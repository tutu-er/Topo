"""Frozen paired actual controller. Import performs no AC, LP or future reads."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import traceback

REPO = next((p for p in Path(__file__).resolve().parents
             if (p/'research_experiments/wzzt_dr_selection').is_dir()),
            Path(os.environ.get('WZZT_REPO_ROOT', r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')))
BASE = REPO / 'research_experiments/wzzt_dr_selection'
CHECKPOINTS = ('early', 'final')
OBS_FIELDS = ('P_terminal', 'Q_terminal', 'V_terminal', 'V_root_observed')

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(p)

def checkpoint_observations(history, packet, checkpoint):
    """early ends at21h: 288+84train,288val; final:480train,480val.

    Event order is train0,train1,val0,val1. Packet may contain later observations,
    but early hashes/fit contain only event0 rows [00:00,21:00).
    """
    training, validation = list(history['training']), list(history['validation'])
    if checkpoint == 'early':
        ids=packet.get('metadata',{}).get('training_event_ids')
        if ids is None: ids=[s.get('actual_event_id') for s in packet['training']]
        if len(ids)!=len(packet['training']) or ids.count(0)!=1:
            raise ValueError('explicit event0 to observed-scene mapping missing/ambiguous')
        first = packet['training'][ids.index(0)]
        if len(first['P_terminal']) != 96: raise ValueError('event0 must have96 original rows')
        scene = {k:first[k].iloc[:84].copy() for k in OBS_FIELDS}
        if 'day_ids' in first: scene['day_ids'] = list(first['day_ids'][:84])
        training.append(scene); expected = (372,288)
    elif checkpoint == 'final':
        if len(packet['training']) != 2 or len(packet['validation']) != 2:
            raise ValueError('final requires all four original observed days')
        for role,expected_ids in (('training',[0,1]),('validation',[2,3])):
            ids=packet.get('metadata',{}).get(role+'_event_ids')
            if ids is None: ids=[s.get('actual_event_id') for s in packet[role]]
            if ids!=expected_ids or any(len(s['P_terminal'])!=96 for s in packet[role]):
                raise ValueError('final event identities/order/row counts do not match protocol')
        training += packet['training']; validation += packet['validation']; expected = (480,480)
    else: raise ValueError('unknown fixed checkpoint')
    rows = tuple(sum(len(s['P_terminal']) for s in split) for split in (training,validation))
    if rows != expected: raise ValueError(f'checkpoint row mismatch {rows} != {expected}')
    return training, validation

def payment_account(receipt, checkpoint, price_rows):
    """Keep early-known commitments separate from post-hoc future realizations."""
    if checkpoint not in CHECKPOINTS: raise ValueError('unknown fixed checkpoint')
    cutoff = 1 if checkpoint == 'early' else 4
    prices = [list(map(float,row)) for row in price_rows]
    if len(prices)!=2 or not prices[0] or len(prices[0])!=len(prices[1]):
        raise ValueError('two aligned price rows required')
    if any(p not in (0.,1.,2.) for row in prices for p in row): raise ValueError('invalid price')
    worst = sum(.2*sum(prices[e%2]) for e in range(4))
    if worst > 1.6+1e-12: raise ValueError('four-event budget exceeded')
    events = {int(r['event']):r for r in receipt.get('event_records',[])}
    def fees(indices):
        values = [events.get(e,{}).get('actual_payment') for e in indices]
        return sum(values) if all(v is not None for v in values) else None
    def commitments(indices):
        values = [events.get(e,{}).get('committed_payment') for e in indices]
        return sum(values) if all(v is not None for v in values) else None
    rebound_ok = all(events.get(e,{}).get('status')=='complete' and
        events[e].get('checks',{}).get('energy_return_pass',False) for e in range(cutoff))
    return dict(paid_through_checkpoint=fees(range(cutoff)),
        accepted_committed_payment_through_checkpoint=commitments(range(cutoff)),
        full_contract_realized_payment=fees(range(4)),
        full_contract_committed_payment_evaluator_only=commitments(range(4)),
        future_realized_payment_evaluator_only=fees(range(cutoff,4)),
        unstarted_event_worst_payment=sum(.2*sum(prices[e%2]) for e in range(cutoff,4)),
        full_contract_worst_payment=worst,
        pending_accepted_rebound_kwh=0. if rebound_ok else None,
        completed_event_rebound_verified=rebound_ok,
        unstarted_event_max_reduction_kwh=sum(.2*sum(p>0 for p in prices[e%2]) for e in range(cutoff,4)),
        future_event_cancellation=False,
        scope='early future actual fees evaluator-only; four events stay scheduled; early/final costs are overlapping, not additive')

def _row(case):
    return case.get('history_row',case.get('source_row',case.get('row',case)))

def prepare(root, case_plans, physical_worker, physical_module, *, pilot_job=None,
            physical_source_root=None, replicate_base=20000, participation_seed=90410000):
    """Freeze32 records for b8/.001, without loading new actual observations."""
    root, case_plans = Path(root).resolve(), Path(case_plans).resolve()
    if root.exists(): raise FileExistsError('existing actual root retained')
    plan_summary = read(case_plans)
    if plan_summary.get('status')!='complete': raise ValueError('price planning must finish before actual freeze')
    cases = [c for c in plan_summary['cases'] if _row(c).get('bundle')==8 and
        float(c.get('voltage_sigma',_row(c).get('voltage_sigma',-1)))==.001]
    if pilot_job: cases = [c for c in cases if c.get('job_id',_row(c).get('job_id'))==pilot_job]
    if len(cases)!=1: raise ValueError('exactly one frozen b8/.001 case required')
    case = cases[0]; row = _row(case)
    if set(case['locks'])!={'local','common'}: raise ValueError('two response laws required')
    sys.path[:0] = [str(REPO),str(REPO/'rnj_wzzt_core')]
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    sources = source_hashes()
    dependencies = {Path(__file__),Path(physical_worker),Path(physical_module),
        BASE/'dr_competition_bridge/physical_bridge.py',BASE/'dr_competition_bridge/coded_shapes.py'}
    dependencies.update(Path(physical_module).parent.glob('*.py'))
    dependencies.update(Path(__file__).parent.glob('*.py'))
    for p in dependencies:
        sources[str(p.resolve())] = sha(p)
    locks, policies = {}, None
    for law, ref in case['locks'].items():
        p = Path(ref['path']).resolve()
        if sha(p)!=ref['sha256']: raise ValueError('price lock hash mismatch')
        lock = read(p)
        if lock.get('status')!='frozen': raise ValueError('unfrozen price lock')
        names = list(lock['plans'])
        if policies is None: policies = names
        if names!=policies or len(names)!=8: raise ValueError('same8 policies required in each law')
        for plan in lock['plans'].values(): payment_account({},'final',plan)
        locks[law] = dict(path=str(p),sha256=sha(p))
    inputs = {str(case_plans):sha(case_plans)}
    for role in ('train','validation'):
        ref = row[role]; p = Path(ref['path']).resolve()
        expected = ref.get('catalog_sha256',ref.get('sha256'))
        if not expected or sha(p)!=expected: raise ValueError('old observation hash mismatch')
        inputs[str(p)] = expected
    archived = {}
    if physical_source_root:
        archived = {str(p.resolve()):sha(p) for p in sorted(Path(physical_source_root).rglob('*.py'))}
        if not archived: raise ValueError('physical source closure missing')
    cfg = dict(status='frozen_before_actual_future',created_utc=now(),case=case,history_row=row,
        locks=locks,policies=policies,checkpoints=list(CHECKPOINTS),planned_records=32,shared_natural_paths=1,
        early=dict(end_hour=21,event=0,first_day_rows=84,training_rows=372,validation_rows=288),
        final=dict(training_rows=480,validation_rows=480),event_order=['train0','train1','val0','val1'],
        source_sha256=sources,input_sha256=inputs,archived_source_sha256=archived,
        case_plans_path=str(case_plans),case_plans_sha256=sha(case_plans),
        physical_worker=str(Path(physical_worker).resolve()),physical_module=str(Path(physical_module).resolve()),
        physical_source_root=str(Path(physical_source_root).resolve()) if physical_source_root else None,
        replicate_base=int(replicate_base),participation_seed=int(participation_seed),event_profiles=[0,2,0,2],
        fit_path_seconds=25.,fit_hard_seconds=180.,physical_hard_seconds=120.,budget=1.6,
        actual_future_loaded_at_prepare=False,truth_passed_to_identifier=False,
        scope='one selected development case;32 paired checkpoint records,16 policy/law trajectories,one shared natural path',
        price_planning_endpoint='final; early is a prelocked mechanism diagnostic, not an early-optimized price',
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform()))
    root.mkdir(parents=True)
    write(root/'actual_protocol.json',cfg)
    write(root/'actual_protocol.sha256.json',dict(sha256=sha(root/'actual_protocol.json')))
    return cfg

def verify(root):
    root = Path(root)
    if sha(root/'actual_protocol.json')!=read(root/'actual_protocol.sha256.json')['sha256']:
        raise RuntimeError('actual protocol drift')
    cfg = read(root/'actual_protocol.json')
    for group in ('source_sha256','input_sha256','archived_source_sha256'):
        for p,h in cfg[group].items():
            if sha(p)!=h: raise RuntimeError(group+' drift: '+p)
    for ref in cfg['locks'].values():
        if sha(ref['path'])!=ref['sha256']: raise RuntimeError('price lock drift')
    return cfg

def fit_worker(root, packet_path, checkpoint, out):
    cfg = verify(root)
    sys.path[:0] = [str(REPO),str(REPO/'rnj_wzzt_core')]
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import load_history
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import fit_observed
    history = load_history(cfg['history_row'])
    packet = pickle.loads(Path(packet_path).read_bytes())
    train,val = checkpoint_observations(history,packet,checkpoint)
    return fit_observed(train,val,out,search_seconds=cfg['fit_path_seconds'])

def _supports(point, labels):
    sets = point.get('support_labels')
    if sets is None: sets = [[labels[i] for i in s] for s in point['supports']]
    r,x = point.get('r_values',point.get('r')),point.get('x_values',point.get('x'))
    return {tuple(sorted(s)) for s,rv,xv in zip(sets,r,x,strict=True) if rv>0 or xv>0}

def summarize(records):
    complete = [r for r in records if r['status']=='complete']; groups=[]
    for checkpoint in CHECKPOINTS:
        for policy in sorted({r['policy'] for r in records}):
            all_rows = [r for r in records if r['checkpoint']==checkpoint and r['policy']==policy]
            rows = [r for r in all_rows if r['status']=='complete']
            costs = [r.get('actual_payment') for r in all_rows]
            g = dict(policy=policy,checkpoint=checkpoint,planned=2,records=len(all_rows),complete=len(rows),
                statuses=dict(Counter(r['status'] for r in all_rows)),structure_denominator=len(rows),
                known_paid_records=sum(v is not None for v in costs),unknown_paid_records=sum(v is None for v in costs),
                known_paid_sum=sum(v for v in costs if v is not None))
            for tag,name in (('wzzt','multistart_original_rule'),('rnj','rnj_bank')):
                g[tag+'_full']=sum(r['evaluation'][name]['full_recovery'] for r in rows)
                g[tag+'_errors']=sum(r['evaluation'][name]['full_block_errors'] for r in rows)
            groups.append(g)
    return dict(status='complete' if len(records)==32 else 'partial',planned=32,records=len(records),
        statuses=dict(Counter(r['status'] for r in records)),
        unique_inputs=len({r['observed_fingerprint'] for r in records if r.get('observed_fingerprint')}),
        unique_complete_fits=len({r['fit_path'] for r in complete}),
        incomplete_scan_records=sum(not r['all_local_scans_complete'] for r in complete),groups=groups,
        scope='32 paired checkpoint records;16 trajectories;one shared four-day path;early/final costs overlap')

def run(root,max_records=32):
    root = Path(root).resolve(); cfg=verify(root); lock=root/'run.lock'
    with lock.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),started_utc=now(),protocol_sha256=sha(root/'actual_protocol.json')),f)
    try: return _run_locked(root,cfg,int(max_records))
    finally: lock.unlink()

def _run_locked(root,cfg,max_records):
    sys.path[:0] = [str(REPO),str(REPO/'rnj_wzzt_core')]
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import load_history
    from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,input_hash
    history=load_history(cfg['history_row']); rp=root/'actual_records.json'
    records=read(rp) if rp.exists() else []; seen={r['record_id'] for r in records}
    cache={r['observed_fingerprint']:r for r in records if r.get('fit_attempted')}
    env=os.environ.copy(); env['PYTHONPATH']=os.pathsep.join([str(REPO),str(REPO/'rnj_wzzt_core')])
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[k]='1'
    new=0
    for law in ('local','common'):
        ref=cfg['locks'][law]; frozen=read(ref['path'])
        for policy in cfg['policies']:
            if new>=max_records: return summarize(records)
            physical=root/'physical'/f'{law}__{policy}'; receipt_file=physical/'events_receipt.json'
            if not receipt_file.exists():
                if physical.exists(): raise RuntimeError('partial physical directory retained: '+str(physical))
                command=[sys.executable,cfg['physical_worker'],'--plan-lock',ref['path'],
                    '--expected-lock-sha',ref['sha256'],'--policy',policy,'--path','0','--dependence',law,
                    '--sigma',str(cfg['case'].get('voltage_sigma',.001)),'--output',str(physical),
                    '--repo-root',str(REPO),'--replicate-base',str(cfg['replicate_base']),
                    '--participation-seed',str(cfg['participation_seed']),'--event-profiles',*map(str,cfg['event_profiles'])]
                if cfg['physical_source_root']: command += ['--source-root',cfg['physical_source_root']]
                launched=now()
                try:
                    proc=subprocess.run(command,env=env,capture_output=True,text=True,timeout=cfg['physical_hard_seconds'])
                    physical.mkdir(parents=True,exist_ok=True)
                    (physical/'controller_worker.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
                    write(physical/'controller_launch.json',dict(started_utc=launched,ended_utc=now(),command=command,returncode=proc.returncode))
                except subprocess.TimeoutExpired as exc:
                    physical.mkdir(parents=True,exist_ok=True)
                    if not receipt_file.exists(): write(receipt_file,dict(status='timeout',event_records=[],actual_payment=None,error=str(exc)))
                    write(physical/'controller_launch.json',dict(started_utc=launched,ended_utc=now(),command=command,status='timeout',error=str(exc)))
                except Exception as exc:
                    physical.mkdir(parents=True,exist_ok=True)
                    if not receipt_file.exists():
                        write(receipt_file,dict(status='worker_launch_failed',event_records=[],actual_payment=None,
                            error_type=type(exc).__name__,error=str(exc)))
                    write(physical/'controller_launch.json',dict(started_utc=launched,ended_utc=now(),command=command,
                        status='failed',error_type=type(exc).__name__,error=str(exc)))
            if not receipt_file.exists():
                write(receipt_file,dict(status='worker_failed_no_receipt',event_records=[],actual_payment=None))
            receipt=read(receipt_file)
            for checkpoint in CHECKPOINTS:
                key=f"{cfg['history_row']['job_id']}__{law}__{policy}__{checkpoint}"
                if key in seen: continue
                if new>=max_records: return summarize(records)
                out=root/'records'/key
                if out.exists(): raise RuntimeError('partial record retained: '+key)
                out.mkdir(parents=True); started=time.perf_counter()
                r=dict(record_id=key,policy=policy,dependence=law,checkpoint=checkpoint,status='running',
                    started_utc=now(),source_sha256=cfg['source_sha256'],protocol_sha256=sha(root/'actual_protocol.json'),
                    price_lock_sha256=ref['sha256'],physical_receipt_path=str(receipt_file),
                    physical_receipt_sha256=sha(receipt_file),actual_payment=None,fit_attempted=False,
                    truth_read_before_fit=False)
                try:
                    r['cost']=payment_account(receipt,checkpoint,frozen['plans'][policy])
                    r['actual_payment']=r['cost']['paid_through_checkpoint']
                    needed=[0] if checkpoint=='early' else list(range(4))
                    events={int(e['event']):e for e in receipt.get('event_records',[])}
                    if any(events.get(e,{}).get('status')!='complete' for e in needed):
                        r['status']='physical_incomplete'; raise ValueError('required physical event incomplete')
                    packet_path=physical/'packet.pkl'; packet=pickle.loads(packet_path.read_bytes())
                    tr,va=checkpoint_observations(history,packet,checkpoint)
                    train,days,_=normalize(tr,'training'); valid,vdays,_=normalize(va,'validation')
                    ih=dict(training=input_hash(train,days),validation=input_hash(valid,vdays))
                    fp=digest(dict(input_sha256=ih,source_sha256=cfg['source_sha256'],search_seconds=cfg['fit_path_seconds']))
                    target=root/'fits'/fp
                    r.update(observed_fingerprint=fp,input_sha256=ih,training_rows=len(train[0]),validation_rows=len(valid[0]),fit_path=str(target/'result.json'))
                    old=cache.get(fp)
                    if old:
                        r.update(fit_reused=True,reused_from=old['record_id'],fit_attempted=True)
                        if not old.get('fit_complete'):
                            r['status']='fit_reused_'+old['status']; raise ValueError('same-input failed attempt retained without retry')
                    else:
                        r.update(fit_attempted=True,fit_reused=False)
                        command=[sys.executable,'-B',str(Path(__file__).resolve()),'fit-worker','--root',str(root),
                            '--packet',str(packet_path),'--checkpoint',checkpoint,'--out',str(target)]
                        ts=time.perf_counter()
                        proc=subprocess.run(command,env=env,capture_output=True,text=True,timeout=cfg['fit_hard_seconds'])
                        r['fit_elapsed_seconds']=time.perf_counter()-ts
                        (out/'fit_worker.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
                        if proc.returncode: raise RuntimeError('identifier returned '+str(proc.returncode))
                    result=read(r['fit_path'])
                    if result.get('status')!='complete': raise RuntimeError('identifier not complete')
                    if result.get('input_sha256')!=ih: raise RuntimeError('identifier input hash mismatch')
                    r['fit_complete']=True
                    # Evaluator truth is read only after the fit output has returned.
                    truth={tuple(sorted(s)) for s in read(physical/'truth_evaluation.json')['true_supports']}
                    r['truth_read_after_identifier_utc']=now(); evaluation={}
                    for name,point in result['models'].items():
                        pred=_supports(point,result['terminal_labels'])
                        evaluation[name]=dict(full_block_errors=len(truth^pred),full_recovery=truth==pred,
                            missed=[list(s) for s in sorted(truth-pred)],extra=[list(s) for s in sorted(pred-truth)],
                            selected_origin=point.get('selected_origin'))
                    r.update(status='complete',evaluation=evaluation,all_local_scans_complete=result['all_local_scans_complete'],
                        LP_calls=len(result['LP_solves']),fit_input_sha256=result['input_sha256'],fit_source_sha256=result['source_sha256'])
                except subprocess.TimeoutExpired as exc:
                    r.update(status='timeout',error=str(exc),failure_retained=True)
                except Exception as exc:
                    if r['status']=='running': r['status']='failed'
                    r.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),failure_retained=True)
                r.update(ended_utc=now(),elapsed_seconds=time.perf_counter()-started)
                if r.get('fit_attempted'): cache[r['observed_fingerprint']]=r
                write(out/'record.json',r); records.append(r); seen.add(key); new+=1
                write(rp,records); write(root/'actual_summary.json',summarize(records))
                print(json.dumps(dict(record=key,status=r['status'],completed=len(records))),flush=True)
    return summarize(records)

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('prepare','run','fit-worker'))
    p.add_argument('--root',type=Path,required=True); p.add_argument('--case-plans',type=Path)
    p.add_argument('--physical-worker',type=Path); p.add_argument('--physical-module',type=Path)
    p.add_argument('--source-root',type=Path); p.add_argument('--pilot-job')
    p.add_argument('--replicate-base',type=int,default=20000); p.add_argument('--participation-seed',type=int,default=90410000)
    p.add_argument('--max-records',type=int,default=32); p.add_argument('--packet',type=Path)
    p.add_argument('--checkpoint',choices=CHECKPOINTS); p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':
        if not all((a.case_plans,a.physical_worker,a.physical_module)): p.error('prepare needs frozen plans and physical source paths')
        result=prepare(a.root,a.case_plans,a.physical_worker,a.physical_module,pilot_job=a.pilot_job,
            physical_source_root=a.source_root,replicate_base=a.replicate_base,participation_seed=a.participation_seed)
    elif a.mode=='fit-worker': result=fit_worker(a.root,a.packet,a.checkpoint,a.out)
    else: result=run(a.root,a.max_records)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','planned_records','planned','records','statuses','unique_complete_fits')}),flush=True)

if __name__=='__main__': main()
