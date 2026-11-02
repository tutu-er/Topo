"""Replay frozen spatial DR with device saturation on identical saved paths.

No optimization, fitting, new case, new random path or rejection-rule change.
The original unbounded batch is read as the paired control; only capped AC is
run. Prices and nominal capacities are copied byte-for-byte from that batch.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,importlib.util,json,pickle,platform,shutil,sys,time,traceback,zipfile
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
PRIVATE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
PARENT=PRIVATE/'wzzt_spatial_correlated_stage/run_v2'
sys.path.insert(0,str(REPO))
BRIDGE=Path(__file__).with_name('ac_bridge.py')
if not BRIDGE.exists():BRIDGE=REPO/'research_experiments/wzzt_dr_selection/dr_spatial_correlated/ac_bridge.py'
spec=importlib.util.spec_from_file_location('_saturation_bridge',BRIDGE)
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):
    if sha(r['path'])!=r['sha256']:raise ValueError('frozen reference changed: '+r['path'])
    return Path(r['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def load(p):
    with Path(p).open('rb') as f:return pickle.load(f)
def dump(p,v):
    with Path(p).open('wb') as f:pickle.dump(v,f,protocol=4)

def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    if not (PARENT/'ended.json').exists():raise RuntimeError('parent batch incomplete')
    old=read(PARENT/'protocol.json');sources=old['source_sha256'].copy()
    for f,h in sources.items():
        if sha(f)!=h:raise ValueError('parent source drift '+f)
    for f in [BRIDGE,*Path(__file__).parent.glob('*.py')]:sources[str(f.resolve())]=sha(f)
    p=dict(id='saturated_spatial_replay_v1',status='frozen',created_utc=now(),
        parent_protocol=ref(PARENT/'protocol.json'),parent_observations=ref(PARENT/'observations_frozen.json'),
        parent_scored=ref(PARENT/'scored_private.json'),parent_truth=ref(PARENT/'truth_private.json'),
        parent_prices=ref(PARENT/'price_freeze.json'),source_sha256=sources,
        source_root=old['source_root'],source_metadata=old['source_metadata'],inputs=old['inputs'],
        days=old['days'],observed_laws=old['observed_laws'],rule=old['rule'],case=old['case'],
        T=96,dt=.25,base_kw=200.,response_main=old['response_main'],response_misspec=old['response_misspec'],
        resource_total_kw=2.,voltage_sigma=.0005,pq_sigma=.005,root_sigma=.003,
        intervention='component a+U projected to [0,2a], equivalently U clipped to [-a,a]; Q increment=.35 times clipped U',
        prices='exact original prepaid fees, quotes, mean commands, allocation; no delivery-linked re-settlement',
        observations='reuse stored natural P/Q/root/Z; original AC and original relative meter innovations/seeds',
        scope='paired retrospective same12-path sensitivity; not new independent validation or saturation-aware optimization',
        planned_records=144,new_optimization=0,new_parameter_fit=0,new_W_search=0,
        pass_condition='report every policy/law; judge retention by final wrong+missing blocks, complete recovery, delivered kWh, fee and equipment violations',
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()))
    write(root/'protocol.json',p)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,f in enumerate(sources):z.write(f,f'{i:03d}_{Path(f).name}')
    for name in ['price_freeze.json','problem_frozen.json']:shutil.copyfile(PARENT/name,root/name)
    write(root/'run_lock.json',dict(status='prepared',pid=None))
    print('locked 144 saturated AC replays, no new paths or quotes',flush=True)

def config(root):
    p=read(root/'protocol.json')
    for file,h in p['source_sha256'].items():
        if sha(file)!=h:raise ValueError('source drift '+file)
    f=read(checked(p['parent_prices']))
    if sha(root/'price_freeze.json')!=p['parent_prices']['sha256']:raise ValueError('prices changed')
    k=load(checked(read(root/'problem_frozen.json')['input']))
    return p,f,k

def observe(root):
    if (root/'observations_frozen.json').exists():raise RuntimeError('already completed; no duplicate AC')
    p,f,k=config(root);parent=read(checked(p['parent_observations']))
    rt=bridge.open_runtime(p['source_root'],p['source_metadata'],repo_root=REPO,protocol_ref=ref(root/'protocol.json'),
        labels=k['pool']['labels'],case='paper15',v_sigma=.0005,pq_sigma=.005)
    rows=[];write(root/'run_lock.json',dict(status='replaying_AC',pid=os.getpid()))
    for old in parent['rows']:
        name=old['record_id'];folder=root/'actual'/name;folder.mkdir(parents=True,exist_ok=True)
        receipt=folder/'receipt.json'
        if receipt.exists():
            r=read(receipt)
            if 'packet' in r:checked(r['packet'])
            rows.append(r);continue
        r={key:old[key] for key in ['record_id','day','profile_index','replicate','response_seed','law','policy','fee']}
        try:
            previous=load(checked(old['packet']));pr=previous['private'];ps=previous['summary']
            pol=f['policies'][old['policy']];u=np.asarray(pol['mean_u']);a=np.asarray(pol['capacity_kw'])
            if not np.array_equal(u,pr['mean_u']) or not np.array_equal(a,pr['capacity_kw']):raise ValueError('old observation does not use frozen policy')
            bg=SimpleNamespace(p_base=pr['natural_P_pu'].copy(),q_base=pr['natural_Q_pu'].copy(),root_true=pr['root_true'].copy(),
                scenario_index=old['profile_index'],replicate=old['replicate'],profile=ps['profile'],shifted_seed=ps['profile_seed'])
            result=bridge.execute_policy(rt,bg,u,pr['Z'],resource_labels=k['control_labels'],capacity_kw=a,
                cv=pr['cv'],multiplier_mean=pr['multiplier_mean'],day_offset=6+old['day'],name=name+'_saturated',delivery_rule='device_clip')
            dump(folder/'result.pkl',result)
            r.update(status='observed',packet=ref(folder/'result.pkl'),paired_uncapped=old['packet'],physical=result['summary'])
        except Exception as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
        write(receipt,r);rows.append(r)
        write(root/'checkpoint.json',dict(completed=len(rows),planned=144,last=name))
        if len(rows)%12==0:print('completed',len(rows),'/144',flush=True)
    write(root/'observations_frozen.json',dict(status='frozen',rows=rows,prices=ref(root/'price_freeze.json'),protocol=ref(root/'protocol.json')))
    write(root/'run_lock.json',dict(status='observations_frozen',pid=None))

def evaluate(root):
    if (root/'ended.json').exists():raise RuntimeError('already evaluated')
    p,f,k=config(root);ledger=read(root/'observations_frozen.json');rows=[]
    for r0 in ledger['rows']:
        row={key:r0[key] for key in ['record_id','day','law','policy','fee']}
        try:
            if r0['status']!='observed':raise RuntimeError('physical failed')
            result=load(checked(r0['packet']));scene=result['scenes']['both']
            if scene is None:raise RuntimeError('physical unresolved')
            s=score(k['pool'],scene,history_gate=True)
            q=next(q for q in s['rules'] if q['new_day_weight']==2 and q['new_day_fraction']==.03)
            losses={r['candidate_id']:r['new_validation_mae'] for r in s['loss_rows']}
            write(root/'scores'/(r0['record_id']+'.json'),s)
            ps=result['summary'];deliv=result['private']['delivered_kw'];raw=result['private']['proposed_delivered_kw']
            row.update(status='predicted',selected=q['selected_candidate_id'],coarse_excluded='coarse' not in q['eligible_candidate_ids'],
                D_micro=1e6*(losses['coarse']-1.03*losses['refined']),margin_micro=1e6*q['exit_margins']['coarse'],
                abs_kwh=.25*float(np.abs(deliv).sum()),net_kwh=.25*float(deliv.sum()),raw_abs_kwh=.25*float(np.abs(raw).sum()),
                removed_kwh=.25*float((np.abs(raw)-np.abs(deliv)).sum()),peak_sum_abs_kw=float(np.abs(deliv).sum(0).max()),
                clipped_cells=ps['saturated_cells'],resource_violation_cells=ps['resource_violation_cells'],
                voltage_violation_cells=ps['voltage_violation_cells'],ac_converged=ps['ac_converged_samples'])
        except Exception as e:row.update(status='unresolved',error=repr(e))
        rows.append(row)
    write(root/'predictions_frozen.json',dict(status='frozen',rows=rows,truth_used_in_decision=False))
    truth={tuple(sorted(x)) for x in read(checked(p['parent_truth']))['missed_blocks']}
    models={}
    for model in k['pool']['models']:
        final={tuple(sorted(x)) for x in model['frozen_output_support_labels']}
        raw={tuple(sorted(k['pool']['labels'][i] for i in b)) for b in model['point']['supports']}
        models[model['candidate_id']]=dict(errors=len(final^truth),missing=len(truth-final),extra=len(final-truth),raw_errors=len(raw^truth))
    oldrows={r['record_id']:r for r in read(checked(p['parent_scored']))['rows']}
    for row in rows:
        old=oldrows[row['record_id']]
        row.update(old_selected=old.get('selected'),old_errors=old.get('errors'),old_D_micro=old.get('D_micro'),old_fee=old['fee'])
        if row['status']=='predicted':row.update(status='complete',**models[row['selected']])
    tables=[];comparisons=[]
    for law in p['observed_laws']:
        for name,policy in f['policies'].items():
            rr=[r for r in rows if r['law']==law and r['policy']==name];valid=[r for r in rr if r['status']=='complete'];paired=[r for r in valid if r['old_errors'] is not None]
            tables.append(dict(law=law,policy=name,planned=12,complete=len(valid),unresolved=12-len(valid),
                old_errors=sum(oldrows[r['record_id']].get('errors',0) for r in rr),errors=sum(r['errors'] for r in valid),
                old_recoveries=sum(oldrows[r['record_id']].get('errors')==0 for r in rr),recoveries=sum(r['errors']==0 for r in valid),
                better_vs_uncapped=sum(r['errors']<r['old_errors'] for r in paired),worse_vs_uncapped=sum(r['errors']>r['old_errors'] for r in paired),same_vs_uncapped=sum(r['errors']==r['old_errors'] for r in paired),
                raw_abs_kwh_per_day=float(np.mean([r['raw_abs_kwh'] for r in valid])) if valid else None,
                abs_kwh_per_day=float(np.mean([r['abs_kwh'] for r in valid])) if valid else None,
                removed_kwh_per_day=float(np.mean([r['removed_kwh'] for r in valid])) if valid else None,
                net_kwh_per_day=float(np.mean([r['net_kwh'] for r in valid])) if valid else None,
                mean_D_micro=float(np.mean([r['D_micro'] for r in valid])) if valid else None,
                max_sum_abs_kw=max([r['peak_sum_abs_kw'] for r in valid],default=None),
                resource_violation_cells=sum(r['resource_violation_cells'] for r in valid),voltage_violation_cells=sum(r['voltage_violation_cells'] for r in valid),fee_per_day=policy['cost'],fees_unchanged=all(r['fee']==r['old_fee'] for r in rr)))
        for other in ['zero','legacy_iid_pair','fixed_correlated','rotation_uniform','rotation_selected_capacity']:
            diff=[]
            for d in p['days']:
                a=next(r for r in rows if r['law']==law and r['day']==d['day'] and r['policy']=='open_correlated');b=next(r for r in rows if r['law']==law and r['day']==d['day'] and r['policy']==other)
                if a['status']==b['status']=='complete':diff.append(a['errors']-b['errors'])
            comparisons.append(dict(law=law,comparison='open_vs_'+other,better=sum(x<0 for x in diff),same=sum(x==0 for x in diff),worse=sum(x>0 for x in diff),unresolved=12-len(diff)))
    write(root/'scored.json',dict(rows=rows))
    write(root/'summary.json',dict(status='complete',tables=tables,comparisons=comparisons,models=models,planned=144,
        complete=sum(r['status']=='complete' for r in rows),background_paths=12,paired_laws=2,new_prices=0,new_fits=0,new_search=0,
        actual_ac_samples=sum(r.get('ac_converged',0) for r in rows),scope=p['scope']))
    write(root/'ended.json',dict(status='complete',summary=ref(root/'summary.json'),completed_utc=now()))
    write(root/'run_lock.json',dict(status='complete',pid=None))
    print(json.dumps(read(root/'summary.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','observe','evaluate']);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();globals()[args.command](args.output)
