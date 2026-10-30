"""Bounded development run; predictions and post-freeze physical data separated."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,hashlib,json,pickle,platform,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from .physical_bridge import load_history,forecast_mean,sha,plain_hash
from .input_design import ordinary_plans,pair_stochastic_plan

REPO=Path(__file__).resolve().parents[3]
BASE=REPO/'research_experiments/wzzt_dr_selection'
POLICIES=('zero','single_input','sync_input','rotation','global_mean_input','pair_stochastic_input','wzzt_response_table')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def timestamp():return datetime.now(timezone.utc).isoformat()


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists():return verify(root)
    source_index=BASE/'evidence/scenario_review_v1/source_inventory.json'
    inventory=read(source_index)
    rows=[r for r in inventory['rows'] if r['case']=='paper15' and r['bundle']==0]
    assert len(rows)==4 and all(r['N']==288 for r in rows)
    sources={str(p):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    for rel in ('dr_pool_v2/engine.py','dr_pool_v2/adapter.py','dr_pool_v2/selection.py','dr_pool_v2/observations.py','end_to_end/physical_case.py'):
        p=BASE/rel;sources[str(p)]=sha(p)
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    sources.update(source_hashes())
    physical_source=Path(r'D:\0-github_workspace\Topo\outputs\wzzt_paper_20261029_full\source\rnj_wzzt')
    physical_hashes={str(p):sha(p) for p in sorted(physical_source.rglob('*.py'))}
    inputs={str(source_index):sha(source_index),str(Path(__file__).with_name('case_audit.json')):sha(Path(__file__).with_name('case_audit.json'))}
    for r in rows:
        inputs[r['existing_result']['path']]=r['existing_result']['catalog_sha256']
        # Historical bytes are verified by load_history; no old test is used.
        for role in ('train','validation'):inputs[r[role]['path']]=r[role]['catalog_sha256']
    protocol=dict(created_utc=timestamp(),base_commit='269bc4c5',implementation_version=2,cases=rows,source_sha256=sources,input_sha256=inputs,
        physical_source_sha256=physical_hashes,
        case_rule='all paper15 b0 reference N288 noise levels; not result-based selection',
        local_price_case='main__paper15__b0__reference__v0010__N288',
        representative_indices=[19,16,7],active_position_rule='seeded representatives of first historical crossing exclusive regions; no truth',
        planning=dict(wall_seconds=600.,path_seconds=15.,max_LP_calls=6000,coefficient_upper=None,
            proposal_scope='historical_representative_union',
            hypotheses='three original raw fits including zero/weak weights, equal heuristic weights',
            menu_domain='full50 menus on two history-selected sites; price sums<=4; zero and single included',
            table='16training masks x3hypotheses; full256train-validation states; validation has no LP refit',
            other_cases='no-cross trigger returns zero, not a statement of structural uniqueness'),
        contract=dict(price=[0,1,2],participation=[0,.5,.75],reduction_kw=.2,reduction_kvar=.07,
            reduce_hour=19,rebound_hour=20,duration_hours=1.,interval_hours=.25,worst_budget=1.6,
            independent_validation_price_reuse=True),
        actual=dict(paths=[0],dependence=['local','common'],event_profiles=[0,2,0,2],replicate_base=10000,
            fit_path_seconds=25.,fit_hard_seconds=180.,physical_hard_seconds=120.,max_records=56,
            full_current_observed_RNJ_WZZT=True,fresh_pool_and_starts=True),
        seeds=dict(site=530013,tie=530014,rotation=530015,participation=90310000),
        status='frozen_before_loading_history_or_actual_future',randomness_scope='one shared actual path perlaw; pilot, not recovery probability estimate',
        no_positive_parameter_lower_bound=True,no_budget_or_noise_change=True,no_old_test_trajectory_use=True,
        production_algorithm_changed=False,new_empirical_behavior_calibration=False)
    write(path,protocol);return protocol


def verify(root):
    p=read(Path(root)/'protocol.json')
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('frozen source changed '+path)
    for path,digest in p['physical_source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('frozen physical source changed '+path)
    # Check metadata/old result fingerprints; NPZ hashes also checked at loading.
    for path,digest in p['input_sha256'].items():
        if Path(path).suffix!='.npz' and sha(path)!=digest:raise RuntimeError('frozen past input changed '+path)
    return p


def active_pair(old,indices,seed):
    points=[old['base_points'][i] for i in indices]
    fams=[{tuple(sorted(s)) for s in p['supports']} for p in points]
    crossings=[]
    for i,f in enumerate(fams):
        for j,g in enumerate(fams[i+1:],start=i+1):
            for c in sorted(f):
                for d in sorted(g):
                    a,b=set(c),set(d)
                    if a&b and not a<=b and not b<=a:
                        crossings.append((i,j,c,d))
    if not crossings:raise RuntimeError('declared historical crossing absent')
    i,j,c,d=sorted(crossings)[0]
    rng=np.random.default_rng(seed)
    first=int(rng.choice(sorted(set(c)-set(d))));second=int(rng.choice(sorted(set(d)-set(c))))
    return [old['labels'][first],old['labels'][second]],dict(hypothesis_indices=[i,j],crossing_support_indices=[c,d],site_seed=seed)


def plan(root):
    root=Path(root);cfg=prepare(root);done=root/'planning_summary.json'
    if done.exists():return read(done)
    summary=dict(status='running',cases=[],protocol_sha256=sha(root/'protocol.json'),actual_future_read=False)
    for row in cfg['cases']:
        job=row['job_id'];out=root/'planning'/job;out.mkdir(parents=True,exist_ok=True)
        if (out/'case_plans.json').exists():
            summary['cases'].append(read(out/'case_plans.json'));continue
        print(json.dumps(dict(stage='planning_history',job=job)),flush=True)
        history=load_history(row)
        write(out/'history_receipt.json',history['receipt'])
        old=read(row['existing_result']['path'])
        ordinary=ordinary_plans(history,cfg['seeds']['rotation'])
        write(out/'ordinary_input_plans.json',ordinary)
        if job==cfg['local_price_case']:
            pair,receipt=active_pair(old,cfg['representative_indices'],cfg['seeds']['site'])
            from .local_replay import plan_case
            replay=plan_case(history,old,cfg['representative_indices'],pair,out/'response_table',
                             wall_seconds=cfg['planning']['wall_seconds'],tie_seed=cfg['seeds']['tie'],
                             path_seconds=cfg['planning']['path_seconds'],max_LP_calls=cfg['planning']['max_LP_calls'],
                             proposal_scope=cfg['planning']['proposal_scope'])
            if replay['status']!='complete':
                summary.update(status='planning_unresolved',failed_case=job,exception=replay.get('exception'))
                write(done,summary);return summary
        else:
            # Explicit trigger restriction: the proposed method targets crossings.
            rng=np.random.default_rng(cfg['seeds']['site'])
            pair=[history['labels'][i] for i in rng.choice(len(history['labels']),2,replace=False)]
            receipt=dict(reason='no-cross trigger for this bounded pilot',site_seed=cfg['seeds']['site'])
            replay=None
        case=dict(job_id=job,voltage_sigma=row['voltage_sigma'],active_labels=pair,
            site_receipt=receipt,locks={},local_replay_used=replay is not None)
        for law in cfg['actual']['dependence']:
            stochastic=pair_stochastic_plan(history,pair,law,cfg['seeds']['tie'])
            write(out/f'stochastic_input_{law}.json',stochastic)
            plans=dict(ordinary['plans']);plans['pair_stochastic_input']=stochastic['menu']
            plans['wzzt_response_table']=replay['plans']['wzzt_'+law] if replay else plans['zero']
            lock=root/'locks'/f'{job}_{law}.json'
            if lock.exists():raise RuntimeError('existing price lock retained')
            write(lock,dict(status='frozen',created_utc=timestamp(),plans=plans,job_id=job,dependence=law,
                active_labels=pair,protocol_sha256=sha(root/'protocol.json'),future_read_before_freeze=False))
            case['locks'][law]=dict(path=str(lock),sha256=sha(lock))
        write(out/'case_plans.json',case);summary['cases'].append(case)
        write(root/'planning_checkpoint.json',summary)
    summary.update(status='complete',finished_utc=timestamp(),policy_count=len(POLICIES))
    write(done,summary);return summary


def fit_worker(root,job,packet_path,out):
    cfg=verify(root);row=next(r for r in cfg['cases'] if r['job_id']==job)
    history=load_history(row);packet=pickle.loads(Path(packet_path).read_bytes())
    if packet['metadata']['status']!='complete':raise RuntimeError('incomplete physical packet')
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import fit_observed
    return fit_observed(history['training']+packet['training'],history['validation']+packet['validation'],out,
                        search_seconds=cfg['actual']['fit_path_seconds'])


def supports(point,labels):
    sets=point.get('support_labels')
    if sets is None:sets=[[labels[i] for i in s] for s in point['supports']]
    r=point.get('r_values',point.get('r'));x=point.get('x_values',point.get('x'))
    return {tuple(sorted(s)) for s,rv,xv in zip(sets,r,x) if rv>0 or xv>0}


def actual(root,max_records):
    root=Path(root);cfg=verify(root);prices=read(root/'planning_summary.json')
    if prices['status']!='complete':raise RuntimeError('prices not complete/frozen')
    directory=root/'actual';directory.mkdir(exist_ok=True)
    output=root/'actual_records.json';records=read(output) if output.exists() else []
    seen={r['record_id'] for r in records};new=0
    fit_cache={r['observed_fingerprint']:r['fit_path'] for r in records if r.get('fit_path')}
    env=os.environ.copy();env['PYTHONPATH']=str(REPO)+';'+str(REPO/'rnj_wzzt_core')
    for case in prices['cases']:
        row=next(r for r in cfg['cases'] if r['job_id']==case['job_id'])
        history=load_history(row)
        for law in cfg['actual']['dependence']:
            lock=case['locks'][law]
            assert sha(lock['path'])==lock['sha256']
            for path in cfg['actual']['paths']:
                for policy in POLICIES:
                    key=f"{case['job_id']}__{law}__p{path}__{policy}"
                    if key in seen:continue
                    if new>=max_records:write(root/'actual_summary.json',summarize(records,56));return
                    out=directory/key
                    if out.exists():raise RuntimeError('partial actual record retained; inspect '+key)
                    out.mkdir()
                    record=dict(record_id=key,job_id=case['job_id'],voltage_sigma=case['voltage_sigma'],
                                dependence=law,path=path,policy=policy,status='running')
                    command=[sys.executable,str(Path(__file__).with_name('actual_worker.py')),
                        '--plan-lock',lock['path'],'--expected-lock-sha',lock['sha256'],'--policy',policy,
                        '--path',str(path),'--dependence',law,'--sigma',str(case['voltage_sigma']),
                        '--output',str(out/'physical'),'--repo-root',str(REPO),
                        '--replicate-base',str(cfg['actual']['replicate_base']),
                        '--participation-seed',str(cfg['seeds']['participation'])]
                    try:
                        child=subprocess.run(command,env=env,capture_output=True,text=True,
                            timeout=cfg['actual']['physical_hard_seconds'])
                        (out/'physical_worker.log').write_text(child.stdout+child.stderr,encoding='utf-8')
                        receipt=read(out/'physical/events_receipt.json');record['physical']=receipt
                        record['actual_payment']=receipt.get('actual_payment')
                        if child.returncode!=0 or receipt['status']!='complete':
                            raise RuntimeError('physical worker '+receipt['status']+': '+receipt.get('message',''))
                        packet=pickle.loads((out/'physical/packet.pkl').read_bytes())
                        from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,input_hash
                        train,days,_=normalize(history['training']+packet['training'],'training')
                        val,vdays,_=normalize(history['validation']+packet['validation'],'validation')
                        fingerprint=plain_hash([input_hash(train,days),input_hash(val,vdays)])
                        record['observed_fingerprint']=fingerprint
                        target=root/'fits'/fingerprint
                        if fingerprint not in fit_cache:
                            command=[sys.executable,'-B','-m',__package__+'.study','fit-worker',
                                '--root',str(root),'--job',case['job_id'],'--packet',str(out/'physical/packet.pkl'),'--out',str(target)]
                            child=subprocess.run(command,env=env,capture_output=True,text=True,timeout=cfg['actual']['fit_hard_seconds'])
                            (out/'fit_worker.log').write_text(child.stdout+child.stderr,encoding='utf-8')
                            if child.returncode!=0:raise RuntimeError('fit worker returned '+str(child.returncode))
                            fit_cache[fingerprint]=str(target/'result.json')
                        result=read(fit_cache[fingerprint]);record['fit_path']=fit_cache[fingerprint]
                        if result['status']!='complete':raise RuntimeError('fit status '+result['status'])
                        # Actual truth is read only after the identifier has returned.
                        truth={tuple(s) for s in read(out/'physical/truth_evaluation.json')['true_supports']}
                        evaluation={}
                        for name,point in result['models'].items():
                            predicted=supports(point,result['terminal_labels'])
                            evaluation[name]=dict(full_block_errors=len(truth^predicted),full_recovery=truth==predicted,
                                missed=[list(s) for s in sorted(truth-predicted)],extra=[list(s) for s in sorted(predicted-truth)],
                                selected_origin=point.get('selected_origin'))
                        old=read(row['existing_result']['path'])
                        stop={}
                        for name in ('RNJ_L1_bank','multistart'):
                            predicted=supports(old['models'][name],old['labels'])
                            stop[name]=dict(full_block_errors=len(truth^predicted),full_recovery=truth==predicted)
                        record.update(status='complete',evaluation=evaluation,stop_evaluation=stop,
                            training_rows=result['training_rows'],validation_rows=result['validation_rows'],
                            LP_calls=len(result['LP_solves']),all_local_scans_complete=result['all_local_scans_complete'],
                            fit_input_sha256=result['input_sha256'])
                    except subprocess.TimeoutExpired as error:
                        record.update(status='timeout',stage=str(error.cmd[:3]),error=str(error))
                    except Exception as error:
                        record.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
                    write(out/'record.json',record);records.append(record);seen.add(key);new+=1
                    write(output,records);write(root/'actual_summary.json',summarize(records,56))
                    print(json.dumps(dict(stage='actual',record=key,status=record['status'],completed=len(records))),flush=True)
    return summarize(records,56)


def summarize(records,planned):
    from collections import Counter
    complete=[r for r in records if r['status']=='complete']
    groups=[]
    for policy in POLICIES:
        rr=[r for r in complete if r['policy']==policy]
        if rr:groups.append(dict(policy=policy,complete=len(rr),
            wzzt_total_errors=sum(r['evaluation']['multistart_original_rule']['full_block_errors'] for r in rr),
            RNJ_total_errors=sum(r['evaluation']['rnj_bank']['full_block_errors'] for r in rr),
            wzzt_full=sum(r['evaluation']['multistart_original_rule']['full_recovery'] for r in rr),
            payment=sum(r['actual_payment'] for r in rr)))
    return dict(status='complete' if len(records)==planned else 'partial',planned=planned,records=len(records),
        statuses=dict(Counter(r['status'] for r in records)),unique_fits=len({r.get('fit_path') for r in complete}),
        policy_groups=groups,scope='one shared actual path perlaw/noise; not independent56 samples')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','plan','actual','fit-worker'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--max-records',type=int,default=56)
    p.add_argument('--job');p.add_argument('--packet');p.add_argument('--out')
    args=p.parse_args()
    if args.mode=='prepare':result=prepare(args.root)
    elif args.mode=='plan':result=plan(args.root)
    elif args.mode=='actual':result=actual(args.root,args.max_records)
    else:result=fit_worker(args.root,args.job,args.packet,args.out)
    if isinstance(result,dict):print(json.dumps({k:v for k,v in result.items() if k in ('status','records','policy_count','planned','failed_case','exception')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
