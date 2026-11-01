"""Freeze first, then generate new AC days and evaluate the original selector.

No price optimization, parameter refitting or new WZZT search is performed.
Physical truth is read only after all predictive scores have been frozen.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pickle,platform,subprocess,sys,traceback,zipfile
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
import numpy as np
from research_experiments.wzzt_dr_selection.dr_day_curve.adapter import build_plan
from research_experiments.wzzt_dr_selection.dr_day_curve.day_model import resource_audit,scenario_summary
from research_experiments.wzzt_dr_selection.dr_frozen_distance.one_day_worker import load_request,audit_plan
from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score as weighted_score

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
PRIVATE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
DESIGN=PRIVATE/'wzzt_day_curve_stage'
POLICIES=('zero','rotation_low','optimized_low','rotation_high','optimized_high')
MAIN_RULE='w2_tau2pct'

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(x):
    p=Path(x['path'])
    if sha(p)!=x['sha256']:raise ValueError('frozen input/source changed: '+str(p))
    return p
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial')
    t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');t.replace(p)
def unpickle(x):
    with checked(x).open('rb') as stream:return pickle.load(stream)


def rotation_curve(resources,payment_bound,phase=1):
    """Predeclared whole-day six-hour alternating exploration, no model score."""
    if phase not in (-1,1) or payment_bound<=0:raise ValueError('positive commitment and signed phase required')
    first=np.repeat([-1.,1.,-1.,1.],24)*phase
    shapes=np.stack([first,-first])
    # Both normalized shapes reduce for12h, lambda=.5, max multiplier1.2.
    price=math.sqrt(payment_bound/(2*12*.5*1.2))
    return .5*price*shapes,np.full(2,price)


def cp_interval(k,n):
    from scipy.stats import beta
    if not 0<=k<=n or n<=0:raise ValueError('binomial counts required')
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def exact_margin(scored):
    a={r['candidate_id']:r for r in scored['loss_rows']}
    if set(a)!= {'coarse','refined'}:raise ValueError('declared historical pair required')
    h=a['coarse']['historical_validation_mae']-1.03*a['refined']['historical_validation_mae']
    g=a['coarse']['new_validation_mae']-1.02*a['refined']['new_validation_mae']
    return float(.6*h+.4*g-1.e-12)


def source_files(source_root):
    files=set(Path(p) for p in subprocess.check_output(['rg','--files','--no-ignore',str(source_root),'-g','*.py'],text=True,encoding='utf-8').splitlines())
    files.update(Path(p) for p in subprocess.check_output(['rg','--files','--no-ignore',str(REPO/'rnj_wzzt_core/rnj_wzzt'),'-g','*.py'],text=True,encoding='utf-8').splitlines())
    task=REPO/'research_experiments/wzzt_dr_selection'
    files.update((task/'dr_day_effect').glob('*.py'));files.update((task/'dr_day_curve').glob('*.py'))
    for relative in ['dr_frozen_distance/one_day_worker.py','dr_frozen_distance/frozen_pool.py',
        'dr_newday_weighted/weighted_selection.py','dr_day_modules/day_schedule.py',
        'dr_day_modules/day_physical_worker.py','dr_competition_bridge/coded_shapes.py',
        'dr_competition_bridge/physical_bridge.py','end_to_end/physical_case.py',
        'dr_conic_pricing/resources.py','dr_conic_pricing/model.py']:
        files.add(task/relative)
    return sorted(p.resolve() for p in files)


def prior_requests():
    paths=subprocess.check_output(['rg','--files','--no-ignore',str(PRIVATE),'-g','*request*.json'],text=True,encoding='utf-8').splitlines()
    identities=set();seen=0
    for name in paths:
        try:d=read(name)
        except (OSError,ValueError):continue
        if isinstance(d,dict) and all(k in d for k in ('case','profile_index','replicate')):
            identities.add((str(d['case']),int(d['profile_index']),int(d['replicate'])));seen+=1
    return identities,dict(request_files_examined=len(paths),requests_with_physical_identity=seen,
        identity_count=len(identities),identity_digest=hashlib.sha256(json.dumps(sorted(identities)).encode()).hexdigest())


def prepare(root,worker_gate,driver_gate):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    old=read(DESIGN/'model_pilot_uncapped_v2/protocol.json')
    if not (DESIGN/'model_pilot_uncapped_v2/ended.json').exists():raise ValueError('planning must have ended')
    c=next(c for c in old['cases'] if c['name']=='paper15_b2_v0005')
    pool_path=checked(c['pool']);pool=read(pool_path)
    old_controls=read(DESIGN/'model_pilot_v1/protocol.json')
    oc=next(x for x in old_controls['cases'] if x['name']==c['name'])
    historical_protocol=read(PRIVATE/'wzzt_newday_weighted_v1/run_v2/protocol.json')
    hc=next(x for x in historical_protocol['cases'] if x['name']==c['name'])
    declared=read(checked(hc['case_protocol']))
    with (DESIGN/'model_pilot_uncapped_v2/cases'/c['name']/'problem.pkl').open('rb') as f:model=pickle.load(f)
    problem,scenarios=model['problem'],model['scenarios']
    paths=[DESIGN/'paper_adapter_retry_v1/result.json',DESIGN/'model_pilot_uncapped_v2/cases'/c['name']/'result.json']
    candidates=[]
    for p in paths:
        r=read(p);audit=resource_audit(problem,r['mean_u'],r['prices']);target=scenario_summary(problem,scenarios,r['mean_u'])
        if not audit['feasible'] or target['passed']<target['required']:raise ValueError('do not deploy an unqualified diagnostic curve')
        candidates.append(dict(result=ref(p),record=r,audit=audit,target=target))
    candidates.sort(key=lambda x:(x['audit']['expected_payment'],x['result']['path']))
    low,high=candidates
    source_plan=read(checked(oc['old_plans']['optimized']));resources=source_plan['resources']
    plans={};plan_audits={}
    inputs={'low':low['result'],'high':high['result'],'pool':c['pool'],
        'declared_case':hc['case_protocol'],'planning_protocol':ref(DESIGN/'model_pilot_uncapped_v2/protocol.json'),
        'problem':ref(DESIGN/'model_pilot_uncapped_v2/cases'/c['name']/'problem.pkl')}
    for tier,entry in [('low',low),('high',high)]:
        r=entry['record'];built=build_plan(resources,r['mean_u'],r['prices'],node_labels=pool['labels'],budget=None,policy='optimized_'+tier)
        path=root/'plans'/('optimized_'+tier+'.json');write(path,built['plan']);plans['optimized_'+tier]=ref(path);plan_audits['optimized_'+tier]=built['audit']
        H=built['plan']['hard_payment_bound']
        for phase in (-1,1):
            u,quote=rotation_curve(resources,H,phase)
            b=build_plan(resources,u,quote,node_labels=pool['labels'],budget=None,policy='rotation_'+tier)
            if abs(b['plan']['hard_payment_bound']-H)>1e-9:raise ValueError('rotation cost not matched before random realization')
            key='rotation_'+tier+('_neg' if phase<0 else '_pos');path=root/'plans'/(key+'.json')
            write(path,b['plan']);plans[key]=ref(path);plan_audits[key]=b['audit']
    zero=build_plan(resources,np.zeros((2,96)),[0.,0.],node_labels=pool['labels'],budget=None,policy='zero')
    path=root/'plans/zero.json';write(path,zero['plan']);plans['zero']=ref(path);plan_audits['zero']=zero['audit']
    # No producer, graph or test future has been loaded to construct these plans.
    identities,index=prior_requests();rng=np.random.default_rng(2026112201)
    trials=[]
    for i in range(12):
        profile=int(rng.integers(3));replicate=910000+97*i
        if ('paper15',profile,replicate) in identities:raise ValueError('new physical identity overlaps a saved request')
        z=np.random.default_rng(np.random.SeedSequence([2026112202,i])).uniform(-1,1,2).tolist()
        phase=int(np.random.default_rng(np.random.SeedSequence([2026112203,i])).choice([-1,1]))
        trials.append(dict(day=i,path=7100+i,profile_index=profile,replicate=replicate,factors=z,rotation_phase=phase))
    files=source_files(Path(declared['source_root']));sources={str(p):sha(p) for p in files}
    protocol=dict(status='frozen',id='frozen24h_actual_effect_v1',created_utc=now(),condition=c['name'],case='paper15',
        qualification=dict(qualified_conditions=1,original_conditions=5,not_qualified={'sample_target_infeasible':2,'sample_target_unresolved':2},
            rule='only planning-target-feasible quoted curves; no test-truth case filtering'),
        inputs=inputs,plans=plans,plan_audits=plan_audits,trials=trials,policies=list(POLICIES),
        selected_low_rule='minimum independently replayed expected cost among previously frozen planning-feasible curves; not global optimum',
        high_role='prelocked sensitivity; not chosen by test outcome',prior_request_identity_audit=index,
        planned_records=60,independent_simulated_days=12,slots=96,primary_rule=MAIN_RULE,
        full_history_rows=[288,288],actual_validation_rows=384,effective_weight_rows=480,
        budget=None,optimizer_payment_cap=None,payment_mode='frozen_plan_bound',
        comparison='24h alternating6h exploration scaled before execution to each optimized plan expected fee and conservative payment bound',
        source_sha256=sources,source_root=declared['source_root'],declared=declared,
        source_metadata=declared['source_metadata'],pool=c['pool'],
        gates=dict(worker=ref(worker_gate),driver=ref(driver_gate)),
        metric='Z=verified physical delivery AND M>=.2 max(-M_noDR,0)+1e-10; paired same-day counterfactual for evaluation only',
        stats='12 iid synthetic profile-mixture trials; per-policy two-sided95percent CP; primary one-sided binomial lower-tail test atp=.9; secondary descriptive',
        failure='no clipping/retry/resampling; unverified attainment0/12, structural error unknown retained; failed contract fee separate from delivered fee',
        scope='fixed historical RNJ+WZZT candidates and parameters; no new W acceptance/search/refit; truth after all scores frozen',
        worker_timeout_seconds=180,
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=version('pandas'),scipy=version('scipy')))
    write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(files):z.write(p,f'{i:03d}_{p.name}')
    runtimes={}
    for key,planref in plans.items():
        base=key.replace('_neg','').replace('_pos','');plan=read(checked(planref))
        runtime=dict(status='frozen',parent_protocol=ref(root/'protocol.json'),source_sha256=sources,
            allowed_voltage_sigma=[declared['voltage_sigma']],voltage_sigma=declared['voltage_sigma'],pq_sigma=declared['pq_sigma'],
            expected_resource_nodes=[r['node_label'] for r in resources],plans={base:planref},frozen_pool=c['pool'],
            payment_mode='frozen_plan_bound',optimizer_payment_cap=None,payment_bound=plan['hard_payment_bound'])
        path=root/'runtime'/f'{key}.json';write(path,runtime);runtimes[key]=ref(path)
    requests=[]
    for day in trials:
        for policy in POLICIES:
            variant=policy+('_pos' if day['rotation_phase']>0 else '_neg') if policy.startswith('rotation') else policy
            identity=f"paper15_day{day['day']:02d}_{policy}"
            request=dict(status='frozen',phase='test',case='paper15',record_id=identity,policy=policy,path=day['path'],
                replicate=day['replicate'],profile_index=day['profile_index'],day_offset=6+day['day'],
                labels=declared['labels'],root_label=declared['root_label'],base_kw=declared['base_kw'],factors=day['factors'],
                voltage_sigma=declared['voltage_sigma'],pq_sigma=declared['pq_sigma'],
                payment_mode='frozen_plan_bound',optimizer_payment_cap=None,payment_bound=read(checked(plans[variant]))['hard_payment_bound'],
                protocol=runtimes[variant],plan=plans[variant],frozen_pool=c['pool'],source_metadata=declared['source_metadata'],
                repo_root=str(REPO),source_root=declared['source_root'],source_sha256=sources)
            path=root/'requests'/f'{identity}.json';write(path,request)
            requests.append(dict(day=day['day'],policy=policy,record_id=identity,variant=variant,request=ref(path)))
    preflight=[]
    for q in requests:
        request=load_request(checked(q['request']),q['request']['sha256']);_,_,fees=audit_plan(request)
        preflight.append(dict(record_id=q['record_id'],status='passed',actual_model_fee=fees['actual_payment'],payment_bound=fees['payment_bound']))
    write(root/'preflight.json',dict(status='complete',passed=len(preflight),denominator=60,rows=preflight,no_graph_or_future_generated=True))
    write(root/'freeze.json',dict(status='frozen',created_utc=now(),protocol=ref(root/'protocol.json'),requests=requests,
        runtimes=runtimes,source_snapshot=ref(root/'source_snapshot.zip'),preflight=ref(root/'preflight.json')))
    write(root/'run_lock.json',dict(status='prepared',pid=os.getpid()))
    print(dict(stage='prepared',days=12,records=60,protocol_sha256=sha(root/'protocol.json')),flush=True)


def config(root):
    root=Path(root).resolve();p=read(root/'protocol.json');f=read(root/'freeze.json');checked(f['protocol']);checked(f['source_snapshot']);checked(f['preflight'])
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('source drift after freeze: '+path)
    return root,p,f


def observe(root):
    root,p,f=config(root);write(root/'run_lock.json',dict(status='running',pid=os.getpid(),started_utc=now()))
    rows=[];halt=False;env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONPATH']=os.pathsep.join([str(REPO),str(REPO/'rnj_wzzt_core')])
    for q in f['requests']:
        folder=root/'actual'/q['record_id'];rowfile=folder/'row.json'
        if rowfile.exists():rows.append(read(rowfile));continue
        folder.mkdir(parents=True,exist_ok=True);row=dict(q,status='started')
        if halt:
            row.update(status='not_attempted_system_block',error_scope='prior infrastructure failure retained')
        else:
            command=[sys.executable,'-u','-B','-m','research_experiments.wzzt_dr_selection.dr_frozen_distance.one_day_worker',
                '--request',str(checked(q['request'])),'--expected-sha',q['request']['sha256'],'--output',str(folder/'physical')]
            try:
                with (folder/'worker.log').open('w',encoding='utf-8') as log:
                    done=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=p['worker_timeout_seconds'])
                receipt=read(folder/'physical/events_receipt.json')
                row.update(status='observed' if receipt['status']=='complete' else receipt['status'],worker_exit=done.returncode,
                    receipt=ref(folder/'physical/events_receipt.json'),physical_folder=str(folder/'physical'),
                    packet=ref(folder/'physical/packet.pkl') if (folder/'physical/packet.pkl').exists() else None)
                if receipt['status']=='worker_failed':halt=True
            except Exception as error:
                row.update(status='infrastructure_unresolved',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc());halt=True
        write(rowfile,row);rows.append(row)
        write(root/'observation_progress.json',dict(status='running',rows=rows,planned=60))
        print(dict(stage='observed',done=len(rows),planned=60,day=q['day'],policy=q['policy'],status=row['status']),flush=True)
    write(root/'observations_frozen.json',dict(status='frozen',rows=rows,planned=60,freeze=ref(root/'freeze.json'),controller_truth_read=False))
    write(root/'run_lock.json',dict(status='observations_complete',pid=os.getpid()))


def score(root):
    root,p,f=config(root);ledger=read(root/'observations_frozen.json');checked(ledger['freeze']);pool=read(checked(p['pool']));rows=[]
    for original in ledger['rows']:
        row=dict(original)
        if row['status']=='observed':
            packet=unpickle(row['packet']);scored=weighted_score(pool,packet['validation'][0],history_gate=True)
            file=root/'scores'/f"{row['record_id']}.json";write(file,scored)
            row.update(status='predictions_frozen',score=ref(file),M=exact_margin(scored),distance=scored['new_prediction_pair_distance_l1'])
        rows.append(row)
    write(root/'predictions_frozen.json',dict(status='frozen',rows=rows,observations=ref(root/'observations_frozen.json'),truth_read=False))
    # First controller access to evaluator topology labels occurs below.
    for row in rows:
        if row['status']!='predictions_frozen':continue
        receipt=read(checked(row['receipt']));truthfile=Path(row['physical_folder'])/'truth_evaluation.json'
        checked(dict(path=str(truthfile),sha256=receipt['private_files']['truth_evaluation.json']['sha256']))
        truth={tuple(sorted(x)) for x in read(truthfile)['true_supports']};s=read(checked(row['score']))
        def errors(supports):
            proposed={tuple(sorted(x)) for x in supports}
            return dict(full_errors=len(proposed^truth),missing=[list(x) for x in sorted(truth-proposed)],extra=[list(x) for x in sorted(proposed-truth)])
        rule=s['primary_rule'];outcome=errors(rule['selected_output_support_labels'])
        row.update(status='complete',**outcome,selected_candidate=rule['selected_candidate_id'],
            coarse_excluded='coarse' not in rule['eligible_candidate_ids'],
            rule_outcomes=[dict(rule_id=q['rule_id'],selected=q['selected_candidate_id'],**errors(q['selected_output_support_labels'])) for q in s['rules']],
            candidate_errors={m['candidate_id']:errors(m['frozen_output_support_labels']) for m in pool['models']},
            delivered_fee=receipt['actual_payment'],all_operation_checks_pass=receipt['all_operation_checks_pass'])
    for row in rows:
        zero=next(x for x in rows if x['day']==row['day'] and x['policy']=='zero')
        row['attained']=False
        if row['status']==zero['status']=='complete':
            target=.2*max(-zero['M'],0.)+1.e-10
            row.update(baseline_M=zero['M'],delta_M=row['M']-zero['M'],target_M=target,
                attained=row['M']>=target-1.e-13,distance_gain=row['distance']-zero['distance'],
                paired_error_reduction=zero['full_errors']-row['full_errors'])
    write(root/'scored.json',dict(status='complete',rows=rows,planned=60,predictions_frozen=ref(root/'predictions_frozen.json')))
    tables=[]
    def mean_micro(group,key):
        values=[r[key] for r in group if key in r]
        return float(np.mean(values))*1e6 if values else None
    for policy in POLICIES:
        group=[r for r in rows if r['policy']==policy];valid=[r for r in group if r['status']=='complete'];k=sum(r['attained'] for r in group)
        tables.append(dict(policy=policy,planned=12,complete=len(valid),unresolved=12-len(valid),attained=k,attainment_ci95=cp_interval(k,12),
            recoveries=sum(r['full_errors']==0 for r in valid),known_full_errors=sum(r['full_errors'] for r in valid),
            full_errors=[r.get('full_errors') for r in group],
            mean_M_micro=mean_micro(valid,'M'),mean_delta_M_micro=mean_micro(valid,'delta_M'),
            mean_distance_gain_micro=mean_micro(valid,'distance_gain'),
            delivered_payment=sum(r['delivered_fee'] for r in valid),
            new_attainment_vs_zero=sum(r['attained'] and not next(z for z in rows if z['day']==r['day'] and z['policy']=='zero')['attained'] for r in group),
            lost_attainment_vs_zero=sum(not r['attained'] and next(z for z in rows if z['day']==r['day'] and z['policy']=='zero')['attained'] for r in group)))
    pairs=[]
    for tier in ('low','high'):
        pair=dict(tier=tier,improved=0,worsened=0,tied=0,unresolved=0,error_reduction_known=0)
        for day in range(12):
            a=next(r for r in rows if r['day']==day and r['policy']=='optimized_'+tier);b=next(r for r in rows if r['day']==day and r['policy']=='rotation_'+tier)
            if a['status']!= 'complete' or b['status']!='complete':pair['unresolved']+=1;continue
            diff=b['full_errors']-a['full_errors'];pair['error_reduction_known']+=diff
            pair['improved' if diff>0 else 'worsened' if diff<0 else 'tied']+=1
        pairs.append(pair)
    primary=next(x for x in tables if x['policy']=='optimized_low');k=primary['attained']
    pvalue=sum(math.comb(12,j)*(.9**j)*(.1**(12-j)) for j in range(k+1))
    write(root/'summary_private.json',dict(status='complete',tables=tables,pairs=pairs,records=rows,
        primary_one_sided_90pct_test=dict(successes=k,denominator=12,pvalue=pvalue,reject_at_05=pvalue<.05),
        unique_simulated_days=12,planned_records=60,completed_records=sum(r['status']=='complete' for r in rows),
        qualification=p['qualification'],new_fits=0,new_price_solves=0,new_W_search=0,
        scope='independent synthetic actual AC days for the sole planning-qualified case; no field or all-topology guarantee'))
    write(root/'ended.json',dict(status='complete',ended_utc=now(),summary=ref(root/'summary_private.json'),scored=ref(root/'scored.json')))
    write(root/'run_lock.json',dict(status='complete',pid=os.getpid()))
    print(dict(stage='scored',complete=sum(r['status']=='complete' for r in rows),planned=60,tables=tables,pairs=pairs),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','observe','score']);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--worker-gate',type=Path);a.add_argument('--driver-gate',type=Path);args=a.parse_args()
    if args.command=='prepare':prepare(args.output,args.worker_gate,args.driver_gate)
    elif args.command=='observe':observe(args.output)
    else:score(args.output)

if __name__=='__main__':main()
