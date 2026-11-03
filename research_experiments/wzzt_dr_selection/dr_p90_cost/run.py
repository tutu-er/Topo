"""Bounded 90% working-probability cost test; all physical/selection code reused."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
import sys,argparse,shutil,zipfile,time,traceback,platform
from pathlib import Path
import numpy as np
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
TASK=REPO/'research_experiments/wzzt_dr_selection'
sys.path.insert(0,str(TASK/'dr_probability_refinement'))
import refine as r
import driver
from cost_only import solve
BASE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_dr_probability_refinement_stage/run_v1')
JOINT=BASE.parent.parent/'wzzt_dr_joint_loss_methods_stage/run_v1'

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    p=r.read(BASE/'protocol.json');f=r.read(BASE/'price_freeze.json');r.checked(f['protocol'])
    if r.read(BASE/'run_lock.json')['status']!='complete':raise RuntimeError('parent still active')
    oldbase=r.checked(p['previous']).parent
    src=set(driver.sources())|{Path(r.__file__),Path(r.__file__).with_name('tail.py')}
    src|=set(Path(__file__).parent.glob('*.py'))
    p.update(id='p90_controlled_cost_v1',created_utc=r.now(),previous=r.ref(BASE/'price_freeze.json'),
        older_base=str(oldbase),parent_complete=r.ref(BASE/'run_lock.json'),
        source_sha256={str(v.resolve()):r.sha(v) for v in sorted(src)},
        planning_samples=128,planning_seed=2026130101,response_seed=2026131101,
        qualification_samples=4096,qualification_seed=2026132101,qualification_response_seed=2026133101,
        planning_target=.90,optimizer=dict(maxiter=40,seconds=180,starts='cheapest qualifying existing quote only'),
        primary_policies=['zero','rotation','old_diagnostic','candidate','nominal_milp'],policies=5,expected_records=360,
        paths=[dict(day=j,profile_index=j%3,replicate=1700000+197*j,response_seed=2026134101+j) for j in range(12)],
        independent_backgrounds_per_case=12,laws=['main','stress'],
        scientific_changes=['90 percent excitation target','one cost-only bounded local stage','nominal MILP output as explicit non-probabilistic comparator'],
        qualification='new independent working-model estimate, SE and Hoeffding bound; never feedback to price',
        actual_scope='same3 cases;12 new independent background days each;existing causal device,AC,noise,frozen output',
        stop_rule='one cost attempt only;no feasible start retained diagnostic,not infeasibility proof;no reprice after holdout',
        no_sampling_claim=False,global_optimal=False,tests_before_protocol=3,
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()))
    inputs=[r.ref(BASE/'protocol.json'),r.ref(BASE/'price_freeze.json'),r.ref(JOINT/'summary.json')]
    for c in p['cases']:
        folder=out/c['name'];folder.mkdir()
        for fn in ['case_protocol.json','pool.json','problem.pkl']:
            srcfile=BASE/c['name']/fn;inputs.append(r.ref(srcfile));shutil.copy2(srcfile,folder/fn)
        for srcfile in [BASE/c['name']/'solver_private.pkl',oldbase/c['name']/'solver_private.pkl',BASE/c['name']/'plans.pkl']:
            inputs.append(r.ref(srcfile))
        r.write(folder/'inputs.json',{fn:r.ref(folder/fn) for fn in ['case_protocol.json','pool.json','problem.pkl']})
    p['locked_inputs']=inputs;r.write(out/'protocol.json',p)
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,path in enumerate(p['source_sha256']):z.write(path,f'{i:03d}_{Path(path).name}')
    r.write(out/'run_lock.json',dict(status='prepared',pid=None))

def check_running(root,phase):
    q=r.read(root/'run_lock.json')
    if q.get('pid') and q['pid']!=os.getpid():raise RuntimeError('check previous process and lock before restart')
    r.write(root/'run_lock.json',dict(status=phase,pid=os.getpid()))

def optimize(out):
    root,p,cfg,alloc=r.config(out)
    if (root/'price_freeze.json').exists():print('prices already frozen');return
    check_running(root,'optimizing');rows=[];freezes={}
    for q in p['locked_inputs']:r.checked(q)
    nominal={v['case']:v for v in r.read(JOINT/'summary.json')['rows'] if v['policy']=='candidate'}
    for ci,c in enumerate(p['cases']):
        folder=root/c['name'];receipt=folder/'price_receipt.json'
        if receipt.exists():
            q=r.read(receipt);r.checked(q['plans']);rows.append(q['summary']);freezes[c['name']]=q;continue
        packet=r.load(folder/'problem.pkl');dev=r.Devices(alloc[c['case']],cfg,r.profiles)
        old=r.load(Path(p['older_base'])/c['name']/'solver_private.pkl')['best']
        current=r.load(BASE/c['name']/'solver_private.pkl')['best']
        prob=r.Probability(packet,dev,p['planning_samples'],p['planning_seed']+1000*ci,p['response_seed']+1000*ci)
        starts=[('zero',np.zeros((dev.R,24))),('old_diagnostic',old['x']),('previous_candidate',current['x'])]
        sol=solve(prob,starts,target=p['planning_target'],**{k:p['optimizer'][k] for k in ['maxiter','seconds']})
        r.dump(folder/'solver_private.pkl',sol);best=sol['best'];fee=best['cost']
        rot=r.simple_plan(dev,'rotation');rc=dev.fees(rot)
        rot*=min(1.,np.sqrt(fee/rc)) if rc else 0.
        alpha=nominal[c['name']]['nominal_milp']['alpha']
        milp=np.zeros_like(best['mean_u']) if alpha is None else alpha*current['mean_u']
        plans=dict(zero=dict(mean_u=np.zeros_like(best['mean_u']),cost=0.),
            rotation=dict(mean_u=rot,cost=dev.fees(rot)),
            old_diagnostic=dict(mean_u=old['mean_u'],cost=old['cost']),
            candidate=dict(mean_u=best['mean_u'],cost=fee),
            nominal_milp=dict(mean_u=milp,cost=dev.fees(milp),alpha=alpha,
                probability_constraint=False,status='no_nominal_solution_zero_fallback' if alpha is None else 'nominal_ray_optimum'))
        for name,item in plans.items():
            item['nominal_audit']=dev.audit(item['mean_u'])
            if not item['nominal_audit']['passed']:raise ArithmeticError('nominal infeasible '+name)
        r.dump(folder/'plans.pkl',plans)
        s=dict(condition=c['name'],target=p['planning_target'],planning_probability=best['probability'],
            qualified=sol['qualified'],status=sol['status'],solver_success=sol['solver_success'],
            start_probabilities=[{k:v for k,v in z.items() if k in ['label','probability','cost','invalid']} for z in sol['records'][:3]],
            costs={k:v['cost'] for k,v in plans.items()},nominal_milp_alpha=alpha,
            retained_iterates=len(sol['records']),seconds=sol['seconds'],global_optimal=False,infeasibility_proven=False)
        q=dict(summary=s,plans=r.ref(folder/'plans.pkl'),solution=r.ref(folder/'solver_private.pkl'))
        r.write(receipt,q);rows.append(s);freezes[c['name']]=q
        print('FROZEN',c['name'],s['qualified'],s['planning_probability'],fee,s['status'],flush=True)
    r.write(root/'planning_summary.json',dict(rows=rows))
    r.write(root/'price_freeze.json',dict(protocol=r.ref(root/'protocol.json'),cases=freezes,status='frozen',truth_read=False))
    r.write(root/'run_lock.json',dict(status='prices_frozen',pid=None))

def qualify(out):
    root,p,cfg,alloc=r.config(out);f=r.read(root/'price_freeze.json');r.checked(f['protocol'])
    if (root/'qualification_summary.json').exists():print('qualification complete');return
    check_running(root,'qualifying');rows=[]
    for ci,c in enumerate(p['cases']):
        dest=root/c['name']/'qualification.json'
        if dest.exists():rows.extend(r.read(dest)['rows']);continue
        dev=r.Devices(alloc[c['case']],cfg,r.profiles);packet=r.load(root/c['name']/'problem.pkl')
        plans=r.load(r.checked(f['cases'][c['name']]['plans']));N=p['qualification_samples']
        vals={k:[] for k in plans};beg=time.perf_counter()
        for batch in range((N+63)//64):
            count=min(64,N-64*batch);prob=r.Probability(packet,dev,count,p['qualification_seed']+ci*1000+batch,p['qualification_response_seed']+ci*1000+batch)
            cache={}
            for name,item in plans.items():
                u=item['mean_u'];key=np.where(u==0.,0.,u).tobytes()
                if key not in cache:
                    U=dev.project(u[None]*prob.multiplier)
                    a=prob.a0+np.concatenate([np.einsum('nr,srt->stn',packet['effect_'+k],U).reshape(count,-1) for k in ['C','F']],axis=1)
                    cache[key]=[r.conditional_probability(v,prob.b,prob.w,packet['threshold'],coefficient_gradients=False).probability for v in a]
                vals[name].extend(cache[key])
            if batch%16==0:print('QUAL',c['name'],batch*64,flush=True)
        case=[]
        for name,a in vals.items():
            a=np.asarray(a);mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(N))
            case.append(dict(condition=c['name'],policy=name,samples=N,probability=mean,standard_error=se,
                fixed_policy_95_lower=max(0.,mean-np.sqrt(np.log(20)/(2*N))),
                simultaneous15_95_lower=max(0.,mean-np.sqrt(np.log(300)/(2*N))),
                target=p['planning_target'],estimate_meets_target=mean>=p['planning_target'],cost=plans[name]['cost'],
                used_for_search=False,scope='assumed controlled working distribution,not real topology success'))
        r.write(dest,dict(rows=case,seconds=time.perf_counter()-beg));rows.extend(case)
        print('QUALIFIED',c['name'],[(v['policy'],round(v['probability'],5)) for v in case],flush=True)
    r.write(root/'qualification_summary.json',dict(rows=rows,price_freeze=r.ref(root/'price_freeze.json')))
    r.write(root/'run_lock.json',dict(status='qualified',pid=None))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','optimize','qualify','observe','evaluate']);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    try:
        if a.phase in ['observe','evaluate']:
            if (a.output/'ended.json').exists():print('physical run complete; nothing repeated')
            else:getattr(r,a.phase)(a.output)
        else:globals()[a.phase](a.output)
    except Exception as e:
        if a.output.exists():r.write(a.output/f'failure_{a.phase}_{int(time.time())}.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
