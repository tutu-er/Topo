"""Bounded conditional-Gaussian pilot on the existing full-history case.

prepare -> solve -> evaluate -> labels.  No future/true topology in planning;
no candidate refit, new W search, clipping, response redraw or new case.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, hashlib, json, pickle, platform, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import beta

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
PRIVATE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
for folder in [REPO,REPO/'rnj_wzzt_core']:
    sys.path.insert(0,str(folder))
from problem import ConditionalProblem,integration_direction

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def checked(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('input drift: '+str(p))
    return p
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def dump(path,value):
    with Path(path).open('wb') as f:pickle.dump(value,f,protocol=4)
def load_pickle(path):
    with Path(path).open('rb') as f:return pickle.load(f)
def source_files():
    return list(Path(__file__).parent.glob('*.py'))+[
        REPO/'research_experiments/wzzt_dr_selection/dr_frozen_distance/frozen_pool.py',
        REPO/'research_experiments/wzzt_dr_selection/dr_newday_weighted/weighted_selection.py',
        REPO/'research_experiments/wzzt_dr_selection/dr_pool_v2/selection.py',
        REPO/'research_experiments/wzzt_dr_selection/dr_pool_v2/observations.py']
def lower_cp(k,n,delta=.05):return float(beta.ppf(delta,k,n-k+1)) if k else 0.

def prepare(root):
    from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import _power,_target
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    old=read(PRIVATE/'wzzt_gaussian_margin_stage/run_v1/protocol.json')
    inputs=old['inputs'];pool=read(checked(inputs['pool']))
    fit_ref=pool['historical_fit'];fit=read(checked(fit_ref))
    protocol=dict(id='conditional_gaussian_original3_v1',created_utc=now(),status='locked_before_prepare',
        inputs={**inputs,'historical_fit':fit_ref},case=old['condition'],history_rows=[288,288],slots=96,
        source_sha256={str(p.resolve()):sha(p) for p in source_files()},
        voltage_predictor='reuse historical_fit.fronts.raw ordered non-tree RX, all training, no refit',
        data_law='Gaussian [P,Q,e0] sample mean/covariance from three training days only; rank <= 2; no calibrated population claim',
        response='U=u*(1+sqrt(.025/3)*Z); independent resource-slot Gaussian in main case; shared multipliers across policies',
        cost='prepaid .25*sum(u^2/.5), bidirectional mean-volume contract',
        control_domain='same resource means [-1,1] kW; no budget cap, no energy equality; no clipping',
        rule=dict(historical_fraction=.03,new_fraction=.03,new_weight=2,epsilon=1e-12,history_gate=True),
        probability_target=.95,integration_samples=128,integration_seed=2026110201,
        qualification_samples=512,qualification_seed=2026110202,evaluation_paths=2000,evaluation_seed=2026110203,
        optimizer=dict(method='SLSQP with exact conditional-boundary derivatives of fixed integration approximation',
            starts=['zero','history_gradient_025'],maxiter=60,seconds_per_start=180,ftol=1e-9,
            fallback='retain best returned unqualified policy; no unbounded restart or seed extension'),
        policy_selection='cheapest among planning-feasible candidates with independent integration estimate >=.95; otherwise best qualification probability marked diagnostic',
        comparisons=['zero','selected_continuous_quote','same_cost_opposed_6h_rotation','same_cost_single_resource_0_if_feasible'],
        scopes=['no topology prior','no global optimality','no new AC','no new RNJ/W search','Gaussian parameter estimation uncertainty not certified'],
        truth_gate='read existing same-case truth only after all new selection results frozen',
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(source_files()):z.write(p,f'{i:02d}_{p.name}')
    write(root/'run_lock.json',dict(pid=os.getpid(),status='preparing',updated_utc=now()))
    packet=load_pickle(checked(inputs['history']));history=packet['history'];del packet
    resources=read(checked(inputs['resources']))['resources'];labels=pool['labels']
    if fit['terminal_labels']!=labels or len(resources)!=2:raise ValueError('identity/resource mismatch')
    models={v['candidate_id']:v for v in pool['models']}
    if set(models)!= {'coarse','refined'}:raise ValueError('locked historical pair changed')
    r0=np.asarray(fit['fronts']['raw']['R']);x0=np.asarray(fit['fronts']['raw']['X'])
    rc,xc=(np.asarray(models['coarse'][k]) for k in ('R','X'))
    rf,xf=(np.asarray(models['refined'][k]) for k in ('R','X'))
    train=[];h={k:[] for k in models}
    for s in history['training']:
        p,q=_power(s,labels);y=_target(s,p.shape)
        if p.shape!=(96,len(labels)):raise ValueError('training shape')
        train.append(np.stack([p,q,y-p@r0.T-q@x0.T]))
    for s in history['validation']:
        p,q=_power(s,labels);y=_target(s,p.shape)
        for k,mod in models.items():h[k].append(float(np.abs(y-p@np.asarray(mod['R']).T-q@np.asarray(mod['X']).T).mean())*1e6)
    if len(train)!=3 or any(len(v)!=3 for v in h.values()):raise ValueError('all three full days required')
    h={k:float(np.mean(v)) for k,v in h.items()}
    for k,mod in models.items():
        if not np.isclose(h[k]/1e6,mod['historical_validation_mae'],rtol=1e-11,atol=1e-12):raise AssertionError('historical loss drift')
    hmin=min(h.values());eligible=[k for k in pool['selection_order'] if h[k]<=1.03*hmin+1e-6]
    if set(eligible)!=set(models):raise ValueError('pair must both survive historical gate')
    a=np.stack(train);mu=a.mean(axis=0);noise=(a-mu).reshape(3,-1).T/np.sqrt(2.)
    residuals={}
    for k,(r,x) in {'C':(rc,xc),'F':(rf,xf)}.items():
        rr=a[:,2]+np.einsum('btn,jn->btj',a[:,0],r0-r)+np.einsum('btn,jn->btj',a[:,1],x0-x)
        rr=rr.reshape(3,-1)*1e6;mm=rr.mean(axis=0)
        residuals['mu_'+k]=mm;residuals['L_'+k]=(rr-mm).T/np.sqrt(2.)
    effects={}
    for k,(r,x) in {'C':(rc,xc),'F':(rf,xf)}.items():
        effects['effect_'+k]=np.stack([(r0-r)[:,labels.index(v['node_label'])]+.35*(x0-x)[:,labels.index(v['node_label'])] for v in resources],axis=1)*1e6/pool['base_kw']
    direction,complement=integration_direction(**dict(mu_c=residuals['mu_C'],mu_f=residuals['mu_F'],lc=residuals['L_C'],lf=residuals['L_F'],kappa=1.03))
    lower=[];upper=[]
    for r in resources:
        base=np.broadcast_to(np.asarray(r['baseline_kw']),96)
        lower.append(np.maximum(np.asarray(r.get('power_lower_kw',0))-base,-1.))
        upper.append(np.minimum(np.asarray(r['power_upper_kw'])-base,1.))
    out=dict(**residuals,**effects,mu=mu,noise=noise,r0=r0,x0=x0,H=h,pool=pool,resources=resources,
        labels=labels,shape=(2,96),cv=np.full((2,96),np.sqrt(.025/3)),elasticities=np.full((2,96),.5),dt=.25,
        lower=np.stack(lower),upper=np.stack(upper),kappa=1.03,threshold=1.5*(1.03*h['refined']-h['coarse'])+2.5e-6,
        direction=direction,complement=complement,response_factor=None)
    dump(root/'problem.pkl',out)
    write(root/'problem_frozen.json',dict(status='frozen',input=ref(root/'problem.pkl'),protocol=ref(root/'protocol.json'),
        historical_losses_micro=h,threshold_micro=out['threshold'],history_eligible=eligible,
        covariance_rank=int(np.linalg.matrix_rank(noise)),controls=192,observations=1440,
        control_labels=[r['node_label'] for r in resources],
        reference_minus_F_frobenius=float(np.linalg.norm(r0-rf)+np.linalg.norm(x0-xf)),
        response_effect_norms={k:float(np.linalg.norm(v)) for k,v in effects.items()},
        source_fit_reused=True,new_fit_calls=0,new_search_calls=0,truth_read=False))
    write(root/'run_lock.json',dict(pid=None,status='prepared',updated_utc=now()))
    print(json.dumps(read(root/'problem_frozen.json'),ensure_ascii=False),flush=True)

def load(root):
    root=Path(root);protocol=read(root/'protocol.json')
    for p,h in protocol['source_sha256'].items():
        if sha(p)!=h:raise ValueError('source drift '+p)
    checked(read(root/'problem_frozen.json')['input'])
    return root,protocol,load_pickle(root/'problem.pkl')

def derivative_audit(problem,seed=2026110210):
    rng=np.random.default_rng(seed);d=int(np.prod(problem.shape))
    u=rng.uniform(-.15,.15,d);r=problem.evaluate(u)
    if r['gradient'] is None:return dict(passed=False,invalid_indices=r['invalid_indices'])
    rows=[]
    for _ in range(4):
        direction=rng.normal(size=d);direction/=np.linalg.norm(direction);eps=2e-5
        fd=(problem.evaluate(u+eps*direction)['probability']-problem.evaluate(u-eps*direction)['probability'])/(2*eps)
        an=float(r['gradient']@direction)
        rows.append(dict(analytic=an,finite_difference=fd,error=abs(fd-an)))
    return dict(passed=all(v['error']<2e-5+2e-3*abs(v['analytic']) for v in rows),rows=rows,
        scope='derivative of fixed conditional sample average, not proof of population gradient interchange')

class StopBudget(Exception):pass

def solve(root):
    root,protocol,p=load(root)
    if (root/'price_freeze.json').exists():raise RuntimeError('price already frozen; do not rerun')
    write(root/'run_lock.json',dict(pid=os.getpid(),status='optimizing',updated_utc=now()))
    problem=ConditionalProblem(p,protocol['integration_samples'],protocol['integration_seed'])
    check=derivative_audit(problem);write(root/'gradient_audit.json',check)
    if not check['passed']:raise RuntimeError('finite integration gradient audit failed')
    zero=np.zeros(np.prod(p['shape']));r0=problem.evaluate(zero)
    if r0['gradient'] is None:raise RuntimeError('zero plan conditional gradient invalid')
    starts={'zero':zero,'history_gradient_025':.25*np.sign(r0['gradient'])}
    bounds=list(zip(p['lower'].ravel(),p['upper'].ravel()))
    results=[]
    for name,start in starts.items():
        path=root/f'solve_{name}.json'
        if path.exists():results.append(read(path));continue
        began=time.perf_counter();trace=[];best=None
        def constraint(u):
            nonlocal best
            if time.perf_counter()-began>protocol['optimizer']['seconds_per_start']:raise StopBudget()
            r=problem.evaluate(u)
            if r['gradient'] is None:raise ArithmeticError('conditional derivative unavailable at iterate')
            cost=problem.cost(u);prob=r['probability']
            if best is None or (prob>=.95-1e-7 and (best['probability']<.95-1e-7 or cost<best['cost'])) or (best['probability']<.95-1e-7 and prob>best['probability']):
                best=dict(mean_u=np.asarray(u).reshape(p['shape']).tolist(),probability=prob,cost=cost)
            return prob-.95
        def jac(u):
            r=problem.evaluate(u)
            if r['gradient'] is None:raise ArithmeticError('conditional derivative unavailable')
            return r['gradient']
        def callback(u):
            r=problem.evaluate(u);trace.append(dict(iteration=len(trace)+1,cost=problem.cost(u),probability=r['probability'],elapsed=time.perf_counter()-began))
            write(root/f'checkpoint_{name}.json',dict(status='running',best=best,trace=trace))
        try:
            result=minimize(problem.cost,start,jac=problem.cost_gradient,method='SLSQP',bounds=bounds,
                constraints=[dict(type='ineq',fun=constraint,jac=jac)],callback=callback,
                options=dict(maxiter=protocol['optimizer']['maxiter'],ftol=protocol['optimizer']['ftol']))
            constraint(result.x)
            status='local_success' if result.success else 'solver_unresolved'
            info=dict(solver_status=int(result.status),message=str(result.message),nit=int(result.nit))
        except StopBudget:
            status='time_limit';info=dict(message='predeclared wall budget reached')
        except Exception as exc:
            status='failed';info=dict(message=repr(exc))
        record=dict(start=name,status=status,**info,elapsed_seconds=time.perf_counter()-began,best=best,trace=trace,
            global_optimum_proven=False,global_lower_bound=None)
        write(path,record);results.append(record)
        print('solve',name,status,'best',None if best is None else (best['probability'],best['cost']),flush=True)
    qualification=ConditionalProblem(p,protocol['qualification_samples'],protocol['qualification_seed'])
    candidates=[]
    for r in results:
        if r['best'] is None:continue
        u=np.asarray(r['best']['mean_u']);rr=qualification.evaluate(u)
        candidates.append(dict(start=r['start'],mean_u=u.tolist(),cost=problem.cost(u),planning_probability=r['best']['probability'],
            qualification_probability=rr['probability'],qualification_se=rr['conditional_mc_se'],
            qualification_gradient_invalid=len(rr['invalid_indices']),numerically_qualified=bool(rr['probability']>=.95 and r['best']['probability']>=.95-1e-7)))
    if not candidates:raise RuntimeError('no returned continuous policy; failures retained')
    qualified=[r for r in candidates if r['numerically_qualified']]
    selected=min(qualified,key=lambda r:r['cost']) if qualified else max(candidates,key=lambda r:r['qualification_probability'])
    u=np.asarray(selected['mean_u']);amp=np.sqrt(np.mean(u*u));pattern=np.where((np.arange(96)//24)%2==0,1.,-1.)
    policies=dict(zero=dict(mean_u=zero.reshape(p['shape']).tolist(),role='no reward natural new day'),
        continuous=dict(mean_u=u.tolist(),role='numerically qualified local quote' if qualified else 'unqualified best returned diagnostic'),
        rotation=dict(mean_u=(amp*np.stack([pattern,-pattern])).tolist(),role='same planned cost, predeclared six-hour rotation'))
    single=np.zeros(p['shape']);single[0]=np.sqrt(2)*amp
    baseline_skips=[]
    if np.all(single<=p['upper']) and np.all(single>=p['lower']):
        policies['single']=dict(mean_u=single.tolist(),role='same cost, predetermined first-resource all-day increase')
    else:baseline_skips.append(dict(policy='single',status='infeasible_at_matched_cost',reason='original mean bound; no clipping'))
    for name,policy in policies.items():
        a=np.asarray(policy['mean_u'])
        if np.any(a<p['lower']-1e-9) or np.any(a>p['upper']+1e-9):raise ValueError('policy outside original mean bounds: '+name)
        policy['planned_cost']=problem.cost(a)
        if name not in ('zero','continuous') and not np.isclose(policy['planned_cost'],selected['cost'],rtol=1e-10):raise AssertionError('cost matching')
    write(root/'planning_results.json',dict(results=results,candidates=candidates,selected=selected,baseline_skips=baseline_skips))
    write(root/'price_freeze.json',dict(status='frozen',created_utc=now(),policies=policies,numerically_qualified=bool(qualified),
        selected_start=selected['start'],planning_results=ref(root/'planning_results.json'),
        independent_evaluation_stream_not_opened=True,probability_target=.95))
    write(root/'run_lock.json',dict(pid=None,status='prices_frozen',updated_utc=now()))

def evaluate(root):
    from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score
    root,protocol,p=load(root)
    if (root/'predictions_frozen.json').exists():raise RuntimeError('evaluation complete; do not redraw')
    freeze=read(root/'price_freeze.json');N=protocol['evaluation_paths'];rng=np.random.default_rng(protocol['evaluation_seed'])
    zbase=rng.normal(size=(N,p['noise'].shape[1]));zdr=rng.normal(size=(N,)+p['shape'])
    np.savez_compressed(root/'evaluation_draws.npz',zbase=zbase,zdr=zdr)
    write(root/'run_lock.json',dict(pid=os.getpid(),status='evaluating',updated_utc=now()))
    base=(p['mu'].ravel()[None]+zbase@p['noise'].T).reshape(N,3,96,len(p['labels']))
    rc0=p['mu_C'][None]+zbase@p['L_C'].T;rf0=p['mu_F'][None]+zbase@p['L_F'].T
    D0=np.abs(rc0).mean(1)-p['kappa']*np.abs(rf0).mean(1)
    records={};summaries=[];calls=0;max_loss_error=0.;max_shared_error=0.
    for name,policy in freeze['policies'].items():
        u=np.asarray(policy['mean_u']);U=u[None]*(1+p['cv'][None]*zdr)
        rc=rc0+np.einsum('nr,srt->stn',p['effect_C'],U).reshape(N,-1)
        rf=rf0+np.einsum('nr,srt->stn',p['effect_F'],U).reshape(N,-1)
        ec=np.abs(rc).mean(1);ef=np.abs(rf).mean(1);D=ec-p['kappa']*ef;event=D>p['threshold']
        choices=[];coarse_exits=[];raw=[];final=[];minp=[]
        for i in range(N):
            pp=base[i,0].copy();qq=base[i,1].copy()
            for j,r in enumerate(p['resources']):
                idx=p['labels'].index(r['node_label']);pp[:,idx]+=U[i,j]/p['pool']['base_kw'];qq[:,idx]+=.35*U[i,j]/p['pool']['base_kw']
            yy=pp@p['r0'].T+qq@p['x0'].T+base[i,2]
            scene=dict(P_terminal=pp,Q_terminal=qq,drop_target=yy,labels=p['labels'])
            out=score(p['pool'],scene,history_gate=True)
            selected=next(v for v in out['rules'] if v['new_day_weight']==2 and v['new_day_fraction']==.03)
            losses={v['candidate_id']:v['new_validation_mae']*1e6 for v in out['loss_rows']}
            max_loss_error=max(max_loss_error,abs(losses['coarse']-ec[i]),abs(losses['refined']-ef[i]))
            byid={v['candidate_id']:v for v in p['pool']['models']}
            expected=pp@(np.asarray(byid['refined']['R'])-np.asarray(byid['coarse']['R'])).T+qq@(np.asarray(byid['refined']['X'])-np.asarray(byid['coarse']['X'])).T
            max_shared_error=max(max_shared_error,float(np.max(np.abs((rc[i]-rf[i]).reshape(96,-1)-expected*1e6))))
            choices.append(selected['selected_candidate_id']);coarse_exits.append('coarse' not in selected['eligible_candidate_ids'])
            raw.append(selected['selected_raw_support_count']);final.append(selected['selected_output_support_count']);minp.append(float(pp.min()));calls+=1
        if not np.array_equal(event,coarse_exits):raise AssertionError('original3% event != existing selection exclusion')
        violations=np.any((U<p['lower'][None]-1e-10)|(U>p['upper'][None]+1e-10),axis=(1,2))
        background_negative=np.any(base[:,0]<0,axis=(1,2));passed=int(event.sum())
        records[name]=dict(D=D,D0=D0,event=event,selected=np.asarray(choices),resource_violation=violations,
            raw_support_count=np.asarray(raw),final_support_count=np.asarray(final),minimum_post_power=np.asarray(minp))
        summaries.append(dict(policy=name,role=policy['role'],paths=N,completed=N,failed=0,timeout=0,
            event_count=passed,event_rate=passed/N,cp_lower95=lower_cp(passed,N),cp_lower_bonferroni95=lower_cp(passed,N,.05/len(freeze['policies'])),
            target_certified_95=bool(lower_cp(passed,N)>=.95),coarse_excluded=int(sum(coarse_exits)),refined_selected=choices.count('refined'),
            planned_cost=policy['planned_cost'],mean_D=float(D.mean()),mean_increment=float(np.mean(D-D0)),threshold=p['threshold'],
            q05_margin=float(np.quantile(D-p['threshold'],.05)),mean_margin=float(np.mean(D-p['threshold'])),
            new_successes_vs_zero=int(np.sum(event&(D0<=p['threshold']))),lost_successes_vs_zero=int(np.sum(~event&(D0>p['threshold']))),
            resource_violation_paths=int(violations.sum()),event_and_resource_valid=int(np.sum(event&~violations)),
            negative_background_paths=int(background_negative.sum()),negative_post_power_paths=int(np.sum(np.asarray(minp)<0))))
        print('evaluated',name,passed,'/',N,'cost',policy['planned_cost'],'resource_bad',int(violations.sum()),flush=True)
    if max_loss_error>1e-7 or max_shared_error>1e-7:raise AssertionError('shared residual or loss unit mismatch')
    np.savez_compressed(root/'evaluation_scalars.npz',**{name+'__'+k:v for name,row in records.items() for k,v in row.items()})
    write(root/'predictions_frozen.json',dict(status='frozen',created_utc=now(),price_freeze=ref(root/'price_freeze.json'),
        rows=summaries,selection_calls=calls,max_loss_error_micro=max_loss_error,max_shared_identity_error_micro=max_shared_error,
        truth_read=False,new_fit_calls=0,new_search_calls=0,new_AC_calls=0,independent_denominator=N,
        rule='existing weighted_selection w2/tau3 explicitly selected; default primary tau2 not used',
        interpretation='frozen-candidate conditional Gaussian model replay, not AC/field recovery'))
    write(root/'run_lock.json',dict(pid=None,status='predictions_frozen',updated_utc=now()))

def labels(root):
    root,protocol,p=load(root);pred=read(root/'predictions_frozen.json')
    checked(pred['price_freeze']);assert pred['truth_read'] is False
    truthpath=next((PRIVATE/'wzzt_day_effect_stage/actual_v1/actual/paper15_day00_zero').rglob('truth_evaluation.json'))
    truth=read(truthpath);assert truth['terminal_labels']==p['labels']
    truthset={tuple(sorted(map(int,x))) for x in truth['true_supports']}
    def error(s):
        s=set(s);return dict(missing=len(truthset-s),extra=len(s-truthset),errors=len(truthset^s),complete=bool(s==truthset))
    models={}
    for model in p['pool']['models']:
        raw={tuple(sorted(p['labels'][int(i)] for i in block)) for block in model['point']['supports']}
        final={tuple(sorted(map(int,b))) for b in model['frozen_output_support_labels']}
        models[model['candidate_id']]=dict(raw=error(raw),final=error(final))
    arr=np.load(root/'evaluation_scalars.npz');rows=[]
    zero=np.asarray([models[str(k)]['final']['errors'] for k in arr['zero__selected']])
    for s in pred['rows']:
        selected=arr[s['policy']+'__selected'];errs=np.asarray([models[str(k)]['final']['errors'] for k in selected])
        rawerrs=np.asarray([models[str(k)]['raw']['errors'] for k in selected])
        rows.append(dict(**s,final_errors_sum=int(errs.sum()),raw_errors_sum=int(rawerrs.sum()),full_recoveries=int(np.sum(errs==0)),
            improved_vs_zero=int(np.sum(errs<zero)),worse_vs_zero=int(np.sum(errs>zero)),equal_vs_zero=int(np.sum(errs==zero))))
    out=dict(status='complete',rows=rows,models=models,truth_source=ref(truthpath),predictions=ref(root/'predictions_frozen.json'),
        scope='same old physical network labels only; new observations are Gaussian working-model simulations',independent_paths=protocol['evaluation_paths'])
    write(root/'summary.json',out)
    write(root/'ended.json',dict(status='complete',summary=ref(root/'summary.json'),new_AC_calls=0))
    write(root/'run_lock.json',dict(pid=None,status='complete',updated_utc=now()))
    print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','solve','evaluate','labels']);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    try:globals()[args.command](args.output)
    except Exception as exc:
        if args.output.exists():write(args.output/f'failure_{args.command}_{int(time.time())}.json',dict(status='failed',error=repr(exc),created_utc=now()))
        raise
