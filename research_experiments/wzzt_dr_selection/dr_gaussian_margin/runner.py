"""One history-only Gaussian SOCP pilot and independent working-law replay.

No truth, old actual-day residuals or future forecasts enter planning. The
three training days define a low-rank Gaussian assumption, not calibration.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle,platform,sys,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.stats import norm,beta
from gaussian_socp import solve_coefficients,evaluate_coefficients,box_bound

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
PRIVATE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
OLD=PRIVATE/'wzzt_day_curve_stage/model_pilot_uncapped_v2/protocol.json'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(v):
    p=Path(v['path'])
    if sha(p)!=v['sha256']:raise ValueError('input drift '+str(p))
    return p
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def abs_mean(mu,sd):
    z=np.divide(mu,sd,out=np.zeros_like(mu),where=sd>0)
    return np.where(sd>0,2*sd*norm.pdf(z)+mu*(2*norm.cdf(z)-1),np.abs(mu))
def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def prepare(root):
    from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import _power,_target
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    old=read(OLD);c=old['cases'][0]
    if c['name']!='paper15_b2_v0005':raise ValueError('case boundary')
    inputs={k:c[k] for k in ('history','pool')};inputs['resources']=c['old_plans']['zero']
    sources=list(Path(__file__).parent.glob('*.py'))+[REPO/'research_experiments/wzzt_dr_selection/dr_frozen_distance/frozen_pool.py',REPO/'research_experiments/wzzt_dr_selection/dr_newday_weighted/weighted_selection.py']
    protocol=dict(status='frozen',id='gaussian_original_margin_v1',created_utc=now(),condition=c['name'],inputs=inputs,
        source_sha256={str(p.resolve()):sha(p) for p in sources},history_rows=[288,288],slots=96,
        optimizer='continuous SOCP, no scenarios, no integer variables, fixed signs plus reference flip compensation',
        data_law='joint (P,Q,reference residual) Gaussian; slotwise mean and 3-day sample covariance, rank<=2',
        response='independent per-resource-slot Gaussian: U=u+sqrt(.025/3)*abs(u)*Z; Q=.35U',
        reference_response='Yafter=Ybaseline+mu_F(DR); reference residual distribution fixed, not a true-topology assertion',
        prices='abs(u)/.5 per slot, direction chosen by signed mean u',
        settlement='planned bidirectional regulation, dt*sum(abs(u)^2/.5), prepaid deterministic conditional on plan',
        budget=None,exact_daily_energy_constraint=False,aggregate_power_constraint=False,
        nominal_power_box='original component [0,2] around baseline1, actual Gaussian excursions retained',
        risk_G=.05,risk_F=.05,reference_risk_allocation='uniform .05/(96*15)',
        target='.2*max(-analytic_E_original_M_zero,0)+.0001 micro; fixed before price solve',
        evaluation_paths=2000,evaluation_seed=2026112401,solver_seconds=30,
        policy_rule='zero; corrected feasible mincost if any; naive signed mincost if any; corrected max-bound diagnostic; same-planned-cost 6h opposite rotation',
        ac_gate='only a corrected original-form target-feasible mincost quote qualifies for a separately frozen AC batch',
        old_actual_days_used=False,truth_used_for_design=False,
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__))
    write(root/'protocol.json',protocol)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,'source/'+p.name)
    write(root/'run_lock.json',dict(status='preparing',created_utc=now()))
    with checked(inputs['history']).open('rb') as f:container=pickle.load(f)
    history=container['history'];del container
    pool=read(checked(inputs['pool']));resources=read(checked(inputs['resources']))['resources']
    labels=pool['labels'];n=len(labels);models={m['candidate_id']:m for m in pool['models']}
    if len(models)!=2 or [len(history[k]) for k in ('training','validation')]!=[3,3]:raise ValueError('full historical domain')
    Rf=np.array(models['refined']['R']);Xf=np.array(models['refined']['X'])
    Rc=np.array(models['coarse']['R']);Xc=np.array(models['coarse']['X'])
    arrays=[];Hs={k:[] for k in models}
    for scene in history['training']:
        p,q=_power(scene,labels);y=_target(scene,p.shape)
        if p.shape!=(96,n):raise ValueError('day shape')
        arrays.append(np.stack([p,q,y-p@Rf.T-q@Xf.T]))
    for scene in history['validation']:
        p,q=_power(scene,labels);y=_target(scene,p.shape)
        for k,m in models.items():Hs[k].append(float(np.abs(y-p@np.array(m['R']).T-q@np.array(m['X']).T).mean())*1e6)
    A=np.stack(arrays);mu=A.mean(axis=0);L=(A-mu).reshape(3,-1).T/np.sqrt(2.)
    rf=A[:,2]*1e6;rc=rf+np.einsum('btn,jn->btj',A[:,0],Rf-Rc)*1e6+np.einsum('btn,jn->btj',A[:,1],Xf-Xc)*1e6
    muc=rc.mean(axis=0).ravel();muf=rf.mean(axis=0).ravel()
    Lc=(rc.reshape(3,-1)-muc).T/np.sqrt(2.);Lf=(rf.reshape(3,-1)-muf).T/np.sqrt(2.)
    sc=np.where(muc>=0,1.,-1.);sf=np.where(muf>=0,1.,-1.)
    alpha=.4/(96*n);beta0=1.02;H={k:float(np.mean(v)) for k,v in Hs.items()}
    K=.6*(H['coarse']-1.03*H['refined'])-1e-6
    contrast=np.stack([(Rf-Rc)[:,labels.index(r['node_label'])]+.35*(Xf-Xc)[:,labels.index(r['node_label'])] for r in resources],axis=1)*1e6/pool['base_kw']
    gradient=alpha*(sc.reshape(96,n)@contrast).T
    m0mean=K+.4*(abs_mean(muc,np.linalg.norm(Lc,axis=1)).mean()-beta0*abs_mean(muf,np.linalg.norm(Lf,axis=1)).mean())
    lower=[];upper=[]
    for r in resources:
        base=np.broadcast_to(np.asarray(r['baseline_kw'],float),(96,))
        lower.append(np.maximum(np.asarray(r.get('power_lower_kw',0.))-base,-1.))
        upper.append(np.minimum(np.asarray(r['power_upper_kw'])-base,1.))
    problem=dict(G_mean_zero=K+alpha*(sc@muc-beta0*(sf@muf)),G_gradient=gradient,
        G_noise_base=alpha*(Lc.T@sc-beta0*(Lf.T@sf)),reference_mean=muf,reference_noise=Lf,reference_sign=sf,
        alpha=alpha,beta=beta0,target=.2*max(-m0mean,0)+1e-4,cv=np.sqrt(.025/3),elasticities=np.full(2,.5),
        u_lower=np.stack(lower),u_upper=np.stack(upper),slot_hours=.25,epsilon_G=.05,epsilon_F=.05)
    packet=dict(problem=problem,mu=mu,noise=L,mu_C=muc,mu_F=muf,L_C=Lc,L_F=Lf,s_C=sc,s_F=sf,
        contrast=contrast,K=K,H=H,pool=pool,labels=labels,resources=resources,baseline_E_M=m0mean)
    with (root/'problem.pkl').open('wb') as f:pickle.dump(packet,f,protocol=4)
    write(root/'problem_frozen.json',dict(status='frozen',input=ref(root/'problem.pkl'),source_protocol=ref(root/'protocol.json'),
        training_days=3,validation_days=3,covariance_rank=int(np.linalg.matrix_rank(L)),
        mu_margin_zero=m0mean,target=problem['target'],zero=evaluate_coefficients(problem,np.zeros((2,96))),
        corrected_box=box_bound(problem),naive_box=box_bound(problem,corrected=False)))
    write(root/'run_lock.json',dict(status='prepared',updated_utc=now()))
    print('prepared Gaussian problem',flush=True)

def load(root):
    root=Path(root);p=read(root/'protocol.json')
    for source,h in p['source_sha256'].items():
        if sha(source)!=h:raise ValueError('source drift '+source)
    checked(read(root/'problem_frozen.json')['input'])
    with (root/'problem.pkl').open('rb') as f:a=pickle.load(f)
    return root,p,a

def solve(root):
    root,p,a=load(root);results={}
    write(root/'run_lock.json',dict(status='solving',updated_utc=now()))
    for name,corrected,mode in [('corrected',True,'min_cost'),('naive',False,'min_cost'),('max_diagnostic',True,'max_bound')]:
        path=root/(name+'.json')
        if path.exists():r=read(path)
        else:
            try:r=solve_coefficients(a['problem'],corrected=corrected,mode=mode,seconds=p['solver_seconds'],log_path=root/(name+'.log'))
            except Exception as e:r=dict(status='failed',error=repr(e),mean_u=None)
            write(path,r)
        results[name]=r
        print(name,r['status'],r.get('audit',{}).get('corrected_margin_bound') if r.get('audit') else None,flush=True)
    policies={'zero':dict(mean_u=np.zeros((2,96)).tolist(),role='natural/no DR')}
    for name in ('corrected','naive','max_diagnostic'):
        r=results[name]
        if r.get('mean_u') is not None:policies[name]=dict(mean_u=r['mean_u'],role=('qualified corrected quote' if name=='corrected' and r.get('original_form_target_feasible') else 'diagnostic only'),solve=ref(root/(name+'.json')))
    reference='corrected' if 'corrected' in policies and results['corrected'].get('original_form_target_feasible') else 'max_diagnostic'
    if reference in policies:
        u=np.asarray(policies[reference]['mean_u']);amplitude=float(np.sqrt(np.square(u).mean()))
        pattern=np.where((np.arange(96)//24)%2==0,1.,-1.)
        rotation=amplitude*np.stack([pattern,-pattern])
        if not evaluate_coefficients(a['problem'],rotation)['nominal_bound_violation']<=1e-7:raise ValueError('matched rotation outside mean box')
        policies['rotation']=dict(mean_u=rotation.tolist(),role='equal planned cost baseline',matched_to=reference)
    write(root/'results.json',dict(results=results))
    write(root/'price_freeze.json',dict(status='frozen',created_utc=now(),policies=policies,results=ref(root/'results.json'),
        no_evaluation_random_stream_opened=True,ac_qualified=bool(results['corrected'].get('original_form_target_feasible') and results['corrected'].get('original_form_nominal_feasible'))))
    write(root/'run_lock.json',dict(status='prices_frozen',updated_utc=now()))

def evaluate(root):
    from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score
    root,p,a=load(root);freeze=read(root/'price_freeze.json');problem=a['problem'];N=p['evaluation_paths']
    models={m['candidate_id']:m for m in a['pool']['models']};Rf=np.array(models['refined']['R']);Xf=np.array(models['refined']['X'])
    # Draws are first created after price_freeze, and shared across policies.
    rng=np.random.default_rng(p['evaluation_seed']);zbase=rng.normal(size=(N,3));zdr=rng.normal(size=(N,2,96))
    np.savez_compressed(root/'evaluation_draws.npz',zbase=zbase,zdr=zdr)
    write(root/'run_lock.json',dict(status='evaluating',updated_utc=now()))
    basevec=a['mu'].ravel()[None,:]+zbase@a['noise'].T
    base=basevec.reshape(N,3,96,len(a['labels']));rf0=a['mu_F'][None,:]+zbase@a['L_F'].T
    rc0=a['mu_C'][None,:]+zbase@a['L_C'].T
    M0=a['K']+.4*(np.abs(rc0).mean(axis=1)-1.02*np.abs(rf0).mean(axis=1))
    summaries=[];rows={};all_checks=0
    for name,policy in freeze['policies'].items():
        u=np.asarray(policy['mean_u']);U=u[None,:,:]+problem['cv']*np.abs(u)[None,:,:]*zdr
        effect=np.einsum('ji,bit->btj',a['contrast'],U).reshape(N,-1)
        rc=rc0+effect;rf=rf0
        ec=np.abs(rc).mean(axis=1);ef=np.abs(rf).mean(axis=1)
        M=a['K']+.4*(ec-1.02*ef)
        G=a['K']+problem['alpha']*(rc@a['s_C']-1.02*(rf@a['s_F']))
        flipF=np.maximum(-rf*a['s_F'],0).sum(axis=1)
        flipC=np.maximum(-rc*a['s_C'],0).sum(axis=1)
        identity=G+2*problem['alpha']*(flipC-1.02*flipF)
        if not np.max(np.abs(identity-M))<1e-8:raise AssertionError('original margin identity')
        # All final choices use the existing observation-only selector.
        selected=[];coarse_excluded=[];mismatches=0
        for j in range(N):
            pp=base[j,0].copy();qq=base[j,1].copy()
            for i,r in enumerate(a['resources']):
                node=a['labels'].index(r['node_label']);pp[:,node]+=U[j,i]/a['pool']['base_kw'];qq[:,node]+=.35*U[j,i]/a['pool']['base_kw']
            yy=pp@Rf.T+qq@Xf.T+rf[j].reshape(96,-1)/1e6
            scene=dict(P_terminal=pp,Q_terminal=qq,drop_target=yy,labels=a['labels'])
            r=score(a['pool'],scene,history_gate=True)['primary_rule']
            selected.append(r['selected_candidate_id']);coarse_excluded.append('coarse' not in r['eligible_candidate_ids'])
            if (M[j]>0)!=coarse_excluded[-1]:mismatches+=1
        if mismatches:raise AssertionError('actual selection margin mismatch')
        bound=evaluate_coefficients(problem,u)
        passed=M>=problem['target'];cost=bound['cost']
        violations=np.any((U<np.asarray(problem['u_lower'])[None]-1e-10)|(U>np.asarray(problem['u_upper'])[None]+1e-10),axis=(1,2))
        negative_natural=np.any(base[:,0]<0,axis=(1,2))
        rows[name]=dict(M=M,G=G,M0=M0,flipF=flipF,selected=np.asarray(selected),coarse_excluded=np.asarray(coarse_excluded),passed=passed,resource_violation=violations)
        summaries.append(dict(policy=name,role=policy['role'],paths=N,completed=N,failures=0,target_passed=int(passed.sum()),target_ci95=interval(int(passed.sum()),N),
            refined_selected=selected.count('refined'),coarse_excluded=int(sum(coarse_excluded)),mean_M=float(M.mean()),p10_M=float(np.quantile(M,.1)),
            mean_increment=float(np.mean(M-M0)),new_passes_vs_zero=int(np.sum(passed & (M0<problem['target']))),lost_passes_vs_zero=int(np.sum(~passed & (M0>=problem['target']))),
            mean_G=float(G.mean()),std_G=float(G.std(ddof=1)),signed_G_passed=int(np.sum(G>=problem['target'])),planned_cost=cost,
            corrected_bound=bound['corrected_margin_bound'],flip_penalty=bound['flip_penalty'],mean_flip_penalty=float(2*problem['alpha']*1.02*flipF.mean()),
            p95_flip_penalty=float(np.quantile(2*problem['alpha']*1.02*flipF,.95)),nominal_power_excursion_paths=int(violations.sum()),
            target_and_no_resource_excursion=int(np.sum(passed & ~violations)),negative_natural_power_paths=int(negative_natural.sum()),
            mean_actual_net_energy_per_resource=(.25*U.sum(axis=2)).mean(axis=0).tolist(),identity_max_error=float(np.max(np.abs(identity-M))),
            selection_rule='existing weighted_selection.score primary w2/tau2, parameters/output maps frozen'))
        all_checks+=N
        print('evaluated',name,int(passed.sum()),'/',N,'cost',cost,flush=True)
    np.savez_compressed(root/'evaluation_scalars.npz',**{name+'__'+k:v for name,row in rows.items() for k,v in row.items()})
    write(root/'predictions_frozen.json',dict(status='frozen',created_utc=now(),price_freeze=ref(root/'price_freeze.json'),rows=summaries,
        truth_read=False,selection_calls=all_checks,new_fit_calls=0,new_W_search_calls=0,new_AC_calls=0))
    write(root/'run_lock.json',dict(status='model_evaluation_complete',updated_utc=now()))
    write(root/'ended.json',dict(status='model_evaluation_complete',price_freeze=ref(root/'price_freeze.json'),predictions=ref(root/'predictions_frozen.json'),
        ac_qualified=freeze['ac_qualified'],no_claim_of_independent_physical_or_field_validation=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','solve','evaluate']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    globals()[a.command](a.output)
if __name__=='__main__':main()
