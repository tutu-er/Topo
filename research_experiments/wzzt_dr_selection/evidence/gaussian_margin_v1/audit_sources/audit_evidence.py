"""Independent reconstruction of history coefficients and saved Gaussian paths.

Reuses input format readers only; never imports the optimizer or runner.
No new random seed, fit, physical simulation, or price solve is performed.
"""
from pathlib import Path
import json, pickle, hashlib, sys, math
import numpy as np
from scipy.special import ndtr
from statistics import NormalDist

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
ROOT=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_gaussian_margin_stage')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'rnj_wzzt_core'))
from research_experiments.wzzt_dr_selection.dr_frozen_distance.frozen_pool import _power,_target

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def absolute_normal_mean(mu,sd):
    nz=sd>0;out=np.abs(mu).copy();z=mu[nz]/sd[nz]
    out[nz]=sd[nz]*math.sqrt(2/math.pi)*np.exp(-z*z/2)+mu[nz]*(2*ndtr(z)-1)
    return out

def main():
    run=ROOT/'run_v1';pro=read(run/'protocol.json')
    with (run/'problem.pkl').open('rb') as f:a=pickle.load(f)
    checks=[];errors=[]
    def check(name,x,y,tol=1e-8):
        x=np.asarray(x);y=np.asarray(y)
        assert x.shape==y.shape,(name,x.shape,y.shape)
        err=float(np.max(np.abs(x-y),initial=0)) if x.dtype.kind not in 'USOb' else float(np.any(x!=y))
        checks.append(name)
        if err>tol:errors.append(dict(name=name,error=err))
        return err
    for k,r in pro['inputs'].items():check('input_hash_'+k,sha(r['path']),r['sha256'])
    with Path(pro['inputs']['history']['path']).open('rb') as f:container=pickle.load(f)
    history=container['history'];del container
    pool=read(pro['inputs']['pool']['path']);labels=pool['labels'];n=len(labels)
    models={x['candidate_id']:x for x in pool['models']}
    rf=np.array(models['refined']['R']);xf=np.array(models['refined']['X'])
    rc=np.array(models['coarse']['R']);xc=np.array(models['coarse']['X'])
    days=[];errors_f=[];errors_c=[];hist=[]
    for scene in history['training']:
        p,q=_power(scene,labels);y=_target(scene,p.shape)
        ef=y-p@rf.T-q@xf.T;ec=y-p@rc.T-q@xc.T
        days.append(np.stack([p,q,ef]));errors_f.append(ef.ravel()*1e6);errors_c.append(ec.ravel()*1e6)
    X=np.array(days);RF=np.array(errors_f);RC=np.array(errors_c)
    check('training_days',len(days),3)
    muf=RF.mean(axis=0);muc=RC.mean(axis=0)
    Lf=((RF-muf)/math.sqrt(2)).T;Lc=((RC-muc)/math.sqrt(2)).T
    common_mu=X.mean(axis=0);common_noise=((X-common_mu).reshape(3,-1)/math.sqrt(2)).T
    check('joint_mean',a['mu'],common_mu,1e-12);check('joint_noise',a['noise'],common_noise,1e-12)
    check('mu_F',a['mu_F'],muf);check('mu_C',a['mu_C'],muc)
    check('L_F',a['L_F'],Lf);check('L_C',a['L_C'],Lc)
    check('reference_centering',Lf.sum(axis=1),np.zeros_like(muf));check('coarse_centering',Lc.sum(axis=1),np.zeros_like(muc))
    rank=int(np.linalg.matrix_rank(common_noise));check('centered_rank_upper',max(rank-2,0),0)
    sf=np.where(muf>=0,1.,-1.);sc=np.where(muc>=0,1.,-1.)
    check('sign_F',sf,a['s_F']);check('sign_C',sc,a['s_C'])
    HH={k:[] for k in models}
    for scene in history['validation']:
        p,q=_power(scene,labels);y=_target(scene,p.shape)
        for k,m in models.items():HH[k].append(np.abs(y-p@np.array(m['R']).T-q@np.array(m['X']).T))
    H={k:float(np.concatenate(v).mean()*1e6) for k,v in HH.items()}
    for k in H:check('history_H_'+k,a['H'][k],H[k])
    alpha=.4/(96*n);beta0=1.02;K=.6*(H['coarse']-1.03*H['refined'])-1e-6
    check('K',a['K'],K)
    coeff=np.stack([(rf-rc)[:,labels.index(r['node_label'])]+.35*(xf-xc)[:,labels.index(r['node_label'])] for r in a['resources']],axis=1)*1e6/pool['base_kw']
    check('contrast',coeff,a['contrast'])
    grad=np.empty((2,96))
    for i in range(2):
        for t in range(96):grad[i,t]=alpha*float(sc.reshape(96,n)[t]@coeff[:,i])
    prob=a['problem'];check('gradient',grad,prob['G_gradient'])
    Gdays=K+alpha*((RC@sc)-beta0*(RF@sf))
    check('G_mean',prob['G_mean_zero'],Gdays.mean())
    check('G_common_loading',prob['G_noise_base'],(Gdays-Gdays.mean())/math.sqrt(2))
    sdc=np.linalg.norm(Lc,axis=1);sdf=np.linalg.norm(Lf,axis=1)
    EabsC=absolute_normal_mean(muc,sdc);EabsF=absolute_normal_mean(muf,sdf)
    EM=K+alpha*(EabsC.sum()-beta0*EabsF.sum())
    check('baseline_E_M',a['baseline_E_M'],EM)
    check('target',prob['target'],.2*max(-EM,0)+1e-4)
    EflipC=.5*(EabsC-sc*muc);EflipF=.5*(EabsF-sf*muf)
    mean_penalty_f=2*alpha*beta0*EflipF.sum()
    mean_positive_c=2*alpha*EflipC.sum()
    check('expected_identity',EM,Gdays.mean()+mean_positive_c-mean_penalty_f)
    diagnostic=dict(reference_mean_flip_penalty=float(mean_penalty_f),coarse_mean_flip_addition=float(mean_positive_c),
        net_mean_sign_correction=float(mean_positive_c-mean_penalty_f),G_mean_zero=float(Gdays.mean()),
        exact_E_M_zero=float(EM),reference_mean_std=float(sdf.mean()),covariance_rank=rank)
    receipt=dict(status='passed' if not errors else 'failed',checks=len(checks),failed=errors,diagnostic=diagnostic,
        inputs={str(run/'protocol.json'):sha(run/'protocol.json'),str(run/'problem.pkl'):sha(run/'problem.pkl')},source_sha256=sha(__file__),new_fit_calls=0,new_AC=0)
    (ROOT/'audit_history_v1.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'stage':'history','status':receipt['status'],'checks':len(checks),'diagnostic':diagnostic}))
    if errors:raise AssertionError(errors)
    if not (run/'predictions_frozen.json').exists():return
    summary=read(run/'predictions_frozen.json');freeze=read(run/'price_freeze.json')
    draws=np.load(run/'evaluation_draws.npz');scalars=np.load(run/'evaluation_scalars.npz')
    zb=draws['zbase'];zd=draws['zdr'];N=len(zb)
    checks=[];errors=[]
    check('paths',N,pro['evaluation_paths'])
    rng=np.random.default_rng(pro['evaluation_seed'])
    check('locked_base_random_stream',zb,rng.normal(size=zb.shape),0)
    check('locked_response_random_stream',zd,rng.normal(size=zd.shape),0)
    basec=muc+zb@Lc.T;basef=muf+zb@Lf.T
    Mzero=K+alpha*(np.abs(basec).sum(axis=1)-beta0*np.abs(basef).sum(axis=1))
    infos={x['policy']:x for x in summary['rows']};out=[]
    for name,policy in freeze['policies'].items():
        u=np.array(policy['mean_u']);dr=u+float(prob['cv'])*np.abs(u)*zd
        effect=np.einsum('ji,bit->btj',coeff,dr).reshape(N,-1)
        residual_c=basec+effect;residual_f=basef
        M=K+alpha*(np.abs(residual_c).sum(axis=1)-beta0*np.abs(residual_f).sum(axis=1))
        G=K+alpha*(residual_c@sc-beta0*(residual_f@sf))
        fC=np.maximum(-residual_c*sc,0).sum(axis=1);fF=np.maximum(-residual_f*sf,0).sum(axis=1)
        check(name+'_M',M,scalars[name+'__M']);check(name+'_G',G,scalars[name+'__G'])
        check(name+'_Mzero',Mzero,scalars[name+'__M0']);check(name+'_flip_F',fF,scalars[name+'__flipF'])
        check(name+'_sign_identity',M,G+2*alpha*(fC-beta0*fF))
        Gaff=float(prob['G_mean_zero'])+zb@np.asarray(prob['G_noise_base'])+np.einsum('it,bit->b',grad,dr)
        check(name+'_Gaussian_affine_formula',G,Gaff)
        # Independent final rule for the two historical candidates.
        eC=np.abs(residual_c).mean(axis=1);eF=np.abs(residual_f).mean(axis=1)
        LC=.6*H['coarse']+.4*eC;LF=.6*H['refined']+.4*eF
        bestF=LF<LC
        threshold=np.where(bestF,LF+.6*.03*H['refined']+.4*.02*eF+1e-6,LC+.6*.03*H['coarse']+.4*.02*eC+1e-6)
        eligibleC=LC<=threshold;eligibleF=LF<=threshold
        hist_min=min(H.values());histC=H['coarse']<=1.03*hist_min+1e-6;histF=H['refined']<=1.03*hist_min+1e-6
        assert histC and histF,'audit final rule expects both candidates in historical gate'
        counts={k:len(m['point']['supports']) for k,m in models.items()}
        chosen=[]
        for j in range(N):
            candidates=[]
            if eligibleC[j]:candidates.append((counts['coarse'],LC[j],0,'coarse'))
            if eligibleF[j]:candidates.append((counts['refined'],LF[j],1,'refined'))
            chosen.append(min(candidates)[3])
        chosen=np.array(chosen)
        check(name+'_final_selection',chosen,scalars[name+'__selected'],0)
        check(name+'_coarse_exclusion',~eligibleC,scalars[name+'__coarse_excluded'],0)
        passed=M>=prob['target'];check(name+'_target_flags',passed,scalars[name+'__passed'],0)
        violations=np.any((dr<np.asarray(prob['u_lower'])[None]-1e-10)|(dr>np.asarray(prob['u_upper'])[None]+1e-10),axis=(1,2))
        check(name+'_excursion_flags',violations,scalars[name+'__resource_violation'],0)
        info=infos[name]
        check(name+'_count_target',passed.sum(),info['target_passed'],0)
        check(name+'_count_refined',np.sum(chosen=='refined'),info['refined_selected'],0)
        check(name+'_count_resource',violations.sum(),info['nominal_power_excursion_paths'],0)
        check(name+'_meanM',M.mean(),info['mean_M']);check(name+'_p10M',np.quantile(M,.1),info['p10_M'])
        fee=.25*np.sum(u*u/.5);check(name+'_fee',fee,info['planned_cost'])
        out.append(dict(policy=name,paths=N,target_passed=int(passed.sum()),refined_selected=int(np.sum(chosen=='refined')),
            mean_M=float(M.mean()),mean_G=float(G.mean()),mean_flip_C_addition=float(2*alpha*fC.mean()),
            mean_flip_F_penalty=float(2*alpha*beta0*fF.mean()),mean_net_sign_correction=float(2*alpha*(fC-beta0*fF).mean()),
            rms_flip_C_addition=float(np.sqrt(np.mean((2*alpha*fC)**2))),
            rms_flip_F_penalty=float(np.sqrt(np.mean((2*alpha*beta0*fF)**2))),
            rms_net_sign_correction=float(np.sqrt(np.mean((2*alpha*(fC-beta0*fF))**2))),
            correlation_flip_terms=float(np.corrcoef(fC,fF)[0,1]),
            p95_flip_C_addition=float(np.quantile(2*alpha*fC,.95)),
            p95_flip_F_penalty=float(np.quantile(2*alpha*beta0*fF,.95)),
            G_mean_analytic=float(prob['G_mean_zero']+np.sum(grad*u)),
            G_std_analytic=float(np.sqrt(np.sum(np.asarray(prob['G_noise_base'])**2)+np.sum((grad*float(prob['cv'])*np.abs(u))**2))),
            G_std_sample=float(np.std(G,ddof=1)),
            target_and_no_resource_excursion=int(np.sum(passed & ~violations)),
            sign_correction_p05=float(np.quantile(2*alpha*(fC-beta0*fF),.05)),
            sign_correction_p95=float(np.quantile(2*alpha*(fC-beta0*fF),.95)),
            nominal_excursions=int(violations.sum()),cost=float(fee)))
    receipt=dict(status='passed' if not errors else 'failed',checks=len(checks),failed=errors,rows=out,
        inputs={str(run/k):sha(run/k) for k in ['problem.pkl','evaluation_draws.npz','evaluation_scalars.npz','predictions_frozen.json','price_freeze.json']},
        source_sha256=sha(__file__),new_fit_calls=0,new_AC=0,new_price_solves=0,random_stream='deterministic replay of saved seed only')
    (ROOT/'audit_evaluation_v1.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'stage':'evaluation','status':receipt['status'],'checks':len(checks),'rows':out}))
    if errors:raise AssertionError(errors)

if __name__=='__main__':main()
