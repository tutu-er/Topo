"""Independent scalar audit; never imports the optimization implementation.

Inputs are private coefficient records. No AC simulation or fit is called.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pickle
from pathlib import Path
from statistics import NormalDist
import numpy as np

ND = NormalDist()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compute(problem, u, corrected=True):
    g=np.asarray(problem['G_gradient'],float)
    u=np.asarray(u,float)
    assert g.shape==u.shape and np.isfinite(u).all()
    def broadcast(key, default):
        a=np.asarray(problem.get(key,default),float)
        if a.shape==(u.shape[0],):a=a[:,None]
        return np.broadcast_to(a,u.shape)
    cv=broadcast('cv',.1);lam=broadcast('elasticities',.5)
    base=np.asarray(problem['G_noise_base'],float).reshape(-1)
    mu=np.asarray(problem['reference_mean'],float).reshape(-1)
    sign=np.asarray(problem.get('reference_sign',np.where(mu>=0,1.,-1.)),float)
    if 'reference_noise' in problem:
        lf=np.asarray(problem['reference_noise'],float)
        std=np.sqrt(np.einsum('ij,ij->i',lf,lf))
    else:std=np.asarray(problem['reference_std'],float)
    alpha=float(problem.get('alpha',.4/len(mu)));beta=float(problem.get('beta',1.02))
    epsf=float(problem.get('epsilon_F',.05));epsg=float(problem.get('epsilon_G',.05))
    risks=np.asarray(problem.get('reference_risks',np.full(len(mu),epsf/len(mu))),float)
    tj=np.array([max(0.,-sign[j]*mu[j]+ND.inv_cdf(1-float(risks[j]))*std[j]) for j in range(len(mu))])
    penalty=2*alpha*beta*math.fsum(tj.tolist()) if corrected else 0.
    m=float(problem['G_mean_zero'])+math.fsum((g*u).ravel().tolist())
    vb=math.fsum((base**2).tolist())
    vd=math.fsum(((g*cv*np.abs(u))**2).ravel().tolist())
    sd=math.sqrt(vb+vd)
    quantile=ND.inv_cdf(1-(epsg if corrected else .1))
    bound=m-quantile*sd-penalty
    lo=broadcast('u_lower',-np.inf);hi=broadcast('u_upper',np.inf)
    violation=max(0.,float(np.max(lo-u)),float(np.max(u-hi)))
    box_mean=float(problem['G_mean_zero'])+float(np.sum(np.maximum(g*lo,g*hi)))
    analytic_box_upper=box_mean-quantile*math.sqrt(vb)-penalty
    fee=float(problem.get('slot_hours',.25))*math.fsum((u*u/lam).ravel().tolist())
    return dict(G_mean=m,G_std=sd,G_variance_base=vb,G_variance_DR=vd,
        gaussian_quantile=quantile,signed_quantile_bound=m-quantile*sd,
        flip_penalty=penalty,corrected_margin_bound=bound,target_surplus=bound-float(problem['target']),
        cost=fee,nominal_bound_violation=violation,analytic_box_upper=analytic_box_upper,
        analytic_box_infeasible=analytic_box_upper<float(problem['target']),
        reference_correction_sum=float(tj.sum()),reference_std_mean=float(std.mean()),
        reference_mu_absolute_mean=float(np.abs(mu).mean()),
        sign_flips_exceed_mean_count=int(np.count_nonzero(tj>0)))

def audit_one(problem,result):
    if result.get('mean_u') is None:
        zero=np.zeros_like(np.asarray(problem['G_gradient'],float))
        metrics=compute(problem,zero,result.get('corrected',True))
        return dict(status='no_incumbent_preserved',solver_status=result.get('status'),checks=1,zero=metrics)
    actual=compute(problem,result['mean_u'],result.get('corrected',True))
    stored=result.get('audit',{})
    failures=[];n=0
    for key,value in actual.items():
        if key not in stored:continue
        n+=1
        if not math.isclose(float(value),float(stored[key]),rel_tol=1e-9,abs_tol=2e-8):
            failures.append(dict(key=key,independent=value,stored=stored[key]))
    n+=1
    if actual['nominal_bound_violation']>1e-7:failures.append(dict(key='nominal_bound_violation',value=actual['nominal_bound_violation']))
    if result.get('mode')=='min_cost' and result.get('status')=='optimal':
        n+=1
        if actual['target_surplus'] < -1e-7:failures.append(dict(key='target_surplus',value=actual['target_surplus']))
    return dict(status='passed' if not failures else 'failed',checks=n,failures=failures,metrics=actual,
        solver_status=result.get('status'),corrected=result.get('corrected'),mode=result.get('mode'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--problem',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.problem.suffix in ('.pkl','.pickle'):
        with args.problem.open('rb') as f:p=pickle.load(f)
    else:p=json.loads(args.problem.read_text(encoding='utf-8'))
    if 'problem' in p:p=p['problem']
    r=json.loads(args.results.read_text(encoding='utf-8'))
    if 'mean_u' in r:rows={'result':r}
    elif 'results' in r:rows=r['results']
    else:rows=r
    audits={key:audit_one(p,value) for key,value in rows.items() if isinstance(value,dict) and ('mean_u' in value or 'status' in value)}
    assert audits,'No result rows found; pass a scoped solver result file'
    receipt=dict(status='passed' if all(x['status']!='failed' for x in audits.values()) else 'failed',
        checks=sum(x['checks'] for x in audits.values()),audits=audits,
        inputs={str(x):sha(x) for x in [args.problem,args.results]},source_sha256=sha(__file__),new_fit_calls=0,new_AC=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'checks':receipt['checks'],'output':str(args.output)}))
    if receipt['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
