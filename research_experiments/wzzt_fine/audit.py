from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha
from research_experiments.wzzt_retune.engine import compatible
from research_experiments.wzzt_budget.audit import independent_lp
from research_experiments.wzzt_refine.core import ridge_alpha
from .common import observed,sample_indices,MAIN,TARGET


def audit(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();protocol=read(output/'protocol.json')
    frame=pd.read_csv(destination/'per_job.csv');manifest=read(output/'manifest.json')
    checks=[dict(check='all frozen producer/data/payload bytes',passed=all(sha(output/p)==h for p,h in manifest.items()))]
    sizes=protocol['sizes'];sets=[set(sample_indices(N)) for N in sizes]
    checks.append(dict(check='nested fine sampling',passed=all(a<=b for a,b in zip(sets,sets[1:]))))
    for case in protocol['cases']:
        for bundle in range(3):
            for split in ('train','validation','test'):
                root=output/'data'/case/f'b{bundle}';true_path=root/f'{split}_truth.npz'
                if not true_path.exists():continue
                with np.load(true_path) as d:truth={k:d[k] for k in ('P','Q','V','root')}
                reference=None
                for sigma in protocol['voltage_sigmas']:
                    file=root/f'v{int(round(sigma*10000)):04d}'/f'{split}_observed.npz'
                    if not file.exists():continue
                    with np.load(file) as d:measured={k:d[k] for k in ('P','Q','V','root')}
                    normalized={k:(measured[k]-truth[k])/(sigma*np.maximum(abs(truth[k]),1e-12)) for k in ('V','root')}
                    if reference is not None:
                        checks.append(dict(check=f'{case}/b{bundle}/{split}/v{sigma} common random numbers',passed=
                            np.array_equal(reference['P'],measured['P']) and np.array_equal(reference['Q'],measured['Q']) and
                            all(np.allclose(reference[k],normalized[k],atol=1e-8,rtol=1e-8) for k in ('V','root'))))
                    reference=dict(P=measured['P'],Q=measured['Q'],**normalized)
                physical=read(root/f'{split}_physical.json')
                checks.append(dict(check=f'{case}/b{bundle}/{split} physical AC convergence',passed=all(r['converged']==96 for r in physical)))
    LP=[];physical=[];fronts=[];independent=[]
    for job in protocol['jobs']:
        folder=output/'jobs'/job
        if not (folder/'result.json').exists():checks.append(dict(check=job+' complete',passed=False));continue
        result=read(folder/'result.json');payload=read(folder/'payload.json')
        train,validation=observed(payload['train'],payload['N']),observed(payload['validation'])
        checks.append(dict(check=job+' observation-only model input',passed=result['truth_used_by_estimator'] is False))
        for name,rows in result['LP'].items():
            LP.append(dict(job_id=job,kind=name,count=len(rows),passed=all(r['status']==2 and max(r['max_constraint_violation'],r['max_bound_violation'])<=1e-7 and abs(r['physical']-r['objective'])<=1e-8 for r in rows)))
        base={tuple(s) for s in result['pool']}
        for name,domain in result['domains'].items():
            checks.append(dict(check=job+'/'+name+' original pool retained',passed=base<={tuple(s) for s in domain}))
        points=list(result['models'].values())+result['base_points']+[p for group in result['candidates'].values() for p in group]
        for search in result['searches'].values():points+=search.get('path',[])
        seen=set();p,q,v,root=train;y=root.to_numpy()[:,None]**2-v.to_numpy()**2
        vp,vq,vv,vr=validation;vy=vr.to_numpy()[:,None]**2-vv.to_numpy()**2
        for point in points:
            key=str((point['supports'],point['r'],point['x']))
            if key in seen:continue
            seen.add(key);family=[tuple(s) for s in point['supports']]
            legal=len(set(family))==len(family) and all((i,) in family for i in range(len(result['labels']))) and all(compatible(family[:i],s) for i,s in enumerate(family))
            bounded=all(-1e-9<=w<=2+1e-9 for w in point['r']+point['x'])
            errors=[]
            for P,Q,Y,name in ((p,q,y,'objective'),(vp,vq,vy,'validation_mae')):
                if name not in point:continue
                pred=np.zeros_like(Y);pp,qq=P.to_numpy(),Q.to_numpy()
                for s,r,x in zip(point['supports'],point['r'],point['x'],strict=True):pred[:,s]+=(r*pp[:,s].sum(axis=1)+x*qq[:,s].sum(axis=1))[:,None]
                errors.append(abs(float(np.mean(abs(pred-Y)))-point[name]))
            physical.append(dict(job_id=job,passed=bool(legal and bounded and max(errors)<1e-8),max_error=max(errors)))
        for front in result['augmentation']['fronts']:
            from research_experiments.wzzt_retune.engine import arrays
            p1,q1,_=arrays(train,front['preprocess']);alpha,scale=ridge_alpha(p1,q1,front['strength'])
            fronts.append(dict(job_id=job,preprocess=front['preprocess'],strength=front['strength'],passed=
                bool(abs(alpha-front['alpha'])<1e-12 and abs(scale-front['gram_scale'])<1e-12 and front['diagnostics'].get('success',False))))
        if payload['bundle']==0 and payload['voltage_sigma']==.001 and payload['N'] in (24,48):
            for method in ('multistart','restart_control','ridge_union'):
                bank=list(result['base_points'])+result['candidates'].get(method,[])
                point=bank[result['models'][method]['selection_index']]
                independent.append(dict(job_id=job,method=method,**independent_lp(point,train)))
    expected=sum(len(MAIN)+(len(TARGET) if read(output/'jobs'/j/'payload.json')['targeted'] else 0) for j in protocol['jobs'])
    checks.append(dict(check='all model rows and input denominators',passed=len(frame)==expected and frame.model_available.all() and not frame.duplicated(['job_id','variant']).any()))
    summary=dict(all_passed=all(r['passed'] for g in (checks,LP,physical,fronts,independent) for r in g),data_checks=len(checks),
        LP_solves=sum(r['count'] for r in LP),LP_groups=len(LP),physical_model_checks=len(physical),ridge_fronts=len(fronts),
        independent_LP_checks=len(independent),inputs=frame.job_id.nunique(),model_rows=len(frame))
    write(destination/'audit_summary.json',summary);write(destination/'data_checks.json',checks);write(destination/'independent_LP.json',independent)
    pd.DataFrame(LP).to_csv(destination/'LP_checks.csv',index=False);pd.DataFrame(physical).to_csv(destination/'physical_checks.csv',index=False)
    pd.DataFrame(fronts).to_csv(destination/'front_checks.csv',index=False)
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');a=p.parse_args();audit(a.output,a.destination)
