"""Independent arithmetic audit and aggregate-only export; no new experiments."""
from pathlib import Path
import json,pickle,hashlib,shutil,zipfile
import numpy as np
from run import BASE,read,write,checked,ref
stage=Path(__file__).parent.parent
first=read(stage/'run_v1/summary.json');second=read(stage/'quadratic_v1/summary.json')
for folder in ['run_v1','quadratic_v1']:
    p=read(stage/folder/'protocol.json')
    for r in p['sources']+p.get('inputs',[]):checked(r)
snapshot=stage/'quadratic_v1/source_snapshot.zip'
if not snapshot.exists():
    with zipfile.ZipFile(snapshot,'x',zipfile.ZIP_DEFLATED) as z:
        for i,r in enumerate(read(stage/'quadratic_v1/protocol.json')['sources']):z.write(r['path'],f'{i:03d}_{Path(r["path"]).name}')
with zipfile.ZipFile(snapshot) as z:
    for i,r in enumerate(read(stage/'quadratic_v1/protocol.json')['sources']):
        assert hashlib.sha256(z.read(f'{i:03d}_{Path(r["path"]).name}')).hexdigest()==r['sha256']
checks=0;failed=[]
def check(flag,label):
    global checks
    checks+=1
    if not flag:failed.append(label)
check(first['complete']==12,'12 completed')
check(second['complete']==6,'6 quadratic returned, numerical convergence separate')
quad={(r['case'],r['policy']):r for r in second['rows']}
scores=read(BASE/'scored_private.json')['rows']
rows=[];unique=set()
for r in first['rows']:
    key=(r['case'],r['policy'])
    p=pickle.loads((BASE/r['case']/'problem.pkl').read_bytes())
    plan=pickle.loads((BASE/r['case']/'plans.pkl').read_bytes())[r['policy']]
    u=np.asarray(plan['mean_u'])
    canonical=hashlib.sha256(r['case'].encode()+np.where(u==0.,0.,u).tobytes()).hexdigest();unique.add(canonical)
    mc=p['mu_C'].reshape(96,-1)+np.einsum('ir,rt->ti',p['effect_C'],u)
    mf=p['mu_F'].reshape(96,-1)+np.einsum('ir,rt->ti',p['effect_F'],u)
    check(abs(float(np.abs(mc).mean()-1.03*np.abs(mf).mean())-r['mean_reference_score'])<1e-8,str(key)+' nominal')
    last=r['moments']['rows'][-1];mean=r['moments']['mean'];var=last['variance_upper'];h=r['threshold']
    check(abs(r['cantelli_lower']-(mean-np.sqrt(19*var)))<1e-9,str(key)+' Cantelli')
    prev=-1
    for q in r['moments']['rows']:
        check(q['variance_lower']>=prev-1e-8,str(key)+' variance monotonic');prev=q['variance_lower']
        check(abs(q['variance_upper']-q['variance_lower']-q['remainder_upper'])<1e-8,str(key)+' remainder')
    # The first audit imposed an unplanned 1e-4 target and failed 8 rows.
    # Keep that failure. Report the actual approximation bound; do not certify
    # spherical integration merely because a looser tolerance would pass.
    perturb=r['sphere']['expected_discard_error']
    check(np.isfinite(perturb) and perturb>=0,str(key)+' spectral perturbation reported')
    check(r['sphere']['certificate'] is False,str(key)+' no numerical certificate')
    check(all(0<=q['p']<=1 for q in r['sphere']['probabilities']),str(key)+' probability range')
    a=r['nominal_milp']['alpha']
    if a is not None:
        sc=p['mu_C']+a*(mc.ravel()-p['mu_C']);sf=p['mu_F']+a*(mf.ravel()-p['mu_F'])
        check(float(np.abs(sc).mean()-1.03*np.abs(sf).mean())>=h+1e-6-1e-7,str(key)+' MILP original loss')
        check(abs(r['nominal_ray_fee']-r['contract_fee']*a*a)<1e-8,str(key)+' ray fee')
    act=[v for v in scores if v['condition']==r['case'] and v['policy']==r['policy'] and v['law']=='main']
    check(len(act)==24,str(key)+' actual denominator')
    check(all(v['status']=='complete' for v in act),str(key)+' actual complete')
    lb=(mean-h)**2/(var+(mean-h)**2) if mean>h else 0.
    out=dict(case=r['case'],policy=r['policy'],fee=r['contract_fee'],threshold=h,mean_D=mean,
        sd_interval=[float(np.sqrt(last['variance_lower'])),float(np.sqrt(var))],
        cantelli_q05_lower=r['cantelli_lower'],cantelli_success_lower=lb,
        cantelli_optimistic_q05=float(mean-np.sqrt(19*last['variance_lower'])),
        gaussian_approx_q05_interval=r['gaussian_approx_q05_range'],
        sphere_p=r['sphere']['probability_estimate'],sphere_grid_spread=r['sphere']['grid_spread'],
        holder_success_lower=r['holder']['lower_success'],holder_quad_unresolved=r['holder']['quadrature_unresolved'],
        nominal_alpha=a,nominal_mean_D=r['mean_reference_score'],
        actual_main_recovery=sum(v['errors']==0 for v in act),actual_main_exclusions=sum(v['coarse_excluded'] for v in act),actual_days=24,
        sphere_dimension=r['sphere']['rank'],sphere_expected_spectral_D_error=perturb,
        sphere_discarded_positive_trace=r['sphere']['discarded_trace'],
        sphere_min_eigenvalue=r['sphere']['negative_eigenvalue'],
        seconds=r['seconds'],canonical_hash=canonical)
    if key in quad:
        q=quad[key]['result'];out['quadratic']=q
        check(abs(q['quadratic_variance']-q['variance_factored'])<1e-7 if 'variance_factored' in q else q['variance_numerical_error']<1e-7,str(key)+' quadratic variance')
        check(q['remainder_variance_upper']>=0,str(key)+' quadratic remainder')
    rows.append(out)
check(len(unique)==9,'nine mathematical plans (signed zero canonicalized)')
audit=dict(checks=checks,failures=failed,passed=not failed,
    audit_scope='Arithmetic and provenance only, not certified probability integration.',
    initial_audit='171 checks;8 failures against an unplanned spectral D-error target 1e-4 retained in audit_initial.json. All integrations remain approximate; no experiment rerun.',
    zero_hash_issue='v1 raw byte hash distinguishes +0 and -0. 11 computed plans but only9 mathematically distinct;2 duplicate computations retained. Delivery normalizes signed zero for future runs.',
    main_records=12,main_compute_calls=11,mathematically_distinct_plans=9,
    quadratic_records=6,quadratic_compute_calls=5,
    quadratic_probability_converged=sum(r['result']['approximate_probability']['status']=='converged' for r in second['rows']),
    quadratic_probability_unresolved=sum(r['result']['approximate_probability']['status']=='numerically_unresolved' for r in second['rows']))
write(stage/'audit.json',audit)
if failed:raise AssertionError(failed)
public=stage/'public';public.mkdir(exist_ok=True)
write(public/'summary.json',dict(scope='frozen affine Gaussian method diagnosis; prior AC results reused only for comparison',
    audit=audit,random_draws=0,new_AC=0,new_fits=0,new_W=0,strict_95_passes=0,rows=rows))
write(public/'audit.json',audit)
shutil.copy2(stage/'audit_initial.json',public/'audit_initial.json')
for folder in ['run_v1','quadratic_v1']:
    p=read(stage/folder/'protocol.json')
    write(public/(folder+'_protocol.json'),{k:v for k,v in p.items() if k not in ['cases','inputs','sources']}|{
        'cases':[c['name'] if isinstance(c,dict) else c for c in p['cases']],
        'inputs_sha256':[{'file':Path(r['path']).name,'sha256':r['sha256']} for r in p['inputs']],
        'source_sha256':[{'file':Path(r['path']).parent.name+'/'+Path(r['path']).name,'sha256':r['sha256']} for r in p['sources']]})
delivery=stage/'delivery';delivery.mkdir(exist_ok=True)
for fn in ['kernel.py','run.py','test_kernel.py','quadratic.py','quadratic_run.py','test_quadratic.py','PLAN.md','LITERATURE.md']:
    shutil.copy2(Path(__file__).parent/fn,delivery/fn)
shutil.copy2(__file__,delivery/'audit_export.py')
print(json.dumps(audit,ensure_ascii=False))
for r in rows:
    if r['policy']=='candidate':
        print(r['case'],r['cantelli_success_lower'],r['sphere_p'],r.get('quadratic',{}).get('approximate_probability'),r.get('quadratic',{}).get('best_corrected_lower'))
