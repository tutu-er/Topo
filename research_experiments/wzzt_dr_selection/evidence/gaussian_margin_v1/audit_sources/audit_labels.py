from pathlib import Path
import json,pickle,hashlib,numpy as np
base=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
run=base/'wzzt_gaussian_margin_stage/run_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
predpath=run/'predictions_frozen.json';pred=read(predpath)
assert pred['status']=='frozen' and pred['truth_read'] is False
freeze_path=run/'price_freeze.json';assert sha(freeze_path)==pred['price_freeze']['sha256']
freeze=read(freeze_path)
poolpath=base/'wzzt_newday_weighted_v1/run_v2/cases/paper15_b2_v0005/frozen_pool.json';pool=read(poolpath);labels=pool['labels']
truthpath=next((base/'wzzt_day_effect_stage/actual_v1/actual/paper15_day00_zero').rglob('truth_evaluation.json'));truth=read(truthpath)
assert truth['terminal_labels']==labels
truthset={tuple(sorted(map(int,x))) for x in truth['true_supports']}
def err(s):
 s=set(s);return dict(support_count=len(s),missing=len(truthset-s),extra=len(s-truthset),full_errors=len(truthset^s),complete_recovery=bool(s==truthset))
models={}
for m in pool['models']:
 raw={tuple(sorted(labels[int(i)] for i in block)) for block in m['point']['supports']}
 final={tuple(sorted(map(int,b))) for b in m['frozen_output_support_labels']}
 models[m['candidate_id']]=dict(raw=err(raw),final=err(final),raw_zero_weight_support_count=sum(float(r)==0 and float(x)==0 for r,x in zip(m['point']['r'],m['point']['x'])))
arr=np.load(run/'evaluation_scalars.npz');draws=np.load(run/'evaluation_draws.npz')
a=pickle.loads((run/'problem.pkl').read_bytes());p=a['problem'];mu=a['mu'];noise=a['noise']
N=draws['zbase'].shape[0];baseline=(mu.ravel()[None,:]+draws['zbase']@noise.T).reshape(N,3,96,len(labels));zdr=draws['zdr']
checks=0;rows=[]
for s in pred['rows']:
 name=s['policy'];selected=arr[name+'__selected'];assert len(selected)==N
 rawerrors=np.array([models[str(v)]['raw']['full_errors'] for v in selected]);errors=np.array([models[str(v)]['final']['full_errors'] for v in selected]);checks+=N
 u=np.asarray(freeze['policies'][name]['mean_u']);U=u[None,:,:]+p['cv']*np.abs(u)[None,:,:]*zdr
 bad=(U<np.asarray(p['u_lower'])[None]-1e-10)|(U>np.asarray(p['u_upper'])[None]+1e-10)
 badpaths=np.any(bad,axis=(1,2));assert np.array_equal(badpaths,arr[name+'__resource_violation']);assert int(badpaths.sum())==s['nominal_power_excursion_paths'];checks+=N+1
 basebad=np.any(baseline[:,0]<0,axis=(1,2));assert int(basebad.sum())==s['negative_natural_power_paths'];checks+=1
 pp=baseline[:,0].copy()
 for i,r in enumerate(a['resources']):pp[:,:,labels.index(r['node_label'])]+=U[:,i,:]/pool['base_kw']
 rows.append(dict(policy=name,denominator=N,conditional_gaussian_raw_full_errors_sum=int(rawerrors.sum()),conditional_gaussian_final_full_errors_sum=int(errors.sum()),conditional_gaussian_full_recovery_count=int(np.sum(errors==0)),resource_power_bound_violation_paths=int(badpaths.sum()),resource_power_bound_violation_cells=int(bad.sum()),negative_natural_active_power_paths=int(basebad.sum()),negative_post_DR_active_power_paths=int(np.any(pp<0,axis=(1,2)).sum()),full_recovery_and_no_resource_excursion=int(np.sum((errors==0)&~badpaths)),complete_no_new_AC_or_physical_checks=True))
out=dict(status='independently_verified',checks=checks,source={str(x):sha(x) for x in (predpath,freeze_path,poolpath,truthpath,run/'evaluation_scalars.npz',run/'evaluation_draws.npz',run/'problem.pkl')},truth_read_gate='predictions_frozen existed and bound price_freeze before truth read',full_effective_truth_blocks=len(truthset),models=models,rows=rows,scope='Labels from existing same-paper15 AC network reused only after all Gaussian selections frozen. These are conditional Gaussian working-law selected-model errors, not new AC or field recovery.',resource_scope='Per-sampled DR U bounds only: original component baseline1 and power[0,2], equivalently U[-1,1]. No prefix/day energy check, no AC, voltage, ampacity or physical delivery qualification. Negative Gaussian background loads retained, never clipped/resampled.')
output=base/'wzzt_gaussian_margin_stage/audit_labels_v1.json';output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ['status','checks','full_effective_truth_blocks','models','rows']},ensure_ascii=False))
