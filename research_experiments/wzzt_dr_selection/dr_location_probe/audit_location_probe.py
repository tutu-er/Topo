"""Arithmetic audit only. No project modules, optimizers, or AC solves are called."""
import argparse,collections,csv,hashlib,importlib.metadata,json,pickle,platform,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd

MODELS=('rnj_bank','baseline25','multistart_original_rule')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def obj(p):return pickle.loads(Path(p).read_bytes())
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b,tol=1e-12):return np.allclose(a,b,rtol=0,atol=tol,equal_nan=False)
def observed(sim,indices):
 o=sim['observed'];assert o is not None
 out=tuple(o[k].iloc[indices].copy() for k in ('P_terminal','Q_terminal','V_terminal','root_voltage'))
 assert all(x.index.equals(out[0].index) and np.isfinite(x.to_numpy()).all() for x in out)
 assert all(x.columns.equals(out[0].columns) for x in out[:3]);assert out[0].index.is_unique and out[0].index.is_monotonic_increasing
 assert close(out[3].to_numpy()[:,None]**2-out[2].to_numpy()**2,o['drop_target'].iloc[indices].to_numpy(),1e-14)
 return out
def join(items):return tuple(pd.concat([x[i] for x in items]) for i in range(4))
def ih(o,days):
 h=hashlib.sha256()
 for x in o:h.update(pd.util.hash_pandas_object(x,index=True).values.tobytes())
 h.update(json.dumps(list(o[0].columns),default=str).encode());h.update(json.dumps(days).encode());return h.hexdigest()
def matrices(point,n):
 r=np.zeros((n,n));x=r.copy()
 for support,a,b in zip(point['supports'],point['r'],point['x'],strict=True):r[np.ix_(support,support)]+=a;x[np.ix_(support,support)]+=b
 return r,x
def loss(point,o):
 p,q,v,u=[x.to_numpy() for x in o];R,X=matrices(point,p.shape[1]);direct=np.zeros_like(p)
 for s,r,x in zip(point['supports'],point['r'],point['x'],strict=True):direct[:,s]+=(r*p[:,s].sum(axis=1)+x*q[:,s].sum(axis=1))[:,None]
 assert close(direct,p@R.T+q@X.T,1e-13)
 return float(np.mean(abs(direct-(u[:,None]**2-v**2))))
def same(a,b):return a['supports']==b['supports'] and np.array_equal(a['r'],b['r']) and np.array_equal(a['x'],b['x'])
def metric(point,truth,labels):
 active={frozenset(labels[i] for i in s) for s,r,x in zip(point['supports'],point['r'],point['x']) if max(r,x)>1e-9};true={frozenset(s) for s in truth}
 internal=lambda v:{s for s in v if 1<len(s)<len(labels)}
 pred,want=internal(active),internal(true);root={s for s in want if not any(s<t for t in want)}
 tp=len(pred&want);den=len(pred)+len(want)
 return dict(internal_f1=2*tp/den if den else 1.,internal_exact=pred==want,complete_effective_family=active==true,root_matched=len(root&pred),internal_matched=tp,internal_false=len(pred-want),internal_missing=len(want-pred))

def main(args):
 root=args.run_root.resolve();repo=args.repo_root.resolve();protocol=read(root/'protocol.json');source=Path(protocol['source']);old=Path(protocol['old_run']);summary=read(root/'summary.json')
 receipt={'mode':'arithmetic replay only; no project, solver, or AC imports/calls','created_at':datetime.now(timezone.utc).isoformat(),'audit_source_sha256':sha(__file__),'protocol_sha256':sha(root/'protocol.json'),'global_issues':[],'fits':[],'cells':[],'physical_records':[]}
 def ck(test,message,out=None):
  if not bool(test):(receipt['global_issues'] if out is None else out['issues']).append(message)
 contract={k:v for k,v in protocol.items() if k not in ('fingerprint','created_at')};ck(fp(contract)==protocol['fingerprint'],'protocol fingerprint mismatch')
 all_hashes={**protocol['source_input_sha256'],**protocol['algorithm_source_sha256'],**protocol['controller_source_sha256']}
 receipt['frozen_hash_count']=len(all_hashes);receipt['changed_frozen_paths']=[p for p,h in all_hashes.items() if not Path(p).is_file() or sha(p)!=h];ck(not receipt['changed_frozen_paths'],'frozen source/input hash changed')
 oldp=read(old/'identification_v1'/'protocol.json');state=read(source/'run_state.json');helpers={}
 for name,anchor in [('research_experiments/wzzt_dr_selection/dr_pool_v2/matrix_runner.py',oldp['controller_source_sha256']),('research_experiments/wzzt_dr_selection/end_to_end/runner.py',state['provenance']['source_sha256'])]:
  p=(repo/name).resolve();expected=next((h for f,h in anchor.items() if Path(f).resolve()==p),None);helpers[str(p)]={'current':sha(p),'frozen':expected,'match':expected==sha(p)}
  ck(helpers[str(p)]['match'],'unlocked helper differs from old frozen source '+name)
 receipt['supplemental_helper_hashes']=helpers
 for fname in ('adapter.py','engine.py','observations.py','selection.py'):
  ck(str((repo/'research_experiments/wzzt_dr_selection/dr_pool_v2'/fname).resolve()) in protocol['algorithm_source_sha256'],'missing algorithm source '+fname)
 bundle=obj(source/'privatebundle.pkl');plan=read(source/'plan.json');meta=read(source/'summary.json');cfg=protocol['config'];labels=meta['case']['terminals'];actions={a['node']:a for a in protocol['actions']};actions_name={a['name']:a for a in protocol['actions']}
 signal=next(Path(p) for p in bundle['data_manifest']['source_hashes'] if p.endswith('price_signals.csv'));price=np.full(24,np.nan)
 for name in ('price_signals.csv','control_training_menu_factors.csv'):
  p=next(Path(p) for p in bundle['data_manifest']['source_hashes'] if p.endswith(name));ck(sha(p)==bundle['data_manifest']['source_hashes'][str(p)],'menu source changed '+name)
 with signal.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   if row['Price_signal']=='P_15':price[int(row['Hour'])-1]=float(row['Experiment_price_NOK_kWh'])
 ck(np.isfinite(price).all(),'P15 price has missing hours')
 period_cache={};baseline_cache={}
 def audit_period(path,noise,node,mode,phase):
  path=Path(path).resolve()
  if path in period_cache:return period_cache[path]
  out={'path':str(path),'sha256':sha(path),'noise':noise,'node':node,'mode':mode if phase=='action' else 0,'phase':phase,'issues':[]}
  data=obj(path);s=data['simulation'];resp=data['response'];truth=s['truth'];check=s['checks'];p=truth['P_kw'];q=truth['Q_kvar']
  generated=path.is_relative_to(root/'physical');out['new_generated']=generated
  if generated:
   rp=path.with_suffix('.receipt.json');ck(rp.exists(),'missing generated physical receipt',out)
   if rp.exists():
    rr=read(rp);ck(rr['sha256']==out['sha256'],'generated physical receipt hash mismatch',out)
    ck(rr['noise']==noise and rr['node']==node and rr['phase']==phase and rr['mode']==out['mode'],'generated physical receipt identity mismatch',out)
  bk=(noise,phase)
  if bk not in baseline_cache:baseline_cache[bk]=obj(old/'physical_v1'/'scenarios'/f'{noise}__response0'/'strategies'/'no_reward'/(phase+'.pkl'))
  base=baseline_cache[bk]['simulation']['truth'];bp=base['P_kw'].to_numpy();bq=base['Q_kvar'].to_numpy();factor=np.ones_like(bp);prices=np.zeros_like(bp)
  if node is not None:
   j=labels.index(node);factor[:,j]=np.asarray(bundle['response_banks'][phase]['P_15'])[mode if phase=='action' else 0,:,0];prices[:,j]=price
  ep,eq=bp*factor,bq*factor
  ck(list(p.columns)==labels and p.index.equals(base['P_kw'].index) and q.index.equals(p.index),'physical alignment',out)
  ck(len(p)==24 and p.index.equals(pd.date_range(cfg[phase+'_date'],periods=24,freq='h')),'physical24h grid',out)
  ck(close(p.to_numpy(),ep) and close(q.to_numpy(),eq),'actual response not frozen bank times baseline',out)
  ck(close(resp['p_kw'],ep) and close(resp['q_kvar'],eq),'response payload mismatch',out)
  credit=np.sum(prices*(np.asarray(plan['forecast_p_kw'])-ep),axis=0);paid=np.maximum(credit,0);bound=float(np.sum(prices*np.asarray(plan['forecast_p_kw'])))
  ck(close(resp['credits_nok_per_node'],credit) and close(resp['payment_nok_per_node'],paid) and close(resp['total_payment_nok'],paid.sum()),'24h account settlement mismatch',out)
  energy=ep.sum(0)-bp.sum(0);fraction=np.divide(energy,bp.sum(0),out=np.zeros_like(energy),where=bp.sum(0)>0)
  audit=resp['audit'];ck(close(audit['daily_energy_change_kwh_per_node'],energy) and close(audit['daily_energy_change_fraction_per_node'],fraction),'energy audit mismatch',out)
  ck(close(audit['total_peak_kw'],ep.sum(1).max()) and close(audit['payment_bound_nok'],bound),'peak or payment bound mismatch',out)
  power_ok=bool(np.all(ep<=cfg['planning']['power_limits_kw']+1e-9));energy_ok=bool(np.all(abs(fraction)<=cfg['planning']['max_daily_energy_change_fraction']+1e-9));budget_ok=bool(paid.sum()<=cfg['planning']['budget_nok']+1e-9)
  ck(audit['valid']==(power_ok and energy_ok and budget_ok) and audit['budget_valid']==budget_ok,'response validity classification mismatch',out)
  ck(audit['clipped_samples']==0 and np.min(ep)>=0,'clipping or negative response',out)
  ac=truth['ac'];vs=ac['V_bus_mag'].to_numpy();voltage_ok=bool(np.isfinite(vs).all() and np.all(vs>=cfg['physical']['voltage_min_pu']) and np.all(vs<=cfg['physical']['voltage_max_pu']));converged=int(np.asarray(ac['converged']).sum())
  ck(check['voltage_limits_pass']==voltage_ok and check['converged_samples']==converged and check['sample_count']==24,'AC voltage/convergence audit mismatch',out)
  ck(check['thermal_limits_checked'] is False,'unexpected thermal certificate',out)
  ck(close(truth['root_voltage_true'],base['root_voltage_true']),'physical root differs across location/noise pairing',out)
  if s['observed'] is not None:
   o=s['observed'];rng=np.random.default_rng(cfg['seeds'][phase]+2000);tp=truth['P_true'].to_numpy();tq=truth['Q_true'].to_numpy();tv=truth['V_terminal_true'].to_numpy();tu=truth['root_voltage_true'].to_numpy()
   mp=tp+rng.normal(0,cfg['physical']['pq_noise_rel']*np.maximum(abs(tp),1e-12),size=tp.shape);mq=tq+rng.normal(0,cfg['physical']['pq_noise_rel']*np.maximum(abs(tq),1e-12),size=tq.shape);mv=tv+rng.normal(0,protocol['noise'][noise]*np.maximum(abs(tv),1e-12),size=tv.shape)
   ur=np.random.default_rng(cfg['seeds'][phase]+70000019);mu=tu+ur.normal(0,protocol['noise'][noise]*np.maximum(abs(tu),1e-12),len(tu))
   ck(all(close(o[k].to_numpy(),v,1e-14) for k,v in [('P_terminal',mp),('Q_terminal',mq),('V_terminal',mv),('root_voltage',mu)]),'frozen meter-seed/noise reconstruction mismatch',out)
   observed(s,np.arange(24))
  deploy=bool(audit['valid'] and check['usable_ac_solution'] and voltage_ok);out.update(payment_nok=float(paid.sum()),converged_samples=converged,power_valid=power_ok,energy_valid=energy_ok,budget_valid=budget_ok,voltage_valid=voltage_ok,deployment_valid=deploy,pass_audit=not out['issues'])
  period_cache[path]=(data,out);receipt['physical_records'].append(out);return data,out
 records=[read(p) for p in sorted((root/'records').glob('*.json'))];expected={(noise,n,a['node'],mode) for noise in protocol['noise'] for n in protocol['sample_sizes'] for a in protocol['actions'] for mode in (0,1)};seen=set();unique={}
 for row in records:
  k=(row['noise'],row['N'],row['node'],row['mode']);ck(k in expected and k not in seen,'unexpected/duplicate task '+str(k));seen.add(k)
  co={'key':row['key'],'status':row['status'],'issues':[]}
  if row['status']!='evaluated':
   ck(row['status']=='failed' and bool(row.get('exception')),'nonterminal or missing failure record',co);receipt['cells'].append(co);continue
  try:
   files=[Path(p) for p in row['physical_files']];ap=next(p for p in files if p.name.startswith('action'));vp=next(p for p in files if p.name.startswith('validation'))
   for p,h in row['physical_files'].items():ck(sha(p)==h,'task physical hash changed '+p,co)
   act,aa=audit_period(ap,row['noise'],row['node'],row['mode'],'action');val,va=audit_period(vp,row['noise'],row['node'],row['mode'],'validation')
   ck(row['deployment_valid']==(aa['deployment_valid'] and va['deployment_valid']),'task deployment classification',co)
   fitpath=Path(row['fit_result']);ck(sha(fitpath)==row['fit_sha256'],'task fit hash mismatch',co);fit=read(fitpath);cache=read(fitpath.parent/'cache_record.json')
   ck(cache['status']=='complete' and fit['status']=='complete' and sha(fitpath)==cache['result_sha256'],'cache status/hash mismatch',co)
   hd=old/'physical_v1'/'scenarios'/f'{row["noise"]}__response0'/'history';ix=np.arange(0,24,96//row['N']);tr=join([observed(obj(hd/(d+'.pkl')),ix) for d in meta['wzzt_training_dates']]+[observed(act['simulation'],ix)]);valid=observed(val['simulation'],np.arange(24));days=[str(d) for d in tr[0].index.normalize()];vd=[str(d) for d in valid[0].index.normalize()]
   inputs={'training':ih(tr,days),'validation':ih(valid,vd)};ident={'training':inputs['training'],'validation':inputs['validation'],'terminal_labels':labels,'root_label':1,'search_seconds':25.,'algorithm_source_sha256':protocol['algorithm_source_sha256']}
   ck(inputs==fit['input_sha256'] and ident==cache['identity'] and fp(ident)==row['observation_key']==cache['key'],'exact observation/cache identity mismatch',co)
   ck(row['old_cache_reused']==fitpath.is_relative_to(old/'identification_v1'/'fit_cache'),'old reuse path classification',co)
   truth=act['simulation']['truth']['truth_supports'];ck(truth==val['simulation']['truth']['truth_supports'],'action validation truth mismatch',co)
   for name in MODELS:
    for field,value in metric(fit['models'][name],truth,labels).items():ck(row['evaluation']['models'][name]['final'][field]==value,name+' recovery metric '+field,co)
   if row['observation_key'] not in unique:
    fo={'observation_key':row['observation_key'],'first_task':row['key'],'result':str(fitpath),'result_sha256':sha(fitpath),'old_cache_reused':row['old_cache_reused'],'issues':[],'models':{}}
    unique[row['observation_key']]=fo;receipt['fits'].append(fo)
    ck(fit['source_sha256']==protocol['algorithm_source_sha256'],'fit source closure differs',fo)
    ck(fit['terminal_labels']==labels and fit['training_rows']==len(tr[0])==row['N'] and fit['validation_rows']==len(valid[0])==24,'dimensions',fo)
    ck(dict(collections.Counter(days))==fit['training_day_counts'] and dict(collections.Counter(vd))==fit['validation_day_counts'],'physical day grouping',fo);ck(not len(tr[0].index.intersection(valid[0].index)),'train validation overlap',fo)
    settings=fit['settings'];ck(settings['final_validation_fraction']==.03 and settings['resolution_factor']==.16 and settings['coefficient_bound']==2. and settings['per_path_seconds']==25. and settings['max_extra_starts']==2 and not settings['final_refit'],'frozen algorithm settings',fo)
    sols=fit['LP_solves'];fo['LP_count']=len(sols);fo['LP_status_counts']=dict(collections.Counter(str(s['status']) for s in sols));ck(all(s['status']==2 and s['accepted'] for s in sols),'LP status not optimal accepted',fo)
    fo['LP_max_objective_discrepancy']=max(abs(s['objective']-s['physical']) for s in sols);ck(fo['LP_max_objective_discrepancy']<1e-8,'LP objective discrepancy',fo);ck(all(s['max_constraint_violation']<=1e-7 and s['max_bound_violation']<=1e-7 for s in sols),'LP reported feasibility',fo)
    errors=[]
    allpoints=fit['selection_candidates']+fit.get('start_preparation',[])
    for pt in allpoints:
     ck(all(np.isfinite(w).all() and min(w)>=-1e-9 and max(w)<=2+1e-9 for w in (pt['r'],pt['x'])),'candidate weights',fo)
     sets=[set(s) for s in pt['supports']];ck(all(not(a&b) or a<=b or b<=a for a,b in __import__('itertools').combinations(sets,2)),'candidate laminarity',fo);ck(all(s in fit['pool_support_indices'] for s in pt['supports']),'candidate outside pool',fo)
     errors.extend([abs(loss(pt,tr)-pt['objective']),abs(loss(pt,valid)-pt['validation_mae'])])
    fo['selection_and_start_points_audited']=len(allpoints);fo['candidate_loss_max_error']=max(errors);ck(max(errors)<1e-12,'candidate raw loss mismatch',fo)
    for name,point in fit['models'].items():
     before=point['selected_uncontracted'];trainloss=loss(point,tr);valloss=loss(point,valid);R,X=matrices(point,len(labels));candidates=fit['bank'] if name=='rnj_bank' else [p for p in fit['selection_candidates'] if name=='multistart_original_rule' or p['origin'] in ('tree_bank','tree_initialized_correction')]
     best=min(p['validation_mae'] for p in candidates);eligible=[i for i,p in enumerate(candidates) if p['validation_mae']<=1.03*best+1e-12];win=min(eligible,key=lambda i:(len(candidates[i]['supports']),candidates[i]['validation_mae'],i));ck(same(before,candidates[win]) and point['selection_index']==win,name+'3percent selection',fo)
     cal=fit['calibrations']['raw'];keep=[i for i,s in enumerate(before['supports']) if len(s)==1 or cal['a']*before['r'][i]+cal['b']*before['x'][i]+1e-10>=.16*cal['median_root_depth']];reduced={k:[before[k][i] for i in keep] for k in ('supports','r','x')};ck(same(point,reduced),name+'0.16 contraction',fo)
     ck(close(R,point['R'],1e-13) and close(X,point['X'],1e-13) and close(R,point['r_matrix'],1e-13) and close(X,point['x_matrix'],1e-13),name+' matrix reconstruction',fo)
     errs=[abs(trainloss-point['train_mae']),abs(trainloss-point['objective']),abs(valloss-point['validation_mae']),abs(loss(before,tr)-before['objective']),abs(loss(before,valid)-before['validation_mae']),abs(loss(before,valid)-point['selection_validation_mae'])];ck(max(errs)<1e-12,name+' final/precontraction losses',fo)
     ck(point['support_labels']==[[labels[i] for i in s] for s in point['supports']],name+' support labels',fo);ck(all([i] in point['supports'] for i in range(len(labels))),name+' stored leaf missing',fo)
     fo['models'][name]={'selection_index':win,'loss_max_error':max(errs),'train_mae':trainloss,'validation_mae':valloss,'precontraction_validation_mae':before['validation_mae']}
    fo['search_stop_statuses']=fit['search_stop_statuses'];ck(fit['search_stop_statuses']=={k:s['reason'] for k,s in fit['searches'].items()},'search reason mismatch',fo)
    fo['pass_audit']=not fo['issues']
  except Exception as exc:co['issues'].append(type(exc).__name__+': '+str(exc));co['traceback']=traceback.format_exc()
  co['pass_audit']=not co['issues'];receipt['cells'].append(co)
 # Also audit generated periods prepared just before a batch boundary.
 for p in sorted((root/'physical').rglob('*.pkl')):
  try:
   noise=p.parent.parent.name;action=actions_name[p.parent.name];phase,mode=p.stem.rsplit('_',1);audit_period(p,noise,action['node'],int(mode),phase)
  except Exception as exc:receipt['global_issues'].append('extra physical '+str(p)+': '+type(exc).__name__+': '+str(exc))
 ck(len(expected)==protocol['planned_cells'],'protocol denominator mismatch');ck(len(records)==summary['recorded_cells'] and sum(r['status']=='evaluated' for r in records)==summary['evaluated_cells'],'summary count mismatch');ck(summary['status']==('complete' if seen==expected else 'partial'),'summary completion mismatch')
 physical_failures=[{'path':str(p),'failure':read(p)} for p in (root/'physical').rglob('*.failure.json')]
 receipt.update(planned_cells=len(expected),recorded_cells=len(records),pending_cells=len(expected-seen),failed_cells=sum(r['status']!='evaluated' for r in records),unique_complete_fits_audited=len(unique),old_unique_fits_audited=sum(x['old_cache_reused'] for x in unique.values()),new_unique_fits_audited=sum(not x['old_cache_reused'] for x in unique.values()),LPs_audited=sum(x['LP_count'] for x in unique.values()),physical_failure_records=physical_failures,generated_periods_audited=sum(p['new_generated'] for p in receipt['physical_records']),generated_simulated_hours=24*sum(p['new_generated'] for p in receipt['physical_records']),no_solves=True)
 receipt['all_passed']=not receipt['global_issues'] and all(not x['issues'] for x in receipt['cells']+receipt['fits']+receipt['physical_records']);receipt['ready_for_unchanged_bounded_resume']=receipt['all_passed'] and receipt['failed_cells']==0 and not physical_failures
 receipt['scope']='Recorded cells and all generated physical periods audited; pending cells retained; arithmetic/record consistency, not global optimality or DR efficacy.'
 args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({k:v for k,v in receipt.items() if k not in ('fits','cells','physical_records')},indent=2,ensure_ascii=False));print('issues',[(x.get('key',x.get('path',x.get('first_task'))),x['issues']) for x in receipt['cells']+receipt['fits']+receipt['physical_records'] if x['issues']]);return 0 if receipt['all_passed'] else 2
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);raise SystemExit(main(p.parse_args()))
