"""Trace already completed P15 position results; no fit or simulation calls."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,pickle
ROOT=Path(__file__).parent
NAMES=('no_reward','node_1_P_15','node_2_P_15')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pack(p,labels):
 return {'supports':[[labels[i] for i in s] for s in p['supports']],'r':p['r'],'x':p['x'],'training_mae':p.get('objective'),'validation_mae':p.get('validation_mae')}
records=[]
for name in NAMES:
 for mode in (0,1):
  path=ROOT/'records'/f'high_voltage_noise__{name}__mode{mode}__N96.json'
  row=read(path);fit_path=Path(row['fit_result']);f=read(fit_path);labels=f['terminal_labels'];cal=f['calibrations']['raw'];tau=.16*cal['median_root_depth'];best=min(c['validation_mae'] for c in f['selection_candidates'])
  truth_path=next(Path(p) for p in row['physical_files'] if 'action' in Path(p).name);tr=pickle.loads(truth_path.read_bytes())['simulation']['truth']['truth_supports'];truth={tuple(s) for s in tr};internal={s for s in truth if 1<len(s)<15}
  def cut(p):
   keep=[i for i,s in enumerate(p['supports']) if len(s)==1 or cal['a']*p['r'][i]+cal['b']*p['x'][i]+1e-10>=tau]
   return {k:[p[k][i] for i in keep] for k in ('supports','r','x')}
  def metrics(p):
   active={tuple(labels[i] for i in s) for s,r,x in zip(p['supports'],p['r'],p['x']) if max(r,x)>1e-9};pred={s for s in active if 1<len(s)<15};tp=len(pred&internal);fp=len(pred-internal);fn=len(internal-pred)
   return {'internal_f1':2*tp/(2*tp+fp+fn),'complete_effective_family':active==truth,'missing':[list(s) for s in sorted(truth-active)],'extra':[list(s) for s in sorted(active-truth)]}
  banks=[]
  for i in (3,4,8,9):
   p=f['bank'][i];start=f['start_preparation'][i];cutpoint=cut(p)
   banks.append({'index':i,'config':p['config'],'native_family':[[labels[k] for k in s] for s in p['native_family']],'raw_fit':pack(p,labels),'raw_fit_structure':metrics(p),'contraction_removed':[[labels[k] for k in s] for s in p['supports'] if s not in cutpoint['supports']],'raw_fit_block_strengths':[{'support':[labels[k] for k in s],'r':r,'x':x,'strength':cal['a']*r+cal['b']*x,'keep':len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=tau} for s,r,x in zip(p['supports'],p['r'],p['x']) if len(s)>1],'contracted_structure':metrics(cutpoint),'contracted_refitted_start':pack(start,labels),'start_structure':metrics(start),'chosen_as_extra_start':i in f['extra_start_sources']})
  cand=[{'index':i,'origin':p['origin'],'config':p.get('config'),'source_config':p.get('source_config'),'path_index':p.get('path_index'),'count_before_contraction':len(p['supports']),'validation_mae':p['validation_mae'],'ratio_to_best':p['validation_mae']/best,'eligible':p['validation_mae']<=1.03*best+1e-12,'after_contraction':{k:v for k,v in metrics(cut(p)).items() if k in ('internal_f1','complete_effective_family')}} for i,p in enumerate(f['selection_candidates'])]
  models={}
  for method,m in f['models'].items():
   selected=m['selected_uncontracted'];origin=m['origin'];initial=origin.startswith('multistart_') and m.get('path_index')==0
   models[method]={'origin':origin,'config':m.get('config'),'source_config':m.get('source_config'),'path_index':m.get('path_index'),'selection_index':m['selection_index'],'ratio_to_global_best':m['selection_validation_mae']/best,'selected_before_contraction':pack(selected,labels),'final':pack(m,labels),'final_structure':metrics(m),'selected_is_initial_point':initial,'selected_equals_path_initial':all(selected[k]==f['searches'][origin]['path'][0][k] for k in ('supports','r','x')) if initial else None,'structure_matches_stored_evaluation':metrics(m)['internal_f1']==row['evaluation']['models'][method]['final']['internal_f1']}
  records.append({'key':row['key'],'node':row['node'],'mode':mode,'record_sha256':sha(path),'fit_sha256':sha(fit_path),'stored_fit_hash_matches':sha(fit_path)==row['fit_sha256'],'truth_pickle_sha256':sha(truth_path),'pool_size':f['pool_size'],'pool_support_labels_sha256':hashlib.sha256(json.dumps(f['pool_support_labels'],sort_keys=True).encode()).hexdigest(),'calibration':cal,'extra_start_sources':f['extra_start_sources'],'models':models,'bank_and_start_details':banks,'selection_candidates':cand,'searches':{k:{'point_count':len(v['path']),'stop_reason':v['reason'],'accepted_steps':len(v['path'])-1} for k,v in f['searches'].items()}})
base=next(r for r in records if r['node'] is None and r['mode']==0)
pos=next(r for r in records if r['node']==103 and r['mode']==0)
neg=next(r for r in records if r['node']==102 and r['mode']==0)
def bank(r,i):return next(x for x in r['bank_and_start_details'] if x['index']==i)
summary={'positive_103_mode0':{'source_bank8_native_family_equals_no_reward':bank(pos,8)['native_family']==bank(base,8)['native_family'],'source_bank8_fitted_supports_equal_no_reward':bank(pos,8)['raw_fit']['supports']==bank(base,8)['raw_fit']['supports'],'mechanism':'Raw RX75 0.16 family joint L1 weights change; 104,105 crosses start-preparation 0.16 threshold. Contracted family is refitted and becomes multistart0 path index0, which wins validation. No accepted greedy step is part of the selected model.','remaining_missing_leaf':[114]},'negative_102_mode0':{'bank9_native_removed_vs_no_reward':[s for s in bank(base,9)['native_family'] if s not in bank(neg,9)['native_family']],'mechanism':'RNJ raw RX75 0.24 drops native block 101,103,104,105, creating a 21-support coarse candidate. This candidate already omits true 103,104,105 and retains false 104,105; removal of another false native block reduces its stored size. It is within 3% of validation best and now wins by size. This degradation is already present in RNJ-only selection.','selected_to_best_validation_ratio':neg['models']['multistart_original_rule']['ratio_to_global_best']},'scope':'Six already evaluated N96 high-noise cells; post-fit mechanism audit, not a learned or validated price rule; full scan and independent numerical audit are separate.'}
c=frozenset((104,105))
compatible_all=all(not(c&frozenset(s)) or c<=frozenset(s) or frozenset(s)<=c for s in truth)
base_family={tuple(s) for s in base['models']['multistart_original_rule']['selected_before_contraction']['supports']}
summary['nested_extra_block_obstruction']={'block':[104,105],'compatible_with_all_physical_true_supports':compatible_all,'true_containing_parent':[103,104,105],'baseline_selected_stored_family_equals_true_plus_extra':base_family==truth|{tuple(sorted(c))},'physical_true_supports':[list(s) for s in sorted(truth,key=lambda s:(len(s),s))],'conditions':['The smaller physical true support family is a feasible subfamily of the larger family.','The extra block is laminar with every retained support.','All shared coefficients have the same bounds and the extra R/X coefficients may be zero.','The objective is unpenalized mean L1 on identical observed data.'],'mathematical_fact':'With identical data, loss normalization and weight bounds, zero-allowed weights imply J(F_true union C) <= J(F_true) for every P,Q,Y, including every DR realization.','proof':'For every feasible coefficient vector on F_true, retain its shared coordinates and append r_C=x_C=0. This is feasible on F_true union C and gives exactly the same prediction and loss, so minimizing on the larger family cannot increase the objective.','search_implication':'Deletion of this compatible extra block alone cannot strictly improve training L1; adding any true block cannot delete it through crossing repair. Its removal requires coefficients becoming zero after another joint refit, explicit contraction/initialization, or a complexity/validation selection mechanism.','scope':'This is a fixed-family nesting fact, independent of correct physical model specification. It does not forbid deletion after a different crossing move or infer that price can never affect fitted coefficients.'}
assert compatible_all and summary['nested_extra_block_obstruction']['baseline_selected_stored_family_equals_true_plus_extra']
out={'created_at':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'no_new_optimization':True,'no_new_simulation':True,'truth_role':'post-fit mechanism diagnosis only','record_count':len(records),'unique_existing_fit_count':len({r['fit_sha256'] for r in records}),'summary':summary,'records':records}
(ROOT/'new_location_mechanisms_102_103.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
