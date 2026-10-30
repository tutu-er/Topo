"""Read-only flatness diagnosis of completed v2 planning records; no fitting."""
from pathlib import Path
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
import numpy as np

ROOT=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_dr_competition_bridge_v2')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def family(p):return tuple(sorted(tuple(s) for s,r,x in zip(p['supports'],p['r'],p['x']) if r>0 or x>0))
def matrices(p):
 r=np.zeros((15,15));x=r.copy()
 for s,rv,xv in zip(p['supports'],p['r'],p['x']):
  r[np.ix_(s,s)]+=rv;x[np.ix_(s,s)]+=xv
 return r,x
def span(values):return dict(min=min(values),max=max(values))
def counter(values):return dict(sorted(Counter(values).items()))
def main():
 protocol=read(ROOT/'protocol.json')
 row=next(r for r in protocol['cases'] if r['voltage_sigma']==.001)
 folder=ROOT/'planning'/row['job_id']/'response_table'
 plan=read(folder/'plan_result.json');pp=read(folder/'plan_protocol.json')
 assert plan['status']=='complete' and plan['training_pairs_completed']==48
 oldfile=Path(row['existing_result']['path']);assert sha(oldfile)==row['existing_result']['catalog_sha256']
 old=read(oldfile);labels=old['labels'];cal=old['calibration']
 sources={str(oldfile):sha(oldfile),str(folder/'plan_result.json'):sha(folder/'plan_result.json')}
 history=[]
 for role in ('train','validation'):
  p=Path(row[role]['path']);assert p.name==role+'_observed.npz' and sha(p)==row[role]['catalog_sha256']
  sources[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as f:
   history.append({k:np.array(f[k],float) for k in ('P','Q','V','root')})
 fP=np.concatenate([d['P'] for d in history]).mean(axis=0)
 fQ=np.concatenate([d['Q'] for d in history]).mean(axis=0)
 positions=[labels.index(v) for v in plan['identity']['active_labels']]
 inputs={}
 for key in range(4):
  p,q=fP.copy(),fQ.copy()
  for j,i in enumerate(positions):
   if (key>>j)&1:
    p[76:80,i]-=.2/200;p[80:84,i]+=.2/200
    q[76:80,i]-=.07/200;q[80:84,i]+=.07/200
  inputs[key]=(p,q)
 vp=history[1]['P'].reshape(288,15);vq=history[1]['Q'].reshape(288,15)
 vy=(history[1]['root'][:,:,None]**2-history[1]['V']**2).reshape(288,15)
 records=[]
 for p in sorted((folder/'training_masks').glob('*.json')):
  r=read(p);assert r['status']=='complete';r['_file']=p;sources[str(p)]=sha(p);records.append(r)
 assert len(records)==48
 summary=[]
 for m in range(3):
  group=[r for r in records if r['model_serial']==m];assert len(group)==16
  h=old['base_points'][group[0]['raw_index']];target=set(family(h));rh,xh=matrices(h)
  zero=next(r for r in group if r['training_mask_key']==[0,0])
  zero_signatures={name:[(tuple(a['candidate']),tuple(tuple(s) for s in a['removed'])) for a in t['accepted_moves']] for name,t in zero['paths'].items()}
  rows=[];raw_min=[];post_min=[];raw_selected=[];post_selected=[]
  raw_argmin=[];post_argmin=[];all_moves=[];zero_row=None
  kernel_changes=[];strength_by_support={}
  for record in group:
   pts=record['candidate_points']
   post=[];raw_errors=[];post_errors=[];components=[]
   for point in pts:
    keep=[i for i,(s,r,x) in enumerate(zip(point['supports'],point['r'],point['x'])) if len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=cal['tau']]
    contracted={k:[point[k][i] for i in keep] for k in ('supports','r','x')}
    post.append(contracted);raw_errors.append(len(set(family(point))^target));post_errors.append(len(set(family(contracted))^target))
    r,x=matrices(point)
    past=float(abs(vp@r.T+vq@x.T-vy).sum())
    future={k:float(abs(p@(r-rh).T+q@(x-xh).T).sum()) for k,(p,q) in inputs.items()}
    components.append((past,future))
   raw_min.append(min(raw_errors));post_min.append(min(post_errors))
   for name,path in record['paths'].items():
    signature=[(tuple(a['candidate']),tuple(tuple(s) for s in a['removed'])) for a in path['accepted_moves']]
    assert signature==zero_signatures[name]
    for move in path['accepted_moves']:
     d=path['decisions'][move['iteration']]
     all_moves.append(dict(path=name,training_key=record['training_mask_key'],candidate=move['candidate'],removed=move['removed'],gain=move['gain'],acceptance_threshold=1e-10+.001*d['current'],acceptance_margin=move['gain']-(1e-10+.001*d['current'])))
   for vr in record['validation_results']:
    a,b=vr['validation_key'];values=[(p+f[a]+f[b])/7200 for p,f in components]
    idx=vr['wzzt_selected_index'];best_idx=min(range(len(values)),key=lambda i:(values[i],i));best=values[best_idx]
    assert abs(values[idx]-vr['wzzt_raw_selection_mae'])<1e-11
    detail=dict(training_key=record['training_mask_key'],validation_key=[a,b],selected_index=idx,selected_mae=values[idx],best_mae=best,relative_gap=(values[idx]-best)/best,
                eligibility_slack=1.03*best+1e-12-values[idx],raw_selected_error=raw_errors[idx],post_selected_error=post_errors[idx],
                raw_argmin_error=raw_errors[best_idx],post_argmin_error=post_errors[best_idx])
    assert detail['post_selected_error']==vr['wzzt_loss']
    rows.append(detail);raw_selected.append(raw_errors[idx]);post_selected.append(post_errors[idx]);raw_argmin.append(raw_errors[best_idx]);post_argmin.append(post_errors[best_idx])
    if detail['training_key']==[0,0] and detail['validation_key']==[0,0]:zero_row=detail
    for s,r,x in zip(pts[idx]['supports'],pts[idx]['r'],pts[idx]['x']):
     key=tuple(s)
     if len(s)>1:strength_by_support.setdefault(key,[]).append(cal['a']*r+cal['b']*x)
  outputs={family(v['selected_wzzt']) for r in group for v in r['validation_results']}
  raw_outputs={family(r['candidate_points'][v['wzzt_selected_index']]) for r in group for v in r['validation_results']}
  rnj_outputs={family(v['selected_RNJ']) for r in group for v in r['validation_results']}
  selected=set(next(iter(outputs)));assert len(outputs)==1
  data=dict(model_serial=m,raw_index=group[0]['raw_index'],conditional_states=256,
    target_positive_blocks=len(target),target_zero_leaf_labels=[labels[i] for i in range(15) if (i,) not in target],
    WZZT_error_histogram=counter(post_selected),RNJ_error_histogram=counter(v['RNJ_loss'] for r in group for v in r['validation_results']),
    WZZT_final_effective_family_count=len(outputs),WZZT_selected_raw_effective_family_count=len(raw_outputs),RNJ_final_effective_family_count=len(rnj_outputs),
    selected_missing_labels=[[labels[j] for j in s] for s in sorted(target-selected)],selected_extra_labels=[[labels[j] for j in s] for s in sorted(selected-target)],
    selected_raw_error_histogram=counter(raw_selected),available_best_raw_error_by_training_mask=counter(raw_min),available_best_postcontract_error_by_training_mask=counter(post_min),
    diagnostic_min_validation_raw_error=counter(raw_argmin),diagnostic_min_validation_then_contract_error=counter(post_argmin),
    diagnostic_not_proposed_algorithm=True,zero_response=zero_row,
    selected_mae=span([r['selected_mae'] for r in rows]),best_mae=span([r['best_mae'] for r in rows]),relative_gap=span([r['relative_gap'] for r in rows]),eligibility_slack=span([r['eligibility_slack'] for r in rows]),
    max_response_change_vs_zero={key:max(abs(r[key]-zero_row[key]) for r in rows) for key in ('selected_mae','best_mae','eligibility_slack')},
    no_response_accepted_signatures={name:[dict(candidate=list(c),removed=[list(s) for s in removed]) for c,removed in signatures] for name,signatures in zero_signatures.items()},
    all16_training_masks_same_accepted_signatures_as_zero=True,
    accepted_move_count=len(all_moves),accepted_crossing_count=sum(bool(r['removed']) for r in all_moves),
    accepted_gain_range=span([r['gain'] for r in all_moves]) if all_moves else None,
    acceptance_margin_range=span([r['acceptance_margin'] for r in all_moves]) if all_moves else None,
    selected_nonleaf_strength_ranges=[dict(indices=list(s),labels=[labels[j] for j in s],strength=span(values),threshold=cal['tau']) for s,values in sorted(strength_by_support.items())],
    validation_details=rows)
  summary.append(data)
 table=plan['response_table'];assert len(table)==256
 assert all(row['wzzt_by_hypothesis']==[0,1,3] and row['RNJ_by_hypothesis']==[0,2,3] for row in table)
 sourceplans=read(ROOT/'planning'/row['job_id']/'ordinary_input_plans.json') if False else read(folder.parent/'ordinary_input_plans.json')
 ordinary={name:[dict(training_event=t,label=labels[j],price=v) for t,rr in enumerate(menu) for j,v in enumerate(rr) if v] for name,menu in sourceplans['plans'].items()}
 contrast=[]
 for idx in [19,16,7]:
  h=old['base_points'][idx];weights={tuple(s):(r,x) for s,r,x in zip(h['supports'],h['r'],h['x'])}
  rc,xc=weights.get((11,14),(0.,0.));rd,xd=weights.get((12,13,14),(0.,0.))
  leaf111=weights.get((10,),(0.,0.))
  contrast.append(dict(raw_index=idx,intersection115_delta_y112_minus_y113_per_deltaP=rc+.35*xc-rd-.35*xd,
                       leaf111_delta_y111_minus_y112_per_deltaP=leaf111[0]+.35*leaf111[1]))
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='passed_readonly_diagnosis',source_script_sha256=sha(__file__),source_files_sha256=sources,
  training_records=48,conditional_states_per_hypothesis=256,mean_WZZT_loss=4/3,mean_RNJ_loss=5/3,
  exact_finite_table_flatness='Every four-event mask has WZZT losses[0,1,3], so any participation distribution on these masks gives the same mean risk; changing only probabilities cannot help',
  raw_min_validation_counterfactual='All768 conditional states choose zero-error raw candidate if pure min validation is used; this is diagnostic only, not a validated production-rule change',
  all_accepted_moves_match_zero_response_patterns=True,accepted_moves=64,accepted_crossing_moves=32,
  models=summary,ordinary_plans=ordinary,alternative_historical_contrasts=contrast,
  site_warning='Original pair[114,112] omits intersection115 and historical zero-leaf disagreement111; ordinary input baselines already select111/115, so the alternative sites are not a unique WZZT discovery',
  recommended_one_next_step='Before new LP or actual data, re-score saved fitted candidates under frozen predictive validation measurement errors at the original declared noise levels, with shared root component; keep3pct/.16 and the same contract. This checks whether noiseless future validation is causing the flat gate. Treat as conditional diagnostic, not DR effect. If flat remains, close this construction.',
  no_recommendation_to_lower_resolution_or_raise_DR_budget=True,solver_calls=0,AC_calls=0,new_actual_future_read=False,truth_files_read=False,
  scope='Finite historical fitted hypotheses; does not prove global parameter coverage or impossibility of all DR')
 output=ROOT/'audit_flatness_v2.json';assert not output.exists()
 output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:result[k] for k in ('status','training_records','mean_WZZT_loss','mean_RNJ_loss','accepted_moves','accepted_crossing_moves','solver_calls','new_actual_future_read')}))
 for model in summary:print(json.dumps({k:model[k] for k in ('model_serial','raw_index','target_zero_leaf_labels','WZZT_error_histogram','selected_missing_labels','selected_extra_labels','relative_gap','eligibility_slack','max_response_change_vs_zero','acceptance_margin_range')}))

if __name__=='__main__':main()
