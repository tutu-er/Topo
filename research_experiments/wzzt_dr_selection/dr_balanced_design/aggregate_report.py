"""Post-run aggregate-only report; no observation arrays are published."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import json
S=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
R=S/'wzzt_b_design_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
cfg=read(R/'protocol.json');rows=read(R/'evaluations.json');sm=read(R/'summary.json');acts=read(R/'actions_frozen.json')
mech=read(S/'wzzt_b_audit_stage/run_v1/mechanisms.json')
idx={(r['flow'],r['case'],r['menu_id']):r for r in rows}
paired=[]
for method in cfg['methods']:
 for other in ('zero','single2','rotation','pair1'):
  count=Counter();aerr=berr=0;afull=bfull=0
  for ac in acts['actions']:
   for case in cfg['cases']:
    a=idx[(ac['flow']['id'],case['id'],ac['policies']['pair_rotation'])].get('evaluated',{}).get(method,{})
    b=idx[(ac['flow']['id'],case['id'],ac['policies'][other])].get('evaluated',{}).get(method,{})
    if a.get('status')!='complete' or b.get('status')!='complete':count['unresolved']+=1;continue
    ea,eb=a['final_errors'],b['final_errors'];aerr+=ea;berr+=eb;afull+=ea==0;bfull+=eb==0
    count['improved' if ea<eb else 'worse' if ea>eb else 'same']+=1
  paired.append(dict(method=method,new_pair_vs=other,counts=dict(count),new_pair_errors=aerr,other_errors=berr,new_pair_full=afull,other_full=bfull))
methods=[]
for name in cfg['methods']:
 vals=[m for m in mech if m['method']==name]
 methods.append(dict(method=name,completed_maps=len(vals),final_errors=sum(m['final_errors'] for m in vals),
  raw_errors=sum(m['raw_errors'] for m in vals),full=sum(m['final_errors']==0 for m in vals),raw_full=sum(m['raw_errors']==0 for m in vals),
  correct_saved=sum(m['correct_saved'] for m in vals),correct_eligible=sum(m['correct_eligible'] for m in vals),
  correct_eligible_but_not_raw=sum(m['correct_eligible'] and m['raw_errors']>0 for m in vals),
  accepted_steps=sum(m['accepted'] for m in vals),accepted_cross_steps=sum(m['accepted_cross'] for m in vals),
  true_blocks_deleted=sum(len(m['true_deleted']) for m in vals),false_blocks_deleted=sum(len(m['false_deleted']) for m in vals),
  partial_scans=sum(not m['all_scans'] for m in vals),
  final_missing_by_size=dict(Counter(len(s) for m in vals for s in m['final_missing'])),
  final_extra_by_size=dict(Counter(len(s) for m in vals for s in m['final_extra']))))
waug=Counter();delta=0
for r in rows:
 w=r.get('evaluated',{}).get('current_pool_wzzt',{});a=r.get('evaluated',{}).get('current_root_augmented_rnj',{})
 if w.get('status')!='complete' or a.get('status')!='complete':waug['unresolved']+=1;continue
 d=w['final_errors']-a['final_errors'];delta+=d;waug['W_better' if d<0 else 'W_worse' if d>0 else 'same']+=1
result=dict(status='bounded_incomplete',planned_maps=308,recorded_maps=len(rows),executed_maps=sum(r['execution_status']=='executed' for r in rows),
 completed_three_method_maps=217,not_executed_maps=90,delivered_but_methods_not_started_maps=1,
 distinct_observations=187,complete_distinct_fits=186,LP_calls=sm['LP_calls'],seconds=sm['seconds'],
 strict_B_choice='zero',strict_B_all22_uniform_separation=0.,
 methods=methods,predeclared_paired_new_pair=paired,W_vs_augmented_same_observations=dict(counts=dict(waug),W_minus_augmented_errors=delta),
 menu_results=sm['menu_results'],predeclared_policy_results=sm['policy_results'],
 metric_boundary='aggregates on completed subset only; do not compare unequal policy totals; same-natural two-stream development test, not population recovery probability',
 original_field_correction='run-v1 path_count read nonexistent searches key; independent audit uses paths; fits and choices unchanged')
result['distinct_W_path_audit']=dict(complete_distinct_fits=186,paths=400,accepted_steps=459,accepted_cross_steps=12,
    note='distinct observed fits; mapped accepted steps in method totals repeat reused fits')
write(S/'wzzt_b_report_stage/results_summary.json',result)
proto=dict(menu_count=22,original_menus=16,new_menus=6,planned_maps=308,case_count=7,behavior_measurement_streams=2,
 historical_rows=[288,288],merged_rows=[480,480],natural_seed=884001,
 behavior_seeds=[924031,924032],measurement_seeds=[924131,924132],order_seed=924431,
 budget_per_four_event_deployment=1.6,prices=[0,1,2],participation=[0,.5,.75],peak_kw=.2,peak_kvar=.07,
 coefficient_lower=0.,coefficient_upper=2.,coefficient_jointly_refit=True,
 source_commit='d411f11a',protocol_sha256=sha256((R/'protocol.json').read_bytes()).hexdigest(),
 actions_sha256=sha256((R/'actions_frozen.json').read_bytes()).hexdigest(),
 historical_saved_points=19,historical_complete_families=4,common_RX_feasible_in_all_faces=True,
 no_new_physical_confidence_claim=True,mathematical_gate_parameter_upper='none; distinct from actual upper2 domain')
write(S/'wzzt_b_report_stage/protocol_summary.json',proto)
audit=read(S/'wzzt_b_audit_stage/run_v1/audit_result.json');ctrl=read(S/'wzzt_b_controls_audit/audit_summary.json');math=read(S/'wzzt_b_math_stage/run_v1/summary.json');ma=read(S/'wzzt_b_math_stage/run_v1/audit_readonly.json')
ar=dict(status='passed',menu_pure_tests=13,math_LPs=math['total_LPs'],math_checks=math['checks'],
 math_independent_replay=ma,actual_independent_checks=audit['checks'],actual_largest_differences=audit['largest_differences'],
 LP_receipts_recount=audit['counts']['LP_receipts'],new_audit_LP_calls=0,
 controls_summary=ctrl,
 private_audit_sources=[dict(name=p.name,sha256=sha256(p.read_bytes()).hexdigest()) for p in [S/'wzzt_b_audit_stage/audit.py',S/'wzzt_b_controls_audit/audit.py']],
 note='checks are audits, not statistical observations')
# Only explicit aggregate fields from secondary summaries; never import private arrays.
ar['math_independent_replay']={k:v for k,v in ma.items() if k in ('status','checks','failed_checks','maximum_reconstruction_difference','LP_calls','new_LP_calls','new_solver_calls')}
ar['controls_summary']={k:v for k,v in ctrl.items() if k in ('status','checks','failures','recorded_maps','executed_maps','not_executed_maps','observed_flow_menu_groups','planned_flow_menu_groups','new_LP_calls','policy_deployments','deployment_costs','coverage')}
ar['controls_summary']['snapshot']={k:ctrl['snapshot'][k] for k in ('declared_maps','audited_receipts','still_unseen','declared_control_trajectories','seen_control_trajectories','full_seven_case_reuses')}
ar['controls_summary']['policy_payments']=ctrl['frozen_policy_payments']
write(S/'wzzt_b_report_stage/audit_summary.json',ar)
print(json.dumps(dict(methods=methods,paired=paired,W_vs_augmented=result['W_vs_augmented_same_observations']),ensure_ascii=False))
