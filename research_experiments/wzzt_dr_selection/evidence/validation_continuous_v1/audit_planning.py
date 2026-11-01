"""Independent read-only audit of completed continuous-price planning records.

No optimization/identification function is imported.  No actual observation,
physical future or true graph is read.  Exact affine tables are replayed with
long-double arithmetic and compared with scalar saved optimization receipts.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np

ROOT = Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_validation_continuous_v1')
checks = Counter()
failures = []
artifacts = {}
phase_statuses = Counter()
problem_rows = []
lp_receipt_count = 0
solver_phase_count = 0
tables_cache = {}


def read(path):
    path = Path(path)
    artifacts[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(category, name, condition, **details):
    checks[category] += 1
    if not bool(condition):
        failures.append(dict(category=category,name=name,**details))


def close(a,b,tolerance=2.e-12):
    return a is None and b is None or a is not None and b is not None and abs(float(a)-float(b))<=tolerance


def checked_ref(ref, name):
    path=Path(ref['path'])
    check('hash',name,sha(path)==ref['sha256'],actual=sha(path),expected=ref['sha256'])
    return path


def load_tables(path):
    path=Path(path)
    if path not in tables_cache:
        artifacts[str(path.relative_to(ROOT))]=sha(path)
        with path.open('rb') as stream: tables_cache[path]=pickle.load(stream)
    return tables_cache[path]


def snapshot_audit(folder,protocol):
    path=folder/'source_snapshot.zip';artifacts[str(path.relative_to(ROOT))]=sha(path)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        actual=Counter(hashlib.sha256(z.read(name)).hexdigest() for name in names)
    expected=Counter(protocol['source_sha256'].values())
    check('snapshot',folder.name+'_source_count',len(names)==len(protocol['source_sha256']))
    check('snapshot',folder.name+'_exact_source_bytes',actual==expected)
    return dict(files=len(names),zip_sha256=sha(path),protocol_source_hashes_match=actual==expected)


def physical(problem,x,prices,label):
    x=np.asarray(x,float);prices=np.asarray(prices,float)
    lam=np.asarray(problem['elasticities'],float)
    d=np.asarray(problem['response_factors'],float)
    hours=np.asarray(problem['reduction_hours'],float)
    caps=np.asarray(problem['actual_amplitude_upper'],float)
    q=np.sum(np.abs(d),axis=1)
    price_upper=np.asarray(problem['price_upper'],float)
    bound=np.minimum(lam*price_upper,caps/(1+q))
    expected=float((hours/lam)@(x*x))
    hard=float((hours/lam*(1+q))@(x*x))
    A=np.asarray(problem['device_matrix'],float)
    b=np.asarray(problem['device_rhs'],float)
    worst=A@x+np.sum(np.abs((A*x)@d),axis=1)
    check('domain',label+'_finite',np.isfinite(x).all() and np.isfinite(prices).all())
    check('domain',label+'_price_link',np.max(np.abs(prices-x/lam))<1.e-7)
    check('domain',label+'_lower',x.min()>=-1.e-7)
    check('domain',label+'_amplitude_cap',np.max(x-bound)<=1.e-7)
    check('domain',label+'_price_cap',np.max(prices-price_upper)<=1.e-7)
    check('domain',label+'_hard_budget',hard<=float(problem['budget'])+1.e-7,value=hard)
    check('domain',label+'_device_rows',np.max(worst-b,initial=0.)<=1.e-7)
    for i,group in enumerate(problem['exclusive_groups']):
        check('domain',label+'_exclusivity_'+str(i),np.count_nonzero(x[group]>1.e-7)<=1)
    return expected,hard


def phase(phase,name,maximizing):
    global solver_phase_count
    if phase is None: return
    solver_phase_count += 1
    phase_statuses[str(phase['status_code'])]+=1
    obj,bound=phase.get('objective_value'),phase.get('best_bound')
    if obj is not None and bound is not None:
        tol=1.e-6*max(1.,abs(obj),abs(bound))
        check('solver',name+'_bound_direction',bound+tol>=obj if maximizing else bound-tol<=obj)
        check('solver',name+'_absolute_gap',close(phase['absolute_gap'],abs(obj-bound),tol))
    if phase.get('solver_reported_optimal'):
        check('solver',name+'_optimal_status',phase['status_code']==2)


def evaluate_tables(tables,prices,kind,coarse_id):
    rows=[];eligible=[];passes=0
    for sid,table in enumerate(tables):
        n=int(table['validation_size'])
        check('table',f'{kind}_s{sid}_validation_size',n==7680)
        losses={}
        flags={}
        for fam in table['families']:
            ident=int(fam['family_id'])
            check('table',f'{kind}_s{sid}_unique_id_{ident}',ident not in losses)
            check('table',f'{kind}_s{sid}_constant_nonnegative_{ident}',fam['constant_loss']>=0.)
            residual=np.asarray(fam['residual_base'],dtype=np.longdouble)
            coefficient=np.asarray(fam['residual_price'],dtype=np.longdouble)
            check('table',f'{kind}_s{sid}_coefficient_shape_{ident}',coefficient.shape==(len(residual),8))
            check('table',f'{kind}_s{sid}_array_finite_{ident}',np.isfinite(residual).all() and np.isfinite(coefficient).all())
            value=np.longdouble(fam['constant_loss'])+np.sum(np.abs(residual+coefficient@np.asarray(prices[8:],np.longdouble)),dtype=np.longdouble)/n
            losses[ident]=float(value);flags[ident]=bool(fam['has_target'])
        if kind=='current_coarse':
            available=coarse_id in losses and bool(losses)
            left=losses.get(coarse_id);right=min(losses.values()) if losses else None
        else:
            no=[v for i,v in losses.items() if not flags[i]]
            yes=[v for i,v in losses.items() if flags[i]]
            available=bool(no and yes and table.get('comparison_available',True))
            left=min(no) if no else None;right=min(yes) if yes else None
        margin=left-1.03*right-1.e-12 if available else None
        gain=float(table['acceptance_margin'])
        accepted=bool(gain>1.e-12 and available)
        passed=bool(accepted and margin>1.e-12)
        if accepted:eligible.append(margin)
        passes += passed
        rows.append(dict(losses=losses,left=left,right=right,margin=margin,gain=gain,
            available=available,accepted=accepted,passed=passed))
    required=int(np.ceil(.9*len(tables)))
    quantile=float(sorted(eligible,reverse=True)[required-1]) if len(eligible)>=required else None
    return rows,quantile,passes


def audit_problem(folder,table_source,problem,kind,coarse_id,train_prices=None):
    name=str(folder.relative_to(ROOT));tables=load_tables(table_source)
    solved=read(folder/'optimization.json');criterion=read(folder/'criterion_check.json')
    check('records',name+'_ten_scenes',len(tables)==10==solved['scenario_count'])
    check('records',name+'_price_incumbent',solved.get('feasible_incumbent',False))
    prices=np.asarray(solved['prices'],float);x=np.asarray(solved['amplitudes'],float)
    expected,hard=physical(problem,x,prices,name)
    check('domain',name+'_expected_fee_receipt',close(expected,solved['original_evaluation']['expected_cost'],1.e-7))
    check('domain',name+'_hard_fee_receipt',close(hard,solved['original_evaluation']['hard_budget_bound'],1.e-7))
    if train_prices is not None:
        target=np.zeros(8);target[[2,6]]=np.asarray(train_prices)*np.asarray(problem['elasticities'])[[2,6]]
    else:target=np.zeros(8)
    check('domain',name+'_training_quote_locked',np.max(np.abs(x[:8]-target))<1.e-9)
    rows,quantile,passes=evaluate_tables(tables,prices,kind,coarse_id)
    check('probability',name+'_full_denominator',criterion['denominator']==10==solved['scenario_count'])
    check('probability',name+'_required_count',solved['required_scenarios']==9)
    check('probability',name+'_passes',criterion['passes']==passes==solved['excitation_pass_count'])
    check('probability',name+'_sample_requirement',bool(solved['sample_requirement_met'])==(passes>=9))
    check('margin',name+'_criterion_quantile',close(quantile,criterion['quantile_margin']))
    check('margin',name+'_solved_quantile',close(None if quantile is None else quantile*1.e6,solved['quantile_micro'],2.e-6))
    check('margin',name+'_solved_gamma',close(None if quantile is None else quantile*1.e6,solved['gamma'],2.e-6))
    largest_loss_error=0.;largest_margin_error=0.
    for sid,row in enumerate(rows):
        check('margin',name+f'_s{sid}_margin',close(row['margin'],solved['scenario_margins'][sid]))
        if row['margin'] is not None:
            largest_margin_error=max(largest_margin_error,abs(row['margin']-solved['scenario_margins'][sid]))
        check('probability',name+f'_s{sid}_comparison_available',bool(solved['comparison_available'][sid])==row['available'])
        check('probability',name+f'_s{sid}_acceptance',bool(solved['acceptance_passes'][sid])==row['accepted'])
        check('probability',name+f'_s{sid}_excitation',bool(solved['excitation_passes'][sid])==row['passed'])
        stored=criterion['rows'][sid]
        check('margin',name+f'_s{sid}_checkfile',close(stored['margin'],row['margin']))
        check('margin',name+f'_s{sid}_gain',close(stored['acceptance_margin'],row['gain']))
        for ident,value in row['losses'].items():
            saved=float(stored['family_losses'][str(ident)])
            check('loss',name+f'_s{sid}_family{ident}',close(value,saved))
            largest_loss_error=max(largest_loss_error,abs(value-saved))
        if kind=='current_coarse':
            # Alias17 is just coarse expression15 on the other side; it adds
            # neither a fitted topology nor an LP solve.
            alias=max(row['losses'])+1
            check('alias',name+f'_s{sid}_alias_count',len(stored['family_losses'])==len(row['losses'])+1)
            check('alias',name+f'_s{sid}_coarse_alias',close(stored['family_losses'][str(alias)],row['losses'][coarse_id]))
            check('margin',name+f'_s{sid}_current_coarse_direct',close(stored['current_coarse_validation'],row['left']))
            check('margin',name+f'_s{sid}_best_all_direct',close(stored['best_pool_validation'],row['right']))
    for label,maximizing in [('primary',True),('secondary',False)]:phase(solved[label],name+'_'+label,maximizing)
    if solved.get('optimizer_gamma_micro') is not None and quantile is not None:
        check('solver',name+'_gamma_epigraph',solved['optimizer_gamma_micro']<=quantile*1.e6+2.e-6)
    if solved.get('secondary_quantile_floor_micro') is not None and quantile is not None:
        check('solver',name+'_secondary_quantile_floor',quantile*1.e6+2.e-6>=solved['secondary_quantile_floor_micro'])
    scopes={f['training_weights_scope'] for t in tables for f in t['families']}
    expected_scope='historical_frozen' if train_prices is None else 'training_refit_then_validation_frozen'
    check('semantics',name+'_parameter_rule_distinct',scopes=={expected_scope})
    row=dict(folder=name,table_sha256=sha(table_source),table_scenarios=len(tables),distinct_families=len(tables[0]['families']),
        criterion_kind=kind,parameter_rule=expected_scope,quantile_margin=quantile,passes=passes,denominator=10,
        expected_payment=expected,hard_budget_bound=hard,
        primary_status=solved['primary']['status_code'],secondary_status=solved['secondary']['status_code'] if solved.get('secondary') else None,
        max_replayed_loss_error=largest_loss_error,max_replayed_margin_error=largest_margin_error,
        source_includes_future_observations=False)
    problem_rows.append(row)
    return row


def main():
    global lp_receipt_count
    v1=ROOT/'plan_v1';v2=ROOT/'coarse_reprice_v2'
    p1=read(v1/'protocol.json');p2=read(v2/'protocol.json')
    summary1=read(v1/'summary.json');summary2=read(v2/'summary.json')
    selection1=read(v1/'price_selection.json');selection2=read(v2/'price_selection.json')
    problem=read(checked_ref(p1['problem'],'original_price_problem'))
    for label,protocol in [('v1',p1),('v2',p2)]:
        for key in ['history','historical_fit','problem']:
            checked_ref(protocol[key],label+'_'+key)
        check('configuration',label+'_budget',protocol['budget']==1.6==problem['budget'])
        check('configuration',label+'_risk',protocol['probability']==.9)
        check('configuration',label+'_samples',protocol['planning_samples']==10)
        check('configuration',label+'_same_planning_seed',protocol['planning_seed']==2026110131)
    check('configuration','bounded_outer_not_global',not selection1['global_outer_optimality'] and not selection2['global_outer_optimality'])
    check('configuration','v1_maximum_not_completed_count',summary1['completed']==13 and summary1['planned_max']==14 and summary1['failed']==0)
    check('configuration','outer_limit_preserved',not summary1['outer_solver']['success'] and summary1['outer_solver']['status']==1)
    check('configuration','v2_reuses_without_LP',summary2['new_training_LP_calls']==0 and summary2['prior_training_LP_calls']==2210)
    coarse_id=int(p2['coarse_family_id'])
    manifest=read(v1/'historical_family_manifest.json')
    check('configuration','historical_pool_complete_count',manifest['selection_points']==21 and manifest['distinct_complete_families']==17)
    check('configuration','v2_coarse_id',coarse_id==15)
    snapshots={folder.name:snapshot_audit(folder,protocol) for folder,protocol in [(v1,p1),(v2,p2)]}
    checked_ref(p2['prior_protocol'],'v2_prior_protocol');checked_ref(p2['prior_summary'],'v2_prior_summary')
    frozen_source=checked_ref(read(v2/'frozen_table_source.json'),'v2_frozen_table_source')
    audit_problem(v1/'historical_frozen',frozen_source,problem,'all_no_target',coarse_id)
    audit_problem(v2/'historical_frozen',frozen_source,problem,'current_coarse',coarse_id)
    records1={r['index']:r for r in summary1['evaluations']}
    records2={r['index']:r for r in summary2['evaluations']}
    check('records','same13_outer_indices',set(records1)==set(records2)==set(range(13)))
    distinct_train=set()
    for index in range(13):
        a,b=records1[index],records2[index]
        check('records',f'eval{index}_completed',a['status']==b['status']=='complete')
        check('records',f'eval{index}_training_quote_reused',a['training_prices']==b['training_prices'])
        distinct_train.add(tuple(a['training_prices']))
        source=ROOT/'plan_v1/outer'/f'eval_{index:03d}'/'private_tables.pkl'
        checked_ref(b['table_source'],f'eval{index}_v2_table_source')
        checked_ref(a['optimization'],f'eval{index}_v1_optimization')
        one=audit_problem(v1/'outer'/f'eval_{index:03d}',source,problem,'all_no_target',coarse_id,a['training_prices'])
        two=audit_problem(v2/'outer'/f'eval_{index:03d}',source,problem,'current_coarse',coarse_id,b['training_prices'])
        check('records',f'eval{index}_v1_summary_quantile',close(one['quantile_margin'],a['quantile_margin']))
        check('records',f'eval{index}_v2_summary_quantile',close(two['quantile_margin'],b['quantile_margin']))
        receipt=read(v1/'outer'/f'eval_{index:03d}'/'training_LP_receipts.json')
        check('LP',f'eval{index}_receipt_count',receipt['calls']==170==len(receipt['solves'])==a['training_LP_calls'])
        lp_receipt_count+=len(receipt['solves'])
        for j,solve in enumerate(receipt['solves']):
            label=f'eval{index}_solve{j}'
            check('LP',label+'_optimal',solve['status']==2 and solve['accepted'])
            check('LP',label+'_objective_recomputed',close(solve['objective'],solve['physical'],1.e-8))
            check('LP',label+'_constraints',solve['max_constraint_violation']<=1.e-7)
            check('LP',label+'_bounds',solve['max_bound_violation']<=1.e-7)
    check('records','13distinct_continuous_train_quotes',len(distinct_train)==13)
    check('records','actual_continuous_nonseed_quotes',any(t not in {(0.,0.),(.8,.8),(1.6,0.),(0.,1.6)} for t in distinct_train))
    check('LP','total2210training_LP',lp_receipt_count==summary1['total_training_LP_calls']==2210)
    for folder,summary,selection in [(v1,summary1,selection1),(v2,summary2,selection2)]:
        chosen=selection['selected_outer_index']
        record=summary['evaluations'][chosen]
        check('selection',folder.name+'_quote_matches_selected_record',selection['quotes']['continuous_validation']==record['amplitudes'])
        check('selection',folder.name+'_best_of_completed_quotes',record['quantile_margin']>=max(r['quantile_margin'] for r in summary['evaluations'])-2.e-12)
        check('selection',folder.name+'_no90claim',not selection['model_90_sample_feasible'] and record['passes']==0)
        check('selection',folder.name+'_negative_frozen_and_refit',summary['refit_arm_quantile']<0 and summary['frozen_arm_quantile']<0)
        for policy,x in selection['quotes'].items():
            physical(problem,x,np.asarray(x)/np.asarray(problem['elasticities']),folder.name+'_frozen_'+policy)
    holdout_protocol=read(ROOT/'model_holdout_v2/protocol.json')
    pinned=checked_ref(holdout_protocol['price_selection'],'holdout_frozen_price_selection')
    freeze_hash=sha(v2/'price_selection.json')
    check('freeze','new_selection_hash',freeze_hash=='317ba149694d9d37e2dfa246923645d87e71b5f9b668637512303c2111ec37f9')
    check('freeze','holdout_uses_frozen_v2',pinned.resolve()==(v2/'price_selection.json').resolve())
    check('freeze','independent_holdout_seed',holdout_protocol['seed']==2026110132!=p1['planning_seed'])
    return dict(status='complete' if not failures else 'failed',created_utc=datetime.now(timezone.utc).isoformat(),
        audit_passed=not failures,check_count=sum(checks.values()),checks_by_category=dict(checks),failures=failures,
        computation=dict(new_solver_calls=0,new_LP=0,new_AC=0,new_actual_observations_read=0,true_graphs_read=0),
        denominators=dict(outer_training_quotes=13,unique_historical_families=17,planning_scenarios=10,
            original_training_LP=lp_receipt_count,validation_optimization_problems_v1=14,
            validation_optimization_problems_v2=14,solver_phases_checked=solver_phase_count,
            corrected_projection_aliases_are_not_new_families=True),
        solver_phase_statuses=dict(phase_statuses),source_snapshots=snapshots,
        source_scope_note='v1 snapshot55files verifies its own protocol only; v2 snapshot15files separately pins newly added reprice code',
        frozen_price_selection_sha256=freeze_hash,planning_records=problem_rows,artifact_sha256=artifacts,
        negative_results_scope='All13 completed fixed-training-quote inner optima and historical-frozen branch have negative sample90 margin. Outercontinuous training search is bounded and not globally optimal.',
        probability_scope='Planning sample excitation only; nofield90 or topologycorrectness assertion. Modelholdout results and actual futures not read.')


if __name__=='__main__':
    try:result=main()
    except Exception as error:
        import traceback
        result=dict(status='failed',audit_passed=False,error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_by_category=dict(checks),failures=failures)
    destination=ROOT/'planning_audit.json'
    if destination.exists():raise FileExistsError(destination)
    result['audit_source_sha256']=sha(__file__)
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ['status','audit_passed','check_count','denominators','solver_phase_statuses','frozen_price_selection_sha256','error_type','message']},ensure_ascii=False))
    if result.get('failures'):print(json.dumps(result['failures'][:20],ensure_ascii=False))
    sys.exit(0 if result['audit_passed'] else 1)
