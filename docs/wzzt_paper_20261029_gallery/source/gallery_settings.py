"""Document the frozen experiment settings by cohort and method, without retuning."""
import json
import pandas as pd

EXTRA_LABELS={
 'ridge_1e3':('Ridge pool / strength 0.001','#009e73','D'),
 'ridge_1e2':('Ridge pool / strength 0.01','#40ad8b','s'),
 'restart100':('Original-pool restart / 100 s','#536e92','s'),
 'ridge100':('Ridge-pool restart / 100 s','#009e73','D'),
 'ridge_two_step':('Predeclared two-step / bundle 0','#009e73','D'),
 'ridge_two_step_guard0':('Predeclared two-step + delete / bundle 0','#7b2cbf','X')}


def settings_table(results):
    rows=[]
    for cohort,method in sorted(set(zip(results.cohort,results.variant))):
        native=method in ('NJ','RNJ_native');bank=method=='RNJ_L1_bank';free=cohort=='free_domain';joint=cohort=='joint_extension';filter_only=cohort=='postfilter_guard'
        delete=method in ('multistart_guard0','ridge_guard0','ridge_two_step_guard0','joint_width5_delete')
        pair=method in ('ridge_two_step','ridge_two_step_guard0','joint_width5','joint_width5_delete')
        ridge=method.startswith('ridge') and method!='ridge_postfilter_guard'
        strengths=[.001] if method=='ridge_1e3' else [.01] if method=='ridge_1e2' else [.001,.01] if ridge or joint or method=='ridge_postfilter_guard' else []
        base_starts=0 if native or bank else 1 if method in ('baseline25','WZZT_leaf_path','WZZT_leaf_bank') or free else 3
        if free:stages=[dict(stage='compatible free-support extension path',seconds=100,per_MIP_seconds=10)]
        elif filter_only or method=='ridge_postfilter_guard':stages=[dict(stage='reuse selected parent; no new search',seconds=0)]
        elif joint:stages=[dict(stage='frozen parent bank; additional proposal path',seconds=25)]
        else:
            stages=[] if native or bank else [dict(stage='base single/leaf path' if base_starts==1 else 'base multistart paths',seconds_each=25,max_starts=base_starts)]
            if method in ('restart_control','ridge_union','ridge_guard0','ridge_1e3','ridge_1e2','ridge_two_step','ridge_two_step_guard0'):stages.append(dict(stage='additional candidate-pool repair',seconds=25))
            if method in ('restart100','ridge100'):stages.append(dict(stage='additional candidate-pool repair',seconds=100))
            if pair:stages.append(dict(stage='joint two-support proposal',seconds=25,width=5))
        if delete:stages.append(dict(stage='validation-admitted backward deletion',seconds=15,max_deletions=1))
        initial='fixed NJ/RNJ family' if native else 'complete RNJ bank' if bank else 'selected ridge parent model' if filter_only or method=='ridge_postfilter_guard' else 'singleton leaves' if free or method.startswith('WZZT_leaf') else 'frozen ridge bank and selected family' if joint else 'RX75 RNJ initial plus selected extra starts'
        rows.append(dict(settings_id=cohort+'::'+method,cohort=cohort,variant=method,
            actual_input_records=int(((results.cohort==cohort)&results.variant.eq(method)).sum()),
            initial_family=initial,uses_frozen_parent_precomputation=joint or free or filter_only or method=='ridge_postfilter_guard',
            nominal_stage_budgets_json=json.dumps(stages),parent_base_start_limit=base_starts,
            fixed_LP_limit_seconds=60 if cohort=='lp60_retry' else 15,
            worker_hard_seconds=None if filter_only or method=='ridge_postfilter_guard' else 360 if free or joint else 900,
            threads_per_solver=1,fixed_LP_method='dual simplex (Method=1)',
            fixed_LP_feasibility_tolerance=1e-9,LP_optimality_tolerance=1e-9,response_coefficient_lower=0.,response_coefficient_upper=2.,
            MILP_relative_gap=.01 if free else None,MILP_per_solve_seconds=10 if free else None,
            greedy_absolute_gain_threshold=None if native or bank or filter_only else 1e-10,
            greedy_relative_gain_threshold=None if native or bank or filter_only else .001,
            candidate_ridge_strengths_json=json.dumps(strengths),
            ridge_alpha_formula='strength * max(||[P,Q]||_F^2/(2*n),1e-12); raw/demean separately' if strengths else None,
            front_preprocessing_json=json.dumps(['raw','demean']),geometry_grid_json=json.dumps(['R','RX75','X']),
            geometry_threshold_grid_json=json.dumps([0.,.04,.08,.16,.24]),
            native_geometry='raw_RX75' if native else None,native_geometry_threshold=.16 if method=='NJ' else .24 if method=='RNJ_native' else None,
            validation_rule='fixed returned family; no pooled selection' if native else 'minimum validation MAE +3% band, fewest supports, MAE, fixed order',
            final_strength_filter='native family retained' if native else 'validation choice among 0/.04/.08/.16' if filter_only or method=='ridge_postfilter_guard' else '.16 * calibrated median root depth; leaves retained',
            max_backward_deletions=1 if delete else 0,deletion_admission_validation_slack=0. if delete else None,
            joint_width=5 if pair else None,joint_first_sweep_budget_fraction=.4 if pair else None,
            actual_selected_floor_field='model_results.final_floor or parameters.point_json.final_floor' if filter_only or method=='ridge_postfilter_guard' else None,
            timing_interpretation='nominal stage budgets; LP calls may cross path checks; not independent deployment cost'))
    return pd.DataFrame(rows)
