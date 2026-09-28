from itertools import combinations
import numpy as np
import pandas as pd
import pytest
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.rnj.rooted_ablation_support import BoundedPath, rooted_scores

def scenarios():
    rng=np.random.default_rng(3090801);p=rng.normal(size=(18,4));q=rng.normal(size=(18,4))
    supports=((0,),(1,),(2,),(3,),(0,1),(2,3))
    r,x=core.build_matrices_from_atoms(4,supports,[.1,.2,.1,.15,.4,.2],[.2,.1,.1,.2,.15,.3])
    return [{'name':'same_scenario','P_terminal':pd.DataFrame(p),'Q_terminal':pd.DataFrame(q),'drop_target':pd.DataFrame(p@r.T+q@x.T)}]

def all_supports(n):
    return tuple(support for size in range(1,n+1) for support in combinations(range(n),size))

def test_enumeration_matches_independent_native_milp():
    s=scenarios();initial=tuple((i,) for i in range(4));pool=all_supports(4)
    native=core.solve_best_laminar_extension_l1(s,initial,r_upper_bound=2,x_upper_bound=2,time_limit=5)
    original=core._solve_extension_prepared
    with BoundedPath(total_seconds=10,solve_seconds=3,enumerate_pool=True,pool=pool) as adapter:
        adapted=core.solve_best_laminar_extension_l1(s,initial,r_upper_bound=2,x_upper_bound=2,time_limit=3)
    assert core._solve_extension_prepared is original
    assert core.solver_diagnostics_prove_optimality(native.diagnostics)
    assert core.solver_diagnostics_prove_optimality(adapted.diagnostics)
    assert abs(native.objective-adapted.objective)<1e-8
    assert adapter.enumerations[0]['all_candidates_certified_optimal']
    assert len(adapter.enumerations[0]['candidate_results'])==11

def test_incomplete_enumeration_has_no_exact_support():
    s=scenarios();initial=tuple((i,) for i in range(4))
    with BoundedPath(total_seconds=1e-12,solve_seconds=2,enumerate_pool=True,pool=((0,1),)) as adapter:
        result=core.solve_best_laminar_extension_l1(s,initial,r_upper_bound=2,x_upper_bound=2,time_limit=2)
    assert result.support is None
    assert not core.solver_diagnostics_prove_optimality(result.diagnostics)

def test_whole_path_full_enumeration_matches_unrestricted_native_milp():
    s=scenarios();initial=tuple((i,) for i in range(4));pool=all_supports(4)
    native=core.fit_laminar_l1_sensitivity(s,validation_scenarios=s,initial_supports=initial,r_upper_bound=2,x_upper_bound=2,time_limit=5)
    with BoundedPath(total_seconds=10,solve_seconds=3,enumerate_pool=True) as adapter:
        result=adapter.fit(s,validation_scenarios=s,initial_supports=initial,candidate_supports=pool)
    assert abs(native.train_mae-result.train_mae)<1e-8
    assert set(native.support_indices)==set(result.support_indices)
    assert np.allclose(native.r_matrix,result.r_matrix,atol=1e-7)

def test_restricted_research_pool_is_retained_by_adapter():
    s=scenarios();initial=tuple((i,) for i in range(4));pool=((0,1),)
    with BoundedPath(total_seconds=10,solve_seconds=3,enumerate_pool=True) as adapter:
        result=adapter.fit(s,validation_scenarios=s,initial_supports=initial,candidate_supports=pool)
    assert adapter.pool==pool
    assert set(result.support_indices)<=set(initial+pool)
    assert any(attempt.support==(0,1) for attempt in adapter.attempts)

def test_removed_native_allowlist_cannot_silently_become_unrestricted():
    s=scenarios();initial=tuple((i,) for i in range(4))
    with BoundedPath(enumerate_pool=False) as adapter:
        with pytest.raises(ValueError,match='replay this historical'):
            adapter.fit(s,validation_scenarios=s,initial_supports=initial,candidate_supports=((0,1),))
    with pytest.raises(ValueError,match='replay this historical'):
        BoundedPath(enumerate_pool=False,pool=())

def test_root_terminal_partition_uses_clades_only():
    truth={frozenset({1,2}),frozenset({1,2,3}),frozenset({4,5})}
    result=rooted_scores(truth,truth,[1,2,3,4,5,6])
    assert result['root_clade_f1']==result['terminal_clade_f1']==result['root_partition_exact']==result['terminal_parent_exact']==1
    missing=rooted_scores(truth-{frozenset({1,2})},truth,[1,2,3,4,5,6])
    assert missing['root_clade_f1']==1
    assert missing['terminal_clade_f1']<1
    assert missing['terminal_parent_exact']==4/6


def test_validation_fallback_can_select_candidate_itself_and_ties_reference():
    from research_experiments.rnj.rooted_ablation_support import validation_selection
    baseline={'method':'rnj_fixed_tree_lp','validation_mae':.2,'clade_f1':.8,'test_rmse':100}
    candidate={'method':'joint','validation_mae':.1,'clade_f1':.3,'test_rmse':200}
    candidate.update(validation_selection(candidate,baseline))
    assert candidate['validation_selected_method']=='joint'
    assert candidate['validation_selected_clade_f1']==.3
    assert candidate['validation_selected_test_rmse']==200
    tie={'method':'joint','validation_mae':.2,'test_rmse':0}
    assert validation_selection(tie,baseline)['validation_selected_method']=='rnj_fixed_tree_lp'


def test_validation_selection_reference_wins_on_validation_despite_worse_test():
    from research_experiments.rnj.rooted_ablation_support import validation_selection
    reference={'method':'rnj_fixed_tree_lp','validation_mae':.1,'test_rmse':100}
    candidate={'method':'joint','validation_mae':.2,'test_rmse':0}
    assert validation_selection(candidate,reference)['validation_selected_method']=='rnj_fixed_tree_lp'
