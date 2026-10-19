"""Analytic centering identities, boundary cases and proxy-bound checks."""
import numpy as np
import pytest

from research_experiments.mobile_ev_mle.history_cost_centering import (center_history_costs, project_history,
    original_bounds_from_common_centering, evaluate_original_ac_witness)
from research_experiments.mobile_ev_mle.ev_information_milp import FrozenPair, solve_pair, support_cut, adaptive_search
from research_experiments.mobile_ev_mle.ev_operations_milp import EVConfig


def _pair(name="one",lower=-2.,upper=2.):
    return FrozenPair(name,np.array([1.,3.,0.]),np.array([[0.],[0.],[1.]]),
        np.array([[1.],[0.],[-1.]]),np.array([[.5]]),np.array([lower]),np.array([upper]),
        metadata={"history_row_widths":[2],"future_row_start":2})


@pytest.mark.parametrize("u",[-2.,0.,.7,3.])
def test_exact_projection_identity_for_all_boxed_parameters_and_qp(u):
    pair = _pair()
    new,summary = center_history_costs([pair])
    centered = new[0]
    assert summary["records"][0]["unrestricted_affine_history_floor"] == pytest.approx(9)
    np.testing.assert_allclose(centered.c[:2],[1.,0.])
    for z in np.linspace(-2,2,17):
        original = pair.residual([u],[z]); changed = centered.residual([u],[z])
        assert original@original == pytest.approx(changed@changed+9,abs=1e-12)
    old,newsol = solve_pair(pair,[u]),solve_pair(centered,[u])
    assert old["q_upper"] == pytest.approx(newsol["q_upper"]+9,abs=1e-10)
    assert np.array_equal(centered.F,pair.F)


def test_nonzero_projected_gradient_and_active_boundary_penalty_retained():
    pair = FrozenPair("boundary",np.array([2.,3.]),np.zeros((2,1)),
        np.array([[1.],[0.]]),np.zeros((0,1)),np.array([0.]),np.array([1.]),
        metadata={"history_row_widths":[2]})
    centered = center_history_costs([pair])[0][0]
    assert 2*pair.A.T@pair.c == pytest.approx(2*centered.A.T@centered.c)
    assert solve_pair(pair,[0])["q_upper"] == pytest.approx(13)
    assert solve_pair(centered,[0])["q_upper"] == pytest.approx(4)
    # Thus removed floor9 is not constrained historical minimum13.
    assert center_history_costs([pair])[1]["records"][0]["unrestricted_affine_history_floor"] == pytest.approx(9)


def test_fixed_columns_absorbed_before_projection_and_auxiliary_F_unchanged():
    pair = FrozenPair("fixed",np.array([1.,1.,0.]),np.array([[0.],[0.],[1.]]),
        np.array([[1.,0.],[0.,1.],[0.,-1.]]),np.array([[2.,3.]]),
        np.array([2.,-1.]),np.array([2.,1.]),metadata={"history_row_widths":[2]})
    new,summary = center_history_costs([pair])
    assert summary["records"][0]["unrestricted_affine_history_floor"] == pytest.approx(9)
    assert summary["records"][0]["fixed_coordinate_count"] == 1
    np.testing.assert_allclose(new[0].c[:2],[-2.,1.])
    for z in np.linspace(-1,1,9):
        oldr,newr = pair.residual([.7],[2.,z]),new[0].residual([.7],[2.,z])
        assert oldr@oldr == pytest.approx(newr@newr+9,abs=1e-12)


def test_all_fixed_history_is_constant_and_rank_deficiency_is_preserved():
    pair = FrozenPair("fixed_all",np.array([1.,2.]),np.zeros((2,1)),np.eye(2),
        np.zeros((0,2)),np.array([2.,3.]),np.array([2.,3.]),metadata={"history_row_widths":[1,1]})
    new,summary = center_history_costs([pair])
    assert summary["records"][0]["free_history_rank"] == 0
    assert summary["records"][0]["unrestricted_affine_history_floor"] == pytest.approx(34)
    np.testing.assert_allclose(new[0].residual([0],[2,3]),[0,0])
    rankdef = FrozenPair("rankdef",np.array([2.,-1.]),np.zeros((2,1)),np.array([[1.,2.],[2.,4.]]),
        np.zeros((0,2)),np.full(2,-1.),np.ones(2),metadata={"history_row_widths":[2]})
    projection = project_history(rankdef)
    assert projection.rank == 1
    assert projection.floor == pytest.approx(5)


def test_joint_history_projection_handles_shared_variables_across_blocks():
    # Separate per-block projection would remove zero, whereas the one common
    # parameter cannot explain [1,-1] together. The true unrestricted floor2
    # belongs to the complete historical prefix.
    pair = FrozenPair("shared",np.array([1.,-1.]),np.zeros((2,1)),np.ones((2,1)),
        np.zeros((0,1)),np.array([-2.]),np.array([2.]),metadata={"history_row_widths":[1,1]})
    projection = project_history(pair)
    assert projection.floor == pytest.approx(2)
    assert np.linalg.norm(projection.projected) < 1e-14


def _constant_pair(name,floor,intercept,slope):
    return FrozenPair(name,np.array([np.sqrt(floor),intercept]),np.array([[0.],[slope]]),
        np.zeros((2,0)),np.zeros((0,0)),np.zeros(0),np.zeros(0),
        metadata={"history_row_widths":[1],"future_row_start":1})


def _outer_score(pairs,u):
    return min(solve_pair(p,[u])["q_upper"] for p in pairs)


def test_pairwise_projection_changes_ranking_and_drops_bad_history_evidence():
    pairs = [_constant_pair("bad_history",101,0,1),_constant_pair("good_history",1,1,-1)]
    projected,meta = center_history_costs(pairs)
    grid = np.linspace(0,1,1001)
    old = np.array([_outer_score(pairs,u) for u in grid])
    new = np.array([_outer_score(projected,u) for u in grid])
    assert grid[old.argmax()] == 0
    assert grid[new.argmax()] == .5
    assert meta["relative_historical_bad_fit_evidence_removed"]
    assert not meta["criterion_action_order_preserved_in_exact_arithmetic"]


def test_common_floor_preserves_outer_order_none_gains_and_restores_bounds():
    pairs = [_constant_pair("first",101,0,1),_constant_pair("second",1,1,-1)]
    before = [p.fingerprint() for p in pairs]
    centered,summary = center_history_costs(pairs,mode="common_floor")
    assert summary["common_floor"] == 1
    assert [p.fingerprint() for p in pairs] == before
    for u in np.linspace(0,1,13):
        assert _outer_score(pairs,u) == pytest.approx(_outer_score(centered,u)+1,abs=1e-12)
        assert _outer_score(pairs,u)-_outer_score(pairs,0) == pytest.approx(
            _outer_score(centered,u)-_outer_score(centered,0),abs=1e-12)
    assert summary["criterion_action_order_preserved_in_exact_arithmetic"]
    restored = original_bounds_from_common_centering(0,1,summary)
    assert restored["lower"] <= 1 <= 2 <= restored["upper"]
    assert np.all(centered[0].A[-1] == 0) and np.all(centered[0].B[-1] == 0)


def test_same_common_baseline_can_normalize_fixed_and_refit_families():
    pair = _pair("refit")
    fixed = FrozenPair("fixed",pair.c,pair.B,pair.A,pair.F,np.array([0.]),np.array([0.]),
        metadata=pair.metadata)
    new,summary = center_history_costs([fixed,pair],mode="common_floor")
    assert summary["common_floor"] == pytest.approx(9)
    for old,changed in zip([fixed,pair],new):
        assert solve_pair(old,[.8])["q_upper"] == pytest.approx(solve_pair(changed,[.8])["q_upper"]+9,abs=1e-11)


def test_truncated_small_columns_have_reported_nonzero_equivalence_defect():
    pair = FrozenPair("tiny",np.array([0.,1.]),np.zeros((2,1)),np.diag([1.,1e-9]),
        np.zeros((0,2)),np.array([-1.,-1e9]),np.array([1.,1e9]),metadata={"history_row_widths":[2]})
    new,summary = center_history_costs([pair],mode="pairwise_projection",rcond=1e-5)
    record = summary["records"][0]
    assert record["free_history_rank"] == 1
    assert record["unrestricted_affine_history_floor"] is None
    assert record["discarded_positive_singular_values"] == 1
    assert record["decomposition_defect_lower"] <= -2
    assert record["decomposition_defect_upper"] >= 2
    for z in ([0.,-1e9],[0.,0.],[0.,1e9]):
        oldr,newr = pair.residual([0],z),new[0].residual([0],z)
        defect = oldr@oldr-newr@newr-record["removed_constant"]
        assert record["decomposition_defect_lower"] <= defect <= record["decomposition_defect_upper"]


def test_common_mode_rejects_positive_truncation_and_default_retains_tiny_column():
    pair = FrozenPair("tiny_common", np.array([0., 1.]), np.zeros((2, 1)), np.diag([1., 1e-9]),
        np.zeros((0, 2)), np.array([-1., -1e9]), np.array([1., 1e9]), metadata={"history_row_widths": [2]})
    with pytest.raises(ValueError, match="positive SVD truncation"):
        center_history_costs([pair], mode="common_floor", rcond=1e-5)
    changed, summary = center_history_costs([pair], mode="common_floor")
    assert summary["common_floor"] == 0.0
    assert summary["records"][0]["free_history_rank"] == 2
    assert summary["criterion_action_order_preserved_in_exact_arithmetic"]
    assert not summary["numerical_action_order_certified"]
    for z in ([0., -1e9], [0., 0.], [0., 1e9]):
        np.testing.assert_allclose(np.r_[pair.residual([0.], z), 0.], changed[0].residual([0.], z), atol=0.0)


def test_full_truncation_defect_bounds_cover_original_box_endpoints_and_interior():
    pair = FrozenPair("truncated_all", np.array([10., 20., .5]), np.array([[0.], [0.], [1.]]),
        np.array([[2., -3.], [5., 7.], [1., -1.]]), np.zeros((0, 2)),
        np.array([-3., -2.]), np.array([4., 6.]), metadata={"history_row_widths": [2]})
    changed, summary = center_history_costs([pair], rcond=2.0)
    record = summary["records"][0]
    assert record["free_history_rank"] == 0 and record["discarded_positive_singular_values"] == 2
    for first in [-3., -1., 0., 4.]:
        for second in [-2., 0., 1., 6.]:
            for control in [-2., .5, 3.]:
                old, new = pair.residual([control], [first, second]), changed[0].residual([control], [first, second])
                defect = float(old @ old - new @ new - record["removed_constant"])
                assert record["decomposition_defect_lower"] <= defect <= record["decomposition_defect_upper"]


def test_deep_metadata_copy_fresh_solver_cache_and_original_arrays_remain_untouched():
    pair = _pair()
    pair.metadata["nested"] = {"physical_indices": [1, 2]}
    original = {key: getattr(pair, key).copy() for key in ("c", "A", "B", "F", "lower", "upper", "scale")}
    pair._solver_cache["sentinel"] = "original"
    changed, _ = center_history_costs([pair], mode="common_floor")
    changed[0].metadata["nested"]["physical_indices"][0] = 99
    assert pair.metadata["nested"]["physical_indices"] == [1, 2]
    assert changed[0]._solver_cache == {}
    assert pair._solver_cache == {"sentinel": "original"}
    for key, values in original.items():
        np.testing.assert_array_equal(getattr(pair, key), values)
    assert "history_cost_centering" not in pair.metadata


def test_qr_hook_preserves_original_history_column_layout_and_checks_full_gradient():
    pair = FrozenPair("qr", np.array([1., 3., 0.]), np.array([[0.], [0.], [1.]]),
        np.array([[1., 0.], [0., 0.], [0., 1.]]), np.zeros((0, 2)),
        np.full(2, -10.), np.full(2, 10.), metadata={"history_row_widths": [2], "future_row_start": 2,
        "free_indices": [0, 1], "variable_layout": [{"history_state": [0, 1]}]})
    changed, _ = center_history_costs([pair], mode="common_floor")
    result = solve_pair(changed[0], [2.])
    assert result["acceleration"] == "history_QR"
    assert result["q_upper"] == pytest.approx(0., abs=1e-12)
    assert result["numeric_gap"] < 1e-10
    assert changed[0].metadata["future_row_start"] == 2
    assert np.all(changed[0].A[2:, 0] == 0.)


def test_qr_hook_falls_back_when_recovered_history_would_violate_original_box():
    pair = FrozenPair("qr_boundary", np.array([2., 3., 0.]), np.array([[0.], [0.], [1.]]),
        np.array([[1., 0.], [0., 0.], [0., 1.]]), np.zeros((0, 2)),
        np.array([0., -10.]), np.array([1., 10.]), metadata={"history_row_widths": [2], "future_row_start": 2,
        "free_indices": [0, 1], "variable_layout": [{"history_state": [0, 1]}]})
    changed, _ = center_history_costs([pair], mode="common_floor")
    result = solve_pair(changed[0], [2.])
    assert result["acceleration"] == "full_BVLS"
    assert result["q_upper"] == pytest.approx(4.)
    assert result["z"][0] == pytest.approx(0.)


def test_ac_adapter_uses_original_residual_and_checks_original_metadata(monkeypatch):
    import research_experiments.mobile_ev_mle.adaptive_history as history_core
    pair = _pair()
    pair.metadata["variable_layout"] = [{"beta": [0, 1]}]
    changed, _ = center_history_costs([pair], mode="common_floor")
    calls = []
    def original_only(context, raw, u, z):
        calls.append(raw)
        np.testing.assert_array_equal(raw["c"], pair.c)
        assert len(raw["c"]) + 1 == len(changed[0].c)
        return {"scope": "one original nonlinear witness", "value": 11.0}
    monkeypatch.setattr(history_core, "evaluate_frozen_pair_actual", original_only)
    result = evaluate_original_ac_witness({}, pair, changed[0], [0.], [0.])
    assert result["value"] == 11.0 and len(calls) == 1
    assert changed[0].metadata["requires_original_pair_for_ac_rollback"]
    assert not changed[0].metadata["original_ac_rollback_compatible"]
    pair.metadata["variable_layout"][0]["beta"] = [1, 2]
    with pytest.raises(ValueError, match="physical metadata changed"):
        evaluate_original_ac_witness({}, pair, changed[0], [0.], [0.])


def test_repeated_centering_and_wrong_original_ac_pair_are_rejected():
    pair = _pair()
    changed, _ = center_history_costs([pair], mode="common_floor")
    with pytest.raises(ValueError, match="already centered"):
        center_history_costs(changed, mode="common_floor")
    with pytest.raises(ValueError, match="does not match"):
        evaluate_original_ac_witness({}, _pair("other"), changed[0], [0.], [0.])


def test_existing_direction_bounds_apply_to_transformed_canonical_norm():
    original = _pair()
    centered = center_history_costs([original],mode="common_floor")[0][0]
    rng = np.random.default_rng(31)
    for _ in range(10):
        direction = rng.normal(size=len(centered.c)+len(centered.F))
        direction /= np.linalg.norm(direction)
        cut = support_cut(centered,direction,[-2],[2])
        for u in np.linspace(-2,2,13):
            distance = np.sqrt(solve_pair(centered,[u])["q_upper"])
            assert cut.value([u]) <= distance+1e-12


def test_two_slot_actual_milp_bounds_remain_valid_and_restore_original():
    pair = FrozenPair("two_slot",np.array([3.,0.]),np.array([[0.,0.],[1.,-1.]]),
        np.zeros((2,0)),np.zeros((0,0)),np.zeros(0),np.zeros(0),metadata={"history_row_widths":[1]})
    centered,summary = center_history_costs([pair],mode="common_floor")
    config = EVConfig(n_slots=2,n_sites=1,net_grid_energy_kwh=1.5,p_max_kw=6.)
    result = adaptive_search(config,centered,max_iterations=4,segments=2,max_segments=8,
        solver_seconds=3,total_seconds=15,absolute_gap=1e-7,relative_gap=1e-7)
    assert result["success"]
    # Independent analytic control domain: u0+u1=6, 0<=ui<=6.
    # max(u0-u1)^2=36; original constant history contributes9.
    assert result["proxy_lower"] <= 36+1e-5
    assert result["proxy_upper"] >= 36-1e-5
    restored = original_bounds_from_common_centering(result["proxy_lower"],result["proxy_upper"],summary)
    assert restored["lower"] <= 45+1e-5 and restored["upper"] >= 45-1e-5


@pytest.mark.parametrize("metadata",[{}, {"history_row_widths":[0]},
    {"history_row_widths":[2],"future_row_start":1}, {"history_row_widths":[4]}])
def test_bad_historical_layout_rejected(metadata):
    pair = _pair()
    malformed = FrozenPair(pair.name,pair.c,pair.B,pair.A,pair.F,pair.lower,pair.upper,metadata=metadata)
    with pytest.raises(ValueError,match="history|historical"):
        project_history(malformed)


def test_invalid_common_baseline_and_control_dependent_history_rejected():
    with pytest.raises(ValueError,match="between"):
        center_history_costs([_pair()],mode="common_floor",common_floor=10)
    with pytest.raises(ValueError,match="only meaningful"):
        center_history_costs([_pair()],common_floor=0)
    pair = _pair(); b = pair.B.copy();b[0,0]=1e-20
    with pytest.raises(ValueError,match="exactly zero"):
        project_history(FrozenPair(pair.name,pair.c,b,pair.A,pair.F,pair.lower,pair.upper,metadata=pair.metadata))
    with pytest.raises(ValueError,match="common restored"):
        original_bounds_from_common_centering(0,1,center_history_costs([pair])[1])
