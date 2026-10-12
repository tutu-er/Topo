"""Observation interface, structure replacement, and real estimator integration."""
import inspect
import numpy as np
import pytest

from research_experiments.mobile_ev_probing import passive_bridge_v9 as bridge


def _six_leaf_data(seed=991, t=36):
    rng = np.random.default_rng(seed)
    family = ((0, 1, 2), (3, 4, 5), (0, 1), (3, 4))
    supports = (tuple(range(6)), *family, *((i,) for i in range(6)))
    theta = np.array([.008, .02, .03, .015, .025, *([.04] * 6)])
    p = rng.uniform(.03, .25, (t, 6))
    q = .3 * p
    y = np.zeros_like(p)
    for s, value in zip(supports, theta, strict=True):
        z = np.array([float(i in s) for i in range(6)])
        y += value * (p[:, s].sum(axis=1) + .5 * q[:, s].sum(axis=1))[:, None] * z
    # A time-varying common source response; not exposed as truth to bridge.
    source_p = np.full(t, .1)
    y += .008 * (source_p + .5 * .3 * source_p)[:, None]
    return p, q, y, family


def test_local_interchange_can_remove_wrong_rnj_cherry():
    # A wrong {0,2} clade is merely a proposal and can become {0,1}.
    wrong = ((0, 1, 2), (3, 4, 5), (0, 2), (3, 4))
    expected = bridge.normalize_family(((0, 1, 2), (3, 4, 5), (0, 1), (3, 4)), 6)
    neighbors = bridge.family_neighbors(wrong, 6)
    assert expected in neighbors
    assert neighbors[expected] == "rooted_interchange"
    assert (0, 2) not in expected
    assert bridge.normalize_family(((0, 1, 2), (3, 4, 5), (3, 4)), 6) in neighbors


def test_neighbors_preserve_laminarity_for_multifurcations():
    family = ((0, 1, 2, 3), (0, 1), (4, 5))
    for candidate in bridge.family_neighbors(family, 6):
        sets = list(map(set, candidate))
        assert all(not (a & b) or a <= b or b <= a for a in sets for b in sets)


def test_real_current_rnj_recovers_informative_six_leaf_example():
    p, q, y, truth_for_test_only = _six_leaf_data()
    result = bridge.build_passive_candidates(p, q, y, run_wzzt=False, max_candidates=16)
    target = bridge.normalize_family(truth_for_test_only, 6)
    found = [row for row in result["candidates"] if tuple(map(tuple, row["supports"])) == target]
    assert found
    assert any(origin.startswith("rnj:") for origin in found[0]["proposal_origins"])
    assert found[0]["linear_contrast_mae"] < 1e-10
    assert all(row["ac_evidence"] is None and row["replaceable"] for row in result["candidates"])
    assert all(item["success"] for item in result["diagnostics"]["rnj"])


def test_actual_wzzt_restarts_with_only_singletons():
    p, q, y, _ = _six_leaf_data(t=10)
    result = bridge.build_passive_candidates(p, q, y, source_p_nominal=.1,
        source_q_nominal=.03, wzzt_time_limit=2., wzzt_max_atoms=7,
        neighbor_seed_count=0, max_candidates=12)
    diagnostics = result["diagnostics"]["wzzt"]
    assert diagnostics["called"] and "error" not in diagnostics
    assert diagnostics["mandatory_supports"] == [[i] for i in range(6)]
    assert not diagnostics["global_topology_optimality"]
    assert diagnostics["solver_attempts"]
    assert any("wzzt:accepted_path:0" in row["proposal_origins"] for row in result["candidates"])


def test_wzzt_failure_keeps_rnj_and_empty_certificates(monkeypatch):
    captured = {}
    def timed_out(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("controlled solver timeout")
    monkeypatch.setattr(bridge, "fit_source_aware_wzzt", timed_out)
    p, q, y, _ = _six_leaf_data()
    result = bridge.build_passive_candidates(p, q, y, max_candidates=8)
    assert captured["certified_supports"] == ()
    assert "initial_supports" not in captured
    assert len(result["candidates"]) == 8
    assert "controlled solver timeout" in result["diagnostics"]["wzzt"]["error"]
    assert result["certified_supports"] == []
    assert all(row["certified_supports"] == [] for row in result["candidates"])


def test_score_profiles_out_common_response_but_does_not_exclude():
    p, q, y, family = _six_leaf_data()
    drift = np.linspace(-.2, .2, len(y))[:, None]
    first = bridge.contrast_rank(p + .5 * q, y, family, .2)
    second = bridge.contrast_rank(p + .5 * q, y + drift, family, .2)
    assert first["linear_contrast_mae"] == pytest.approx(second["linear_contrast_mae"], abs=1e-12)
    assert "excluded" not in first


def test_input_interface_has_no_truth_or_certification_argument():
    parameters = inspect.signature(bridge.build_passive_candidates).parameters
    assert "truth" not in parameters and "certified_supports" not in parameters
    p, q, y, _ = _six_leaf_data()
    old = (p.copy(), q.copy(), y.copy())
    bridge.build_passive_candidates(p, q, y, run_wzzt=False, neighbor_seed_count=0)
    assert all(np.array_equal(a, b) for a, b in zip(old, (p, q, y), strict=True))


def test_wrong_prior_family_can_be_outranked_and_replaced():
    p, q, y, _ = _six_leaf_data()
    wrong = ((0, 1, 2), (3, 4, 5), (0, 2), (3, 4))
    result = bridge.build_passive_candidates(p, q, y, run_wzzt=False,
        prior_proposals=(wrong,), max_candidates=24)
    old = next(row for row in result["candidates"] if "prior_uncertified:0" in row["proposal_origins"])
    best = result["candidates"][0]
    assert old["replaceable"] and not old["certified_supports"]
    assert old["linear_contrast_mae"] > 1e-5
    assert best["linear_contrast_mae"] < 1e-10
    assert [0, 2] not in best["supports"]
    assert not result["rnj_supports_fixed_in_wzzt"]


@pytest.mark.parametrize("kwargs", [dict(source_p_error=.01), dict(kappa=-1),
    dict(max_candidates=0), dict(tolerance_factors=(-.1,)), dict(wzzt_time_limit=0)])
def test_invalid_assumptions_fail_before_solver(kwargs):
    p, q, y, _ = _six_leaf_data()
    with pytest.raises(ValueError):
        bridge.build_passive_candidates(p, q, y, run_wzzt=False, **kwargs)


def test_crossing_supports_rejected():
    with pytest.raises(ValueError, match="laminar"):
        bridge.normalize_family(((0, 1), (1, 2)), 6)
