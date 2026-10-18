"""Physical conservation, input boundaries and frozen local-case checks."""
from copy import deepcopy

import numpy as np
import pytest

from .. import likelihood as lk
from ..reference_benchmark.networks import build_network
from ..reference_benchmark.representation import physical_tree_ac
from .cases import make_case, candidate_bank, public_domain


@pytest.mark.parametrize("network", ("paper11", "paper15"))
@pytest.mark.parametrize("regime", ("ordinary", "correlated"))
def test_native_subtree_and_independent_physical_replay(network, regime):
    c = make_case(network, regime, 31001)
    truth = c["truth"]
    physical = truth["physical_tree"]
    assert len(physical["branches"]) == 4
    assert len(physical["buses"]) == 5
    assert truth["candidate_name"] == "star"
    native_edges = {(e["from_bus"], e["to_bus"]): e
                    for e in build_network(network)["branches"]}
    for e in physical["branches"]:
        assert e == native_edges[e["from_bus"], e["to_bus"]]
    base = physical["base_kw"]
    replay = physical_tree_ac(
        physical["branches"], physical["observation_bus_ids"], physical["root_bus"],
        physical["base_kv"], base, truth["p"] * base, truth["q"] * base,
        physical["source_bus"], root_v=truth["root_v"])
    assert np.max(abs(replay["v"] - truth["v"])) < 2e-11
    assert replay["ac_equation_residual"] < 2e-11
    assert replay["power_balance_residual"] < 2e-11
    assert c["provenance"]["physical_parameter_source"]["new_impedance_multiplier"] == 1.
    assert not c["provenance"]["original_paper_parameters"]


def test_candidate_refinements_are_all_preserved_and_do_not_read_truth():
    candidates = candidate_bank()
    assert [c.name for c in candidates] == ["star", "pair01", "pair02", "pair12"]
    assert candidates[0].m == 4
    assert [c.m for c in candidates[1:]] == [5, 5, 5]
    for c, pair in zip(candidates[1:], ((0, 1), (0, 2), (1, 2))):
        assert pair in c.supports
    assert all(c.n == 3 for c in candidates)


@pytest.mark.parametrize("network", ("paper11", "paper15"))
def test_boundary_controls_are_exactly_paired_and_unknown_boundary_has_no_root(network):
    u = make_case(network, "ordinary", 31002, boundary="U")
    k = make_case(network, "ordinary", 31002, boundary="K")
    assert set(u["observed"]) == {"p", "q", "v"}
    assert set(k["observed"]) == {"p", "q", "v", "root_known"}
    for key in ("p", "q", "v"):
        np.testing.assert_array_equal(u["observed"][key], k["observed"][key])
    np.testing.assert_array_equal(k["observed"]["root_known"], k["truth"]["root_v"])
    assert not u["provenance"]["original_cut_bus_metered"]
    assert k["provenance"]["known_boundary_is_additional_ideal_calibrated_sensor"]
    assert np.ptp(k["truth"]["root_v"]) > .01


@pytest.mark.parametrize("regime", ("ordinary", "correlated"))
def test_history_and_holdout_are_prefix_reproducible(regime):
    short = make_case("paper15", regime, 31003, total=96, boundary="K")
    long = make_case("paper15", regime, 31003, total=384, boundary="K")
    for section, keys in (("observed", ("p", "q", "v", "root_known")),
                          ("truth", ("p", "q", "v", "root_v"))):
        for key in keys:
            np.testing.assert_array_equal(short[section][key], long[section][key][:96])
    assert long["provenance"]["holdout_start"] == 288
    assert long["provenance"]["holdout_stop"] == 384


@pytest.mark.parametrize("network", ("paper11", "paper15"))
def test_declared_whole_library_domain_covers_assets_and_local_physics(network):
    c = make_case(network, "ordinary", 31001)
    d, truth = c["domain"], c["truth"]
    zbase = c["nominal"]["zbase_ohm"]
    all_z = [complex(e["r_ohm"], e["x_ohm"])
             for name in ("paper11", "paper15")
             for e in build_network(name)["branches"]]
    assert d["library"]["asset_count"] == 36
    assert d["z_min"] == pytest.approx(min(map(abs, all_z)) / zbase)
    assert d["z_max"] == pytest.approx(2. * max(map(abs, all_z)) / zbase)
    assert d["ratio_bounds"] == (.25, 1.5)
    for key in ("r", "x"):
        assert np.all(truth[key] >= d[key + "_bounds"][0])
        assert np.all(truth[key] <= d[key + "_bounds"][1])
    assert np.all(np.hypot(truth["r"], truth["x"]) >= d["z_min"])
    assert np.all(np.hypot(truth["r"], truth["x"]) <= d["z_max"])
    for key in ("p", "q"):
        assert np.all(truth[key] > d[key + "_bounds"][0])
        assert np.all(truth[key] < d[key + "_bounds"][1])
    assert np.all(np.hypot(truth["p"], truth["q"]) < d["power_caps"])
    assert np.min(truth["v"]) > d["v_bounds"][0]
    assert np.max(truth["root_v"]) < d["v_bounds"][1]


def test_public_domain_is_identical_without_local_selection():
    a = make_case("paper11", "ordinary", 31001)["domain"]
    b = make_case("paper15", "correlated", 31003)["domain"]
    for key in ("r_bounds", "x_bounds", "z_min", "z_max", "ratio_bounds", "library"):
        assert a[key] == b[key]
    assert public_domain()["z_min"] == a["z_min"]


def test_noise_is_present_in_every_meter_and_sigma_is_nominal_not_truth_scaled():
    c = make_case("paper11", "ordinary", 31001)
    for key in ("p", "q", "v"):
        sigma = c["meter"]["sigma_" + key]
        assert sigma.shape == (3,)
        error = (c["observed"][key] - c["truth"][key]) / sigma
        assert .85 < np.std(error) < 1.15
        assert abs(np.mean(error)) < .12
    np.testing.assert_array_equal(c["meter"]["sigma_p"], .01 * c["nominal"]["p"])
    np.testing.assert_array_equal(c["meter"]["sigma_q"],
                                  .01 * np.maximum(c["nominal"]["q"], .2 * c["nominal"]["p"]))


def test_pv_is_added_once_and_q_is_not_a_constant_multiple_of_p():
    for regime in ("ordinary", "correlated"):
        c = make_case("paper15", regime, 31001)
        t = c["truth"]
        np.testing.assert_allclose(t["p"] + t["pv_p"], t["demand_p"], rtol=0, atol=2e-17)
        assert np.any(t["pv_p"][:, 0] > 0)
        assert np.count_nonzero(t["pv_p"][:, 1:]) == 0
        assert np.min(np.std(t["q"] / t["p"], axis=0)) > .001
    a = make_case("paper15", "ordinary", 31001)["truth"]["p"]
    b = make_case("paper15", "correlated", 31001)["truth"]["p"]
    off_diag = np.triu_indices(3, 1)
    assert np.mean(np.corrcoef(a.T)[off_diag]) < np.mean(np.corrcoef(b.T)[off_diag])


def test_truth_mutation_cannot_modify_observed_data():
    c = make_case("paper11", "ordinary", 31001, boundary="K")
    saved = deepcopy(c["observed"])
    for key in ("p", "q", "v", "root_v", "r", "x"):
        c["truth"][key][:] = -999.
    for key in saved:
        np.testing.assert_array_equal(saved[key], c["observed"][key])


@pytest.mark.parametrize("kwargs", [dict(network="case33"), dict(regime="flat"),
                                   dict(seed=-1), dict(seed=True),
                                   dict(total=0), dict(total=1.2), dict(boundary="X")])
def test_invalid_configuration_fails(kwargs):
    values = dict(network="paper11", regime="ordinary", seed=31001)
    values.update(kwargs)
    with pytest.raises(ValueError):
        make_case(**values)


def test_alias_and_fresh_objects():
    a = make_case("soumalas11", "ordinary", 31001, total=1)
    b = make_case("paper11", "ordinary", 31001, total=1)
    np.testing.assert_array_equal(a["observed"]["p"], b["observed"]["p"])
    a["domain"]["library"]["asset_count"] = -1
    assert b["domain"]["library"]["asset_count"] == 36
    p = b["truth"]
    replay = lk.ac_predict(b["candidates"][0], p["r"], p["x"], p["p"], p["q"], root_v=p["root_v"])
    np.testing.assert_array_equal(replay["v"], p["v"])
