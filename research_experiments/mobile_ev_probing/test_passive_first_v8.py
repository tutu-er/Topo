"""Lifecycle/authorization-boundary tests for the linear passive-first study."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import passive_first_v8 as p


def history(mode="flat", topology="cherry12", profile="symmetric"):
    obs, _ = p.simulate(topology, p.d.PROFILES[profile], 8190, mode)
    return p.d.observation_slice(obs, slice(0, 12))


def test_rich_passive_data_resolve_each_topology_and_do_not_design_ev():
    for topology in p.d.NAMES:
        result = p.plan(history("rich", topology))
        assert result["state"] == "singleton_verified_skip_ev"
        assert result["screening"]["retained"] == [topology]
        assert result["selected_action"] is None and "design" not in result


def test_failed_solver_and_unverified_status_never_exclude():
    def failed(*args, **kwargs):
        return dict(status="excluded", solver_success=False, reliable_lower_bound=.1)
    result = p.screen_history(history(), solver=failed)
    assert result["retained"] == list(p.d.NAMES)
    assert all(r["classification"] == "unresolved" for r in result["records"].values())


def test_singleton_with_missing_feasible_witness_is_not_called_verified(monkeypatch):
    def solver(obs, h, **kwargs):
        if h == "cherry12":
            return dict(status="failure", reliable_lower_bound=None)
        return dict(reliable_lower_bound=.01, exact_box_corrected_dual_verified=True)
    monkeypatch.setattr(p.d, "solve_qp", lambda *args, **kwargs: dict(status="failure"))
    result = p.plan(history(), solver=solver)
    assert result["state"] == "singleton_unverified_numerically_unresolved"
    assert result["selected_action"] is None


def test_inconsistent_history_is_flagged_and_does_not_probe():
    obs = history()
    obs["Y_drop"] += .5
    result = p.plan(obs)
    assert result["state"] == "history_inconsistent_with_declared_family"
    assert result["screening"]["retained"] == [] and result["selected_action"] is None


def test_reduced_candidate_design_never_reintroduces_excluded_candidates():
    retained = ("cherry12", "cherry13")
    result = p.design_retained(history(), retained)
    assert result["state"] == "probe_selected"
    assert set(result["representatives"]) == set(retained)
    assert len(result["qp_records"]) == 6
    assert all(r["candidate"] in retained and r["reference"] in retained for r in result["qp_records"])


def test_action_pairing_is_exact_and_truth_does_not_enter_observations():
    outputs = [p.simulate("cherry13", p.d.PROFILES["asymmetric"], 8191, "single_node", 1, pol)
               for pol in ("none", "constant", "endpoint")]
    for obs, truth in outputs:
        assert "topology" not in obs and "theta" not in obs
        assert all(truth["checks"].values())
        np.testing.assert_array_equal(obs["Y_drop"][:12], outputs[0][0]["Y_drop"][:12])
        assert truth["source_delta"] == outputs[0][1]["source_delta"]
        assert truth["noise_draw"] == outputs[0][1]["noise_draw"]
    assert outputs[1][1]["energy_kwh"] == outputs[2][1]["energy_kwh"]


def test_static_history_needs_probing_but_weak_edge_remains_unresolved():
    obs = history()
    assert len(p.screen_history(obs)["retained"]) == 3
    result = p.plan(obs)
    assert result["state"] == "probe_selected"
    site = int(result["selected_action"][4]) - 1
    for profile, expect in (("symmetric", 2), ("weak", 0)):
        data, _ = p.simulate("cherry12", p.d.PROFILES[profile], 8190, "flat", site, "endpoint")
        wrong = [p.v6.solve_lp(data, h, parameter_box=p.d.BOX) for h in ("cherry13", "cherry23")]
        assert sum(p.classify_fit(r) == "excluded" for r in wrong) == expect


def test_weak_profile_is_covered_by_every_rival_at_all_admissible_vertices():
    # Direct matrix calculation independent of fitting. Each residual
    # coordinate is affine in u; its absolute maximum is at u=0 or u=.3.
    matrices = {}
    for h, pair in p.v6.CANDIDATES.items():
        z = np.zeros(3)
        z[list(pair)] = 1
        matrices[h] = 1e-6 * np.outer(z, z)
    for h in matrices:
        for g in matrices:
            for site in range(3):
                for u in (0., .3):
                    power = np.full(3, .2)
                    power[site] += u
                    separation = np.max(abs((matrices[h] - matrices[g]) @ power))
                    assert separation <= 7e-7 + 1e-20
                    assert separation + 8e-6 + 1e-16 < 1e-5
