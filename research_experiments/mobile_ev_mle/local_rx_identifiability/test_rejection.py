"""Protocol, invariance and conservative failure tests for local rejection."""
from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from ..label_free_bounds.bounds import Measurements, PhysicalDomain, TreeSpec, certify_candidate
from . import rejection


def _inputs():
    tree = TreeSpec(0, ((0, 1), (1, 2), (1, 3), (1, 4)), (2, 3, 4))
    p = np.array([[.025, .03, .035], [.03, .025, .04], [.04, .03, .025]])
    q = .3 * p
    v = np.array([[.998, .997, .996], [.9975, .998, .9955], [.996, .997, .998]])
    domain = PhysicalDomain(.003, .025, .90, 1.10, [.06, .07, .08],
                            ratio_min=.2, ratio_max=2.)
    return tree, p, q, v, domain


def _call(tree, p, q, v, domain, **kwargs):
    return rejection.certify_observed_candidate(
        tree, p, q, v, .001, .001, .0005, domain, solver="scipy", **kwargs)


def test_api_has_no_truth_or_local_fit_loss_input():
    parameters = inspect.signature(rejection.certify_observed_candidate).parameters
    assert not any("truth" in name or "sse" in name or "fit" in name for name in parameters)


@pytest.mark.parametrize("root_v", [None, 1.])
def test_two_alphas_and_two_tests_remain_separate(root_v):
    result = _call(*_inputs(), root_v=root_v)
    assert result["status"] == "completed"
    assert result["combined_reject"] is None
    assert result["raw_noise_dimension"] == 27
    assert [row["alpha"] for row in result["alpha_results"]] == [.05, .01]
    for row in result["alpha_results"]:
        assert set(("raw", "residual")) <= row.keys()
        assert sum(row["residual"]["alpha_split"]) == pytest.approx(row["alpha"])
        assert row["residual"]["output_rank_df"] == (6 if root_v is None else 9)
    assert result["alpha_results"][1]["raw"]["threshold"] > result["alpha_results"][0]["raw"]["threshold"]


@pytest.mark.parametrize("root_v", [None, 1.])
def test_terminal_permutation_preserves_bounds(root_v):
    tree, p, q, v, domain = _inputs()
    order = [2, 0, 1]
    changed_tree = replace(tree, observed_nodes=tuple(tree.observed_nodes[j] for j in order))
    changed_domain = replace(domain, power_caps=np.asarray(domain.power_caps)[order])
    first = _call(tree, p, q, v, domain, root_v=root_v)
    second = _call(changed_tree, p[:, order], q[:, order], v[:, order], changed_domain, root_v=root_v)
    for left, right in zip(first["alpha_results"], second["alpha_results"]):
        for field in ("raw", "residual"):
            assert left[field]["reject"] == right[field]["reject"]
            assert left[field]["threshold"] == pytest.approx(right[field]["threshold"], rel=1e-10)
        assert left["raw"]["sse_lower"] == pytest.approx(right["raw"]["sse_lower"], abs=1e-7)
        assert left["residual"]["qp_lower"] == pytest.approx(right["residual"]["qp_lower"], abs=1e-12)


def test_unknown_root_projection_removes_reference_and_common_stem():
    tree, p, q, v, domain = _inputs()
    projected = replace(tree, source_node=1)
    obs = Measurements(p, q, v, np.array([.96, 1., 1.04]), .001, .001, .0005)
    left = certify_candidate(projected, obs, domain, source_projection=True, solver="scipy")
    # Alter root independently by time, and only the common-edge impedance cap.
    right = certify_candidate(
        projected, replace(obs, root_v=np.array([1.08, .92, .99])),
        replace(domain, z_max=[.25, .025, .025, .025]),
        source_projection=True, solver="scipy")
    for name in ("ac_sse_lower", "remainder_norm", "lipschitz_k",
                 "chi_square_threshold", "residual_chi_threshold"):
        # Bounds include solver/dual floating-point guards; the physical noise
        # and remainder envelopes are invariant to machine precision.
        tolerance = 1e-7 if name == "ac_sse_lower" else 1e-12
        assert getattr(left, name) == pytest.approx(getattr(right, name), abs=tolerance)
    assert left.qp_support_lower == pytest.approx(right.qp_support_lower, abs=1e-12)
    assert left.common_prefix_edges_removed == [0]


def test_zero_edge_sensitivity_is_kept_unsupported():
    tree, p, q, v, domain = _inputs()
    result = _call(tree, p, q, v, replace(domain, z_min=[.003, 0., .003, .003]))
    assert result["status"] == "unsupported_zero_impedance"
    assert len(result["alpha_results"]) == 2
    assert all(not row[field]["reject"] for row in result["alpha_results"] for field in ("raw", "residual"))


def test_numerical_failure_keeps_candidate_unrejected(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate solver failure")
    monkeypatch.setattr(rejection, "certify_candidate", fail)
    result = _call(*_inputs())
    assert result["status"] == "failed"
    assert result["failed_alpha_count"] == 2
    assert all("deliberate solver failure" in row["reason"] for row in result["alpha_results"])
    assert all(not row[field]["reject"] for row in result["alpha_results"] for field in ("raw", "residual"))


def test_large_primal_or_local_loss_cannot_cause_rejection(monkeypatch):
    def certificate(*args, **kwargs):
        result = SimpleNamespace(
            ac_sse_lower=0., chi_square_threshold=50.,
            qp_support_lower=0., residual_chi_threshold=.001,
            residual_chi_df=6, residual_chi_alpha_split=[kwargs["alpha"] / 3] * 3,
            qp_primal=1e30, reject=True, residual_chi_reject=True)
        result.as_dict = lambda: {"qp_primal": result.qp_primal}
        return result
    monkeypatch.setattr(rejection, "certify_candidate", certificate)
    result = _call(*_inputs())
    assert all(not row[field]["reject"] for row in result["alpha_results"] for field in ("raw", "residual"))


def test_bootstrap_failures_and_nonstationarity_are_conservative():
    records = [
        {"success": True, "stationary": True, "sse": 2.},
        {"success": True, "stationary": True, "sse": 5.},
        {"success": False, "stationary": True, "sse": 0.},
        {"success": True, "stationary": False, "sse": 0.},
        {"success": True, "stationary": True, "sse": np.nan},
        {"success": True, "sse": 0.},
    ]
    result = rejection.bootstrap_pvalue(5., records)
    assert result["pvalue"] == pytest.approx(6 / 7)
    assert result["valid_replicates"] == 2
    assert result["failed_or_nonstationary_replicates"] == 4
    assert result["upper_tail_count_including_failures"] == 5
    assert "approximate" in result["coverage"]


def test_bootstrap_empty_all_failed_and_resolution():
    assert rejection.bootstrap_pvalue(1., [])["pvalue"] == 1.
    assert rejection.bootstrap_pvalue(1., [{}, None])["pvalue"] == 1.
    result = rejection.bootstrap_pvalue(1., [{"success": True, "stationary": True, "sse": 0.}] * 39)
    assert result["pvalue"] == result["resolution"] == .025


def test_bootstrap_requires_boolean_success_and_stationarity():
    records = [{"success": "False", "stationary": True, "sse": 0.},
               {"success": True, "stationary": "False", "sse": 0.}]
    result = rejection.bootstrap_pvalue(1., records)
    assert result["pvalue"] == 1.
    assert result["failed_or_nonstationary_replicates"] == 2


@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_invalid_observed_bootstrap_loss_is_explicit(value):
    with pytest.raises(ValueError, match="observed_sse"):
        rejection.bootstrap_pvalue(value, [])
