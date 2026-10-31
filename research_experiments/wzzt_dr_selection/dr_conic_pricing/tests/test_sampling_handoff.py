"""Whole-response sampling, private output handoff and artifact checks."""
import importlib
import json

import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_conic_pricing.examples import synthetic_problem
from research_experiments.wzzt_dr_selection.dr_conic_pricing.identification import fit_full_history
from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem, evaluate_plan
from research_experiments.wzzt_dr_selection.dr_conic_pricing.sampling import (
    realize_factors, validate_frozen_plan,
)


def test_joint_realization_keeps_correlations_and_increment_only():
    p = synthetic_problem()
    x = np.array([.2, 0., 0., .15])
    z = np.array([[-1., 1., -.5], [.5, .2, 1.]])
    r = realize_factors(p, x, z)
    actual = x*(1+z@p.response_factors.T)
    assert np.array_equal(r["actual_amplitudes"], actual)
    assert np.allclose(r["net_load"]-r["background"], actual@p.profiles.T)
    assert np.allclose(r["benefits"], r["net_load"]@p.benefit_directions.T+p.benefit_offsets)
    assert np.allclose(r["payments"], actual@((x/p.elasticities)*p.reduction_hours))
    assert np.array_equal(r["actual_amplitudes"][:, 1:3], np.zeros((2, 2)))


def test_independent_complete_episode_validation_and_paired_baseline():
    p = synthetic_problem()
    x = [.2, 0., 0., .2]
    v = validate_frozen_plan(p, x, samples=10000, seed=29)
    assert v["successes"]+v["failures"] == 10000
    assert v["samples"] == v["paired_no_dr"]["samples"]
    assert v["budget_failures"] == v["device_failures"] == v["amplitude_failures"] == 0
    assert v["empirical_success_rate"] >= .9
    assert np.allclose(v["benefit_means"], v["analytic_benefit_means"], atol=.002)
    assert np.allclose(v["benefit_standard_deviations"],
                       v["analytic_benefit_standard_deviations"], atol=.001)
    assert abs(v["mean_payment"]-v["analytic_expected_payment"]) < .003
    assert v["raw_trajectory_exported"] is False


def test_incremental_realizations_cancel_background_even_with_shared_factors():
    d = synthetic_problem().to_dict()
    d["benefit_mode"] = "incremental"
    p = PriceProblem.from_dict(d)
    q = PriceProblem.from_dict(dict(d, mu0=np.full(16, -10.).tolist(),
                                   background_factors=np.full((16, 3), 3.).tolist()))
    z = np.random.default_rng(13).uniform(-1, 1, (40, 3))
    a = realize_factors(p, [.2, 0., 0., .15], z)
    b = realize_factors(q, [.2, 0., 0., .15], z)
    assert np.array_equal(a["benefits"], b["benefits"])
    assert not np.allclose(a["net_load"], b["net_load"])


def test_deterministic_threshold_not_lost_by_matrix_reassociation():
    d = synthetic_problem().to_dict()
    d["response_factors"] = np.zeros((4, 3)).tolist()
    d["background_factors"] = np.zeros((16, 3)).tolist()
    p = PriceProblem.from_dict(d)
    e = evaluate_plan(p, [.2, 0., 0., .2])
    v = validate_frozen_plan(p, [.2, 0., 0., .2], threshold=e["reliable_benefit"], samples=100)
    assert v["successes"] == 100


@pytest.mark.parametrize("kwargs", [dict(samples=0), dict(seed=-1), dict(batch_size=0), dict(threshold=float("nan"))])
def test_sampling_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        validate_frozen_plan(synthetic_problem(), [0., 0., 0., 0.], **kwargs)


def test_handoff_preserves_all_history_and_original_settings(monkeypatch, tmp_path):
    adapter = importlib.import_module("research_experiments.wzzt_dr_selection.dr_pool_v2.adapter")
    captured = {}

    def fake(training, validation, output_dir, **kwargs):
        captured.update(training=training, validation=validation, kwargs=kwargs)
        return {"status": "test_double_only"}

    monkeypatch.setattr(adapter, "fit_observed", fake)
    ht, hv, ft, fv = [object(), object()], [object()], [object()], [object(), object()]
    result = fit_full_history(ht, hv, ft, fv, tmp_path, root_label=1, search_seconds=25., lp_method=2)
    assert result["status"] == "test_double_only"
    assert captured["training"] == ht+ft
    assert captured["validation"] == hv+fv
    assert captured["kwargs"] == dict(root_label=1, search_seconds=25., lp_method=2, resume=False)
    assert len(ht) == 2 and len(hv) == 1


def test_cli_creates_frozen_input_and_does_not_overwrite(tmp_path):
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.cli import main
    problem_path, out = tmp_path/"input.json", tmp_path/"solve"
    assert main(["example", "--output", str(problem_path)]) == 0
    assert main(["solve", "--problem", str(problem_path), "--output", str(out),
                 "--mode", "socp", "--allowed", "0,3", "--samples", "50"]) == 0
    protocol = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen"
    assert protocol["validation_samples"] == 50
    assert protocol["source_sha256"]
    result = json.loads((out/"optimization.json").read_text(encoding="utf-8"))
    assert result["feasible_incumbent"]
    assert (out/"validation.json").exists()
    assert json.loads((out/"ended.json").read_text(encoding="utf-8"))["topology_identification_executed"] is False
    with pytest.raises(FileExistsError):
        main(["solve", "--problem", str(problem_path), "--output", str(out), "--mode", "misocp"])


@pytest.mark.parametrize("mode", ["socp", "misocp"])
def test_full_day_original_resource_to_solver_to_continuous_response(mode):
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.examples import resource_day_problem
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.solver import solve_price
    p = resource_day_problem()
    assert p.profiles.shape == (384, 4)
    result = solve_price(p, mode=mode, allowed_contracts=[0, 3] if mode == "socp" else None)
    assert result["feasible_incumbent"]
    v = validate_frozen_plan(p, result["amplitudes"], samples=500, seed=310)
    assert v["device_failures"] == v["budget_failures"] == 0
    assert v["samples"] == 500
