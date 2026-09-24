"""M7 contracts: units, truth isolation, uncertainty, and stage persistence."""

import numpy as np
import pandas as pd
import pytest

from theft_wzzt.theft.loss_models import amplitude_envelope, estimate_loss
from theft_wzzt.theft.theft_model import TheftTree, nominal_edge_weights, tree_from_network
from theft_wzzt.theft.theft_simulation import simulate_theft_scenarios


def test_l1_units_alignment_truth_isolation_and_clean_ac_agreement():
    tree = TheftTree(((0, 1), (1, 2), (1, 3)), (3, 2), (1, 2, 3))
    index = pd.Index([10, 20])
    measured = {
        "P_terminal": pd.DataFrame([[1., 2.], [1., 2.]], index=index, columns=[2, 3]),
        "Q_terminal": pd.DataFrame([[.5, .75], [.5, .75]], index=index, columns=[2, 3]),
        "root_voltage": pd.Series([1., 2.], index=index),
    }
    weights = ([.2, .4, .6], [.1, .2, .3])
    lp, lq = estimate_loss(measured, weights, tree, model="L1")
    # Hand calculation: 0.1*(3^2+1.25^2)+0.2*(1^2+.5^2)+0.3*(2^2+.75^2).
    np.testing.assert_allclose(lp, [2.675, 2.675 / 4])
    np.testing.assert_allclose(lq, lp / 2)
    assert lp.index.equals(index)
    with pytest.raises(ValueError, match="index"):
        estimate_loss(dict(measured, root_voltage=measured["root_voltage"].iloc[::-1]),
                      weights, tree, model="L1")
    net, scenario = simulate_theft_scenarios("paper15", pq_noise_rel=0, v_noise_rel=0,
                                            master_noise_rel=0, root_observation="exact")
    inputs = {k: scenario[k] for k in ("P_terminal", "Q_terminal", "root_voltage")}
    estimate, _ = estimate_loss(inputs, nominal_edge_weights(net), tree_from_network(net), model="L1")
    assert np.median(np.abs(estimate / scenario["loss_p_notheft"] - 1)) < .1


def test_loss_controls_identity_random_reproducibility_and_nonnegative_ratio():
    _, scenario = simulate_theft_scenarios("paper15")
    l0 = estimate_loss(scenario)
    l2 = estimate_loss(scenario, model="L2", bias=0)
    for a, b in zip(l0, l2):
        pd.testing.assert_series_equal(a, b)
    with pytest.raises(ValueError, match="bias"):
        estimate_loss(scenario, model="L2", bias=-1.01)
    for a, b in zip(estimate_loss(scenario, model="L3", sigma=4, seed=7),
                    estimate_loss(scenario, model="L3", sigma=4, seed=7)):
        pd.testing.assert_series_equal(a, b)
        assert (a >= 0).all()
    ratio = [float(v.sum() / scenario["P_terminal"].sum(axis=1).sum()) for v in l0]
    inputs = {k: scenario[k].copy() for k in ("P_terminal", "Q_terminal")}
    inputs["P_terminal"].iloc[0] *= -1
    for v in estimate_loss(inputs, model="L4", ratio=ratio):
        assert (v >= 0).all() and v.iloc[0] == 0
    with pytest.raises(ValueError, match="ratios"):
        estimate_loss(inputs, model="L4", ratio=[-1, 0])


def test_envelope_bounds_clipping_and_undefined_zero_relative_width():
    p = pd.DataFrame({1: [1., 1., 1.]})
    scenario = {"P_terminal": p, "P0_measured": pd.Series([2., 1.5, 1.])}
    loss = pd.Series([.5, .5, .5])
    env = amplitude_envelope(scenario, loss)
    np.testing.assert_allclose(env[["lower", "point", "upper"]],
                               [[.375, .5, .625], [0, 0, .125], [0, 0, 0]])
    assert env.relative_halfwidth.iloc[0] == pytest.approx(.25)
    assert env.relative_halfwidth.iloc[1:].isna().all()
    wide = amplitude_envelope(scenario, loss, beta_lo=-.5, beta_hi=.5)
    assert (wide.lower <= env.lower).all() and (wide.upper >= env.upper).all()
    with pytest.raises(ValueError):
        amplitude_envelope(scenario, loss, beta_lo=.1)
    with pytest.raises(ValueError):
        amplitude_envelope(scenario, -loss)


def test_m7_checkpoint_resume_failure_and_incomplete_report(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "m7_test_runner", Path(__file__).resolve().parents[1] / "experiments/run_theft.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    context = {"signature": "test", "fixed_q95": 5.454994777891962}
    monkeypatch.setattr(runner, "_m7_context", lambda: context)
    monkeypatch.setattr(runner, "_m7_specs", lambda stage: [])
    runner._save({"gain_threshold_q95": context["fixed_q95"]}, "summary")
    # Partial report must never declare success when expected stage rows are missing.
    original_specs = [("L0", 0., "null", 30)]
    monkeypatch.setattr(runner, "_m7_specs", lambda stage: original_specs if stage == "m7_l1" else [])
    summary = runner.run_m7_report()
    assert summary["status"] == "incomplete" and len(summary["missing_or_failed"]) == 1
    count = []
    def fake_detection(*args, **kwargs):
        count.append(kwargs)
        return {"gain": 1., "seconds_total": 0.}
    monkeypatch.setattr(runner, "run_detection", fake_detection)
    spec = original_specs[0]
    first = runner._m7_run_one(spec, context)
    assert runner._m7_run_one(spec, context) == first and len(count) == 1
    alias = runner._m7_run_one(("L2", 0., "null", 30), context)
    assert alias["gain"] == first["gain"] and "reused_from" in alias and len(count) == 1
    with pytest.raises(ValueError, match="Stale"):
        runner._m7_run_one(spec, dict(context, signature="changed"))
    def failure(*args, **kwargs):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(runner, "run_detection", failure)
    with pytest.raises(RuntimeError, match="injected"):
        runner._m7_run_one(("L1", 0., "null", 31), context)
    assert runner._load(runner._m7_name("L1", 0., "null", 31))["status"] == "failed"
    monkeypatch.setattr(runner, "run_detection", fake_detection)
    assert runner._m7_run_one(("L1", 0., "null", 31), context)["status"] == "ok"
