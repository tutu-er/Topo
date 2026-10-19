"""Regression checks for actual AC evaluation defaults and saved fixed states."""
from types import SimpleNamespace

import numpy as np

from . import ev_mobile_v4_study as study
from .likelihood import Observations
from .simulation import model


def inputs():
    def observed(count):
        return Observations(p=np.full((count, 1), .1), q=np.full((count, 1), .03),
            v=np.ones((count, 1)), master_p=np.full(count, .1),
            master_q=np.full(count, .03), root_v=np.ones(count))
    D, future = observed(2), observed(1)
    reference = SimpleNamespace(r=np.array([.002]), x=np.array([.0005]),
        source_p=np.array([.10, .11]), source_q=np.array([.03, .04]))
    bank = [SimpleNamespace(name="a", n=1)]
    return bank, D, model(1), future, model(1), {"a": reference}


def patch_evaluation(monkeypatch, *, mutate_stage=False, fail_name=None):
    calls = []
    def source_fit(candidate, observations, meter, **kwargs):
        assert kwargs["voltage_bounds"] == (.85, 1.15)
        assert "r_bounds" in kwargs and "ratio_bounds" in kwargs
        calls.append(("source", candidate.name))
        if candidate.name == fail_name:
            raise ValueError("declared candidate failure")
        return SimpleNamespace(success=True, source_p=np.array([.12]), source_q=np.array([.05]))
    def fixed_score(candidate, observed, meter, state, options, lengths, labels):
        assert options["voltage_bounds"] == (.85, 1.15)
        assert lengths == (2, 1) and labels == ("observed", "future")
        assert observed.t == 3
        np.testing.assert_array_equal(state["source_p"], [.10, .11, .12])
        np.testing.assert_array_equal(state["source_q"], [.03, .04, .05])
        calls.append(("score", candidate.name))
        return dict(loss=2.)
    def joint_stage(candidate, observed, meter, options, witnesses, lengths, labels, seed, counter):
        assert options["voltage_bounds"] == (.85, 1.15)
        assert witnesses[0][0] == "D_reference_fixed_source"
        state = witnesses[0][1]
        if mutate_stage:
            state["source_p"][0] = 9.
            state["r"][0] = 8.
        calls.append(("stage", candidate.name))
        return dict(success=True, loss=1.), state
    monkeypatch.setattr(study, "fit_rx_inputs", source_fit)
    monkeypatch.setattr(study.refit, "_score", fixed_score)
    monkeypatch.setattr(study.refit, "_stage", joint_stage)
    return calls


def test_actual_evaluation_normalizes_missing_voltage_defaults_before_all_fits(monkeypatch):
    args = inputs()
    calls = patch_evaluation(monkeypatch)
    options = dict(multistart=1, max_nfev=7)
    result = study.evaluate_future(*args, options, seed=13)
    assert result["complete"]
    assert result["failures"] == {}
    assert calls == [("source", "a"), ("score", "a"), ("stage", "a")]
    assert options == dict(multistart=1, max_nfev=7)
    fixed = result["comparisons"]["a"]["fixed"]
    np.testing.assert_array_equal(fixed["state"]["source_p"], [.10, .11, .12])
    np.testing.assert_array_equal(fixed["state"]["source_q"], [.03, .04, .05])
    np.testing.assert_array_equal(fixed["state"]["r"], [.002])
    np.testing.assert_array_equal(fixed["state"]["x"], [.0005])


def test_fixed_state_saved_as_copy_before_joint_stage(monkeypatch):
    args = inputs()
    patch_evaluation(monkeypatch, mutate_stage=True)
    result = study.evaluate_future(*args, {}, seed=13)
    assert result["complete"]
    saved = result["comparisons"]["a"]["fixed"]["state"]
    np.testing.assert_array_equal(saved["source_p"], [.10, .11, .12])
    np.testing.assert_array_equal(saved["r"], [.002])
    reference = args[-1]["a"]
    np.testing.assert_array_equal(reference.source_p, [.10, .11])
    np.testing.assert_array_equal(reference.r, [.002])


def test_actual_candidate_failure_remains_explicit_and_prevents_complete_flag(monkeypatch):
    bank, D, dm, future, fm, fits = inputs()
    bank.append(SimpleNamespace(name="b", n=1))
    fits["b"] = fits["a"]
    calls = patch_evaluation(monkeypatch, fail_name="b")
    result = study.evaluate_future(bank, D, dm, future, fm, fits, {}, seed=13)
    assert not result["complete"]
    assert set(result["comparisons"]) == {"a"}
    assert result["failures"] == {"b": "ValueError: declared candidate failure"}
    assert calls[-1] == ("source", "b")
