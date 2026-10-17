"""Simulator/planner separation and unit tests, using no fitted candidates.

Artificial raw-import windows make each causal dependency explicit. Forward
AC solves only verify data generation; no optimization/fitting is invoked.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json

import numpy as np
import pandas as pd
import pytest

from .. import likelihood as lk
from . import cases


CHANNELS = ("p", "q", "v", "master_p", "master_q")
ALLOWLIST = {"history", "history_meter", "future_meter", "timestamps", "horizon", "base_kw"}


def synthetic_window(H=8, T=3, n=8):
    rows = np.arange(H + T, dtype=float)[:, None]
    columns = np.arange(n, dtype=float)[None, :]
    raw_kw = .8 + .04 * rows + .11 * columns + .03 * np.sin(.7 * rows + columns)
    stamps = pd.date_range("2016-06-01T10:00Z", periods=H + T, freq="15min")
    return dict(history=dict(p_kw=raw_kw[:H].copy(), timestamps_utc=[t.isoformat() for t in stamps[:H]]),
                future=dict(p_kw=raw_kw[H:].copy(), timestamps_utc=[t.isoformat() for t in stamps[H:]]),
                provenance=dict(raw_sha256="artificial_unit_fixture", normalization="none", columns=list(range(n))))


def specification(n=6, mode="none", kind=0):
    return dict(n=n, source_mode=mode, cluster_index=kind, seed=724,
                physical_seed=915)


def configuration(H=8, T=3):
    return dict(H=H, T=T, base_kw=20., total_precision_per_channel=30.)


def make(n=6, mode="none", kind=0, window=None):
    return cases.make_case(specification(n, mode, kind), synthetic_window() if window is None else window,
                           configuration())


def comparable(value):
    if is_dataclass(value):
        return comparable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [comparable(item) for item in value]
    return value


def snapshot(value):
    return json.dumps(comparable(value), sort_keys=True, allow_nan=False)


def history_physical(case):
    H, truth = case["H"], case["truth"]
    return lk.ac_predict(truth["candidate"], truth["r"], truth["x"],
        case["p"][:H], case["q"][:H], case["source_p"][:H], case["source_q"][:H], case["root_v"][:H])


def standardized_noise(case):
    physical, measured, meter = history_physical(case), case["history"], case["history_meter"]
    result = {}
    for channel in CHANNELS:
        bias = getattr(meter, "bias_" + channel, 0.)
        result[channel] = (getattr(measured, channel) - physical[channel] - bias) / getattr(meter, channel + "_sigma")
    return result


def future_white(T, n):
    rng = np.random.default_rng(1881)
    return {key: rng.standard_normal((T, n) if key in ("p", "q", "v") else T) for key in CHANNELS}


def test_scaling_uses_only_history_and_retains_declared_factors():
    original = synthetic_window()
    changed = deepcopy(original)
    changed["future"]["p_kw"] *= 1.4
    first, second = make(window=original), make(window=changed)
    scale = first["provenance"]["scaling"]
    expected_mean = np.mean(original["history"]["p_kw"][:, :6], axis=0)
    expected_factor = 2. / np.maximum(expected_mean, .1)
    np.testing.assert_allclose(scale["history_mean_kw"], expected_mean)
    np.testing.assert_allclose(scale["factors"], expected_factor)
    assert scale == second["provenance"]["scaling"]
    np.testing.assert_array_equal(first["p"][:first["H"]], second["p"][:second["H"]])
    np.testing.assert_array_equal(first["q"][:first["H"]], second["q"][:second["H"]])
    assert snapshot(first["history"]) == snapshot(second["history"])
    assert not np.allclose(first["p"][first["H"]:], second["p"][second["H"]:])
    assert scale["future_used"] is False


def test_small_history_mean_has_fixed_physical_floor():
    window = synthetic_window()
    window["history"]["p_kw"][:, 0] = .05
    case = make(window=window)
    assert case["provenance"]["scaling"]["factors"][0] == 20.
    np.testing.assert_allclose(case["p"][:case["H"], 0] * case["base_kw"], 1.)


def test_planning_inputs_exact_allowlist_and_counterfactual_truth_isolation():
    case = make(mode="DER")
    inputs = cases.planning_inputs(case)
    assert set(inputs) == ALLOWLIST
    assert inputs["history"] is case["history"]
    assert inputs["history"].source_p is None and inputs["history"].source_q is None
    before = snapshot(inputs)
    H = case["H"]
    case["p"][H:] += .7
    case["q"][H:] *= 4
    case["source_p"][H:] -= .4
    case["source_q"][H:] += .3
    case["truth"]["r"] *= 1.2
    case["truth"]["x"] *= .9
    case["truth"]["candidate"] = lk.RadialCandidate.from_supports("changed_truth", 6, ())
    case["spec"]["source_mode"] = "unpassed_counterfactual"
    case["provenance"]["secret_future"] = "different outcome"
    assert before == snapshot(cases.planning_inputs(case))
    assert set(cases.planning_inputs(case)) == ALLOWLIST


@pytest.mark.parametrize("n", [6, 8])
def test_source_conditions_are_physically_paired_and_share_meter_noise(n):
    none, der = make(n=n), make(n=n, mode="DER")
    assert none["truth"]["candidate"].supports == der["truth"]["candidate"].supports
    for key in ("r", "x"):
        np.testing.assert_array_equal(none["truth"][key], der["truth"][key])
    for key in ("p", "q", "root_v"):
        np.testing.assert_array_equal(none[key], der[key])
    assert snapshot(none["history_meter"]) == snapshot(der["history_meter"])
    assert snapshot(none["future_meter"]) == snapshot(der["future_meter"])
    assert np.all(none["source_p"] == 0) and np.all(none["source_q"] == 0)
    assert np.any(der["source_p"] < 0) and np.any(der["source_q"] != 0)
    left, right = standardized_noise(none), standardized_noise(der)
    for channel in CHANNELS:
        np.testing.assert_allclose(left[channel], right[channel], atol=1e-10, rtol=1e-10)
    # The caller-supplied future noise is shared without sharing the actual
    # source-dependent physical response.
    white = future_white(der["T"], n)
    none_future, none_response, _ = cases.actual_observation(none, np.zeros((none["T"], n)), white)
    der_future, der_response, _ = cases.actual_observation(der, np.zeros((der["T"], n)), white)
    for channel in CHANNELS:
        meter = der["future_meter"]
        bias = getattr(meter, "bias_" + channel, 0.)
        sigma = getattr(meter, channel + "_sigma")
        a = (getattr(none_future, channel) - none_response[channel] - bias) / sigma
        b = (getattr(der_future, channel) - der_response[channel] - bias) / sigma
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
    physical = history_physical(der)
    candidate = der["truth"]["candidate"]
    assert len(candidate.supports[0]) == n
    np.testing.assert_allclose(physical["loads"][:, 0],
        der["source_p"][:der["H"]] + 1j * der["source_q"][:der["H"]])
    assert physical["ac_equation_residual"] < 1e-10
    assert physical["power_balance_residual"] < 1e-10


@pytest.mark.parametrize("n", [6, 8])
def test_ev_is_added_once_to_P_in_fixed_units_and_never_added_to_Q(n):
    case = make(n=n, mode="DER")
    H, T, base = case["H"], case["T"], case["base_kw"]
    power = np.zeros((T, n)); power[:, 1] = [3., -1.5, .75]
    white = future_white(T, n)
    obs, physical, actual_background = cases.actual_observation(case, power, white)
    expected_p = case["p"][H:H + T] + power / base
    np.testing.assert_allclose(physical["p"], expected_p)
    np.testing.assert_array_equal(physical["q"], case["q"][H:H + T])
    np.testing.assert_array_equal(actual_background["p"], case["p"][H:H + T])
    assert case["provenance"]["EV_Q"] == 0.
    meter = case["future_meter"]
    reconstructed = obs.p - meter.bias_p - np.asarray(meter.p_sigma) * white["p"]
    np.testing.assert_allclose(reconstructed, expected_p)
    np.testing.assert_allclose((reconstructed - case["p"][H:H + T]) * base, power, atol=1e-13)
    independent = lk.ac_predict(case["truth"]["candidate"], case["truth"]["r"], case["truth"]["x"],
        expected_p, case["q"][H:H + T], case["source_p"][H:H + T],
        case["source_q"][H:H + T], case["root_v"][H:H + T])
    for channel in CHANNELS:
        np.testing.assert_allclose(physical[channel], independent[channel], atol=1e-13)


def test_clean_removes_only_future_random_noise_and_preserves_bias_and_history():
    case = make(mode="DER")
    before = snapshot(case["history"])
    T, n = case["T"], case["history"].n
    power = np.zeros((T, n)); power[:, 0] = [2., 0., -1.]
    white = future_white(T, n)
    noisy, physical, _ = cases.actual_observation(case, power, white)
    clean, clean_physical, _ = cases.actual_observation(case, power, white, clean=True)
    meter = case["future_meter"]
    for channel in CHANNELS:
        np.testing.assert_array_equal(physical[channel], clean_physical[channel])
        np.testing.assert_allclose(getattr(noisy, channel) - getattr(clean, channel),
            np.asarray(getattr(meter, channel + "_sigma")) * white[channel], atol=1e-13)
        np.testing.assert_allclose(getattr(clean, channel),
            physical[channel] + getattr(meter, "bias_" + channel, 0.), atol=1e-13)
    assert before == snapshot(case["history"])
    assert np.any(np.abs(standardized_noise(case)["v"]) > 1e-5)


@pytest.mark.parametrize("n", [6, 8])
def test_four_registered_families_are_distinct_complete_valid_radial_trees(n):
    families = [cases.clades(n, kind, seed=16) for kind in range(4)]
    assert len(set(families)) == 4
    for kind, family in enumerate(families):
        sets = [set(value) for value in family]
        assert all(1 < len(value) < n for value in family)
        for i, left in enumerate(sets):
            for right in sets[i + 1:]:
                assert left <= right or right <= left or left.isdisjoint(right)
        candidate = lk.RadialCandidate.from_supports(f"family_{kind}", n, family)
        assert tuple(range(n)) in candidate.supports
        assert all((j,) in candidate.supports for j in range(n))
        assert len(candidate.terminal_edges) == n and len(set(candidate.terminal_edges)) == n
        assert candidate.B.shape == (candidate.m, candidate.m)
        assert np.linalg.matrix_rank(candidate.B) == candidate.m
