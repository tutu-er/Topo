"""Independent convex-step checks and short full-AC period-state-machine tests."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.optimize import lsq_linear

from .. import likelihood as lk
from ..reference_benchmark.fitting import fit_rx_inputs
from .fixed_source import fit_fixed_sources
from .batched_source import solve_source_boxes, fit_batched_sources
from .test_fixed_source import fixture


def run(candidate, data, meter, options, state, **kwargs):
    return fit_batched_sources(candidate, data, meter, fixed_r=state["r"], fixed_x=state["x"],
                               **dict(options, **kwargs))


@pytest.mark.parametrize("damping", [0., 1e-8, .2, 100.])
def test_two_coordinate_boxes_against_independent_dense_lsq(damping):
    rng = np.random.default_rng(1618033)
    jac = rng.normal(size=(40, 8, 2)); residual = rng.normal(size=(40, 8))
    lower = -rng.uniform(0., .5, (40, 2)); upper = rng.uniform(0., .5, (40, 2))
    result = solve_source_boxes(residual, jac, lower, upper, damping)
    assert np.all(result["success"])
    assert np.max(result["kkt"]) < 1e-10
    for time in range(40):
        a = np.vstack((jac[time], np.sqrt(damping)*np.eye(2)))
        b = np.r_[-residual[time], np.zeros(2)]
        reference = lsq_linear(a, b, bounds=(lower[time], upper[time]), method="bvls", tol=1e-12)
        assert reference.success
        assert abs(result["objective"][time]-.5*np.sum((a@reference.x-b)**2)) < 1e-10
        assert np.max(abs(result["step"][time]-reference.x)) < 1e-9


def test_rank_deficient_zero_column_and_fixed_step_boxes():
    jac = np.array([[[1., 1.], [2., 2.]], [[0., 0.], [0., 0.]], [[0., 1.], [0., 2.]]])
    residual = np.array([[2., 4.], [1., -1.], [2., 4.]])
    lower = np.array([[-.2, -.3], [0., -.1], [0., -.3]])
    upper = np.array([[.2, .3], [0., .1], [0., .3]])
    result = solve_source_boxes(residual, jac, lower, upper)
    assert np.all(result["rank_deficient"])
    assert np.all(result["success"])
    assert np.array_equal(result["step"][:, 0][1:], [0., 0.])
    assert np.max(result["kkt"]) < 1e-10


@pytest.mark.parametrize("varying,biases", [(False, False), (True, False), (True, True)])
def test_noisy_objective_agrees_with_both_original_paths(varying, biases):
    c, data, meter, options, state = fixture(periods=7, noisy=True, varying_sigma=varying, biases=biases)
    batched = run(c, data, meter, options, state)
    dense_rows = fit_fixed_sources(c, data, meter, fixed_r=state["r"], fixed_x=state["x"], **options)
    sparse = fit_rx_inputs(c, data, meter, fixed_r=state["r"], fixed_x=state["x"], **options)
    assert batched.success and dense_rows.success and sparse.success, batched.message
    assert abs(batched.loss-sparse.loss) < 2e-7
    assert abs(batched.loss-dense_rows.loss) < 2e-7
    assert np.max(abs(batched.source_p-sparse.source_p)) < 2e-8
    assert np.max(abs(batched.source_q-sparse.source_q)) < 2e-8
    assert batched.diagnostics["full_history_replay"]["passed"]
    assert batched.diagnostics["ac_equation_residual"] < 2e-10
    assert batched.diagnostics["power_balance_residual"] < 2e-10
    assert not batched.diagnostics["legacy_global_budget_equivalent"]


def test_noiseless_interior_and_independent_stops():
    c, data, meter, options, state = fixture(periods=4)
    p, q = state["source_p"].copy(), state["source_q"].copy()
    p[1] += .04; q[1] -= .04
    fit = run(c, data, meter, options, state,
              initial=dict(source_p=p, source_q=q), max_nfev=1)
    assert not fit.success and fit.status == 0
    assert fit.diagnostics["failed_periods"] == [1]
    assert all(fit.diagnostics["periods"][time]["status"] == 1 for time in [0, 2, 3])
    assert all(fit.diagnostics["periods"][time]["trace"] == [] for time in [0, 2, 3])
    assert np.isfinite(fit.loss)
    assert fit.diagnostics["full_history_replay"]["passed"]


def test_two_active_bounds_and_varying_source_limits():
    c, data, meter, options, state = fixture(periods=4, varying_sigma=True)
    sp, sq = np.full(data.t, .035), np.full(data.t, -.03)
    physical = lk.ac_predict(c, state["r"], state["x"], data.p, data.q, sp, sq)
    data = replace(data, v=physical["v"], master_p=physical["master_p"], master_q=physical["master_q"])
    hi = np.linspace(.003, .006, data.t); lo = -hi
    options.update(source_p_bounds=(lo, hi), source_q_bounds=(lo, hi))
    fit = run(c, data, meter, options, state)
    reference = fit_rx_inputs(c, data, meter, fixed_r=state["r"], fixed_x=state["x"], **options)
    assert fit.success and reference.success
    assert np.max(abs(fit.source_p-hi)) < 1e-10
    assert np.max(abs(fit.source_q-lo)) < 1e-10
    assert abs(fit.loss-reference.loss) < 1e-6
    assert fit.diagnostics["scaled_projected_gradient_inf"] < 1e-8


def test_counters_multistart_and_frozen_periods():
    c, data, meter, options, state = fixture(periods=6, noisy=True)
    counter = {}
    fit = run(c, data, meter, options, state, multistart=2, call_counter=counter, seed=42)
    assert fit.success
    assert counter["fit_calls"] == 1 and counter["solver_calls"] == 2
    assert counter["source_period_solver_calls"] == 2*data.t
    assert counter["ac_calls"] == fit.diagnostics["ac_evaluations"]
    assert counter["ac_time_states"] == fit.diagnostics["ac_time_states"]
    assert all(len(row["restarts"]) == 2 for row in fit.diagnostics["periods"])
    assert fit.diagnostics["damping_and_acceptance_per_period"]
    assert fit.diagnostics["converged_periods_frozen"]


def test_one_ac_failure_does_not_fail_other_periods(monkeypatch):
    c, data, meter, options, state = fixture(periods=3)
    original = lk.ac_predict
    def guarded(candidate, r, x, p, q, *args, **kwargs):
        if any(np.array_equal(row, data.p[1]) for row in p):
            raise RuntimeError("injected failed AC period")
        return original(candidate, r, x, p, q, *args, **kwargs)
    monkeypatch.setattr(lk, "ac_predict", guarded)
    fit = run(c, data, meter, options, state)
    assert not fit.success and np.isinf(fit.loss)
    assert fit.diagnostics["failed_periods"] == [1]
    assert fit.diagnostics["periods"][0]["success"]
    assert fit.diagnostics["periods"][2]["success"]
    assert not fit.diagnostics["full_history_replay"]["attempted"]


def test_full_physical_domain_is_checked():
    c, data, meter, options, state = fixture(periods=3)
    fit = run(c, data, meter, options, state, voltage_bounds=(.99999, 1.00001))
    assert not fit.success and np.isfinite(fit.loss)
    assert not fit.diagnostics["physical_voltage_feasible"]


@pytest.mark.parametrize("key,value", [("max_nfev", 0), ("multistart", 0), ("tol", 0.)])
def test_invalid_numerical_budget(key, value):
    c, data, meter, options, state = fixture(periods=2)
    with pytest.raises(ValueError, match=key):
        run(c, data, meter, options, state, **{key: value})
