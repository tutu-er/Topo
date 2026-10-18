"""Small, independent checks of period separation and retained failure semantics."""
from dataclasses import replace
import numpy as np
import pytest

from .. import likelihood as lk
from ..reference_benchmark.fitting import fit_rx_inputs as batched_fit
from .fixed_source import fit_fixed_sources, _SourceRow
from ..ev_fixed_train_fit import _FixedInputProblem


def fixture(periods=5, noisy=False, varying_sigma=False, biases=False):
    rng = np.random.default_rng(240018)
    candidate = lk.RadialCandidate.from_supports("source_pair_fixture", 3, [(0, 1)])
    r = np.linspace(.004, .008, candidate.m); x = .5*r
    p = rng.uniform(.02, .08, (periods, 3)); q = rng.uniform(.1, .5, p.shape)*p
    sp = rng.uniform(-.012, .012, periods); sq = rng.uniform(-.008, .008, periods)
    physical = lk.ac_predict(candidate, r, x, p, q, sp, sq)
    v_sigma = rng.uniform(.00007, .0002, (periods, 3)) if varying_sigma else .0001
    mp_sigma = rng.uniform(.00006, .0002, periods) if varying_sigma else .0001
    mq_sigma = rng.uniform(.00007, .0003, periods) if varying_sigma else .0001
    bp = np.array([.001, -.0008, .0006]) if biases else 0.
    bq = np.array([-.0005, .0002, .0001]) if biases else 0.
    bv = rng.uniform(-.0003, .0003, (periods, 3)) if biases else 0.
    voltage = physical["v"]+bv
    mp, mq = physical["master_p"].copy(), physical["master_q"].copy()
    if noisy:
        voltage = voltage+rng.normal(size=voltage.shape)*v_sigma
        mp += rng.normal(size=periods)*mp_sigma
        mq += rng.normal(size=periods)*mq_sigma
    data = lk.Observations(p+bp, q+bq, voltage, master_p=mp, master_q=mq)
    model = lk.MeasurementModel(.0001, .0001, v_sigma,
        master_p_sigma=mp_sigma, master_q_sigma=mq_sigma,
        bias_p=bp, bias_q=bq, bias_v=bv)
    options = dict(r_bounds=(0., .05), x_bounds=(0., .03), ratio_bounds=(.1, 1.),
        source_p_bounds=(-.1, .1), source_q_bounds=(-.1, .1),
        voltage_bounds=(.8, 1.2), multistart=1, max_nfev=100, tol=1e-10)
    state = dict(r=r, x=x, source_p=sp, source_q=sq)
    return candidate, data, model, options, state


def fit_fixture(candidate, data, model, options, state, **kwargs):
    return fit_fixed_sources(candidate, data, model,
        fixed_r=state["r"], fixed_x=state["x"], **dict(options, **kwargs))


def test_noiseless_periods_and_full_replay():
    c, data, meter, options, state = fixture()
    counter = {}
    fit = fit_fixture(c, data, meter, options, state, call_counter=counter)
    assert fit.success and fit.loss < 1e-12
    assert np.max(abs(fit.source_p-state["source_p"])) < 1e-9
    assert np.max(abs(fit.source_q-state["source_q"])) < 1e-9
    assert fit.diagnostics["source_periods_independent"]
    assert fit.diagnostics["full_history_replay"]["passed"]
    assert fit.diagnostics["ac_equation_residual"] < 2e-10
    assert fit.diagnostics["power_balance_residual"] < 2e-10
    assert counter["fit_calls"] == 1
    assert counter["solver_calls"] == data.t
    assert counter["ac_time_states"] == fit.diagnostics["ac_time_states"]
    assert counter["ac_time_states"] >= counter["ac_calls"]


@pytest.mark.parametrize("varying_sigma,biases", [(False, False), (True, False), (True, True)])
def test_noisy_same_objective_as_batched(varying_sigma, biases):
    c, data, meter, options, state = fixture(noisy=True,
        varying_sigma=varying_sigma, biases=biases)
    split = fit_fixture(c, data, meter, options, state)
    batch = batched_fit(c, data, meter, fixed_r=state["r"], fixed_x=state["x"], **options)
    assert split.success and batch.success
    assert abs(split.loss-batch.loss) < 2e-7
    assert np.max(abs(split.source_p-batch.source_p)) < 2e-8
    assert np.max(abs(split.source_q-batch.source_q)) < 2e-8
    assert abs(split.loglik+.5*(split.loss+split.diagnostics["output_normalizer"])) < 1e-10


def test_exact_two_column_row_jacobian_and_finite_difference():
    c, data, meter, options, state = fixture(varying_sigma=True, biases=True)
    args = {k: options[k] for k in ("r_bounds", "x_bounds", "source_p_bounds",
                                  "source_q_bounds", "voltage_bounds")}
    base = _FixedInputProblem(c, data, meter, fixed_r=state["r"], fixed_x=state["x"], **args)
    source = base.start.reshape(data.t, 2)
    full = base.jacobian(base.start)
    for time in range(data.t):
        row = _SourceRow(base, time)
        actual = row.jacobian(source[time])
        block = full[time*(data.n+2):(time+1)*(data.n+2), 2*time:2*time+2]
        assert actual.shape == (data.n+2, 2)
        assert np.max(abs(actual-block)) < 1e-10
        delta = np.array([.7, -.3]); h = 1e-7
        numeric = (row.evaluate(source[time]+h*delta)-row.evaluate(source[time]-h*delta))/(2*h)
        assert np.max(abs(numeric-actual@delta)) < 2e-5


@pytest.mark.parametrize("both_active", [False, True])
def test_active_source_bounds_match_batch_and_kkt(both_active):
    c, data, meter, options, state = fixture(periods=3)
    sp = np.full(data.t, .035)
    sq = np.full(data.t, -.03) if both_active else np.zeros(data.t)
    physical = lk.ac_predict(c, state["r"], state["x"], data.p, data.q, sp, sq)
    data = replace(data, v=physical["v"], master_p=physical["master_p"], master_q=physical["master_q"])
    options.update(source_p_bounds=(-.005, .005), source_q_bounds=(-.005, .005))
    split = fit_fixture(c, data, meter, options, state)
    batch = batched_fit(c, data, meter, fixed_r=state["r"], fixed_x=state["x"], **options)
    assert split.success and batch.success
    assert np.max(abs(split.source_p-.005)) < 1e-9
    if both_active:
        assert np.max(abs(split.source_q+.005)) < 1e-9
    assert abs(split.loss-batch.loss) < 1e-6
    assert split.diagnostics["scaled_projected_gradient_inf"] < 1e-4


def test_per_period_budget_failure_is_not_hidden_by_total_loss():
    c, data, meter, options, state = fixture(periods=4, noisy=True)
    initial = dict(source_p=np.full(data.t, .04), source_q=np.full(data.t, -.04))
    fit = fit_fixture(c, data, meter, options, state, initial=initial, max_nfev=1)
    assert not fit.success and fit.status == 0
    assert np.isfinite(fit.loss)
    assert fit.diagnostics["failed_period_count"] == data.t
    assert all(row["selected_status"] == 0 for row in fit.diagnostics["periods"])
    assert fit.nfev == data.t
    assert fit.diagnostics["nfev_selected_max"] == 1
    assert not fit.diagnostics["legacy_global_budget_equivalent"]


def test_one_period_budget_failure_preserves_other_converged_periods():
    c, data, meter, options, state = fixture(periods=3)
    source_p, source_q = state["source_p"].copy(), state["source_q"].copy()
    source_p[1] += .04
    source_q[1] -= .04
    initial = dict(source_p=source_p, source_q=source_q)
    fit = fit_fixture(c, data, meter, options, state, initial=initial, max_nfev=1)
    assert not fit.success and fit.status == 0
    assert fit.diagnostics["failed_periods"] == [1]
    assert fit.diagnostics["periods"][0]["success"]
    assert fit.diagnostics["periods"][2]["success"]
    assert fit.diagnostics["full_history_replay"]["passed"]


def test_multistart_records_every_period_and_is_reproducible():
    c, data, meter, options, state = fixture(periods=3, noisy=True)
    options.update(multistart=2)
    one = fit_fixture(c, data, meter, options, state, seed=17)
    two = fit_fixture(c, data, meter, options, state, seed=17)
    assert one.success and two.success
    assert one.loss == two.loss
    assert all(len(row["restarts"]) == 2 for row in one.diagnostics["periods"])
    assert one.diagnostics["solver_calls"] == 2*data.t


def test_physical_voltage_acceptance_uses_all_periods():
    c, data, meter, options, state = fixture(periods=3)
    fit = fit_fixture(c, data, meter, options, state,
                      voltage_bounds=(.99999, 1.00001))
    assert not fit.success and np.isfinite(fit.loss)
    assert not fit.diagnostics["physical_voltage_feasible"]
    assert fit.diagnostics["failed_period_count"] == data.t


def test_one_ac_failed_period_retains_other_periods(monkeypatch):
    c, data, meter, options, state = fixture(periods=3)
    original = lk.ac_predict
    def guarded(candidate, r, x, p, q, *args, **kwargs):
        if len(p) == 1 and np.array_equal(p[0], data.p[1]):
            raise RuntimeError("injected one-period AC failure")
        return original(candidate, r, x, p, q, *args, **kwargs)
    monkeypatch.setattr(lk, "ac_predict", guarded)
    fit = fit_fixture(c, data, meter, options, state)
    assert not fit.success and np.isinf(fit.loss)
    assert fit.diagnostics["failed_periods"] == [1]
    assert fit.diagnostics["periods"][0]["success"]
    assert fit.diagnostics["periods"][2]["success"]
    assert np.isnan(fit.source_p[1])
    assert not fit.diagnostics["full_history_replay"]["attempted"]


@pytest.mark.parametrize("key,value,match", [
    ("multistart", 0, "multistart"), ("max_nfev", 0, "max_nfev"), ("tol", 0., "tol")])
def test_invalid_budget_rejected(key, value, match):
    c, data, meter, options, state = fixture(periods=2)
    options[key] = value
    with pytest.raises(ValueError, match=match):
        fit_fixture(c, data, meter, options, state)


def test_outside_ratio_cone_and_free_bias_rejected():
    c, data, meter, options, state = fixture(periods=2)
    bad = dict(state, x=2*state["r"])
    with pytest.raises(ValueError, match="ratio cone"):
        fit_fixture(c, data, meter, options, bad)
    with pytest.raises(ValueError, match="known calibrated"):
        fit_fixture(c, data, replace(meter, fit_bias=True), options, state)
