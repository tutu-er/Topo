"""Independent finite differences, held-input checks and time-slicing contracts."""
from dataclasses import replace

import numpy as np
import pytest

from . import likelihood as lk
from .ev_fixed_train_fit import fit_fixed_inputs, subset_observations, subset_model, _FixedInputProblem


def _case(t=14, seed=120):
    c = lk.RadialCandidate.from_supports("pair01", 3, [(0, 1)])
    rng = np.random.default_rng(seed)
    r = np.linspace(.008, .026, c.m)
    x = np.array([.006, .009, .012, .004, .018])
    p = rng.uniform(.025, .17, (t, 3))
    q = rng.uniform(-.012, .085, (t, 3))
    sp, sq = rng.uniform(.015, .08, t), rng.uniform(-.02, .03, t)
    root = rng.uniform(.995, 1.006, t)
    meter = lk.MeasurementModel(.0006, .0004, .00012, .001, .001,
        bias_p=np.array([.001, -.0008, .0005]), bias_q=np.array([-.0003, .0006, .0002]),
        bias_v=np.array([.0002, -.0001, .0003]))
    phy = lk.ac_predict(c, r, x, p, q, sp, sq, root)
    obs = lk.Observations(p+meter.bias_p, q+meter.bias_q, phy["v"]+meter.bias_v, root,
        phy["master_p"], phy["master_q"])
    return c, r, x, sp, sq, obs, meter, phy


def test_training_recovers_lines_and_source_without_pq_states():
    c, r, x, sp, sq, obs, meter, phy = _case()
    fit = fit_fixed_inputs(c, obs, meter, multistart=2, max_nfev=250, seed=17)
    assert fit.success, fit.diagnostics
    assert fit.loss < 1e-10
    np.testing.assert_allclose(fit.r, r, atol=2e-7, rtol=0)
    np.testing.assert_allclose(fit.x, x, atol=2e-7, rtol=0)
    np.testing.assert_allclose(fit.source_p, sp, atol=2e-9, rtol=0)
    np.testing.assert_allclose(fit.source_q, sq, atol=2e-9, rtol=0)
    assert fit.diagnostics["joint_parameter_count"] == 2*c.m+2*obs.t
    assert fit.diagnostics["local_identifiability_full_column_rank"]
    assert not fit.diagnostics["global_optimality_certified"]
    assert len(fit.diagnostics["restarts"]) == 2
    assert "attained_state" in fit.diagnostics["restarts"][0]


def test_noisy_pq_are_exactly_held_and_output_normalizer_excludes_input_sigmas():
    c, r, x, sp, sq, obs, meter, _ = _case(t=4)
    observed = replace(obs, p=obs.p+.002, q=obs.q-.001)
    fit = fit_fixed_inputs(c, observed, meter, fixed_r=r, fixed_x=x, multistart=1)
    np.testing.assert_array_equal(fit.p, observed.p-np.asarray(meter.bias_p))
    np.testing.assert_array_equal(fit.q, observed.q-np.asarray(meter.bias_q))
    np.testing.assert_allclose(fit.predicted["p"], observed.p, atol=2e-17, rtol=0)
    assert fit.diagnostics["joint_parameter_count"] == 2*obs.t
    changed = replace(meter, p_sigma=1., q_sigma=2.)
    same = fit_fixed_inputs(c, observed, changed, fixed_r=r, fixed_x=x, multistart=1)
    assert fit.loss == same.loss and fit.loglik == same.loglik
    normalizer = obs.t*(3*np.log(2*np.pi*meter.v_sigma**2)
        + np.log(2*np.pi*meter.master_p_sigma**2)+np.log(2*np.pi*meter.master_q_sigma**2))
    assert fit.loglik == pytest.approx(-.5*(fit.loss+normalizer), abs=1e-10)


def test_analytic_training_and_fixed_line_jacobians_match_finite_differences():
    c, r, x, sp, sq, obs, meter, _ = _case(t=2)
    for fixed in (False, True):
        kw = dict(fixed_r=r, fixed_x=x) if fixed else {}
        problem = _FixedInputProblem(c, obs, meter,
            initial=dict(r=r, x=x, source_p=sp, source_q=sq), **kw)
        point = problem.start
        analytic = problem.jacobian(point)
        numerical = np.zeros_like(analytic)
        for j in range(len(point)):
            low, high = point.copy(), point.copy()
            low[j] -= 2e-6
            high[j] += 2e-6
            numerical[:, j] = (problem.evaluate(high)-problem.evaluate(low))/4e-6
        np.testing.assert_allclose(analytic, numerical, atol=3e-5, rtol=3e-6)
        # A source change in one slot has no effect on another slot's output.
        source_offset = 0 if fixed else 2*c.m
        assert np.count_nonzero(analytic[5:, source_offset:source_offset+2]) == 0


def test_validation_and_new_ev_data_keep_training_r_x_exactly():
    c, r, x, sp, sq, obs, meter, _ = _case(t=16)
    ids = np.arange(12)
    training = subset_observations(obs, ids)
    train_meter = subset_model(meter, obs, ids)
    trained = fit_fixed_inputs(c, training, train_meter, multistart=1)
    future = subset_observations(obs, np.arange(12, 16))
    fm = subset_model(meter, obs, np.arange(12, 16))
    # A future EV pulse changes total P once; matching V/root are generated once.
    p = future.p-fm.bias_p
    p[:, 2] += np.array([.015, -.01, .02, 0.])
    q = future.q-fm.bias_q
    phy = lk.ac_predict(c, r, x, p, q, sp[12:], sq[12:], future.root_v)
    ev = lk.Observations(p+fm.bias_p, q+fm.bias_q, phy["v"]+fm.bias_v,
        future.root_v, phy["master_p"], phy["master_q"])
    fit = fit_fixed_inputs(c, ev, fm, initial=trained, fixed_r=trained.r, fixed_x=trained.x, multistart=2)
    assert fit.success and fit.loss < 1e-10
    np.testing.assert_array_equal(fit.r, trained.r)
    np.testing.assert_array_equal(fit.x, trained.x)
    np.testing.assert_array_equal(fit.p, ev.p-fm.bias_p)
    assert fit.diagnostics["line_parameters_fixed"]
    assert fit.diagnostics["shared_parameter_count"] == 0
    assert fit.diagnostics["source_node"] == 1


def test_missing_root_joint_bias_and_partial_fixed_line_inputs_rejected():
    c, r, x, sp, sq, obs, meter, _ = _case(t=3)
    no_root = lk.Observations(obs.p, obs.q, obs.v, obs.root_v)
    with pytest.raises(ValueError, match="requires root"):
        fit_fixed_inputs(c, no_root, meter)
    with pytest.raises(ValueError, match="known calibrated"):
        fit_fixed_inputs(c, obs, replace(meter, fit_bias=True))
    with pytest.raises(ValueError, match="supplied together"):
        fit_fixed_inputs(c, obs, meter, fixed_r=r)
    with pytest.raises(ValueError, match="line bounds"):
        fit_fixed_inputs(c, obs, meter, fixed_r=r, fixed_x=x, r_bounds=(.04, .1))


def test_subset_broadcasts_on_original_times_then_slices_all_meter_fields():
    c, r, x, sp, sq, obs, meter, _ = _case(t=7)
    row = np.arange(7.)
    sigma_v = .0001+np.arange(21.).reshape(7, 3)*1e-6
    bp = np.arange(21.).reshape(7, 3)*1e-4
    model = replace(meter, v_sigma=sigma_v, master_p_sigma=.001+row*1e-5,
        master_q_sigma=.002+row*1e-5, p_sigma=np.array([.001, .002, .003]), bias_p=bp)
    ids = np.array([5, 1, 6])
    small = subset_observations(obs, ids)
    sm = subset_model(model, obs, ids)
    np.testing.assert_array_equal(small.master_q, obs.master_q[ids])
    np.testing.assert_array_equal(small.root_v, obs.root_v[ids])
    np.testing.assert_array_equal(sm.v_sigma, sigma_v[ids])
    np.testing.assert_array_equal(sm.master_p_sigma, np.asarray(model.master_p_sigma)[ids])
    np.testing.assert_array_equal(sm.p_sigma, np.broadcast_to(model.p_sigma, (7, 3))[ids])
    np.testing.assert_array_equal(sm.bias_p, bp[ids])
    np.testing.assert_array_equal(sm.bias_v, np.broadcast_to(model.bias_v, (7, 3))[ids])
    with pytest.raises(ValueError, match="at least one"):
        subset_observations(obs, [])


def test_failed_local_fit_retains_attainment_and_exact_inputs_and_counts():
    c, r, x, sp, sq, obs, meter, _ = _case(t=5)
    counter = {}
    fit = fit_fixed_inputs(c, obs, meter, multistart=2, max_nfev=1, call_counter=counter)
    assert not fit.success and np.isfinite(fit.loss)
    np.testing.assert_array_equal(fit.p, obs.p-meter.bias_p)
    assert fit.diagnostics["failure_kind"] == "computation"
    assert len(fit.diagnostics["restarts"]) == 2
    assert all(r["status"] == 0 and "attained_state" in r for r in fit.diagnostics["restarts"])
    assert counter["fit_calls"] == 1 and counter["solver_calls"] == 2
    assert counter["ac_calls"] == fit.diagnostics["ac_evaluations"]


def test_future_source_change_is_free_but_remains_at_single_interface():
    c, r, x, sp, sq, obs, meter, phy = _case(t=4)
    next_sp = sp + np.array([.06, -.025, .02, -.01])
    next_sq = sq + np.array([-.02, .035, -.01, .025])
    next_phy = lk.ac_predict(c, r, x, obs.p-meter.bias_p, obs.q-meter.bias_q,
        next_sp, next_sq, obs.root_v)
    changed = replace(obs, v=next_phy["v"]+meter.bias_v,
        master_p=next_phy["master_p"], master_q=next_phy["master_q"])
    fit = fit_fixed_inputs(c, changed, meter, fixed_r=r, fixed_x=x, multistart=1)
    assert fit.success and fit.loss < 1e-13
    np.testing.assert_allclose(fit.source_p, next_sp, atol=1e-10, rtol=0)
    np.testing.assert_allclose(fit.source_q, next_sq, atol=1e-10, rtol=0)
    assert fit.diagnostics["source_spatial_location_shared"]
