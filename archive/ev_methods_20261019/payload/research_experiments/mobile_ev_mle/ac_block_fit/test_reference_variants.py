"""Actual short AC fits plus inspection of the temporary SciPy call options."""
import numpy as np
import pytest

from .. import likelihood as lk
from ..reference_benchmark import fitting
from ..reference_benchmark.test_fitting import fixture
from .reference_variants import fit_reference_jac


def noisy_fixture():
    candidate, observations, meter, options, state = fixture()
    rng = np.random.default_rng(81021)
    noisy = lk.Observations(observations.p, observations.q,
        observations.v+rng.normal(0, .0001, observations.v.shape),
        master_p=observations.master_p+rng.normal(0, .0001, observations.t),
        master_q=observations.master_q+rng.normal(0, .0001, observations.t))
    options.update(max_nfev=150, tol=1e-8)
    return candidate, noisy, meter, options, state


@pytest.mark.parametrize('fixed', [False, True])
def test_actual_jac_scale_same_complete_ac_objective_and_restoration(monkeypatch, fixed):
    candidate, data, meter, options, state = noisy_fixture()
    if fixed:
        options.update(fixed_r=state['r'], fixed_x=state['x'])
    reference = fitting.fit_rx_inputs(candidate, data, meter, initial=state, **options)
    original = fitting.least_squares
    calls = []

    def inspect_solver(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, 'least_squares', inspect_solver)
    result = fit_reference_jac(candidate, data, meter, initial=state, **options)
    assert fitting.least_squares is inspect_solver
    assert reference.success and result.success
    assert calls and all(call['x_scale'] == 'jac' for call in calls)
    assert all(call['method'] == 'trf' and call['tr_solver'] == 'lsmr' for call in calls)
    assert all(call['max_nfev'] == options['max_nfev'] for call in calls)
    expected_width = (0 if fixed else 2*candidate.m)+2*data.t
    for call in calls:
        assert call['tr_options'] == dict(atol=1e-10, btol=1e-10, maxiter=max(200, 4*expected_width))
        assert call['ftol'] == call['xtol'] == call['gtol'] == options['tol']
    # Replay through the original physical routine, independently of the hook.
    physical = lk.ac_predict(candidate, result.r, result.x, data.p, data.q,
                             result.source_p, result.source_q, data.root_v)
    residual = np.column_stack(((physical['v']-data.v)/meter.v_sigma,
        (physical['master_p']-data.master_p)/meter.master_p_sigma,
        (physical['master_q']-data.master_q)/meter.master_q_sigma)).ravel()
    assert result.loss == pytest.approx(float(residual@residual), abs=1e-9)
    assert result.loglik == pytest.approx(-.5*(result.loss+result.diagnostics['output_normalizer']))
    assert abs(reference.loss-result.loss) < 1e-5
    assert result.diagnostics['reference_scientific_objective_unchanged']
    assert result.diagnostics['inner_lsmr_tolerance_override'] is None
    assert result.diagnostics['global_optimality_certified'] is False


def test_lsmr_precision_override_keeps_maxiter_and_outer_options(monkeypatch):
    candidate, data, meter, options, state = noisy_fixture()
    original = fitting.least_squares
    calls = []

    def inspect_solver(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, 'least_squares', inspect_solver)
    result = fit_reference_jac(candidate, data, meter, initial=state,
                               inner_lsmr_tol=1e-8, **options)
    assert fitting.least_squares is inspect_solver
    assert calls
    for call in calls:
        assert call['tr_options']['atol'] == call['tr_options']['btol'] == 1e-8
        assert call['tr_options']['maxiter'] == max(200, 4*(2*candidate.m+2*data.t))
        assert call['max_nfev'] == options['max_nfev']
        assert call['gtol'] == options['tol']
    assert result.diagnostics['inner_lsmr_tolerance_override'] == 1e-8


def test_original_solver_restored_after_uncaught_exception(monkeypatch):
    class IntentionalAbort(BaseException):
        pass

    candidate, data, meter, options, state = noisy_fixture()

    def failing_solver(*args, **kwargs):
        assert kwargs['x_scale'] == 'jac'
        raise IntentionalAbort('independent abort')

    monkeypatch.setattr(fitting, 'least_squares', failing_solver)
    with pytest.raises(IntentionalAbort):
        fit_reference_jac(candidate, data, meter, initial=state, **options)
    assert fitting.least_squares is failing_solver
    # A second call proves that the active-process guard is also restored.
    with pytest.raises(IntentionalAbort):
        fit_reference_jac(candidate, data, meter, initial=state, **options)
    assert fitting.least_squares is failing_solver


def test_original_solver_restored_when_reference_retains_numerical_failure(monkeypatch):
    candidate, data, meter, options, state = noisy_fixture()

    def failed_solver(*args, **kwargs):
        raise ValueError('injected numerical failure')

    monkeypatch.setattr(fitting, 'least_squares', failed_solver)
    result = fit_reference_jac(candidate, data, meter, initial=state, **options)
    assert fitting.least_squares is failed_solver
    assert not result.success
    assert result.diagnostics['failure_kind'] == 'computation'


@pytest.mark.parametrize('tolerance', [0., -1., np.inf, np.nan])
def test_invalid_lsmr_precision_does_not_patch_global_solver(tolerance):
    original = fitting.least_squares
    with pytest.raises(ValueError, match='inner_lsmr_tol'):
        fit_reference_jac(inner_lsmr_tol=tolerance)
    assert fitting.least_squares is original
