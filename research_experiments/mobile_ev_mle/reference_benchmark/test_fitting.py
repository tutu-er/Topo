"""Same-objective sparse/dense comparisons and zero-stub initialization checks."""
import numpy as np
from scipy.sparse import issparse
import pytest

from .. import likelihood as lk
from ..ev_rx_fit import _RxInputProblem, fit_rx_inputs as dense_fit
from .fitting import fit_rx_inputs, history_linear_initial, _sparse_jacobian, _SourceInputProblem
from .representation import encode_physical_tree


def fixture(zero_stub=False):
    rng = np.random.default_rng(812)
    if zero_stub:
        branches = [dict(from_bus=0, to_bus=1, r_ohm=.01, x_ohm=.005),
                    dict(from_bus=1, to_bus=2, r_ohm=.02, x_ohm=.01),
                    dict(from_bus=1, to_bus=3, r_ohm=.03, x_ohm=.015)]
        encoded = encode_physical_tree(branches, [1, 2, 3], 0, .4, 100.)
        c, r, x = encoded.candidate, encoded.r, encoded.x
    else:
        c = lk.RadialCandidate.from_supports('small', 3, [(0, 1)])
        r = np.linspace(.004, .008, c.m); x = .5*r
    p = rng.uniform(.02, .08, (12, 3)); q = p*rng.uniform(.1, .6, p.shape)
    sp = rng.uniform(-.01, .01, 12); sq = rng.uniform(-.004, .004, 12)
    phy = lk.ac_predict(c, r, x, p, q, sp, sq)
    D = lk.Observations(p, q, phy['v'], master_p=phy['master_p'], master_q=phy['master_q'])
    meter = lk.MeasurementModel(p_sigma=.0001, q_sigma=.0001, v_sigma=.0001,
                               master_p_sigma=.0001, master_q_sigma=.0001)
    opts = dict(r_bounds=(0., .05), x_bounds=(0., .03), ratio_bounds=(.1, 1.),
                source_p_bounds=(-.1, .1), source_q_bounds=(-.1, .1),
                voltage_bounds=(.8, 1.2), multistart=1, max_nfev=300, tol=1e-9)
    return c, D, meter, opts, dict(r=r, x=x, source_p=sp, source_q=sq)


def test_sparse_jacobian_is_same_exact_jacobian():
    c, D, meter, opts, state = fixture()
    problem = _RxInputProblem(c, D, meter, initial=state,
        **{key:value for key, value in opts.items() if key not in ('multistart', 'max_nfev', 'tol')})
    sparse = _sparse_jacobian(problem, problem.start)
    assert issparse(sparse)
    assert np.array_equal(sparse.toarray(), problem.jacobian(problem.start))
    direction = np.random.default_rng(40).normal(size=problem.width)
    step = 1e-7
    numeric = (problem.evaluate(problem.start+step*direction)-problem.evaluate(problem.start-step*direction))/(2*step)
    assert np.max(abs(numeric-sparse@direction)) < 2e-5


def test_data_only_linear_initial_is_legal():
    c, D, meter, opts, _ = fixture()
    initial, record = history_linear_initial(c, D, meter,
        **{key:value for key, value in opts.items() if key in (
            'r_bounds', 'x_bounds', 'ratio_bounds', 'source_p_bounds', 'source_q_bounds')})
    assert not record['truth_used'] and not record['future_used']
    assert np.all(initial['r'] >= 0) and np.all(initial['x'] >= .1*initial['r'])
    assert np.all(initial['x'] <= initial['r'])
    assert np.isfinite(lk.ac_predict(c, initial['r'], initial['x'], D.p, D.q,
                       initial['source_p'], initial['source_q'])['v']).all()


def test_sparse_fit_matches_dense_same_objective_on_small_case():
    c, D, meter, opts, _ = fixture()
    initial, _ = history_linear_initial(c, D, meter,
        **{key:value for key, value in opts.items() if key in (
            'r_bounds', 'x_bounds', 'ratio_bounds', 'source_p_bounds', 'source_q_bounds')})
    sparse = fit_rx_inputs(c, D, meter, initial=initial, **opts)
    dense = dense_fit(c, D, meter, initial=initial, **opts)
    assert sparse.success and dense.success
    assert sparse.loss < 1e-5 and dense.loss < 1e-5
    assert abs(sparse.loss-dense.loss) < 1e-5
    assert np.max(abs(sparse.predicted['v']-dense.predicted['v'])) < 1e-6
    assert sparse.diagnostics['information_spectrum_available'] is False
    assert sparse.diagnostics['global_optimality_certified'] is False


def test_zero_virtual_stub_approaches_correct_physical_fit():
    c, D, meter, opts, _ = fixture(zero_stub=True)
    fit = fit_rx_inputs(c, D, meter, **opts)
    assert fit.success
    assert fit.loss < 1e-3
    assert np.max(abs(fit.predicted['v']-D.v)) < 1e-6
    assert fit.diagnostics['ratio_constraints']['feasible']
    assert fit.diagnostics['source_node'] == 1


def test_failed_budget_is_retained_and_not_labeled_success():
    c, D, meter, opts, _ = fixture()
    opts.update(max_nfev=1, multistart=2)
    counter = {}
    fit = fit_rx_inputs(c, D, meter, call_counter=counter, **opts)
    assert not fit.success
    assert len(fit.diagnostics['restarts']) == 2
    assert all(not row['success'] for row in fit.diagnostics['restarts'])
    assert np.isfinite(fit.loss)
    assert counter['fit_calls'] == 1 and counter['solver_calls'] == 2
    assert counter['ac_calls'] >= 2


def test_fixed_line_calls_preserve_original_source_only_path():
    c, D, meter, opts, state = fixture()
    fit = fit_rx_inputs(c, D, meter, fixed_r=state['r'], fixed_x=state['x'], **opts)
    assert fit.success
    assert fit.loss < 1e-8
    assert fit.diagnostics['fixed_lines']


def test_sparse_fixed_source_fit_matches_original_dense_noisy_optimum():
    c, D, meter, opts, state = fixture()
    rng = np.random.default_rng(220)
    noisy = lk.Observations(D.p, D.q, D.v+rng.normal(0, .0001, D.v.shape),
        master_p=D.master_p+rng.normal(0, .0001, D.t),
        master_q=D.master_q+rng.normal(0, .0001, D.t))
    new = fit_rx_inputs(c, noisy, meter, fixed_r=state['r'], fixed_x=state['x'], **opts)
    old = dense_fit(c, noisy, meter, fixed_r=state['r'], fixed_x=state['x'], **opts)
    assert new.success and old.success
    assert abs(new.loss-old.loss) < 1e-6
    assert np.max(abs(new.source_p-old.source_p)) < 1e-7
    assert np.max(abs(new.source_q-old.source_q)) < 1e-7
    assert new.diagnostics['information_spectrum_available'] is False


def test_source_only_sparse_jacobian_matches_original_and_finite_difference():
    c, D, meter, opts, state = fixture()
    args = {key:value for key, value in opts.items() if key not in ('multistart', 'max_nfev', 'tol')}
    problem = _SourceInputProblem(c, D, meter, initial=None,
                                  fixed_r=state['r'], fixed_x=state['x'], call_counter=None, **args)
    jac = _sparse_jacobian(problem, problem.start)
    assert np.array_equal(jac.toarray(), problem.base.jacobian(problem.start))
    direction = np.random.default_rng(221).normal(size=problem.width)
    h = 1e-7
    numeric = (problem.evaluate(problem.start+h*direction)-problem.evaluate(problem.start-h*direction))/(2*h)
    assert np.max(abs(numeric-jac@direction)) < 2e-5


def test_invalid_budget_is_rejected():
    c, D, meter, opts, _ = fixture()
    opts['multistart'] = 0
    with pytest.raises(ValueError, match='multistart'):
        fit_rx_inputs(c, D, meter, **opts)


def test_equal_box_bounds_are_explicitly_rejected():
    c, D, meter, opts, state = fixture(zero_stub=True)
    rhi = np.full(c.m, .05); xhi = np.full(c.m, .03)
    zero = (state['r'] == 0) & (state['x'] == 0)
    rhi[zero] = 0.; xhi[zero] = 0.
    opts.update(r_bounds=(np.zeros(c.m), rhi), x_bounds=(np.zeros(c.m), xhi))
    # The experiment uses the same 0-lower/positive-upper box for every
    # candidate; it never fixes true virtual locations inside the planner.
    # The legacy validator requires strict box bounds, even at zero corners.
    with pytest.raises(ValueError, match='lower < upper'):
        fit_rx_inputs(c, D, meter, **opts)


def test_bad_initial_ac_shrinks_start_without_changing_domain():
    c, D, meter, opts, state = fixture()
    opts.update(r_bounds=(0., 10.), x_bounds=(0., 6.), max_nfev=1)
    initial = dict(state, r=np.full(c.m, 5.), x=np.full(c.m, 2.5))
    fit = fit_rx_inputs(c, D, meter, initial=initial, **opts)
    trials = fit.diagnostics['restarts'][0]['initial_ac_trials']
    assert not trials[0]['ac_success']
    assert trials[-1]['ac_success']
    assert fit.diagnostics['r_bounds_original'][1] == [10.]*c.m
    assert fit.diagnostics['x_bounds_original'][1] == [6.]*c.m
