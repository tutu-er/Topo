import numpy as np
import pytest
from ..reference_benchmark.test_fitting import fixture
from ..reference_benchmark.fitting import _SourceInputProblem, _sparse_jacobian
from ..ev_rx_fit import _RxInputProblem
from .blocks import problem_blocks, dense_jacobian, matvec, gradient


@pytest.mark.parametrize('fixed', [False, True])
@pytest.mark.parametrize('zero_stub', [False, True])
def test_blocks_equal_reference_and_finite_difference(fixed, zero_stub):
    c, obs, model, opts, initial = fixture(zero_stub)
    if zero_stub and not fixed:
        # Centered differences require an interior point. Zero-edge boundary
        # directions are checked separately using feasible one-sided steps.
        initial = dict(initial, r=initial['r']+.001, x=.5*(initial['r']+.001))
    opts = {k: v for k, v in opts.items() if k not in ('multistart','max_nfev','tol')}
    if fixed:
        problem = _SourceInputProblem(c, obs, model, initial=initial,
            fixed_r=initial['r'], fixed_x=initial['x'], call_counter=None, **opts)
    else:
        problem = _RxInputProblem(c, obs, model, initial=initial, **opts)
    x = problem.start
    line, source = problem_blocks(problem, x)
    full = dense_jacobian(line, source)
    assert np.array_equal(full, _sparse_jacobian(problem, x).toarray())
    direction = np.random.default_rng(813).normal(size=len(x))
    h = 1e-7
    numeric = (problem.evaluate(x+h*direction)-problem.evaluate(x-h*direction))/(2*h)
    assert np.max(abs(numeric-full@direction)) < 3e-5
    assert np.allclose(matvec(line,source,direction).ravel(), full@direction, atol=1e-10)
    residual = problem.evaluate(x)
    assert np.allclose(gradient(line,source,residual), full.T@residual, atol=1e-10)


def test_zero_edge_one_sided_feasible_derivative():
    c, obs, model, opts, initial = fixture(True)
    opts = {k:v for k,v in opts.items() if k not in ('multistart','max_nfev','tol')}
    problem = _RxInputProblem(c,obs,model,initial=initial,**opts)
    line,source = problem_blocks(problem,problem.start)
    direction = np.zeros(problem.width)
    direction[:c.m] = 1.
    h = 1e-7
    numeric = (problem.evaluate(problem.start+h*direction)-problem.evaluate(problem.start))/h
    assert np.max(abs(numeric-matvec(line,source,direction).ravel())) < .01
