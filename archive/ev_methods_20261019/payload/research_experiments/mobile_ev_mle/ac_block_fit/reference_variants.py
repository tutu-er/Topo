"""Isolated-process numerical controls for the unchanged reference AC fitter.

``fit_reference_jac`` changes only SciPy's variable scale to ``'jac'`` and,
when explicitly requested, its LSMR atol/btol.  It does not edit the reference
source or change its residual, exact derivative, bounds, starts or budgets.
The temporary module hook is process-local and must not be used concurrently
with other fits in the same process; benchmark workers are isolated processes.
"""
from contextlib import contextmanager
import numpy as np

from ..reference_benchmark import fitting


_ACTIVE = False


@contextmanager
def _jac_scaled_solver(inner_lsmr_tol):
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError('reference variant requires one isolated fit per process')
    original = fitting.least_squares

    def jac_scaled(*args, **kwargs):
        kwargs['x_scale'] = 'jac'
        if inner_lsmr_tol is not None:
            options = dict(kwargs.get('tr_options', {}))
            options.update(atol=inner_lsmr_tol, btol=inner_lsmr_tol)
            kwargs['tr_options'] = options
        return original(*args, **kwargs)

    _ACTIVE = True
    fitting.least_squares = jac_scaled
    try:
        yield
    finally:
        fitting.least_squares = original
        _ACTIVE = False


def fit_reference_jac(*args, inner_lsmr_tol=None, **kwargs):
    """Call the original reference fitter with Jacobian column scaling.

    ``inner_lsmr_tol=None`` retains the original atol/btol (currently 1e-10).
    An explicit positive finite tolerance changes both; LSMR maxiter, outer
    tolerances, max_nfev and all physical/statistical inputs remain unchanged.
    The original solver function is restored even after an exception.
    """
    if inner_lsmr_tol is not None:
        inner_lsmr_tol = float(inner_lsmr_tol)
        if not np.isfinite(inner_lsmr_tol) or inner_lsmr_tol <= 0:
            raise ValueError('inner_lsmr_tol must be finite and positive')
    with _jac_scaled_solver(inner_lsmr_tol):
        result = fitting.fit_rx_inputs(*args, **kwargs)
    result.diagnostics.update(reference_variant='original_TRF_with_jac_column_scale',
        variable_scale='jac', reference_scientific_objective_unchanged=True,
        inner_lsmr_tolerance_override=inner_lsmr_tol,
        reference_hook_scope='single isolated worker process; restored in finally')
    return result


__all__ = ['fit_reference_jac']
