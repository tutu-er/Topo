"""Independent full-box BVLS/closed-form checks for exact history elimination."""
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import lsq_linear

try:
    from .history_qp_acceleration import accelerate_history_qp
except ImportError:
    from history_qp_acceleration import accelerate_history_qp


def _pair():
    # Column order: first line, first H state, second line, second H state.
    matrix = np.array([[1., 2., .5, 0.], [-.2, 0., .8, 3.],
        [2., .2, -1., .4], [1., 0., .1, 0.], [.2, 0., 2., 0.]])
    return SimpleNamespace(A=matrix, F=np.empty((0, 4)),
        c=np.array([2., 3., 4., -.7, .2]),
        B=np.array([[0., 0.], [0., 0.], [0., 0.], [.1, .4], [.3, -.2]]),
        lower=np.array([-.5, -100., -.5, -100.]),
        upper=np.array([.5, 100., .5, 100.]), scale=np.array([.2, 2., .3, 3.]),
        metadata=dict(variable_layout=[dict(beta=(0, 1), history_state=(1, 2)),
            dict(beta=(2, 3), history_state=(3, 4))], free_indices=[0, 1, 2, 3], future_row_start=3),
        _solver_cache={})


def _inputs(pair, u):
    matrix = np.vstack((pair.A, pair.F))
    d = np.r_[pair.c+pair.B@np.asarray(u), np.zeros(len(pair.F))]
    free = pair.lower < pair.upper
    z = np.clip(np.zeros(len(pair.lower)), pair.lower, pair.upper)
    z[~free] = pair.lower[~free]
    return matrix, d, z, free


def _full_bvls(pair, matrix, d, z, free):
    """Independent complete original-coordinate box least-squares reference."""
    z = z.copy()
    adjusted = d+matrix[:, ~free]@z[~free]
    fit = lsq_linear(matrix[:, free]*pair.scale[free], -adjusted,
        bounds=(pair.lower[free]/pair.scale[free], pair.upper[free]/pair.scale[free]),
        method="bvls", tol=1e-12, max_iter=1000)
    assert fit.success
    z[free] = fit.x*pair.scale[free]
    return z, float(np.sum((d+matrix@z)**2))


def _tangent_gap(pair, C, d, z):
    residual = d+C@z
    gradient = 2*C.T@residual
    # The original-box convex first-order lower bound, not reduced BVLS status.
    correction = np.minimum(gradient*(pair.lower-z), gradient*(pair.upper-z)).sum()
    return float(max(0., -correction))


def test_wide_inactive_history_boxes_match_complete_bvls_for_kw_steps():
    pair = _pair()
    for control_kw in ([0., 0.], [.5, 0.], [0., 3.], [6., 6.]):
        C, d, z, free = _inputs(pair, control_kw)
        fast = accelerate_history_qp(pair, C, d, z, free)
        assert fast is not None and fast["success"]
        reference, objective = _full_bvls(pair, C, d, z, free)
        np.testing.assert_allclose(fast["z"], reference, atol=1e-10)
        assert np.sum((d+C@fast["z"])**2) == pytest.approx(objective, abs=1e-12)
        assert _tangent_gap(pair, C, d, fast["z"]) < 2e-10
    assert len(pair._solver_cache["history_qp_qr"]) == 1


def test_nonzero_history_intercept_and_active_line_have_closed_form_optimum():
    # Residual (7+2h+3b, -5+4b); b in [-.4,.4]. Optimum b=.4,
    # h=-(7+3*.4)/2=-4.1 and q=(-5+4*.4)^2. Dropping 7 loses h.
    pair = SimpleNamespace(lower=np.array([-.4, -10.]), upper=np.array([.4, 10.]),
        scale=np.array([.1, 2.]), metadata=dict(variable_layout=[dict(history_state=(1, 2))],
            free_indices=[0, 1], future_row_start=1), _solver_cache={})
    C, d = np.array([[3., 2.], [4., 0.]]), np.array([7., -5.])
    fast = accelerate_history_qp(pair, C, d, np.zeros(2), np.ones(2, bool))
    assert fast is not None
    np.testing.assert_allclose(fast["z"], [.4, -4.1], atol=1e-12)
    assert np.sum((d+C@fast["z"])**2) == pytest.approx(11.56)
    assert _tangent_gap(pair, C, d, fast["z"]) < 1e-11


def test_recovered_active_history_bound_returns_none_and_never_clips():
    pair = _pair()
    pair.lower[[1, 3]], pair.upper[[1, 3]] = -.01, .01
    C, d, z, free = _inputs(pair, [0., 0.])
    assert accelerate_history_qp(pair, C, d, z, free) is None
    full, objective = _full_bvls(pair, C, d, z, free)
    assert np.all(full >= pair.lower-1e-14) and np.all(full <= pair.upper+1e-14)
    assert objective > 1.
    np.testing.assert_array_equal(z, np.zeros(4))


def test_preeliminated_and_current_fixed_coordinates_are_both_preserved():
    pair = _pair()
    # Original coordinate 2 was already substituted into c by build_frozen;
    # current coordinate 0 is additionally fixed by this QP's original box.
    pair.metadata.update(free_indices=[0, 1, 3, 4], variable_layout=[
        dict(beta=(0, 1), history_state=(1, 2)), dict(beta=(3, 4), history_state=(4, 5))])
    pair.lower[0] = pair.upper[0] = .3
    pair.c += np.array([.2, -.1, .4, .3, -.2])  # retained pre-substitution offset
    C, d, z, free = _inputs(pair, [1., 2.])
    fast = accelerate_history_qp(pair, C, d, z, free)
    assert fast is not None and fast["z"][0] == .3
    reference, objective = _full_bvls(pair, C, d, z, free)
    np.testing.assert_allclose(fast["z"], reference, atol=1e-10)
    assert np.sum((d+C@fast["z"])**2) == pytest.approx(objective, abs=1e-12)


def test_rank_deficient_or_nonzero_future_history_columns_fall_back():
    pair = _pair()
    pair.A[:, 3] = pair.A[:, 1]
    assert accelerate_history_qp(pair, *_inputs(pair, [0., 0.])) is None
    pair = _pair()
    pair.A[-1, 1] = 1e-15
    assert accelerate_history_qp(pair, *_inputs(pair, [0., 0.])) is None


def test_metadata_absent_or_missing_mapping_falls_back_without_global_cache():
    pair = _pair()
    pair.metadata = {}
    assert accelerate_history_qp(pair, *_inputs(pair, [0., 0.])) is None
    assert pair._solver_cache == {}
    pair = _pair()
    pair.metadata["fullfree_indices"] = pair.metadata.pop("free_indices")
    del pair._solver_cache
    assert accelerate_history_qp(pair, *_inputs(pair, [0., 0.])) is not None


def test_only_history_free_columns_have_exact_analytic_recovery():
    pair = _pair()
    pair.lower[[0, 2]], pair.upper[[0, 2]] = .1, .1
    C, d, z, free = _inputs(pair, [2., 3.])
    fast = accelerate_history_qp(pair, C, d, z, free)
    assert fast is not None and fast["remaining_free_count"] == 0
    reference = z.copy()
    reference[free] = np.linalg.lstsq(C[:, free], -(d+C[:, ~free]@z[~free]), rcond=None)[0]
    np.testing.assert_allclose(fast["z"], reference, atol=1e-12)


def test_cache_is_pair_local_and_checks_changed_supplied_matrix():
    first, second = _pair(), _pair()
    assert accelerate_history_qp(first, *_inputs(first, [0., 0.])) is not None
    assert second._solver_cache == {}
    first.A[0, 0] += .4
    fast = accelerate_history_qp(first, *_inputs(first, [0., 0.]))
    C, d, z, free = _inputs(first, [0., 0.])
    reference, _ = _full_bvls(first, C, d, z, free)
    np.testing.assert_allclose(fast["z"], reference, atol=1e-10)
    assert len(first._solver_cache["history_qp_qr"]) == 2


def _ac_pair():
    # Actual Stage-A AC Jacobians and boxes; the independent reference still
    # solves every original column instead of following the QR implementation.
    try:
        from .test_adaptive_history import _context, build_frozen_pairs
    except ImportError:
        from test_adaptive_history import _context, build_frozen_pairs
    given = build_frozen_pairs(_context(source_delta_bound=.05), source_mode="local_refit")[0]
    return SimpleNamespace(**given, _solver_cache={})


def test_actual_ac_frozen_pair_matches_full_bvls_at_kw_controls():
    pair = _ac_pair()
    controls = [np.zeros(9), np.full(9, .5), np.tile([0., 3., 0.], 3),
                np.array([0., 0., 0., 0., 0., 0., 6., 0., 0.])]
    for u in controls:
        C, d, z, free = _inputs(pair, u)
        fast = accelerate_history_qp(pair, C, d, z, free)
        assert fast is not None and fast["recovered_history_box_feasible"]
        _, reference_objective = _full_bvls(pair, C, d, z, free)
        objective = float(np.sum((d+C@fast["z"])**2))
        assert objective == pytest.approx(reference_objective, rel=2e-7, abs=1e-7)
        assert _tangent_gap(pair, C, d, fast["z"]) < max(1e-5, 1e-7*objective)


def benchmark_actual_ac_qps(n_calls=200):
    """Optional reproducible timing; no asserted speed or scientific change."""
    pair = _ac_pair()
    controls = np.random.default_rng(9127).uniform(0., 6., size=(n_calls, pair.B.shape[1]))
    inputs = [_inputs(pair, u) for u in controls]
    start = perf_counter()
    fast = [accelerate_history_qp(pair, *values) for values in inputs]
    accelerated_seconds = perf_counter()-start
    start = perf_counter()
    full = [_full_bvls(pair, *values)[1] for values in inputs]
    full_seconds = perf_counter()-start
    discrepancies = [abs(float(np.sum((values[1]+values[0]@result["z"])**2))-q)
        for values, result, q in zip(inputs, fast, full) if result is not None]
    return dict(calls=n_calls, residual_rows=len(pair.c)+len(pair.F), columns=len(pair.lower),
        accelerated_seconds=accelerated_seconds, full_bvls_seconds=full_seconds,
        fallbacks=sum(result is None for result in fast), max_objective_difference=max(discrepancies, default=None),
        scope="same actual AC frozen QP matrices; timing diagnostic, no experiment-cohort change")
