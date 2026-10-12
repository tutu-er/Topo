"""Independent physical LP oracles and failure contracts for the research fitter."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from rnj_wzzt.estimation import laminar_l1_milp as core
from rnj_wzzt.estimation.unmetered_inputs import PreparedSourceInputs
from research_experiments.mobile_ev_probing import wzzt_adapter as adapter


def _inputs(p, q):
    return PreparedSourceInputs(tuple(p), tuple(q),
        tuple(pd.RangeIndex(len(a)) for a in p), tuple("independent_test" for _ in p))


def _physical(supports, r, x, h, p, q, sp, sq):
    n = p.shape[1]
    R, X, gr, gx = np.zeros((n, n)), np.zeros((n, n)), np.zeros(n), np.zeros(n)
    for s, rk, xk, on in zip(supports, r, x, h, strict=True):
        z = np.array([int(i in s) for i in range(n)])
        R += rk * np.outer(z, z)
        X += xk * np.outer(z, z)
        gr += rk * on * z
        gx += xk * on * z
    return p @ R.T + q @ X.T + sp[:, None] * gr + sq[:, None] * gx


def _example(counts=(10,)):
    rng = np.random.default_rng(642)
    supports = ((0,), (1,), (2,), (0, 1), (0, 1, 2))
    r = np.array([.1, .15, .12, .2, .25])
    x = np.array([.08, .1, .13, .11, .14])
    h = np.array([0, 0, 0, 1, 1])
    scenarios, source_p, source_q = [], [], []
    for t in counts:
        p, q = rng.uniform(.01, .2, (t, 3)), rng.uniform(-.1, .1, (t, 3))
        sp, sq = rng.uniform(.01, .2, t), rng.uniform(-.08, .08, t)
        y = _physical(supports, r, x, h, p, q, sp, sq)
        scenarios.append(dict(P_terminal=p, Q_terminal=q, drop_target=y))
        source_p.append(sp)
        source_q.append(sq)
    return scenarios, _inputs(source_p, source_q), supports, r, x, h


def _oracle(scenarios, supports, source_inputs, bp=None, bq=None,
            r_path=None, x_path=None):
    """Enumerate physical source locations, eliminate every binary/product."""
    k, n = len(supports), scenarios[0]["P_terminal"].shape[1]
    obs = sum(len(s["P_terminal"]) * n for s in scenarios)
    best = np.inf
    for location in supports:
        h = [int(set(location) <= set(s)) for s in supports]
        design, band, target = [], [], []
        for w, scenario in enumerate(scenarios):
            p, q, y = (scenario[key] for key in ("P_terminal", "Q_terminal", "drop_target"))
            for t in range(len(p)):
                for i in range(n):
                    rr, xx, br, bx = [], [], [], []
                    for a, s in enumerate(supports):
                        membership = int(i in s)
                        rr.append(membership * (sum(p[t, j] for j in s) + source_inputs.p[w][t] * h[a]))
                        xx.append(membership * (sum(q[t, j] for j in s) + source_inputs.q[w][t] * h[a]))
                        br.append(membership * h[a] * (0 if bp is None else bp[w][t]))
                        bx.append(membership * h[a] * (0 if bq is None else bq[w][t]))
                    design.append(rr + xx)
                    band.append(br + bx)
                    target.append(y[t, i])
        design, band, target = np.asarray(design), np.asarray(band), np.asarray(target)
        constraint_rows = [np.c_[design - band, -np.eye(obs)],
                           np.c_[-design - band, -np.eye(obs)]]
        constraint_upper = [target, -target]
        membership = np.array([[int(i in s) for s in supports] for i in range(n)])
        for path, start in ((r_path, 0), (x_path, k)):
            if path is not None:
                row = np.zeros((n, 2 * k + obs))
                row[:, start:start + k] = membership
                constraint_rows.append(row)
                constraint_upper.append(np.broadcast_to(path, (n,)))
        result = linprog(np.r_[np.zeros(2 * k), np.ones(obs) / obs],
            A_ub=np.vstack(constraint_rows), b_ub=np.concatenate(constraint_upper),
            bounds=[(0, 1)] * (2 * k) + [(0, None)] * obs, method="highs")
        assert result.success, result.message
        best = min(best, result.fun)
    return best


def _extensions(n, supports):
    existing = list(map(set, supports))
    for bits in range(1, 1 << n):
        s = {i for i in range(n) if bits & (1 << i)}
        if s not in existing and all(not s & a or s <= a or a <= s for a in existing):
            yield tuple(sorted(s))


@pytest.mark.parametrize("with_allowance", [False, True])
def test_fixed_source_fit_matches_independent_enumerated_lp(with_allowance):
    scenarios, inputs, supports, *_ = _example((5, 4))
    for w, s in enumerate(scenarios):
        s["drop_target"] += .007 * np.sin(np.arange(s["drop_target"].size).reshape(-1, 3) + w)
    bp = tuple(np.full(len(p), .013) for p in inputs.p) if with_allowance else None
    bq = tuple(np.full(len(q), .009) for q in inputs.q) if with_allowance else None
    fit = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
                                          source_p_error=bp, source_q_error=bq)
    assert fit.objective == pytest.approx(_oracle(scenarios, supports, inputs, bp, bq), abs=2e-8)
    assert fit.train_mae + 1e-10 >= fit.adjusted_train_mae
    assert core.solver_diagnostics_prove_optimality(fit.diagnostics)


def test_first_extension_matches_independent_full_support_domain():
    scenarios, inputs, *_ = _example((6,))
    initial = ((0,), (1,), (2,))
    expected = min(_oracle(scenarios, (*initial, s), inputs) for s in _extensions(3, initial))
    fit = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs, max_atoms=4)
    assert len(fit.path) == 2
    assert fit.selected.objective == pytest.approx(expected, abs=2e-8)
    assert [a.stage for a in fit.attempted_solves] == ["initial", "extension", "refit"]
    assert all(a.accepted for a in fit.attempted_solves)


def test_zero_source_matches_ordinary_core_with_same_singleton_start():
    scenarios, inputs, *_ = _example((12,))
    zeros = _inputs(tuple(np.zeros_like(a) for a in inputs.p), tuple(np.zeros_like(a) for a in inputs.q))
    plain = core.fit_laminar_l1_sensitivity(scenarios, initial_supports=((0,), (1,), (2,)),
                                          r_upper_bound=1, x_upper_bound=1, max_atoms=5)
    absent = adapter.fit_source_aware_wzzt(scenarios, max_atoms=5)
    zero = adapter.fit_source_aware_wzzt(scenarios, source_inputs=zeros, max_atoms=5)
    assert absent.selected.objective == pytest.approx(plain.train_mae, abs=1e-8)
    np.testing.assert_allclose(zero.r_matrix, absent.r_matrix, atol=1e-7)
    np.testing.assert_allclose(zero.x_matrix, absent.x_matrix, atol=1e-7)
    assert not np.any(zero.h)
    assert not np.any(zero.g_r)


def test_source_prediction_validation_and_refit_keep_one_consistent_path():
    train, inputs, supports, r, x, h = _example((18,))
    validation, v_inputs, *_ = _example((7,))
    result = adapter.fit_source_aware_wzzt(train, source_inputs=inputs,
        certified_supports=supports, max_atoms=5, validation_scenarios=validation,
        validation_source_inputs=v_inputs)
    fit = result.selected
    assert set(fit.support_indices) == set(supports)
    assert fit.train_mae < 1e-8
    assert fit.validation_mae < 1e-8
    np.testing.assert_allclose(fit.r_values, r, atol=1e-7)
    np.testing.assert_allclose(fit.x_values, x, atol=1e-7)
    np.testing.assert_array_equal(fit.h, h)
    reconstructed = _physical(fit.support_indices, fit.r_values, fit.x_values,
        fit.h, train[0]["P_terminal"], train[0]["Q_terminal"], inputs.p[0], inputs.q[0])
    np.testing.assert_allclose(reconstructed, train[0]["drop_target"], atol=1e-8)


def test_source_bound_contains_true_amplitude_error_and_records_raw_loss():
    scenarios, inputs, supports, *_ = _example((15,))
    exact = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs)
    perturbation = .01 * np.sin(np.arange(15))
    shifted = _inputs((inputs.p[0] + perturbation,), inputs.q)
    fit = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=shifted,
        source_p_error=np.abs(perturbation), source_q_error=0)
    assert fit.adjusted_train_mae < 1e-8
    assert fit.train_mae > 1e-5
    # The physical original fit is feasible for the outer-approximation bands.
    raw = np.abs(scenarios[0]["drop_target"] - _physical(supports,
        exact.r_values, exact.x_values, exact.h, scenarios[0]["P_terminal"],
        scenarios[0]["Q_terminal"], shifted.p[0], shifted.q[0]))
    assert np.max(raw - np.abs(perturbation)[:, None] * exact.g_r) < 1e-8


def test_new_call_restarts_without_previous_unverified_rnj_supports():
    scenarios, inputs, *_ = _example((10,))
    result = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs, max_atoms=3)
    assert result.mandatory_supports == ((0,), (1,), (2,))
    assert result.support_indices == result.mandatory_supports
    assert len(result.path) == 1


def test_source_aware_pruning_refits_and_rejects_a_lossy_removal():
    scenarios, inputs, *_ = _example((10,))
    result = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs,
                                         max_atoms=4, zero_tolerance=.99)
    pruning = [a for a in result.attempted_solves if a.stage == "pruning_refit"]
    assert len(pruning) == 1
    assert not pruning[0].accepted
    assert len(result.support_indices) == 4
    expected = min(_oracle(scenarios, (*result.mandatory_supports, s), inputs)
                   for s in _extensions(3, result.mandatory_supports))
    assert result.selected.objective == pytest.approx(expected, abs=1e-8)


def test_zero_source_validation_is_allowed_without_reselecting_training_path():
    train, inputs, supports, r, x, h = _example((18,))
    validation, vi, *_ = _example((6,))
    zeros = _inputs((np.zeros(6),), (np.zeros(6),))
    validation[0]["drop_target"] = _physical(supports, r, x, h,
        validation[0]["P_terminal"], validation[0]["Q_terminal"], zeros.p[0], zeros.q[0])
    result = adapter.fit_source_aware_wzzt(train, source_inputs=inputs,
        certified_supports=supports, max_atoms=5, validation_scenarios=validation,
        validation_source_inputs=zeros, validation_source_p_error=.001)
    assert result.selected.validation_mae < 1e-8
    assert result.selected.adjusted_validation_mae < 1e-8
    np.testing.assert_array_equal(result.h, h)
    assert len(result.attempted_solves) == 1


def test_nonoptimal_extension_is_not_accepted_and_diagnostics_survive(monkeypatch):
    scenarios, inputs, *_ = _example((8,))
    run = adapter.core._run_milp
    calls = 0

    def fail_extension(*args, **kwargs):
        nonlocal calls
        calls += 1
        result, diagnostics = run(*args, **kwargs)
        if calls > 1:
            return SimpleNamespace(x=result.x, fun=result.fun), replace(diagnostics,
                status=1, success=False, message="test time limit", mip_gap=.1, dual_bound=None)
        return result, diagnostics

    monkeypatch.setattr(adapter.core, "_run_milp", fail_extension)
    fit = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs)
    assert fit.stop_reason == "extension_not_proven_optimal"
    assert len(fit.path) == 1
    assert fit.attempted_solves[-1].diagnostics.status == 1
    assert not fit.attempted_solves[-1].accepted
    assert fit.attempted_solves[-1].added_support is None


def test_nonoptimal_fixed_fit_raises_error_with_diagnostics(monkeypatch):
    scenarios, inputs, supports, *_ = _example((6,))
    real = adapter.core._run_milp

    def fail(*args, **kwargs):
        result, diag = real(*args, **kwargs)
        return result, replace(diag, status=1, success=False, mip_gap=.01, dual_bound=None)

    monkeypatch.setattr(adapter.core, "_run_milp", fail)
    with pytest.raises(adapter.SourceSolveError) as exc:
        adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs)
    assert exc.value.diagnostics.status == 1


def test_input_guardrails_and_zero_uncertainty_model_boundary():
    scenarios, inputs, *_ = _example((5,))
    with pytest.raises(ValueError, match="nonzero source uncertainty"):
        adapter.fit_source_aware_wzzt(scenarios, source_p_error=.1)
    with pytest.raises(ValueError, match="nonnegative"):
        adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs, source_p_error=-1)
    with pytest.raises(ValueError, match="mandatory"):
        adapter.fit_source_aware_wzzt(scenarios, max_atoms=2)
    with pytest.raises(ValueError, match="laminar"):
        adapter.fit_source_aware_wzzt(scenarios, certified_supports=((0, 1), (1, 2)))


def test_certified_R_lower_bound_is_imposed_in_every_retained_fit():
    scenarios, inputs, supports, *_ = _example((10,))
    fixed = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
                                             certified_r_lower_bounds={(0, 1): .22})
    assert fixed.r_values[fixed.support_indices.index((0, 1))] >= .22 - 1e-8
    result = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs,
        certified_supports=((0, 1),), certified_r_lower_bounds={(0, 1): .22}, max_atoms=5)
    for point in result.path:
        assert point.r_values[point.support_indices.index((0, 1))] >= .22 - 1e-8
    assert result.summary()["certified_r_lower_bounds"] == [{"support": [0, 1], "lower": .22}]
    with pytest.raises(ValueError, match="retained certified"):
        adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs,
                                     certified_r_lower_bounds={(0, 1): .1})
    with pytest.raises(ValueError, match="exceeds"):
        adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
                                         certified_r_lower_bounds={(0, 1): 2.})


def test_fixed_path_bounds_match_independent_physical_lp_oracle():
    scenarios, inputs, supports, *_ = _example((10,))
    rp, xp = (.48, .55, .4), (.3, .34, .31)
    fit = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
        r_path_upper_bound=rp, x_path_upper_bound=xp)
    expected = _oracle(scenarios, supports, inputs, r_path=rp, x_path=xp)
    assert fit.objective == pytest.approx(expected, abs=2e-8)
    assert np.all(np.diag(fit.r_matrix) <= np.asarray(rp) + 1e-8)
    assert np.all(np.diag(fit.x_matrix) <= np.asarray(xp) + 1e-8)
    assert np.all(fit.g_r <= np.diag(fit.r_matrix) + 1e-8)
    assert np.all(fit.g_x <= np.diag(fit.x_matrix) + 1e-8)
    assert fit.r_path_upper_bound == rp


def test_extension_path_bounds_include_new_atom_diagonal_products():
    scenarios, inputs, *_ = _example((8,))
    rp, xp = (.49, .53, .35), (.31, .32, .26)
    initial = ((0,), (1,), (2,))
    expected = min(_oracle(scenarios, (*initial, s), inputs, r_path=rp, x_path=xp)
                   for s in _extensions(3, initial))
    result = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs, max_atoms=4,
        r_path_upper_bound=rp, x_path_upper_bound=xp)
    assert len(result.path) == 2
    assert result.selected.objective == pytest.approx(expected, abs=2e-8)
    for fit in result.path:
        assert np.all(np.diag(fit.r_matrix) <= np.asarray(rp) + 1e-8)
        assert np.all(np.diag(fit.x_matrix) <= np.asarray(xp) + 1e-8)
    assert result.summary()["r_path_upper_bound"] == rp
    # Zero-source branch uses core extension auxiliaries, not source blocks.
    plain = adapter.fit_source_aware_wzzt(scenarios, max_atoms=4,
        r_path_upper_bound=.4, x_path_upper_bound=.3)
    assert np.max(np.diag(plain.r_matrix)) <= .4 + 1e-8
    assert np.max(np.diag(plain.x_matrix)) <= .3 + 1e-8


def test_incompatible_certified_lower_and_path_upper_bounds_are_not_accepted():
    scenarios, inputs, supports, *_ = _example((5,))
    with pytest.raises(adapter.SourceSolveError) as failure:
        adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
            certified_r_lower_bounds={(0, 1): .3}, r_path_upper_bound=.2)
    assert failure.value.diagnostics.status == 2
    result = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs,
        certified_supports=((0, 1),), certified_r_lower_bounds={(0, 1): .3},
        r_path_upper_bound=.2)
    assert result.selected is None
    assert result.stop_reason == "initial_not_proven_optimal"


def test_certified_X_bound_is_separate_from_R_and_retained_with_path_bound():
    scenarios, inputs, supports, *_ = _example((8,))
    result = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs,
        certified_supports=supports, certified_x_lower_bounds={(0, 1): .14},
        r_path_upper_bound=.8, x_path_upper_bound=.5, max_atoms=5)
    for fit in result.path:
        index = fit.support_indices.index((0, 1))
        assert fit.x_values[index] >= .14 - 1e-8
        assert np.max(np.diag(fit.x_matrix)) <= .5 + 1e-8
    assert result.certified_r_lower_bounds == ()
    assert result.summary()["certified_x_lower_bounds"] == [{"support": [0, 1], "lower": .14}]
    with pytest.raises(ValueError, match="retained certified"):
        adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs,
            certified_x_lower_bounds={(0, 1): .1})


@pytest.mark.parametrize("invalid", [-1, np.inf, True, [.1, .2], [.1, .2, np.nan]])
def test_path_bound_guardrails(invalid):
    scenarios, inputs, supports, *_ = _example((5,))
    with pytest.raises(ValueError, match="r_path_upper_bound"):
        adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
                                          r_path_upper_bound=invalid)


def test_certified_clade_weight_does_not_force_source_into_that_clade():
    scenarios, inputs, supports, r, x, _ = _example((18,))
    outside_source_path = np.array([0, 0, 1, 0, 1])
    scenarios[0]["drop_target"] = _physical(supports, r, x, outside_source_path,
        scenarios[0]["P_terminal"], scenarios[0]["Q_terminal"], inputs.p[0], inputs.q[0])
    fit = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
        certified_r_lower_bounds={(0, 1): .19}, certified_x_lower_bounds={(0, 1): .1},
        r_path_upper_bound=.7, x_path_upper_bound=.5)
    assert fit.objective < 1e-8
    np.testing.assert_array_equal(fit.h, outside_source_path)
    assert fit.r_values[3] >= .19 - 1e-8 and fit.x_values[3] >= .1 - 1e-8
    assert fit.h[3] == 0
    assert np.all(fit.g_r <= np.diag(fit.r_matrix) + 1e-8)


def test_outputwise_source_allowance_does_not_claim_a_shared_amplitude_fit():
    # Fix g_R=(1,1) through a known aggregate weight and path bounds.
    # Opposite terminal errors both lie in outputwise bands, but no scalar
    # shared source error can create them. This guards the stated relaxation.
    error_bound = .1
    scenarios = [dict(P_terminal=np.zeros((1, 2)), Q_terminal=np.zeros((1, 2)),
                      drop_target=np.array([[1 + error_bound, 1 - error_bound]]))]
    inputs = _inputs((np.ones(1),), (np.zeros(1),))
    fit = adapter.fit_fixed_source_supports(scenarios, ((0,), (1,), (0, 1)),
        source_inputs=inputs, source_p_error=error_bound,
        certified_r_lower_bounds={(0, 1): 1}, r_path_upper_bound=1,
        x_path_upper_bound=0)
    np.testing.assert_allclose(fit.g_r, np.ones(2), atol=1e-8)
    assert fit.adjusted_train_mae < 1e-8
    assert fit.train_mae == pytest.approx(error_bound, abs=1e-8)
    residual = scenarios[0]["drop_target"][0] - fit.g_r
    for shared_error in np.linspace(-error_bound, error_bound, 31):
        assert np.mean(np.abs(residual - shared_error * fit.g_r)) >= error_bound - 1e-8


def test_all_zero_nominal_source_with_nonzero_error_is_an_explicit_model_boundary():
    scenarios, inputs, supports, *_ = _example((5,))
    zeros = _inputs((np.zeros(5),), (np.zeros(5),))
    with pytest.raises(ValueError, match="active nonzero source amplitudes"):
        adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=zeros,
                                          source_p_error=.01)


def test_native_optimal_but_out_of_bounds_vector_is_a_recorded_failed_solve(monkeypatch):
    scenarios, inputs, supports, *_ = _example((6,))
    original = adapter.core._run_milp

    def violate_bound(*args, **kwargs):
        result, diagnostics = original(*args, **kwargs)
        result.x = result.x.copy()
        result.x[0] = -2e-8  # Larger than the core's unchanged 1e-8 tolerance.
        return result, diagnostics

    monkeypatch.setattr(adapter.core, "_run_milp", violate_bound)
    with pytest.raises(adapter.SourceSolveError) as failure:
        adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs)
    diagnostics = failure.value.diagnostics
    assert diagnostics.status == 4 and not diagnostics.success
    assert diagnostics.raw_status == 0
    assert diagnostics.dual_bound is None and diagnostics.mip_gap is None
    assert "physical_postcheck_failed" in diagnostics.message
    assert "tolerance=1e-08" in diagnostics.message
    assert len(failure.value.diagnostics_history) == 2
    assert all(d.status == 4 for d in failure.value.diagnostics_history)
    assert diagnostics.presolve_history == (True, False)
    result = adapter.fit_source_aware_wzzt(scenarios, source_inputs=inputs, max_atoms=3)
    assert result.selected is None
    assert result.attempted_solves[0].diagnostics.status == 4
    assert not result.attempted_solves[0].accepted


def test_only_tolerated_atom_boundary_roundoff_is_removed(monkeypatch):
    scenarios, inputs, supports, r, x, h = _example((18,))
    r[0] = 0.
    scenarios[0]["drop_target"] = _physical(supports, r, x, h,
        scenarios[0]["P_terminal"], scenarios[0]["Q_terminal"], inputs.p[0], inputs.q[0])
    original = adapter.core._run_milp

    def tolerated_roundoff(*args, **kwargs):
        result, diagnostics = original(*args, **kwargs)
        assert abs(result.x[0]) < 1e-8
        result.x = result.x.copy()
        result.x[0] = -5e-9
        return result, diagnostics

    monkeypatch.setattr(adapter.core, "_run_milp", tolerated_roundoff)
    fit = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs)
    assert fit.r_values[0] == 0.
    assert fit.objective < 1e-8
    assert adapter.core.solver_diagnostics_prove_optimality(fit.diagnostics)


def test_physical_path_violation_is_rejected_even_with_valid_atom_bounds(monkeypatch):
    scenarios, inputs, supports, r, x, _ = _example((18,))
    R, X = adapter.core.build_matrices_from_atoms(3, supports, r, x)
    original = adapter.core._run_milp

    def violate_path(*args, **kwargs):
        result, diagnostics = original(*args, **kwargs)
        result.x = result.x.copy()
        result.x[0] += 1e-5  # Remains below atom upper=1 but exceeds path=diag R.
        return result, diagnostics

    monkeypatch.setattr(adapter.core, "_run_milp", violate_path)
    with pytest.raises(adapter.SourceSolveError) as failure:
        adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
            r_path_upper_bound=np.diag(R), x_path_upper_bound=np.diag(X))
    assert failure.value.diagnostics.status == 4
    assert "physical R path exceeds" in failure.value.diagnostics.message


def test_one_presolve_recovery_preserves_domain_and_both_diagnostics(monkeypatch):
    import json

    scenarios, inputs, supports, *_ = _example((18,))
    original, presolves, models = adapter.core._run_milp, [], []

    def first_presolve_is_numerically_invalid(variables, rows, **kwargs):
        presolves.append(kwargs["presolve"])
        models.append(tuple(np.asarray(v).copy() for v in (
            variables.lower, variables.upper, variables.objective, variables.integrality,
            rows.lower, rows.upper, rows.row_index, rows.col_index, rows.data)))
        result, diagnostics = original(variables, rows, **kwargs)
        if kwargs["presolve"]:
            result.x = result.x.copy()
            result.x[0] = -2e-8
        return result, diagnostics

    monkeypatch.setattr(adapter.core, "_run_milp", first_presolve_is_numerically_invalid)
    fit = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs,
        certified_r_lower_bounds={(0, 1): .18}, certified_x_lower_bounds={(0, 1): .08},
        r_upper_bound=.8, x_upper_bound=.7, r_path_upper_bound=.7, x_path_upper_bound=.5,
        source_p_error=.001, source_q_error=.002)
    assert presolves == [True, False]
    for first, second in zip(models[0], models[1]):
        np.testing.assert_array_equal(first, second)
    assert len(fit.diagnostics_history) == 2
    assert fit.diagnostics_history[0].status == 4
    assert fit.diagnostics_history[0].raw_status == 0
    assert fit.diagnostics_history[1].status == 0
    assert fit.diagnostics.presolve_history == (True, False)
    assert adapter.core.solver_diagnostics_prove_optimality(fit.diagnostics)
    assert fit.r_values[fit.support_indices.index((0, 1))] >= .18 - 1e-8
    assert fit.x_values[fit.support_indices.index((0, 1))] >= .08 - 1e-8
    assert np.max(np.diag(fit.r_matrix)) <= .7 + 1e-8
    assert np.max(np.diag(fit.x_matrix)) <= .5 + 1e-8
    assert len(json.loads(json.dumps(fit.diagnostics.to_dict()))["diagnostics_history"]) == 2


def test_auxiliary_roundoff_uses_objective_guard_in_its_own_units(monkeypatch):
    scenarios, inputs, supports, *_ = _example((18,))
    original, count = adapter.core._run_milp, 0

    def auxiliary_roundoff(variables, rows, **kwargs):
        nonlocal count
        count += 1
        result, diagnostics = original(variables, rows, **kwargs)
        result.x = result.x.copy()
        index = variables.slices["absolute_residual"].start
        result.x[index] = -2e-7  # Auxiliary feasibility scale, not an R/X coefficient.
        result.fun = float(np.dot(variables.objective, result.x))
        diagnostics = replace(diagnostics, objective=result.fun, dual_bound=result.fun)
        return result, diagnostics

    monkeypatch.setattr(adapter.core, "_run_milp", auxiliary_roundoff)
    fit = adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs)
    assert fit.objective < 1e-8
    assert np.min(fit.r_values) >= 0 and np.min(fit.x_values) >= 0
    assert count == 1  # Does not create spurious physical-postcheck/recovery failures.
    assert len(fit.diagnostics_history) == 1


def test_nonoptimal_status_does_not_trigger_numerical_recovery(monkeypatch):
    scenarios, inputs, supports, *_ = _example((5,))
    original, count = adapter.core._run_milp, 0

    def time_limit(*args, **kwargs):
        nonlocal count
        count += 1
        result, diagnostics = original(*args, **kwargs)
        return result, replace(diagnostics, status=1, success=False, mip_gap=.1, dual_bound=None)

    monkeypatch.setattr(adapter.core, "_run_milp", time_limit)
    with pytest.raises(adapter.SourceSolveError) as failure:
        adapter.fit_fixed_source_supports(scenarios, supports, source_inputs=inputs)
    assert count == 1
    assert len(failure.value.diagnostics_history) == 1
