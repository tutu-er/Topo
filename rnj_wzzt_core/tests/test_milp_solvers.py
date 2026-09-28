"""Backend parity, independent tiny LP enumeration, and status boundaries."""
import builtins
from dataclasses import replace
import sys

import numpy as np
import pytest

from rnj_wzzt.estimation import laminar_l1_milp as core
from rnj_wzzt.estimation.gurobi_milp import _normalized_status
from test_independent_milp_oracle import (
    _direct_mae, _extension_oracle, _lp_oracle, _scenarios,
)


@pytest.fixture
def gurobi():
    # Missing optional dependency skips real solves. An installed but unusable
    # license fails the solve instead of silently disguising a test failure.
    return pytest.importorskip("gurobipy")


def _solve(variables, rows, **kwargs):
    return core._run_milp(variables, rows, **{
        "time_limit": 20, "mip_rel_gap": 0, "presolve": True,
        "disp": False, "solver": "gurobi", **kwargs,
    })


def test_optional_dependency_is_required_only_when_requested(monkeypatch):
    original_import = builtins.__import__

    def without_gurobi(name, *args, **kwargs):
        if name == "gurobipy":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_gurobi)
    scenarios = _scenarios(1, ((0,),), (.3,), (.2,))
    options = dict(r_upper_bound=2, x_upper_bound=2)
    solution = core.solve_fixed_support_l1(scenarios, ((0,),), **options)
    assert solution.diagnostics.solver == "highs"
    with pytest.raises(ImportError, match="requires gurobipy"):
        core.solve_fixed_support_l1(scenarios, ((0,),), solver="gurobi", **options)


@pytest.mark.parametrize("solve", [core.solve_fixed_support_l1,
                                 core.solve_best_laminar_extension_l1])
def test_unknown_solver_is_rejected(solve):
    with pytest.raises(ValueError, match="solver must"):
        solve([], (), r_upper_bound=2, x_upper_bound=2, solver="typo")


@pytest.mark.parametrize("raw,expected", [
    (2, 0), (3, 2), (5, 3), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1),
    (15, 1), (16, 1), (17, 1), (1, 4), (4, 4), (6, 4), (12, 4),
    (13, 4), (14, 4), (999, 4),
])
def test_native_status_cannot_be_mistaken_for_a_scipy_certificate(raw, expected):
    status = _normalized_status(raw)
    assert status == expected
    diagnostic = core.SolverDiagnostics(
        status=status, success=status == 0, message="test", objective=1.,
        dual_bound=1., mip_gap=0., node_count=0, runtime_seconds=0.,
        variable_count=1, binary_variable_count=1, constraint_count=1,
        solver="gurobi", raw_status=raw,
    )
    assert core.solver_diagnostics_prove_optimality(diagnostic) == (raw == 2)
    if raw == 2:
        assert not core.solver_diagnostics_prove_optimality(
            replace(diagnostic, dual_bound=.9, mip_gap=.1))


def test_cli_and_pipeline_forward_solver_choice(monkeypatch, tmp_path):
    from rnj_wzzt import cli, pipeline

    captured = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda *a, **kw: captured.update(kw))
    monkeypatch.setattr(sys, "argv", ["run.py", "--milp-solver", "gurobi"])
    cli.main()
    assert captured["milp_solver"] == "gurobi"
    captured.clear()
    cli.advanced_main(runner=lambda *a, **kw: captured.update(kw))
    assert captured["milp_solver"] == "gurobi"
    with pytest.raises(ValueError, match="solver must"):
        pipeline.run(tmp_path / "invalid", milp_solver="typo")
    assert not (tmp_path / "invalid").exists()
    with pytest.raises(ValueError, match="solver must"):
        core.fit_laminar_l1_sensitivity([], solver="typo")


def test_sparse_rows_free_variables_and_integer_domains(gurobi):
    v, a = core._VariableBuilder(), core._ConstraintBuilder()
    v.add("free", 1, lower=-np.inf, objective=1)
    v.add("binary", 1, upper=1, objective=-10, integral=True)
    v.add("integer", 1, upper=4, objective=1, integral=True)
    v.add("positive", 1, objective=.1)
    a.add({0: 1}, lower=-3, upper=-2)
    a.add({1: 2}, upper=1.5)
    a.add({2: 1}, lower=1.2)
    a.add({0: 1, 1: 1, 3: 1}, lower=-1, upper=-1)
    a.add({})  # Entirely unbounded row imposes no restriction.
    result, diagnostic = _solve(v, a)
    assert result.x == pytest.approx([-3, 0, 2, 2], abs=1e-8)
    assert result.fun == pytest.approx(-.8, abs=1e-8)
    assert diagnostic.raw_status == gurobi.GRB.OPTIMAL
    assert diagnostic.dual_bound == pytest.approx(result.fun, abs=1e-8)
    assert diagnostic.constraint_count == a.size
    assert core.solver_diagnostics_prove_optimality(diagnostic)


@pytest.mark.parametrize("integral", [False, True])
def test_infeasible_model_does_not_access_missing_incumbent(gurobi, integral):
    v, a = core._VariableBuilder(), core._ConstraintBuilder()
    v.add("v", 1, upper=1, integral=integral)
    a.add({0: 1}, lower=2)
    result, diagnostic = _solve(v, a)
    assert result.x is None and result.fun is None
    assert diagnostic.status == 2 and diagnostic.raw_status == gurobi.GRB.INFEASIBLE
    assert not core.solver_diagnostics_prove_optimality(diagnostic)


def test_unbounded_lp_is_not_reported_as_infeasible(gurobi):
    v, a = core._VariableBuilder(), core._ConstraintBuilder()
    v.add("v", 1, objective=-1)
    result, diagnostic = _solve(v, a)
    assert diagnostic.status == 3 and diagnostic.raw_status == gurobi.GRB.UNBOUNDED
    assert diagnostic.dual_bound is None
    assert not core.solver_diagnostics_prove_optimality(diagnostic)


def test_real_time_limit_returns_uncertified_result(gurobi):
    v, a = core._VariableBuilder(), core._ConstraintBuilder()
    v.add("v", 30, upper=1, objective=-1, integral=True)
    a.add({i: i + 1 for i in range(30)}, upper=115.5)
    # A tiny positive budget can finish this model before the next clock check.
    # Zero deterministically exercises the low-level adapter's timeout branch;
    # public fitting APIs still require a strictly positive time limit.
    result, diagnostic = _solve(v, a, time_limit=0)
    assert diagnostic.status == 1 and diagnostic.raw_status == gurobi.GRB.TIME_LIMIT
    assert not core.solver_diagnostics_prove_optimality(diagnostic)
    assert result.x is None or np.isfinite(result.x).all()


@pytest.mark.parametrize("mode", ["normal", "collinear", "no_q"])
@pytest.mark.parametrize("noise", [0., .03])
def test_fixed_lp_matches_independent_residual_split_oracle(gurobi, mode, noise):
    family = ((0,), (1,), (2,), (0, 1))
    scenarios = _scenarios(3, family, (.2, .3, .4, .6), (.1, .2, .1, .4),
                          mode=mode, noise=noise, sizes=(9, 13))
    solved = core.solve_fixed_support_l1(
        scenarios, family, r_upper_bound=2, x_upper_bound=2, solver="gurobi")
    expected = _lp_oracle(scenarios, family)
    assert solved.objective == pytest.approx(expected, abs=5e-8)
    assert _direct_mae(scenarios, family, solved) == pytest.approx(expected, abs=5e-8)
    assert solved.diagnostics.solver == "gurobi"


@pytest.mark.parametrize("mode", ["normal", "collinear"])
@pytest.mark.parametrize("noise", [0., .03])
def test_extension_matches_independent_enumeration_and_highs(gurobi, mode, noise):
    initial = ((0,), (1,), (2,))
    family = (*initial, (1, 2))
    scenarios = _scenarios(3, family, (.2, .3, .4, .6), (.1, .2, .1, .4), noise=noise, mode=mode)
    options = dict(r_upper_bound=2, x_upper_bound=2)
    expected = _extension_oracle(scenarios, initial)
    highs = core.solve_best_laminar_extension_l1(scenarios, initial, **options)
    solved = core.solve_best_laminar_extension_l1(scenarios, initial, solver="gurobi", **options)
    assert core.solver_diagnostics_prove_optimality(solved.diagnostics)
    assert solved.objective == pytest.approx(min(expected.values()), abs=5e-8)
    assert expected[solved.support] == pytest.approx(solved.objective, abs=5e-8)
    assert solved.objective == pytest.approx(highs.objective, abs=5e-8)
    assert _direct_mae(scenarios, (*initial, solved.support), solved) == pytest.approx(
        solved.objective, abs=5e-8)


def test_full_path_and_saturated_support_domain_use_gurobi(gurobi):
    initial = ((0,), (1,), (2,))
    family = (*initial, (0, 1))
    scenarios = _scenarios(3, family, (.2, .3, .4, .6), (.1, .2, .1, .4))
    options = dict(initial_supports=initial, r_upper_bound=2, x_upper_bound=2,
                   solver="gurobi", validation_scenarios=scenarios)
    result = core.fit_laminar_l1_sensitivity(scenarios, max_atoms=4, **options)
    assert result.train_mae < 1e-8
    assert result.summary()["solver"] == "gurobi"
    for point in result.path:
        assert point.solver.solver == "gurobi"
        assert point.train_mae == pytest.approx(_lp_oracle(scenarios, point.support_indices), abs=5e-8)
    saturated = _scenarios(1, ((0,),), (.2,), (.1,))
    exhausted = core.solve_best_laminar_extension_l1(
        saturated, ((0,),), r_upper_bound=2, x_upper_bound=2, solver="gurobi")
    assert exhausted.support is None
    assert exhausted.diagnostics.raw_status == gurobi.GRB.INFEASIBLE


def test_unproven_gurobi_incumbent_is_not_an_exact_extension(gurobi, monkeypatch):
    from rnj_wzzt.estimation import gurobi_milp as backend

    actual = backend.solve_gurobi_milp

    def limited(*args, **kwargs):
        result = actual(*args, **kwargs)
        result.status, result.raw_status, result.success = 1, gurobi.GRB.TIME_LIMIT, False
        result.mip_dual_bound, result.mip_gap = 0., 1.
        return result

    monkeypatch.setattr(backend, "solve_gurobi_milp", limited)
    scenarios = _scenarios(2, ((0, 1),), (.3,), (.2,), noise=.03)
    extension = core.solve_best_laminar_extension_l1(
        scenarios, (), r_upper_bound=2, x_upper_bound=2, solver="gurobi")
    assert extension.objective is not None
    assert extension.support is None
    assert extension.r_values.size == 0
