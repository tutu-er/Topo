import numpy as np
import pytest

from research_experiments.local_rebuild.matched import FiniteLP, all_hierarchies, gate, single_closure
from research_experiments.local_rebuild.model import fit_fixed, solve_rebuild, unions
from research_experiments.local_rebuild.run import scenes


def test_hierarchy_domain_counts():
    assert len(all_hierarchies(4)) == 26
    assert len(all_hierarchies(5)) == 236


def test_lp_primal_dual_and_old_fixed_solver_agree():
    rng = np.random.default_rng(4401)
    p, q = rng.random((2, 12, 4))
    y = rng.random((12, 4))
    prepared = scenes(p, q, y)
    mandatory = [(i,) for i in range(4)] + [tuple(range(4))]
    pool = [*mandatory, *(s for _, s in unions(tuple((i,) for i in range(4))))]
    bank = FiniteLP(prepared, pool, mandatory=mandatory)
    for extra in ([], [(0, 1)], [(0, 2), (1, 3)], [(0, 1), (0, 1, 2)]):
        point = bank.fit(extra)
        old = fit_fixed(prepared, [*mandatory, *extra])
        assert point["objective"] == pytest.approx(old["objective"], abs=1e-8)
        certificate = bank.solves[point["solve_index"]]
        assert certificate["valid"] and certificate["absolute_gap"] <= 1e-8


def test_stationary_requires_a_complete_sweep_and_gate_is_shared():
    p = np.eye(3)
    z = np.array([0., 1., 1.])
    prepared = scenes(p, p * 0., p + np.outer(z, z))
    mandatory = [(i,) for i in range(3)] + [tuple(range(3))]
    pool = [*mandatory, (0, 1), (0, 2), (1, 2)]
    bank = FiniteLP(prepared, pool, mandatory=mandatory)
    result = single_closure(bank, [*mandatory, (0, 1)])
    assert result["stationary"] and result["sweeps"][-1]["complete"]
    assert result["point"]["objective"] == pytest.approx(0., abs=1e-9)
    assert (1, 2) in result["point"]["supports"]
    assert result["sweeps"][0]["threshold"] == pytest.approx(gate(result["path"][0]["objective"]))
    stopped = single_closure(bank, mandatory, seconds=0.)
    assert not stopped["stationary"] and stopped["stop_reason"] == "wall_limit"


def test_known_incumbent_is_embedded_in_rebuild_constraints():
    p = np.eye(3)
    z = np.array([0., 1., 1.])
    prepared = scenes(p, p * 0., p + np.outer(z, z))
    family = ((0,), (1,), (2,), (0, 1), (0, 1, 2))
    old = fit_fixed(prepared, family)
    result = solve_rebuild(prepared, family, range(3), ((0,), (1,), (2,)),
                           incumbent=old, presolve=False)
    assert result["known_feasible_witness"]["maximum_violation"] <= 1e-9
    assert result["known_feasible_witness"]["objective"] == pytest.approx(old["objective"])
    assert result["point"]["objective"] < old["objective"]
    invalid = {**old, "r": [100.] * len(old["r"])}
    with pytest.raises(ValueError, match="not feasible"):
        solve_rebuild(prepared, family, range(3), ((0,), (1,), (2,)), incumbent=invalid)


def test_solver_bound_contradicting_witness_is_rejected(monkeypatch):
    from research_experiments.local_rebuild.run import core
    from types import SimpleNamespace
    p = np.eye(3)
    prepared = scenes(p, p * 0., p)
    family = ((0,), (1,), (2,), (0, 1, 2))
    old = fit_fixed(prepared, family)
    diagnostics = core.SolverDiagnostics(0, True, "optimal", .1, .1, 0., 0, .1, 1, 1, 1)
    monkeypatch.setattr(core,"_run_milp",lambda *a,**k:(SimpleNamespace(x=None,fun=.1),diagnostics))
    result = solve_rebuild(prepared, family, range(3), ((0,), (1,), (2,)), incumbent=old)
    assert not result["certified"]
    assert result["certificate_error"] == "dual_bound_exceeds_verified_feasible_objective"
