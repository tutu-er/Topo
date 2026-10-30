from types import SimpleNamespace

import numpy as np
import pytest

from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import (
    build_model, drop_zero, fit_fixed, frontiers, scan, solve_rebuild, unions,
)


def data(p, q, y):
    return [dict(P_terminal=p, Q_terminal=q, drop_target=y)]


def example():
    p = np.eye(3)
    z = np.array([0., 1., 1.])
    scenes = data(p, p * 0., p + np.outer(z, z))
    family = ((0,), (1,), (2,), (0, 1), (0, 1, 2))
    return scenes, family


def test_counterexample_and_hard_prior():
    scenes, family = example()
    old = fit_fixed(scenes, family)
    assert old["objective"] * 9 == pytest.approx(2.)
    fine = solve_rebuild(scenes, family, (0, 1, 2), ((0,), (1,), (2,)))
    assert fine["audit"]["valid"] and fine["certified"]
    assert fine["point"]["objective"] == pytest.approx(0., abs=1e-10)
    assert (1, 2) in fine["point"]["supports"]
    hard = solve_rebuild(scenes, family, (0, 1, 2), ((0,), (1,), (2,)), hard=((0, 1),))
    assert (0, 1) in hard["point"]["supports"]
    assert hard["point"]["objective"] * 9 == pytest.approx(2.)


def test_refinement_domain_and_malformed_frontier():
    scenes, family = example()
    coarse = solve_rebuild(scenes, family, (0, 1, 2), ((0, 1), (2,)))
    fine = solve_rebuild(scenes, family, (0, 1, 2), ((0,), (1,), (2,)))
    assert fine["point"]["objective"] <= coarse["point"]["objective"]
    with pytest.raises(ValueError, match="complete subtree"):
        solve_rebuild(scenes, family, (0, 1, 2), ((0, 2), (1,)))


def test_small_exact_enumeration_with_independent_lp():
    # scipy.linprog builds a separate dense L1 LP, without core model builders.
    from scipy.optimize import linprog
    rng = np.random.default_rng(8401)
    n = 4
    true = ((0,), (1,), (2,), (3,), (0, 2), (1, 3), (0, 1, 2, 3))
    p, q = rng.normal(size=(12, n)), rng.normal(size=(12, n))
    R, X = core.build_matrices_from_atoms(n, true, rng.uniform(.1, .8, len(true)),
                                         rng.uniform(.1, .8, len(true)))
    y = p @ R.T + q @ X.T + rng.normal(scale=.02, size=p.shape)
    scenes = data(p, q, y)
    base = ((0,), (1,), (2,), (3,), (0, 1, 2, 3))
    old = (*base, (0, 1), (2, 3))
    candidates = [s for _, s in unions(((0,), (1,), (2,), (3,)))]
    best, count = np.inf, 0
    for bits in range(1 << len(candidates)):
        selected = [s for k, s in enumerate(candidates) if bits & (1 << k)]
        if not core.is_laminar_family(selected):
            continue
        family = (*base, *selected)
        z = np.array([[float(i in s) for i in range(n)] for s in family])
        f = np.column_stack([((v @ a)[:, None] * a).ravel() for v in (p, q) for a in z])
        m = y.size
        result = linprog(np.r_[np.zeros(f.shape[1]), np.ones(m) / m],
                         A_ub=np.vstack([np.c_[f, -np.eye(m)], np.c_[-f, -np.eye(m)]]),
                         b_ub=np.r_[y.ravel(), -y.ravel()],
                         bounds=[(0., 2.)] * f.shape[1] + [(0., None)] * m, method="highs")
        assert result.success
        best = min(best, result.fun)
        count += 1
    assert count == 26
    for cuts in ("none", "global", "subset"):
        fitted = solve_rebuild(scenes, old, range(n), tuple((i,) for i in range(n)), cuts=cuts)
        assert fitted["certified"] and fitted["audit"]["valid"]
        assert fitted["point"]["objective"] == pytest.approx(best, abs=1e-8)
        assert fitted["point"]["objective"] <= fit_fixed(scenes, old)["objective"] + 1e-9


def test_hierarchy_cuts_strengthen_relaxation_and_count_fixed_priors():
    scenes, family = example()
    bounds = []
    for cuts in ("none", "global", "subset"):
        model, meta = build_model(core._prepare_scenarios(scenes), family, range(3),
                                  ((0,), (1,), (2,)), cuts=cuts)
        model.variables.objective = [0.] * model.variables.size
        for i in range(meta["selection"].start, meta["selection"].stop):
            model.variables.objective[i] = -1.
        model.variables.integrality = [0] * model.variables.size
        result, _ = core._run_milp(model.variables, model.rows, time_limit=3.,
                                   mip_rel_gap=0., presolve=True, disp=False)
        bounds.append(-result.fun)
    assert bounds == pytest.approx([1.5, 1., 1.])
    hard = solve_rebuild(scenes, family, range(3), ((0,), (1,), (2,)), hard=((0, 1),))
    assert hard["cardinality_cuts"] == 1
    assert all(v <= 1e-8 for v in hard["selection_values"])


def test_all_subset_cuts_still_allow_a_fractional_odd_cycle():
    # Five crossing pair supports form an induced C5. A6 leaves the vector
    # 1/2 on that cycle feasible, whereas every integer family uses at most 2.
    n = 5
    prepared = core._prepare_scenarios(data(np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n))))
    family = tuple((i,) for i in range(n)) + (tuple(range(n)),)
    model, meta = build_model(prepared, family, range(n), tuple((i,) for i in range(n)))
    values = np.zeros(model.variables.size)
    cycle = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
    for k, s in enumerate(meta["candidates"]):
        if s in cycle:
            values[meta["selection"].start + k] = .5
    constraints = model.rows.build(model.variables.size)
    activity = constraints.A @ values
    assert np.all(activity >= constraints.lb - 1e-10)
    assert np.all(activity <= constraints.ub + 1e-10)
    assert values[meta["selection"]].sum() == 2.5
    assert max(sum(bool(bits & (1 << k)) for k in range(5)) for bits in range(32)
               if core.is_laminar_family([s for k, s in enumerate(sorted(cycle)) if bits & (1 << k)])) == 2


def test_scan_drops_zero_temporary_atoms_and_never_worsens():
    scenes, family = example()
    result = scan(scenes, family)
    assert (0, 1) in result["removed_initial_zero"]
    assert result["point"]["objective"] == pytest.approx(0., abs=1e-9)
    assert (1, 2) in result["point"]["supports"]
    objectives = [p["objective"] for p in result["path"]]
    assert all(b <= a + 1e-8 for a, b in zip(objectives, objectives[1:]))
    for v, blocks in frontiers(result["point"]["supports"], 3):
        assert set(i for b in blocks for i in b) == set(v)
    dropped = drop_zero(core._prepare_scenarios(scenes), fit_fixed(scenes, family))
    assert (0, 1) not in dropped["supports"]
    kept = drop_zero(core._prepare_scenarios(scenes), fit_fixed(scenes, family), hard=((0, 1),))
    assert (0, 1) in kept["supports"]


def test_uncertified_solve_retains_failure_without_fake_point(monkeypatch):
    scenes, family = example()
    diagnostics = core.SolverDiagnostics(1, False, "time limit", None, None, None,
                                         None, .1, 1, 1, 1)
    monkeypatch.setattr(core, "_run_milp", lambda *a, **k:
                        (SimpleNamespace(x=None, fun=None), diagnostics))
    result = solve_rebuild(scenes, family, range(3), ((0,), (1,), (2,)))
    assert not result["certified"] and result["point"] is None
    assert result["diagnostics"]["status"] == 1


@pytest.mark.parametrize("b, count", [(3, 3), (4, 10), (5, 25)])
def test_candidate_counts(b, count):
    assert len(unions(tuple((i,) for i in range(b)))) == count
