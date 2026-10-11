"""Independent checks of historical projection, QP bounds and design scope.

The explicit-history comparison keeps source variables; it does not implement
the pairwise interval elimination being tested. No generation truth is passed
to the design API. This is a small mechanism audit, not a performance suite.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parent))
import design_v7 as study


_AUDIT = []
_PAIR = {"cherry12": (0, 1), "cherry13": (0, 2), "cherry23": (1, 2)}


def _matrix(theta, pair):
    a, b, *c = theta
    z = np.zeros(3)
    z[list(pair)] = 1
    return a * np.ones((3, 3)) + b * np.outer(z, z) + np.diag(c)


def _history():
    """Independent synthetic history with nonconstant P and heterogeneous boxes."""
    p = np.array([[.2, .2, .2], [.25, .2, .2], [.2, .24, .2], [.2, .2, .26]])
    theta = np.array([.024, .009, .04, .07, .05])
    nominal = np.array([.12, .11, .14, .1])
    delta = np.array([.01, -.02, .015, -.01])
    y = p @ _matrix(theta, (0, 1)).T + theta[0] * (nominal + delta)[:, None]
    eps = np.tile([.00031, .00035, .00033], (4, 1))
    return dict(P_total=p, Y_drop=y, source_nominal=nominal,
                source_error_bound=np.array([.05, .06, .04, .08]),
                drop_error_bound=eps, batch_id=np.zeros(4, dtype=int))


def _explicit_history_lp(history, candidate, direction):
    """Optimize a physical theta objective with explicit per-time common w."""
    p = history["P_total"]
    y = history["Y_drop"]
    n = len(p)
    z = np.zeros(3)
    z[list(_PAIR[candidate])] = 1
    basis = [np.ones((3, 3)), np.outer(z, z)]
    basis.extend(np.diag(np.eye(3)[i]) for i in range(3))
    rows, rhs = [], []
    for t in range(n):
        design = np.column_stack([mat @ p[t] for mat in basis])
        design[:, 0] += history["source_nominal"][t]
        for i in range(3):
            row = np.zeros(5+n)
            row[:5] = design[i]
            row[5+t] = 1
            rows.extend((row, -row))
            eps = history["drop_error_bound"][t, i]
            rhs.extend((y[t, i]+eps, -y[t, i]+eps))
        for sign in (-1., 1.):
            row = np.zeros(5+n)
            row[0] = -history["source_error_bound"][t]
            row[5+t] = sign
            rows.append(row)
            rhs.append(0.)
    return linprog(np.r_[direction, np.zeros(n)], A_ub=np.array(rows),
                   b_ub=np.array(rhs), bounds=list(study.BOX)+[(None, None)]*n,
                   method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                             "dual_feasibility_tolerance": 1e-9})


@pytest.fixture(scope="session", autouse=True)
def _save_audit():
    yield
    out = Path(__file__).with_name("results_v7") / "independent_checks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(
        implementation_sha256=hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Independent explicit-source LP, analytic convex QP, projection and information-boundary checks",
        records=_AUDIT), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def test_history_projection_matches_explicit_source_lp():
    history = _history()
    directions = np.r_[np.eye(5), -np.eye(5), [[.4, -.7, .2, .9, -.3]]]
    for h in _PAIR:
        poly = study.build_history_polytope(history, h)
        widths = np.array(list(map(float, poly.width)))
        offset = np.array(list(map(float, poly.offset)))
        for g in directions:
            original = _explicit_history_lp(history, h, g)
            projected = linprog(g*widths, A_ub=poly.a, b_ub=poly.b_outer,
                                bounds=[(0., 1.)]*5, method="highs",
                                options={"primal_feasibility_tolerance": 1e-9,
                                         "dual_feasibility_tolerance": 1e-9})
            assert original.success == projected.success
            assert original.success, "The declared independent example must have feasible competitors."
            delta = abs(original.fun-(projected.fun+float(g@offset)))
            assert delta < 2e-8
            _AUDIT.append(dict(check="explicit_source_projection", candidate=h,
                               direction=g.tolist(), absolute_optimum_difference=delta))


def test_qp_certificate_encloses_analytic_active_constraint_optimum():
    # Euclidean projection of (.75,.75,.5,.5,.5) onto x1+x2<=.5 is
    # (.25,.25,.5,.5,.5), with objective exactly 1/2.
    row = (F(1), F(1), F(0), F(0), F(0))
    poly = study.Polytope((row,), (F(1, 2),), np.array([list(map(float, row))]),
                         np.array([.5]), np.array([.5-2e-10]),
                         (F(0),)*5, (F(1),)*5, "analytic")
    q = study.Quadratic(tuple(tuple(F(i == j) for j in range(5)) for i in range(5)),
                        (F(3, 4), F(3, 4), F(1, 2), F(1, 2), F(1, 2)), (F(1),)*5)
    result = study.solve_qp(poly, q)
    assert result["status"] == "certified_interval"
    assert result["lower"] <= .5 <= result["upper"]
    assert result["gap"] < 1e-8
    assert np.linalg.norm(np.array(result["x"])-[.25, .25, .5, .5, .5]) < 1e-7
    _AUDIT.append(dict(check="analytic_qp", result=result, exact_optimum="1/2"))


def test_pairwise_quadratic_is_the_orthogonal_source_contrast():
    poly = study.build_history_polytope(_history(), "cherry23")
    x = np.array([.25, .3, .4, .5, .6])
    theta = poly.theta(x)
    reference = dict(candidate="cherry12", theta=[.023, .012, .045, .07, .055])
    p = np.array([[.2, .2, .2], [.5, .2, .2], [.2, .2, .2], [.2, .35, .2]])
    q = study.design_objective(poly, reference, p)
    exact, _ = q.exact_value_gradient(x)
    c = np.array([[1., -1., 0.], [1., 1., -2.]]) / np.array([[np.sqrt(2)], [np.sqrt(6)]])
    difference = _matrix(reference["theta"], (0, 1)) - _matrix(theta, (1, 2))
    direct = float(np.sum((p @ difference.T @ c.T)**2))
    assert abs(float(exact)-direct) < 1e-16
    _AUDIT.append(dict(check="orthogonal_contrast", pairwise_value=float(exact),
                       independent_matrix_value=direct))


def test_design_api_does_not_access_generator_or_truth_sidecar(monkeypatch):
    class ForbiddenTruth:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Design attempted to read simulation truth.")

        def __getitem__(self, key):
            raise AssertionError("Design attempted to read simulation truth.")

    def forbidden_generator(*args, **kwargs):
        raise AssertionError("Design attempted to generate/access future outputs.")

    history = _history()
    history["evaluation_truth"] = ForbiddenTruth()
    monkeypatch.setattr(study, "simulate", forbidden_generator)
    monkeypatch.setattr(study, "PROFILES", ForbiddenTruth())
    design = study.design_scores(history)
    assert design["local_selection"]["selected"].endswith("endpoint")
    assert len(design["representatives"]) == 3
    assert all(r.get("lower") is not None for r in design["representatives"].values())
    # Recheck the rounded physical theta actually consumed as the reference,
    # not merely its pre-conversion normalized incumbent.
    for h, ref in design["representatives"].items():
        poly = study.build_history_polytope(history, h)
        x_exact = [(F(float(t))-lo)/width for t, lo, width in
                   zip(ref["theta"], poly.offset, poly.width)]
        assert all(0 <= x <= 1 for x in x_exact)
        assert all(sum(a*x for a, x in zip(row, x_exact)) <= b
                   for row, b in zip(poly.exact_a, poly.exact_b))
    _AUDIT.append(dict(check="history_only_interface",
                       reference_selection=design["reference_selection"],
                       local_selection=design["local_selection"],
                       qp_records=len(design["qp_records"]),
                       physical_reference_feasible=True))


def test_counterexamples_disallow_unqualified_endpoint_claim():
    # A time-varying background: competing losses are 2t² and 2(1-t)².
    endpoint = min(F(0), F(2))
    constant = min(2*F(1, 2)**2, 2*(1-F(1, 2))**2)
    assert endpoint == 0 and constant == F(1, 2)
    # Static background but unequal time weights.
    endpoint_weighted = min(F(100), F(1))
    constant_weighted = F(101, 4)
    assert endpoint_weighted == 1 and constant_weighted > endpoint_weighted
    # L2 design and L-infinity execution can rank actions differently.
    assert 2**2 < 2*1.5**2 and 2 > 1.5
    _AUDIT.append(dict(check="scope_counterexamples", timevarying_endpoint=0,
                       timevarying_constant=.5, weighted_endpoint=1,
                       weighted_constant=25.25, l2_linf_order_can_reverse=True))
