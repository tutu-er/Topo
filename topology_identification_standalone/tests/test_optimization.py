"""Independent small-domain checks of the bounded QP and structural MIQPs."""
from __future__ import annotations

from itertools import combinations
from time import perf_counter
import unittest

import numpy as np
from scipy.optimize import lsq_linear

from algorithm import SearchConfig, _clean
from data import Observation
from optimization import QPFitter, StructuralSolver, criterion
from trees import basis, canonical


def proper_subsets(n):
    return tuple(C for size in range(2, n) for C in combinations(range(n), size))


def laminar(family):
    sets = list(map(set, family))
    return not any(A & B and A-B and B-A for i, A in enumerate(sets) for B in sets[i+1:])


def design(data, family):
    """Build columns through explicit rank-one matrices, independently of fitter."""
    columns = []
    for values in (data.p, data.q):
        for support in family:
            z = np.zeros(data.n)
            z[list(support)] = 1.
            columns.append((np.outer(z, z)@values.T).T.ravel())
    return np.column_stack(columns)


def oracle(data, family, *, reference=1., alpha=.003):
    family = tuple(sorted(set(tuple(C) for C in family), key=lambda C: (len(C), C)))
    F = design(data, family)
    solved = lsq_linear(F, data.y.ravel(), bounds=(0., 2.), method="bvls", tol=1e-13, max_iter=10000)
    if not solved.success:
        raise AssertionError(solved.message)
    mse = float(np.mean((F@solved.x-data.y.ravel())**2))
    return mse/reference+alpha*sum(1 < len(C) < data.n for C in family), mse


def synthetic(n, *, pairs=(), seed=None):
    if seed is None:
        p, q = np.eye(n), np.zeros((n, n))
    else:
        rng = np.random.default_rng(seed)
        p, q = rng.normal(size=(16, n)), rng.normal(size=(16, n))
    R, X = np.eye(n), np.zeros((n, n))
    for support, r, x in pairs:
        z = np.array([i in support for i in range(n)], float)
        R += r*np.outer(z, z); X += x*np.outer(z, z)
    return Observation(p, q, p@R.T+q@X.T, tuple(range(101, 101+n)))


class ContinuousOptimizationTests(unittest.TestCase):
    def test_bounded_QP_normalization_and_invalid_family(self):
        data = Observation(np.eye(3), np.zeros((3, 3)), 3*np.eye(3), (1, 2, 3))
        fitter = QPFitter(data, reference_mse=.25, bound=2.)
        fitted = fitter.fit([(0,), (1,), (2,)])
        self.assertTrue(np.allclose(fitted["r"], 2., atol=1e-12))
        self.assertTrue(np.allclose(fitted["x"], 0., atol=1e-12))
        self.assertAlmostEqual(fitted["mse"], 1/3, places=12)
        self.assertAlmostEqual(fitted["objective"], (1/3)/.25, places=12)
        self.assertLessEqual(fitter.records[-1]["certificate_gap"], 1e-10)
        with self.assertRaises(ValueError):
            fitter.fit([*basis(3), (0, 1), (1, 2)])
        with self.assertRaises(ValueError):
            QPFitter(data, reference_mse=0.)

    def test_numerical_zero_release_preserves_basis_and_protected_supports(self):
        data, config = synthetic(4), SearchConfig()
        fitter = QPFitter(data)
        current = fitter.fit([*basis(4), (0, 1), (2, 3)])
        closed, events = _clean(fitter, current, config, protected=((0, 1),))
        remaining = set(map(tuple, closed["supports"]))
        self.assertTrue(set(basis(4)) | {(0, 1)} <= remaining)
        self.assertNotIn((2, 3), remaining)
        self.assertTrue(any(row["accepted"] for row in events))
        self.assertLessEqual(abs(closed["mse"]-current["mse"]), config.gain_tol)
        with self.assertRaises(ValueError):
            _clean(fitter, fitter.fit(basis(4)), config, protected=((0, 1),))


class StructuralOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import gurobipy as gp
        except ImportError:
            raise unittest.SkipTest("MIQP tests require gurobipy; continuous tests still run")
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start(); env.dispose()
        except gp.GurobiError as exc:
            raise unittest.SkipTest(f"Gurobi license unavailable (error {exc.errno}); no MIQP evidence produced")

    def solver(self, data):
        fitter = QPFitter(data)
        config = SearchConfig(mip_gap=1e-8, mip_abs_gap=1e-10)
        native = StructuralSolver(fitter, config)
        self.addCleanup(native.close)
        return fitter, native, config

    def verify_native(self, data, raw, evidence, expected_J, *, alpha=.003):
        self.assertIsNotNone(raw)
        self.assertEqual(evidence["status"], 2)
        self.assertLessEqual(evidence["gap"], 1e-7)
        self.assertTrue(laminar(raw["supports"]))
        theta = np.r_[raw["r"], raw["x"]]
        self.assertGreaterEqual(float(theta.min()), -1e-12)
        self.assertLessEqual(float(theta.max()), 2.+1e-12)
        F = design(data, raw["supports"])
        replayed = float(np.mean((F@theta-data.y.ravel())**2))
        replayed += alpha*sum(1 < len(C) < data.n for C in raw["supports"])
        self.assertAlmostEqual(replayed, evidence["objective"], delta=1e-7)
        self.assertAlmostEqual(replayed, expected_J, delta=1e-7)
        self.assertLessEqual(evidence["bound"], expected_J+1e-7)
        self.assertLessEqual(expected_J, evidence["objective"]+1e-7)
        for key in ("constraint_violation", "bound_violation", "integrality_violation"):
            self.assertLessEqual(evidence[key], 1e-7)

    def test_joint_selection_replaces_crossing_wrong_pair(self):
        data = synthetic(3, pairs=(((1, 2), 1., 0.),))
        fitter, native, config = self.solver(data)
        old = fitter.fit([*basis(3), (0, 1)])
        # With squared loss the free full-set weight is 1/3; the six
        # off-diagonal residuals give MSE 4/27 (different from the L1 toy).
        self.assertAlmostEqual(old["mse"], 4/27, places=10)
        raw, evidence = native.select([*basis(3), *proper_subsets(3)], basis(3), old,
                                      deadline=perf_counter()+15.)
        self.verify_native(data, raw, evidence, .003)
        self.assertIn((1, 2), set(map(tuple, raw["supports"])))
        self.assertNotIn((0, 1), set(map(tuple, raw["supports"])))

    def test_joint_selection_matches_all_four_terminal_laminar_families(self):
        data = synthetic(4, pairs=(((0, 1), .5, .2), ((0, 1, 2), .4, .3)), seed=42)
        candidates = proper_subsets(4)
        self.assertEqual(len(candidates), 10)
        values = []
        for mask in range(1 << len(candidates)):
            selected = [C for k, C in enumerate(candidates) if mask & (1 << k)]
            if laminar(selected):
                values.append(oracle(data, [*basis(4), *selected])[0])
        self.assertEqual(len(values), 26)
        fitter, native, config = self.solver(data)
        raw, evidence = native.select([*basis(4), *candidates], basis(4), fitter.fit(basis(4)),
                                      deadline=perf_counter()+15.)
        self.verify_native(data, raw, evidence, min(values))

    def test_unknown_W_support_matches_exhaustive_compatible_extensions(self):
        data = synthetic(4, pairs=(((0, 1), .5, 0.), ((2, 3), .7, 0.)))
        fitter, native, config = self.solver(data)
        family = canonical([*basis(4), (0, 1)])
        current = fitter.fit(family)
        compatible_extensions = [C for C in proper_subsets(4) if C not in family and laminar([*family, C])]
        choices = [oracle(data, family)[0], *[oracle(data, [*family, C])[0] for C in compatible_extensions]]
        # Only this inferior candidate is supplied as a warm suggestion. The
        # MIQP must discover the better (2,3) connection outside that shortlist.
        suggested = (0, 1, 2)
        raw, evidence = native.extend(current, suggested, current, deadline=perf_counter()+15.)
        self.verify_native(data, raw, evidence, min(choices))
        self.assertIn((2, 3), set(map(tuple, raw["supports"])))
        self.assertNotIn(suggested, set(map(tuple, raw["supports"])))

    def test_pure_deletion_can_increase_MSE_while_improving_regularized_J(self):
        epsilon = .1
        data = synthetic(3, pairs=(((0, 1), epsilon, 0.),))
        fitter, native, config = self.solver(data)
        old, deleted = fitter.fit([*basis(3), (0, 1)]), fitter.fit(basis(3))
        self.assertAlmostEqual(old["mse"], 0., places=12)
        self.assertAlmostEqual(deleted["mse"], 4*epsilon**2/27, places=12)
        self.assertGreater(deleted["mse"], old["mse"])
        self.assertLess(criterion(deleted, 3, config.alpha), criterion(old, 3, config.alpha))
        raw, evidence = native.select([*basis(3), (0, 1)], basis(3), old,
                                      deadline=perf_counter()+15.)
        self.verify_native(data, raw, evidence, deleted["objective"])
        self.assertNotIn((0, 1), set(map(tuple, raw["supports"])))


if __name__ == "__main__":
    unittest.main()
