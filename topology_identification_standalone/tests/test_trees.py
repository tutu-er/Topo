"""Independent geometry, exact-tree and frontier checks; no MIQP license needed."""
import unittest
from itertools import combinations
from types import SimpleNamespace

import numpy as np

from trees import (basis, canonical, children, compatible, fit_sensitivity,
                   frontier_candidates, geometry, is_laminar, max_direct_blocks,
                   nj, rnj, rnj_guidance)


def response(n, family, values):
    result = np.zeros((n, n))
    for support, value in zip(family, values):
        z = np.array([i in support for i in range(n)], float)
        result += value*np.outer(z, z)
    return result


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.n = 6
        self.family = canonical([*basis(6), (0, 1), (2, 3), (0, 1, 2, 3), (4, 5)])
        self.R = response(6, self.family, [.2+.03*i for i in range(len(self.family))])
        self.X = response(6, self.family, [.1+.02*i for i in range(len(self.family))])

    def test_hierarchy_counts_and_crossings(self):
        self.assertTrue(is_laminar(self.family))
        self.assertFalse(compatible((0, 1), (1, 2)))
        self.assertEqual(max_direct_blocks(self.family, 6), 2)
        self.assertEqual(children(self.family, 6)[tuple(range(6))], ((4, 5), (0, 1, 2, 3)))
        with self.assertRaises(ValueError):
            children([*self.family, (1, 2)], 6)

    def test_rnj_and_classical_nj_exact_additive_tree(self):
        shared, depth, _ = geometry(self.R, self.X)
        self.assertEqual(rnj(shared, depth), self.family)
        self.assertEqual(nj(shared, depth), self.family)

    def test_star_and_single_terminal(self):
        shared = np.diag([.3, .5, .7, .9]); depth = np.diag(shared)
        self.assertEqual(rnj(shared, depth), basis(4))
        self.assertEqual(nj(shared, depth), basis(4))
        self.assertEqual(rnj(np.array([[1.]]), np.array([1.])), ((0,),))
        self.assertEqual(nj(np.array([[1.]]), np.array([1.])), ((0,),))

    def test_ordered_QP_on_exact_full_rank_observations(self):
        rng = np.random.default_rng(712)
        p, q = rng.normal(size=(80, 6)), rng.normal(size=(80, 6))
        train = SimpleNamespace(p=p, q=q, y=p@self.R+q@self.X)
        R, X, diagnostics = fit_sensitivity(train)
        self.assertTrue(diagnostics["success"])
        self.assertLess(np.max(abs(R-self.R)), 1e-8)
        self.assertLess(np.max(abs(X-self.X)), 1e-8)
        for matrix in (R, X):
            self.assertGreaterEqual(matrix.min(), -1e-12)
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
            self.assertTrue((np.diag(matrix)[:, None] >= matrix-1e-10).all())
        guide = rnj_guidance(train)
        self.assertEqual(len(guide["warm_families"]), 15)
        self.assertIn(self.family, guide["warm_families"])
        self.assertEqual(set(guide["pool"]), {s for f in guide["warm_families"] for s in f})
        self.assertFalse(guide["fitted_zero_pruning"])

    def test_rank_deficient_QP_compares_predictions(self):
        rng = np.random.default_rng(713)
        p = rng.normal(size=(50, 6)); q = .2*p
        y = p@self.R+q@self.X
        R, X, info = fit_sensitivity(dict(p=p, q=q, y=y))
        self.assertTrue(info["success"])
        self.assertLess(np.mean((y-p@R-q@X)**2), 1e-10)

    def test_all_frontiers_match_independent_antichain_enumeration(self):
        n = 8; six = tuple(range(6))
        old = canonical([*basis(n), six, (0, 1, 2), (3, 4, 5)])
        domain = frontier_candidates(old, n, 5)
        expected_frontiers, expected_pool = set(), set(old)
        for parent in old:
            candidates = [s for s in old if set(s) <= set(parent)]
            for count in range(3, 6):
                for blocks in combinations(candidates, count):
                    if set().union(*map(set, blocks)) != set(parent):
                        continue
                    if any(set(a) & set(b) for a, b in combinations(blocks, 2)):
                        continue
                    expected_frontiers.add((parent, canonical(blocks)))
                    for size in range(2, count):
                        expected_pool.update(tuple(sorted(i for block in group for i in block))
                                             for group in combinations(blocks, size))
        actual = {(tuple(row["parent"]), canonical(row["blocks"])) for row in domain["frontiers"]}
        self.assertEqual(actual, expected_frontiers)
        self.assertEqual(set(domain["pool"]), expected_pool)
        self.assertIn(six, domain["released"])
        self.assertIn((0, 1, 2, 6), domain["pool"])
        self.assertIn(tuple(range(7)), domain["pool"])
        protected = frontier_candidates(old, n, 5, hard=(six,))
        self.assertIn(six, protected["keep"])
        self.assertNotIn(six, protected["released"])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            frontier_candidates(self.family, 6, block_limit=2)
        with self.assertRaises(ValueError):
            geometry(np.eye(2), np.eye(3))
        with self.assertRaises(ValueError):
            fit_sensitivity(dict(p=np.ones((2, 2)), q=np.ones((3, 2)), y=np.ones((2, 2))))


if __name__ == "__main__":
    unittest.main()
