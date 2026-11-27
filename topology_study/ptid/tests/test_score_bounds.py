"""Small deterministic checks of full-pool bounds and threshold exclusions.

Run from topology_study: python -B -m ptid.tests.test_score_bounds
"""
from itertools import combinations
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np

from ptid.method.labels import DECIDED, OPEN, UNSETTLED, flip_test, label_groupings
from ptid.method.miqp import _pool_claim, tree_miqp
from ptid.method.model import Fitter, Model
from ptid.method.sets import basis, compatible


class ScoreBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(19)
        p = rng.uniform(.01, .05, (48, 4))
        q = rng.uniform(.005, .02, p.shape)
        y = .12*p+.04*q+.15*p.sum(axis=1, keepdims=True)
        y[:, :2] += .35*p[:, :2].sum(axis=1, keepdims=True)
        y += rng.normal(0., .0005, p.shape)
        cls.model = Model(p, q, y, np.ones(len(p)))
        cls.pool = ((0, 1), (1, 2), (2, 3), (0, 1, 2))
        fit = Fitter(cls.model, cls.pool)
        cls.scores = {}
        for count in range(len(cls.pool)+1):
            for tree in combinations(cls.pool, count):
                if all(compatible(a, b) for a, b in combinations(tree, 2)):
                    cls.scores[tree] = fit.crit(tree)

    def test_flat_design_at_attained_level(self):
        p = np.full((64, 4), .03125)
        y = .04+np.random.default_rng(5).normal(0., 1e-5, p.shape)
        model = Model(p, np.zeros_like(p), y, np.ones(len(p)))
        best = Fitter(model, ()).crit(())
        result = tree_miqp(model, (), 5., cutoff=best, tighten_at=best, warm=())
        self.assertIsNotNone(result["family"])
        self.assertFalse(result["threshold_excluded"])
        self.assertAlmostEqual(result["crit"], best)
        self.assertLessEqual(result["bound"], best+1e-8)

    def test_tightening_contradicted_by_warm_tree_is_skipped(self):
        tree = min(self.scores, key=self.scores.get)
        best = self.scores[tree]
        with patch("ptid.method.miqp.response_bounds", return_value=None):
            result = tree_miqp(self.model, self.pool, 5., cutoff=best, tighten_at=best,
                               warm=tree, gap=1e-8)
        self.assertFalse(result["tightened"])
        self.assertFalse(result["threshold_excluded"])
        self.assertIsNotNone(result["family"])

    def test_original_minimum_and_flip_intervals_against_enumeration(self):
        best = min(self.scores.values())
        direct = tree_miqp(self.model, self.pool, 5., gap=1e-8)
        self.assertLessEqual(direct["bound"], best+1e-6)
        self.assertAlmostEqual(direct["crit"], best, places=6)
        target = (0, 1)
        for inside in (False, True):
            flipped = {t: score for t, score in self.scores.items() if (target in t) != inside}
            optimum = min(flipped.values())
            worst = max(flipped.values())
            for tighten in (False, True):
                for cutoff, expected in ((optimum-.05, DECIDED), (optimum+.05, OPEN)):
                    with self.subTest(inside=inside, tighten=tighten, cutoff=cutoff):
                        r = flip_test(self.model, self.pool, target, cutoff, 5., inside=inside,
                                      tighten=tighten, known=worst)
                        self.assertEqual(r["label"], expected)
                        self.assertEqual(r["threshold_excluded"], expected == DECIDED)
                        if r["low"] is not None:
                            self.assertLessEqual(r["low"], optimum+1e-6)
                        if expected == OPEN:
                            self.assertLessEqual(optimum, r["high"]+1e-6)
                            self.assertEqual(target in r["rival"], not inside)

    def test_restricted_bound_and_outside_witness(self):
        # Below 10 is excluded. A raw bound of 20 for the restricted model is
        # compatible with an original feasible tree at 12 outside the level.
        protected = 10.+1.1e-8
        for status in (gp.GRB.CUTOFF, gp.GRB.TIME_LIMIT):
            result = _pool_claim(status, 20., 12., protected, 10.)
            self.assertEqual(result, (status, protected, True))
        self.assertEqual(_pool_claim(gp.GRB.INFEASIBLE, None, 10., protected, 10.),
                         (gp.GRB.NUMERIC, None, False))
        self.assertEqual(_pool_claim(gp.GRB.NUMERIC, 20., None, protected, 10.),
                         (gp.GRB.NUMERIC, None, False))
        self.assertEqual(_pool_claim(gp.GRB.TIME_LIMIT, 5., None, protected, 10.),
                         (gp.GRB.TIME_LIMIT, 5., False))

    def test_flip_incumbent_needs_a_feasible_opposite_tree(self):
        target = (0, 1)
        base = basis(self.model.leaves)
        # Last row: a checked incumbent remains a witness after a numerical solver exit.
        cases = [
            (True, (*base, (0, 2)), UNSETTLED),                 # outside the pool
            (True, (*base, (1, 2), (2, 3)), UNSETTLED),        # crossing
            (True, (*base, target), UNSETTLED),                # still present
            (False, base, UNSETTLED),                          # still absent
            (False, (*base, target), OPEN),
        ]
        threshold = max(self.scores.values())+1.
        for inside, family, expected in cases:
            with self.subTest(inside=inside, family=family):
                result = dict(status=gp.GRB.NUMERIC, seconds=0., bound=None, family=family, threshold_excluded=False)
                with patch("ptid.method.labels.tree_miqp", return_value=result):
                    r = flip_test(self.model, self.pool, target, threshold, 5., inside=inside)
                self.assertEqual(r["label"], expected)
                self.assertIsNone(r["low"])
                if expected == OPEN:
                    self.assertEqual(r["rival"], (target,))
                    self.assertAlmostEqual(r["high"], self.scores[(target,)])
                else:
                    self.assertIsNone(r["rival"])
                    self.assertIsNone(r["high"])

    def test_local_upper_bound_keeps_its_rival(self):
        target = (0, 1)
        tree = (target,)
        fit = Fitter(self.model, self.pool)
        for status, expected in ((gp.GRB.CUTOFF, DECIDED), (gp.GRB.TIME_LIMIT, UNSETTLED)):
            with self.subTest(status=status):
                excluded = status == gp.GRB.CUTOFF
                result = dict(status=status, seconds=0., bound=fit.crit(tree)+1e-8 if excluded else None,
                              family=None, threshold_excluded=excluded)
                with patch("ptid.method.labels.tree_miqp", return_value=result):
                    labels, _ = label_groupings(self.model, tree, self.pool, 0., 5.)
                row = labels[target]
                self.assertEqual(row["label"], expected)
                self.assertIsNotNone(row["rival"])
                self.assertNotIn(target, row["rival"])
                self.assertAlmostEqual(row["gap"][1], (fit.crit(row["rival"])-fit.crit(tree))*self.model.N/2)


if __name__ == "__main__":
    unittest.main()
