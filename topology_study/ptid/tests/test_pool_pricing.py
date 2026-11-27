"""Small regressions for the scope of a pricing certificate and the lifetime of a candidate."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np

from ptid.method import identify as identification
from ptid.method import search
from ptid.method.model import Fitter, Model
from ptid.method.sets import basis, canonical, children, unions


class PoolPricingTest(unittest.TestCase):
    def test_priced_line_survives_resolve(self):
        new, alternative = (0, 1), (2, 3)
        model = SimpleNamespace(n=4, leaves=4)
        domains = []
        labelled = []

        def solve(_model, pool, starts, seconds, settings, require):
            domain = canonical(pool)
            domains.append(domain)
            chosen = (alternative,) if new in domain else ()
            return chosen, dict(domain=dict(lines=domain, require=require), tree=chosen, status=gp.GRB.TIME_LIMIT,
                                crit=2. if chosen else 4., bound=1., gap_closed=False)

        def labels(_model, tree, pool, *args):
            labelled.append(canonical(pool))
            return {}, {}

        priced = dict(certified=True, enumerated=1, by_miqp=0, unproved=0)
        st = identification.Settings(K=2, rounds=1)
        with patch.object(identification, "label_groupings", side_effect=labels), \
             patch.object(identification, "solve", side_effect=solve), \
             patch.object(identification, "best_addition", side_effect=[(new, {}), (None, priced), (None, priced)]), \
             patch.object(identification, "Fitter") as fitted:
            fitted.return_value.crit.side_effect = lambda tree: 2. if alternative in tree else 3.
            result = identification.refine(model, (), (alternative,), st)

        self.assertEqual(domains, [(alternative,), canonical([new, alternative])])
        self.assertEqual(result["tree"], (alternative,))
        self.assertEqual(result["pool"], canonical([alternative, new]))
        self.assertEqual(labelled[-1], result["pool"])
        after = result["rounds"][0]["price"]["re_solve"]
        self.assertEqual((after["domain"]["lines"], after["bound"], after["crit"]), (domains[-1], 1., 2.))
        self.assertEqual(identification.scope(result["rounds"])["restricted_solve"], "price.re_solve")

    def test_wide_node_bounds_control_single_line_certificate(self):
        model = SimpleNamespace(leaves=3, alpha=.1)
        current = 1.
        trials = [
            (gp.GRB.OPTIMAL, current-1e-5, False),  # solver optimum within its own MIPGap
            (gp.GRB.TIME_LIMIT, None, False),
            (gp.GRB.TIME_LIMIT, current-search.PRICING_TOL/2, True),
            (gp.GRB.OPTIMAL, current+model.alpha+.1, False),  # exceeds a known feasible upper bound
        ]
        for status, bound, expected in trials:
            with self.subTest(status=status, bound=bound), patch.object(search, "Fitter") as fit, \
                 patch.object(search, "node_miqp", return_value=(None, None, status, bound)):
                fit.return_value.crit.return_value = current
                line, info = search.best_addition(model, (), max_children=2)
                self.assertIsNone(line)
                self.assertEqual(info["certified"], expected)
                self.assertEqual(info["unproved"], int(not expected))
                self.assertEqual(info["wide_nodes"][0]["no_improvement"], expected)
                if bound is not None and bound > current+model.alpha:
                    self.assertIsNone(info["wide_nodes"][0]["bound"])

    def test_unproved_node_prevents_convergence(self):
        model = SimpleNamespace(n=4, leaves=4)
        for certified in (False, True):
            with self.subTest(certified=certified):
                price = dict(added=[], still_paying=None, certified=certified, re_solve=None)
                with patch.object(identification, "label_groupings", return_value=({}, {})), \
                     patch.object(identification, "solve", return_value=((), dict(status=gp.GRB.OPTIMAL, gap_closed=True))), \
                     patch.object(identification, "price", return_value=((), price)):
                    result = identification.refine(model, (), (), identification.Settings(K=2, rounds=1))
                self.assertEqual(result["stop"].startswith("converged"), certified)
                self.assertEqual(identification.scope(result["rounds"])["single_line"],
                                 "complete" if certified else "incomplete")

    def test_node_bound_matches_all_unions(self):
        rng = np.random.default_rng(17)
        p, q = rng.normal(0, .01, (48, 4)), rng.normal(0, .005, (48, 4))
        y = .02*p+.03*q+rng.normal(0, .0001, p.shape)
        model = Model(p, q, y, np.ones(48))
        kids = children(basis(4), 4)[tuple(range(4))]
        options = unions(kids)
        fit = Fitter(model, options)
        exact = min(fit.crit([line]) for line in options)
        line, _, _, lower = search.node_miqp(model, (), kids, 10., return_bound=True)
        self.assertIsNotNone(line)
        self.assertIsNotNone(lower)
        self.assertLessEqual(lower, exact+search.PRICING_TOL)
        self.assertLessEqual(exact, fit.crit([line])+search.PRICING_TOL)


if __name__ == "__main__":
    unittest.main()
