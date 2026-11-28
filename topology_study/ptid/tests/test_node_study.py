"""Checks of the node MIQP study code (experiments/node_miqp). Run from topology_study:
    python -m unittest ptid.tests.test_node_study
About a minute; needs Gurobi, no case files."""
import math
import unittest

import numpy as np

from ptid.method import probe
from ptid.method.search import best_addition, node_miqp, node_price
from ptid.runs import node_probe
from ptid.runs.node_study import (BASE_KW, Feeder, Simulator, enumerate_node, model_of, node_scan, score, truth,
                                  widest)


def small(seed=3, r=.012, hours=72):
    f = Feeder(blocks=6, per=2, shared=(0, 2, 3), r_shared=r, excitation=.32, hours=hours, seed=seed)
    data = Simulator(f).history()
    model, _ = model_of(data)
    return f, data, model


class NodeStudyTest(unittest.TestCase):
    def test_truth_and_network(self):
        f = Feeder(blocks=5, per=2, shared=(1, 3), r_shared=.01)
        true, flat, target = truth(f)
        self.assertEqual(target, (2, 3, 6, 7))
        self.assertEqual(len(flat), 5)
        self.assertIn(target, true)

    def test_paired_noise(self):
        f = Feeder(blocks=4, per=2, hours=24, seed=5)
        sim = Simulator(f)
        a, b = sim.hours(24, 2), sim.hours(24, 2)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        order = np.zeros(f.n)
        order[3] = 3/BASE_KW
        c = sim.hours(24, 2, order)
        gain = sim.meters.base.p_gain[3]
        # The block is measured by its meter; every other meter's power reading is unchanged (same noise draw).
        np.testing.assert_allclose(c["P"][:, 3]-a["P"][:, 3], order[3]*(1+gain), rtol=.05)
        np.testing.assert_array_equal(np.delete(c["P"], 3, 1), np.delete(a["P"], 3, 1))
        self.assertTrue(np.all(c["Y"][:, 3] > a["Y"][:, 3]))       # more load, larger drop at that meter

    def test_node_flip_test_matches_enumeration(self):
        f, data, model = small()
        _, flat, target = truth(f)
        scan, ref = node_scan(model, flat, 60.), enumerate_node(model, flat)
        self.assertEqual(scan["line"], ref["line"])
        self.assertAlmostEqual(scan["gain"], ref["gain"], delta=1e-3)
        self.assertEqual(scan["runner"], ref["runner"])
        self.assertAlmostEqual(scan["runner_gap"], ref["runner_gap"], delta=1e-3)
        self.assertGreaterEqual(scan["runner_gap_low"], ref["runner_gap"]-1e-3*(1+ref["runner_gap"]))

    def test_warm_start_changes_nothing(self):
        f, data, model = small()
        _, flat, _ = truth(f)
        cold, warm = node_scan(model, flat, 60.), node_scan(model, flat, 60., warm=True)
        for key in ("line", "runner", "decision"):
            self.assertEqual(cold[key], warm[key])
        self.assertAlmostEqual(cold["gain"], warm["gain"], delta=1e-6)

    def test_exclude_cut_forbids_the_union(self):
        f, data, model = small()
        _, flat, _ = truth(f)
        _, kids = widest(model, flat)
        line, _, _ = node_miqp(model, flat, kids, 30.)
        chosen = {a for a, kid in enumerate(kids) if set(kid) <= set(line)}
        other, _, _ = node_miqp(model, flat, kids, 30., exclude=[chosen])
        self.assertNotEqual(tuple(line), tuple(other))

    def test_one_home_order_is_the_drafts_block(self):
        f, data, model = small(hours=48)
        _, flat, target = truth(f)
        trees = {"flat": flat, "S": (*flat, target)}
        fc = probe.forecast(data, data["P"], data["Q"], 48)
        pred = {k: probe.prediction(model, t, fc) for k, t in trees.items()}
        H = {k: score(model, t) for k, t in trees.items()}
        lead = min(H, key=H.get)
        rivals = {k: pred[k] for k in trees if k != lead}
        now = {k: dict(high=H[k]-H[lead], kind="inside") for k in rivals}
        up = down = np.full(f.n, 3/BASE_KW)
        ref = probe.choose_block(model, pred[lead], rivals, now, fc, up, down, 20., BASE_KW, node_probe.PS, 4.6)
        J, u, cost, homes = node_probe.choose_order(model, pred[lead], rivals, now, fc, up, down, 20., 1)
        self.assertAlmostEqual(J, ref[0], places=9)
        self.assertAlmostEqual(cost, ref[2], places=9)
        if ref[1] is None:
            self.assertIsNone(u)
        else:
            np.testing.assert_allclose(u, ref[1])

    def test_node_price(self):
        f, data, model = small()
        _, flat, target = truth(f)
        self.assertAlmostEqual(node_price(model, 6, 1.)*model.N/2, math.log(2**6-6-2))
        self.assertEqual(node_price(model, 6, 0.), 0.)
        plain, info = best_addition(model, flat, 9)
        same, info0 = best_addition(model, flat, 9, node_gamma=0.)
        self.assertEqual(plain, same)
        self.assertAlmostEqual(info["gain"], info0["gain"])
        if plain is not None:
            # a price above the line's gain removes it; the reported gain is net of the extra price
            g = info["gain"]*model.N/2
            none, _ = best_addition(model, flat, 9, node_gamma=(g+1.)/math.log(2**6-6-2))
            self.assertIsNone(none)
            kept, info1 = best_addition(model, flat, 9, node_gamma=.5*g/math.log(2**6-6-2))
            self.assertEqual(kept, plain)
            self.assertAlmostEqual(info1["gain"]*model.N/2, .5*g, places=6)

    def test_multi_line_node_miqp_matches_enumeration(self):
        from itertools import combinations
        from ptid.method.model import Fitter
        from ptid.method.search import node_miqp_multi
        from ptid.runs.node_study import root_feeder
        f = root_feeder(2, 3, 1, 2, .006, hours=72, seed=4)            # 7 laterals, two root lines of 3
        data = Simulator(f).history()
        model, _ = model_of(data)
        _, flat, _ = truth(f)
        _, kids = widest(model, flat)
        c = len(kids)
        U = lambda S: tuple(sorted(i for a in S for i in kids[a]))
        crit = lambda t: Fitter(model, t).crit(t)
        singles = [frozenset(S) for k in range(2, c) for S in combinations(range(c), k)]
        configs = [[S] for S in singles]+[[S, T] for S, T in combinations(singles, 2) if not S & T]
        best = min(configs, key=lambda cfg: crit([*flat, *(U(S) for S in cfg)]))
        found, value, status, bound = node_miqp_multi(model, flat, kids, 60., 2)
        self.assertEqual(status, 2)
        self.assertEqual(sorted(found), sorted(U(S) for S in best))
        self.assertAlmostEqual(value, crit([*flat, *found]), delta=1e-6*(1+abs(value)))

    def test_pair_test_matches_enumeration(self):
        from itertools import combinations
        from ptid.method.model import Fitter
        from ptid.runs.node_study import node_label, root_feeder, root_lines
        f = root_feeder(2, 3, 1, 2, .006, hours=72, seed=4)            # 7 laterals, two root lines of 3
        data = Simulator(f).history()
        model, _ = model_of(data)
        _, flat, _ = truth(f)
        C, D = root_lines(f)
        tree = [*flat, C, D]
        label = node_label(model, tree, C, 60., pairs=[C, D])
        self.assertEqual(len(label["pairs"]), 1)
        pair = label["pairs"][0]
        _, kids = widest(model, flat)
        U = lambda S: tuple(sorted(i for a in S for i in kids[a]))
        crit = lambda t: Fitter(model, t).crit(t)
        singles = [frozenset(S) for k in range(2, len(kids)) for S in combinations(range(len(kids)), k)]
        configs = [[S] for S in singles]+[[S, T] for S, T in combinations(singles, 2) if not S & T]
        others = [crit([*flat, *(U(S) for S in cfg)]) for cfg in configs if sorted(map(U, cfg)) != sorted([C, D])]
        h = crit(tree)
        self.assertAlmostEqual(pair["gap_other"], (min(others)-h)*model.N/2, delta=1e-4)
        self.assertLessEqual(pair["gap_low"], pair["gap_other"]+1e-6)
        self.assertAlmostEqual(pair["gap_none"], (crit(flat)-h)*model.N/2, delta=1e-6)

    def test_chain_root_lines(self):
        from ptid.runs.node_study import lateral_parent, network, root_feeder, root_lines
        f = root_feeder(1, 4, 2, 2, .006, inner=2, seed=3)          # a root line over 4 laterals, one inside over 2
        (outer, _, p0), (inner, _, p1) = f.roots
        self.assertEqual((len(outer), len(inner), p0, p1), (4, 2, -1, 0))
        self.assertTrue(set(inner) < set(outer))
        self.assertEqual({lateral_parent(f, j) for j in inner}, {201})
        self.assertEqual({lateral_parent(f, j) for j in set(outer)-set(inner)}, {200})
        true, flat, _ = truth(f)
        self.assertEqual(len(true), len(flat)+2)
        self.assertTrue(all(rl in true for rl in root_lines(f)))
        edges = {(e["from_bus"], e["to_bus"]) for e in network(f)[0]["branches"]}
        self.assertTrue({(1, 200), (200, 201)} <= edges)

    def test_verdicts(self):
        scan = dict(decision="present", line=[0, 1, 2, 3], gain=9.)
        self.assertEqual(node_probe.verdict(scan, (0, 1, 2, 3)), "right")
        self.assertEqual(node_probe.verdict(scan, None), "wrong")
        self.assertEqual(node_probe.leader(dict(scan, decision="open", gain=.5), (0, 1, 2, 3)), "right")
        self.assertEqual(node_probe.leader(dict(scan, gain=-.5), None), "right")


if __name__ == "__main__":
    unittest.main()
