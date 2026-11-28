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

    def test_verdicts(self):
        scan = dict(decision="present", line=[0, 1, 2, 3], gain=9.)
        self.assertEqual(node_probe.verdict(scan, (0, 1, 2, 3)), "right")
        self.assertEqual(node_probe.verdict(scan, None), "wrong")
        self.assertEqual(node_probe.leader(dict(scan, decision="open", gain=.5), (0, 1, 2, 3)), "right")
        self.assertEqual(node_probe.leader(dict(scan, gain=-.5), None), "right")


if __name__ == "__main__":
    unittest.main()
