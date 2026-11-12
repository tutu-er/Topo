"""The convex device restriction has no integers and replays exact original U."""
import unittest
from time import perf_counter
import numpy as np

from ...dr.pipeline import load_resources
from .convex_steps import ConvexDeviceStep, supporting_plane_start
from .evidence import Evidence


class Tests(unittest.TestCase):
    def test_all_device_types_preserve_original_feasibility_and_fee(self):
        import gurobipy as gp
        original, problem, _ = load_resources("paper15", labels=tuple(range(101, 116)))
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0); env.start()
            for r in range(4):
                step = ConvexDeviceStep(problem, r, np.zeros((1, 24)), 1., env)
                try:
                    delivery, report = step.solve(.1, np.sin(np.arange(96)/12), 10., perf_counter()+2.)
                    self.assertTrue(report["optimal"])
                    self.assertEqual(report["binaries"], 0)
                    raw = original.scale[r]*np.repeat(step.requests[0], 4)
                    np.testing.assert_allclose(delivery, raw, atol=1e-7)
                    assembled = np.zeros((original.R, 96)); assembled[r] = delivery
                    self.assertTrue(original.audit(assembled)["passed"])
                    self.assertAlmostEqual(step.resource["fee"].getValue(), original.fees(assembled), places=7)
                finally:
                    step.close()

    def test_zero_supporting_plane_yields_original_feasible_gain(self):
        original, problem, _ = load_resources("paper15", labels=tuple(range(101, 116)))
        effect = np.zeros((1, 1, original.R)); effect[0, 0, 1] = 1.
        evidence = Evidence(-np.ones((1, 96, 1)), effect, np.ones(1), .7)
        seed = supporting_plane_start(problem, original, evidence, 1., seconds=5.)
        self.assertTrue(seed["qualified"])
        self.assertTrue(original.audit(seed["mean_u"])["passed"])
        self.assertGreaterEqual(evidence.gains(seed["mean_u"])[0], .7)


if __name__ == "__main__":
    unittest.main()
