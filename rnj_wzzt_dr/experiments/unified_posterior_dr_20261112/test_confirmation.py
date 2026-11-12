"""Fee-matched original rotation, fixed-scale scoring, and payment semantics."""
import unittest
import numpy as np

from ...dr.pipeline import load_resources
from ...observations import Observation
from .confirmation import rotation, score_actual
from .physical import execute


class ConfirmationTests(unittest.TestCase):
    def test_rotation_uses_original_devices_and_matches_nominal_fee(self):
        labels = tuple(range(101, 116))
        original, _, _ = load_resources("paper15", labels=labels)
        full, _ = rotation(original, 1e20)
        desired = .2*original.fees(full)
        delivery, receipt = rotation(original, desired)
        self.assertTrue(receipt["matched"])
        self.assertTrue(original.audit(delivery)["passed"])
        self.assertLessEqual(abs(original.fees(delivery)/desired-1), .01)

    def test_zero_execution_uses_original_meter_law_and_no_extra_payment(self):
        original, _, _ = load_resources("paper15", labels=tuple(range(101, 116)))
        p = np.full((96, 15), .01); q = p*.2; v = np.ones_like(p); root = np.ones(96)
        background = dict(p=p, q=q, root=root, voltage=v, labels=tuple(range(101, 116)),
            network=dict(base_kw=1000.), ac=dict(iterations=1),
            metadata=dict(pq_sigma=.005, voltage_sigma=.0005), noise=[np.zeros_like(p)]*3+[np.zeros(96)])
        observed, summary, private = execute(None, background, original, np.zeros((original.R, 96)), np.ones((original.R, 96)))
        np.testing.assert_array_equal(observed.p, p)
        np.testing.assert_array_equal(observed.y, np.zeros_like(p))
        self.assertEqual(summary["ac_day_calls"], 0)
        self.assertEqual(summary["contract_payment"], 0.)
        self.assertFalse(summary["wear_proxy_is_additional_payment"])

    def test_actual_score_retains_frozen_variance_and_accumulates_prefix(self):
        score = dict(sigma2=.25, parameters=[dict(H=[[1., 0.]], history_sse=1.), dict(H=[[2., 0.]], history_sse=2.)],
            models=[dict(topology_id="A", active_clades=[]), dict(topology_id="B", active_clades=[])])
        observation = Observation([[1.]], [[0.]], [[1.]], (1,))
        from .posterior import log_weights
        result = score_actual(score, observation, "A", "B")
        self.assertAlmostEqual(result["p_true"], 1/(1+np.exp(-4)), places=13)
        self.assertTrue(result["decisions"]["0.9"]["correct_unique"])
        np.testing.assert_allclose(result["prefix_log_probability"][-1], log_weights([1., 3.], .25))


if __name__ == "__main__":
    unittest.main()
