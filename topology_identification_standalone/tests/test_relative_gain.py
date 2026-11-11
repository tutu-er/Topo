"""Relative-J acceptance algebra; hand-made points, no QP or MIQP solves."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm import SearchConfig, _accept
from optimization import criterion


N = 5
BASIS = [*((i,) for i in range(N)), tuple(range(N))]
A, B, PARENT = (0, 1), (2, 3), (0, 1, 2, 3)


def point(supports=(), *, loss=1., weights=None):
    """loss is normalized training MSE, not the count-regularized score."""
    family = [*BASIS, *supports]
    values = weights or {}
    pairs = [values.get(tuple(C), (.1, .05)) for C in family]
    return dict(supports=[list(C) for C in family], r=[r for r, _ in pairs],
                x=[x for _, x in pairs], objective=float(loss))


def accept(old, new, config):
    fitter = SimpleNamespace(data=SimpleNamespace(n=N))
    return _accept(old, new, fitter, config, stage="E", solver={})


class RelativeGainTests(unittest.TestCase):
    def test_default_zero_kappa_matches_original_absolute_gate(self):
        config = SearchConfig()
        self.assertEqual(config.relative_gain, 0.)
        old = point((A,))
        for decrease in (0., .5*config.gain_tol, 2*config.gain_tol, .01):
            new = point((A,), loss=1.-decrease)
            chosen, event = accept(old, new, config)
            gain = criterion(old, N, config.alpha)-criterion(new, N, config.alpha)
            self.assertEqual(event["accepted"], gain > config.gain_tol)
            self.assertIs(chosen, new if event["accepted"] else old)
            self.assertEqual(event["required_gain"], config.gain_tol)
            self.assertEqual(event["numerical_threshold"], config.gain_tol)
            self.assertEqual(event["relative_gain"], 0.)

    def test_relative_gate_has_a_strict_boundary(self):
        # Binary-representable scores make equality an exact floating-point
        # equality, avoiding a rounding-based pseudo-test around .003.
        config = SearchConfig(relative_gain=.125)
        old = point((A,), loss=1.-config.alpha)
        self.assertEqual(criterion(old, N, config.alpha), 1.)
        for score, expected in ((.875, False), (.875-2**-20, True), (.875+2**-20, False)):
            new = point((A,), loss=score-config.alpha)
            self.assertEqual(criterion(new, N, config.alpha), score)
            _, event = accept(old, new, config)
            self.assertEqual(event["required_gain"], .125)
            self.assertEqual(event["accepted"], expected)

    def test_numeric_floor_and_relative_floor_use_max_not_sum(self):
        config = SearchConfig(relative_gain=.125, gain_tol=.25)
        old = point((A,), loss=1.-config.alpha)
        new = point((A,), loss=.6875-config.alpha)
        _, event = accept(old, new, config)
        self.assertEqual(event["required_gain"], .25)
        self.assertTrue(event["accepted"])
        # gain=.3125 clears max(.125,.25), but not their sum=.375.

    def test_net_add_delete_and_same_count_follow_the_same_equation(self):
        config = SearchConfig(relative_gain=.001)
        cases = [((A,), (A, PARENT)), ((A, B), (B,)), ((A,), ((1, 2),))]
        reference = 2.7
        for before, after in cases:
            old = point(before, loss=1.)
            delta_k = len(after)-len(before)
            required = max(config.gain_tol, config.relative_gain*criterion(old, N, config.alpha))
            for margin, expected in ((1e-6, True), (-1e-6, False)):
                new = point(after, loss=old["objective"]-config.alpha*delta_k-required-margin)
                _, event = accept(old, new, config)
                physical_improvement = reference*(old["objective"]-new["objective"])
                physical_requirement = reference*(config.alpha*delta_k+required)
                self.assertEqual(physical_improvement > physical_requirement, expected)
                self.assertEqual(event["accepted"], expected)
                self.assertAlmostEqual(event["required_gain"], required, places=15)

    def test_invalid_kappa_is_rejected_by_config_validation(self):
        for value in (-.001, 1., 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(kappa=value), self.assertRaises(ValueError):
                SearchConfig(relative_gain=value).validate()
        for value in (0., .001, .003, .999999):
            SearchConfig(relative_gain=value).validate()

    def test_numerical_zero_deletion_bypasses_only_the_relative_floor(self):
        config = SearchConfig(relative_gain=.003)
        zero_old = point((A,), loss=1., weights={A: (0., 0.)})
        new = point((), loss=1.)
        chosen, zero_event = accept(zero_old, new, config)
        self.assertTrue(zero_event["accepted"])
        self.assertIs(chosen, new)
        self.assertEqual(zero_event["required_gain"], config.gain_tol)
        self.assertEqual(zero_event["relative_gain"], config.relative_gain)
        self.assertLess(zero_event["gain"], config.relative_gain*zero_event["before_score"])
        nonzero_old = point((A,), loss=1.)
        chosen, normal_event = accept(nonzero_old, new, config)
        self.assertFalse(normal_event["accepted"])
        self.assertIs(chosen, nonzero_old)
        self.assertNotEqual(zero_event["acceptance_rule"], normal_event["acceptance_rule"])
        self.assertEqual(normal_event["required_gain"], config.relative_gain*normal_event["before_score"])

    def test_one_active_RX_channel_prevents_zero_bypass(self):
        config = SearchConfig(relative_gain=.003)
        for pair in ((0., .1), (.1, 0.), (1.0001*config.zero_tol, 0.)):
            old = point((A,), weights={A: pair})
            _, event = accept(old, point(()), config)
            self.assertFalse(event["accepted"])
            self.assertGreater(event["required_gain"], config.gain_tol)
        old = point((A,), weights={A: (config.zero_tol, .5*config.zero_tol)})
        _, event = accept(old, point(()), config)
        self.assertTrue(event["accepted"])
        self.assertEqual(event["required_gain"], config.gain_tol)

    def test_zero_bypass_requires_no_additions_and_nonworsening_fit(self):
        config = SearchConfig(relative_gain=.003)
        old = point((A,), weights={A: (0., 0.)})
        # Adding another support is a structural proposal even when the removed
        # old support was zero. It must not borrow the cleanup exception.
        replacement = point((B,), loss=.999, weights={B: (0., 0.)})
        _, event = accept(old, replacement, config)
        self.assertFalse(event["accepted"])
        self.assertGreater(event["required_gain"], config.gain_tol)
        _, allowed = accept(old, point((), loss=1.+.5*config.gain_tol), config)
        self.assertTrue(allowed["accepted"])
        _, worsened = accept(old, point((), loss=1.+2*config.gain_tol), config)
        self.assertFalse(worsened["accepted"])
        self.assertGreater(worsened["required_gain"], config.gain_tol)

    def test_every_removed_support_must_be_numerically_zero(self):
        config = SearchConfig(relative_gain=.01)
        mixed = point((A, B), weights={A: (0., 0.), B: (.1, 0.)})
        _, event = accept(mixed, point(()), config)
        self.assertFalse(event["accepted"])
        self.assertGreater(event["required_gain"], config.gain_tol)
        zeros = point((A, B), weights={A: (0., 0.), B: (0., 0.)})
        _, event = accept(zeros, point(()), config)
        self.assertTrue(event["accepted"])
        self.assertEqual(event["required_gain"], config.gain_tol)

    def test_zero_bypass_requires_unchanged_active_topology(self):
        config = SearchConfig(relative_gain=.003)
        cases = (
            # A retained numerical-zero support becomes a genuine active clade.
            (point((A, B), weights={A: (0., 0.), B: (0., 0.)}),
             point((A,), weights={A: (.1, 0.)})),
            # A retained active support becomes numerical zero after the refit.
            (point((A, B), weights={A: (.1, 0.), B: (0., 0.)}),
             point((A,), weights={A: (0., 0.)})),
        )
        for old, new in cases:
            with self.subTest(old_weights=old["r"], new_weights=new["r"]):
                chosen, event = accept(old, new, config)
                self.assertFalse(event["accepted"])
                self.assertIs(chosen, old)
                self.assertEqual(event["acceptance_rule"], "relative_regularized_training_gain")
                self.assertGreater(event["required_gain"], config.gain_tol)


if __name__ == "__main__":
    unittest.main()
