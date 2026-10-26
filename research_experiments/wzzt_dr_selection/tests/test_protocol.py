"""Protocol checks: information isolation, feasible responses and resumable evidence.

These tests exercise the independent three-terminal reference only.  They do not
claim coverage of historical coefficient faces or production WZZT behavior.
"""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from research_experiments.wzzt_dr_selection import experiment as exp
from research_experiments.wzzt_dr_selection import exact_patterns as exact


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((exp.ROOT / "protocol.json").read_text(encoding="utf-8"))
        cls.specifications = exp.cases()

    def test_design_has_no_evaluator_or_test_rng_dependency(self):
        _, history, forecast, _ = self.specifications[0]
        latent = np.array(list(product((0.05, 0.95), repeat=3)))
        before = [a.copy() for a in (history.P, history.Q, history.Y)]
        baseline = exp.clean(exp.design(history, forecast, latent))
        # Access through any of these routes would cross the design boundary.
        with patch.object(exp, "evaluate", side_effect=AssertionError("test evaluator accessed")), \
             patch.object(exp, "cases", side_effect=AssertionError("truth catalog accessed")), \
             patch.object(exp, "Evaluator", side_effect=AssertionError("truth constructed")), \
             patch.object(exp.np.random, "default_rng", side_effect=AssertionError("hidden RNG accessed")):
            isolated = exp.clean(exp.design(history, forecast, latent))
        self.assertEqual(baseline, isolated)
        self.assertFalse(isolated["pool"]["parameter_coverage"])
        for original, current in zip(before, (history.P, history.Q, history.Y)):
            assert_allclose(current, original, rtol=0, atol=0)

    def test_all_response_paths_preserve_energy_power_and_payment_cap(self):
        cfg = self.config["a2"]
        for menu, mode, latent in product(exp.MENUS,
                                         ("independent", "common", "ignores_price_common"),
                                         product((0., .1, .5, .9, 1.), repeat=3)):
            P, Q, cost, pattern = exp.response(menu, np.array(latent), mode)
            flexible = P - cfg["background_power_per_node"]
            assert_allclose(flexible.sum(axis=1), cfg["per_node_energy"])
            self.assertGreaterEqual(float(flexible.min()), 0.)
            self.assertLessEqual(float(flexible.max()), cfg["per_node_power_cap"])
            assert_allclose(Q, 0.)
            self.assertGreaterEqual(cost, 0.)
            self.assertLessEqual(cost, cfg["pathwise_payment_cap"] + 1e-12)
            assert_allclose(flexible[:, 0], pattern)
            if mode != "independent":
                self.assertEqual(len(set(pattern)), 1)

    def test_misspecified_response_really_ignores_price(self):
        for first in (.05, .45, .55, .95):
            outcomes = [exp.response(m, np.array([first, .2, .8]), "ignores_price_common")
                        for m in exp.MENUS]
            for P, Q, _, pattern in outcomes[1:]:
                assert_allclose(P, outcomes[0][0])
                assert_allclose(Q, outcomes[0][1])
                self.assertEqual(pattern, outcomes[0][3])

    def test_summary_retains_failures_and_fixed_denominators(self):
        designs = {"fixture": {"choices": {"metric": 0}, "pool": {}, "scores": []}}
        common = {"case": "fixture", "menu": exp.MENUS[0]["name"]}
        rows = [common | {"replicate": 0, "outcome": "correct", "strict_correct": True, "cost": .2},
                common | {"replicate": 1, "outcome": "failed", "strict_correct": False, "cost": None},
                common | {"replicate": 2, "outcome": "stopped", "strict_correct": False, "cost": .1}]
        summary = exp.summarize(rows, designs, expected=4)
        first = summary["cases"]["fixture"]["menus"][0]
        self.assertEqual(first["completed"], 3)
        self.assertEqual(first["denominator"], 4)
        self.assertEqual(sum(first["counts"].values()), 3)
        self.assertEqual(first["counts"]["failed"], 1)
        self.assertEqual(first["strict_correct"], 1)
        self.assertEqual(first["cost_observations"], 2)
        self.assertEqual(summary["planned_independent_test_realizations_per_case"], 4)
        self.assertEqual(summary["cases"]["fixture"]["observed_replicates"], 3)
        self.assertEqual(summary["cases"]["fixture"]["complete_paired_replicates"], 0)
        self.assertTrue(summary["not_a_confidence_audit"])

    def test_partial_summary_distinguishes_observed_and_complete_pairs(self):
        designs = {"fixture": {"choices": {"metric": 0}, "pool": {}, "scores": []}}
        rows = [{"case": "fixture", "replicate": 0, "menu": menu["name"],
                 "outcome": "failed", "strict_correct": False, "cost": None}
                for menu in exp.MENUS]
        rows.extend({"case": "fixture", "replicate": 1, "menu": menu["name"],
                     "outcome": "stopped", "strict_correct": False, "cost": .1}
                    for menu in exp.MENUS[:2])
        summary = exp.summarize(rows, designs, expected=4)
        case = summary["cases"]["fixture"]
        self.assertEqual(summary["completed_updates"], 8)
        self.assertEqual(summary["planned_updates"], 24)
        self.assertEqual(summary["planned_independent_test_realizations_per_case"], 4)
        self.assertEqual(case["observed_replicates"], 2)
        # A failed solve is still a completed attempt; it must not silently vanish.
        self.assertEqual(case["complete_paired_replicates"], 1)
        self.assertEqual(case["menus"][0]["counts"]["failed"], 1)
        self.assertEqual(case["menus"][0]["denominator"], 4)

    def resume_fixture(self):
        designs = {name: {"fixture_design": name} for name, *_ in self.specifications}
        name = self.specifications[0][0]
        latent = np.random.default_rng(self.config["a2"]["seed_test"]).random(
            (self.config["a2"]["test_draws"], 3))[0]
        row = {"case": name, "replicate": 0, "menu": exp.MENUS[0]["name"],
               "latent": latent.tolist(), "design_fingerprint": exp.digest(designs[name]),
               "outcome": "failed", "correct": False, "strict_correct": False,
               "cost": None, "error": "intentional retained failure"}
        return designs, row

    def test_resume_accepts_and_retains_failed_rows(self):
        designs, row = self.resume_fixture()
        done = exp.validate_resume_rows([row], designs, self.specifications, self.config)
        self.assertEqual(done, {(row["case"], 0, row["menu"])})

    def test_resume_rejects_duplicate_or_foreign_keys(self):
        designs, row = self.resume_fixture()
        invalids = [row | {"case": "unknown"}, row | {"menu": "unknown"},
                    row | {"replicate": -1},
                    row | {"replicate": self.config["a2"]["test_draws"]},
                    row | {"replicate": 0.0}, row | {"replicate": True}]
        for changed in invalids:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                exp.validate_resume_rows([changed], designs, self.specifications, self.config)
        with self.assertRaises(ValueError):
            exp.validate_resume_rows([row, deepcopy(row)], designs, self.specifications, self.config)

    def test_resume_rejects_changed_design_or_future_realization(self):
        designs, row = self.resume_fixture()
        wrong_latent = list(row["latent"])
        wrong_latent[0] = 1. - wrong_latent[0]
        invalids = [row | {"design_fingerprint": "incorrect"},
                    row | {"latent": wrong_latent}, row | {"latent": [float("nan")] * 3},
                    row | {"latent": row["latent"][:2]}]
        for changed in invalids:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                exp.validate_resume_rows([changed], designs, self.specifications, self.config)

    def test_resume_rejects_invalid_or_inconsistent_outcomes(self):
        designs, row = self.resume_fixture()
        invalids = [row | {"outcome": "missing"}, row | {"correct": True},
                    row | {"strict_correct": True}, row | {"outcome": "correct"}]
        for changed in invalids:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                exp.validate_resume_rows([changed], designs, self.specifications, self.config)

    def test_resume_rejects_environment_change(self):
        # Use only a temporary A1 manifest; existing evidence is never modified.
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), stage="a1", limit=None)
            with patch.object(exp, "run_a1", return_value={"status": "test_fixture"}):
                exp.run(args)
                with patch.object(exp.platform, "python_version", return_value="changed-environment"):
                    with self.assertRaises(RuntimeError):
                        exp.run(args)


class ExactPatternTests(unittest.TestCase):
    def test_probabilities_are_normalized_and_match_response_marginals(self):
        for menu, mode in product(exp.MENUS, ("independent", "common", "ignores_price_common")):
            weights = np.array([exact.probability(p, menu, mode) for p in exact.PATTERNS])
            self.assertGreaterEqual(float(weights.min()), 0.)
            self.assertAlmostEqual(float(weights.sum()), 1.)
            marginals = weights @ np.array(exact.PATTERNS)
            target = np.array(menu["q"]) if mode == "independent" else np.full(
                3, np.mean(menu["q"]) if mode == "common" else .5)
            assert_allclose(marginals, target)
            if mode != "independent":
                for p, weight in zip(exact.PATTERNS, weights):
                    if len(set(p)) > 1:
                        self.assertEqual(weight, 0.)

    def test_exact_patterns_and_costs_match_the_evaluator_response(self):
        for menu, pattern in product(exp.MENUS, exact.PATTERNS):
            latent = np.array([0. if bit else 1. for bit in pattern])
            _, _, cost, actual_pattern = exp.response(menu, latent, "independent")
            self.assertEqual(actual_pattern, pattern)
            self.assertAlmostEqual(exact.pattern_cost(pattern, menu), cost)
        for menu in exp.MENUS:
            expected = sum(exact.probability(p, menu, "independent") * exact.pattern_cost(p, menu)
                           for p in exact.PATTERNS)
            self.assertAlmostEqual(expected, 0. if menu["preferred"] is None else .27)

    def test_exact_integration_keeps_wrong_and_stopped_outcomes(self):
        result = exact.run()
        self.assertEqual(result["independent_mc_samples"], 0)
        expected = {"no_reward": .75, "uniform_early": .27, "uniform_late": .27,
                    "node0_early": .91, "node1_early": .91, "node2_early": .91}
        for state in ("omitted_lines", "weak_line"):
            for row in result["cases"][state]["menus"]:
                self.assertAlmostEqual(row["actual_strict_correct_probability"], expected[row["menu"]])
                self.assertAlmostEqual(sum(row["actual_outcome_probabilities"].values()), 1.)
        # Success in the weak-line state means selection of a true block, not necessarily the weak block.
        for outcome in result["cases"]["weak_line"]["pattern_outcomes"]:
            if outcome["outcome"] == "correct":
                self.assertEqual(outcome["chosen"], [0, 1, 2])
        for row in result["cases"]["common_mode"]["menus"]:
            self.assertAlmostEqual(row["actual_outcome_probabilities"]["stopped"], 1.)
        for row in result["cases"]["forecast_mismatch"]["menus"]:
            self.assertAlmostEqual(row["actual_outcome_probabilities"]["wrong"], 1.)
        omitted = {r["menu"]: r for r in result["cases"]["omitted_lines"]["menus"]}
        # A demonstrated ranking reversal must remain visible even when the
        # screen's selected menu happens to lie on the same optimal plateau.
        self.assertLess(omitted["node2_early"]["frozen_screen"], omitted["no_reward"]["frozen_screen"])
        self.assertGreater(omitted["node2_early"]["actual_strict_correct_probability"],
                           omitted["no_reward"]["actual_strict_correct_probability"])


if __name__ == "__main__":
    unittest.main()
