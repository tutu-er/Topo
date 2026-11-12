"""Frozen top-two ranking and native-bound contradiction checks."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from algorithm import SearchConfig, _close_native, _output
from competition import ambiguity_gate, competition
from data import Observation
from optimization import predict
from trees import basis
from tools.analyze_strict96 import verify_competition


def point(n, *, singleton=1., extra=None, weight=0.):
    family = list(basis(n)) + ([extra] if extra is not None else [])
    return dict(supports=[list(C) for C in family],
                r=[singleton if len(C) == 1 else weight if C == extra else 0. for C in family],
                x=[0.]*len(family))


def library(observations, points):
    config = SearchConfig()
    selected, diagnostic, bank = _output(points, observations, observations, 1., config)
    return dict(method="101", selected=selected, selected_mae_view=diagnostic,
                frozen_candidates=bank, path=points, events=[], reference_mse=1.,
                config=dict(validation_tie=config.validation_tie, zero_tol=config.zero_tol),
                stop_reason="no_gain_or_unresolved")


class CompetitionTests(unittest.TestCase):
    def test_zero_auxiliary_clade_is_not_a_second_topology(self):
        obs = Observation(np.eye(3), np.zeros((3, 3)), np.eye(3), (101, 102, 103))
        result = library(obs, [point(3, extra=(0, 1)), point(3), point(3, extra=(1, 2), weight=.3)])
        before = json.dumps(result, sort_keys=True)
        answer = competition(result, obs)
        self.assertEqual(answer["primary_index"], result["selected"]["index"])
        self.assertEqual(answer["primary_index"], 1)
        self.assertIsNone(answer["secondary_index"])
        self.assertEqual(answer["best_distinct_index"], 2)
        self.assertEqual(answer["output_count"], 1)
        self.assertEqual(answer["distinct_topologies"], 2)
        self.assertEqual(answer["only_secondary"], [])
        self.assertEqual(answer["decision"], "single_separated")
        self.assertEqual(answer["global_identification"], "not_established")
        self.assertEqual(json.dumps(result, sort_keys=True), before)

    def test_only_close_alternatives_are_returned_and_widening_preserves_primary(self):
        obs = Observation(np.eye(3), np.zeros((3, 3)), 2*np.eye(3), (1, 2, 3))
        result = library(obs, [point(3), point(3, extra=(0, 1), weight=.01)])
        snapshot = json.dumps(result, sort_keys=True)
        narrow = competition(result, obs, kappa=.003)
        wide = competition(result, obs, kappa=.02)
        self.assertEqual(narrow["output_count"], 1)
        self.assertEqual(wide["output_count"], 2)
        self.assertEqual(narrow["primary_index"], wide["primary_index"])
        self.assertEqual(wide["dr_candidate_indices"], [1, 0])
        self.assertEqual(len(wide["models"]), 2)
        self.assertEqual(narrow["dr_candidate_indices"], [])
        self.assertAlmostEqual(wide["ambiguity_gate"]["relative_mse_gap"], .0132)
        self.assertEqual(json.dumps(result, sort_keys=True), snapshot)
        result["config"]["ambiguity_kappa"] = .02
        self.assertEqual(competition(result, obs)["output_count"], 2)

    def test_same_family_different_coefficients_does_not_create_an_alternative(self):
        obs = Observation(np.eye(3), np.zeros((3, 3)), np.eye(3), (1, 2, 3))
        answer = competition(library(obs, [point(3), point(3, singleton=1.2)]), obs)
        self.assertIsNone(answer["secondary_index"])
        self.assertEqual(answer["observed_stability"], "no_distinct_saved_alternative")
        self.assertEqual(answer["statistical_ambiguity"], "not_calibrated")
        self.assertEqual(answer["decision"], "single_no_saved_alternative")

    def test_explicit_block_reversal_is_descriptive_not_a_probability(self):
        p = np.array([[1., 0., 0.], [0., 0., 1.]])
        zero = Observation(p, np.zeros_like(p), np.zeros_like(p), (1, 2, 3))
        first, second = point(3, singleton=0., extra=(0, 1), weight=1.), point(3, singleton=0., extra=(1, 2), weight=1.)
        y = np.vstack((predict(zero, first)[0], predict(zero, second)[1]))
        obs = Observation(p, np.zeros_like(p), y, (1, 2, 3))
        result = library(obs, [first, second])
        blocks = [dict(name="first", start=0, stop=1), dict(name="second", start=1, stop=2)]
        answer = competition(result, obs, blocks=blocks)
        self.assertGreater(answer["blocks"][0]["delta_mse"], 0.)
        self.assertLess(answer["blocks"][1]["delta_mse"], 0.)
        self.assertEqual(answer["observed_stability"], "observed_blocks_not_strictly_separated")
        self.assertEqual(answer["statistical_ambiguity"], "not_calibrated")
        single = competition(result, obs, blocks=[dict(name="one", start=0, stop=2)])
        self.assertEqual(single["observed_stability"], "insufficient_blocks_for_stability_assessment")

    def test_incomplete_or_overlapping_blocks_are_rejected(self):
        obs = Observation(np.eye(3), np.zeros((3, 3)), np.eye(3), (1, 2, 3))
        result = library(obs, [point(3), point(3, extra=(0, 1), weight=.1)])
        for blocks in ([dict(name="gap", start=0, stop=2)],
                       [dict(name="a", start=0, stop=2), dict(name="b", start=1, stop=3)]):
            with self.assertRaises(ValueError):
                competition(result, obs, blocks=blocks)

    def test_scores_and_labels_must_belong_to_the_supplied_observations(self):
        obs = Observation(np.eye(3), np.zeros((3, 3)), np.eye(3), (1, 2, 3))
        result = library(obs, [point(3), point(3, extra=(0, 1), weight=.1)])
        stale = copy.deepcopy(result); stale["frozen_candidates"][1]["validation_scaled_mse"] += .1
        with self.assertRaises(ValueError):
            competition(stale, obs)
        wrong = copy.deepcopy(result); wrong["frozen_candidates"][0]["labels"] = [3, 2, 1]
        with self.assertRaises(ValueError):
            competition(wrong, obs)


class AmbiguityGateTests(unittest.TestCase):
    def test_independent_replay_supports_legacy_and_conditional_outputs(self):
        first = dict(index=0, selection_scaled_mse=.9)
        other = dict(index=1, selection_scaled_mse=1.)
        config = dict(ambiguity_kappa=.003, validation_tie=1e-12)
        self.assertIs(verify_competition(dict(secondary_index=1), first, other, config), other)
        saved = dict(selection_rule="relative_selection_mse_ambiguity_v1", ambiguity_kappa=.003,
                     best_distinct_index=1, secondary_index=None, output_count=1,
                     models=[dict(index=0)], dr_candidate_indices=[], ambiguity_gate=dict(ambiguous=False))
        self.assertIsNone(verify_competition(saved, first, other, config))
        tampered = dict(saved, secondary_index=1)
        with self.assertRaises(ValueError):
            verify_competition(tampered, first, other, config)

    def test_exact_boundary_is_kept_and_larger_separation_is_rejected(self):
        self.assertTrue(ambiguity_gate(.875, 1., kappa=.125, tie=0.)["ambiguous"])
        self.assertFalse(ambiguity_gate(.874, 1., kappa=.125, tie=0.)["ambiguous"])

    def test_zero_losses_and_numerical_sparse_tie(self):
        self.assertTrue(ambiguity_gate(0., 0., kappa=0.)["ambiguous"])
        self.assertFalse(ambiguity_gate(0., 1., kappa=.999)["ambiguous"])
        tied = ambiguity_gate(1.+1e-13, 1., kappa=0.)
        self.assertTrue(tied["ambiguous"])
        self.assertEqual(tied["relative_mse_gap"], 0.)
        self.assertTrue(ambiguity_gate(1.+1e-12, 1., kappa=0.)["ambiguous"])
        self.assertTrue(ambiguity_gate(1., 1.+1e-12, kappa=0.)["ambiguous"])

    def test_widening_is_monotone_and_training_gate_is_independent(self):
        self.assertEqual([ambiguity_gate(.99, 1., kappa=k)["ambiguous"]
                          for k in (0., .003, .02, .1)], [False, False, True, True])
        config = SearchConfig(ambiguity_kappa=.1)
        config.validate()
        self.assertEqual(config.relative_gain, 0.)

    def test_invalid_parameters_are_rejected(self):
        for k in (-.1, 1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ambiguity_gate(1., 1., kappa=k)
            with self.assertRaises(ValueError):
                SearchConfig(ambiguity_kappa=k).validate()
        for pair in ((-1., 0.), (float("nan"), 1.), (2., 1.)):
            with self.assertRaises(ValueError):
                ambiguity_gate(*pair)


class NativeClosureTests(unittest.TestCase):
    def test_feasible_fixed_family_fit_can_contradict_a_reported_native_bound(self):
        fitted = dict(point(3), objective=.8)
        fitter = SimpleNamespace(data=SimpleNamespace(n=3), fit=lambda family: fitted)
        for lower, expected in ((1., False), (.7, True)):
            native = dict(bound=lower, objective=1.1, status=2, gap=0.)
            closed, _ = _close_native(fitted, fitter, SearchConfig(), native)
            self.assertIs(closed, fitted)
            self.assertEqual(native["bound_consistent_with_fixed_family_qp"], expected)
            self.assertEqual(native["bound"], lower)  # Preserve original evidence.
            self.assertEqual(native["status"], 2)
            if not expected:
                self.assertIn("bound_warning", native)


if __name__ == "__main__":
    unittest.main()
