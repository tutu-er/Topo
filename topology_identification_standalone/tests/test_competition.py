"""Frozen top-two ranking and native-bound contradiction checks."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from algorithm import SearchConfig, _close_native, _output
from competition import competition
from data import Observation
from optimization import predict
from trees import basis


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
        self.assertEqual(answer["secondary_index"], 2)
        self.assertEqual(answer["distinct_topologies"], 2)
        self.assertEqual(answer["only_secondary"], [[102, 103]])
        self.assertEqual(answer["global_identification"], "not_established")
        self.assertEqual(json.dumps(result, sort_keys=True), before)

    def test_same_family_different_coefficients_does_not_create_an_alternative(self):
        obs = Observation(np.eye(3), np.zeros((3, 3)), np.eye(3), (1, 2, 3))
        answer = competition(library(obs, [point(3), point(3, singleton=1.2)]), obs)
        self.assertIsNone(answer["secondary_index"])
        self.assertEqual(answer["observed_stability"], "no_distinct_saved_alternative")
        self.assertEqual(answer["statistical_ambiguity"], "not_calibrated")

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
