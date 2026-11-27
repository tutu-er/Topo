"""Regression checks for the load rival, voltage bands and target error counts."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ptid.method import probe
from ptid.runs import probe_cases


class ProbeTests(unittest.TestCase):
    def test_stationary_meter_must_satisfy_band_for_every_tree(self):
        settings = probe.ProbeSettings()
        fc = {"v0sq": 1.}
        dy = np.diag([1., 0.])
        leader = {"dy": dy, "y": np.zeros(2)}
        for drop in (-.2, .2):
            rival = {"dy": dy, "y": np.array([0., drop])}
            low, high = probe.band_interval([leader, rival], 0, 1., fc, 0., settings)
            self.assertGreater(low, high)
        low, high = probe.band_interval([leader], 0, 1., fc, 0., settings)
        self.assertEqual(low, 0.)
        self.assertAlmostEqual(high, 1.-settings.band[0]**2)

    def test_different_path_and_gap_relative_to_actual_tree(self):
        model = SimpleNamespace(n=3, N=20)
        group, carrying, all_meters = (0, 1), (0, 1, 3), (0, 1, 2)
        tree = (group, carrying)     # both lines project to the same meter group
        scores = {tree: 1., (carrying,): .5, (group,): .9, (group, all_meters): 1.2}
        with patch.object(probe, "Fitter") as fitter:
            fitter.return_value.crit.side_effect = scores.__getitem__
            gap, rival = probe.placement_rival(model, tree)
        self.assertEqual(rival, (group,))
        self.assertAlmostEqual(gap, -1.)   # the cheaper tested tree with the same path was excluded

    def test_pcc_and_common_trunk_are_distinct_placements(self):
        model = SimpleNamespace(n=3, N=20)
        pcc = ((0, 1, 2),)
        with patch.object(probe, "Fitter") as fitter:
            fitter.return_value.crit.side_effect = {(): 1., pcc: 2.}.__getitem__
            place = probe.placement_rival(model, ())
        self.assertEqual(place, (10., pcc))
        result = probe.states(model, (), [], place, ["place"], 4.6)
        self.assertEqual(result["place"]["label"], probe.DECIDED)

    def test_position_truth_is_unassessed_and_excluded_from_group_errors(self):
        group = (1, 2)
        record = dict(n=3, source=None, truth=[])
        gate = dict(targets={str(group): [probe.DECIDED, 8.], "place": [probe.DECIDED, 6.]},
                    block=None, seconds=0.)
        day = dict(tree=[group], gates=[gate], cost=0., stop="end of day", targets=[group, "place"])
        with patch.object(probe_cases.library, "open_case", return_value=record), \
                patch.object(probe_cases.library, "readings", return_value={}), \
                patch.object(probe_cases, "World"), patch.object(probe_cases, "identify", return_value={"tree": []}), \
                patch.object(probe_cases, "probe_day", return_value=day):
            row = probe_cases.run_case("case", "T0", None, probe.ProbeSettings())
        self.assertEqual(row["wrong_decided_groupings"], 1)
        self.assertEqual(row["decided_groupings"], 1)
        self.assertIsNone(row["targets"][1]["true"])
        self.assertIsNone(row["targets"][1]["wrong"])
