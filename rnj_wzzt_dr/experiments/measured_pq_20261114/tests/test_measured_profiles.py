"""Independent acceptance checks for measured, synchronized P/Q profiles.

These use small synthetic source records, not the measured study outcomes.
Run from the repository root with:
    python -m unittest discover -s rnj_wzzt_dr/experiments/measured_pq_20261114/tests -v
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rnj_wzzt_dr.experiments.measured_pq_20261114 import profiles


def fixture():
    """Three distinct sources with known training RMS, reversals, and zero P."""
    p = np.tile(np.array([[3., 5., 2.], [-4., -5., -2.],
                          [-3., 0., 2.], [4., np.sqrt(50.), -2.]]), (96, 1))
    q = np.tile(np.array([[-4., 2., -1.], [0., -3., 4.],
                          [1., 7., 0.], [7., 1., -5.]]), (96, 1))
    # Deliberately different future magnitudes: they must not set the scale.
    p[192:] *= 13.
    q[192:] *= 7.
    start = datetime(2023, 1, 10, 7, tzinfo=timezone.utc)  # Berlin 08:00.
    timestamps = [(start + timedelta(minutes=15 * k)).isoformat().replace("+00:00", "Z")
                  for k in range(384)]
    return dict(labels=["terminal_c", "terminal_a", "terminal_b"], p_kw=p,
                q_kvar=q, timestamps=timestamps,
                nominal_p_kw=np.array([2. * np.sqrt(12.5), 15., 1.]) / .65,
                feeder_ids=[1, 2, 3],
                source_metadata={"archive_sha256": "a" * 64,
                                 "missing_policy": "reject"})


class MeasuredProfileAcceptance(unittest.TestCase):
    def test_same_positive_scale_preserves_pq_signs_and_source_order(self):
        args = fixture()
        result = profiles.build_from_arrays(**args)
        expected_scale = np.array([2., 3., .5])
        np.testing.assert_allclose(result["p_kw"], args["p_kw"] * expected_scale,
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result["q_kvar"], args["q_kvar"] * expected_scale,
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(np.sign(result["p_kw"]), np.sign(args["p_kw"]))
        np.testing.assert_array_equal(np.sign(result["q_kvar"]), np.sign(args["q_kvar"]))
        self.assertEqual(list(result["labels"]), args["labels"])
        mapping = result["metadata"]["node_mapping"]
        self.assertEqual([row["label"] for row in mapping], args["labels"])
        self.assertEqual([row["feeder_id"] for row in mapping], [1, 2, 3])
        np.testing.assert_allclose([row["scale"] for row in mapping], expected_scale)
        # Includes P=0 with Q!=0: deriving new Q from a P/Q ratio would fail.
        self.assertEqual(result["p_kw"][2, 1], 0.)
        self.assertEqual(result["q_kvar"][2, 1], 21.)

    def test_future_changes_cannot_set_training_scale(self):
        args = fixture()
        before = profiles.build_from_arrays(**args)
        changed = deepcopy(args)
        changed["p_kw"][192:] = -103. * changed["p_kw"][192:] + 17.
        changed["q_kvar"][192:] = 71. * changed["q_kvar"][192:] - 13.
        after = profiles.build_from_arrays(**changed)
        np.testing.assert_array_equal(before["p_kw"][:192], after["p_kw"][:192])
        np.testing.assert_array_equal(before["q_kvar"][:192], after["q_kvar"][:192])
        self.assertEqual([row["scale"] for row in before["metadata"]["node_mapping"]],
                         [row["scale"] for row in after["metadata"]["node_mapping"]])
        self.assertEqual(after["metadata"]["scale_fit_split"], "train")

    def test_four_day_split_is_disjoint_and_chronological(self):
        args = fixture()
        result = profiles.build_from_arrays(**args)
        self.assertEqual(result["split"],
                         {"train": [0, 192], "validation": [192, 288], "new_day": [288, 384]})
        self.assertEqual(list(result["timestamps"]), args["timestamps"])
        self.assertEqual(result["p_kw"].shape, (384, 3))
        self.assertEqual(result["q_kvar"].shape, (384, 3))
        self.assertEqual(result["metadata"]["archive_sha256"], "a" * 64)
        self.assertIn("reject", result["metadata"]["missing_policy"].lower())

    def test_nonfinite_selected_records_are_rejected(self):
        for field, row, bad in [("p_kw", 11, np.nan), ("q_kvar", 220, np.nan),
                                ("p_kw", 330, np.inf), ("q_kvar", 383, -np.inf)]:
            with self.subTest(field=field, row=row):
                args = fixture()
                args[field][row, 1] = bad
                with self.assertRaises((ValueError, RuntimeError)):
                    profiles.build_from_arrays(**args)

    def test_duplicate_source_or_label_is_rejected(self):
        for key, bad in [("feeder_ids", [1, 1, 3]),
                         ("labels", ["same", "same", "different"])]:
            with self.subTest(key=key):
                args = fixture()
                args[key] = bad
                with self.assertRaises((ValueError, RuntimeError)):
                    profiles.build_from_arrays(**args)

    def test_invalid_time_grids_and_incomplete_window_are_rejected(self):
        for kind in ["duplicate", "out_of_order", "gap", "short_window"]:
            with self.subTest(kind=kind):
                args = fixture()
                if kind == "duplicate":
                    args["timestamps"][10] = args["timestamps"][9]
                elif kind == "out_of_order":
                    args["timestamps"][10], args["timestamps"][11] = (
                        args["timestamps"][11], args["timestamps"][10])
                elif kind == "gap":
                    args["timestamps"][10] = "2023-01-10T09:31:00Z"
                else:
                    args["timestamps"] = args["timestamps"][:-1]
                    args["p_kw"] = args["p_kw"][:-1]
                    args["q_kvar"] = args["q_kvar"][:-1]
                with self.assertRaises((ValueError, RuntimeError)):
                    profiles.build_from_arrays(**args)

    def test_zero_training_rms_and_invalid_nominal_power_are_rejected(self):
        for kind in ["zero_training_rms", "zero_nominal", "negative_nominal", "nan_nominal"]:
            with self.subTest(kind=kind):
                args = fixture()
                if kind == "zero_training_rms":
                    args["p_kw"][:192, 1] = 0.
                else:
                    args["nominal_p_kw"][1] = {"zero_nominal": 0., "negative_nominal": -1.,
                                                "nan_nominal": np.nan}[kind]
                with self.assertRaises((ValueError, RuntimeError)):
                    profiles.build_from_arrays(**args)


class QuarterHourAcceptance(unittest.TestCase):
    def test_joint_validity_and_twelve_minute_threshold(self):
        timestamps = pd.date_range("2023-01-10T07:00:00Z", periods=30, freq="min")
        minute = np.arange(30, dtype=float)
        frame = pd.DataFrame({"timestamp_UTC": timestamps, "active_power_kW": minute + 1.,
                              "reactive_power_kVar": 5. - 2. * minute,
                              "voltage_one_phase_V": np.full(30, 230.)})
        frame.loc[11:14, "voltage_one_phase_V"] = 84.  # 11 valid minutes: reject.
        frame.loc[15, "voltage_one_phase_V"] = 85.    # Inclusive lower boundary.
        frame.loc[16, "voltage_one_phase_V"] = 264.   # Inclusive upper boundary.
        frame.loc[27, "active_power_kW"] = np.nan
        frame.loc[28, "reactive_power_kVar"] = np.nan
        frame.loc[29, "voltage_one_phase_V"] = 265.  # Exactly 12 jointly valid.
        result = profiles.quarter_hour_frame(frame)
        np.testing.assert_array_equal(result["valid_minutes"].to_numpy(), [11, 12])
        self.assertTrue(np.isnan(result.iloc[0]["p_kw"]))
        self.assertTrue(np.isnan(result.iloc[0]["q_kvar"]))
        self.assertAlmostEqual(float(result.iloc[1]["p_kw"]), 21.5)
        self.assertAlmostEqual(float(result.iloc[1]["q_kvar"]), -36.)
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertEqual(result.index[0], timestamps[0])
        self.assertEqual(result.index[1], timestamps[15])

    def test_duplicate_minutes_cannot_inflate_coverage(self):
        frame = pd.DataFrame({
            "timestamp_UTC": pd.date_range("2023-01-10T07:00:00Z", periods=11, freq="min"),
            "active_power_kW": np.ones(11), "reactive_power_kVar": -np.ones(11),
            "voltage_one_phase_V": np.full(11, 230.)})
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            profiles.quarter_hour_frame(frame)


class ArchiveAcceptance(unittest.TestCase):
    """Exercise the public loader against a tiny locally constructed archive."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "measured.npz"
        args = fixture()
        p = np.tile(np.arange(1., 19.), (384, 1))
        q = np.tile(np.arange(-9., 9.), (384, 1))
        p[:, :3], q[:, :3] = args["p_kw"], args["q_kvar"]
        self.payload = dict(p_kw=p, q_kvar=q, valid_minutes=np.full((384, 18), 15, dtype=np.uint8),
                            timestamps_utc=np.asarray(args["timestamps"]), feeder_ids=np.arange(1, 19),
                            metadata_json=np.asarray(json.dumps({"dataset": "synthetic acceptance fixture"})))
        self.network = {"buses": [
            {"bus_id": 0, "is_terminal": False, "pd_kw": 0.},
            *[{"bus_id": name, "is_terminal": True, "pd_kw": float(nominal)}
              for name, nominal in zip(args["labels"], args["nominal_p_kw"])]]}
        self.loader_patch = patch("topology_identification_standalone.data.load_case",
                                  return_value=self.network)
        self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)
        self.write_archive()

    def write_archive(self):
        np.savez_compressed(self.path, **self.payload)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_content_hash_source_mapping_and_synchronized_window(self):
        expected = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = profiles.build_case("paper15", self.path, "2023-01-10", expected_sha256=expected)
        self.assertEqual(result["metadata"]["archive_sha256"], expected)
        self.assertEqual(result["metadata"]["source_archive_sha256"], expected)
        self.assertEqual([row["feeder_id"] for row in result["metadata"]["node_mapping"]], [1, 2, 3])
        self.assertEqual(result["timestamps"], fixture()["timestamps"])
        self.assertEqual(result["metadata"]["source_timestamps_utc"], result["timestamps"])
        np.testing.assert_allclose(result["p_kw"], self.payload["p_kw"][:, :3] * [2., 3., .5])
        # ZIP readers often tolerate trailing bytes; the locked hash must still reject them.
        with self.path.open("ab") as stream:
            stream.write(b"changed after locking")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            profiles.build_case("paper15", self.path, "2023-01-10", expected_sha256=expected)

    def test_counts_and_missing_values_are_enforced_independently(self):
        for kind in ["low_count", "missing_p", "missing_q"]:
            with self.subTest(kind=kind):
                self.payload["valid_minutes"][:] = 15
                self.payload["p_kw"][200, 0] = 5.
                self.payload["q_kvar"][200, 0] = 3.
                if kind == "low_count":
                    self.payload["valid_minutes"][200, 0] = 11
                elif kind == "missing_p":
                    self.payload["p_kw"][200, 0] = np.nan
                else:
                    self.payload["q_kvar"][200, 0] = np.nan
                digest = self.write_archive()
                with self.assertRaises(ValueError):
                    profiles.build_case("paper15", self.path, "2023-01-10", expected_sha256=digest)

    def test_source_library_requires_eighteen_distinct_feeders(self):
        for ids in [np.arange(1, 18), np.r_[np.arange(1, 18), 17]]:
            with self.subTest(ids=ids.tolist()):
                self.payload["feeder_ids"] = ids
                digest = self.write_archive()
                with self.assertRaisesRegex(ValueError, "distinct feeders"):
                    profiles.build_case("paper15", self.path, "2023-01-10", expected_sha256=digest)

    def test_window_across_daylight_saving_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "daylight-saving"):
            profiles.build_case("paper15", self.path, "2024-03-30")


if __name__ == "__main__":
    unittest.main()
