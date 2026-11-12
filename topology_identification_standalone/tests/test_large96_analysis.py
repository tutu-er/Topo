"""Independent aggregation, ambiguity and strict-seal regression checks."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools import analyze_large96 as audit


class Large96AnalysisTests(unittest.TestCase):
    def test_selection_tie_and_ambiguity_use_distinct_roles(self):
        first = dict(index=0, internal=1, selection_scaled_mse=.995)
        alternate = dict(index=1, internal=2, selection_scaled_mse=1.)
        self.assertEqual(audit.choose([alternate, first], 1e-12), first)
        self.assertFalse(audit.near_pair(first, alternate, .003, 1e-12))
        self.assertTrue(audit.near_pair(first, alternate, .01, 1e-12))
        self.assertFalse(audit.near_pair(first, None, .9, 1e-12))
        self.assertTrue(audit.near_pair(dict(selection_scaled_mse=1.), dict(selection_scaled_mse=1.+1e-12), 0., 1e-12))

    def test_failures_keep_denominator_and_no_false_mean(self):
        rows = [dict(method="101", status="verified", F1=.8, exact=True, total_seconds=2.),
                dict(method="101", status="failed", F1=None, exact=None, total_seconds=3.)]
        summary = audit.grouped(rows, ["method"])[0]
        self.assertEqual(summary["planned"], 2)
        self.assertEqual(summary["verified"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["F1_verified_mean"], .8)
        self.assertEqual(summary["F1_failure_zero_mean"], .4)
        self.assertEqual(summary["exact_rate_planned"], .5)
        self.assertEqual(summary["total_seconds_recorded"], 5.)

    def test_factorial_recovers_known_negative_and_interaction(self):
        rows = []
        for method in audit.FACTORIAL:
            R, T, E = map(int, method)
            rows.append(dict(job="j", case="c", replicate=0, method=method, status="verified",
                             F1=.5+.1*R-.04*T+.02*E-.03*R*T, exact=False, total_seconds=1.))
        effects, interactions, orders, shapley = audit.factorial_statistics(rows, audit.FACTORIAL)
        self.assertEqual((len(effects), len(interactions), len(orders), len(shapley)), (12, 7, 18, 3))
        rt = [r for r in interactions if r["interaction"] == "RT" and r["condition_value"] == 0][0]
        self.assertAlmostEqual(rt["delta_F1"], -.03)
        self.assertTrue(all(r["delta_F1"] < 0 for r in effects if r["module"] == "T"))
        pressure = [r for r in rows if r["method"] in {"000", "100", "101", "111"}]
        pe, _, po, ps = audit.factorial_statistics(pressure, [r["method"] for r in pressure])
        self.assertEqual(len(pe), 3)
        self.assertEqual(po, [])
        self.assertTrue(all(not r["valid"] for r in ps))

    def test_bootstrap_keeps_noise_siblings_in_case_day_clusters(self):
        rows = []
        for case in ("a", "b"):
            for replicate in range(2):
                for sigma in (.0002, .001):
                    for method in ("100", "101"):
                        rows.append(dict(job=f"{case}{replicate}{sigma}", case=case, replicate=replicate,
                            method=method, status="verified", F1=.5+(.1 if method == "101" else 0.),
                            exact=False, total_seconds=1.))
        intervals = audit.cluster_intervals(rows, ("100", "101"), draws=20)
        pair = next(r for r in intervals if r["old"] == "100" and r["new"] == "101" and r["metric"] == "F1")
        self.assertEqual(pair["independent_case_day_clusters"], 4)
        self.assertEqual(pair["paired"], 8)
        self.assertAlmostEqual(pair["estimate"], .1)
        self.assertAlmostEqual(pair["interval_low"], .1)
        self.assertAlmostEqual(pair["interval_high"], .1)

    def test_all_generation_failures_need_no_observations_or_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); runtime = root/"runtime"; runtime.mkdir()
            (runtime/"stub.py").write_text("# frozen source\n")
            inputs = {"j": dict(path=None, sha256=None, labels=None, training_sha256=None,
                                 selection_sha256=None, generation_failed=True)}
            protocol = dict(inputs=inputs, methods=["000", "101"], config={},
                source_sha256={"stub.py": audit.sha(runtime/"stub.py")},
                split=dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False))
            audit.write_json(root/"protocol.json", protocol); audit.write_json(runtime/"protocol.json", protocol)
            audit.write_json(root/"evaluation_map.json", {"j": dict(case="paper15", replicate=0,
                voltage_sigma=.0005, profile="baseline", level=1., cohort="main")})
            hashes = {}
            for method in protocol["methods"]:
                name = f"runs/j/{method}/result.json"
                audit.write_json(root/name, dict(status="failed", failure_stage="generation", method_started=False,
                                                total_seconds=0., fresh_seconds=0., common_seconds=0.))
                hashes[name] = audit.sha(root/name)
            audit.write_json(root/"selection_seal.json", dict(sealed=True, all_planned_terminal=True,
                protocol_sha256=audit.sha(root/"protocol.json"), result_sha256=hashes))
            audit.write_json(root/"completion.json", dict(all_planned_terminal=True,
                selection_seal_sha256=audit.sha(root/"selection_seal.json")))
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit.analyze(root, bootstrap=5)
            self.assertFalse((root/"evaluation").exists())
            self.assertEqual([r["failed"] for r in result["methods"]], [1, 1])
            receipt = json.loads((root/"analysis/pre_truth_audit.json").read_text())
            self.assertEqual(receipt["observed_file_sha256"], {})
            self.assertFalse(receipt["truth_opened"])

    def test_unsealed_batch_is_rejected_before_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit.write_json(root/"protocol.json", {})
            audit.write_json(root/"selection_seal.json", dict(sealed=False))
            audit.write_json(root/"completion.json", dict(all_planned_terminal=False))
            with self.assertRaises(ValueError):
                audit.analyze(root)


if __name__ == "__main__":
    unittest.main()
