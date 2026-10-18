"""Data-boundary and native-network checks, independent of optimizer success."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_experiments.mobile_ev_mle.label_free_bounds import experiment as exp


class DataProtocolTests(unittest.TestCase):
    def test_public_candidates_do_not_export_generating_lines(self):
        public=exp.public_assets()
        self.assertEqual(len(public["candidates"]),10)
        self.assertEqual(public["observed_nodes"],list(range(2,34)))
        for candidate in public["candidates"]:
            self.assertEqual(len(candidate["edges"]),32)
            self.assertNotIn("true",candidate["id"])
            self.assertNotIn("r",candidate); self.assertNotIn("x",candidate)
            self.assertTrue(all(len(edge)==2 for edge in candidate["edges"]))
        self.assertGreater(public["domain"]["z_min"],0)

    def test_generator_native_internal_loads_and_source_Q(self):
        meter,truth=exp.generate_history(96,20261018071,"pcc_PQ")
        self.assertEqual(np.asarray(meter["p"]).shape,(96,32))
        self.assertNotIn("generating_candidate_id",meter)
        self.assertNotIn("branch_table",meter)
        self.assertNotIn("standardized_noise_squared",meter)
        self.assertGreater(float(np.std(truth["source_q"])),0)
        self.assertGreater(float(np.std(truth["source_p"])),0)
        self.assertTrue(np.all(np.asarray(truth["q"])>0))
        self.assertLess(truth["ac_equation_residual"],1e-10)
        self.assertLess(truth["power_balance_residual"],1e-10)
        self.assertGreater(float(np.std(np.asarray(truth["p"])[:,0])),0)

    def test_source_scenarios_pair_load_and_meter_noise(self):
        m0,t0=exp.generate_history(96,20261018071,"none")
        m1,t1=exp.generate_history(96,20261018071,"pcc_PQ")
        for key in ("p","q"):
            np.testing.assert_array_equal(t0[key],t1[key])
            np.testing.assert_array_equal(m0[key],m1[key])
        self.assertFalse(np.allclose(t0["v"],t1["v"]))
        self.assertFalse(m0["source_projection"])
        self.assertTrue(m1["source_projection"])

    def test_files_separate_truth_and_meters(self):
        temporary_root=Path(__file__).resolve().parents[3]/"outputs"/"label_free_bounds_unit_tmp"
        temporary_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            manifest=exp.generate_bundle(folder,pilot=True)
            row=manifest[0]
            self.assertNotEqual(row["meter_file"],row["truth_file"])
            m=exp.read(Path(folder)/row["meter_file"])
            t=exp.read(Path(folder)/row["truth_file"])
            self.assertEqual(m["id"],t["id"])
            self.assertNotIn("branch_table",m)
            self.assertIn("branch_table",t)
            self.assertEqual(row["meter_sha256"],exp.digest(Path(folder)/row["meter_file"]))


if __name__=="__main__": unittest.main()
