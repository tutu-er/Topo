"""Single-day batch boundary, native RNJ worker and no-budget-reset checks."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm import SearchConfig
import data
from tools import large96


class Large96Tests(unittest.TestCase):
    def test_declared_matrices_and_sentinel_are_balanced(self):
        main, pressure = large96.specifications("main"), large96.specifications("pressure")
        self.assertEqual((len(main), len(pressure), len(large96.METHODS)), (240, 160, 10))
        self.assertEqual(len({(r['case'], r['replicate']) for r in main}), 80)
        self.assertEqual({r['replicate'] for r in main[:24]}, {0, 1})
        for case in large96.CASES:
            self.assertEqual(sum(r['case'] == case for r in main[:24]), 6)

    def test_native_worker_has_sole96_and_can_read_its_own_common(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"study"; runtime = output/"runtime"; runtime.mkdir(parents=True)
            for name in large96.SOURCES:
                target = runtime/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, target)
            obs, _ = data.generate_day96("paper15", 999, .0005, study_seed=927)
            path = runtime/"inputs/input000.npz"; path.parent.mkdir()
            np.savez_compressed(path, P=obs.p, Q=obs.q, V=obs.voltage, root=obs.root, labels=obs.labels)
            train, selection = large96.split96(path)
            config = SearchConfig(total_seconds=30., per_mip_seconds=10., qp_seconds=5., output_reserve_seconds=3.)
            protocol = dict(inputs=dict(input000=dict(path="inputs/input000.npz", sha256=large96.sha(path),
                labels=list(obs.labels), training_sha256=large96.fingerprint(train), selection_sha256=large96.fingerprint(selection))),
                methods=["rnj"], config=asdict(config), project_root=str(Path(temporary)),
                source_sha256=large96.source_hashes(runtime))
            large96.write(output/"protocol.json", protocol); large96.write(runtime/"protocol.json", protocol)
            for method in ("common", "rnj"):
                process = subprocess.run([sys.executable, "-I", "-B", str(runtime/"tools/large96.py"), "worker",
                    str(output), "--job", "input000", "--method", method], capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
                record = large96.read(large96.result_path(output, "input000", None if method == "common" else method))
                self.assertEqual(record['status'], 'fitted', record.get('error'))
                self.assertEqual(record['observed_npz_opened'], [str(path.resolve())])
                self.assertEqual((record['train_points'], record['selection_points']), (64, 32))
            # Extra runtime truth/observation files are rejected before a batch.
            (runtime/"truth.json").write_text('{}')
            with self.assertRaises(ValueError):
                large96.check_runtime(output, protocol)

    def test_unfinished_attempt_cannot_receive_a_fresh_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            large96.write(output/"protocol.json", dict(inputs={"input000": {}}, methods=["101"], workers=1))
            large96.write(output/"runs/input000/101/attempt.json", dict(state="running", pid=-1))
            with patch.object(large96, 'check_runtime'):
                with self.assertRaisesRegex(RuntimeError, 'budgets cannot be reset'):
                    large96.run(output)

    def test_truth_requires_sealed_results_and_frozen_case_parser_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            large96.write(output/"protocol.json", {})
            with self.assertRaises(FileNotFoundError):
                large96.expose_truth(output)
        for case in large96.CASES:
            network = data.load_case(case)
            labels = [b['bus_id'] for b in network['buses'] if b['is_terminal']]
            expected = sorted(list(C) for C in data.case_truth(case, labels) if 1 < len(C) < len(labels))
            self.assertEqual(large96.frozen_truth(ROOT/"cases"/case, labels), expected)


if __name__ == '__main__':
    unittest.main()
