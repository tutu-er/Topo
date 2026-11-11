"""Strict data-budget tests; synthetic stub workers never run an optimizer."""
from __future__ import annotations

import builtins
from dataclasses import dataclass
import importlib.util
import io
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
SPEC = importlib.util.spec_from_file_location("strict96_under_test", ROOT/"tools/strict96.py")
strict = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(strict)


ALGORITHM = '''
from dataclasses import dataclass
import numpy as np
@dataclass
class SearchConfig:
    total_seconds: float = 14400.
    per_mip_seconds: float = 3600.
    relative_gain: float = 0.
    validation_tie: float = 1e-12
    def validate(self): pass
def prepare_common(train, config):
    assert train.points == 64
    return dict(seconds=0., reference_mse=1.)
def fit(train, selection, method, config, common):
    assert train.points == 64 and selection.points == 32 and method == '101'
    r = {0.: .002, .001: .001, .003: .003}[config.relative_gain]
    supports = [[i] for i in range(train.n)] + [list(range(train.n))]
    point = dict(supports=supports, r=[r]+[0.]*train.n, x=[0.]*(train.n+1))
    alternate = dict(supports=supports+[[0,1]], r=point['r']+[.0001], x=point['x']+[0.])
    prediction = selection.p @ np.diag([r]+[0.]*(train.n-1))
    loss = float(np.mean((selection.y-prediction)**2))
    selected = dict(index=0, point=point, validation_mse=loss, validation_scaled_mse=loss)
    return dict(selected=selected, selected_mse_view=selected, reference_mse=1.,
        frozen_candidates=[selected, dict(index=1, point=alternate)],
        competition=dict(primary_index=0, secondary_index=1),
        MIQP=[dict(bound_consistent_with_fixed_family_qp=False)], total_cap_exceeded=False)
'''

DATA = '''
from dataclasses import dataclass
@dataclass
class Observation:
    p: object
    q: object
    y: object
    labels: object
    voltage: object = None
    root: object = None
    @property
    def n(self): return len(self.labels)
    @property
    def points(self): return len(self.p)
'''


def observed(path, count=96, ids=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = np.repeat(np.arange(count)[:, None], 3, axis=1) if ids else np.ones((count, 3))
    np.savez(path, P=p[None], Q=np.zeros((1, count, 3)), V=np.ones((1, count, 3)),
             root=np.ones((1, count)), labels=np.array([101, 102, 103]))


@dataclass
class Configuration:
    total_seconds: float = 14400.
    per_mip_seconds: float = 3600.
    relative_gain: float = 0.
    validation_tie: float = 1e-12
    def validate(self):
        pass


def fixture(root):
    base = root/"project"; (base/"tools").mkdir(parents=True)
    shutil.copyfile(ROOT/"tools/strict96.py", base/"tools/strict96.py")
    (base/"algorithm.py").write_text(ALGORITHM, encoding="utf-8")
    (base/"data.py").write_text(DATA, encoding="utf-8")
    for name in ("optimization.py", "trees.py", "competition.py"):
        (base/name).write_text("# Synthetic worker stub: no solver.\n", encoding="utf-8")
    catalog = {}
    for case in strict.CASES:
        train = f"data/{case}/train_observed.npz"; observed(base/train)
        # These files must never be staged or hashed by the strict entry.
        for name in ("validation_observed.npz", "test_observed.npz", "diagnostics.json"):
            (base/"data"/case/name).write_bytes(b"FORBIDDEN SENTINEL")
        catalog[case] = dict(job=case, case=case, seed_index=0, voltage_sigma=.0005, train=train)
        folder = base/"cases"/case; folder.mkdir(parents=True)
        (folder/"metadata.json").write_text('{"root_bus": 1}', encoding="utf-8")
        (folder/"branches.csv").write_text("from_bus,to_bus,status\n1,2,True\n2,101,True\n2,102,True\n1,103,True\n", encoding="utf-8")
    strict.write(base/"data/catalog.json", catalog)
    (base/"data/manifest.json").write_text("FORBIDDEN MANIFEST", encoding="utf-8")
    return base


def freeze_fixture(base, output):
    fake = type("Module", (), {"SearchConfig": Configuration})
    with patch.object(strict, "BASE", base), patch.object(strict.importlib, "import_module", return_value=fake):
        return strict.freeze(output)


class Strict96Tests(unittest.TestCase):
    def test_exact_chronological_split_and_private_array_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"train_observed.npz"; observed(path, ids=True)
            train, selection = strict.split96(path)
            np.testing.assert_array_equal(train["p"][:, 0], np.arange(64))
            np.testing.assert_array_equal(selection["p"][:, 0], np.arange(64, 96))
            self.assertEqual(train["labels"], (101, 102, 103))
            for key in ("p", "q", "y", "voltage", "root"):
                self.assertIsNone(train[key].base); self.assertIsNone(selection[key].base)
                self.assertFalse(np.shares_memory(train[key], selection[key]))
                self.assertFalse(train[key].flags.writeable)
            for count in (95, 97):
                observed(path, count)
                with self.assertRaises(ValueError): strict.split96(path)

    def test_freeze_never_reads_other_observation_or_case_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base = fixture(root); output = root/"run"
            original_open, original_io = builtins.open, io.open
            def guarded(original):
                def opening(file, *args, **kwargs):
                    if not isinstance(file, int):
                        path = Path(file); name = path.name
                        if name in {"validation_observed.npz", "test_observed.npz", "diagnostics.json", "manifest.json"} or "cases" in path.parts:
                            raise AssertionError("Forbidden freeze read: "+str(path))
                    return original(file, *args, **kwargs)
                return opening
            with patch("builtins.open", guarded(original_open)), patch("io.open", guarded(original_io)):
                freeze_fixture(base, output)
            runtime = output/"runtime"; protocol = strict.read(output/"protocol.json")
            self.assertEqual(len(list(runtime.rglob("*.npz"))), 4)
            self.assertFalse((runtime/"cases").exists()); self.assertFalse((runtime/"data").exists())
            for info in protocol["inputs"].values():
                self.assertEqual(set(info), {"path", "sha256", "labels", "training_sha256", "selection_sha256"})
            with self.assertRaises(FileExistsError): freeze_fixture(base, output)

    def test_python_io_guard_blocks_parent_project_and_other_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory); runtime = project/"runtime"; runtime.mkdir()
            own, other = runtime/"own.npz", runtime/"other.npz"
            own.write_bytes(b"a"); other.write_bytes(b"b"); secret = project/"case.json"; secret.write_text("secret")
            code = '''
import importlib.util, pathlib, sys
spec=importlib.util.spec_from_file_location('guard',sys.argv[1]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
runtime,project,own,other,secret=map(pathlib.Path,sys.argv[2:])
module.python_read_guard(runtime,project,[own])
assert own.read_bytes()==b'a'
for path in (other,secret):
    try: path.read_bytes()
    except PermissionError: pass
    else: raise AssertionError('unapproved read succeeded')
'''
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT/"tools/strict96.py"),
                                     str(runtime), str(project), str(own), str(other), str(secret)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_stub_workers_seal_kappa_then_grade_without_extra_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base = fixture(root); output = root/"run"; freeze_fixture(base, output)
            result = strict.run(output)  # Twelve synthetic stubs; no native optimizer is imported.
            self.assertEqual((result["planned"], result["fitted"], result["selected_kappa"]), (12, 12, .001))
            self.assertEqual(result["selected_kappa_by_input"], {f"input{i:02d}": .001 for i in range(4)})
            sealed = strict.read(output/"selection_seal.json"); self.assertEqual(len(sealed["result_sha256"]), 12)
            before = strict.sha(output/"selection_seal.json")
            with self.assertRaises(FileExistsError): strict.run(output)
            graded = subprocess.run([sys.executable, "-I", "-B", str(base/"tools/strict96.py"), "grade", str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(graded.returncode, 0, graded.stdout+graded.stderr)
            report = strict.read(output/"grade.json")
            self.assertTrue(all(row["top2_contains_true"] == 4 for row in report["summary"]))
            self.assertTrue(all(row["native_bound_check_false"] == 4 for row in report["summary"]))
            self.assertEqual(report["selected_policy_summary"]["top2_contains_true"], 4)
            self.assertEqual(len(report["selected_rows"]), 4)
            self.assertEqual(before, strict.sha(output/"selection_seal.json"))

    def test_one_selection_suffix_cannot_change_other_inputs_kappa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base = fixture(root)
            first, second = root/"first", root/"second"
            freeze_fixture(base, first); strict.run(first)
            previous = strict.read(first/"selection_seal.json")
            path = base/"data/paper15/train_observed.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            # Only this input's last32 observed voltage values change. Its
            # training64 and every byte of the other three inputs stay fixed.
            arrays["V"][0, 64:, 0] = np.sqrt(1.-.003)
            np.savez(path, **arrays)
            freeze_fixture(base, second); strict.run(second)
            current = strict.read(second/"selection_seal.json")
            self.assertEqual(previous["selected_kappa_by_input"]["input00"], .001)
            self.assertEqual(current["selected_kappa_by_input"]["input00"], .003)
            self.assertIsNone(current["selected_kappa"])
            one, two = strict.read(first/"protocol.json"), strict.read(second/"protocol.json")
            for identity in one["inputs"]:
                self.assertEqual(one["inputs"][identity]["training_sha256"], two["inputs"][identity]["training_sha256"])
                if identity != "input00":
                    self.assertEqual(previous["selected_kappa_by_input"][identity], current["selected_kappa_by_input"][identity])
                    self.assertEqual(one["inputs"][identity], two["inputs"][identity])
            self.assertEqual(current["aggregate_mse_role"], "diagnostic_only_not_used_for_selection")

    def test_source_tampering_and_unsealed_truth_scoring_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base = fixture(root); output = root/"run"; freeze_fixture(base, output)
            with patch.object(strict, "truth", side_effect=AssertionError("truth before sealing")) as loader:
                with self.assertRaises(FileNotFoundError): strict.grade(output)
                loader.assert_not_called()
            with (output/"runtime/algorithm.py").open("a", encoding="utf-8") as stream: stream.write("# changed\n")
            with self.assertRaises(ValueError): strict.run(output)
            self.assertFalse((output/"run_started.json").exists())


if __name__ == "__main__":
    unittest.main()
