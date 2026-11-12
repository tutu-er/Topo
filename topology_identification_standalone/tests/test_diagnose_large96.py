"""Small post-seal diagnostic interfaces; no MIQP or actual panel is run."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import diagnose_large96 as diagnostic


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf8")


def score(p, point):
    result = np.zeros_like(p)
    for C, r in zip(point["supports"], point["r"], strict=True):
        result[:, C] += (p[:, C].sum(axis=1)*r)[:, None]
    return result


def point(extra=None, weight=.03):
    family = [[0], [1], [2], [0, 1, 2]]+([list(extra)] if extra else [])
    return dict(supports=family, r=[.05 if len(C) == 1 else weight if C == list(extra or ()) else 0. for C in family], x=[0.]*len(family))


def fixture(root, *, collapse=False, invalid_first=False):
    output = root/"study"; runtime = output/"runtime"; runtime.mkdir(parents=True)
    for name in ("data.py", "optimization.py", "trees.py"):
        shutil.copyfile(ROOT/name, runtime/name)
    methods = ["000", "100", "101"]
    count = 2 if invalid_first else 1
    p = .1*np.tile(np.eye(3), (32, 1)); q = np.zeros_like(p); labels = (11, 12, 13)
    clean = point((1, 2), weight=0. if collapse else .03)
    y = score(p, clean); voltage = np.sqrt(1.-y)
    inputs = {}
    for number in range(count):
        job = f"input{number:03d}"; path = runtime/"inputs"/(job+".npz"); path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, P=p, Q=q, V=voltage, root=np.ones(96), labels=labels)
        measured_y=1.-voltage**2
        info = dict(path=f"inputs/{job}.npz", sha256=diagnostic.sha(path), labels=list(labels))
        for start, stop, key in ((0, 64, "training_sha256"), (64, 96, "selection_sha256")):
            h=hashlib.sha256(repr(labels).encode())
            for value in (p[start:stop],q[start:stop],measured_y[start:stop]):
                value=np.ascontiguousarray(value,dtype=np.float64);h.update(repr(value.shape).encode());h.update(value.tobytes())
            info[key]=h.hexdigest()
        inputs[job]=info
    protocol=dict(inputs=inputs, methods=methods, planned_inputs=count, planned_methods=count*len(methods),
        split=dict(total=96,train=[0,64],selection=[64,96],shuffle=False),project_root=str(root),
        config=dict(response_bound=2.,qp_seconds=300.,alpha=.003,zero_tol=1e-9),
        source_sha256={name:diagnostic.sha(runtime/name) for name in ("data.py","optimization.py","trees.py")})
    write(output/"protocol.json",protocol);write(runtime/"protocol.json",protocol)
    source_protocol=diagnostic.sha(output/"protocol.json")
    common_hashes, result_hashes = {}, {}
    for number,job in enumerate(inputs):
        common=output/"common"/job/"common.json"
        write(common,dict(status="fitted",protocol_sha256=source_protocol,input_sha256=inputs[job]["sha256"],
                          common=dict(reference_mse=.001,training_sha256=inputs[job]["training_sha256"])))
        common_hashes[common.relative_to(output).as_posix()]=diagnostic.sha(common)
        proper=[[0,1],[1,2]] if invalid_first and number==0 else [[1,2]]
        write(output/"evaluation"/(job+".json"),dict(labels=list(labels),proper_clades=proper))
        for method in methods:
            target=output/"runs"/job/method/"result.json"
            record=dict(status="failed" if method=="100" else "fitted",protocol_sha256=source_protocol)
            if method!="100":
                record["result"]=dict(selected=dict(index=0,point=point() if method=="000" else point((0,1))))
                record.update(input_sha256=inputs[job]["sha256"],common_sha256=diagnostic.sha(common))
            write(target,record);result_hashes[target.relative_to(output).as_posix()]=diagnostic.sha(target)
    seal=dict(sealed=True,all_planned_terminal=True,protocol_sha256=source_protocol,
              source_sha256=protocol["source_sha256"],result_sha256=result_hashes,common_sha256=common_hashes)
    write(output/"selection_seal.json",seal)
    write(output/"completion.json",dict(all_planned_terminal=True,selection_seal_sha256=diagnostic.sha(output/"selection_seal.json")))
    return output


class TrueFamilyDiagnosticTests(unittest.TestCase):
    def run_diagnostic(self, output):
        return subprocess.run([sys.executable,"-I","-B",str(ROOT/"tools/diagnose_large96.py"),str(output)],capture_output=True,text=True)

    def test_unsealed_or_tampered_study_is_rejected_before_qp(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=fixture(Path(temporary))
            seal=output/"selection_seal.json";saved=seal.read_bytes();seal.unlink()
            with self.assertRaises(FileNotFoundError): diagnostic.verify_study(output)
            self.assertFalse((output/"true_family_diagnostics").exists())
            seal.write_bytes(saved)
            target=output/"runs/input000/000/result.json";target.write_bytes(target.read_bytes()+b" ")
            result=self.run_diagnostic(output)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("Frozen bytes changed",result.stderr)
            self.assertFalse((output/"true_family_diagnostics").exists())

    def test_one_fixed_qp_per_input_decomposition_and_original_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=fixture(Path(temporary));_,_,before=diagnostic.verify_study(output)
            result=self.run_diagnostic(output)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            dest=output/"true_family_diagnostics";summary=diagnostic.read(dest/"summary.json")
            self.assertEqual((summary["true_family_fit_calls"],summary["compared_methods"],summary["unavailable_methods"]),(1,2,1))
            self.assertEqual(summary["native_MIQP_calls"],0)
            oracle=diagnostic.read(dest/"inputs/input000.json")
            self.assertEqual((oracle["stored_internal_count"],oracle["active_internal_count"]),(1,1))
            self.assertFalse(oracle["zero_weight_collapses_true_family"])
            self.assertLess(oracle["train_mse"],1e-20)
            with (dest/"decompositions.csv").open(encoding="utf-8-sig",newline="") as stream: rows=list(csv.DictReader(stream))
            for row in rows:
                if row["status"]=="compared":
                    self.assertAlmostEqual(float(row["delta_J"]),float(row["normalized_training_fit_difference"])+float(row["count_penalty_difference"]),places=12)
            first=next(row for row in rows if row["method"]=="000")
            self.assertAlmostEqual(float(first["count_penalty_difference"]),-.003)
            self.assertTrue(all(diagnostic.sha(output/name)==expected for name,expected in before.items()))

    def test_bad_true_family_is_recorded_and_next_input_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=fixture(Path(temporary),invalid_first=True);result=self.run_diagnostic(output)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            summary=diagnostic.read(output/"true_family_diagnostics/summary.json")
            self.assertEqual((summary["planned_inputs"],summary["fitted_inputs"],summary["failed_inputs"]),(2,1,1))
            self.assertEqual(summary["planned_method_comparisons"],6)
            self.assertEqual(summary["true_family_fit_calls"],1)

    def test_zero_weight_truth_clade_keeps_raw_count_and_reports_collapse(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=fixture(Path(temporary),collapse=True);result=self.run_diagnostic(output)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            oracle=diagnostic.read(output/"true_family_diagnostics/inputs/input000.json")
            self.assertEqual((oracle["stored_internal_count"],oracle["active_internal_count"]),(1,0))
            self.assertTrue(oracle["zero_weight_collapses_true_family"])
            self.assertEqual(oracle["zero_weight_true_clades"],[[1,2]])


if __name__=="__main__": unittest.main()
