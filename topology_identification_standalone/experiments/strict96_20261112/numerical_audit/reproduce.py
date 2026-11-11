"""Replay the sealed numerical fixture without the original project outputs."""
from pathlib import Path
import argparse
import importlib.util
import json
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new result directory; never overwritten")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("sealed_numerical_audit", root/"audit_t_qp_20261112.py")
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    fixture = json.loads((root/"training_fixture.json").read_text(encoding="utf-8"))
    with np.load(root/"train_observed.npz", allow_pickle=False) as observed:
        n = len(observed["labels"])
        p, q, v = [np.asarray(observed[k], float).reshape(-1, n) for k in ("P", "Q", "V")]
        target = np.asarray(observed["root"], float).reshape(-1, 1)**2-v**2
    if len(p) != 96:
        raise ValueError("The sealed numerical fixture must use exactly 96 time points")
    args.output.mkdir(parents=True, exist_ok=False)
    runs = [audit.solve(fixture, p, q, target, args.output, selection=False,
                        name="fixed_gurobi_QP", limit=300.)]
    for name, presolve, focus in (("T_auto", -1, 0), ("T_presolve0", 0, 0), ("T_numericfocus3", -1, 3)):
        runs.append(audit.solve(fixture, p, q, target, args.output, selection=True,
                               name=name, presolve=presolve, focus=focus, limit=300.))
    audit.write(args.output/"replay.json", dict(scope="train96 numerical fixture only", runs=runs,
        observation_sha256=audit.sha(root/"train_observed.npz"), validation_test_truth_used=False))
