"""Recompute saved objectives and compare repair in the same synthetic domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run import ROOT, scenes, sha, write
from .model import repair_control, unions


def independent_loss(p, q, y, point):
    n = p.shape[1]
    R, X = np.zeros((n, n)), np.zeros((n, n))
    family = [set(s) for s in point["supports"]]
    if len({tuple(sorted(s)) for s in family}) != len(family):
        raise AssertionError("duplicate supports")
    for i, (s, r, x) in enumerate(zip(family, point["r"], point["x"], strict=True)):
        if not s or not s <= set(range(n)) or not (-1e-8 <= min(r, x) <= max(r, x) <= 2. + 1e-8):
            raise AssertionError("support or coefficient bound violation")
        for t in family[:i]:
            if s & t and not (s <= t or t <= s):
                raise AssertionError("crossing output family")
        for a in s:
            for b in s:
                R[a, b] += r
                X[a, b] += x
    return float(np.abs(y - p @ R.T - q @ X.T).mean())


def verify(directory, *, compare=True):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checks, errors, controls = 0, [], []
    for name, digest in manifest["sources"].items():
        if sha(ROOT / name) != digest:
            errors.append("source hash changed: " + name)
        checks += 1
    synthetic = json.loads((directory / "synthetic.json").read_text(encoding="utf-8"))
    for row in synthetic:
        if row["status"] != "completed":
            errors.append("synthetic failed: " + row["id"])
            continue
        file = directory / (row["id"] + ".npz")
        if sha(file) != row["data_sha256"]:
            errors.append("synthetic hash changed: " + row["id"])
        with np.load(file) as d:
            for point in (row["baseline"], row["rebuild"]["point"]):
                value = independent_loss(d["P"], d["Q"], d["Y"], point)
                if abs(value - point["objective"]) > 1e-8:
                    errors.append("objective mismatch: " + row["id"])
                checks += 1
            if compare:
                n = row["n"]
                pool = [s for _, s in unions(tuple((i,) for i in range(n)))]
                result = repair_control(scenes(d["P"], d["Q"], d["Y"]),
                                        row["baseline"]["supports"], pool, seconds=5.)
                point = result["point"]
                predicted = {tuple(s) for s, r, x in zip(point["supports"], point["r"], point["x"])
                             if 1 < len(s) < n and max(r, x) > 1e-9}
                true = {tuple(s) for s in row["rebuild_score"]["truth"]}
                controls.append(dict(id=row["id"], repair=result, exact=predicted == true,
                                     f1=2 * len(predicted & true) / (len(predicted) + len(true)),
                                     same_domain_MILP_objective=row["rebuild"]["point"]["objective"],
                                     MILP_minus_repair=row["rebuild"]["point"]["objective"] - point["objective"],
                                     test_mae=independent_loss(d["testP"], d["testQ"], d["testY"], point)))
                print(row["id"], "same-domain repair", result["stop_reason"], controls[-1]["f1"], flush=True)
    real_file = directory / "real.json"
    if real_file.exists():
        from .run import from_observed
        real = json.loads(real_file.read_text(encoding="utf-8"))
        for row in real:
            parent = ROOT / "outputs/wzzt_paper_20261029_full/jobs" / row["id"]
            payload = json.loads((parent / "payload.json").read_text(encoding="utf-8"))
            for split in ("train", "validation", "test"):
                if sha(payload[split]) != row["input_sha256"][split]:
                    errors.append("real hash mismatch: " + row["id"] + split)
                prepared = from_observed(payload[split], 48 if split == "train" else None)
                for name, metric in row["metrics"].items():
                    value = independent_loss(prepared.p[0], prepared.q[0], prepared.target[0], metric["point"])
                    if abs(value - metric[split + "_mae"]) > 1e-8:
                        errors.append("real objective mismatch: " + row["id"] + name + split)
                    checks += 1
            for attempt in row["rebuild"]["attempts"]:
                point = attempt["point"]
                if attempt["accepted"] and (point is None or not attempt["audit"]["valid"] or
                                            point["objective"] >= attempt["before_objective"]):
                    errors.append("invalid acceptance: " + row["id"])
                checks += 1
    if compare:
        write(directory / "synthetic_control.json", controls)
    result = dict(checks=checks, errors=errors, passed=not errors,
                  verification_source_sha256=sha(__file__),
                  independent_scope="saved support bounds, laminarity, original P/Q/Y objectives and accepted improvements",
                  controls=len(controls), control_seconds_per_case=5.,
                  control_scope="same complete block-union domain, original single-candidate repair acceptance rule")
    write(directory / "verification.json", result)
    if errors:
        raise AssertionError(errors)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-control", action="store_true")
    args = parser.parse_args()
    verify(args.directory, compare=not args.no_control)
