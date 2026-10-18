"""Audit saved AC96 paths after fitting; never select an estimator using truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from research_experiments.rnj.audit_source_ac96_benchmark import (
    clade_metrics, physical_truth, relative_error,
)

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit_paths(run, dataset, case):
    rows, jobs = [], []
    for directory in sorted((run / "jobs").glob(f"{case}__*")):
        if not (directory / "result.json").exists():
            continue
        result, model = read(directory / "result.json"), read(directory / "model.json")
        if result["method"] not in {"WZZT", "RNJ+WZZT"}:
            continue
        labels = model["terminal_labels"]
        truth = physical_truth(dataset / case / "network", labels)
        with np.load(directory / "model.npz") as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        points = []
        for index, point in enumerate(model["path"]):
            supports = point["supports"]
            r, x = np.asarray(point["r_weights"]), np.asarray(point["x_weights"])
            z = np.zeros((len(supports), len(labels)))
            for k, support in enumerate(supports):
                z[k, support] = 1
            R, X = z.T @ (r[:, None] * z), z.T @ (x[:, None] * z)
            prediction = arrays["P"] @ R.T + arrays["Q"] @ X.T
            prediction += arrays["P_source"][:, None] * np.asarray(point["source_gR"])
            prediction += arrays["Q_source"][:, None] * np.asarray(point["source_gX"])
            mae = float(np.mean(np.abs(prediction - arrays["Y"])))
            if not np.isclose(mae, point["train_mae"], atol=1e-10, rtol=1e-8):
                raise ValueError(f"Saved path objective differs: {directory.name}/{index}")
            clades = [[labels[i] for i in support]
                      for support, rv, xv in zip(supports, r, x, strict=True)
                      if 1 < len(support) < len(labels) and max(rv, xv) > 1e-8]
            scores = clade_metrics(clades, truth["clades"], labels)
            row = {"case": case, "variant": result["variant"], "method": result["method"],
                   "job_id": directory.name, "path_index": index,
                   "support_count": len(supports), "active_nontrivial_clades": len(clades),
                   "clade_f1": scores["clade_f1"], "precision": scores["precision"],
                   "recall": scores["recall"], "train_mae": mae,
                   "accepted_gain": point.get("gain"),
                   "r_relative_error": relative_error(R, truth["R"]),
                   "x_relative_error": relative_error(X, truth["X"]),
                   "bound_active": point.get("coefficient_bound_active"),
                   "active_clades": json.dumps(clades, separators=(",", ":"))}
            rows.append(row)
            points.append(row)
        if not points:
            continue
        best = max(points, key=lambda item: item["clade_f1"])
        jobs.append({"job_id": directory.name, "status": result["status"],
                     "stop_reason": result["stop_reason"], "path_points": len(points),
                     "final_f1": points[-1]["clade_f1"],
                     "maximum_saved_path_f1": best["clade_f1"],
                     "first_best_path_index": best["path_index"],
                     "exact_path_points": [p["path_index"] for p in points if p["clade_f1"] == 1],
                     "best_path_use": "post-hoc diagnostic only; truth was not available for selection",
                     "final_bound_active": model.get("coefficient_bound_active"),
                     "model_json_sha256": hashlib.sha256((directory / "model.json").read_bytes()).hexdigest()})
    return rows, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/source_ac96_benchmark_20261018")
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/source_ac96_20261018")
    parser.add_argument("--case", choices=("soumalas11", "pengwah18"), default="soumalas11")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/source_ac96_diagnosis_20261018")
    args = parser.parse_args()
    rows, jobs = audit_paths(args.run, args.dataset, args.case)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"{args.case}_saved_path_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["job_id"])
        writer.writeheader()
        writer.writerows(rows)
    report = {"case": args.case, "mode": "saved-model audit; no fitting, solving or tuning",
              "path_objectives_independently_reproduced": len(rows), "jobs": jobs}
    (args.output / f"{args.case}_saved_path_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case": args.case, "path_points": len(rows), "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
