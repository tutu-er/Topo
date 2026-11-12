"""Recompute candidate objectives and AC scores without reading physical truth."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .profile_gaps import ROOT, RUN, RUNTIME, read, write, sha
from .policy import extension_decision


def observations(job):
    payload = read(RUN/"jobs"/job/"payload.json")
    directory = Path(payload["input_dir"])/"observed"
    frames = [pd.read_csv(directory/f"{name}.csv", index_col=0, float_precision="round_trip")
              for name in ("P_terminal", "Q_terminal", "V_terminal")]
    for frame in frames:
        frame.columns = frame.columns.astype(int)
    root = pd.read_csv(directory/"root_voltage.csv", index_col=0, float_precision="round_trip").iloc[:, 0]
    return (*frames, root)


def decode_first_extension(job, event, primal):
    initial = read(RUN/"jobs"/job/"initialization.json")["initial_supports"]
    family = [tuple(s) for s in event.get("old_support_indices", initial)]
    slices = {k: slice(*v) for k, v in event["variable_slices"].items()}
    z = primal[slices["z"]]
    if np.max(abs(z-np.rint(z))) > 1e-7:
        raise ValueError("nonintegral support")
    groups = event.get("terminal_partition") or [[i] for i in range(len(z))]
    support = tuple(sorted(i for group, flag in zip(groups, np.rint(z), strict=True) if flag for i in group))
    if not support or support in family:
        raise ValueError("empty or duplicate support")
    family.append(support)
    r = np.maximum(0., np.r_[primal[slices["old_r"]], primal[slices["new_r"]]])
    x = np.maximum(0., np.r_[primal[slices["old_x"]], primal[slices["new_x"]]])
    return family, r, x


def candidate_point(labels, supports, r, x, observed, *, refit):
    from .ac_screen import ac_score
    p, q, voltage, root = observed
    y = root.to_numpy()[:, None]**2-voltage.to_numpy()**2
    matrices = []
    for coefficients in (r, x):
        matrix = np.zeros((len(labels), len(labels)))
        for support, weight in zip(supports, coefficients, strict=True):
            matrix[np.ix_(support, support)] += weight
        matrices.append(matrix)
    objective = float(np.mean(abs(y-p.to_numpy()@matrices[0].T-q.to_numpy()@matrices[1].T)))
    return {"supports": [list(s) for s in supports], "r": list(r), "x": list(x),
        "physical_l1_objective": objective,
        "ac": ac_score(labels, supports, r, x, observed, refit=refit)}


def analyze(output):
    sys.path.insert(0, str(RUNTIME))
    from .ac_screen import select_ac_one_se
    protocol = read(output/"protocol.json")
    rows, evidence = [], {}
    for model in protocol["models"]:
        job = model["job_id"]
        directory = RUN/"jobs"/job
        saved = read(directory/"model.json")
        labels = saved["terminal_labels"]
        observed = observations(job)
        base = saved["path"][0]
        points = [{"name": "current", **candidate_point(labels, base["support_indices"],
            base["r_values"], base["x_values"], observed, refit=True)}]
        targets = [("historical_exact", np.load(ROOT/model["raw"])["primal"], None)]
        for gap in protocol["gaps"]:
            target = output/f"{job}__gap_{gap:g}"
            result = read(target/"result.json")
            if (target/"primal.npz").exists():
                targets.append((f"gap_{gap:g}", np.load(target/"primal.npz")["primal"], result))
        for name, primal, result in targets:
            supports, r, x = decode_first_extension(job, model["event"], primal)
            point = candidate_point(labels, supports, r, x, observed, refit=True)
            point["name"] = name
            if result:
                if abs(point["physical_l1_objective"]-result["objective"]) > 1e-8:
                    raise ValueError("decoded candidate objective disagrees with raw MILP")
                point["decision"] = extension_decision(model["current_objective"], result["objective"],
                    result["bound"], feasible=result["primal_audit"]["feasible"])
            points.append(point)
            print(job, name, point["physical_l1_objective"], point["ac"].get("relative_voltage_rmse"), flush=True)
        rows.append({"job_id": job, "points": points, "ac_one_se_index": select_ac_one_se(points),
            "scope": "first-extension candidate comparison; AC criteria use all fitting rows"})
        write(output/"ac_analysis.json", rows)
        for path in directory.glob("*.json"):
            if path.name in ("model.json", "initialization.json", "payload.json"):
                evidence[str(path.relative_to(ROOT))] = sha(path)
        for path in (Path(read(directory/"payload.json")["input_dir"])/"observed").glob("*.csv"):
            evidence[str(path.relative_to(ROOT))] = sha(path)
    for path in (Path(__file__), Path(__file__).with_name("ac_screen.py"), Path(__file__).with_name("policy.py")):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    write(output/"ac_analysis_manifest.json", {"input_sha256": evidence,
        "truth_used": False, "impedance_refit": "local voltage residual fit, measured PQ/root fixed; 12 function evaluations; bounds [0,1] pu"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/wzzt_solver_tuning_20261027/gap_profiles")
    args = parser.parse_args()
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
