"""Audit saved physics/data, summarize all denominators, and package evidence.

This does not select parameters, fit models, or tune a scientific setting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from .assets import build_network
from .experiment import ROOT, sha, write, trajectory, simulate, inference_payload


def finish(folder, evidence):
    folder, evidence = Path(folder), Path(evidence)
    if evidence.exists():
        raise FileExistsError("use a fresh evidence folder")
    config = json.loads((folder/"configuration.json").read_text(encoding="utf-8"))
    results = json.loads((folder/"results.json").read_text(encoding="utf-8"))
    completed = json.loads((folder/"completed.json").read_text(encoding="utf-8"))
    issues = []
    for relative, expected in config["source_sha256"].items():
        if sha(ROOT/relative.replace("\\", "/")) != expected:
            issues.append(f"source changed: {relative}")
    if completed["results_sha256"] != sha(folder/"results.json"):
        issues.append("saved result hash mismatch")
    if results["config_sha256"] != sha(folder/"configuration.json"):
        issues.append("saved configuration hash mismatch")
    audit = dict(source_files_checked=len(config["source_sha256"]), raw_files_checked=0,
                 ac_replays=0, v_replay_max_pu=0., noisy_reading_replay_max=0.,
                 independent_history_seed_count=len(config["seeds"]),
                 exact_source_hashes=True, same_noise_across_profiles=True,
                 reserved_rows_used_for_fitting_or_selection=False,
                 actual_mle_fits=results["mle_fits"], actual_ev_planning_calls=results["ev_planning_calls"])
    paired = {}
    by_condition = defaultdict(list)
    for row in results["rows_data"]:
        by_condition[row["network"], row["profile"], row["seed"]].append(row)
    manifest = {}
    for (name, profile, seed), rows in by_condition.items():
        if any(r["status"] != "completed" for r in rows):
            continue  # Failures stay in the main result denominator.
        relative = rows[0]["raw_file"]; path = folder/relative
        if sha(path) != rows[0]["raw_sha256"]:
            issues.append(f"raw hash mismatch: {relative}")
        manifest[relative] = sha(path)
        with np.load(path, allow_pickle=False) as saved:
            net = build_network(name, profile); data = trajectory(net, seed)
            state = simulate(net, data)
            payload = inference_payload(net, data, state, 672)
            for key in ("p_kw", "q_kvar", "root_v"):
                if not np.array_equal(saved[key], data[key]):
                    issues.append(f"truth generation mismatch: {relative}/{key}")
            v_error = float(np.max(abs(state["v"]-saved["v_true"])))
            audit["v_replay_max_pu"] = max(audit["v_replay_max_pu"], v_error)
            for key in ("p", "q", "v"):
                error = float(np.max(abs(saved[f"{key}_obs"]-payload["observed"][key])))
                audit["noisy_reading_replay_max"] = max(audit["noisy_reading_replay_max"], error)
            if not np.array_equal(saved["root_obs"], payload["root_meter"][0]):
                issues.append(f"root meter generation mismatch: {relative}")
            current = tuple(saved[k].tobytes() for k in ("p_kw", "q_kvar", "root_v"))
            current += tuple(data["white"][k].tobytes() for k in ("p", "q", "v", "root"))
            paired_key = (name, seed)
            if paired_key in paired and paired[paired_key] != current:
                issues.append(f"profile paths not paired: {name}/{seed}")
            paired[paired_key] = current
        audit["raw_files_checked"] += 1; audit["ac_replays"] += 1
    if audit["v_replay_max_pu"] > 1e-12 or audit["noisy_reading_replay_max"] > 1e-12:
        issues.append("saved-state replay error")
    groups = defaultdict(list)
    for row in results["rows_data"]:
        groups[row["network"], row["profile"], row["N"]].append(row)
    stats = []
    for (name, profile, n), rows in sorted(groups.items()):
        good = [r for r in rows if r["status"] == "completed"]
        record = dict(network=name, profile=profile, N=n, total=len(rows), completed=len(good),
                      failed=len(rows)-len(good))
        if good:
            for key in ("root_drop_mean_pu", "root_drop_max_pu", "voltage_min",
                        "max_ampacity_fraction", "edge_drop_rms_pu", "terminal_difference_rms_pu",
                        "wrong_root_v_rmse_pu", "naive_port_v_rmse_pu"):
                values = [r[key] for r in good if r.get(key) is not None]
                if values:
                    record[key] = dict(mean=float(np.mean(values)), minimum=min(values), maximum=max(values))
            record["voltage_envelope_failures"] = sum(not r["operational_voltage_ok"] for r in good)
            record["ampacity_failures"] = sum(r["ampacity_ok"] is False for r in good)
        stats.append(record)
    audit.update(issues=issues, passed=not issues,
                 failed_conditions=results["failed_conditions"], total_conditions=results["conditions"],
                 prefix_rows=len(results["rows_data"]), source_config_sha256=sha(folder/"configuration.json"),
                 run_seconds=results["seconds"], result_sha256=sha(folder/"results.json"))
    evidence.mkdir(parents=True)
    # Preserve original source bytes, including line endings, independently of
    # Git autocrlf and future checkout paths. Numerical libraries are versioned
    # in the configuration rather than copied into this small archive.
    archive = evidence/"scientific_sources.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for relative in sorted(config["source_sha256"]):
            member = relative.replace("\\", "/")
            z.writestr(member, (ROOT/member).read_bytes())
    with zipfile.ZipFile(archive) as z:
        for relative, expected in config["source_sha256"].items():
            if hashlib.sha256(z.read(relative.replace("\\", "/"))).hexdigest() != expected:
                issues.append(f"source archive recovery mismatch: {relative}")
    audit.update(scientific_source_archive_sha256=sha(archive),
                 audit_script_sha256=sha(__file__), issues=issues, passed=not issues)
    for name in ("configuration.json", "results.json", "conditions.csv", "completed.json"):
        shutil.copyfile(folder/name, evidence/name)
    shutil.copytree(folder/"networks", evidence/"networks")
    write(evidence/"independent_audit.json", audit)
    write(evidence/"aggregate.json", stats)
    write(evidence/"all_raw_hashes.json", manifest)
    for name in ("paper15_cable_v1_35001.npz", "paper11_mixed_v1_35001.npz"):
        path = folder/"raw"/name
        if path.exists():
            (evidence/"representative_raw").mkdir(exist_ok=True)
            shutil.copyfile(path, evidence/"representative_raw"/name)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if issues:
        raise AssertionError("saved-state audit failed; evidence retains issues")
    return audit, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    finish(args.input, args.evidence)
