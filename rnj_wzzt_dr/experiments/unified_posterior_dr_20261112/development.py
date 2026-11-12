"""Observed ranking -> offline eligibility -> bounded zero-DR development.

No actual future day or line parameters enter the planning calculations.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from ... import io
from ...dr.pipeline import load_resources
from ..posterior_mse_20261112.deliver import write_csv
from .history_scan import BETAS, PACKAGE, ROOT, load_request, require, sha, utc
from .prospective import ForecastLaw, active_allocation, binomial_interval, failure_bound, simulate_target, spectral_moment, stream


def observed_rank(scored):
    models = scored["models"]
    ordered = sorted(range(len(models)), key=lambda i: (-models[i]["log_probability"], models[i]["index"]))
    a, b = ordered[:2]
    difference = set(map(tuple, models[a]["active_clades"]))^set(map(tuple, models[b]["active_clades"]))
    ratios = []
    for index in (a, b):
        point = scored["parameters"][index]["point"]
        active = [(tuple(sorted(scored["labels"][i] for i in C)), max(r, x))
                  for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
                  if 1 < len(C) < len(scored["labels"]) and max(r, x) > 1e-9]
        median = float(np.median([value for _, value in active])) if active else None
        if median:
            ratios.extend(value/median for C, value in active if C in difference)
    return dict(id=scored["id"], cohort=scored["cohort"], job=scored["job"], case=scored["case"],
        p_map=scored["p_map"], topologies=len(models), structure_strength=min(ratios) if ratios else None,
        disputed_supports=len(difference), targets=ordered[:2])


def prepare(run):
    require((run/"stage_a_complete.json").exists(), "Complete historical scan required")
    out = run/"development"; require(not out.exists(), "Existing development is preserved")
    out.mkdir()
    requests = {r["id"]: r for r in io.read(run/"requests.json")}
    seal = io.read(run/"selection_seal.json")
    candidates = []
    for identity, expected in seal["scores"].items():
        if requests[identity]["method"] != "101":
            continue
        path = run/"scores"/(identity+".json")
        require(sha(path) == expected, "Historical score changed")
        scored = io.read(path)
        if scored["status"] == "complete" and len(scored["models"]) > 1 and scored["p_map"] < .99:
            candidates.append(observed_rank(scored))
    candidates.sort(key=lambda r: (r["case"], -(r["structure_strength"] if r["structure_strength"] is not None else -1.), r["cohort"], r["job"]))
    io.write(out/"observed_ranking.json", dict(created_utc=utc(), rule="structure strength then cohort/job; no truth fields", candidates=candidates))
    # Truth now supplies only offline sample eligibility, never ranking or a planner input.
    evaluation = {r["id"]: r for r in io.read(run/"evaluation.json")}
    selected, counters, ledger = [], Counter(), []
    for candidate in candidates:
        grade = evaluation[candidate["id"]]
        eligible = grade["truth_covered"] and grade["top2_contains_truth"] and grade["sets"]["0.99"]["truth_retained"]
        chosen = bool(eligible and counters[candidate["case"]] < 4)
        reason = "selected" if chosen else "budget_rank_limit" if eligible else "offline_truth_qualification_failed"
        ledger.append(dict(candidate, eligible=eligible, selected=chosen, reason=reason))
        if chosen:
            selected.append(candidate); counters[candidate["case"]] += 1
    require(len(selected) <= 16, "Candidate input budget exceeded")
    io.write(out/"qualification_ledger.json", ledger)
    sources = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        destination = out/"source_snapshot"/path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
        sources[path.name] = io.ref(path)
    protocol = dict(created_utc=utc(), selection_seal=io.ref(run/"selection_seal.json"),
        observed_ranking=io.ref(out/"observed_ranking.json"), qualification=io.ref(out/"qualification_ledger.json"),
        selected=selected, sources=sources, counts_by_network=counters, betas=BETAS, eta=.9,
        paths=2048, confirmation_paths_reserved=2048, wall_seconds=3600, workers=1,
        forecast="repeat the sole already observed historical day; no true/new power",
        error_law=dict(power_relative_sd=.05, time_ar=.7, same_node_PQ_shared=True,
            independent_nodes=True, residual="iid zero-mean Gaussian at frozen historical sigma2; independent of power"),
        exponent=.5, resources="original resource contracts/capacities; original zero executor",
        beta_selection_rule="first of .90,.95,.99,.80 having a development gap for both top targets of any checked input; otherwise .90. No truth labels in this choice.",
        scope="conditional planning-law development; independent confirmation and physical AC unopened")
    io.write(out/"protocol.json", protocol)
    return out, protocol, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(); run = args.run.resolve(); began = time.perf_counter()
    out, protocol, requests = prepare(run)
    rows = []
    for number, selected in enumerate(protocol["selected"], 1):
        require(time.perf_counter()-began < protocol["wall_seconds"], "Development budget exhausted")
        identity = selected["id"]; request = requests[identity]
        scored = io.read(run/"scores"/(identity+".json"))
        native, training, validation = load_request(request)
        forecast = np.column_stack((np.vstack((training.p, validation.p)), np.vstack((training.q, validation.q))))
        law = ForecastLaw(forecast)
        H = np.asarray([p["H"] for p in scored["parameters"]])
        log_history = np.array([m["log_probability"] for m in scored["models"]])
        original, adapter, base_kw = load_resources(request["case"], labels=training.labels)
        point = adapter.nominal(adapter.zero)
        zero_device = adapter.audit_delivery(point["mean_u"])
        require(zero_device["passed"] and point["fee"] == 0 and np.all(point["mean_u"] == 0), "Original zero delivery is not feasible/free")
        for target in selected["targets"]:
            row = dict(id=identity, case=request["case"], cohort=request["cohort"], job=request["job"], target=target,
                target_original_index=scored["models"][target]["index"], target_topology_id=scored["models"][target]["topology_id"],
                historical_weight=scored["models"][target]["probability"], structure_strength=selected["structure_strength"],
                candidate_count=len(H), zero_device=zero_device, zero_fee=0., status="complete", gates={})
            try:
                moments = {j: spectral_moment(H[target]-H[j], law, scored["sigma2"]) for j in range(len(H)) if j != target}
                simulation = simulate_target(H, log_history, target, law, scored["sigma2"],
                    stream("posterior_dr_model_development_20261112", identity, target), protocol["paths"])
                np.savez_compressed(out/(identity+f"__target{target}_development.npz"), log_target_probability=simulation)
                for beta in BETAS:
                    bound = failure_bound(log_history, moments, target, beta)
                    allocation = active_allocation(bound["pairs"])
                    count = int(np.sum(simulation >= np.log(beta)))
                    interval = binomial_interval(count, len(simulation))
                    active_effect = []
                    for j in allocation["active"]:
                        dh = H[target]-H[j]; n = len(training.labels)
                        effect = dh[:, original.node]+dh[:, n+original.node]*original.qratio[None]
                        active_effect.extend(np.flatnonzero(np.any(effect != 0, axis=0)).tolist())
                    row["gates"][str(beta)] = dict(bound=bound, allocation=allocation,
                        successes=count, paths=len(simulation), probability=count/len(simulation), interval95=interval,
                        zero_certified=bound["success_lower"] >= .9,
                        gap_evidence=interval[1] < .9,
                        active_resource_count=len(set(active_effect)),
                        minimum_fee=0. if bound["success_lower"] >= .9 else None)
            except Exception as exc:
                row.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            io.write(out/"targets"/(identity+f"__target{target}.json"), row)
            rows.append(row)
        print(f"zero-DR development {number}/{len(protocol['selected'])}: {identity}", flush=True)
        io.write(out/"state.json", dict(stage="zero_development", inputs=number, planned=len(protocol["selected"]), targets=len(rows)))
    # Threshold choice is intentionally computed before joining target correctness.
    both = {}
    for beta in BETAS:
        identities = {r["id"] for r in rows}
        both[str(beta)] = sum(all(r["status"] == "complete" and r["gates"][str(beta)]["gap_evidence"]
            for r in rows if r["id"] == identity) for identity in identities)
    beta_star = next((b for b in (.9, .95, .99, .8) if both[str(b)]), .9)
    io.write(out/"threshold_lock.json", dict(created_utc=utc(), beta_star=beta_star, eta=.9,
        observed_rule=protocol["beta_selection_rule"], both_target_gap_counts=both,
        truth_labels_used_for_threshold=False, actual_future_opened=False))
    evaluation = {r["id"]: r for r in io.read(run/"evaluation.json")}
    for row in rows:
        grade = evaluation[row["id"]]
        row["target_correct"] = row["target_topology_id"] == grade["truth_id"]
        row["truth_retained_at_beta_star"] = grade["sets"][str(beta_star)]["truth_retained"]
    io.write(out/"results.json", rows)
    summary = dict(inputs=len(protocol["selected"]), targets=len(rows), beta_star=beta_star,
        by_beta={str(beta): dict(complete=sum(r["status"] == "complete" for r in rows),
            zero_certified=sum(r["status"] == "complete" and r["gates"][str(beta)]["zero_certified"] for r in rows),
            gap_evidence=sum(r["status"] == "complete" and r["gates"][str(beta)]["gap_evidence"] for r in rows),
            correct_target_gaps=sum(r["status"] == "complete" and r["target_correct"] and r["gates"][str(beta)]["gap_evidence"] for r in rows)) for beta in BETAS},
        elapsed_seconds=time.perf_counter()-began, no_new_AC=True)
    io.write(out/"summary.json", summary)
    io.write(out/"complete.json", dict(status="complete", created_utc=utc(), summary_sha256=sha(out/"summary.json"),
        protocol_sha256=sha(out/"protocol.json"), threshold_lock_sha256=sha(out/"threshold_lock.json")))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
