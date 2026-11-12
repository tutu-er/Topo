"""Read-only structural diagnostics of the finished factorial experiment.

No LP, MILP, refit or new validation selection is run. Candidate coverage is
scored against independently reconstructed physical descendant sets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def supports(point):
    return tuple(sorted(tuple(C) for C in point["supports"]))


def topology(point, n):
    return frozenset(tuple(C) for C, r, x in zip(point["supports"], point["r"], point["x"])
                     if 1 < len(C) < n and max(r, x) > 1e-9)


def selected_stage(result):
    trajectory = result["trajectory"]
    path = trajectory["path"]
    index = result["selected"]["index"]
    if index == len(path):
        return "singleton"
    exit_accepted = bool(result.get("exit_record", {}).get("accepted"))
    exit_zero = any(row.get("accepted") and row.get("stage") == "final_exit"
                    for row in trajectory.get("zero_events", []))
    prefix_length = len(path) - int(exit_accepted) - int(exit_zero)
    return "pre_exit_path" if index < prefix_length else "exit_derived_path"


def collect_diagnostics(output: Path) -> dict:
    """Return lost-exact and candidate-pool coverage tables plus provenance."""
    output = Path(output).resolve()
    receipt = read(output / "analysis/verification.json")
    assert receipt["passed"] and receipt["verified"] == receipt["planned"] == 960
    protocol = read(output / "protocol.json")
    catalog = read(output / "catalog.json")
    frame = pd.read_csv(output / "analysis/per_model.csv")
    assert len(frame) == 960 and frame.status.eq("verified").all()
    re = frame[frame.method == "R1T0E1"].set_index("job")
    full = frame[frame.method == "R1T1E1"].set_index("job").reindex(re.index)
    lost_jobs = set(re.index[re.exact & ~full.exact])
    assert len(lost_jobs) == 6 and not (full.exact & ~re.exact).any()
    coverage, lost, mechanisms = [], [], []
    truth_cache = {}
    source_hashes = {}

    def saved(path):
        source_hashes[str(path.relative_to(output)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)

    for job in protocol["jobs"]:
        meta = catalog[job]
        common = saved(output / "common" / job / "common.json")
        labels = common["labels"]
        n = len(labels)
        key = (meta["truth_network"], tuple(labels))
        if key not in truth_cache:
            physical = physical_truth(Path(meta["truth_network"]), labels)
            positions = {v: i for i, v in enumerate(labels)}
            truth_cache[key] = frozenset(tuple(sorted(positions[v] for v in C)) for C in physical["clades"])
        truth = truth_cache[key]
        row_header = {k: meta[k] for k in ("panel", "case", "N", "voltage_sigma", "condition")}
        results = {}
        for config in protocol["methods"]:
            method = config["id"]
            result = saved(output / "jobs" / job / method / "result.json")
            results[method] = result
            t = result["trajectory"]
            tails = [e for e in t.get("attempts", []) if e.get("kind") == "joint_tail"]
            zeros = [e for e in t.get("zero_events", []) if e.get("accepted")]
            mechanisms.append(dict(job=job, **row_header, method=method, selected_stage=selected_stage(result),
                                   path_length=len(t["path"]), selected_index=result["selected"]["index"],
                                   tail_calls=len(tails), tail_accepted=sum(bool(e.get("accepted")) for e in tails),
                                   exit_accepted=bool(result.get("exit_record", {}).get("accepted")),
                                   exit_warm_origin=result.get("exit_record", {}).get("warm_origin"),
                                   zero_events=len(zeros), zero_released_sets=sum(len(e["removed"]) for e in zeros),
                                   stop_reason=t.get("stop_reason")))
            if not config["exit"]:
                continue
            event = result["exit_event"]
            pool = frozenset(tuple(C) for C in event["pool"])
            coverage.append(dict(job=job, **row_header, method=method, all_true_in_pool=truth <= pool,
                                 true_clades=len(truth), true_in_pool=len(truth & pool),
                                 true_recall=len(truth & pool) / len(truth) if truth else 1.,
                                 pool_count=len(pool), accepted=bool(result["exit_record"].get("accepted"))))
        if job not in lost_jobs:
            continue
        a, b = results["R1T0E1"], results["R1T1E1"]
        a_choice, b_choice = a["selected"], b["selected"]
        a_point, b_point = a_choice["point"], b_choice["point"]
        a_topology = topology(a_point, n)
        candidates = b["output_candidates"]
        same_family = [q for q in candidates if supports(q["point"]) == supports(a_point)]
        same_topology = [q for q in candidates if topology(q["point"], n) == a_topology]
        tails = [e for e in b["trajectory"].get("attempts", []) if e.get("kind") == "joint_tail"]
        lost.append(dict(job=job, **row_header,
            RE_F1=float(re.loc[job, "clade_f1"]), RTE_F1=float(full.loc[job, "clade_f1"]),
            RE_train=float(a_point["objective"]), RTE_train=float(b_point["objective"]),
            RE_validation=float(a_choice["validation"]), RTE_validation=float(b_choice["validation"]),
            RE_internal=len(topology(a_point, n)), RTE_internal=len(topology(b_point, n)),
            RE_path_length=len(a["trajectory"]["path"]), RTE_path_length=len(b["trajectory"]["path"]),
            RE_selected_index=a_choice["index"], RTE_selected_index=b_choice["index"],
            RE_selected_stage=selected_stage(a), RTE_selected_stage=selected_stage(b),
            RE_exit_accepted=bool(a["exit_record"].get("accepted")),
            RTE_exit_accepted=bool(b["exit_record"].get("accepted")),
            RTE_tail_calls=len(tails), RTE_tail_accepted=sum(bool(e.get("accepted")) for e in tails),
            RE_selected_family_in_RTE_candidates=bool(same_family),
            RE_selected_topology_in_RTE_candidates=bool(same_topology),
            best_RTE_same_family_validation=min((q["validation"] for q in same_family), default=None),
            best_RTE_same_topology_validation=min((q["validation"] for q in same_topology), default=None),
            RE_stop=a["trajectory"].get("stop_reason"), RTE_stop=b["trajectory"].get("stop_reason")))
    coverage = pd.DataFrame(coverage)
    coverage_summary = coverage.groupby("method", sort=False).agg(
        inputs=("job", "size"), all_true_in_pool=("all_true_in_pool", "sum"),
        true_in_pool_mean=("true_in_pool", "mean"), true_recall=("true_recall", "mean"),
        pool_count=("pool_count", "mean"), exit_accepted=("accepted", "sum")).reset_index()
    lost = pd.DataFrame(lost)
    mechanisms = pd.DataFrame(mechanisms)
    assert len(coverage) == 480 and len(lost) == 6
    assert not lost.RE_exit_accepted.any() and not lost.RTE_exit_accepted.any()
    assert (lost.RTE_tail_accepted > 0).all()
    assert (lost.RTE_validation > lost.RE_validation).all()
    return dict(lost_exact_cases=lost, exit_coverage=coverage, coverage_summary=coverage_summary,
                mechanism_inputs=mechanisms,
                metadata=dict(new_solver_calls=0, lost_exact_cases=6,
                    exact_RE_topology_absent_from_RTE_output_candidates=int((~lost.RE_selected_topology_in_RTE_candidates).sum()),
                    all_lost_cases_have_both_exits_rejected=True,
                    all_lost_cases_have_accepted_tail=True,
                    all_lost_cases_have_worse_RTE_validation=True,
                    source_hashes=source_hashes,
                    diagnostic_note="saved candidate/path provenance and physical coverage; no refit, selection or optimization"))
