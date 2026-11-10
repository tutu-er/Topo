"""Replay saved stage endpoints after the complete native benchmark verifies.

No optimization or refitting. Endpoint indices follow the frozen engine's path
append rules, never nearest-loss matching. Nonterminal/interleaved T traces are
reported unresolved rather than assigning an inferred tree to a stage.

Direct execution needs the Topo Python environment (NumPy and pandas), with no
PYTHONPATH setup: the independent physical-truth helper is loaded by its absolute
verified frozen-source filename and has no project-package imports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ENGINE_SHA256 = "8b3022104745a15eee6c69e4a7625fbf4672de7b449c45165fd60904f194df2a"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def point_key(point):
    """Canonical exact coefficients, excluding derived scalar scores."""
    rows = sorted((tuple(C), float(r), float(x)) for C, r, x in zip(point["supports"], point["r"], point["x"]))
    return tuple(rows)


def saved_timing(result, *, tolerance=1e-6):
    """Disjoint top-level accounting; T call clocks are a prefix subset."""
    trajectory = result["trajectory"]
    tail = [event for event in trajectory["attempts"] if event.get("kind") == "joint_tail"]
    tail_seconds = [float(event["wall_seconds"]) for event in tail]
    total = float(result["total_seconds"])
    common = float(result["common_seconds"])
    prefix = float(trajectory["seconds"])
    exit_seconds = float(trajectory["final_exit_seconds"])
    values = [total, common, prefix, exit_seconds, *tail_seconds]
    if any(not np.isfinite(value) or value < -tolerance for value in values):
        raise ValueError("A saved stage wall clock is nonfinite or negative.")
    tail_total = sum(tail_seconds)
    overhead = total-common-prefix-exit_seconds
    if overhead < -tolerance:
        raise ValueError("Saved top-level timing breakdown has negative remaining overhead.")
    if tail_total > prefix+tolerance:
        raise ValueError("T-call wall clocks exceed their containing prefix clock.")
    successful = float(result.get("successful_total_seconds", total))
    if abs(successful-total) > tolerance:
        raise ValueError("Native total is not the successful returned-attempt total.")
    return dict(common_seconds=common, total_seconds=total, prefix_seconds=prefix,
                T_call_wall_seconds=tail_total, T_calls=len(tail), E_finish_seconds=exit_seconds,
                remaining_overhead_seconds=overhead, timing_nonnegative_tolerance=tolerance,
                timing_valid=True, T_is_subset_of_prefix=True, T_must_not_be_added_again=True,
                interrupted_proxy_charge_seconds=float(result.get("interrupted_attempt_seconds", 0.)),
                operational_proxy_total_seconds=float(result.get("logical_total_seconds", total)),
                timing_scope="saved successful-attempt wall clocks; shared frontend charged once",
                T_clock_scope="sum joint_tail event.wall_seconds only, excluding surrounding cleanup and other prefix work",
                E_clock_scope="trajectory.final_exit_seconds: complete native finish call",
                overhead_scope="remaining setup, I/O, output fitting/selection/export combined; no saved finer split")


def locate_stages(result):
    """Locate E and a contiguous terminal T suffix by saved control metadata.

    Every accepted E adds one point, and final zero cleanup adds at most one.
    In a terminal T-only suffix, each nonfinal T call must have accepted cleanup
    (otherwise search breaks); hence it adds one cleanup point independently of
    the number of zero-release rows produced inside that cleanup call.
    """
    trajectory, exit_record = result["trajectory"], result["exit_record"]
    path = trajectory["path"]
    if not path or exit_record.get("factorial_exit_on") is not True:
        raise ValueError("This locator expects the frozen benchmark's final-E path.")
    accepted_exit = bool(exit_record["accepted"])
    final_clean = any(event.get("stage") == "final_exit" and event.get("accepted")
                      for event in trajectory.get("zero_events", []))
    prefix_end = len(path)-1-int(accepted_exit)-int(final_clean)
    if prefix_end < 0 or point_key(path[-1]) != point_key(trajectory["point"]):
        raise ValueError("Native final-point/path append identity failed.")
    exit_point_index = prefix_end + int(accepted_exit)
    if point_key(path[exit_point_index]) != point_key(exit_record["point"]):
        raise ValueError("Saved E-refit point does not match its deterministic path position.")
    n = max(i for C in path[0]["supports"] for i in C)+1
    penalty = float(trajectory["penalty"])
    count = lambda point: sum(1 < len(C) < n for C in point["supports"])
    actual_before_score = path[prefix_end]["objective"] + penalty*count(path[prefix_end])
    if abs(actual_before_score-exit_record["before_score"]) > 1e-9:
        raise ValueError("E-before score does not validate the control-derived endpoint.")
    locations = dict(prefix_end_index=prefix_end,
        E=dict(status="resolved", before_index=prefix_end, after_index=len(path)-1,
               accepted_refit_index=exit_point_index if accepted_exit else None,
               accepted=accepted_exit, zero_cleanup_changed=final_clean,
               location_basis="accepted-E and final-cleanup append flags; exact saved-point identity"))
    attempts = trajectory["attempts"]
    tail_indices = [i for i, event in enumerate(attempts) if event.get("kind") == "joint_tail"]
    if not tail_indices:
        locations["T"] = dict(status="not_triggered", calls=0,
                              reason="module_disabled" if not trajectory.get("tail_on") else "no_tail_call")
        locations["T_calls"] = []
        return locations
    first = tail_indices[0]
    if any(event.get("kind") != "joint_tail" for event in attempts[first:]):
        locations["T"] = dict(status="unresolved", calls=len(tail_indices),
                               reason="T and WZZT interleaved; no explicit per-stage path indices saved")
        locations["T_calls"] = []
        return locations
    if trajectory["stop_reason"] not in {"joint_tail_gap_met", "joint_tail_unresolved"}:
        locations["T"] = dict(status="unresolved", calls=len(tail_indices),
                               reason="forward stop does not certify terminal T cleanup/append count")
        locations["T_calls"] = []
        return locations
    tail = [attempts[i] for i in tail_indices]
    tail_appends = sum(bool(event["accepted"]) for event in tail) + len(tail)-1
    before = prefix_end-tail_appends
    if before < 0:
        raise ValueError("Terminal T suffix has more declared appends than saved points.")
    zero_rows = [event for event in trajectory.get("zero_events", [])
                 if event.get("stage") == "tail_joint" and event.get("accepted")]
    if len(zero_rows) < len(tail)-1:
        raise ValueError("Terminal T suffix lacks the required accepted-cleanup records.")
    calls, cursor = [], before
    for j, (attempt_index, event) in enumerate(zip(tail_indices, tail)):
        old = cursor
        if abs(path[old]["objective"]-event["before"]) > 1e-9:
            raise ValueError("Control-derived T-before point fails its saved scalar audit.")
        if event["accepted"]:
            cursor += 1
            if abs(path[cursor]["objective"]-event["decoded_objective"]) > 1e-9:
                raise ValueError("Control-derived accepted T proposal fails its saved scalar audit.")
        proposal = cursor
        cleanup = j < len(tail)-1
        if cleanup:
            cursor += 1
            if not set(map(tuple,path[cursor]["supports"])) < set(map(tuple,path[proposal]["supports"])):
                raise ValueError("Required T cleanup is not a strict support deletion.")
        calls.append(dict(attempt_index=attempt_index, iteration=event.get("iteration"),
                          before_index=old, accepted_proposal_index=proposal if event["accepted"] else None,
                          after_index=cursor, accepted=bool(event["accepted"]), cleanup_changed=cleanup,
                          candidate_count=event.get("candidate_count"),
                          kept_internal_count=sum(1 < len(C) < n for C in event["keep"]),
                          released_count=len(event.get("released", [])),
                          penalty=event.get("penalty"), rejected_proposal_weights_available=bool(event["accepted"])))
    if cursor != prefix_end:
        raise ValueError("T suffix did not consume exactly the certified prefix path.")
    locations["T"] = dict(status="resolved", before_index=before, after_index=prefix_end,
                          calls=len(tail), accepted_calls=sum(bool(e["accepted"]) for e in tail),
                          cleanup_calls=len(tail)-1,
                          location_basis="contiguous terminal T-only suffix; accepted proposals plus one cleanup append per nonterminal call")
    locations["T_calls"] = calls
    return locations


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(int).tolist()
        n = len(labels)
        p, q, v = (np.asarray(archive[key], float).reshape(-1,n) for key in ("P","Q","V"))
        root = np.asarray(archive["root"], float).reshape(-1)
    return labels, (p,q,root[:,None]**2-v**2)


def metrics(point, data, truth):
    p,q,y = data
    n = p.shape[1]
    z = np.asarray([[i in C for i in range(n)] for C in point["supports"]],float)
    R = (z.T*np.asarray(point["r"]))@z
    X = (z.T*np.asarray(point["x"]))@z
    predicted = {tuple(C) for C,r,x in zip(point["supports"],point["r"],point["x"])
                 if 1 < len(C) < n and max(r,x)>1e-9}
    tp,fp,fn = len(predicted&truth),len(predicted-truth),len(truth-predicted)
    return dict(F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1., exact=predicted==truth,
                train_MAE=float(np.mean(abs(y-p@R.T-q@X.T))),
                positive_internal=len(predicted), false_positive=fp, false_negative=fn)


def main_analysis(output):
    output = Path(output).resolve()
    protocol_file, catalog_file = output/"protocol.json", output/"catalog.json"
    protocol,catalog = read(protocol_file),read(catalog_file)
    protocol_hash = sha(protocol_file)
    methods = [method["id"] for method in protocol["methods"]]
    planned = len(protocol["jobs"])*len(methods)
    if planned != 144 or protocol["planned"] != planned:
        raise ValueError("Stage analysis requires all 144 declared native evaluations.")
    native_file = output/"analysis/verification.json"
    native = read(native_file)
    if native.get("passed") is not True or native.get("complete") is not True or native.get("verified") != planned or native.get("planned") != planned:
        raise RuntimeError("Stage scoring waits for the complete, verified planned result set.")
    if native.get("protocol_sha256") != protocol_hash:
        raise ValueError("Native verification references a different frozen protocol.")
    analyzer_hash = sha(Path(__file__).with_name("analyze.py"))
    if native.get("analyzer_sha256") != analyzer_hash:
        raise ValueError("Native analyzer differs from the version that verified this batch.")
    if protocol.get("catalog_sha256") != sha(catalog_file):
        raise ValueError("Catalog differs from the frozen protocol.")
    for name, digest in protocol["source_hashes"].items():
        if sha(output/"source"/name) != digest:
            raise ValueError("Frozen source changed: "+name)
    engine = output/"source/research_experiments/full_ablation/engine.py"
    if sha(engine) != SUPPORTED_ENGINE_SHA256:
        raise RuntimeError("Stage locator has not been audited for this engine version.")
    module_file = output/"source/research_experiments/rnj/audit_source_ac96_benchmark.py"
    spec = importlib.util.spec_from_file_location("day96_stage_physical_truth", module_file)
    physical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(physical)
    rows,calls,locations,hashes,timings=[],[],{},{},[]
    input_audit_hashes, training_hashes = {}, {}
    for job in protocol["jobs"]:
        meta = catalog[job]
        input_audit_file = output/"analysis/jobs"/job/"verified.json"
        audited_job = read(input_audit_file)
        input_audit_hashes[job] = sha(input_audit_file)
        if (audited_job.get("protocol_sha256") != protocol_hash
                or audited_job.get("analyzer_sha256") != analyzer_hash
                or audited_job.get("exporter_sha256") != native.get("exporter_sha256")):
            raise ValueError("Per-input numerical audit has a different source or protocol: "+job)
        if set(audited_job["raw_hashes"]) != set(methods):
            raise ValueError("Per-input numerical audit does not cover every method: "+job)
        expected_hashes = protocol["input_hashes"][job]
        training_hashes[job] = sha(meta["train"])
        if training_hashes[job] != expected_hashes["train"]:
            raise ValueError("Training observations differ from the frozen protocol: "+job)
        network_dir = Path(meta["truth_network"])
        if {name:sha(network_dir/name) for name in audited_job["truth_source_hashes"]} != audited_job["truth_source_hashes"]:
            raise ValueError("Physical network differs from the independent numerical audit: "+job)
        labels,training = arrays(meta["train"])
        if labels != meta["labels"]:
            raise ValueError("Training terminal order differs from the frozen catalog: "+job)
        label_index = {label:i for i,label in enumerate(labels)}
        target = physical.physical_truth(Path(meta["truth_network"]), labels)["clades"]
        truth = {tuple(sorted(label_index[label] for label in C)) for C in target}
        for method in protocol["methods"]:
            name = method["id"]
            file = output/"jobs"/job/name/"result.json"
            result = read(file)
            hashes[job+"/"+name] = sha(file)
            if audited_job["raw_hashes"].get(name) != hashes[job+"/"+name]:
                raise ValueError("Native result changed after numerical verification: "+job+"/"+name)
            if result.get("protocol_sha256") != protocol_hash or result.get("input_hashes") != {key:expected_hashes[key] for key in ("train","validation")}:
                raise ValueError("Native result references different historical inputs: "+job+"/"+name)
            located = locate_stages(result)
            locations[job+"/"+name] = located
            timing = saved_timing(result)
            timings.append(dict(job=job,method=name,case=meta["case"],bundle=meta["bundle"],
                                voltage_sigma=meta["voltage_sigma"],**timing))
            path = result["trajectory"]["path"]
            output_rows = {row["index"]:row for row in result["output_candidates"]}
            if len(output_rows) != len(path)+1 or set(output_rows) != set(range(len(path)+1)):
                raise ValueError("Saved output library does not map one-to-one to path indices plus singleton.")
            for stage in ("T","E"):
                location = located[stage]
                row = dict(job=job,method=name,stage=stage,status=location["status"],
                           case=meta["case"],bundle=meta["bundle"],voltage_sigma=meta["voltage_sigma"],
                           **{k:v for k,v in location.items() if k!="status"})
                row["saved_stage_wall_seconds"] = timing["T_call_wall_seconds" if stage=="T" else "E_finish_seconds"]
                row["stage_timing_scope"] = timing["T_clock_scope" if stage=="T" else "E_clock_scope"]
                row["stage_time_is_prefix_subset"] = stage=="T"
                if location["status"]=="resolved":
                    for side in ("before","after"):
                        index = location[side+"_index"]
                        raw = metrics(path[index],training,truth)
                        contracted = metrics(output_rows[index]["point"],training,truth)
                        if abs(raw["train_MAE"]-path[index]["objective"])>1e-9:
                            raise ValueError("Saved raw training objective does not replay.")
                        row.update({"raw_"+side+"_"+k:v for k,v in raw.items()})
                        row.update({"output24_"+side+"_"+k:v for k,v in contracted.items()})
                    for view in ("raw","output24"):
                        row[view+"_delta_F1"] = row[view+"_after_F1"]-row[view+"_before_F1"]
                        row[view+"_delta_train_MAE"] = row[view+"_after_train_MAE"]-row[view+"_before_train_MAE"]
                rows.append(row)
            for call in located["T_calls"]:
                row = dict(job=job,method=name,**call)
                row["saved_T_call_wall_seconds"] = result["trajectory"]["attempts"][call["attempt_index"]]["wall_seconds"]
                for side,index in (("before",call["before_index"]),("after",call["after_index"])):
                    row.update({side+"_"+k:v for k,v in metrics(path[index],training,truth).items()})
                calls.append(row)
    dest=output/"stage_effects"
    dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/"stage_inputs.csv",rows)
    write_csv(dest/"tail_calls.csv",calls)
    write_csv(dest/"timing_breakdown.csv",timings)
    aggregate=[]
    for method in protocol["methods"]:
        for stage in ("T","E"):
            subset=[r for r in rows if r["method"]==method["id"] and r["stage"]==stage]
            resolved=[r for r in subset if r["status"]=="resolved"]
            row=dict(method=method["id"],stage=stage,planned=len(protocol["jobs"]),
                     resolved=len(resolved),unresolved=sum(r["status"]=="unresolved" for r in subset),
                     not_triggered=sum(r["status"]=="not_triggered" for r in subset),
                     conditional_resolved_stage_only=True)
            row.update(stage_timing_inputs=len(subset),
                       saved_stage_wall_seconds_total=sum(r["saved_stage_wall_seconds"] for r in subset),
                       saved_stage_wall_seconds_mean=float(np.mean([r["saved_stage_wall_seconds"] for r in subset])),
                       stage_time_is_prefix_subset=stage=="T",
                       stage_timing_not_restricted_to_resolved_F1_rows=True)
            for view in ("raw","output24"):
                for side in ("before","after"):
                    row[view+"_"+side+"_F1"]=float(np.mean([r[view+"_"+side+"_F1"] for r in resolved])) if resolved else None
                    row[view+"_"+side+"_train_MAE"]=float(np.mean([r[view+"_"+side+"_train_MAE"] for r in resolved])) if resolved else None
                    row[view+"_"+side+"_exact"]=sum(r[view+"_"+side+"_exact"] for r in resolved)
            aggregate.append(row)
    write_csv(dest/"stage_summary.csv",aggregate)
    timing_summary=[]
    fields=("common_seconds","prefix_seconds","T_call_wall_seconds","E_finish_seconds",
            "remaining_overhead_seconds","total_seconds","interrupted_proxy_charge_seconds",
            "operational_proxy_total_seconds")
    for method in protocol["methods"]:
        subset=[row for row in timings if row["method"]==method["id"]]
        timing_summary.append(dict(method=method["id"],planned=len(protocol["jobs"]),inputs=len(subset),
                                  **{field:sum(row[field] for row in subset) for field in fields},
                                  T_is_subset_of_prefix=True,T_must_not_be_added_again=True))
    write_csv(dest/"timing_summary.csv",timing_summary)
    write(dest/"locations.json",locations)
    receipt=dict(passed=True,planned_native_results=protocol["planned"],native_results_complete=True,
                 native_verification_sha256=sha(native_file),native_protocol_sha256=protocol_hash,
                 native_catalog_sha256=sha(catalog_file),native_analyzer_sha256=analyzer_hash,
                 native_per_input_audit_sha256=input_audit_hashes,training_input_sha256=training_hashes,
                 frozen_source_hashes_checked=len(protocol["source_hashes"]),
                 physical_truth_source_sha256=sha(module_file),
                 physical_truth_import="absolute verified frozen-source filename; NumPy/pandas only, no PYTHONPATH dependency",
                 source_engine_sha256=sha(engine),stage_analysis_sha256=sha(__file__),result_hashes=hashes,
                 raw_effects_use_saved_fitted_path_points=True,output24_effects_use_saved_refitted_library=True,
                 loss_used_for_nearest_point_matching=False,index_derivation="frozen control-flow append counts plus exact saved point validation",
                 interleaved_or_nonterminal_tail_is_unresolved=True,
                 unresolved_T=sum(r["stage"]=="T" and r["status"]=="unresolved" for r in rows),
                 stage_deltas_are_not_factorial_ablation_effects=True,
                 stage_timings_use_saved_wall_clocks_only=True,T_timing_is_prefix_subset=True,
                 top_level_timing_identity="total = common + prefix + E_finish + remaining_overhead; T is contained in prefix",
                 finer_CPU_or_unrecorded_phase_times_inferred=False,
                 all_saved_timing_breakdowns_valid=all(row["timing_valid"] for row in timings),
                 solver_calls=0,fit_calls=0,test_observations_opened=False,truth_used_only_for_postanalysis_scoring=True)
    write(dest/"verification.json",receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!="result_hashes"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",nargs="?",default=str(ROOT/"outputs/day96_t_benchmark_20261110"))
    main_analysis(parser.parse_args().output)
