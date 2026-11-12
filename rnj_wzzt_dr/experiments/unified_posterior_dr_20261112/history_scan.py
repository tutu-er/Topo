"""Stage A: audit original seals -> score 800 libraries -> seal -> grade.

Only existing observed NPZ and frozen parameter points enter the scorer.
Truth is first opened by grade(), after all weights have been sealed.
"""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from ... import io
from ...observations import Observation, fingerprint, load
from ...identification.optimization import response_matrices
from ..posterior_mse_20261112.scoring import independent_check, score, topology_id
from ..posterior_mse_20261112.deliver import write_csv
from .posterior import retained

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT/"rnj_wzzt_dr"
STANDALONE = ROOT/"topology_identification_standalone"
ARCHIVES = STANDALONE/"experiments/large96_20261112/records"
BETAS = (.8, .9, .95, .99)
BATCHES = {
    "main": ("large96_main_20261112_001", 240, "ceac935650e280105c507e26ce780191fde4340cdd3a8fd3ba29028f8a6a8fc4"),
    "pressure": ("large96_pressure_20261112_001", 160, "20877c32c732d5083c2765908ccd37701251bc23bf40a23f6aa5a49de2016ec8"),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_snapshot(out):
    references = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        target = out/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        references[relative.as_posix()] = io.ref(path)
    return references


def prepare(out):
    require(not out.exists(), "Existing run must be preserved; use a new output")
    out.mkdir(parents=True)
    sources = source_snapshot(out)
    protocol = dict(schema="unified_posterior_dr_history_v1", created_utc=utc(),
        stage="history_only_before_truth", plan=io.ref(PACKAGE/"plans/unified_posterior_dr_20261112/PLAN.md"),
        commit_at_lock=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=dict(io.environment(), executable=sys.executable, threads=1),
        source_references=sources, archive_manifest=io.ref(ARCHIVES/"manifest.json"),
        betas=BETAS, methods=["101", "111"], expected=800, train_slots=64, validation_slots=32,
        formula="uniform distinct active topologies; training-only representative; sigma2=min validation MSE",
        random_draws=0, optimization_calls=0, scoring_wall_seconds=900,
        scope="completed historical development data, not a first independent blind test",
        zero_variance="failed, no artificial floor", future_rule="freeze sigma2; add SSE only")
    io.write(out/"protocol.json", protocol)
    manifest = io.read(ARCHIVES/"manifest.json")
    for volume in manifest["volumes"]:
        require(sha(ARCHIVES/volume["path"]) == volume["sha256"], "Archive volume changed")
    checked = {}

    def archive_checked(path, batch):
        relative = path.relative_to(batch).as_posix()
        member = batch.name+"/"+relative
        expected = manifest["members"][member]["sha256"]
        digest = sha(path)
        require(digest == expected, "Archived source/input/result changed: "+member)
        checked[str(path)] = digest
        return digest

    requests, batches = [], {}
    for cohort, (name, count, expected_protocol) in BATCHES.items():
        batch = STANDALONE/"outputs"/name
        require(not (batch/"run.lock").exists(), "An original manager is still locked")
        for relative in ("protocol.json", "completion.json", "selection_seal.json", "analysis/pre_truth_audit.json", "analysis/verification.json"):
            archive_checked(batch/relative, batch)
        original = io.read(batch/"protocol.json")
        completion, seal = io.read(batch/"completion.json"), io.read(batch/"selection_seal.json")
        pre, verification = io.read(batch/"analysis/pre_truth_audit.json"), io.read(batch/"analysis/verification.json")
        require(sha(batch/"protocol.json") == expected_protocol == seal["protocol_sha256"], "Protocol differs")
        require(completion["selection_seal_sha256"] == sha(batch/"selection_seal.json"), "Completion seal differs")
        require(completion["all_planned_terminal"] and seal["sealed"] and seal["all_planned_terminal"], "Batch incomplete")
        require(pre["truth_opened"] is False and pre["invalid"] == 0 and pre["model_audits"] == original["planned_methods"], "Pre-truth audits incomplete")
        require(verification["verified"] and verification["all_fitted_records_valid"] and verification["invalid_methods"] == 0, "Owning verification failed")
        require(original["split"] == dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False), "Observation split differs")
        require(len(original["inputs"]) == count, "Planned input count differs")
        for relative, expected in original["source_sha256"].items():
            path = batch/"runtime"/relative
            require(archive_checked(path, batch) == expected, "Frozen source differs")
        require(archive_checked(batch/"runtime/protocol.json", batch) == expected_protocol, "Runtime protocol differs")
        archive_checked(batch/"evaluation_map.json", batch)
        mapping = io.read(batch/"evaluation_map.json")  # IDs/noise/profile only, no true clades.
        require(sha(batch/"evaluation_map.json") == original["evaluation_map_sha256"], "Case ID map differs")
        for job, info in sorted(original["inputs"].items()):
            observed = batch/"runtime"/info["path"]
            require(archive_checked(observed, batch) == info["sha256"], "Observed input differs")
            common = batch/"common"/job/"common.json"
            common_hash = archive_checked(common, batch)
            require(seal["common_sha256"][common.relative_to(batch).as_posix()] == common_hash, "Common record differs")
            for method in ("101", "111"):
                path = batch/"runs"/job/method/"result.json"
                result_hash = archive_checked(path, batch)
                require(seal["result_sha256"][path.relative_to(batch).as_posix()] == result_hash, "Method seal differs")
                wrapper = io.read(path)
                require(wrapper["status"] == "fitted" and not wrapper.get("truth_used"), "Method terminal invalid")
                require(wrapper["protocol_sha256"] == expected_protocol and wrapper["input_sha256"] == info["sha256"], "Worker used foreign input")
                require(wrapper["common_sha256"] == common_hash and wrapper["source_sha256_before"] == original["source_sha256"] == wrapper["source_sha256_after"], "Worker provenance differs")
                require(mapping[job]["cohort"] == cohort, "Case cohort differs")
                requests.append(dict(id=f"{cohort}__{job}__{method}", job=job, method=method,
                    **mapping[job], result=io.ref(path), observed=io.ref(observed),
                    training_sha256=info["training_sha256"], selection_sha256=info["selection_sha256"],
                    batch=str(batch), truth_member=name+"/evaluation/"+job+".json"))
        batches[cohort] = dict(path=str(batch), protocol_sha256=expected_protocol,
            inputs=count, native_methods=original["planned_methods"], completion=io.ref(batch/"completion.json"),
            verification=io.ref(batch/"analysis/verification.json"))
    require(len(requests) == 800, "Expected exactly 800 planned libraries")
    io.write(out/"requests.json", requests)
    io.write(out/"source_audit.json", dict(status="passed", created_utc=utc(), sources_checked=checked,
        archive_volumes=[dict(path=v["path"], sha256=v["sha256"]) for v in manifest["volumes"]],
        batches=batches, requests_sha256=sha(out/"requests.json"), truth_opened=False,
        process_inventory="CIM unavailable; original locks absent and all terminal seals verified"))
    print("Source audit passed: 400 inputs / 800 frozen libraries; no original fits rerun", flush=True)
    return protocol, requests


def load_request(request):
    for field in ("result", "observed"):
        require(sha(request[field]["path"]) == request[field]["sha256"], "Frozen request changed")
    native = io.read(request["result"]["path"])["result"]
    require(not native["truth_used"] and not native["test_used"], "Identifier used truth")
    day = load(request["observed"]["path"])
    require(day.points == 96, "Original day has extra observations")
    train = Observation(day.p[:64], day.q[:64], day.y[:64], day.labels)
    valid = Observation(day.p[64:], day.q[64:], day.y[64:], day.labels)
    require(fingerprint(train) == request["training_sha256"] and fingerprint(valid) == request["selection_sha256"], "Split fingerprint differs")
    return native, train, valid


def calculate(out, protocol, requests):
    began = time.perf_counter()
    scores = []
    for number, request in enumerate(requests, 1):
        require(time.perf_counter()-began <= protocol["scoring_wall_seconds"], "Historical scoring wall budget exceeded")
        try:
            native, train, valid = load_request(request)
            scored = score(native, train, valid)
            scored["independent_verification"] = independent_check(native, train, valid, scored)
            scored["parameters"] = []
            for model in scored["models"]:
                point = native["frozen_candidates"][model["index"]]["point"]
                R, X = response_matrices(point, valid.n)
                scored["parameters"].append(dict(index=model["index"], point=point,
                    H=np.column_stack((R, X)), history_sse=model["validation_mse"]*valid.y.size))
            scored["sets"] = {str(beta): retained([m["log_probability"] for m in scored["models"]], beta) for beta in BETAS}
            scored.update({k: request[k] for k in ("id", "cohort", "job", "method", "case", "profile", "level", "replicate", "voltage_sigma")})
        except Exception as exc:
            scored = dict(status="failed", **{k: request[k] for k in ("id", "cohort", "job", "method", "case")},
                error=repr(exc), traceback=traceback.format_exc())
        io.write(out/"scores"/(request["id"]+".json"), scored)
        scores.append(scored)
        if number % 50 == 0:
            io.write(out/"state.json", dict(stage="scoring", completed=number, planned=800,
                failed=sum(s["status"] != "complete" for s in scores), elapsed_seconds=time.perf_counter()-began))
            print(f"scored {number}/800", flush=True)
    for relative, ref in protocol["source_references"].items():
        require(sha(ref["path"]) == ref["sha256"], "Scoring source changed: "+relative)
    seal = dict(status="weights_frozen_before_this_task_truth_read", created_utc=utc(),
        protocol_sha256=sha(out/"protocol.json"), requests_sha256=sha(out/"requests.json"),
        source_audit_sha256=sha(out/"source_audit.json"),
        scores={s["id"]: sha(out/"scores"/(s["id"]+".json")) for s in scores},
        scoring_seconds=time.perf_counter()-began, complete=sum(s["status"] == "complete" for s in scores))
    io.write(out/"selection_seal.json", seal)
    return scores


def grade(out, protocol, requests, scores):
    seal = io.read(out/"selection_seal.json")
    require(seal["protocol_sha256"] == sha(out/"protocol.json"), "Protocol changed before grading")
    manifest = io.read(ARCHIVES/"manifest.json")
    truths, rows, audit_refs = {}, [], {}
    for request, scored in zip(requests, scores, strict=True):
        require(seal["scores"][request["id"]] == sha(out/"scores"/(request["id"]+".json")), "Sealed score changed")
        row = {k: request[k] for k in ("id", "cohort", "job", "method", "case", "profile", "level", "replicate", "voltage_sigma")}
        if scored["status"] != "complete":
            rows.append(dict(row, status="failed")); continue
        key = request["cohort"], request["job"]
        if key not in truths:
            path = Path(request["batch"])/"evaluation"/(request["job"]+".json")
            require(sha(path) == manifest["members"][request["truth_member"]]["sha256"], "Original truth record changed")
            truth = io.read(path)
            require(truth["labels"] == scored["labels"], "Truth node order differs")
            clades = tuple(sorted(tuple(sorted(scored["labels"][i] for i in C)) for C in truth["proper_clades"]))
            truths[key] = clades, topology_id(clades)
            audit_refs[str(path)] = sha(path)
        real, truth_id = truths[key]
        models = scored["models"]
        ordered = sorted(range(len(models)), key=lambda i: (-models[i]["log_probability"], models[i]["index"]))
        best = models[ordered[0]]
        present = next((i for i, m in enumerate(models) if m["topology_id"] == truth_id), None)
        predicted = set(map(tuple, best["active_clades"])); reality = set(real)
        row.update(status="complete", truth_id=truth_id, truth_clades=real,
            truth_covered=present is not None, map_correct=best["topology_id"] == truth_id,
            native_correct=scored["native_topology_id"] == truth_id, map_matches_native=scored["map_matches_native_topology"],
            missing=len(reality-predicted), extra=len(predicted-reality), p_map=scored["p_map"],
            p_true=models[present]["probability"] if present is not None else None,
            raw_points=scored["raw_points"], topologies=len(models), sigma2=scored["sigma2"], cells=scored["cells"],
            top2_contains_truth=any(models[i]["topology_id"] == truth_id for i in ordered[:2]),
            sets={})
        for beta in BETAS:
            kept = scored["sets"][str(beta)]
            contains = present in kept if present is not None else False
            row["sets"][str(beta)] = dict(size=len(kept), correct_singleton=len(kept) == 1 and contains,
                wrong_singleton=len(kept) == 1 and not contains, true_excluded=present is not None and not contains,
                truth_retained=contains, wrong_rejected=len(models)-len(kept)-(int(present is not None and not contains)))
        rows.append(row)
    io.write(out/"evaluation.json", rows)
    io.write(out/"evaluation_receipt.json", dict(created_utc=utc(), truth_read_after_seal=True,
        selection_seal_sha256=sha(out/"selection_seal.json"), truth_references=audit_refs))
    return rows


def deliver(out, protocol, rows):
    public = out/"public"; public.mkdir()
    summary = {}
    for cohort in BATCHES:
        for method in ("101", "111"):
            all_rows = [r for r in rows if r["cohort"] == cohort and r["method"] == method]
            valid = [r for r in all_rows if r["status"] == "complete"]
            summary[cohort+"/"+method] = dict(planned=len(all_rows), valid=len(valid),
                failed=len(all_rows)-len(valid), covered=sum(r["truth_covered"] for r in valid),
                map_correct=sum(r["map_correct"] for r in valid), native_correct=sum(r["native_correct"] for r in valid),
                canonical_changes=sum(not r["map_matches_native"] for r in valid),
                topologies=dict(Counter(r["topologies"] for r in valid)),
                gates={str(beta): dict(correct_singleton=sum(r["sets"][str(beta)]["correct_singleton"] for r in valid),
                    wrong_singleton=sum(r["sets"][str(beta)]["wrong_singleton"] for r in valid),
                    true_excluded=sum(r["sets"][str(beta)]["true_excluded"] for r in valid),
                    retained_sizes=dict(Counter(r["sets"][str(beta)]["size"] for r in valid))) for beta in BETAS})
    io.write(out/"summary.json", summary)
    io.write(public/"summary.json", summary)
    flat = []
    for row in rows:
        if row["status"] != "complete":
            flat.append({k: row[k] for k in ("id", "cohort", "job", "method", "case", "status")}); continue
        for beta in BETAS:
            flat.append({**{k: row[k] for k in ("id", "cohort", "job", "method", "case", "profile", "level", "replicate", "voltage_sigma", "status", "truth_covered", "map_correct", "native_correct", "p_map", "p_true", "topologies", "missing", "extra", "top2_contains_truth")},
                         "beta": beta, **row["sets"][str(beta)]})
    write_csv(public/"history_libraries.csv", flat)
    lines = ["# 已封存large96数据的统一后验评价", "", "复用原64训练＋32验证；无新拟合、控制或AC。", "",
             "| 批次／方法 | 有效／计划 | 库含真 | MAP完整恢复 | 90%正确唯一 | 90%错误唯一 | 90%真族误拒 | 99%仍多候选 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in summary.items():
        gate = group["gates"]["0.9"]
        many = sum(n for size, n in group["gates"]["0.99"]["retained_sizes"].items() if int(size) > 1)
        lines.append(f"| {name} | {group['valid']}/{group['planned']} | {group['covered']} | {group['map_correct']} | {gate['correct_singleton']} | {gate['wrong_singleton']} | {gate['true_excluded']} | {many} |")
    lines += ["", "800库包含复用观测和配对方法，只报告描述性计数。库缺真与库含真误拒分别保留。",
              "", "原始MSE、sigma²、拟合参数和完整来源留在私人输出。资格检查和DR尚未完成。"]
    (public/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    io.write(public/"RECEIPT.json", dict(stage="history_scoring_complete", planned=800,
        valid=sum(r["status"] == "complete" for r in rows),
        protocol_sha256=sha(out/"protocol.json"), selection_seal_sha256=sha(out/"selection_seal.json"),
        source_audit_sha256=sha(out/"source_audit.json"), environment=protocol["environment"],
        optimization_calls=0, new_physical_observations=0))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    began = time.perf_counter()
    try:
        protocol, requests = prepare(out)
        with (out/"run.lock").open("x", encoding="utf-8") as lock:
            lock.write(json.dumps(dict(pid=os.getpid(), started_utc=utc())))
        scores = calculate(out, protocol, requests)
        rows = grade(out, protocol, requests, scores)
        deliver(out, protocol, rows)
        require(all(r["status"] == "complete" for r in rows), "Historical stage has failed libraries")
        io.write(out/"stage_a_complete.json", dict(status="complete", created_utc=utc(), planned=800,
            valid=800, elapsed_seconds=time.perf_counter()-began,
            selection_seal_sha256=sha(out/"selection_seal.json")))
    except Exception:
        if out.exists():
            io.write(out/"failure.json", dict(status="failed", created_utc=utc(), traceback=traceback.format_exc()))
        raise
    finally:
        if (out/"run.lock").exists():
            (out/"run.lock").unlink()


if __name__ == "__main__":
    main()
