"""Frozen continuous-price effect pilot; reuse existing physical and RNJ/W calls.

Observe -> original fit workers -> freeze and score -> aggregate. No topology
search or power-flow equations are implemented in this driver.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[4]
TASK = REPO/"research_experiments/wzzt_dr_selection"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def bound(item):
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise RuntimeError("frozen input hash changed: "+str(path))
    return path


def setup():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    for folder in (REPO, REPO/"rnj_wzzt_core"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))


def configuration(root):
    setup()
    from research_experiments.wzzt_dr_selection.dr_day_modules.run_day_trial import config
    return config(root)


def observe(root):
    """Run new physical workers only after the complete price/source lock."""
    setup()
    import numpy as np
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import observation_identity
    root, c = configuration(root)
    phase = root/"test"
    phase.mkdir(exist_ok=True)
    if (phase/"observations.json").exists() or (phase/"observation_progress.json").exists():
        raise FileExistsError("an observation batch already started; keep its receipt rather than silently restart")
    lock = phase/"observations.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    rows, jobs = [], {}
    planned = len(c["policies"])*len(c["test_paths"])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(REPO), str(REPO/"rnj_wzzt_core")))
    try:
        with bound(c["history_packet"]).open("rb") as stream:
            history = pickle.load(stream)["history"]
        for path in c["test_paths"]:
            # These joint response factors are frozen independently of price,
            # observations and truth. Same case/path shares them across policies.
            factors = [np.random.default_rng(np.random.SeedSequence(
                       [c["response_seed"], c["case_index"], path, event])).uniform(-1, 1, 2).tolist()
                       for event in range(4)]
            for policy in c["policies"]:
                rid = f"{policy}__path{path}"
                folder = phase/"records"/rid
                folder.mkdir(parents=True, exist_ok=False)
                request = dict(status="frozen", case=c["case"], case_index=c["case_index"],
                    repo_root=str(REPO), source_root=c["source_root"], source_metadata=c["source_metadata"],
                    labels=c["labels"], root_label=c["root_label"], base_kw=c["base_kw"],
                    voltage_sigma=c["voltage_sigma"], pq_sigma=c["pq_sigma"], phase="test",
                    record_id=rid, policy=policy, path=path, replicate_base=c["replicate_base"],
                    event_profiles=c["event_profiles"], response_seed=c["response_seed"],
                    factors_by_event=factors, plan_packet=c["plans"][policy],
                    problem=c["problem"], protocol=ref(root/"protocol.json"),
                    source_sha256=c["source_sha256"], incentive_budget=c["hard_four_event_budget"],
                    created_utc=now())
                request_path = folder/"request.json"
                write(request_path, request)
                row = dict(record_id=rid, case=c["case"], policy=policy, path=path,
                    status="started", request_sha256=sha(request_path),
                    physical_output=str(folder/"physical"), factors_by_event=factors)
                try:
                    command = [sys.executable, "-B", "-m",
                        "research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test.physical_worker",
                        "--request", str(request_path), "--expected-sha", sha(request_path),
                        "--output", str(folder/"physical")]
                    with (folder/"physical.log").open("w", encoding="utf-8") as log:
                        process = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, timeout=240)
                    receipt_path = folder/"physical/events_receipt.json"
                    receipt = read(receipt_path)
                    row.update(physical_status=receipt["status"],
                        operation_checks=receipt.get("all_operation_checks_pass"),
                        actual_payment=receipt.get("actual_payment"),
                        worst_payment=receipt.get("worst_payment"),
                        physical_receipt_sha256=sha(receipt_path),
                        physical_exit_code=process.returncode,
                        excitation_physical=receipt.get("excitation_physical"),
                        excitation_observed=receipt.get("excitation_observed"),
                        excitation_threshold=receipt.get("excitation_threshold"),
                        excitation_physical_joint_pass=receipt.get("excitation_physical_joint_pass"),
                        excitation_observed_joint_pass=receipt.get("excitation_observed_joint_pass"))
                    if receipt["status"] != "complete" or not receipt.get("all_operation_checks_pass"):
                        row["status"] = "physical_failed"
                    else:
                        packet_path = folder/"physical/packet.pkl"
                        with packet_path.open("rb") as stream:
                            packet = pickle.load(stream)
                        if sum(len(v["P_terminal"]) for v in packet["training"]) != 192:
                            raise RuntimeError("future training is not two complete days")
                        if sum(len(v["P_terminal"]) for v in packet["validation"]) != 192:
                            raise RuntimeError("future validation is not two complete days")
                        key, identity = observation_identity(
                            history["training"]+packet["training"], history["validation"]+packet["validation"],
                            root_label=c["root_label"], search_seconds=c["search_seconds"],
                            sources=c["algorithm_source_sha256"], lp_method=c["lp_method"])
                        job = jobs.setdefault(key, dict(key=key, identity=identity, packet=ref(packet_path), record_ids=[]))
                        job["record_ids"].append(rid)
                        row.update(status="observed", packet=ref(packet_path), fit_key=key)
                except Exception as error:
                    row.update(status="physical_failed", error_type=type(error).__name__,
                               message=str(error), traceback=traceback.format_exc())
                rows.append(row)
                write(phase/"observation_progress.json", dict(status="running", planned=planned,
                      rows=rows, unique_jobs=len(jobs)))
                print(json.dumps(dict(stage="physical", case_index=c["case_index"], done=len(rows),
                    planned=planned, policy=policy, path=path, status=row["status"])), flush=True)
        ordered = [jobs[key] for key in sorted(jobs)]
        for index, job in enumerate(ordered):
            job["worker"] = index % c["workers"]
        write(phase/"observations.json", dict(status="frozen", protocol_sha256=sha(root/"protocol.json"),
              rows=rows, jobs=ordered, planned=planned, created_utc=now()))
    finally:
        lock.unlink(missing_ok=True)


def fit(root, worker):
    setup()
    from research_experiments.wzzt_dr_selection.dr_day_modules.run_day_trial import fit_worker
    return fit_worker(root, "test", worker)


def finish(root):
    setup()
    from research_experiments.wzzt_dr_selection.dr_day_modules.run_day_trial import finish as existing
    existing(root, "test")
    root, c = configuration(root)
    target = tuple(c.get("target_labels", []))
    mechanisms = []
    for row in read(root/"test/scored.json")["rows"]:
        record = dict(record_id=row["record_id"], policy=row["policy"], path=row["path"], status=row["status"])
        if row["status"] == "complete" and target:
            result = read(bound(row["result"]))
            labels = result["terminal_labels"]
            def has(point):
                return target in {tuple(sorted(labels[i] for i in support)) for support in point["supports"]}
            candidates = result["selection_candidates"]
            best = min(point["validation_mae"] for point in candidates)
            with_target = [p["validation_mae"] for p in candidates if has(p)]
            without_target = [p["validation_mae"] for p in candidates if not has(p)]
            model = result["models"]["multistart_original_rule"]
            record.update(target_labels=list(target),
                target_in_pool=list(target) in result["pool_support_labels"],
                target_ever_accepted=any(has(p) for search in result["searches"].values() for p in search["path"][1:]),
                best_validation_with_target=min(with_target) if with_target else None,
                best_validation_without_target=min(without_target) if without_target else None,
                no_target_exclusion_margin=min(without_target)-1.03*best-1e-12 if without_target else None,
                selected_raw_target=has(model["selected_uncontracted"]), selected_final_target=has(model),
                final_errors=row["models"]["multistart_original_rule"]["full_errors"],
                raw_errors=row["trace"]["pre_contraction_errors"], selected_origin=model["selected_origin"])
        mechanisms.append(record)
    write(root/"test/target_mechanisms.json", dict(status="complete", rows=mechanisms,
          source="same frozen actual complete RNJ+WZZT outputs; no refit"))


def aggregate(root):
    """Per-case paired results, retaining all failed/undecided path mass."""
    root, c = configuration(root)
    scored = read(root/"test/scored.json")
    allrows, tables, pairs = scored["rows"], [], []
    models = ["multistart_original_rule", "rnj_bank", "baseline25",
              "geometric_RNJ_RX75_0.16", "geometric_NJ_RX75_0.16"]
    cap = 4*len(c["labels"])-2
    for policy in c["policies"]:
        rows = [r for r in allrows if r["policy"] == policy]
        completed = [r for r in rows if r["status"] == "complete"]
        n, unresolved = len(rows), len(rows)-len(completed)
        table = dict(policy=policy, denominator=n, completed=len(completed), unresolved=unresolved,
            mean_actual_payment=(sum(r["actual_payment"] for r in rows)/n
                                 if all(r.get("actual_payment") is not None for r in rows) else None),
            physical_excitation_evaluated=sum(r.get("excitation_physical_joint_pass") is not None for r in rows),
            physical_excitation_passes=sum(r.get("excitation_physical_joint_pass") is True for r in rows),
            observed_excitation_evaluated=sum(r.get("excitation_observed_joint_pass") is not None for r in rows),
            observed_excitation_passes=sum(r.get("excitation_observed_joint_pass") is True for r in rows),
            models={})
        for model in models:
            total = sum(r["models"][model]["full_errors"] for r in completed)
            recoveries = sum(r["models"][model]["full_recovery"] for r in completed)
            table["models"][model] = dict(full_error_sum_known=total,
                mean_full_errors=total/n if not unresolved else None,
                mean_full_error_bounds=[total/n, (total+cap*unresolved)/n],
                recoveries=recoveries, recovery_fraction_bounds=[recoveries/n, (recoveries+unresolved)/n])
        tables.append(table)
    for path in c["test_paths"]:
        zero = next(r for r in allrows if r["policy"] == "zero" and r["path"] == path)
        for row in allrows:
            if row["path"] != path or row["policy"] == "zero":
                continue
            pair = dict(path=path, policy=row["policy"], status=row["status"], zero_status=zero["status"], models={})
            for model in models:
                z = zero["models"][model]["full_errors"] if zero["status"] == "complete" else None
                p = row["models"][model]["full_errors"] if row["status"] == "complete" else None
                pair["models"][model] = dict(zero_errors=z, paid_errors=p,
                    improvement_zero_minus_paid=z-p if z is not None and p is not None else None,
                    recovered_by_paid=(z > 0 and p == 0) if z is not None and p is not None else None)
            pairs.append(pair)
    summary = dict(status="complete", case=c["case"], case_index=c["case_index"], protocol_sha256=sha(root/"protocol.json"),
        requested_records=len(allrows), unique_fits=scored["unique_jobs"], unique_complete=scored["unique_complete"],
        unique_failed=scored["unique_failed"], actual_LP_calls=scored["actual_LP_calls"],
        independent_case_paths=len(c["test_paths"]), tables=tables, paired=pairs,
        planner_scope="fixed historical two-fit signed prediction difference; not final structural error",
        actual_background_law="unchanged source natural-load generator; planner factor law is an approximation",
        confidence_scope="planning90pct excitation only; two actual paths do not certify actual90pct", new_PV=False)
    write(root/"summary.json", summary)
    print(json.dumps(dict(status="complete", records=len(allrows), unique=scored["unique_jobs"],
                         LP=scored["actual_LP_calls"], case_index=c["case_index"])), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("observe", "fit", "finish", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args(argv)
    if args.command == "fit" and args.worker not in (0, 1):
        parser.error("fit requires --worker 0 or --worker 1")
    if args.command == "observe":
        observe(args.root)
    elif args.command == "fit":
        fit(args.root, args.worker)
    elif args.command == "finish":
        finish(args.root)
    else:
        aggregate(args.root)


if __name__ == "__main__":
    main()
