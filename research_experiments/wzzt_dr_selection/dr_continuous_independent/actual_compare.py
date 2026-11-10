"""Paired original-AC evaluation of four frozen contracts on one sealed day.

Only execution requests and meter observations are sampled. Natural P/Q and
physical root voltage remain the same sealed trajectory. The independent day
denominator is one; the independent execution/meter-path denominator is 16.
All run directories must stay private because observations contain P/Q/V.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import pickle
import sys
import time
import traceback
import zipfile

for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[option] = "1"

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
REPO = TASK.parents[1]
sys.path[:0] = [str(HERE), str(TASK / "dr_heterogeneous_pilot"), str(REPO)]

import run as bounded
from sample_compare import load_policies
from research_experiments.wzzt_dr_selection.dr_heterogeneous_pilot import driver as original
from research_experiments.wzzt_dr_selection.dr_paper15_delivery_margin.ac_replay import score_scene


RESPONSE_SEED = 2026111204
METER_SEED = 2026111205
POLICY_ORDER = ("zero", "rotation", "old95", "current90")


def array_hash(array):
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(value.tobytes()).hexdigest()


def physical_fingerprints(background):
    return {key: array_hash(getattr(background, key))
            for key in ("p_base", "q_base", "root_true")}


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(pickle.dumps(value, protocol=4))
    temporary.replace(path)


def policy_summary(rows, policy, requested):
    selected = [row for row in rows if row["policy"] == policy]
    complete = [row for row in selected if row["status"] == "complete"]
    successes = sum(row["excluded"] for row in complete)
    return dict(requested=requested, observed=len(selected), complete=len(complete),
                failures=len(selected)-len(complete), not_run=requested-len(selected),
                coarse_excluded=successes,
                exclusion_fraction=successes/len(complete) if complete else None,
                wilson_95=bounded.wilson(successes, len(complete)),
                complete_recoveries=sum(row["complete_recovery"] for row in complete),
                total_wrong_or_missing_blocks=sum(row["errors"] for row in complete),
                mean_D=float(np.mean([row["D_micro"] for row in complete])) if complete else None,
                min_D=float(min(row["D_micro"] for row in complete)) if complete else None,
                max_D=float(max(row["D_micro"] for row in complete)) if complete else None,
                fee=selected[0]["fee"] if selected else None,
                mean_delivered_absolute_kwh=float(np.mean([row["delivered_absolute_kwh"] for row in complete]))
                    if complete else None,
                converged_AC_slots=sum(row.get("converged",0) for row in selected),
                requested_AC_slots=requested*96,
                voltage_violation_cells=sum(row.get("voltage_violation_cells",0) for row in complete))


def paired_results(rows, first, second):
    by_key = {(row["path"],row["policy"]):row for row in rows}
    outcomes = []
    for path in sorted({row["path"] for row in rows}):
        a,b = by_key.get((path,first)),by_key.get((path,second))
        if a and b and a["status"]==b["status"]=="complete":
            outcomes.append((a["errors"],b["errors"]))
    return dict(first=first,second=second,paired_complete_paths=len(outcomes),
                first_better=sum(a<b for a,b in outcomes),
                equal=sum(a==b for a,b in outcomes),
                first_worse=sum(a>b for a,b in outcomes))


def main(args, *, policy_loader=None, policy_order=None, response_seed=RESPONSE_SEED,
         meter_seed=METER_SEED, extra_sources=(), comparison_pairs=None,
         stage_metadata=None, has_new_quote=False):
    policy_order=POLICY_ORDER if policy_order is None else tuple(policy_order)
    if policy_loader is None:
        policy_loader=load_policies
    if comparison_pairs is None:
        comparison_pairs=[("current90","zero"),("current90","rotation"),("old95","current90")]
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock=out/"running.lock"
    with lock.open("x",encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    started=time.perf_counter()
    try:
        archive,public=args.archive.resolve(),args.public.resolve()
        problem=bounded.build_problem(archive,public)
        policies,policy_references=policy_loader(problem,archive)
        if len(set(policy_order))!=len(policy_order) or set(policies)!=set(policy_order):
            raise ValueError("Frozen policy bank differs from explicit order")
        if any(a not in policies or b not in policies for a,b in comparison_pairs):
            raise ValueError("Unknown policy in paired comparisons")
        artifacts=bounded.read(archive/bounded.CASE/"inputs.json")
        condition=bounded.read(bounded.checked(artifacts["case_protocol.json"]))
        pool=bounded.read(bounded.checked(artifacts["pool.json"]))
        prepared=args.sealed.resolve()/"prepared"
        receipt=bounded.read(prepared/"receipt.json")
        natural=prepared/"natural_private.pkl"
        if bounded.sha(natural)!=receipt["files"]["natural_private.pkl"]:
            raise ValueError("Sealed natural-day hash changed")
        if bounded.sha(public/"forecast.npy")!=receipt["files"]["public/forecast.npy"]:
            raise ValueError("Published forecast differs from sealed day")
        sources=dict(problem["sources"])
        for filename in [Path(__file__),HERE/"sample_compare.py",
                         TASK/"dr_heterogeneous_pilot/driver.py",
                         TASK/"dr_paper15_delivery_margin/ac_replay.py", *extra_sources]:
            sources[str(filename.relative_to(TASK))]=bounded.sha(filename)
        inputs=dict(case=bounded.CASE,paths=args.paths,requested_policy_observations=args.paths*len(policy_order),
                    max_seconds=args.seconds,response_seed=response_seed,meter_seed=meter_seed,
                    original_archive_protocol=bounded.reference(archive/"protocol.json"),
                    sealed_day=bounded.reference(natural),sealed_receipt=bounded.reference(prepared/"receipt.json"),
                    historical_and_forecast_inputs=problem["inputs"],policies=policy_references,
                    fees={name:policies[name]["fee"] for name in policy_order},sources=sources,
                    python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                    natural_day_count=1,rerun_natural_generator=False,
                    response_law="independent primitive Gaussian requests plus original Devices.project",
                    meter_law="original shared-root and terminal relative observation noises",
                    no_parameter_refit=True,no_W_search=True,
                    selection="complete historical gate, weight2, common3%, frozen .16 support map")
        if stage_metadata is not None:
            inputs["extension"]=stage_metadata
        if (out/"request.json").exists() and bounded.read(out/"request.json")!=bounded.plain(inputs):
            raise ValueError("Changed inputs or source in existing AC sampling stage")
        bounded.write(out/"request.json",inputs)
        if (out/"complete.json").exists():
            print("Already complete",flush=True)
            return
        if not (out/"source_snapshot.zip").exists():
            with zipfile.ZipFile(out/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as snapshot:
                for filename in sources:
                    snapshot.write(TASK/filename,filename)
        # Freeze contracts before unsealing the evaluation day or constructing
        # its true AC network. None of the following results feeds price design.
        frozen=out/"contracts_private.pkl"
        if not frozen.exists():
            dump(frozen,{name:policies[name]["mean_u"] for name in policy_order})
        bounded.write(out/"policy_freeze.json",dict(contracts=bounded.reference(frozen),
                      sources=policy_references,fees=inputs["fees"],test_truth_used=False))
        runtime=original.open_runtime(condition["source_root"],condition["source_metadata"],
                    repo_root=REPO,protocol_ref=original.ref(archive/"protocol.json"),
                    labels=condition["labels"],case="paper15",
                    v_sigma=condition["voltage_sigma"],pq_sigma=condition["pq_sigma"])
        background=original.load(natural)
        fingerprints=physical_fingerprints(background)
        device=problem["device"]
        response_children=np.random.SeedSequence(response_seed).spawn(args.paths)
        meter_children=np.random.SeedSequence(meter_seed).spawn(args.paths)
        rows=bounded.read(out/"observed_rows.json") if (out/"observed_rows.json").exists() else []
        done={(row["path"],row["policy"]) for row in rows}
        observation_dir=out/"observations"
        observation_dir.mkdir(exist_ok=True)
        for path in range(args.paths):
            if time.perf_counter()-started>args.seconds:
                break
            rng=np.random.default_rng(response_children[path])
            multiplier=1.+device.relative_sd*rng.standard_normal((device.R,device.T))
            meter_rng=np.random.default_rng(meter_children[path])
            shifted_seed=int(meter_rng.integers(1,2**62))
            meter_replicate=int(meter_rng.integers(1,2**31))
            observed_background=replace(background,shifted_seed=shifted_seed,replicate=meter_replicate)
            if physical_fingerprints(observed_background)!=fingerprints:
                raise ArithmeticError("Meter-only seed override changed physical background")
            seeds=dict(response=response_seed,response_spawn_key=list(response_children[path].spawn_key),
                       meter=meter_seed,meter_spawn_key=list(meter_children[path].spawn_key),
                       terminal_meter_seed=shifted_seed+2000,
                       root_meter_seed=70000019+100003*meter_replicate+int(background.scenario_index),
                       overridden_fields_are_meter_only=True)
            for policy in policy_order:
                if (path,policy) in done:
                    continue
                if time.perf_counter()-started>args.seconds:
                    break
                row=dict(path=path,policy=policy,fee=policies[policy]["fee"],seeds=seeds)
                began=time.perf_counter()
                try:
                    record=f"{bounded.CASE}:paired_ac_sampling:{path}:{policy}"
                    result=original.execute(runtime,observed_background,device,
                                           policies[policy]["mean_u"],multiplier,record)
                    destination=observation_dir/f"path_{path:02d}_{policy}.pkl"
                    dump(destination,result)
                    summary=result["summary"]
                    row.update(status=summary["status"],observation=bounded.reference(destination),
                               converged=summary["converged"],
                               voltage_violation_cells=summary["voltage_violation_cells"],
                               resource_audit_passed=summary["resource_audit"]["passed"])
                    if row["status"]=="complete":
                        scored=original.score(pool,result["scene"],history_gate=True)
                        rule=next(rule for rule in scored["rules"]
                                  if rule["new_day_weight"]==2 and rule["new_day_fraction"]==.03)
                        losses={r["candidate_id"]:r["new_validation_mae"] for r in scored["loss_rows"]}
                        row.update(selected=rule["selected_candidate_id"],
                                   excluded="coarse" not in rule["eligible_candidate_ids"],
                                   D_micro=1e6*(losses["coarse"]-1.03*losses["refined"]),
                                   delivered_absolute_kwh=summary["resource_audit"]["absolute_DR_kwh"])
                except Exception as exc:
                    row.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
                row["seconds"]=time.perf_counter()-began
                rows.append(row)
                done.add((path,policy))
                bounded.write(out/"observed_rows.json",rows)
            print(f"path {path+1}/{args.paths}, observations {len(rows)}/{args.paths*len(policy_order)}",flush=True)
        if physical_fingerprints(background)!=fingerprints:
            raise ArithmeticError("Physical background mutated during evaluation")
        bounded.write(out/"predictions_frozen.json",dict(rows=rows,truth_used=False,
                      physical_background=fingerprints,one_natural_day=True))
        # Query evaluation truth only after every recorded selection is frozen.
        truth={tuple(sorted(block)) for block in runtime["legacy"].evaluate_supports(
                [],runtime["context"].net)["missed_blocks"]}
        for row in rows:
            if row["status"]!="complete":
                continue
            observation=original.load(bounded.checked(row["observation"]))["scene"]
            selected,excluded,errors,gap=score_scene(observation,pool,truth)
            if (selected,excluded)!=(row["selected"],row["excluded"]) or abs(gap-row["D_micro"])>1e-8:
                raise ArithmeticError("Independent frozen selector replay failed")
            row.update(errors=errors,complete_recovery=errors==0)
        summary=dict(case=bounded.CASE,role="one fixed natural-day paired AC execution sampling",
                     natural_days=1,requested_execution_paths=args.paths,
                     execution_paths_with_any_observation=len({row["path"] for row in rows}),
                     requested_policy_observations=args.paths*len(policy_order),observed_policy_records=len(rows),
                     complete_policy_records=sum(row["status"]=="complete" for row in rows),
                     failed_or_unresolved_records=sum(row["status"]!="complete" for row in rows),
                     seconds=time.perf_counter()-started,physical_background=fingerprints,
                     original_day_profile=background.profile,original_day_replicate=int(background.replicate),
                     threshold_micro=problem["packet"]["threshold"],
                     response_seed=response_seed,meter_seed=meter_seed,
                     response_independent=True,original_meter_observer=True,
                     copied_optimizer_current90=not has_new_quote,no_new_policy_optimization=not has_new_quote,
                     thermal_limits_checked=False,
                     policies={p:policy_summary(rows,p,args.paths) for p in policy_order},
                     paired=[paired_results(rows,a,b) for a,b in comparison_pairs])
        bounded.write(out/"evaluated_rows.json",rows)
        bounded.write(out/"summary.json",summary)
        status="complete" if len(rows)==args.paths*len(policy_order) else "bounded_partial"
        bounded.write(out/"complete.json",dict(status=status,summary_sha256=bounded.sha(out/"summary.json"),
                      request_sha256=bounded.sha(out/"request.json"),all_selections_before_truth=True))
        print(json.dumps(bounded.plain(summary),ensure_ascii=True),flush=True)
    except Exception as exc:
        bounded.write(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("archive","public","sealed","out"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--paths",type=int,default=16)
    parser.add_argument("--seconds",type=float,default=300.)
    main(parser.parse_args())
