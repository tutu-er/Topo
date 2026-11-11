"""Independent audit of the fixed four-input simplified-RTE pilot.

No native optimizer is imported. Historical training/validation observations
replay complete predictors and selectors; truth/test are opened afterwards.
Failed, invalid and unfinished inputs retain the planned denominator of four.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np
import pandas as pd

from research_experiments.miqp_loss_benchmark_20261111 import analyze as independent
from .run import read, sha, digest, write, parameters, choose, check_sources


ROOT = Path(__file__).resolve().parents[2]
NEW, OLD = "simplified_RTE", "parent_T003"


def close(a, b, name, *, atol=1e-9):
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(a)-float(b)) > atol+1e-7*max(abs(float(a)), abs(float(b))):
        raise ValueError(f"{name} does not reproduce: {a!r} vs {b!r}")


def assert_offline():
    bad = [n for n in sys.modules if n == "gurobipy" or n.startswith("gurobipy.")
           or n.endswith(".squared_backend") or n == "research_experiments.simplified_rte_20261111.engine"]
    if bad: raise RuntimeError("Independent post-analysis imported native optimization: "+", ".join(bad))


def base_row(job, meta, method):
    return dict(job=job, case=meta["case"], seed_index=meta["seed_index"], voltage_sigma=meta["voltage_sigma"],
                method=method, status="pending", returned=False, available=False)


def verify_new(job, meta, output, protocol, splits):
    file = output/"jobs"/job/"result.json"; result = read(file)
    if (result["job"] != job or result["method"] != NEW or result["protocol_sha256"] != sha(output/"protocol.json")
            or result["common_sha256"] != protocol["parent_common_sha256"][job]
            or result["parent_T003_result_sha256"] != protocol["parent_T003_result_sha256"][job]):
        raise ValueError("Pilot result provenance changed.")
    expected = {s: protocol["input_hashes"][job][s] for s in ("train", "validation")}
    if result["input_hashes"] != expected or any(result.get(k) is not False for k in ("truth_used", "test_opened", "final_RNJ_fallback", "output_contraction")):
        raise ValueError("Pilot result uses an incompatible information/output policy.")
    parent = Path(protocol["parent_output"]); common = read(parent/"common"/job/"common.json")
    K, reference = float(common["normalization"]["K"]), float(common["reference"])
    close(result["reference"], reference, "frozen reference")
    close(result["normalization"]["K"], K, "loss normalization")
    close(result["common_seconds"], common["seconds"], "reused common cost")
    close(result["total_seconds"], result["common_seconds"]+result["fresh_seconds"], "timing identity")
    penalty, n = .003*reference, len(meta["labels"])
    path = result["trajectory"]["path"]
    if not path or independent.support_count(path[0], n) != 0: raise ValueError("RTE must start from the basis without RNJ seed blocks.")
    scores, path_checks = [], []
    for index, point in enumerate(path):
        train = independent.scores(splits["train"], point)
        close(point["objective"], K*train["mse"], "raw path scaled training MSE")
        J = K*train["mse"]+penalty*independent.support_count(point, n)
        if scores and J > scores[-1]+1e-9: raise ValueError("Accepted raw path increases the common regularized score.")
        scores.append(J)
        path_checks.append(dict(job=job, index=index, J=J, train_mse=train["mse"],
                                proper_count=independent.support_count(point, n)))
    if independent.fingerprint(path[-1]) != independent.fingerprint(result["trajectory"]["point"]):
        raise ValueError("Final trajectory point differs from the last accepted path point.")
    trajectory = result["trajectory"]
    if trajectory.get("initial_seeds") != [] or trajectory.get("truth_used") is not False or trajectory.get("validation_used") is not False:
        raise ValueError("Simplified training trajectory has incompatible seed/information flags.")
    basis = {*((i,) for i in range(n)), tuple(range(n))}
    expected_R = set(independent.canonical([*common["initial"], *(C for p in common["bank"] for C in p["supports"]), *basis]))
    if set(map(tuple, trajectory["rnj_pool"])) != expected_R:
        raise ValueError("Actual RNJ guidance domain is not derived from the declared RNJ families.")
    tail = trajectory["tail_domain"]
    expected_E = expected_R | {tuple(C) for point in path for C in point["supports"]} | set(map(tuple, tail["pool"])) | set(combinations(range(n), 2)) | basis
    if set(map(tuple, trajectory["e_pool"])) != expected_E:
        raise ValueError("E domain differs from R guidance, path, T domain and all pairs.")
    for frontier in tail.get("frontiers", []):
        blocks = [set(C) for C in frontier["blocks"]]
        if not 3 <= len(blocks) <= 5 or set().union(*blocks) != set(frontier["parent"]):
            raise ValueError("A T frontier is not a <=5-block partition of its parent.")
        if any(C & D for i, C in enumerate(blocks) for D in blocks[i+1:]):
            raise ValueError("T frontier blocks overlap.")
    stages = [event["stage"] for event in trajectory["events"]]
    if (stages.count("T") > 1 or stages.count("E") != 1 or stages[-1] != "E"
            or any(stage not in {"W", "T", "E"} for stage in stages)
            or ("T" in stages and any(stage == "W" for stage in stages[stages.index("T")+1:]))):
        raise ValueError("Expected W, at most one T, then one E without returning to W.")
    solver_rows, event_checks = [], []
    cursor, current_J = 0, scores[0]
    visited = {independent.canonical(path[0]["supports"])}
    for index, event in enumerate(trajectory["events"]):
        if "before_score" in event:
            close(event["before_score"], current_J, "event before-J")
        if "point" in event:
            candidate = event["point"]
            L = independent.scores(splits["train"], candidate)["mse"]
            J = K*L+penalty*independent.support_count(candidate, n)
            close(event["after_score"], J, "event proposed after-J")
            close(event["gain"], event["before_score"]-event["after_score"], "event gain", atol=1e-12)
            numerical_passed = float(event["gain"]) > float(event["numerical_threshold"])
            repeated = bool(event.get("rejected_by_visited", False))
            if repeated:
                if (event["stage"] != "W" or event["accepted"] or not numerical_passed
                        or event.get("numerical_gain_passed") is not True
                        or independent.canonical(candidate["supports"]) not in visited):
                    raise ValueError("A visited-structure numerical rejection does not reproduce.")
            elif bool(event["accepted"]) != numerical_passed:
                raise ValueError("Structural update does not use its declared common J threshold.")
            if event["accepted"] and not repeated:
                cursor += 1
                if cursor >= len(path) or independent.fingerprint(candidate) != independent.fingerprint(path[cursor]):
                    raise ValueError("Accepted structural event does not reproduce its actual path append.")
                current_J = scores[cursor]; visited.add(independent.canonical(candidate["supports"]))
            event_checks.append(dict(job=job, event=index, stage=event["stage"], accepted=event["accepted"],
                before_J=event["before_score"], proposed_J=J, gain=event["gain"], repeated_W_not_appended=repeated))
        native = event.get("solver", {})
        if native.get("solve_started"):
            upper, lower = native.get("objective"), native.get("dual_bound")
            if upper is not None and lower is not None and lower > upper+1e-8: raise ValueError("A native lower bound exceeds its incumbent.")
            warm = native.get("warm_start", {}).get("objective")
            if warm is not None and lower is not None and lower > warm+1e-8: raise ValueError("A native lower bound exceeds its known feasible warm score.")
            if native.get("audit", {}).get("has_incumbent") and native["audit"].get("valid") is not True:
                raise ValueError("Native structural incumbent failed its feasibility audit.")
            requested = float(native["requested_seconds"])
            if requested <= 0 or (event["stage"] == "W" and requested > protocol["limits"]["per_mip"]+1e-6):
                raise ValueError("The requested W native cap does not match the pilot protocol.")
            solver_rows.append(dict(job=job, stage=event["stage"], status=native.get("status"),
                objective=upper, dual_bound=lower, relative_gap=native.get("relative_gap"),
                seconds=native.get("seconds"), requested_seconds=requested,
                timeout=native.get("status") == 9, gap_met=(native.get("relative_gap") is not None and native["relative_gap"] <= .0001+1e-9)))
    if cursor != len(path)-1: raise ValueError("Path contains an unaccounted accepted point.")
    library = result["output_candidates"]
    if len(library) != len(path)+1: raise ValueError("Raw output library lost path points or its singleton baseline.")
    for index, row in enumerate(library):
        point = path[index] if index < len(path) else result["star"]
        if row["index"] != index or independent.fingerprint(row["point"]) != independent.fingerprint(point):
            raise ValueError("Pilot output changed raw fitted coefficients through a contraction/refit.")
        validation = independent.scores(splits["validation"], point)
        close(row["validation_mae"], validation["mae"], "raw validation MAE", atol=1e-12)
        close(row["validation_mse"], validation["mse"], "raw validation MSE", atol=1e-14)
        close(row["validation_scaled_mse"], K*validation["mse"], "scaled validation MSE")
        if row["internal"] != independent.support_count(point, n): raise ValueError("Tie count changed.")
    for name, metric in (("selected", "mae"), ("selected_mse_view", "mse")):
        replay = choose(library, metric)
        if replay["index"] != result[name]["index"] or independent.fingerprint(replay["point"]) != independent.fingerprint(result[name]["point"]):
            raise ValueError("Strict raw-library selection does not reproduce.")
    export_file = file.with_name("candidates.json"); exported = read(export_file)
    if sha(export_file) != result["candidate_export_sha256"] or exported["pool_sha256"] != digest({k: v for k, v in exported.items() if k != "pool_sha256"}):
        raise ValueError("Raw frozen candidate export hash changed.")
    if exported["contraction"] is not False or exported["weights_frozen"] is not True or exported["future_refit_allowed"] is not False:
        raise ValueError("Raw candidate contract was changed.")
    by_id = {row["model_id"]: row for row in exported["models"]}
    if len(by_id) != len(exported["models"]): raise ValueError("Candidate IDs repeat.")
    for model_id, model in by_id.items():
        point = model["point"]; R, X = independent.matrices(point, n)
        if model_id != digest(parameters(point)) or model["parameter_sha256"] != model_id: raise ValueError("Candidate coefficients changed.")
        if not np.allclose(model["R"], R, atol=1e-12, rtol=0.) or not np.allclose(model["X"], X, atol=1e-12, rtol=0.):
            raise ValueError("Frozen response matrices do not reproduce.")
        for split in ("train", "validation"):
            actual = independent.scores(splits[split], point)
            for key in ("mae", "mse"): close(model[split+"_"+key], actual[key], "candidate "+split+" "+key, atol=1e-12)
    if set(row["model_id"] for row in library) != set(by_id): raise ValueError("Export is not exactly raw path plus singleton.")
    for phase in ("loaded_sources_before", "loaded_sources_after"):
        for module in result[phase].values():
            if module["sha256"] != protocol["source_hashes"].get(module["relative_path"]): raise ValueError("A loaded source differs from frozen manifest.")
    profile = independent.independent_qp(splits["train"], result["selected"]["point"]["supports"])
    fixed_check = independent.fixed_qp_check(result["selected"]["point"], splits["train"], profile, K)
    return result, solver_rows, path_checks, event_checks, fixed_check


def analyze(output):
    output = Path(output).resolve(); started = perf_counter()
    protocol, catalog = read(output/"protocol.json"), read(output/"catalog.json")
    if protocol["planned"] != 4 or len(catalog) != 4: raise ValueError("This analyzer is restricted to the fixed four-input pilot.")
    if any(m["seed_index"] != 0 or m["voltage_sigma"] != .0005 for m in catalog.values()): raise ValueError("Pilot input subset changed.")
    check_sources(output, protocol)
    if sha(output/"catalog.json") != protocol["catalog_sha256"]: raise ValueError("Pilot catalog changed.")
    parent = Path(protocol["parent_output"])
    if read(parent/"analysis/verification.json")["analyzer_sha256"] != sha(Path(independent.__file__)):
        raise ValueError("Independent helper differs from the verified parent analyzer.")
    rows, cached, solvers, paths, events, fixed, errors = [], {}, [], [], [], [], []
    source_hashes, truth_hashes, input_hashes, failure_hashes = {}, {}, {}, {}
    for job in protocol["jobs"]:
        meta = catalog[job]
        for split in ("train", "validation"):
            if sha(meta[split]) != protocol["input_hashes"][job][split]: raise ValueError("Frozen observed input changed.")
        splits = {s: independent.load_split(meta[s]) for s in ("train", "validation")}
        baseline_file = parent/"jobs"/job/"T003/result.json"
        old = read(baseline_file)
        baseline = base_row(job, meta, OLD); baseline.update(status="historical_verified", returned=True)
        cached[job] = dict(splits=splits, old=old, old_row=baseline)
        row = base_row(job, meta, NEW); file = output/"jobs"/job/"result.json"
        if not file.exists():
            row["status"] = "failed" if file.with_name("error.json").exists() else "pending"
            if file.with_name("error.json").exists(): failure_hashes[job] = sha(file.with_name("error.json"))
            rows.append(row); continue
        row["returned"] = True
        source_hashes[job] = sha(file)
        try:
            result, s, p, e, q = verify_new(job, meta, output, protocol, splits)
            cached[job]["new"] = result; cached[job]["new_row"] = row
            solvers.extend(s); paths.extend(p); events.extend(e)
            fixed.append(dict(job=job, **q)); row["status"] = "numerically_verified"
        except Exception as exc:
            row.update(status="invalid", error=str(exc)); errors.append(dict(job=job, error=str(exc), traceback=traceback.format_exc()))
        rows.append(row)
    # Model fitting/selection has completed before opening physical truth/test.
    for job, bundle in cached.items():
        meta = catalog[job]; splits = bundle["splits"]
        if sha(meta["test"]) != protocol["input_hashes"][job]["test"]: raise ValueError("Frozen test input changed.")
        input_hashes[job] = {s: sha(meta[s]) for s in ("train", "validation", "test")}
        splits["test"] = independent.load_split(meta["test"])
        prior = read(parent/"analysis/jobs"/job/"verified.json")
        if prior["errors"] or prior["raw_hashes"]["scaled_mse/T003"] != protocol["parent_T003_result_sha256"][job]:
            raise ValueError("Parent baseline numerical audit changed.")
        network = Path(meta["truth_network"])
        hashes = {name: sha(network/name) for name in ("metadata.json", "buses.csv", "branches.csv")}
        if hashes != prior["truth_source_hashes"]: raise ValueError("Physical truth changed.")
        truth_hashes[job] = hashes; truth = independent.physical_truth(parent, meta, meta["labels"])
        old_metrics = independent.metrics(bundle["old"]["selected"]["point"], splits, truth)
        original_row = next(r for r in prior["rows"] if r["fit_loss"] == "scaled_mse" and r["method"] == "T003")
        for key in ("clade_f1", "train_mae", "train_mse", "validation_mae", "validation_mse", "test_mae", "test_mse"):
            close(old_metrics[key], original_row[key], "parent baseline "+key, atol=1e-12)
        bundle["old_row"].update(**old_metrics, status="verified", available=True,
            total_seconds=bundle["old"]["total_seconds"], fresh_seconds=bundle["old"]["fresh_seconds"],
            common_seconds=bundle["old"]["common_seconds"], output_policy="historical .24 contraction/QP refit")
        rows.append(bundle["old_row"])
        if "new" in bundle:
            new = bundle["new"]
            bundle["new_row"].update(**independent.metrics(new["selected"]["point"], splits, truth), status="verified", available=True,
                total_seconds=new["total_seconds"], fresh_seconds=new["fresh_seconds"], common_seconds=new["common_seconds"],
                cap_exceeded=new["total_seconds"] > protocol["limits"]["total"]+1., output_policy="raw path+singleton; no contraction",
                MIPs=sum(s["job"] == job for s in solvers), MIP_timeouts=sum(s["job"] == job and s["timeout"] for s in solvers),
                stop_reason=new["trajectory"]["stop_reason"], path_length=len(new["trajectory"]["path"]),
                zero_releases=sum(bool(z.get("accepted")) for z in new["trajectory"]["zero_events"]))
    assert_offline(); frame = pd.DataFrame(rows); summaries, paired = [], []
    for method in (OLD, NEW):
        group = frame[frame.method == method]; valid = group[group.status == "verified"]
        summaries.append(dict(method=method, planned=4, returned=int(group.returned.sum()), verified=len(valid),
            failed=int((group.status == "failed").sum()), invalid=int((group.status == "invalid").sum()), pending=int((group.status == "pending").sum()),
            F1_verified_mean=float(valid.clade_f1.mean()) if len(valid) else None, exact=int(valid.exact.sum()) if len(valid) else 0,
            exact_fraction_planned=float(valid.exact.sum()/4) if len(valid) else 0.,
            total_seconds_verified=float(valid.total_seconds.sum()) if len(valid) else None, complete=len(valid) == 4))
    for job in protocol["jobs"]:
        old, new = frame[(frame.job == job) & (frame.method == OLD)].iloc[0], frame[(frame.job == job) & (frame.method == NEW)].iloc[0]
        ok = old.status == new.status == "verified"
        paired.append(dict(job=job, status="paired" if ok else new.status, delta_F1=new.clade_f1-old.clade_f1 if ok else None,
                          delta_exact=int(new.exact)-int(old.exact) if ok else None,
                          delta_reported_seconds=new.total_seconds-old.total_seconds if ok else None,
                          runtime_interpretation="shared old preprocessing reused; output policies and structural algorithms differ; not independent fair-budget evidence"))
    target = output/"analysis"; target.mkdir(parents=True, exist_ok=True)
    artifacts = {"per_model.csv": frame, "summary.csv": pd.DataFrame(summaries), "paired_inputs.csv": pd.DataFrame(paired),
        "solver.csv": pd.DataFrame(solvers), "path_scores.csv": pd.DataFrame(paths), "events.csv": pd.DataFrame(events),
        "fixed_QP_checks.csv": pd.DataFrame(fixed), "failures.csv": pd.DataFrame(errors)}
    for name, table in artifacts.items(): table.to_csv(target/name, index=False, lineterminator="\n")
    complete = all(row["complete"] for row in summaries)
    verification = dict(passed=complete and not errors and not bool(frame.get("cap_exceeded", pd.Series(False)).fillna(False).any()),
        complete=complete, planned_new=4, verified_new=sum(row.get("method") == NEW and row.get("status") == "verified" for row in rows),
        baseline_planned=4, baseline_verified=4, all_denominators_preserved=True, errors=errors,
        native_optimizer_imported=False, new_native_solver_calls_by_analyzer=0, independent_selected_family_QP_checks=len(fixed),
        model_and_selector_replayed_without_contraction=True, accepted_path_J_monotonic=True if not errors else None,
        truth_test_used_only_in_postanalysis=True, shared_old_common_reused=True, independent_fair_runtime_comparison=False,
        comparison_scope=protocol["comparison_scope"], protocol_sha256=sha(output/"protocol.json"), analyzer_sha256=sha(__file__),
        independent_helper_sha256=sha(Path(independent.__file__)), native_result_sha256=source_hashes,
        failure_file_sha256=failure_hashes, MIP_timeouts=sum(row["timeout"] for row in solvers),
        MIP_scope="each finite conditional W/T/E domain; feasible outputs do not assert global tree optimality",
        truth_source_hashes=truth_hashes, input_hashes=input_hashes, result_hashes={name: sha(target/name) for name in artifacts},
        seconds=perf_counter()-started)
    write(target/"verification.json", verification); print(pd.DataFrame(summaries).to_string(index=False)); return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT/"outputs/simplified_rte_20261111")
    analyze(parser.parse_args().output)
