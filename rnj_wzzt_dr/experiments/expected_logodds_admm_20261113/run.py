"""Freeze a two-input protocol, compare sharing ADMM with direct QCQP."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from pathlib import Path
from time import perf_counter
import shutil
import subprocess
import traceback
import numpy as np

from ... import io
from ...dr.pipeline import load_resources
from ..unified_posterior_dr_20261112.history_scan import ROOT, load_request, require, sha, utc
from ..unified_posterior_dr_20261112.confirmation import rotation
from .evidence import build, direct_gains
from .admm import run as admm_run
from .device_steps import OriginalDeviceStep, initial_requests, native_direct

REFERENCE = ROOT/"outputs/unified_posterior_dr_20261112_v1"
IDENTITIES = ("main__input054__101", "main__input121__101")


def prepare(out):
    require(not out.exists(), "Keep every existing attempt; use a new output directory")
    require((REFERENCE/"execution_complete.json").exists(), "Completed source run is required")
    prior = io.read(REFERENCE/"planning/protocol.json")
    selected = [x for x in prior["selected"] if x["id"] in IDENTITIES]
    require(len(selected) == 2, "Missing preregistered input")
    out.mkdir(parents=True)
    sources = {}
    for folder in (Path(__file__).parent, ROOT/"rnj_wzzt_dr/dr",
                   ROOT/"rnj_wzzt_dr/experiments/unified_posterior_dr_20261112"):
        for p in sorted(folder.glob("*.py")):
            relative = p.relative_to(ROOT); dest = out/"source_snapshot"/relative
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
            sources[relative.as_posix()] = io.ref(p)
    targets = []
    old_seal = io.read(REFERENCE/"selection_seal.json")
    for item in selected:
        scored_path = REFERENCE/"scores"/(item["id"]+".json")
        require(sha(scored_path) == old_seal["scores"][item["id"]], "Frozen candidate bank changed")
        for target in item["targets"]:
            data = REFERENCE/"development/targets"/(item["id"]+f"__target{target}.json")
            row = io.read(data); require("target_correct" not in row, "Planner source contains a truth flag")
            competitors = row["gates"]["0.9"]["allocation"]["active"]
            targets.append(dict(id=item["id"], case=item["case"], target=target, competitors=competitors,
                                observed_qualification=io.ref(data), scores=io.ref(scored_path)))
    protocol = dict(created_utc=utc(), reference=io.ref(REFERENCE/"protocol.json"), selected=selected,
        commit_at_lock=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        targets=targets, sources=sources, environment=io.environment(), gain_required=float(np.log(2)),
        numerical_gain_buffer=1e-5, objective="minimum original nominal contract fee subject to prescribed expected log-posterior odds increments",
        competitor_scope="pre-existing history-only major sets; all original candidates remain in posterior evaluation",
        changes_from_previous="expected evidence gain; no exponential chance bound or failure-risk allocation in planning",
        admm=dict(starts=["zero", "rotation25"], seconds_per_start=30., iterations=300, initial_rho=10.,
                  rho_balance_every=10, rho_ratio=10., rho_multiplier=2., local_seconds=.05,
                  original_integer_device_graph=True, convex_device_claim=False),
        direct=dict(seconds_per_target=60., starts="zero only; no ADMM result used", gap=.01, threads=1),
        timing="Per-method total includes starts, model construction and all local solves; shared frozen-input preparation recorded separately",
        initialization="zero and fixed quarter-amplitude rotation; both original-device feasible, neither assumed gain-feasible",
        comparison="same target, competitor scope, gain, original domain and one-thread 60-second method budget",
        model_confirmation=dict(paths=2048, namespace="expected_logodds_admm_model_20261113", study_seed=2026111301),
        physical_confirmation=dict(paths=32, study_seed=2026111301, replicate_start=22000, policies=["no_dr", "rotation", "admm"]),
        future_opened=False, new_identifier_fits=0, new_parameter_fits=0,
        truth_role="not an input to planning; reused inputs already studied, truth joined only for evaluation",
        public="probabilities, evidence increments, times and normalized fees; raw observations, fit parameters and trajectories private",
        known_development_failure="Symmetric two-competitor example stalls at zero; rho=2 with a feasible nonzero start only retained the initial plan. Rho=10 reaches the analytic branch optimum in that example.")
    io.write(out/"protocol.json", protocol)
    io.write(out/"lock.json", dict(pid=os.getpid(), created_utc=utc(), protocol_sha256=sha(out/"protocol.json")))
    return protocol


def load_target(item, gain):
    requests = {r["id"]: r for r in io.read(REFERENCE/"requests.json")}
    request = requests[item["id"]]
    _, training, validation = load_request(request)
    require(sha(item["scores"]["path"]) == item["scores"]["sha256"], "Target score changed")
    score = io.read(item["scores"]["path"])
    forecast = np.column_stack((np.vstack((training.p, validation.p)), np.vstack((training.q, validation.q))))
    H = np.asarray([p["H"] for p in score["parameters"]])
    original, problem, base = load_resources(item["case"], labels=score["labels"])
    evidence = build(H, item["target"], item["competitors"], forecast, score["sigma2"], original, base, gain)
    original_full_rotation, _ = rotation(original, 1e100)
    fee_scale = original.fees(original_full_rotation)
    require(fee_scale > 0, "Positive original fee unit required")
    return dict(request=request, score=score, H=H, forecast=forecast, original=original,
                problem=problem, base=base, evidence=evidence, fee_scale=fee_scale)


def solve_admm(data, directory, protocol):
    import gurobipy as gp
    evidence, problem, original = data["evidence"], data["problem"], data["original"]
    attempts = []; candidates = []
    for start_kind in protocol["admm"]["starts"]:
        target = directory/start_kind; target.mkdir(parents=True)
        began = perf_counter(); requests = initial_requests(problem, original, start_kind)
        requests[~evidence.active.any(axis=0)] = 0.
        initial = original.nominal(requests); steps = []
        try:
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0); env.start()
                for r in range(original.R):
                    steps.append(OriginalDeviceStep(problem, r, requests[r:r+1], data["fee_scale"], env)
                                 if evidence.active[:, r].any() else None)
                setup = perf_counter()-began
                result = admm_run(evidence, steps, initial, original.fees,
                    seconds=protocol["admm"]["seconds_per_start"], iterations=protocol["admm"]["iterations"],
                    rho=protocol["admm"]["initial_rho"], setup_seconds=setup,
                    checkpoint=lambda row: io.write(target/"last_iteration.json", row))
                if result["best"] is not None:
                    best = result["best"]
                    x = np.vstack([part if part is not None else np.zeros((1, 24)) for part in best["requests"]])
                    replay = original.nominal(x)
                    audit = original.audit(replay)
                    independent = direct_gains(data["H"], data["item"]["target"], data["item"]["competitors"],
                        data["forecast"], data["score"]["sigma2"], original, data["base"], replay)
                    require(np.max(abs(replay-best["delivery"])) < 1e-7, "Assembled resource policy did not replay")
                    require(np.max(abs(independent-evidence.gains(replay))) < 1e-8, "Independent mean evidence differs")
                    require(audit["passed"] and min(independent) >= evidence.gain_required-1e-8, "Retained quote is not gain/resource feasible")
                    np.savez_compressed(target/"quote.npz", x=x, mean_u=replay)
                    candidates.append(dict(path=io.ref(target/"quote.npz"), fee=original.fees(replay),
                        start=start_kind, retained_iteration=best["iteration"], gains=independent,
                        first_feasible_seconds=result["first_feasible_seconds"], device=audit))
                io.write(target/"trace.json", result.pop("trace"))
                best = result.pop("best")
                result["retained_iteration"] = best["iteration"] if best else None
                result["retained_fee"] = best["fee"] if best else None
                result.update(start=start_kind, initialization_seconds=setup, initial_device_audit=original.audit(initial))
                io.write(target/"summary.json", result); attempts.append(result)
                for step in steps:
                    if step is not None:
                        step.close()
                steps = []
        finally:
            for step in steps:
                if step is not None:
                    step.close()
    best = min(candidates, key=lambda r: r["fee"]) if candidates else None
    result = dict(qualified=best is not None, best=best, attempts=attempts,
        total_seconds=sum(r["total_seconds"] for r in attempts),
        first_feasible_seconds=min((sum(a["total_seconds"] for a in attempts[:i])+a["first_feasible_seconds"]
            for i, a in enumerate(attempts) if a["first_feasible_seconds"] is not None), default=None),
        global_lower_bound=None, global_optimality_proved=False,
        device_step="original mixed-integer graph; bounded, potentially inexact subsolves")
    if best:
        shutil.copyfile(best["path"]["path"], directory/"quote.npz")
    io.write(directory/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); protocol = prepare(out); rows = []
    for item in protocol["targets"]:
        path = out/"planning"/(item["id"]+f"__target{item['target']}"); path.mkdir(parents=True)
        began = perf_counter()
        try:
            data = load_target(item, protocol["gain_required"]); data["item"] = item
            io.write(path/"input.json", dict(item=item, parameter_sha256=data["score"]["parameter_sha256"],
                sigma2=data["score"]["sigma2"], forecast_sha256=io.contract_hash(data["forecast"]),
                fee_scale=data["fee_scale"], gain_required=protocol["gain_required"],
                preparation_seconds=perf_counter()-began))
            (path/"admm").mkdir(); admm = solve_admm(data, path/"admm", protocol)
            print(f"ADMM {item['id']} target {item['target']}: qualified={admm['qualified']}, seconds={admm['total_seconds']:.2f}", flush=True)
            (path/"direct").mkdir()
            direct = native_direct(data["problem"], data["original"], data["evidence"], data["fee_scale"], path/"direct",
                                   seconds=protocol["direct"]["seconds_per_target"])
            if admm["qualified"] and direct["lower_bound"] is not None:
                direct["bound_consistent_with_ADMM_feasible_quote"] = direct["lower_bound"] <= admm["best"]["fee"]+1e-6
                if not direct["bound_consistent_with_ADMM_feasible_quote"]:
                    direct["bound_warning"] = "withdrawn: native lower bound exceeds independently replayed ADMM feasible quote"
            io.write(path/"direct/summary.json", direct)
            rows.append(dict(**item, status="complete", admm=admm, direct=direct))
            print(f"Direct {item['id']} target {item['target']}: qualified={direct['qualified']}, status={direct['status']}", flush=True)
        except Exception as exc:
            rows.append(dict(**item, status="failed", error=repr(exc), traceback=traceback.format_exc()))
            print(f"Retained failure {item['id']} target {item['target']}: {exc!r}", flush=True)
        io.write(out/"planning_results.json", rows)
    io.write(out/"planning_complete.json", dict(created_utc=utc(), status="bounded_attempts_complete", targets=len(rows),
        protocol_sha256=sha(out/"protocol.json"), results_sha256=sha(out/"planning_results.json"),
        failures=sum(r["status"] != "complete" for r in rows)))


if __name__ == "__main__":
    main()
