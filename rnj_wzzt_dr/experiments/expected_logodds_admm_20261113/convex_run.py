"""A second, explicitly restricted prototype after the original-device pilot.

No physical/confirmation sample is opened to choose this change. Original
integer-domain results stay preserved and are not replaced or relabelled.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import traceback
import numpy as np

from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT, require, sha, utc
from .run import load_target
from .admm import run as admm_run
from .convex_steps import ConvexDeviceStep, supporting_plane_start
from .evidence import direct_gains

FIRST = ROOT/"outputs/expected_logodds_admm_20261113_v1"


def prepare(out):
    require((FIRST/"planning_complete.json").exists(), "Finish and preserve first pilot before the second protocol")
    require(not out.exists(), "Existing attempt is preserved")
    out.mkdir(parents=True)
    protocol = io.read(FIRST/"protocol.json")
    sources = {}
    for folder in (Path(__file__).parent, ROOT/"rnj_wzzt_dr/dr",
                   ROOT/"rnj_wzzt_dr/experiments/unified_posterior_dr_20261112"):
        for p in sorted(folder.glob("*.py")):
            relative = p.relative_to(ROOT); dest = out/"source_snapshot"/relative
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
            sources[relative.as_posix()] = io.ref(p)
    protocol.update(created_utc=utc(), version="convex_device_subset_with_sharing_ADMM",
        commit_at_lock=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        original_device_pilot=io.ref(FIRST/"planning_results.json"), sources=sources,
        development_reason="Original integer-resource subsolves left only about 12-13 ADMM iterations per 30-second start. Preserve them; separately test a declared convex restriction before new evaluation samples.",
        device_domain="hourly raw schedules already satisfy original power, energy, viability, reset and deadline constraints; fixed two-hour battery signs; original nominal clipping inactive; subset of original domain, not a relaxation",
        admm=dict(starts=["zero", "supporting_plane_if_feasible"], total_seconds_per_target=60.,
                  seed_max_seconds=10., iterations=500, initial_rho=10.,
                  original_integer_device_graph=False, continuous_device_QP=True,
                  seed_scope="one convex supporting-plane QP at zero; its generation cost included, no original-domain lower bound"),
        direct=dict(reused=True, source=io.ref(FIRST/"planning_results.json"),
                    scope="previous 60-second original-domain direct solve, same gamma and same input; no rerun"),
        model_confirmation=dict(paths=2048, namespace="expected_logodds_admm_model_20261113_v2", study_seed=2026111302),
        physical_confirmation=dict(paths=32, study_seed=2026111302, replicate_start=23000,
                                   policies=["no_dr", "rotation", "admm"]),
        future_opened=False, new_identifier_fits=0, new_parameter_fits=0)
    io.write(out/"protocol.json", protocol)
    io.write(out/"lock.json", dict(pid=os.getpid(), created_utc=utc(), protocol_sha256=sha(out/"protocol.json")))
    return protocol


def solve(data, directory, protocol):
    import gurobipy as gp
    evidence, original, problem = data["evidence"], data["original"], data["problem"]
    began = perf_counter()
    anchor = supporting_plane_start(problem, original, evidence, data["fee_scale"],
                                    seconds=protocol["admm"]["seed_max_seconds"])
    if anchor["qualified"]:
        np.savez_compressed(directory/"convex_seed.npz", x=anchor["x"], mean_u=anchor["mean_u"])
    io.write(directory/"convex_seed.json", {k:v for k,v in anchor.items() if k not in ("x", "mean_u")})
    starts = [("zero", np.zeros(problem.shape))]
    if anchor["qualified"]:
        starts.append(("supporting_plane", anchor["x"]))
    per_start = max(.01, (protocol["admm"]["total_seconds_per_target"]-(perf_counter()-began))/len(starts))
    results = []; candidates = []
    for label, requests in starts:
        target = directory/label; target.mkdir()
        setup_start = perf_counter(); steps = []
        requests = requests.copy(); requests[~evidence.active.any(axis=0)] = 0.
        initial = original.nominal(requests)
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0); env.start()
            try:
                for r in range(original.R):
                    steps.append(ConvexDeviceStep(problem, r, requests[r:r+1], data["fee_scale"], env)
                                 if evidence.active[:, r].any() else None)
                result = admm_run(evidence, steps, initial, original.fees, seconds=per_start,
                    iterations=protocol["admm"]["iterations"], rho=protocol["admm"]["initial_rho"],
                    setup_seconds=perf_counter()-setup_start,
                    checkpoint=lambda row:io.write(target/"last_iteration.json", row))
                if result["best"]:
                    best = result["best"]
                    x = np.vstack([a if a is not None else np.zeros((1, 24)) for a in best["requests"]])
                    replay = original.nominal(x)
                    independent = direct_gains(data["H"], data["item"]["target"], data["item"]["competitors"],
                        data["forecast"], data["score"]["sigma2"], original, data["base"], replay)
                    require(np.max(abs(independent-evidence.gains(replay))) < 1e-8, "Independent expected evidence differs")
                    require(original.audit(replay)["passed"] and min(independent) >= evidence.gain_required-1e-8,
                            "Convex-subset quote fails original device/gain check")
                    np.savez_compressed(target/"quote.npz", x=x, mean_u=replay)
                    candidates.append(dict(start=label, fee=original.fees(replay), gains=independent,
                        retained_iteration=best["iteration"], quote=io.ref(target/"quote.npz"),
                        origin="input_supporting_plane_seed" if best["iteration"] == -1 else "ADMM_iterate"))
                io.write(target/"trace.json", result.pop("trace"))
                best = result.pop("best")
                result["retained_iteration"] = best["iteration"] if best else None
                result["retained_fee"] = best["fee"] if best else None
                result.update(start=label); results.append(result)
                io.write(target/"summary.json", result)
            finally:
                for step in steps:
                    if step is not None:
                        step.close()
    best = min(candidates, key=lambda a:a["fee"]) if candidates else None
    if best:
        shutil.copyfile(best["quote"]["path"], directory/"quote.npz")
    summary = dict(qualified=best is not None, best=best, attempts=results,
        seed={k:v for k,v in anchor.items() if k not in ("x", "mean_u")},
        total_seconds=perf_counter()-began,
        first_feasible_seconds=anchor["total_seconds"] if anchor["qualified"] else
            next((r["first_feasible_seconds"] for r in results if r["first_feasible_seconds"] is not None), None),
        seed_cost_included=True, global_lower_bound=None, global_optimality_proved=False,
        domain_scope="fixed battery-sign, no nominal clipping convex subset of original devices")
    io.write(directory/"summary.json", summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();protocol=prepare(out);rows=[]
    originals={(r["id"],r["target"]):r for r in io.read(FIRST/"planning_results.json")}
    for item in protocol["targets"]:
        directory=out/"planning"/(item["id"]+f"__target{item['target']}");directory.mkdir(parents=True)
        try:
            data=load_target(item,protocol["gain_required"]);data["item"]=item
            io.write(directory/"input.json",dict(item=item,fee_scale=data["fee_scale"],
                parameter_sha256=data["score"]["parameter_sha256"],forecast_sha256=io.contract_hash(data["forecast"])))
            (directory/"admm").mkdir();result=solve(data,directory/"admm",protocol)
            direct=originals[item["id"],item["target"]].get("direct")
            if result["qualified"] and direct and direct.get("lower_bound") is not None:
                require(direct["lower_bound"] <= result["best"]["fee"]+1e-6,
                        "Original-domain native bound exceeds a replayed subset feasible quote")
            rows.append(dict(**item,status="complete",admm=result,direct_reused=direct))
            print(f"Convex-subset {item['id']} target {item['target']}: qualified={result['qualified']}, seed={result['seed']['qualified']}, seconds={result['total_seconds']:.2f}",flush=True)
        except Exception as exc:
            rows.append(dict(**item,status="failed",error=repr(exc),traceback=traceback.format_exc()))
            print(f"Retained failure {item['id']} target {item['target']}: {exc!r}",flush=True)
        io.write(out/"planning_results.json",rows)
    io.write(out/"planning_complete.json",dict(status="bounded_attempts_complete",created_utc=utc(),
        protocol_sha256=sha(out/"protocol.json"),results_sha256=sha(out/"planning_results.json"),targets=len(rows),
        failures=sum(r["status"]!="complete" for r in rows)))


if __name__=="__main__":
    main()
