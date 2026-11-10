"""Publish one paper15 forecast, freeze a price using only it and history, then score AC.

Keep all run directories private. The planner's two inputs are the released
directory and its own integration seeds. The evaluator process owns the
natural trajectory, forecast-noise seed, physical generator, and response seed.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import sys
import uuid

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
REPO = TASK.parents[1]
sys.path[:0] = [str(HERE), str(TASK / "dr_heterogeneous_pilot"), str(REPO)]
import driver as original
from forecast import condition_packet, publish_forecast, training_scale
from research_experiments.wzzt_dr_selection.dr_p90_cost.cost_only import solve
from research_experiments.wzzt_dr_selection.dr_paper15_delivery_margin.ac_replay import score_scene

CASE = "paper15_b2_v0005"
PHYSICAL = "paper15"
PLANS = ("zero", "rotation", "candidate")


def code_hashes():
    paths = [Path(__file__), HERE / "forecast.py", Path(original.__file__),
             TASK / "dr_heterogeneous_pilot/devices.py", TASK / "dr_heterogeneous_pilot/planner.py",
             TASK / "dr_p90_cost/cost_only.py", TASK / "dr_paper15_delivery_margin/ac_replay.py",
             TASK / "dr_conditional_gaussian/conditional_integral.py", TASK / "dr_conditional_gaussian/problem.py",
             TASK / "dr_frozen_distance/frozen_pool.py", TASK / "dr_heterogeneous_resources/build_allocations.py",
             TASK / "dr_newday_weighted/weighted_selection.py", TASK / "dr_spatial_correlated/ac_bridge.py"]
    return {str(p.relative_to(TASK)): original.sha(p) for p in paths}


def inventory(folder):
    return {f.relative_to(folder).as_posix(): original.sha(f)
            for f in sorted(folder.rglob("*")) if f.is_file() and f != folder / "receipt.json"}


def verify(folder):
    receipt = original.read(folder / "receipt.json")
    if receipt["files"] != inventory(folder):
        raise ValueError(f"Changed or incomplete frozen stage: {folder}")
    return receipt


@contextmanager
def stage(destination, inputs):
    """A completed immutable stage is reused; no partial result is published."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if verify(destination)["inputs"] != inputs:
            raise ValueError(f"Different inputs for completed stage: {destination}")
        print(f"Already complete: {destination}", flush=True)
        yield None
        return
    lock = destination.with_name(destination.name + ".lock")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()}\n")
    tmp = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    try:
        if destination.exists():
            if verify(destination)["inputs"] != inputs:
                raise ValueError("Different inputs in concurrently completed stage")
            yield None
            return
        tmp.mkdir()
        original.write(tmp / "request.json", inputs)
        yield tmp
        original.write(tmp / "receipt.json", dict(inputs=inputs, files=inventory(tmp)))
        tmp.rename(destination)
    except Exception as exc:
        if tmp.exists():
            original.write(tmp / "failure.json", {"error": repr(exc)})
        raise
    finally:
        lock.unlink()


def runtime(archive, condition):
    return original.open_runtime(condition["source_root"], condition["source_metadata"], repo_root=REPO,
                                 protocol_ref=original.ref(archive / "protocol.json"), labels=condition["labels"],
                                 case=PHYSICAL, v_sigma=condition["voltage_sigma"], pq_sigma=condition["pq_sigma"])


def stack_training(condition):
    hist = original.load(original.checked(condition["history_packet"]))
    hist = hist.get("history", hist)
    pqs = [np.column_stack(original._power(day, condition["labels"])) for day in hist["training"]]
    if np.shape(pqs) != (3, 96, 30) or sum(len(day["P_terminal"]) for day in hist["validation"]) != 288:
        raise ValueError("Expected three full training days and 288 historical validation slots")
    return np.asarray(pqs)


def prepare(args):
    archive = args.archive.resolve()
    folder = archive / CASE
    # Protocol/receipts/solver starts contain only data available before the old AC holdout.
    protocol = original.read(archive / "protocol.json")
    locked = original.read(folder / "inputs.json")
    condition = original.read(original.checked(locked["case_protocol.json"]))
    price_receipt = original.read(folder / "price_receipt.json")
    root = args.root.resolve()
    inputs = dict(case=CASE, archive_protocol=original.ref(archive / "protocol.json"),
                  frozen_inputs=locked, prior_training_solution=price_receipt["solution"]["sha256"],
                  profile=args.profile, replicate=args.replicate, forecast_seed=args.forecast_seed,
                  execution_seed=args.actual_seed, sources=code_hashes(),
                  python=platform.python_version(), numpy=np.__version__)
    if args.forecast_seed == args.actual_seed:
        raise ValueError("Require separate forecast and execution streams")
    with stage(root / "prepared", inputs) as tmp:
        if tmp is None:
            return
        public = tmp / "public"
        public.mkdir()
        training = stack_training(condition)
        packet = original.load(original.checked(locked["problem.pkl"]))
        if list(packet["labels"]) != condition["labels"] or not packet["has_dispute"]:
            raise ValueError("Frozen disputed pair/labels required")
        cfg = original.read(original.checked(protocol["library"]))
        alloc = original.read(original.checked(protocol["allocations"]))["cases"]
        allocation = next(item for item in alloc if item["case"] == PHYSICAL)
        initial = original.load(original.checked(price_receipt["solution"]))["best"]["x"]
        # Do not publish private source paths or evaluator seeds in planner data.
        solver_fields = ["mu_C", "mu_F", "L_C", "L_F", "effect_C", "effect_F", "r0", "x0",
                         "labels", "threshold", "eligible", "has_dispute"]
        solver_packet = {k: packet[k] for k in solver_fields}
        original.dump(public / "history.pkl", dict(training=training, packet=solver_packet,
                      config=cfg, allocation=allocation, initial=initial))
        rt = runtime(archive, condition)
        bg = original.build_background(rt, args.profile, args.replicate)
        if list(bg.p_base.columns) != condition["labels"] or list(bg.q_base.columns) != condition["labels"]:
            raise ValueError("Natural producer order differs from frozen terminals")
        natural = np.column_stack((bg.p_base, bg.q_base))
        issued = publish_forecast(natural, training_scale(training), np.random.default_rng(args.forecast_seed))
        np.save(public / "forecast.npy", issued, allow_pickle=False)
        public_inputs = dict(case=CASE, labels=condition["labels"], train_slots=288, validation_slots=288,
                             history_sha256=condition["history_packet"]["sha256"],
                             historical_pair_sha256=locked["pool.json"]["sha256"],
                             prior_training_solution=price_receipt["solution"]["sha256"],
                             planning_code=code_hashes())
        original.write(public / "receipt.json", dict(inputs=public_inputs, files=inventory(public)))
        # Save the exact day including original meter draws/seed; no regeneration in evaluation.
        original.dump(tmp / "natural_private.pkl", bg)
        print("Prepared one forecast; public SHA256", original.sha(public / "forecast.npy"), flush=True)


def planner(args):
    public = args.public.resolve()
    info = verify(public)
    if info["inputs"]["planning_code"] != code_hashes() or args.background_seed == args.response_seed:
        raise ValueError("Changed planning code or reused seed")
    inputs = dict(case=CASE, public_receipt=original.sha(public / "receipt.json"), sources=code_hashes(),
                  background_seed=args.background_seed, response_seed=args.response_seed,
                  samples=128, maxiter=40, seconds=180, target=.90)
    with stage(args.out.resolve(), inputs) as tmp:
        if tmp is None:
            return
        history = original.load(public / "history.pkl")
        issued = np.load(public / "forecast.npy", allow_pickle=False)
        packet = condition_packet(history["packet"], history["training"], issued)
        dev = original.Devices(history["allocation"], history["config"], original.profiles)
        prob = original.Probability(packet, dev, inputs["samples"], args.background_seed, args.response_seed)
        start = np.asarray(history["initial"])
        sol = solve(prob, [("zero", np.zeros_like(start)), ("historical_quote", start)],
                    target=inputs["target"], maxiter=inputs["maxiter"], seconds=inputs["seconds"])
        best = sol["best"]
        candidate = best["mean_u"]
        rotation = original.simple_plan(dev, "rotation")
        full_cost = dev.fees(rotation)
        rotation *= min(1., np.sqrt(best["cost"] / full_cost)) if full_cost else 0.
        plans = dict(zero=np.zeros_like(candidate), rotation=rotation, candidate=candidate)
        fees = {k: dev.fees(u) for k, u in plans.items()}
        if not all(dev.audit(u)["passed"] for u in plans.values()):
            raise ValueError("Infeasible frozen nominal")
        summary = dict(qualified=sol["qualified"], solver_status=sol["status"], solver_success=sol["solver_success"],
                       role="qualified" if sol["qualified"] else "unqualified_diagnostic",
                       planning_probability=best["probability"], fees=fees,
                       rotation_matched=bool(np.isclose(fees["rotation"], fees["candidate"], atol=1e-7, rtol=0)),
                       starts=[{k:v for k,v in row.items() if k in ("label","probability","cost","invalid")}
                               for row in sol["records"][:2]], iterations=len(sol["trace"]),
                       seconds=sol["seconds"], forecast_sha256=original.sha(public / "forecast.npy"),
                       forecast_conditioning=packet["forecast_conditioning"])
        original.dump(tmp / "plans.pkl", plans)
        original.dump(tmp / "solver_private.pkl", sol)
        original.write(tmp / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)


def evaluate(args):
    root, archive = args.root.resolve(), args.archive.resolve()
    frozen = root / "planned"
    prepared = root / "prepared"
    plan_receipt = verify(frozen)
    day_receipt = verify(prepared)
    public = prepared / "public"
    if plan_receipt["inputs"]["public_receipt"] != original.sha(public / "receipt.json"):
        raise ValueError("Price does not belong to published forecast")
    if day_receipt["inputs"]["archive_protocol"] != original.ref(archive / "protocol.json"):
        raise ValueError("Archive differs from the one used to seal the day")
    seeds = [day_receipt["inputs"][k] for k in ["forecast_seed","execution_seed"]]
    seeds += [plan_receipt["inputs"][k] for k in ["background_seed","response_seed"]]
    if len(set(seeds)) != len(seeds) or day_receipt["inputs"]["sources"] != code_hashes():
        raise ValueError("Invalid seed segregation or changed source")
    inputs = dict(case=CASE, price_receipt=original.sha(frozen/"receipt.json"),
                  day_receipt=original.sha(prepared/"receipt.json"), sources=code_hashes())
    with stage(root / "evaluated", inputs) as tmp:
        if tmp is None:
            return
        artifacts = day_receipt["inputs"]["frozen_inputs"]
        condition = original.read(original.checked(artifacts["case_protocol.json"]))
        pool = original.read(original.checked(artifacts["pool.json"]))
        rt = runtime(archive, condition)
        bg = original.load(prepared / "natural_private.pkl")
        history = original.load(public / "history.pkl")
        dev = original.Devices(history["allocation"], history["config"], original.profiles)
        multiplier = dev.draw(seeds[1], 1, stress=False)[0]
        prices = original.load(frozen / "plans.pkl")
        qualification = original.read(frozen / "summary.json")
        rows = []
        for policy in PLANS:
            u = prices[policy]
            row = dict(policy=policy, fee=dev.fees(u), planning_qualified=qualification["qualified"]
                       if policy=="candidate" else None)
            try:
                result = original.execute(rt, bg, dev, u, multiplier, f"{CASE}:forecast_day:{policy}")
                original.dump(tmp / (policy + "_observation.pkl"), result)
                row.update(status=result["summary"]["status"], converged=result["summary"]["converged"])
                if row["status"] == "complete":
                    scored = original.score(pool, result["scene"], history_gate=True)
                    rule = next(s for s in scored["rules"] if s["new_day_weight"]==2 and s["new_day_fraction"]==.03)
                    losses = {s["candidate_id"]:s["new_validation_mae"] for s in scored["loss_rows"]}
                    row.update(selected=rule["selected_candidate_id"], excluded="coarse" not in rule["eligible_candidate_ids"],
                               D_micro=1e6*(losses["coarse"]-1.03*losses["refined"]),
                               delivered_kwh=result["summary"]["resource_audit"]["absolute_DR_kwh"])
            except Exception as exc:
                row.update(status="failed", error=repr(exc))
            rows.append(row)
        original.write(tmp / "predictions_frozen.json", dict(rows=rows, true_support_used=False))
        truth = {tuple(sorted(b)) for b in rt["legacy"].evaluate_supports([],rt["context"].net)["missed_blocks"]}
        for row in rows:
            if row["status"]=="complete":
                obs = original.load(tmp / (row["policy"]+"_observation.pkl"))["scene"]
                selected, excluded, errors, gap = score_scene(obs, pool, truth)
                if (selected,excluded)!=(row["selected"],row["excluded"]) or abs(gap-row["D_micro"])>1e-8:
                    raise ArithmeticError("Independent selection replay failed")
                row.update(errors=errors, complete_recovery=errors==0)
        forecast_error = np.load(public/"forecast.npy",allow_pickle=False)-np.column_stack((bg.p_base,bg.q_base))
        n=len(pool["labels"])
        stats={key:dict(rmse_pu=float(np.sqrt((a*a).mean())),mean_signed_error_pu=float(a.mean()))
               for key,a in [("P",forecast_error[:,:n]),("Q",forecast_error[:,n:])]}
        summary=dict(natural_days=1, execution_draws=1, policy_denominator=len(rows), response_law="main",
                     forecast_error=stats, threshold_micro=history["packet"]["threshold"], rows=rows)
        original.write(tmp / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for key in ("archive","root"):
        prep.add_argument("--"+key, type=Path, required=True)
    for key in ("profile","replicate","forecast-seed","actual-seed"):
        prep.add_argument("--"+key, type=int, required=True)
    plan = sub.add_parser("plan")
    for key in ("public","out"):
        plan.add_argument("--"+key, type=Path, required=True)
    for key in ("background-seed","response-seed"):
        plan.add_argument("--"+key, type=int, required=True)
    evaluation = sub.add_parser("evaluate")
    for key in ("root","archive"):
        evaluation.add_argument("--"+key, type=Path, required=True)
    args = cli.parse_args()
    {"prepare":prepare, "plan":planner, "evaluate":evaluate}[args.action](args)
