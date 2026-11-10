"""Fresh held-out sampling of four already frozen paper15 contracts.

Three explicitly different probability models are evaluated. No result in
this file changes a quote, a model parameter, or the historical candidate pool.
Run directories must remain private. The output contains aggregate statistics
and privately stored batch scores, never test curves used by a planner.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback
import zipfile

for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[option] = "1"

import numpy as np
import scipy

import run as bounded
from continuous import IndependentResiduals
from gauss_frequency import gauss_probability


POLICY_ORDER = ("zero", "rotation", "old95", "current90")
MODES = {
    "conditional": dict(samples=32768, seed=2026111201, batch=256),
    "independent_full": dict(samples=8192, seed=2026111202, batch=64),
    "old_working": dict(samples=8192, seed=2026111203, batch=64),
}


def load_policies(problem, archive):
    """Use exactly the historical frozen plans, with a scalar replay audit."""
    folder = Path(archive) / bounded.CASE
    receipt_path = folder / "price_receipt.json"
    receipt = bounded.read(receipt_path)
    plans = bounded.unpickle(bounded.checked(receipt["plans"]))
    solution = bounded.unpickle(bounded.checked(receipt["solution"]))
    old = next(row for row in solution["records"] if row["label"] == "old_diagnostic")
    replay = problem["device"].nominal(np.asarray(old["x"]))
    if not np.allclose(replay, plans["old_diagnostic"]["mean_u"], atol=1e-10, rtol=0):
        raise ValueError("Old 95% contract does not replay its frozen requests")
    if not np.allclose(problem["device"].nominal(problem["frozen_x"]),
                       plans["candidate"]["mean_u"], atol=1e-10, rtol=0):
        raise ValueError("Current 90% contract does not replay")
    names = dict(zero="zero", rotation="rotation", old95="old_diagnostic", current90="candidate")
    probabilities = dict(zero=solution["records"][0]["probability"], rotation=None,
                         old95=old["probability"], current90=solution["best"]["probability"])
    policies = {}
    for name, source in names.items():
        value = np.asarray(plans[source]["mean_u"], float)
        fee = problem["device"].fees(value)
        if not problem["device"].audit(value)["passed"]:
            raise ValueError("Frozen contract violates original resource constraints")
        if not np.isclose(fee, plans[source]["cost"], atol=1e-8, rtol=0):
            raise ValueError("Contract fee changed")
        policies[name] = dict(mean_u=value, fee=fee,
                              planning_probability=probabilities[name], source_policy=source,
                              planning_probability_scope=("not used for planning" if name=="rotation" else
                                  "original 90pct stage fixed-outer reevaluation" if name in ("zero","old95") else
                                  "original 90pct stage selected planning value"))
    references = dict(price_receipt=bounded.reference(receipt_path),
                      plans=receipt["plans"], solution=receipt["solution"])
    return policies, references


def condition_means(problem, policies):
    means = {}
    for name, policy in policies.items():
        means[name] = {}
        for k in ("C", "F"):
            background = problem["c"] + problem["mean_power"] @ (problem["h0"] - problem["matrices"][k]).T
            effect = np.asarray(problem["packet"]["effect_" + k])
            means[name][k] = background + np.einsum("ir,rt->ti", effect, policy["mean_u"])
    return means


def conditional_predictions(problem, policies, *, policy_order=None, normalization_policy="current90"):
    policy_order = POLICY_ORDER if policy_order is None else tuple(policy_order)
    means = condition_means(problem, policies)
    baseline = IndependentResiduals(means[normalization_policy]["C"].ravel(),
                                    means[normalization_policy]["F"].ravel(), problem["sigma"].ravel())
    normalization = baseline.sum_mean, baseline.sum_sd
    results = {}
    for name in policy_order:
        model = IndependentResiduals(means[name]["C"].ravel(), means[name]["F"].ravel(),
                                     problem["sigma"].ravel())
        results[name] = gauss_probability(model, problem["packet"]["threshold"],
                                          cutoff=512., order=16, refinements=2,
                                          tolerance=1e-7, normalization=normalization)
    return results


def resource_valid(device, delivered):
    """Vectorized counterpart of Devices.audit, retaining every sampled path."""
    power = np.maximum(device.delta_lower_kw[None] - delivered, 0.).max(axis=(1, 2))
    power = np.maximum(power, np.maximum(delivered - device.delta_upper_kw[None], 0.).max(axis=(1, 2)))
    energy = np.zeros(delivered.shape[:2])
    violation = np.zeros(delivered.shape[0])
    deadline = np.zeros(delivered.shape[0])
    for t in range(96):
        energy = np.where(device.reset[None, :, t], 0., energy)
        slot = delivered[:, :, t]
        energy += device.dt * np.where(slot >= 0, device.eta, 1. / device.eta) * slot
        violation = np.maximum(violation, np.maximum(device.energy_low[None] - energy, 0.).max(axis=1))
        violation = np.maximum(violation, np.maximum(energy - device.energy_high[None], 0.).max(axis=1))
        ends = (device.kind != 1) & ((t == 95) | ((device.kind == 2) & (t == 31)))
        if np.any(ends):
            deadline = np.maximum(deadline, np.abs(energy[:, ends]).max(axis=1))
    valid = np.maximum.reduce([power, violation, deadline]) < 1e-7
    audit = dict(max_power_violation_kw=float(power.max()),
                 max_energy_violation_kwh=float(violation.max()),
                 max_deadline_violation_kwh=float(deadline.max()))
    return valid, audit


def sample_batch(problem, policies, means, mode, index, size, *, policy_order=None, mode_spec=None):
    policy_order = POLICY_ORDER if policy_order is None else tuple(policy_order)
    spec = MODES[mode] if mode_spec is None else mode_spec
    seed = np.random.SeedSequence([spec["seed"], index])
    independent_seed, response_seed = seed.spawn(2)
    rng = np.random.default_rng(independent_seed)
    device = problem["device"]
    if mode == "conditional":
        voltage = rng.standard_normal((size, 96, 15)) * problem["sigma"]
    elif mode == "independent_full":
        power = problem["mean_power"] + rng.standard_normal((size, 96, 30)) * problem["power_sd"]
        multiplier = 1. + device.relative_sd[None] * rng.standard_normal((size, device.R, 96))
        meters = rng.standard_normal((size, 96, 30))
        voltage = rng.standard_normal((size, 96, 15)) * problem["sigma"]
    else:
        dimension = np.asarray(problem["packet"]["L_C"]).shape[1]
        background = rng.standard_normal((size, dimension))
        child = int(response_seed.generate_state(1)[0])
        multiplier = device.draw(child, size, stress=False)
    gaps, valid, energy = [], [], []
    audits = {}
    for name in policy_order:
        policy = policies[name]
        if mode == "conditional":
            residual = {k: means[name][k] + voltage for k in ("C", "F")}
            valid_policy = np.ones(size, bool)
            kwh = np.full(size, .25 * np.abs(policy["mean_u"]).sum())
        else:
            delivered = device.project(policy["mean_u"][None] * multiplier)
            valid_policy, audits[name] = resource_valid(device, delivered)
            kwh = .25 * np.abs(delivered).sum(axis=(1, 2))
            residual = {}
            if mode == "old_working":
                for k in ("C", "F"):
                    base = (np.asarray(problem["packet"]["mu_" + k])[None]
                            + background @ np.asarray(problem["packet"]["L_" + k]).T)
                    effect = np.asarray(problem["packet"]["effect_" + k])
                    residual[k] = base.reshape(size, 96, 15) + np.einsum("ir,srt->sti", effect, delivered)
            else:
                node = np.eye(15)[device.node]
                dp = np.einsum("srt,rn->stn", delivered, node) / problem["base_kw"]
                dq = np.einsum("srt,rn,r->stn", delivered, node, device.qratio) / problem["base_kw"]
                gross = power + np.concatenate((dp, dq), axis=2)
                meter_error = meters * problem["pq_sigma"] * np.abs(gross)
                for k in ("C", "F"):
                    natural = power @ (problem["h0"] - problem["matrices"][k]).T
                    action = np.einsum("ir,srt->sti", np.asarray(problem["packet"]["effect_" + k]), delivered)
                    residual[k] = problem["c"] + natural + action + voltage - meter_error @ problem["matrices"][k].T
        gap = (np.abs(residual["C"]).mean(axis=(1, 2))
               - 1.03 * np.abs(residual["F"]).mean(axis=(1, 2)))
        valid_policy &= np.isfinite(gap)
        gaps.append(gap)
        valid.append(valid_policy)
        energy.append(kwh)
    return dict(gap=np.asarray(gaps), valid=np.asarray(valid), energy=np.asarray(energy)), audits


def aggregate_batches(folder, spec, threshold, *, policy_order=None, main_policy="current90"):
    policy_order = POLICY_ORDER if policy_order is None else tuple(policy_order)
    arrays = []
    for file in sorted(folder.glob("batch_*.npz")):
        with np.load(file, allow_pickle=False) as saved:
            arrays.append({k: saved[k] for k in ("gap", "valid", "energy")})
    if not arrays:
        return dict(requested=spec["samples"], completed=0, policies={}, paired={})
    values = {k: np.concatenate([a[k] for a in arrays], axis=1) for k in arrays[0]}
    completed = values["gap"].shape[1]
    policies = {}
    passed = values["gap"] > threshold
    for i, name in enumerate(policy_order):
        valid = values["valid"][i]
        successes = int(np.sum(passed[i] & valid))
        count = int(valid.sum())
        failures = completed - count
        missing = spec["samples"] - completed
        policies[name] = dict(successes=successes, valid=count, attempted=completed,
                              requested=spec["samples"], failures=failures, missing=missing,
                              probability=successes / count if count else None,
                              standard_error=math.sqrt((successes/count)*(1-successes/count)/count) if count else None,
                              wilson_95=bounded.wilson(successes, count),
                              unresolved_probability_bounds=[successes/spec["samples"],
                                  (successes+failures+missing)/spec["samples"]],
                              mean_D=float(values["gap"][i, valid].mean()) if count else None,
                              sd_D=float(values["gap"][i, valid].std(ddof=1)) if count>1 else None,
                              mean_absolute_DR_kwh=float(values["energy"][i, valid].mean()) if count else None)
    paired = {}
    main = policy_order.index(main_policy)
    for i, name in enumerate(policy_order):
        if name == main_policy:
            continue
        valid = values["valid"][main] & values["valid"][i]
        difference = passed[main, valid].astype(float) - passed[i, valid].astype(float)
        n = difference.size
        paired[name] = dict(valid=n, requested=spec["samples"],
                            **{main_policy+"_only":int(np.sum(difference>0))}, comparator_only=int(np.sum(difference<0)),
                            same=int(np.sum(difference==0)),
                            probability_difference=float(difference.mean()) if n else None,
                            paired_standard_error=float(difference.std(ddof=1)/np.sqrt(n)) if n>1 else None,
                            mean_D_difference=float((values["gap"][main, valid]-values["gap"][i, valid]).mean()) if n else None)
    return dict(requested=spec["samples"], completed=completed, seed=spec["seed"], policies=policies, paired=paired)


def run_sampling(problem, policies, means, mode, out, seconds=180, *,
                 policy_order=None, mode_spec=None, main_policy="current90"):
    policy_order = POLICY_ORDER if policy_order is None else tuple(policy_order)
    folder = out / mode
    folder.mkdir(exist_ok=True)
    spec = MODES[mode] if mode_spec is None else mode_spec
    if (folder / "complete.json").exists():
        return bounded.read(folder / "summary.json")
    start = time.perf_counter()
    timing = bounded.read(folder / "timing.json") if (folder / "timing.json").exists() else {}
    for index, position in enumerate(range(0, spec["samples"], spec["batch"])):
        destination = folder / f"batch_{index:05d}.npz"
        if destination.exists():
            continue
        if time.perf_counter()-start > seconds:
            break
        size = min(spec["batch"], spec["samples"]-position)
        batch_start = time.perf_counter()
        try:
            arrays, audits = sample_batch(problem, policies, means, mode, index, size,
                                          policy_order=policy_order, mode_spec=spec)
        except Exception as exc:
            bounded.write(folder / f"batch_{index:05d}_failure.json", dict(index=index,size=size,error=repr(exc),
                                                                        traceback=traceback.format_exc()))
            raise
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
        timing[str(index)] = dict(seconds=time.perf_counter()-batch_start, size=size, audits=audits)
        bounded.write(folder / "timing.json", timing)
    result = aggregate_batches(folder, spec, problem["packet"]["threshold"],
                               policy_order=policy_order, main_policy=main_policy)
    result.update(seconds=float(sum(item["seconds"] for item in timing.values())),
                  batch_count=len(timing), model=mode)
    bounded.write(folder / "summary.json", result)
    bounded.write(folder / "complete.json", dict(status="complete" if result["completed"]==spec["samples"] else "time_limit",
                                                summary_sha256=bounded.sha(folder / "summary.json")))
    return result


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "complete.json").exists():
        print("Already complete", flush=True)
        return
    lock = args.out / "running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        problem = bounded.build_problem(args.archive.resolve(), args.public.resolve())
        policies, quote_refs = load_policies(problem, args.archive.resolve())
        sources = dict(problem["sources"])
        for file in (Path(__file__), bounded.HERE/"gauss_frequency.py"):
            sources[str(file.relative_to(bounded.TASK))] = bounded.sha(file)
        request = dict(case=bounded.CASE, protocol=bounded.reference(args.protocol.resolve()),
                       inputs=problem["inputs"], quotes=quote_refs, sources=sources,
                       policy_order=POLICY_ORDER, specs=MODES, seconds_per_mode=180,
                       batch_seed_rule="SeedSequence([mode_seed,batch_index]).spawn(2)",
                       gaussian_inversion=dict(cutoff=512,order=16,refinements=2,tolerance=1e-7),
                       python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                       no_price_search=True,no_AC_run=True,new_two_variable_result_deduplicated_to="current90")
        path = args.out / "request.json"
        if path.exists() and bounded.read(path) != bounded.plain(request):
            raise ValueError("Changed inputs or sources in sampling comparison stage")
        bounded.write(path, request)
        if not (args.out / "source_snapshot.zip").exists():
            with zipfile.ZipFile(args.out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
                for filename in sources:
                    snapshot.write(bounded.TASK / filename, filename)
        results_path = args.out / "results.json"
        results = bounded.read(results_path) if results_path.exists() else {}
        results["quotes"] = {name:{k:v for k,v in row.items() if k!="mean_u"} for name,row in policies.items()}
        if "conditional_predictions" not in results:
            results["conditional_predictions"] = conditional_predictions(problem, policies)
            bounded.write(results_path, results)
        means = condition_means(problem, policies)
        for mode in MODES:
            if mode not in results:
                results[mode] = run_sampling(problem, policies, means, mode, args.out)
                bounded.write(results_path, results)
        finished = all(results[mode]["completed"] == MODES[mode]["samples"] for mode in MODES)
        bounded.write(args.out / "complete.json", dict(status="complete" if finished else "bounded_time_limit",results_sha256=bounded.sha(results_path),
                   mode_completion={mode:results[mode]["completed"] for mode in MODES},no_new_quotes=True,no_AC=True))
        print(json.dumps(bounded.plain(results),ensure_ascii=True),flush=True)
    except Exception as exc:
        bounded.write(args.out / "failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "public", "protocol", "out"):
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
