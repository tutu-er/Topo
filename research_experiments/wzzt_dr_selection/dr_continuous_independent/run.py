"""Bounded paper15 continuous-inversion checks using frozen historical inputs.

The inversion stage conditions on a fixed forecast and delivered nominal power.
The full independent-primitive benchmark is a separate linear-reference Monte
Carlo diagnostic. It is not an AC experiment or a no-sampling total probability.
All run directories must remain private because they contain derived powers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback
import zipfile

for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[option] = "1"

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
REPO = TASK.parents[1]
sys.path[:0] = [str(HERE), str(TASK / "dr_heterogeneous_pilot"), str(REPO)]
from continuous import IndependentResiduals
from devices import Devices
from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles
from research_experiments.wzzt_dr_selection.dr_paper15_forecast_day.forecast import BIAS, RELATIVE_SD, training_scale

CASE = "paper15_b2_v0005"
POLICIES = {"zero": [0., 0.], "half": [.5, .5], "frozen": [1., 1.]}
CONDITIONAL_SEED = 2026111101
FULL_SEED = 2026111102


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def checked(item):
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError("Changed archived input: " + str(path))
    return path


def unpickle(path):
    return pickle.loads(Path(path).read_bytes())


def build_problem(archive, public):
    """Read historical fit, one issued forecast, and a pre-existing frozen price."""
    folder = archive / CASE
    public_receipt = read(public / "receipt.json")
    for filename, fingerprint in public_receipt["files"].items():
        if sha(public / filename) != fingerprint:
            raise ValueError("Changed published planner input: " + filename)
    inputs = read(folder / "inputs.json")
    condition = read(checked(inputs["case_protocol.json"]))
    pool = read(checked(inputs["pool.json"]))
    packet = unpickle(checked(inputs["problem.pkl"]))
    price_receipt = read(folder / "price_receipt.json")
    best = unpickle(checked(price_receipt["solution"]))["best"]
    published = unpickle(public / "history.pkl")
    training = np.asarray(published["training"])
    forecast = np.load(public / "forecast.npy", allow_pickle=False)
    history = unpickle(checked(condition["history_packet"]))
    history = history.get("history", history)
    if training.shape != (3, 96, 30) or forecast.shape != (96, 30):
        raise ValueError("Expected complete paper15 historical/forecast channels")
    if list(packet["labels"]) != pool["labels"]:
        raise ValueError("Candidate terminal order changed")

    device = Devices(published["allocation"], published["config"], profiles)
    frozen_x = np.asarray(best["x"]).reshape(device.R, 24)
    if not np.isclose(device.fees(device.nominal(frozen_x)), best["cost"], atol=1e-8, rtol=0):
        raise ArithmeticError("Archived quote does not replay")
    reference_matrix = 1e6 * np.column_stack((packet["r0"], packet["x0"]))
    matrices = {}
    for identity, code in (("coarse", "C"), ("refined", "F")):
        model = next(m for m in pool["models"] if m["candidate_id"] == identity)
        matrices[code] = 1e6 * np.column_stack((model["R"], model["X"]))
    mean_training = training.mean(axis=0)
    constants = {k: np.asarray(packet["mu_" + k]).reshape(96, 15)
                 - mean_training @ (reference_matrix - matrices[k]).T for k in matrices}
    if not np.allclose(constants["C"], constants["F"], atol=1e-8, rtol=0):
        raise ArithmeticError("Candidates do not share one reference mean residual")

    scale = training_scale(training)
    mean_power = forecast - BIAS[:, None] * scale
    power_sd = RELATIVE_SD[:, None] * scale
    voltage = np.stack([np.asarray(day["V_terminal"], float) for day in history["training"]])
    root = np.stack([np.asarray(day["V_root_observed"], float).reshape(96, 1)
                     for day in history["training"]])
    # This deliberately assumes independent squared-drop innovations, matching
    # each channel's first-order root-plus-terminal variance. Shared root
    # correlation is omitted under the user's independent-noise permission.
    sigma = 2 * condition["voltage_sigma"] * np.sqrt(
        np.mean(root**4, axis=0) + np.mean(voltage**4, axis=0)) * 1e6
    if sigma.shape != (96, 15) or not np.isfinite(sigma).all() or not np.all(sigma > 0):
        raise ValueError("Positive finite voltage innovation scales required")
    sources = [Path(__file__), HERE / "continuous.py", TASK / "dr_heterogeneous_pilot/devices.py",
               TASK / "dr_paper15_forecast_day/forecast.py",
               TASK / "dr_heterogeneous_resources/build_allocations.py"]
    fingerprints = {str(path.relative_to(TASK)): sha(path) for path in sources}
    fingerprints_inputs = {"archived_inputs": inputs,
                           "archived_quote": price_receipt["solution"],
                           "historical_packet": condition["history_packet"],
                           "public_history": reference(public / "history.pkl"),
                           "issued_forecast": reference(public / "forecast.npy")}
    return dict(device=device, frozen_x=frozen_x, matrices=matrices, h0=reference_matrix,
                packet=packet, c=constants["C"], mean_power=mean_power, power_sd=power_sd,
                sigma=sigma, pq_sigma=condition["pq_sigma"], base_kw=pool["base_kw"],
                sources=fingerprints, inputs=fingerprints_inputs)


def nominal(problem, theta):
    device = problem["device"]
    scale = np.repeat(np.asarray(theta, float), 12)
    x = problem["frozen_x"] * scale
    delivered = device.nominal(x)
    if not device.audit(delivered)["passed"]:
        raise ArithmeticError("Nominal output violates the unchanged resource domain")
    means = {}
    for k in ("C", "F"):
        base = problem["c"] + problem["mean_power"] @ (problem["h0"] - problem["matrices"][k]).T
        effect = np.asarray(problem["packet"]["effect_" + k])
        means[k] = (base + np.einsum("ir,rt->ti", effect, delivered)).ravel()
    return dict(x=x, delivered=delivered, a_C=means["C"], a_F=means["F"],
                fee=device.fees(delivered), absolute_kwh=.25 * np.abs(delivered).sum())


def kernel(problem, theta, options, directions=False):
    value = nominal(problem, theta)
    model = IndependentResiduals(value["a_C"], value["a_F"], problem["sigma"].ravel())
    options = dict(options, normalization=problem["normalization"])
    if directions:
        step = 1e-5
        dc, df, fee_gradient = [], [], []
        for j in range(2):
            left, right = np.asarray(theta, float).copy(), np.asarray(theta, float).copy()
            left[j] = max(0., left[j]-step)
            right[j] = min(1., right[j]+step)
            width = right[j]-left[j]
            if width <= 0:
                raise ValueError("Probability directions require a feasible theta point")
            a, b = nominal(problem, left), nominal(problem, right)
            dc.append((b["a_C"] - a["a_C"]) / width)
            df.append((b["a_F"] - a["a_F"]) / width)
            fee_gradient.append((b["fee"] - a["fee"]) / width)
        options = dict(options, daC=np.column_stack(dc), daF=np.column_stack(df))
    result = model.probability(problem["packet"]["threshold"], **options)
    result.update(theta=list(np.asarray(theta, float)), fee=value["fee"],
                  absolute_kwh=value["absolute_kwh"])
    if directions:
        result["fee_gradient"] = fee_gradient
    return result


def wilson(successes, count):
    if not count:
        return None
    z = float(norm.ppf(.975))
    center = (successes + z*z/2) / (count + z*z)
    radius = z * np.sqrt(successes * (count-successes)/count + z*z/4) / (count + z*z)
    return [float(center-radius), float(center+radius)]


def monte_carlo(problem, full=False, limit=120):
    """Independent validation only; no Monte Carlo value is used by the optimizer."""
    count = 2048 if full else 32768
    batch = 64 if full else 256
    seed = FULL_SEED if full else CONDITIONAL_SEED
    policies = POLICIES if full else {"frozen": POLICIES["frozen"]}
    frozen = {name: nominal(problem, theta) for name, theta in policies.items()}
    summaries = {name: {"successes": 0, "count": 0, "sum_D": 0., "sum_D2": 0.,
                        "delivery_kwh_sum": 0., "failures": 0} for name in policies}
    rng = np.random.default_rng(seed)
    began = time.perf_counter()
    device = problem["device"]
    n = 15
    for start in range(0, count, batch):
        if time.perf_counter() - began > limit:
            break
        size = min(batch, count-start)
        if full:
            power_error = rng.standard_normal((size, 96, 30)) * problem["power_sd"]
            delivery_error = rng.standard_normal((size, device.R, 96))
            metering_error = rng.standard_normal((size, 96, 30))
        voltage_error = rng.standard_normal((size, 96, 15)) * problem["sigma"]
        for name, value in frozen.items():
            row = summaries[name]
            try:
                if full:
                    delivered = device.project(value["delivered"][None] *
                        (1. + device.relative_sd[None] * delivery_error))
                    dp = np.einsum("srt,rn->stn", delivered,
                        np.eye(n)[device.node]) / problem["base_kw"]
                    dq = np.einsum("srt,rn,r->stn", delivered,
                        np.eye(n)[device.node], device.qratio) / problem["base_kw"]
                    gross_power = problem["mean_power"] + power_error + np.concatenate((dp, dq), axis=2)
                    pq_noise = metering_error * problem["pq_sigma"] * np.abs(gross_power)
                    residuals = {}
                    for k in ("C", "F"):
                        natural = (problem["mean_power"] + power_error) @ (problem["h0"]-problem["matrices"][k]).T
                        effect = np.asarray(problem["packet"]["effect_" + k])
                        action = np.einsum("ir,srt->sti", effect, delivered)
                        residuals[k] = (problem["c"] + natural + action + voltage_error
                                        - pq_noise @ problem["matrices"][k].T)
                    kwh = .25 * np.abs(delivered).sum(axis=(1, 2))
                else:
                    residuals = {k: value["a_"+k].reshape(96, n) + voltage_error for k in ("C", "F")}
                    kwh = np.full(size, value["absolute_kwh"])
                gap = (np.abs(residuals["C"]).mean(axis=(1, 2))
                       - 1.03*np.abs(residuals["F"]).mean(axis=(1, 2)))
                row["successes"] += int(np.sum(gap > problem["packet"]["threshold"]))
                row["count"] += size
                row["sum_D"] += float(gap.sum())
                row["sum_D2"] += float(gap @ gap)
                row["delivery_kwh_sum"] += float(kwh.sum())
            except Exception:
                row["failures"] += size
                raise
    for row in summaries.values():
        total = row["count"]
        row.update(probability=row["successes"]/total if total else None,
                   wilson_95=wilson(row["successes"], total),
                   mean_D=row["sum_D"]/total if total else None,
                   sd_D=float(np.sqrt(max(0., row["sum_D2"]/total-(row["sum_D"]/total)**2))) if total else None)
    return dict(role="full_independent_primitive_linear_reference_diagnostic" if full
                else "conditional_independent_voltage_validation_only", seed=seed, requested=count,
                completed=min(row["count"] for row in summaries.values()),
                seconds=time.perf_counter()-began, policies=summaries)


def optimize(problem, options, out, max_seconds=120):
    began = time.perf_counter()
    records = []
    cache = {}

    def evaluate(theta):
        key = tuple(float(v) for v in theta)
        if key not in cache:
            if time.perf_counter()-began > max_seconds:
                raise TimeoutError("Two-variable time limit")
            result = kernel(problem, theta, options, directions=True)
            if not np.isfinite(result["probability"]) or result.get("gradient") is None:
                raise ArithmeticError("Unresolved continuous probability derivative")
            cache[key] = result
            records.append(result)
            write(out / "optimization_trace.json", records)
        return cache[key]

    status = "not_started"
    success = False
    try:
        starts = [(name, evaluate(theta)) for name, theta in POLICIES.items()]
        best_name, best = max(starts, key=lambda item: item[1]["probability"])
        initial = np.asarray(best["theta"])
        feasible = [row for _, row in starts if row["probability"] >= .9]
        if not feasible:
            phase = minimize(lambda x: -evaluate(x)["probability"], initial,
                             jac=lambda x: -np.asarray(evaluate(x)["gradient"]), bounds=[(0.,1.)]*2,
                             method="SLSQP", options=dict(maxiter=15, ftol=1e-7))
            evaluate(phase.x)
            feasible = [row for row in records if row["probability"] >= .9]
            status = "phase_one: " + str(phase.message)
        if feasible:
            initial = np.asarray(min(feasible, key=lambda row:row["fee"])["theta"])
            phase = minimize(lambda x:evaluate(x)["fee"], initial,
                             jac=lambda x:np.asarray(evaluate(x)["fee_gradient"]),
                             constraints=[dict(type="ineq",fun=lambda x:evaluate(x)["probability"]-.9,
                                               jac=lambda x:np.asarray(evaluate(x)["gradient"]))],
                             bounds=[(0.,1.)]*2, method="SLSQP", options=dict(maxiter=15,ftol=1e-7))
            evaluate(phase.x)
            status = "cost_phase: " + str(phase.message)
            success = bool(phase.success)
    except Exception as exc:
        status = repr(exc)
    feasible = [row for row in records if row["probability"] >= .9]
    best = min(feasible,key=lambda row:row["fee"]) if feasible else max(records,key=lambda row:row["probability"],default=None)
    return dict(status=status, solver_success=success, qualified=bool(feasible), best=best,
                seconds=time.perf_counter()-began, evaluations=len(records),
                global_optimal=False, target_scope="conditional_voltage_noise_only")


def gradient_check(problem, options):
    """Compare analytic inversion directions against independent central values."""
    point = np.array([.5, .5])
    analytic = kernel(problem, point, options, directions=True)
    rows = []
    for step in (1e-3, 1e-4):
        for coordinate in range(2):
            left, right = point.copy(), point.copy()
            left[coordinate] -= step
            right[coordinate] += step
            lo = kernel(problem, left, options)
            hi = kernel(problem, right, options)
            estimate = (hi["probability"]-lo["probability"])/(2*step)
            expected = analytic["gradient"][coordinate]
            rows.append(dict(step=step, coordinate=coordinate,
                             finite_difference=estimate, analytic=expected,
                             absolute_difference=abs(estimate-expected),
                             lower_probability=lo["probability"], upper_probability=hi["probability"],
                             seconds=lo["seconds"]+hi["seconds"]))
    return dict(point=point.tolist(), analytic=analytic, rows=rows,
                max_absolute_difference=max(row["absolute_difference"] for row in rows),
                scope="fixed frequency nodes and normalization; no total-response probability")


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "complete.json").exists():
        print("Already complete", flush=True)
        return
    lock = args.out / "running.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        problem = build_problem(args.archive.resolve(), args.public.resolve())
        baseline = nominal(problem, POLICIES["frozen"])
        baseline_model = IndependentResiduals(baseline["a_C"], baseline["a_F"], problem["sigma"].ravel())
        problem["normalization"] = (baseline_model.sum_mean, baseline_model.sum_sd)
        request = dict(case=CASE, sources=problem["sources"], inputs=problem["inputs"],
                       policies=POLICIES, independent_voltage_innovations=True,
                       conditional_mc=dict(samples=32768,seed=CONDITIONAL_SEED),
                       full_mc=dict(samples=2048,seed=FULL_SEED,seconds=120),
                       optimization=dict(variables=2,maxiter=15,seconds=120,box=[0,1]),
                       python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                       normalization=problem["normalization"],
                       voltage_sigma_summary=dict(min=float(problem["sigma"].min()),
                                                  max=float(problem["sigma"].max()),
                                                  median=float(np.median(problem["sigma"]))))
        if (args.out/"request.json").exists() and read(args.out/"request.json")!=request:
            raise ValueError("Changed configuration/source in existing run")
        write(args.out/"request.json", request)
        if not (args.out/"source_snapshot.zip").exists():
            with zipfile.ZipFile(args.out/"source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
                for filename in problem["sources"]:
                    snapshot.write(TASK / filename, filename)
        results = read(args.out/"results.json") if (args.out/"results.json").exists() else {}
        options = dict(tolerance=1e-5, step=.05, refinements=2, max_cutoff=2048)
        if "frozen" not in results:
            results["frozen"] = kernel(problem, POLICIES["frozen"], options)
            write(args.out/"results.json", results)
        if "conditional_mc" not in results:
            results["conditional_mc"] = monte_carlo(problem)
            write(args.out/"results.json", results)
        if "full_mc" not in results:
            results["full_mc"] = monte_carlo(problem,full=True)
            write(args.out/"results.json", results)
        conditional_validation = results["conditional_mc"]["policies"]["frozen"]
        interval = conditional_validation["wilson_95"]
        computed = results["frozen"]["probability"]
        compatible = (interval[0]-.001 <= computed <= interval[1]+.001)
        gate = (results["frozen"]["seconds"] <= 10
                and results["frozen"].get("numerical_pass",False) and compatible
                and results["conditional_mc"]["completed"] == 32768)
        fixed_options = dict(tolerance=1e-5, cutoff=results["frozen"]["cutoff"],
                             step=.025, refinements=0, max_cutoff=2048)
        if "gradient_check" not in results:
            results["gradient_check"] = gradient_check(problem,fixed_options) if gate else dict(
                status="skipped_kernel_runtime_or_precision_gate",runtime_gate_seconds=10)
            write(args.out/"results.json", results)
        if "optimization" not in results:
            results["optimization"] = optimize(problem,fixed_options,args.out) if gate else dict(
                status="skipped_kernel_runtime_or_precision_gate",runtime_gate_seconds=10)
            if gate and results["optimization"].get("best") is not None:
                best = results["optimization"]["best"]
                results["optimization"]["refined_frozen_best"] = kernel(problem,best["theta"],options)
            write(args.out/"results.json",results)
        write(args.out/"complete.json",dict(status="complete",results_sha256=sha(args.out/"results.json"),
                                            conditional_only_optimizer=True,no_AC_run=True))
        print(json.dumps(plain(results),ensure_ascii=True),flush=True)
    except Exception as exc:
        write(args.out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "public", "out"):
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
