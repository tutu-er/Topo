"""Independent paired evaluation of a frozen all-node/hour paper15 quote.

This module supplies five explicit frozen contracts to the existing sampling
and original-AC evaluators. It does not implement a new identifier, device
controller, observation model, or topology score. All output paths are private.
"""
from __future__ import annotations

import argparse
import json
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
from scipy.optimize import brentq

import run as bounded
import sample_compare as sampling
import actual_compare as actual


POLICY_ORDER = ("zero", "rotation", "old95", "old90", "new")
SPECS = {
    "conditional": dict(samples=32768, seed=2026111301, batch=256),
    "independent_full": dict(samples=8192, seed=2026111302, batch=64),
}
RESPONSE_SEED = 2026111303
METER_SEED = 2026111304


def matched_rotation(device, original, target):
    """Match the new fee along a pre-existing, truth-independent rotation ray.

    Nominal fees are quadratic at fixed signs. Below the old rotation fee,
    sqrt scaling is exact. Above that fee the original causal controller is
    applied to fixed scalar multiples; one continuous bracket is solved. The
    finite ray search gives a feasible comparator, not a maximum-cost theorem.
    """
    original = np.asarray(original, float)
    initial = float(device.fees(original))
    if not np.isfinite(target) or target < 0:
        raise ValueError("Invalid target contract fee")
    records = []

    def point(scale):
        raw = scale * original
        value = device.project(raw)
        fee = float(device.fees(value))
        audit = device.audit(value)
        if not audit["passed"]:
            raise ArithmeticError("Original controller returned invalid rotation")
        records.append(dict(scale=float(scale), fee=fee, passed=True))
        return value, fee

    if target == 0:
        value = np.zeros_like(original)
        return value, dict(matched=True, target_fee=0., fee=0., original_fee=initial,
                           scale=0., method="zero", ray_search=records)
    if initial <= 0:
        raise ValueError("Existing rotation has no adjustable resource")
    simple_scale = float(np.sqrt(target / initial))
    value, fee = point(simple_scale)
    if abs(fee - target) <= 1e-7:
        return value, dict(matched=True, target_fee=target, fee=fee, original_fee=initial,
                           scale=simple_scale, method="quadratic fee ray with original controller",
                           ray_search=records)

    # This search is locked before opening actual future curves. No performance
    # score is queried: it only matches an existing simple schedule's fee.
    tested = [(0., np.zeros_like(original), 0.)]
    grid = sorted(set([1., simple_scale, 1.25, 1.5, 2., 3., 4., 6., 8., 12., 16., 24., 32., 64.]))
    bracket = None
    for scale in grid:
        candidate, amount = point(scale)
        tested.append((scale, candidate, amount))
        if (tested[-2][2] - target) * (amount - target) <= 0:
            bracket = tested[-2][0], scale
            break
    if bracket is not None:
        def residual(scale):
            return point(scale)[1] - target
        scale = float(brentq(residual, *bracket, xtol=1e-12, rtol=1e-12))
        value, fee = point(scale)
        matched = abs(fee - target) <= 1e-7
        if not matched:
            raise ArithmeticError("Continuous rotation fee match did not close")
        return value, dict(matched=matched, target_fee=target, fee=fee, original_fee=initial,
                           scale=scale, method="bounded scalar ray and original controller",
                           ray_search=records)
    scale, value, fee = max(tested, key=lambda row: row[2])
    return value, dict(matched=False, target_fee=target, fee=fee, original_fee=initial,
                       scale=scale, method="bounded ray comparator; equal fee unattained",
                       largest_tested_ray_fee=fee, ray_search=records)


def load_five_policies(problem, archive, quote_path, *, allow_unqualified=False):
    """Read only historical fits, issued forecasts, devices, and frozen quotes."""
    quote_path = Path(quote_path).resolve()
    folder = quote_path.parent
    freezing = bounded.read(folder / "price_freeze.json")
    complete = bounded.read(folder / "complete.json")
    if bounded.checked(freezing["plan"]) != quote_path:
        raise ValueError("Requested quote differs from optimization freeze")
    if bounded.checked(complete["price_freeze"]) != folder / "price_freeze.json":
        raise ValueError("Optimization completion does not reference quote freeze")
    request_path = bounded.checked(freezing["request"])
    quote = bounded.unpickle(quote_path)
    if quote["source_request_sha256"] != bounded.sha(request_path):
        raise ValueError("Frozen quote request mismatch")
    if freezing.get("true_future_used") is not False:
        raise ValueError("Quote information boundary is not declared")
    if bool(quote["qualified"]) != bool(freezing["qualified"]):
        raise ValueError("Quote qualification changed")
    if not quote["qualified"] and not allow_unqualified:
        raise ValueError("New quote did not reach planning target; explicit diagnostic flag required")

    old, old_refs = sampling.load_policies(problem, archive)
    device = problem["device"]
    x = np.asarray(quote["x"], float)
    if x.shape != (device.R, 24):
        raise ValueError("New quote shape differs from complete hourly resource domain")
    value = device.nominal(x)
    if not np.allclose(value, quote["mean_u"], atol=1e-10, rtol=0):
        raise ArithmeticError("New quote does not replay the original controller")
    audit = device.audit(value)
    fee = float(device.fees(value))
    if not audit["passed"] or abs(fee - float(quote["fee"])) > 1e-8:
        raise ArithmeticError("New quote resource or fee replay failed")
    rotation, rotation_audit = matched_rotation(device, old["rotation"]["mean_u"], fee)
    policies = {name:dict(old[name]) for name in ("zero", "old95")}
    policies["old90"] = dict(old["current90"])
    policies["rotation"] = dict(mean_u=rotation, fee=float(device.fees(rotation)),
                                planning_probability=None,
                                planning_probability_scope="not selected by probability or future outcome",
                                source_policy="same existing simple rotation scaled to new quote fee")
    policies["new"] = dict(mean_u=value, fee=fee, planning_probability=quote["probability"],
                           planning_probability_scope=quote["target_scope"],
                           qualified=bool(quote["qualified"]), nominal_audit=audit,
                           source_policy="new all-resource/hour frozen quote")
    references = dict(old=old_refs, new_quote=bounded.reference(quote_path),
                      price_freeze=bounded.reference(folder/"price_freeze.json"),
                      optimizer_complete=bounded.reference(folder/"complete.json"),
                      optimizer_request=bounded.reference(request_path),
                      rotation_match=rotation_audit)
    return policies, references


def check_default_interfaces(problem, archive):
    """Tiny historical-only replay proving optional arguments preserve defaults."""
    old, _ = sampling.load_policies(problem, archive)
    means = sampling.condition_means(problem, old)
    checks = []
    for mode in ("conditional", "independent_full"):
        default, audit0 = sampling.sample_batch(problem, old, means, mode, 0, 2)
        explicit, audit1 = sampling.sample_batch(problem, old, means, mode, 0, 2,
                                                policy_order=sampling.POLICY_ORDER,
                                                mode_spec=sampling.MODES[mode])
        good = all(np.array_equal(default[k], explicit[k]) for k in default)
        good = good and audit0 == audit1
        checks.append(dict(mode=mode, size=2, bitwise_equal=good))
    if not all(row["bitwise_equal"] for row in checks):
        raise ArithmeticError("Optional evaluator interface changed default calculations")
    return dict(passed=True, checks=checks, actual_physics_and_selection_reimplemented=False)


def main(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    try:
        problem = bounded.build_problem(args.archive.resolve(), args.public.resolve())
        policies, references = load_five_policies(problem, args.archive.resolve(), args.quote,
                                                 allow_unqualified=args.allow_unqualified)
        sources = dict(problem["sources"])
        for filename in (Path(__file__), bounded.HERE/"sample_compare.py", bounded.HERE/"actual_compare.py",
                         bounded.HERE/"gauss_frequency.py"):
            sources[str(filename.relative_to(bounded.TASK))] = bounded.sha(filename)
        request = dict(case=bounded.CASE, protocol=bounded.reference(args.protocol.resolve()),
                       historical_forecast_inputs=problem["inputs"], frozen_quotes=references,
                       policy_order=POLICY_ORDER, specs=SPECS, sources=sources,
                       seconds_per_sampling_model=180, AC_paths=32, AC_seconds=600,
                       response_seed=RESPONSE_SEED, meter_seed=METER_SEED,
                       allow_unqualified_diagnostic=bool(args.allow_unqualified),
                       python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                       policy_information="complete history, issued forecast, resource data; no actual future",
                       sampling_information="new independent execution paths after all five contracts frozen",
                       identifier="existing frozen-RX historical RNJ+WZZT pool and original weighted3% selector",
                       seed_rule="sample model SeedSequence([seed,batch_index]); AC per-path child streams",
                       no_parameter_refit=True, no_new_W_search=True)
        request_path = out/"request.json"
        if request_path.exists() and bounded.read(request_path) != bounded.plain(request):
            raise ValueError("Changed sources, plans or evaluation configuration in existing stage")
        bounded.write(request_path, request)
        if (out/"complete.json").exists():
            print("Already complete", flush=True)
            return
        if not (out/"source_snapshot.zip").exists():
            with zipfile.ZipFile(out/"source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as snapshot:
                for filename in sources:
                    snapshot.write(bounded.TASK/filename, filename)
        frozen_path = out/"contracts_private.pkl"
        if not frozen_path.exists():
            actual.dump(frozen_path, policies)
        saved = bounded.unpickle(frozen_path)
        if any(not np.array_equal(saved[name]["mean_u"], policies[name]["mean_u"]) for name in POLICY_ORDER):
            raise ValueError("Frozen evaluation policy bank changed")
        bounded.write(out/"policy_freeze.json", dict(contracts=bounded.reference(frozen_path),
                      sources=references, test_truth_used=False,
                      fees={name:policies[name]["fee"] for name in POLICY_ORDER}))
        results_path = out/"results.json"
        results = bounded.read(results_path) if results_path.exists() else {}
        results["interface_regression"] = check_default_interfaces(problem, args.archive.resolve())
        results["quotes"] = {name:{k:v for k,v in row.items() if k!="mean_u"} for name,row in policies.items()}
        if "conditional_predictions" not in results:
            results["conditional_predictions"] = sampling.conditional_predictions(
                problem, policies, policy_order=POLICY_ORDER, normalization_policy="old90")
            bounded.write(results_path, results)
        means = sampling.condition_means(problem, policies)
        for mode, spec in SPECS.items():
            if mode not in results:
                results[mode] = sampling.run_sampling(problem, policies, means, mode, out,
                    seconds=180, policy_order=POLICY_ORDER, mode_spec=spec, main_policy="new")
                bounded.write(results_path, results)

        # Explicit callback, not a patched module global. The exact same bank
        # is supplied to the existing AC engine before it opens the sealed day.
        def frozen_loader(rebuilt_problem, archive):
            if rebuilt_problem["inputs"] != problem["inputs"]:
                raise ValueError("AC engine rebuilt a different historical/forecast problem")
            return policies, references

        ac_out = out/"actual"
        ac_args = argparse.Namespace(archive=args.archive, public=args.public, sealed=args.sealed,
                                    out=ac_out, paths=32, seconds=600.)
        actual.main(ac_args, policy_loader=frozen_loader, policy_order=POLICY_ORDER,
                    response_seed=RESPONSE_SEED, meter_seed=METER_SEED,
                    extra_sources=(Path(__file__),),
                    comparison_pairs=[("new", name) for name in POLICY_ORDER if name!="new"],
                    stage_metadata=dict(protocol=bounded.reference(args.protocol.resolve()),
                                        outer_freeze=bounded.reference(out/"policy_freeze.json"),
                                        allow_unqualified_diagnostic=bool(args.allow_unqualified)),
                    has_new_quote=True)
        results["actual"] = bounded.read(ac_out/"summary.json")
        results["seconds"] = time.perf_counter()-started
        results["new_quote_qualified"] = policies["new"]["qualified"]
        results["rotation_matching"] = references["rotation_match"]
        bounded.write(results_path, results)
        model_done = all(results[k]["completed"] == SPECS[k]["samples"] for k in SPECS)
        ac_done = results["actual"]["observed_policy_records"] == 32*len(POLICY_ORDER)
        bounded.write(out/"complete.json", dict(status="complete" if model_done and ac_done else "bounded_partial",
                      results=bounded.reference(results_path), request=bounded.reference(request_path),
                      new_quote_qualified=policies["new"]["qualified"],
                      model_completion={k:results[k]["completed"] for k in SPECS},
                      AC_records=results["actual"]["observed_policy_records"]))
        print(json.dumps(bounded.plain(results), ensure_ascii=True), flush=True)
    except Exception as exc:
        bounded.write(out/"failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "public", "sealed", "quote", "protocol", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--allow-unqualified", action="store_true",
                        help="Evaluate a bounded diagnostic explicitly; never labels it a 90pct contract")
    main(parser.parse_args())
