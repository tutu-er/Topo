"""Exploratory candidate-relative plug-in bootstrap on sealed observations.

The statistic T_k = S_k - min_j S_j cancels fitted-noise contributions common
to every candidate; candidate-specific noise fitting still remains.
Its calibration reuses each candidate's existing 39 same-candidate bootstrap
fits and adds the three competing fits on exactly the same simulated meters.
No truth files or true labels are read.  Local optimization and plug-in
generation do not provide a uniform finite-sample probability guarantee.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

import numpy as np
import scipy

from .cases import candidate_bank
from .fitting import fit
from .rejection import bootstrap_pvalue


REPEATS = 39
CANDIDATE_NAMES = ("star", "pair01", "pair02", "pair12")


def _read(path):
    p = Path(path)
    body = p.read_bytes()
    return json.loads(gzip.decompress(body) if p.suffix == ".gz" else body)


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def _write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def finite(item):
        if isinstance(item, dict):
            return {k:finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item
    p.write_text(json.dumps(finite(value), default=_json, ensure_ascii=False, allow_nan=False,
                           separators=(",", ":")), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(spec):
    return f"{spec['network']}_{spec['regime']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def tasks():
    return [(dict(network=network, regime=regime, seed=31001, N=96, boundary="K"), candidate)
            for network in ("paper15", "paper11") for regime in ("ordinary", "correlated")
            for candidate in CANDIDATE_NAMES]


def replay_measurements(sealed, candidate, repeats=REPEATS):
    """Replay experiment.bootstrap_one's seed and p/q/v draw order exactly."""
    spec = sealed["spec"]
    index = CANDIDATE_NAMES.index(candidate)
    fitted = sealed["fits"][candidate]
    sigma = {key:np.asarray(sealed["sigma"][key], float) for key in ("p", "q", "v")}
    rng = np.random.default_rng(np.random.SeedSequence([
        spec["seed"], 170, index, 11 if spec["network"] == "paper11" else 15,
        0 if spec["regime"] == "ordinary" else 1]))
    for b in range(repeats):
        boot = {key:np.asarray(fitted[key], float)
                + rng.normal(size=np.shape(fitted[key]))*sigma[key]
                for key in ("p", "q", "v")}
        yield b, boot


def measurement_sha256(boot):
    """Fixed-order binary digest of the replayed floating-point meters."""
    digest = hashlib.sha256()
    for key in ("p", "q", "v"):
        array = np.asarray(boot[key], dtype="<f8", order="C")
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _usable_own(record):
    try:
        loss = float(record.get("sse", np.nan))
        return (record.get("success") is True and record.get("stationary") is True
                and np.isfinite(loss) and loss >= 0)
    except (TypeError, ValueError, AttributeError):
        return False


def relative_statistic(fits, candidate):
    """Only qualified local values may form a reported relative statistic."""
    invalid = []
    losses = {}
    for name in CANDIDATE_NAMES:
        item = fits.get(name, {})
        try:
            value = float(item.get("loss", np.nan))
            usable = item.get("qualified") is True and np.isfinite(value) and value >= 0
        except (TypeError, ValueError, AttributeError):
            usable = False
        if not usable:
            invalid.append(name)
        else:
            losses[name] = value
    if invalid:
        return dict(valid=False, statistic=None, invalid_candidates=invalid, losses=losses)
    minimum = min(losses.values())
    return dict(valid=True, statistic=max(0., losses[candidate]-minimum),
                invalid_candidates=[], losses=losses,
                minimum_loss=minimum,
                minimizing_candidates=[name for name in CANDIDATE_NAMES if losses[name] == minimum])


def _fit_summary(value):
    fields = ("status", "qualified", "loss", "projected_gradient", "seconds", "ac_calls", "r", "x")
    result = {key:value[key] for key in fields if key in value}
    # Keep record serialization valid if a numerical failure returned NaN.
    for key in ("loss", "projected_gradient", "seconds"):
        if key in result and result[key] is not None and not np.isfinite(result[key]):
            result[key] = None
    return result


def run_one(directory, spec, candidate, *, fit_function=None):
    """Add the competing fits; leave all sealed input artifacts unchanged."""
    started = perf_counter()
    directory = Path(directory)
    key = _key(spec)
    destination = directory/"relative_bootstrap"/f"{key}_{candidate}.json"
    value = dict(spec=spec, candidate=candidate, statistic="T_k=S_k-min_j_S_j",
                 method="exploratory_candidate_relative_plugin_bootstrap_local_fits",
                 coverage="approximate_not_uniform_frequentist_guarantee",
                 truth_read=False, alpha=.05, reject=False,
                 reused_own_fit_count=0, extra_fit_calls=0, extra_ac_calls=0,
                 records=[], source_sha256={})
    for source in (Path(__file__), Path(__file__).with_name("fitting.py"),
                   Path(__file__).with_name("cases.py"), Path(__file__).with_name("rejection.py"),
                   Path(__file__).with_name("experiment.py"),
                   Path(__file__).parents[1]/"likelihood.py",
                   Path(__file__).parents[2]/"mobile_ev_probing"/"ac_general_v9.py"):
        value["source_sha256"][str(source.resolve())] = _sha(source)
    try:
        if candidate not in CANDIDATE_NAMES or (spec, candidate) not in tasks():
            raise ValueError("only the predeclared four N96/K/seed31001 conditions are supported")
        inference_path = directory/"inference"/f"{key}.json.gz"
        bootstrap_path = directory/"bootstrap"/f"{key}_{candidate}.json"
        sealed, original = _read(inference_path), _read(bootstrap_path)
        value["input_sha256"] = dict(inference=_sha(inference_path), own_bootstrap=_sha(bootstrap_path))
        if (directory/"source_manifest.json").exists():
            value["input_sha256"]["original_source_manifest"] = _sha(directory/"source_manifest.json")
        if sealed.get("spec") != spec or original.get("spec") != spec or original.get("candidate") != candidate:
            raise ValueError("sealed inference/bootstrap specification mismatch")
        if sealed.get("truth_used") is not False:
            raise ValueError("sealed inference must explicitly declare truth_used=False")
        observed = relative_statistic(sealed["fits"], candidate)
        value["observed"] = observed
        value["fit_rule"] = dict(multistart=2, max_iter=160,
            seed="spec.seed + candidate index in [star,pair01,pair02,pair12]",
            initialization="same observed-only initialization as experiment.bootstrap_one")
        if not observed["valid"]:
            value.update(status="observed_statistic_unavailable", decision=None,
                         reason="at least one of four observed fits is unqualified; no rejection")
        elif original.get("status") != "completed":
            value.update(status="own_bootstrap_unavailable", decision=None,
                         reason=original.get("status", "unknown"))
        else:
            records = original.get("records", [])
            if len(records) != REPEATS:
                raise ValueError(f"expected {REPEATS} own records, found {len(records)}")
            own_observed = float(original["observed_sse"])
            if not np.isclose(own_observed, observed["losses"][candidate], atol=1e-8, rtol=1e-10):
                raise ValueError("own bootstrap observed loss does not match sealed inference")
            bank = {c.name:c for c in candidate_bank()}
            sigma = {key:np.asarray(sealed["sigma"][key], float) for key in ("p", "q", "v")}
            known = np.asarray(sealed["root_known"], float)
            solver = fit if fit_function is None else fit_function
            for b, boot in replay_measurements(sealed, candidate):
                repeat_started = perf_counter()
                own = records[b]
                own_valid = _usable_own(own)
                fits = {candidate:dict(loss=float(own["sse"]) if own_valid else None,
                                      qualified=own_valid, source="reused_same_candidate_bootstrap",
                                      seconds=own.get("seconds"), projected_gradient=own.get("projected_gradient"))}
                value["reused_own_fit_count"] += 1
                # Once the reused own fit fails, this repeat already contributes
                # +infinity to the conservative tail. Avoid three wasted solves.
                if own_valid:
                    for j, name in enumerate(CANDIDATE_NAMES):
                        if name == candidate:
                            continue
                        try:
                            fitted = solver(bank[name], boot, sigma, sealed["domain"], root_known=known,
                                            multistart=2, max_iter=160, seed=spec["seed"]+j)
                        except Exception as exc:
                            fitted = dict(status="exception", qualified=False, loss=None,
                                          error=f"{type(exc).__name__}: {exc}")
                        fits[name] = _fit_summary(fitted)
                        if "error" in fitted:
                            fits[name]["error"] = fitted["error"]
                        value["extra_fit_calls"] += 1
                        value["extra_ac_calls"] += int(fitted.get("ac_calls", 0))
                relative = relative_statistic(fits, candidate)
                good = relative["valid"]
                value["records"].append(dict(
                    replicate=b, simulated_measurements_sha256=measurement_sha256(boot),
                    sse=relative["statistic"] if good else None,
                    success=good, stationary=good, valid=good,
                    invalid_statistic_interpretation=None if good else "+infinity; counts in upper tail",
                    relative=relative, fits=fits,
                    skipped_competitor_fit_count=0 if own_valid else 3,
                    extra_seconds=perf_counter()-repeat_started))
            decision = bootstrap_pvalue(observed["statistic"], value["records"])
            value.update(status="completed", decision=decision,
                         reject=bool(decision["pvalue"] <= value["alpha"]),
                         original_absolute_bootstrap_pvalue=original.get("decision", {}).get("pvalue"),
                         own_bootstrap_recorded_seconds=original.get("seconds"))
    except Exception as exc:
        value.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc(), decision=None, reject=False)
    value["extra_elapsed_seconds"] = perf_counter()-started
    _write(destination, value)
    return {k:v for k, v in value.items() if k not in ("records", "traceback")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    started = perf_counter()
    summary = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, args.directory, spec, candidate) for spec, candidate in tasks()]
        for future in as_completed(futures):
            item = future.result()
            summary.append(item)
            print(json.dumps({key:item.get(key) for key in ("spec", "candidate", "status", "decision",
                              "extra_fit_calls", "extra_elapsed_seconds")}, ensure_ascii=False), flush=True)
    summary.sort(key=lambda item:(_key(item["spec"]), CANDIDATE_NAMES.index(item["candidate"])))
    report = dict(schema="exploratory_relative_bootstrap_v1", configured_slots=16,
        results=summary, truth_read=False, extra_elapsed_seconds=perf_counter()-started,
        extra_fit_calls=sum(x["extra_fit_calls"] for x in summary),
        extra_ac_calls=sum(x["extra_ac_calls"] for x in summary),
        completed_slots=sum(x["status"] == "completed" for x in summary),
        failed_slots=sum(x["status"] == "failed" for x in summary),
        original_batch_modified=False,
        interpretation="candidate-relative plug-in local bootstrap; exploratory; no uniform coverage claim",
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         numpy=np.__version__, scipy=scipy.__version__))
    _write(Path(args.directory)/"relative_bootstrap_summary.json", report)


if __name__ == "__main__":
    main()
