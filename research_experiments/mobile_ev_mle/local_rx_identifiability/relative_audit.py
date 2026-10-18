"""Read-only statistical replay of the exploratory relative bootstrap.

No new meter samples are used for inference and no fit is run.  Seed replay
only verifies saved simulated-meter digests.  Truth labels are read in a final
evaluation pass, after every statistic and decision has been recomputed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


NAMES = ("star", "pair01", "pair02", "pair12")


def _read(path):
    p = Path(path)
    body = p.read_bytes()
    return json.loads(gzip.decompress(body) if p.suffix == ".gz" else body)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(spec):
    return f"{spec['network']}_{spec['regime']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def _expected():
    return [(dict(network=n, regime=r, seed=31001, N=96, boundary="K"), c)
            for n in ("paper15", "paper11") for r in ("ordinary", "correlated") for c in NAMES]


def _check(issues, passed, code, path, detail):
    if not passed:
        issues.append(dict(code=code, path=str(path), detail=str(detail)))


def _close(left, right, tol=1e-9):
    try:
        left, right = float(left), float(right)
        return bool(np.isfinite(left) and np.isfinite(right) and abs(left-right) <= tol*(1+abs(left)))
    except (TypeError, ValueError):
        return False


def recompute_relative(fits, target):
    """Independent implementation; no relative_bootstrap helpers imported."""
    losses, invalid = {}, []
    for name in NAMES:
        value = fits.get(name, {})
        try:
            loss = float(value.get("loss", np.nan))
            valid = value.get("qualified") is True and np.isfinite(loss) and loss >= 0
        except (TypeError, ValueError):
            valid = False
        if valid:
            losses[name] = loss
        else:
            invalid.append(name)
    if invalid:
        return dict(valid=False, statistic=None, invalid=invalid, losses=losses)
    return dict(valid=True, statistic=losses[target]-min(losses.values()), invalid=[], losses=losses)


def recompute_tail(observed, records):
    """Every incomplete or nonstationary statistic contributes to upper tail."""
    failed = tail = 0
    for record in records:
        valid = record["valid"]
        if not valid:
            failed += 1
            tail += 1
        else:
            tail += int(record["statistic"] >= observed)
    return dict(pvalue=(1+tail)/(1+len(records)), replicates=len(records), failed=failed,
                valid=len(records)-failed, upper_tail=tail)


def _own_usable(record):
    try:
        value = float(record.get("sse", np.nan))
        return record.get("success") is True and record.get("stationary") is True and np.isfinite(value) and value >= 0
    except (TypeError, ValueError):
        return False


def _digests(sealed, target, repeats):
    spec = sealed["spec"]
    rng = np.random.default_rng(np.random.SeedSequence([
        spec["seed"], 170, NAMES.index(target), 11 if spec["network"] == "paper11" else 15,
        0 if spec["regime"] == "ordinary" else 1]))
    center = sealed["fits"][target]
    for _ in range(repeats):
        digest = hashlib.sha256()
        for channel in ("p", "q", "v"):
            values = np.asarray(center[channel], float)
            noise = rng.normal(size=values.shape)
            sample = np.asarray(values+noise*np.asarray(sealed["sigma"][channel]), dtype="<f8", order="C")
            digest.update(channel.encode("ascii"))
            digest.update(np.asarray(sample.shape, dtype="<i8").tobytes())
            digest.update(sample.tobytes())
        yield digest.hexdigest()


def audit_relative(directory, *, destination=None):
    started = perf_counter()
    root = Path(directory)
    issues, rows, missing = [], [], []
    artifacts, sources = {}, {}
    for spec, target in _expected():
        key = _key(spec)
        path = root/"relative_bootstrap"/f"{key}_{target}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        row = dict(spec=spec, candidate=target, file=path.name)
        rows.append(row)
        try:
            result = _read(path)
            artifacts[str(path.resolve())] = _sha(path)
            _check(issues, result.get("spec") == spec and result.get("candidate") == target,
                   "relative_spec", path, result.get("spec"))
            _check(issues, result.get("truth_read") is False, "truth_read_flag", path, result.get("truth_read"))
            row["status"] = result.get("status")
            for label, source in (("inference", root/"inference"/(key+".json.gz")),
                                  ("own_bootstrap", root/"bootstrap"/f"{key}_{target}.json"),
                                  ("original_source_manifest", root/"source_manifest.json")):
                if label == "original_source_manifest" and not source.exists():
                    continue
                digest = _sha(source)
                artifacts[str(source.resolve())] = digest
                _check(issues, digest == result.get("input_sha256", {}).get(label),
                       "input_hash_changed", path, label)
            for source, digest in result.get("source_sha256", {}).items():
                current = _sha(source)
                sources[source] = current
                _check(issues, current == digest, "source_hash_changed", path, source)
            sealed = _read(root/"inference"/(key+".json.gz"))
            original = _read(root/"bootstrap"/f"{key}_{target}.json")
            observed = recompute_relative(sealed["fits"], target)
            row["observed"] = observed
            saved_observed = result.get("observed", {})
            _check(issues, observed["valid"] == saved_observed.get("valid"), "observed_valid_flag", path, observed)
            if observed["valid"]:
                _check(issues, _close(observed["statistic"], saved_observed.get("statistic")),
                       "observed_statistic", path, observed)
            else:
                _check(issues, not result.get("reject"), "unavailable_observed_rejected", path, observed)
            if result.get("status") != "completed":
                row.update(failure_preserved=True, reason=result.get("reason", result.get("error")))
                continue
            saved_records, original_records = result.get("records", []), original.get("records", [])
            _check(issues, len(saved_records) == len(original_records) == 39,
                   "replicate_count", path, (len(saved_records), len(original_records)))
            checked = []
            own_failures = competitor_failures = extra_calls = extra_ac_calls = skipped = 0
            extra_fit_seconds = 0.
            for b, (record, own, digest) in enumerate(zip(saved_records, original_records,
                                                       _digests(sealed, target, len(saved_records)))):
                loc = f"{path.name}.replicate{b}"
                _check(issues, record.get("replicate") == b, "replicate_index", loc, record.get("replicate"))
                _check(issues, digest == record.get("simulated_measurements_sha256"),
                       "simulation_noise_digest", loc, digest)
                fits = record.get("fits", {})
                own_valid = _own_usable(own)
                own_failures += int(not own_valid)
                _check(issues, fits.get(target, {}).get("qualified") == own_valid,
                       "reused_own_qualification", loc, own)
                if own_valid:
                    _check(issues, _close(fits.get(target, {}).get("loss"), own["sse"]),
                           "reused_own_loss", loc, own)
                for name in NAMES:
                    if name == target:
                        continue
                    value = fits.get(name)
                    if value is not None:
                        extra_calls += 1
                        extra_ac_calls += int(value.get("ac_calls", 0))
                        extra_fit_seconds += float(value.get("seconds", 0.) or 0.)
                        competitor_failures += int(value.get("qualified") is not True)
                expected_skipped = 0 if own_valid else 3
                skipped += expected_skipped
                _check(issues, record.get("skipped_competitor_fit_count") == expected_skipped,
                       "skipped_fit_count", loc, expected_skipped)
                relative = recompute_relative(fits, target)
                checked.append(relative)
                _check(issues, record.get("valid") == relative["valid"] and
                       record.get("success") == relative["valid"] and
                       record.get("stationary") == relative["valid"], "simulated_valid_flag", loc, relative)
                if relative["valid"]:
                    _check(issues, _close(record.get("sse"), relative["statistic"]) and
                           _close(record.get("relative", {}).get("statistic"), relative["statistic"]),
                           "simulated_statistic", loc, relative)
                else:
                    _check(issues, record.get("sse") is None, "failed_statistic_not_infinite_tail", loc, record.get("sse"))
            tail = recompute_tail(observed["statistic"], checked)
            saved_decision = result.get("decision", {})
            _check(issues, _close(tail["pvalue"], saved_decision.get("pvalue"), tol=1e-14),
                   "relative_pvalue", path, tail)
            _check(issues, tail["failed"] == saved_decision.get("failed_or_nonstationary_replicates"),
                   "conservative_tail_failure_count", path, tail)
            _check(issues, tail["upper_tail"] == saved_decision.get("upper_tail_count_including_failures"),
                   "upper_tail_count", path, tail)
            reject = tail["pvalue"] <= float(result["alpha"])
            _check(issues, reject == result.get("reject"), "relative_decision", path, tail)
            _check(issues, extra_calls == result.get("extra_fit_calls"), "extra_fit_calls", path, extra_calls)
            _check(issues, extra_ac_calls == result.get("extra_ac_calls"), "extra_ac_calls", path, extra_ac_calls)
            _check(issues, result.get("reused_own_fit_count") == len(original_records),
                   "reused_fit_count", path, result.get("reused_own_fit_count"))
            row.update(decision=tail, reject=reject,
                absolute_pvalue=original.get("decision", {}).get("pvalue"),
                absolute_reject=original.get("reject", False),
                own_failed_replicates=own_failures, competitor_unqualified_fits=competitor_failures,
                skipped_competitor_fits=skipped, extra_fit_calls=extra_calls, extra_ac_calls=extra_ac_calls,
                extra_fit_seconds_sum=extra_fit_seconds,
                extra_slot_seconds=float(result.get("extra_elapsed_seconds", 0.)),
                reused_own_bootstrap_seconds=float(original.get("seconds", 0.)))
        except Exception as exc:
            issues.append(dict(code="relative_audit_exception", path=path.name,
                               detail=f"{type(exc).__name__}: {exc}"))
    # Only now read the evaluation labels. They did not enter any statistic,
    # simulated response, threshold, or recomputed p-value above.
    conditions = {}
    for row in rows:
        key = _key(row["spec"])
        try:
            true_name = _read(root/"truth"/(key+".json.gz"))["candidate_name"]
            row["evaluation_true_candidate"] = row["candidate"] == true_name
            group = conditions.setdefault(key, dict(spec=row["spec"], truth_label=true_name,
                retained=[], rejected=[], unavailable=[]))
            if "reject" not in row:
                group["unavailable"].append(row["candidate"])
            elif row["reject"]:
                group["rejected"].append(row["candidate"])
            else:
                group["retained"].append(row["candidate"])
        except Exception as exc:
            issues.append(dict(code="evaluation_truth_missing", path=key,
                               detail=f"{type(exc).__name__}: {exc}"))
    completed_rows = [r for r in rows if r.get("status") == "completed" and "reject" in r]
    true_rows = [r for r in completed_rows if r.get("evaluation_true_candidate")]
    wrong_rows = [r for r in completed_rows if r.get("evaluation_true_candidate") is False]
    wall = None
    summary_path = root/"relative_bootstrap_summary.json"
    if summary_path.exists():
        saved_summary = _read(summary_path)
        wall = saved_summary.get("extra_elapsed_seconds")
        _check(issues, saved_summary.get("completed_slots") == len(completed_rows),
               "summary_completed_slots", summary_path, len(completed_rows))
    result = dict(schema="independent_relative_bootstrap_audit_v1", directory=str(root.resolve()),
        complete=not missing and len(rows) == 16,
        audit_pass=not missing and len(rows) == 16 and not issues,
        expected_candidate_slots=16, saved_candidate_slots=len(rows),
        completed_candidate_slots=len(completed_rows), missing_files=missing,
        original_history_conditions=4,
        original_history_randomness_families=2,
        history_pairing="ordinary/correlated share same network/seed RNG streams; four conditions are paired, not four independent histories",
        true_candidate_denominator=4, true_candidate_evaluable=len(true_rows),
        false_reject=sum(r["reject"] for r in true_rows),
        wrong_candidate_denominator=12, wrong_candidate_evaluable=len(wrong_rows),
        wrong_reject=sum(r["reject"] for r in wrong_rows),
        absolute_false_reject=sum(bool(r.get("absolute_reject")) for r in true_rows),
        absolute_wrong_reject=sum(bool(r.get("absolute_reject")) for r in wrong_rows),
        relative_failed_statistic_repeats=sum(r.get("decision", {}).get("failed", 0) for r in rows),
        configured_simulated_datasets=16*39,
        audited_simulated_datasets=sum(r.get("decision", {}).get("replicates", 0) for r in rows),
        reused_own_failed_repeats=sum(r.get("own_failed_replicates", 0) for r in rows),
        competitor_unqualified_fits=sum(r.get("competitor_unqualified_fits", 0) for r in rows),
        skipped_competitor_fits=sum(r.get("skipped_competitor_fits", 0) for r in rows),
        extra_fit_calls=sum(r.get("extra_fit_calls", 0) for r in rows),
        extra_ac_calls=sum(r.get("extra_ac_calls", 0) for r in rows),
        extra_fit_seconds_sum=sum(r.get("extra_fit_seconds_sum", 0.) for r in rows),
        extra_slot_seconds_sum=sum(r.get("extra_slot_seconds", 0.) for r in rows),
        extra_batch_wall_seconds=wall,
        reused_own_bootstrap_seconds_sum=sum(r.get("reused_own_bootstrap_seconds", 0.) for r in rows),
        conditions=conditions, rows=rows, issues=issues,
        artifact_sha256=artifacts, source_sha256=sources,
        auditor_sha256=_sha(__file__), environment=dict(python=sys.version, numpy=np.__version__),
        newly_fitted_states=0, audit_seconds=perf_counter()-started,
        scope="exploratory candidate plug-in local bootstrap; no uniform frequentist or posterior claim")
    target = Path(destination) if destination else root/"relative_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args()
    result = audit_relative(args.directory, destination=args.destination)
    print(json.dumps({k:result[k] for k in ("complete", "audit_pass", "saved_candidate_slots",
        "false_reject", "wrong_reject", "relative_failed_statistic_repeats", "extra_fit_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
