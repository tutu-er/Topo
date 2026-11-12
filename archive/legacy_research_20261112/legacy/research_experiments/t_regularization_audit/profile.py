"""Profile the frozen T domain by selected binary cardinality.

This is an oracle diagnostic, not a retuned search algorithm.  It reopens six
already documented failures and solves min training MAE for each possible
number of optional selected connections.  No new positive weight floor is
introduced: selected zero-weight candidates may pad an exact cardinality.
Protected supports consume a constant count, even if their weight is zero.

Run this file directly; optimizer imports are forced to the immutable source
snapshot.  Existing results and production defaults are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def lambda_envelope(profiles, true_loss, true_q, *, bound_key, tol=1e-10):
    """A necessary (incumbent) or sufficient (bound) lambda interval.

    For q>true_q: lambda >= (L_true-phi(q))/(q-true_q).
    For q<true_q: lambda <= (phi(q)-L_true)/(true_q-q).
    Equal-cardinality competitors are unaffected by a count penalty.
    """
    lower, upper, lower_at, upper_at = 0., None, None, None
    same_count_margin = None
    for row in profiles:
        value = row.get(bound_key)
        if value is None:
            continue
        q = row["selected_candidates"]
        if q > true_q:
            limit = (true_loss - value) / (q - true_q)
            if limit > lower:
                lower, lower_at = limit, q
        elif q < true_q:
            limit = (value - true_loss) / (true_q - q)
            if upper is None or limit < upper:
                upper, upper_at = limit, q
        else:
            same_count_margin = value - true_loss
    return dict(lower=lower, upper=upper, lower_at_q=lower_at, upper_at_q=upper_at,
                same_count_margin=same_count_margin,
                same_count_strictly_better=same_count_margin is not None and same_count_margin < -tol,
                interval_empty=upper is not None and lower > upper + tol,
                tolerance=tol)


def run_job(args, job):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare, solve, reconstruction_start, audit_vector, loss
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.local_rebuild.model import canonical, compatible
    from research_experiments.joint_five.model import build_joint
    from research_experiments.full_ablation.analyze import independent_profile, load_split
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth

    started = perf_counter()
    old, out = Path(args.snapshot).resolve(), Path(args.output).resolve()
    target = out / "jobs" / job / "profile.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    protocol, catalog = read(old / "protocol.json"), read(old / "catalog.json")
    native_source = old / "source"
    # Every frozen optimizer source is checked before numerical execution.
    source_checks = {name: sha(native_source / name) == digest for name, digest in protocol["source_hashes"].items()}
    assert all(source_checks.values()), "Frozen source changed"
    for module in (prepare, build_joint, solve, FloorBank):
        import inspect
        assert Path(inspect.getfile(module)).resolve().is_relative_to(native_source), inspect.getfile(module)
    meta = catalog[job]
    result_path = old / "jobs" / job / "R1T1E1/result.json"
    result = read(result_path)
    common_path = old / "common" / job / "common.json"
    common = read(common_path)
    hashes = {name: sha(meta[name]) for name in ("train", "validation")}
    assert hashes == result["input_hashes"]
    assert sha(common_path) == result["common_sha256"]
    event = next(e for e in result["trajectory"]["attempts"] if e.get("kind") == "joint_tail")
    penalty = float(result["trajectory"]["penalty"])
    data = prepare(observed(meta["train"], meta["N"]))
    n = data.n
    matches = [p for p in result["trajectory"]["path"]
               if abs(p["objective"] - event["before"]) <= 1e-12
               and abs(p["objective"] + penalty * sum(1 < len(s) < n for s in p["supports"]) - event["before_score"]) <= 1e-12]
    assert matches, "T-entry point not found in saved path"
    current = matches[-1]
    keep = canonical(event["keep"])
    deadline = started + args.job_seconds

    class TimedBank(FloorBank):
        def fit(self, family, *, preserve=False):
            remaining = deadline - perf_counter()
            if remaining <= .05:
                raise TimeoutError("Diagnostic job budget before fixed-family LP")
            self.seconds = min(args.lp_seconds, remaining)
            self.lp.model.Params.TimeLimit = self.seconds
            return super().fit(family, preserve=preserve)

    bank = TimedBank(data, common["pool"], common["calibration"], seconds=args.lp_seconds)
    try:
        model, domain = build_joint(bank.scaled, keep)
        assert canonical(domain["candidates"]) == canonical(event["candidates"])
        assert set(canonical(current["supports"])) <= set(domain["all_supports"])
        assert abs(loss(data, current) - event["before"]) <= 1e-12
        assert audit_vector(model, reconstruction_start(bank, model, domain, current))["valid"]
        protected = sum(1 < len(s) < n for s in keep)
        capacity = sum(max(0, len(r["blocks"]) - 2) for r in domain["regions"])
        completion = []
        for c in domain["candidates"]:
            if all(compatible(c, s) for s in (*keep, *completion)):
                completion.append(c)
        assert len(completion) == capacity, "Greedy hierarchy did not attain the known local count capacity"
        common_receipt = dict(job=job, source_snapshot=str(native_source), source_files_checked=len(source_checks),
                              input_hashes=hashes, saved_result_sha256=sha(result_path), common_sha256=sha(common_path),
                              driver_sha256=sha(__file__), truth_used_only_for_diagnostic=True, test_opened=False,
                              limits=dict(profile_MIP_seconds=args.mip_seconds, job_seconds=args.job_seconds,
                                          LP_seconds=args.lp_seconds, mip_gap=.0001, threads=1),
                              n=n, keep=keep, candidates=domain["candidates"], protected_count=protected,
                              optional_capacity=capacity, entry=current, old_event=event,
                              reference=result["reference"], current_lambda=penalty,
                              definition="phi(q)=min raw training MAE over exact q selected candidate binaries; optional zero weights allowed; protected count constant",
                              profiles=[])
        write(target.with_name("progress.json"), common_receipt)
        for q in range(capacity + 1):
            if deadline - perf_counter() <= .1:
                break
            model, domain = build_joint(bank.scaled, keep)
            row = {i: 1. for i in range(domain["selection"].start, domain["selection"].stop)}
            model.rows.add(row, lower=float(q), upper=float(q))
            warm = bank.fit([*keep, *completion[:q]], preserve=True)
            start = reconstruction_start(bank, model, domain, warm)
            values, record = solve(bank, model, start,
                                   seconds=min(args.mip_seconds, args.job_seconds - (perf_counter() - started)))
            item = dict(selected_candidates=q, encoded_total_count=protected + q, **record)
            if values is not None:
                chosen = [c for k, c in enumerate(domain["candidates"]) if values[domain["selection"].start + k] > .5]
                assert len(chosen) == q
                point = bank.fit([*keep, *chosen], preserve=True)
                physical = loss(data, point)
                assert physical <= record["objective"] + 1e-8
                assert record["dual_bound"] is None or record["dual_bound"] <= physical + 1e-8
                # Binary q is deliberately distinct from positive-weight count.
                actual = sum(1 < len(c) < n and max(r, x) > 1e-9
                             for c, r, x in zip(point["supports"], point["r"], point["x"]))
                item.update(point=point, physical_objective=physical, positive_internal_count=actual,
                            phi_upper=physical, phi_lower=record["dual_bound"])
            common_receipt["profiles"].append(item)
            write(target.with_name("progress.json"), common_receipt)

        # Truth is opened after all cardinality searches, never used for starts.
        labels = common["labels"]
        truth = physical_truth(Path(meta["truth_network"]), labels)
        label_index = {label: i for i, label in enumerate(labels)}
        true_internal = canonical(tuple(label_index[label] for label in C) for C in truth["clades"])
        true_family = canonical([*true_internal, *((i,) for i in range(n)), tuple(range(n))])
        absent = canonical(set(true_internal) - set(domain["all_supports"]))
        extras = canonical(s for s in keep if 1 < len(s) < n and s not in set(true_internal))
        reference = dict(internal=true_internal, truth_count=len(true_internal), missing_from_domain=absent,
                         extra_protected=extras, feasible_topology=not absent)
        if not absent:
            reference_point = bank.fit(true_family, preserve=True)
            lookup = {tuple(c): (r, x) for c, r, x in zip(reference_point["supports"], reference_point["r"], reference_point["x"])}
            embedded = canonical([*true_family, *keep])
            embedded_point = dict(supports=embedded,
                                  r=[lookup.get(c, (0., 0.))[0] for c in embedded],
                                  x=[lookup.get(c, (0., 0.))[1] for c in embedded],
                                  objective=reference_point["objective"])
            truth_q = len(set(true_internal) & set(domain["candidates"]))
            plain_model, plain_domain = build_joint(bank.scaled, keep)
            witness = reconstruction_start(bank, plain_model, plain_domain, embedded_point)
            witness_audit = audit_vector(plain_model, witness)
            assert witness_audit["valid"]
            independent = independent_profile(load_split(meta["train"], meta["N"]), true_family)
            assert abs(independent["objective"] - reference_point["objective"]) <= 2e-8
            reference.update(point=reference_point, embedded_point=embedded_point, selected_candidates=truth_q,
                             encoded_total_count=protected + truth_q, witness_audit=witness_audit,
                             independent_LP=independent, loss=reference_point["objective"])
            common_receipt["necessary_lambda_interval"] = lambda_envelope(common_receipt["profiles"], reference["loss"], truth_q, bound_key="phi_upper")
            common_receipt["sufficient_lambda_interval"] = lambda_envelope(common_receipt["profiles"], reference["loss"], truth_q, bound_key="phi_lower")
        elapsed = perf_counter() - started
        common_receipt.update(truth_reference=reference, complete_profiles=len(common_receipt["profiles"]) == capacity + 1,
                              all_profiles_gap_met=all(row.get("status") == 2 for row in common_receipt["profiles"]),
                              LP_failures=bank.failures, seconds=elapsed,
                              budget_exceeded=elapsed > args.job_seconds + 1.)
        write(target, common_receipt)
        print(f"PROFILE {job}: {len(common_receipt['profiles'])}/{capacity+1} q-values, {common_receipt['seconds']:.2f}s", flush=True)
        return common_receipt
    finally:
        bank.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=str(ROOT / "outputs/full_ablation_20261102"))
    parser.add_argument("--output", default=str(ROOT / "outputs/t_regularization_audit_20261102"))
    parser.add_argument("--job", action="append")
    parser.add_argument("--mip-seconds", type=float, default=300.)
    parser.add_argument("--lp-seconds", type=float, default=300.)
    parser.add_argument("--job-seconds", type=float, default=3600.)
    args = parser.parse_args()
    source = Path(args.snapshot).resolve() / "source"
    sys.path[:0] = [str(source), str(source / "rnj_wzzt_core")]
    if args.job:
        jobs = args.job
    else:
        with (ROOT / "docs/full_ablation_20261102/lost_exact_cases.csv").open(encoding="utf-8", newline="") as stream:
            jobs = [row["job"] for row in csv.DictReader(stream)]
    for job in jobs:
        run_job(args, job)


if __name__ == "__main__":
    main()
