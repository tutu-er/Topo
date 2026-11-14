"""Bounded validation of measured P/Q, RNJ, and a separate true-structure QP.

Default: four unchanged networks and four fixed seasonal four-day blocks.
No alternative date is substituted after a failed/missing input. No DFS is run.
Old fitted libraries must not be reused with these newly built histories.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...history import fit_observed, freeze_history
from ...identification.algorithm import SearchConfig
from ...identification.optimization import QPFitter, errors
from ...observations import Observation, fingerprint
from .physical import CASES, SPLIT, ac_backend, generate, offline_truth


DEFAULT_DATES = ("2024-01-15", "2024-04-15", "2024-07-15", "2024-10-14")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _environment():
    environment = io.environment()
    for name in ("pandas", "pyarrow"):
        try:
            environment["packages"][name] = version(name)
        except PackageNotFoundError:
            environment["packages"][name] = None
    return environment


def _git_receipt(source_paths):
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        relative = [str(Path(path).resolve().relative_to(root)) for path in source_paths]
        status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *relative],
                                         cwd=root, text=True).splitlines()
        return dict(git_commit=commit, dirty_task_sources=status,
                    dirty_task_sources_meaning="Git status of the exact hashed source/dependency files; staged, unstaged and untracked changes are retained. Source hashes identify the executed bytes even before a commit.")
    except (OSError, subprocess.CalledProcessError) as exc:
        return dict(git_commit=None, dirty_task_sources=None,
                    git_receipt_error=type(exc).__name__+": "+str(exc))


def pq_diagnostics(observed):
    """Report actual observed P/Q variation; no fitted voltage or physical truth."""
    p, q = observed.p, observed.q
    x = np.column_stack((p, q)); centered = x-x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    threshold = np.finfo(float).eps*max(centered.shape)*singular[0] if singular.size else 0.
    condition = float(singular[0]/singular[-1]) if singular.size and singular[-1] > threshold else None
    sigma = centered.std(axis=0)
    standardized = np.divide(centered, sigma, out=np.zeros_like(centered), where=sigma > 0)
    normalized_singular = np.linalg.svd(standardized, compute_uv=False)
    corr, slopes = [], []
    for a, b in zip(p.T, q.T):
        ac, bc = a-a.mean(), b-b.mean()
        norm = np.linalg.norm(ac)*np.linalg.norm(bc)
        corr.append(float(ac@bc/norm) if norm > 0 else None)
        slopes.append(float(a@b/(a@a)) if a@a > 0 else None)
    return dict(points=observed.points, labels=list(observed.labels), per_node_PQ_correlation=corr,
        per_node_Q_on_P_origin_slope=slopes, centered_design_rank=int(np.linalg.matrix_rank(centered)),
        centered_design_columns=2*observed.n, centered_design_condition_number=condition,
        centered_design_singular_values=singular.tolist(),
        standardized_design_singular_values=normalized_singular.tolist(),
        negative_P_cells=int((p < 0).sum()), negative_Q_cells=int((q < 0).sum()),
        definition="SVD of centered [P,Q], not the topology-constrained QP feature matrix")


def _active(point, n, zero=1e-9, *, internal=False):
    return {tuple(C) for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
            if max(r, x) > zero and (not internal or 1 < len(C) < n)}


def _topology(point, truth, n, zero):
    predicted = _active(point, n, zero)
    actual = {tuple(C) for C in truth}
    pi = {C for C in predicted if 1 < len(C) < n}
    ti = {C for C in actual if 1 < len(C) < n}
    return dict(full_exact=predicted == actual, full_missing=len(actual-predicted),
        full_extra=len(predicted-actual), internal_exact=pi == ti,
        internal_missing=len(ti-pi), internal_extra=len(pi-ti),
        active_supports=[list(c) for c in sorted(predicted, key=lambda x: (len(x), x))])


def _save_observations(directory, generated):
    for name, observed in generated["splits"].items():
        np.savez_compressed(directory/(name+".npz"), P=observed.p, Q=observed.q,
                            Y=observed.y, V=observed.voltage, root=observed.root, labels=observed.labels)
    p = generated["private"]
    np.savez_compressed(directory/"private_physical.npz", P=p["p"], Q=p["q"], V=p["voltage"],
                        root=p["root"], labels=p["labels"], timestamps=p["timestamps"])


def _config(seconds):
    return SearchConfig(total_seconds=float(seconds), per_mip_seconds=float(seconds),
                        qp_seconds=max(10., float(seconds)), output_reserve_seconds=min(3., .1*seconds))


def _fit_identifier(name, training, validation, directory, seconds):
    started = perf_counter()
    try:
        result = fit_observed(training, validation, method=name, config=_config(seconds))
        # Save all native statuses/paths before any new-day score or offline truth.
        io.write(directory/(name+"_native.json"), result)
        frozen = freeze_history(result, validation)
        return dict(status="fitted", seconds=perf_counter()-started, result=result, frozen=frozen)
    except Exception as exc:
        failure = dict(status="failed", seconds=perf_counter()-started,
                       error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
        io.write(directory/(name+"_failure.json"), failure)
        return failure


def _evaluate_identifier(fitted, splits, truth):
    if fitted["status"] != "fitted":
        return {key: value for key, value in fitted.items() if key != "traceback"}
    result, frozen = fitted["result"], fitted["frozen"]
    before = result["selected"]["index"]
    after_receipt = frozen.select(splits["new_day"])
    after = after_receipt["selected"]
    points = [row["point"] for row in result["frozen_candidates"]]
    zero, n = result["config"]["zero_tol"], splits["train"].n
    return dict(status="complete", seconds=fitted["seconds"], bank_size=len(points),
        history_selected=before, cumulative_selected=after,
        history_topology=_topology(points[before], truth, n, zero),
        cumulative_topology=_topology(points[after], truth, n, zero),
        bank_has_full_truth=any(_topology(p, truth, n, zero)["full_exact"] for p in points),
        bank_has_internal_truth=any(_topology(p, truth, n, zero)["internal_exact"] for p in points),
        history_selected_losses={key: errors(observed, points[before]) for key, observed in splits.items()},
        cumulative_selected_losses={key: errors(observed, points[after]) for key, observed in splits.items()},
        cumulative_selection=after_receipt, stop_reason=result["stop_reason"],
        total_cap_exceeded=result["total_cap_exceeded"],
        recorded_native_QP_count=len(result["QP"]), MIQP_count=len(result["MIQP"]),
        common_seconds=result["common_seconds"], common_QP_records_available=False,
        native_returned_result_saved_in_full=True,
        scoring_rows="96 historical validation + 96 new-day rows; training is not rescored",
        topology_metric="internal_exact matches previous proper-clade recovery; full_exact also checks singleton and shared-trunk supports")


def _oracle_fit(splits, truth, directory):
    """Fixed physical support only; R and X are freely fitted to training P/Q/V."""
    started = perf_counter()
    fitter = QPFitter(splits["train"], reference_mse=1., bound=2., qp_seconds=30.)
    try:
        point = fitter.fit(truth)
        output = dict(status="complete", seconds=perf_counter()-started,
            role="offline true-structure parameter-fit diagnostic, excluded from all identifier banks",
            coefficient_rule="separate freely estimated R/X response coefficients in [0,2]; no fixed R/X ratio",
            point=point, QP=fitter.records,
            losses={key: errors(observed, point) for key, observed in splits.items()},
            retained_structure=_topology(point, truth, splits["train"].n, 1e-9))
    except Exception as exc:
        output = dict(status="failed", seconds=perf_counter()-started, QP=fitter.records,
                      error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
    io.write(directory/"offline_true_structure_qp.json", output)
    return {k: v for k, v in output.items() if k not in {"point", "traceback"}}


def run(archive_path, output, *, cases=CASES, start_dates=DEFAULT_DATES,
        seed=20261114, voltage_sigma=.0005, pq_sigma=.005, root_sigma=.003,
        method101_seconds=None):
    from . import profiles
    cases, dates = tuple(cases), tuple(start_dates)
    if not cases or len(set(cases)) != len(cases) or not set(cases) <= set(CASES):
        raise ValueError("Use a nonempty, unique subset of the four original networks")
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("Use unique, explicitly selected seasonal start dates")
    if method101_seconds is not None and (not np.isfinite(method101_seconds) or method101_seconds <= 0):
        raise ValueError("Optional method101 budget must be positive")
    output, archive_path = Path(output).resolve(), Path(archive_path).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous outputs; choose a new directory")
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    output.mkdir(parents=True)
    lock = output/"lock.json"
    io.write(lock, dict(pid=os.getpid(), created_utc=_utc()))
    source_paths = [Path(__file__), Path(__file__).with_name("physical.py"), Path(profiles.__file__),
                    Path(profiles.__file__).with_name("fetch.py"), Path(ac_backend.__file__),
                    Path(__file__).parents[2]/"identification"/"algorithm.py",
                    Path(__file__).parents[2]/"identification"/"optimization.py",
                    Path(__file__).parents[2]/"identification"/"trees.py",
                    Path(__file__).parents[2]/"identification"/"competition.py",
                    Path(__file__).parents[2]/"history.py",
                    Path(__file__).parents[2]/"observations.py",
                    Path(__file__).parents[2]/"dr"/"selection_mse.py",
                    Path(__file__).parents[2]/"io.py"]
    source_paths.extend(ac_backend.ROOT/"cases"/case/name for case in cases
                        for name in ("metadata.json", "buses.csv", "branches.csv"))
    sources = [io.ref(path) for path in source_paths]
    tasks = [dict(case=case, start_date=date, seed=int(seed)+i)
             for i, (date, case) in enumerate((d, c) for d in dates for c in cases)]
    protocol = dict(created_utc=_utc(), archive=io.ref(archive_path), sources=sources,
        tasks=tasks, split=SPLIT, environment=_environment(), **_git_receipt(source_paths),
        methods=["rnj"]+(["101"] if method101_seconds is not None else []),
        method_configs={"rnj": asdict(_config(30.)),
                        **({"101": asdict(_config(method101_seconds))} if method101_seconds is not None else {})},
        extra_meter_noise=dict(pq_relative_sd=pq_sigma, voltage_relative_sd=voltage_sigma),
        root_physical_variation_sigma=root_sigma,
        truth_isolation="No truth/new-day inputs to RNJ/101. True support QP separate after native fit freeze.",
        no_old_candidate_library_reuse=True, added_new_day_DER_baseline=False, DFS_run=False,
        failure_rule="Keep every fixed input and failure; no alternate dates, power rescaling or seed replacement",
        voltage_rule="Report .90–1.10/.95–1.05 excursions; do not alter or omit inputs",
        use="Measured-profile-driven simulated networks, not field topology recovery")
    io.write(output/"protocol.json", protocol)
    rows = []
    try:
        for task in tasks:
            case, date = task["case"], task["start_date"]
            directory = output/(case+"__"+date)
            directory.mkdir()
            started = perf_counter()
            try:
                profile = profiles.build_case(case, archive_path, date,
                                              expected_sha256=protocol["archive"]["sha256"])
                generated = generate(case, profile, seed=task["seed"], voltage_sigma=voltage_sigma,
                                     pq_sigma=pq_sigma, root_sigma=root_sigma)
                _save_observations(directory, generated)
                io.write(directory/"physical_receipt.json", generated["diagnostics"])
                splits = generated["splits"]
                observations = {name: dict(ref=io.ref(directory/(name+".npz")), sha256=fingerprint(obs))
                                for name, obs in splits.items()}
                io.write(directory/"observations_sealed.json", dict(observations=observations,
                    profile_metadata=profile.get("metadata", {}), created_utc=_utc()))
                fitted = {"rnj": _fit_identifier("rnj", splits["train"], splits["validation"], directory, 30.)}
                if method101_seconds is not None:
                    fitted["101"] = _fit_identifier("101", splits["train"], splits["validation"], directory, method101_seconds)
                # Original native banks are now saved/frozen. Truth is evaluation only.
                truth = offline_truth(case, splits["train"].labels)
                methods = {name: _evaluate_identifier(value, splits, truth) for name, value in fitted.items()}
                oracle = _oracle_fit(splits, truth, directory)
                clean = generated["private"]
                clean_pq = {}
                for name, (start, stop) in SPLIT.items():
                    source = Observation(clean["p"][start:stop], clean["q"][start:stop],
                        clean["root"][start:stop, None]**2-clean["voltage"][start:stop]**2,
                        clean["labels"])
                    clean_pq[name] = pq_diagnostics(source)
                row = dict(**task, status="complete", seconds=perf_counter()-started,
                    physical=generated["diagnostics"],
                    pq={key: pq_diagnostics(obs) for key, obs in splits.items()},
                    source_pq_before_additional_meter_noise=clean_pq,
                    methods=methods, offline_true_structure_qp=oracle)
            except Exception as exc:
                row = dict(**task, status="failed", seconds=perf_counter()-started,
                           error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
            io.write(directory/"result.json", row)
            rows.append(row)
            io.write(output/"progress.json", dict(completed=len(rows), total=len(tasks), rows=rows))
            print(f"{len(rows)}/{len(tasks)} {case} {date}: {row['status']}", flush=True)
        source_unchanged = all(io.ref(ref["path"])["sha256"] == ref["sha256"] for ref in sources)
        report = dict(status="complete", planned_inputs=len(tasks), completed_inputs=len(rows),
            input_failures=sum(row["status"] == "failed" for row in rows),
            method_failures=sum(value["status"] != "complete" for row in rows for value in row.get("methods", {}).values()),
            oracle_failures=sum(row.get("offline_true_structure_qp", {}).get("status") == "failed" for row in rows),
            source_unchanged=source_unchanged, protocol=io.ref(output/"protocol.json"), rows=rows)
        io.write(output/"REPORT.json", report)
        io.write(output/"complete.json", dict(created_utc=_utc(), report=io.ref(output/"REPORT.json"),
                                             source_unchanged=source_unchanged))
        return report
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--dates", default=",".join(DEFAULT_DATES))
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--voltage-sigma", type=float, default=.0005)
    parser.add_argument("--pq-sigma", type=float, default=.005)
    parser.add_argument("--root-sigma", type=float, default=.003)
    parser.add_argument("--method101-seconds", type=float)
    args = parser.parse_args()
    report = run(args.archive, args.out, cases=args.cases.split(","), start_dates=args.dates.split(","),
        seed=args.seed, voltage_sigma=args.voltage_sigma, pq_sigma=args.pq_sigma,
        root_sigma=args.root_sigma, method101_seconds=args.method101_seconds)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
