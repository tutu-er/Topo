"""Index saved WZZT result families without importing estimators or solvers.

Run from any directory with the Topo Python environment. Only the explicitly
selected report directory is written; original outputs are read without changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    ("ac96_wrapper", "source_ac96_benchmark_20261018", "summary.csv", 64, 96, 0,
     "clade_f1", ("independent_benchmark_audit.json",)),
    ("ac96_full_64_32", "source_ac96_full_history_pengwah18_20261018", "summary.csv", 6, 64, 32,
     "family_clade_f1", ("independent_full_history_audit.json",)),
    ("pengwah18_ac288", "source_ac288_benchmark_20261018", "reports_final/summary.csv", 16, 288, 0,
     "family_clade_f1", ("reports_final/independent_ac288_audit.json",)),
    ("soumalas11_rx75_source", "rx75_source_comparison_20261018", "reports_final/per_job.csv", 30, 288, 0,
     "resolution_clade_f1", ("reports_final/independent_audit.json",)),
    ("rnj_blocks96", "rnj_blocks96_20261019", "reports_final/effective_results.csv", 96, 96, 0,
     "resolution_clade_f1", ("audits/final/original.json", "audits/final/recording_recovery.json")),
)
FIELDS = (
    "suite", "job_id", "case", "replicate", "variant", "method", "budget_multiplier",
    "fit_rows", "validation_rows", "test_rows", "primary_f1_field", "primary_f1",
    "family_f1", "active_f1", "resolution_f1", "r_error_percent", "x_error_percent",
    "status", "stop_reason", "has_model", "retained_fallback", "reused_baseline",
    "recovery_applied", "original_status", "compute_seconds", "combined_spent_seconds",
    "per_solve_seconds", "method_wall_seconds", "selected_rnj_blocks", "selected_path_index",
    "source_table", "source_row", "result_directory", "suite_protocol_sha256", "attempt_protocol_sha256",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def is_true(value):
    return str(value).lower() == "true"


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite metric: {value!r}")
    return result


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(output):
    rows, suites, evidence = [], [], {}

    def record(path):
        evidence[relative(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}

    for suite, directory, table, expected, fit, validation, f1, audits in SUITES:
        run = ROOT / "outputs" / directory
        table_path = run / table
        source = read_csv(table_path)
        if len(source) != expected or len({r["job_id"] for r in source}) != expected:
            raise ValueError(f"Missing or duplicate jobs in {table_path}")
        protocol_path = run / "protocol.json"
        protocol = read_json(protocol_path)
        declared_jobs = protocol.get("jobs")
        if declared_jobs is None:
            declared_jobs = [dict(
                case=case, variant=variant, method=method,
                job_id=f"{case}__{variant}__{method}",
                per_solve_seconds=protocol["per_solve_seconds"] if "WZZT" in method else None,
                method_wall_seconds=protocol["method_wall_seconds"],
            ) for case, variant, method in product(protocol["cases"], protocol["variants"], protocol["methods"])]
        jobs = {j["job_id"]: j for j in declared_jobs}
        protocol_sha = digest(protocol_path)
        for path in (table_path, protocol_path, run / "completion.json", run / "execution_complete.json"):
            record(path)
        stored_audits = []
        for name in audits:
            path = run / name
            data = read_json(path)
            if not data.get("passed") or data.get("failed_count") != 0:
                raise ValueError(f"Stored audit did not pass: {path}")
            stored_audits.append({"path": relative(path), **{
                k: data[k] for k in ("passed", "check_count", "failed_count", "audited_final_jobs") if k in data
            }})
            record(path)
        suite_rows = []
        for line, raw in enumerate(source, 2):
            job = jobs.get(raw["job_id"], {})
            reused = is_true(raw.get("reused_baseline", False))
            if not job and not reused:
                raise ValueError(f"Job absent from frozen protocol: {raw['job_id']}")
            result_dir = Path(raw.get("result_directory") or run / "jobs" / raw["job_id"])
            if not (result_dir / "result.json").is_file():
                raise ValueError(f"Missing saved result: {result_dir}")
            saved = read_json(result_dir / "result.json")
            if saved.get("job_id") != raw["job_id"] or saved.get("status") != raw["status"]:
                raise ValueError(f"CSV/result identity or status mismatch: {result_dir}")
            record(result_dir / "result.json")
            attempt_protocol = Path(raw.get("attempt_root") or run) / "protocol.json"
            record(attempt_protocol)
            producer_sha = saved.get("producer_protocol_sha256") or saved.get("protocol_sha256")
            if not producer_sha:
                raise ValueError(f"Missing producer protocol identity: {result_dir}")
            if not reused and producer_sha != digest(attempt_protocol):
                raise ValueError(f"Result/attempt protocol mismatch: {result_dir}")
            if reused and producer_sha != digest(ROOT / "outputs/source_ac96_benchmark_20261018/protocol.json"):
                raise ValueError(f"Unexpected historical producer protocol: {result_dir}")
            r_pct = number(raw.get("r_error_percent"))
            x_pct = number(raw.get("x_error_percent"))
            if r_pct is None and raw.get("r_relative_error"):
                r_pct = 100 * number(raw["r_relative_error"])
            if x_pct is None and raw.get("x_relative_error"):
                x_pct = 100 * number(raw["x_relative_error"])
            status = raw["status"]
            has_model = is_true(raw.get("has_model"))
            row = dict(
                suite=suite, job_id=raw["job_id"], case=raw.get("case") or job.get("case"),
                replicate=raw.get("replicate", ""), variant=raw["variant"], method=raw["method"],
                budget_multiplier=raw.get("budget_multiplier", ""), fit_rows=fit,
                validation_rows=validation, test_rows=0, primary_f1_field=f1,
                primary_f1=number(raw.get(f1)), family_f1=number(raw.get("family_clade_f1")),
                active_f1=number(raw.get("active_clade_f1")),
                resolution_f1=number(raw.get("resolution_clade_f1")),
                r_error_percent=r_pct, x_error_percent=x_pct, status=status,
                stop_reason=raw["stop_reason"], has_model=has_model,
                retained_fallback=is_true(raw.get("retained_fallback")) or (has_model and status != "complete"),
                reused_baseline=reused, recovery_applied=is_true(raw.get("recovery_applied")),
                original_status=raw.get("original_status", status),
                compute_seconds=number(raw.get("compute_seconds") or raw.get("elapsed_seconds")),
                combined_spent_seconds=number(raw.get("combined_spent_seconds")),
                per_solve_seconds=number(raw.get("per_solve_seconds") or job.get("per_solve_seconds")),
                method_wall_seconds=number(raw.get("method_wall_seconds") or job.get("method_wall_seconds")),
                selected_rnj_blocks=raw.get("selected_rnj_blocks", raw.get("strict_selected_count", "")),
                selected_path_index=raw.get("selected_path_index", ""),
                source_table=relative(table_path), source_row=line, result_directory=relative(result_dir),
                suite_protocol_sha256=protocol_sha, attempt_protocol_sha256=producer_sha,
            )
            for key in ("primary_f1", "family_f1", "active_f1", "resolution_f1"):
                if row[key] is not None and not 0 <= row[key] <= 1:
                    raise ValueError(f"Invalid F1 in {raw['job_id']}: {key}")
            suite_rows.append(row)
        rows.extend(suite_rows)
        suites.append(dict(
            suite=suite, planned_rows=expected, returned_rows=len(suite_rows),
            status_counts=dict(Counter(r["status"] for r in suite_rows)),
            original_status_counts=dict(Counter(r["original_status"] for r in suite_rows)),
            reused_baselines=sum(r["reused_baseline"] for r in suite_rows),
            recoveries=sum(r["recovery_applied"] for r in suite_rows),
            fit_rows=fit, validation_rows=validation, test_rows=0, primary_f1_field=f1,
            stored_audits=stored_audits, raw_completion=read_json(run / "completion.json"),
        ))

    # Match the exact same case / replicate / Source cells across every method.
    # The budget experiment is deliberately excluded: its NJ/RNJ controls have
    # no budget-multiplier label and must not be silently duplicated.
    comparisons = []
    for suite in ("rnj_blocks96", "soumalas11_rx75_source"):
        subset = [r for r in rows if r["suite"] == suite]
        methods = sorted({r["method"] for r in subset})
        cells = defaultdict(dict)
        for row in subset:
            key = (row["case"], row["replicate"], row["variant"])
            if row["method"] in cells[key]:
                raise ValueError(f"Duplicate comparison cell: {key}")
            cells[key][row["method"]] = row
        accepted = {k: v for k, v in cells.items() if set(v) == set(methods)
                    and all(r["status"] == "complete" and r["primary_f1"] is not None for r in v.values())}
        for method in methods:
            paired = [v[method] for v in accepted.values()]
            if not paired:
                raise ValueError(f"No complete matched conditions for {suite}")
            comparisons.append(dict(
                suite=suite, method=method, complete_common_conditions=len(paired),
                planned_conditions=len(cells),
                excluded_conditions=json.dumps([list(k) for k in sorted(set(cells) - set(accepted))]),
                mean_f1=sum(r["primary_f1"] for r in paired) / len(paired),
                mean_r_error_percent=sum(r["r_error_percent"] for r in paired) / len(paired),
                mean_x_error_percent=sum(r["x_error_percent"] for r in paired) / len(paired),
                mean_compute_seconds=sum(r["compute_seconds"] for r in paired) / len(paired),
                reused_baselines=sum(r["reused_baseline"] for r in paired),
            ))
    rows.sort(key=lambda r: (r["suite"], r["case"], str(r["replicate"]), r["variant"], r["method"], str(r["budget_multiplier"])))
    record(Path(__file__))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_job.csv", rows, FIELDS)
    incomplete = [r for r in rows if r["status"] != "complete"]
    write_csv(output / "incomplete.csv", incomplete, FIELDS)
    write_csv(output / "paired_complete.csv", comparisons, list(comparisons[0]))
    metadata = dict(
        schema_version=1, suites=suites, table_rows=len(rows),
        incomplete_rows=len(incomplete),
        scope="Index of saved result families; overlapping data and reused baselines are not independent trials",
        verification="CSV counts, identities, paths, ranges, stored audit summaries and SHA256 only; no solver or independent audit rerun",
    )
    (output / "suite_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "evidence_manifest.json").write_text(json.dumps({
        "scope": "Consumed tables, protocols, completion markers, result.json files, stored audits and index producer; not all raw NPZ/checkpoints",
        "files": dict(sorted(evidence.items())),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "indexed_rows": len(rows), "incomplete_rows": len(incomplete), "evidence_files": len(evidence)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/wzzt_case_results_20261027")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # Keep the index away from the original experimental evidence.
    if not output.is_relative_to(ROOT / "docs"):
        parser.error("--output-dir must be within this repository's docs directory")
    build(output)


if __name__ == "__main__":
    main()
