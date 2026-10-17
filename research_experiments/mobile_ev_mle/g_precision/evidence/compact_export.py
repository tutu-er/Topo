"""Export lossless scientific endpoints and replay their independent AC audit.

Original gzip records are never changed.  Only repeated arrays in per-start
optimization traces and saved final-block residual vectors are replaced by
explicit SHA/size receipts.  Observations, measurement models, response
coefficients, all operations/parents/MILP logs, fitted endpoint states, losses,
decisions, noises, costs and failure descriptions remain available.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys


STAGE = Path(__file__).resolve().parents[4]


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    if path.suffix == ".gz":
        # mtime=0 makes this export deterministic for a fixed source record.
        data = gzip.compress(data, compresslevel=9, mtime=0)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def receipt(value):
    data = canonical(value)
    return dict(canonical_json_sha256=sha256(data).hexdigest(),
                canonical_json_bytes=len(data))


def numeric_tree(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, list) and bool(value) and all(numeric_tree(item) for item in value)


def shape(value):
    if not isinstance(value, list):
        return []
    children = [shape(item) for item in value]
    return [len(value)] + (children[0] if children and all(s == children[0] for s in children) else [])


def compact_start_tree(value, removed):
    if isinstance(value, dict):
        return {key: compact_start_tree(item, removed) for key, item in value.items()}
    if isinstance(value, list):
        if numeric_tree(value):
            removed["start_numeric_arrays"] += 1
            return dict(compact_omitted_numeric_array=True, shape=shape(value), **receipt(value))
        return [compact_start_tree(item, removed) for item in value]
    return value


def compact_record(original):
    record = copy.deepcopy(original)
    removed = dict(starts=0, start_numeric_arrays=0, final_block_residuals=0)
    for action in record.get("evaluation", {}).get("actions", {}).values():
        for actual in action.get("regimes", {}).values():
            for label in ("noisy", "clean"):
                refit = actual.get(label + "_refit", {})
                for comparison in refit.get("comparisons", {}).values():
                    if "starts" in comparison:
                        original_starts = comparison["starts"]
                        compact_starts = []
                        for start in original_starts:
                            entry = compact_start_tree(start, removed)
                            if isinstance(entry, dict):
                                entry["compact_original_start"] = receipt(start)
                            compact_starts.append(entry)
                            removed["starts"] += 1
                        comparison["starts"] = compact_starts
                        comparison["compact_original_starts"] = receipt(original_starts)
                    for route in ("fixed", "joint"):
                        for block in comparison.get(route, {}).get("blocks", {}).values():
                            if "residual" in block:
                                values = block.pop("residual")
                                block["compact_removed_residual"] = dict(shape=shape(values), **receipt(values))
                                removed["final_block_residuals"] += 1
    record["compact_export"] = dict(version=1, removed=removed,
        scope="only repeated per-start numeric arrays and final fitted-block residual vectors; historical fit record untouched; all failure descriptions retained")
    return record, removed


VOLATILE = {"generated_utc", "record_path", "record_sha256"}


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items() if key not in VOLATILE}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def export(source, destination, *, replay=True):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("compact destination must be outside the original raw output tree")
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    for phase in ("development", "evaluation"):
        for path in sorted((source / phase).glob("history_*.json.gz")):
            relative = path.relative_to(source)
            compact, removed = compact_record(read(path))
            target = destination / relative
            write(target, compact)
            inventory.append(dict(path=relative.as_posix(), raw_absolute_path=str(path),
                raw_sha256=digest(path), raw_bytes=path.stat().st_size,
                compact_sha256=digest(target), compact_bytes=target.stat().st_size,
                original_canonical_json=receipt(read(path)), removed=removed))
    for name in ("freeze.json", "environment.json", "source_snapshot.zip", "checkpoint.json", "progress.jsonl"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    originals = destination / "original_analysis"
    originals.mkdir(exist_ok=True)
    for name in ("summary.json", "audit.json", "history_pairs.json", "report.md"):
        shutil.copy2(source / name, originals / name)
    archive = source / "development_audit_passed"
    if archive.exists():
        shutil.copytree(archive, destination / "development_audit_passed", dirs_exist_ok=True)
    analysis_path = STAGE / "research_experiments/mobile_ev_mle/g_precision/analysis.py"
    shutil.copy2(analysis_path, destination / "analysis_postprocessor.py")
    archive_path = destination / "source_snapshot.zip"
    manifest = dict(version=1, source_output=str(source), compact_output=str(destination),
        raw_retained_untouched=True, files=inventory,
        raw_total_bytes=sum(row["raw_bytes"] for row in inventory),
        compact_total_bytes=sum(row["compact_bytes"] for row in inventory),
        source_archive_sha256=digest(archive_path), source_archive_bytes=archive_path.stat().st_size,
        export_script_sha256=digest(__file__), analysis_sha256=digest(analysis_path),
        exclusions=list(sorted(VOLATILE)),
        scientific_rule="no new fits, plans, simulated data, or outcome-dependent selection; only deterministic endpoint-preserving compaction")
    write(destination / "raw_and_export_sha256.json", manifest)
    replay_command = [sys.executable, "-m", "research_experiments.mobile_ev_mle.g_precision.analysis",
                      "--output", str(destination)]
    completed = None
    if replay:
        completed = subprocess.run(replay_command, cwd=STAGE, text=True,
                                   capture_output=True, check=False)
        (destination / "replay_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (destination / "replay_stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError("compact replay failed; see replay_stdout.log and replay_stderr.log")
    comparison = dict(replay_command=replay_command, working_directory=str(STAGE),
        exported_histories=len(inventory), source_archive_sha256=manifest["source_archive_sha256"],
        export_script_sha256=manifest["export_script_sha256"], analysis_sha256=manifest["analysis_sha256"],
        generated_utc=datetime.now(timezone.utc).isoformat(), ignored_comparison_fields=sorted(VOLATILE),
        compared_files={})
    if replay:
        for name in ("summary.json", "audit.json", "history_pairs.json"):
            original, compact = read(originals / name), read(destination / name)
            same = normalized(original) == normalized(compact)
            comparison["compared_files"][name] = dict(scientific_content_identical=same,
                original_sha256=digest(originals / name), replay_sha256=digest(destination / name))
        audit = read(destination / "audit.json")
        comparison["audit_passed"] = audit["passed"]
        comparison["audit_issues"] = len(audit["issues"])
        comparison["forward_AC_replay_calls"] = audit["forward_AC_replay_calls"]
        comparison["all_scientific_content_identical"] = all(
            row["scientific_content_identical"] for row in comparison["compared_files"].values())
    write(destination / "compact_replay_comparison.json", comparison)
    restore_command = (
        "Expand-Archive -LiteralPath '<evidence>/source_snapshot.zip' -DestinationPath '<restore-root>'\n"
        "Copy-Item -LiteralPath '<evidence>/analysis_postprocessor.py' -Destination '<restore-root>/research_experiments/mobile_ev_mle/g_precision/analysis.py'\n"
        "Set-Location -LiteralPath '<restore-root>'\n"
        "python -m research_experiments.mobile_ev_mle.g_precision.analysis --output '<absolute-evidence-path>'")
    text = ["# Compact frozen G-precision evidence", "",
        "All ten raw records remain unchanged in the original D-stage output. Their exact SHA256 and sizes, and each compact export SHA256, are in raw_and_export_sha256.json.", "",
        "The compact records preserve observations, meter models, noise, configuration, candidate structures, trained history, response coefficients, plans, parents, complete MILP logs/bounds, final FIX/JNT states, block/channel SSE, decisions, costs and failures. Repeated numerical arrays inside optimization starts and final residual vectors are replaced by explicit hash/size receipts; no endpoint is replaced or refitted.", "",
        f"Raw gzip records: {manifest['raw_total_bytes']:,} bytes. Compact gzip records: {manifest['compact_total_bytes']:,} bytes.", "",
        "source_snapshot.zip is the frozen scientific source. The analysis postprocessor is intentionally outside the science hash closure and is archived separately with its own SHA256. original_analysis preserves the full-batch original summary and audit; development_audit_passed preserves the first successful development audit.", "",
        "To replay the independent forward-AC and SSE audit after restoring the source:", "", "```powershell", restore_command, "```", "",
        "This requires the recorded Python/numerical environment but no MILP solver or new fitting. Complete simulation and fitting cost remains the original experiment's cost; replay calls are recorded separately.", ""]
    if replay:
        text.extend([f"Replay audit passed: {comparison['audit_passed']}; issues: {comparison['audit_issues']}; forward AC calls: {comparison['forward_AC_replay_calls']}.",
            f"Scientific summary, audit and history-pair content identical: {comparison['all_scientific_content_identical']}.", ""])
    (destination / "README.md").write_text("\n".join(text), encoding="utf-8")
    if replay and (not comparison["audit_passed"] or not comparison["all_scientific_content_identical"]):
        raise RuntimeError("compact scientific comparison failed; preserved comparison.json describes the discrepancy")
    return manifest, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=STAGE / "outputs/g_precision_v1")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parent / "compact")
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    manifest, comparison = export(args.source, args.destination, replay=not args.skip_replay)
    print(json.dumps(dict(raw_bytes=manifest["raw_total_bytes"], compact_bytes=manifest["compact_total_bytes"],
        histories=len(manifest["files"]), audit_passed=comparison.get("audit_passed"),
        scientific_content_identical=comparison.get("all_scientific_content_identical"))))


if __name__ == "__main__":
    main()
