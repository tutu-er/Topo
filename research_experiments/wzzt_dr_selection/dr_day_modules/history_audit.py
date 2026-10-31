"""Observed-input audit for full-day DR design; no topology truth or solver.

prepare reads metadata and hashes only. run must be requested separately after
the protocol is frozen. Pair diagnostics describe observed near-proportional
inputs; they are not identifiability certificates, probabilities, or price scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np


CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
ZERO_CONTROLS = (
    "main__flynn16__b5__reference__v0002__N288",
    "main__pengwah18__b5__reference__v0002__N288",
)
REPO = Path(__file__).resolve().parents[3]
SLOTS = 96
POWER_BASE_KW = 200.0


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_metadata(inventory):
    """Project permitted metadata before retaining any source row."""
    selected = []
    for row in inventory["rows"]:
        if row["N"] != 288 or row["condition"] != "reference":
            continue
        if row["voltage_sigma"] != .0002 or row["pq_sigma"] != .005:
            continue
        item = {key: row[key] for key in (
            "job_id", "case", "bundle", "N", "condition", "voltage_sigma",
            "pq_sigma", "terminal_count", "declared_data_error",
        )}
        expected_id = f"main__{item['case']}__b{item['bundle']}__reference__v0002__N288"
        if item["job_id"] != expected_id:
            raise ValueError("source ID does not match the main normal-noise panel")
        item["sources"] = {}
        for role in ("train", "validation"):
            source = row[role]
            item["sources"][role] = {
                "path": source["path"],
                "catalog_sha256": source.get("catalog_sha256"),
            }
        item["retained_zero_gain_control"] = item["job_id"] in ZERO_CONTROLS
        item["previous_closed_b7_control"] = item["job_id"] == (
            "main__flynn16__b7__reference__v0002__N288"
        )
        selected.append(item)
    expected = {(case, bundle) for case in CASES for bundle in range(10)}
    keys = {(item["case"], item["bundle"]) for item in selected}
    if len(selected) != 40 or keys != expected:
        raise ValueError("normal N288 source metadata is not the complete 4x10 pool")
    return sorted(selected, key=lambda item: item["job_id"])


def pair_channel(target, predictor, *, sigma=.005, power_floor=1e-9):
    """Two scalar input vectors, zero-intercept positive-proportion LS.

    The reference uses observed powers as a plug-in for the unknown noise-free
    levels. Estimating alpha from the same noisy data is not a statistical test.
    """
    x = np.asarray(target, dtype=float)
    y = np.asarray(predictor, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not x.size:
        raise ValueError("aligned nonempty one-dimensional channels required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite observed power; do not clip or interpolate")
    if not np.isfinite([sigma, power_floor]).all() or min(sigma, power_floor) <= 0:
        raise ValueError("positive sigma and numerical power floor required")
    rx = float(np.sqrt(np.mean(x*x)))
    ry = float(np.sqrt(np.mean(y*y)))
    cx = float(np.sqrt(np.mean((x-x.mean())**2)))
    cy = float(np.sqrt(np.mean((y-y.mean())**2)))
    result = dict(
        count=int(x.size), target_rms=rx, predictor_rms=ry,
        target_centered_rms=cx, predictor_centered_rms=cy,
        alpha_unconstrained=None, alpha_nonnegative=None,
        residual_rms=None, rms_normalized_residual=None,
        noise_reference_rms=None, residual_over_noise_reference=None,
        at_one_noise_reference=None, at_two_noise_reference=None,
        target_has_negative=bool((x < 0).any()),
        predictor_has_negative=bool((y < 0).any()),
    )
    if min(rx, ry) <= power_floor:
        return dict(result, status="near_zero_power_unresolved")
    raw_alpha = float(np.dot(x, y) / np.dot(y, y))
    alpha = max(0.0, raw_alpha)
    residual = float(np.sqrt(np.mean((x-alpha*y)**2)))
    noise = float(sigma*np.sqrt(np.mean(x*x + alpha*alpha*y*y)))
    result.update(
        alpha_unconstrained=raw_alpha, alpha_nonnegative=alpha,
        residual_rms=residual, noise_reference_rms=noise,
        residual_over_noise_reference=residual/noise if noise > power_floor*sigma else None,
    )
    if raw_alpha <= 0:
        return dict(result, status="nonpositive_best_proportion")
    if min(cx, cy) <= power_floor:
        return dict(result, status="constant_channel_not_standardized")
    result["rms_normalized_residual"] = residual/rx
    ratio = result["residual_over_noise_reference"]
    if ratio is None:
        return dict(result, status="noise_reference_unresolved")
    result.update(
        status="positive_proportion_evaluated",
        at_one_noise_reference=bool(ratio <= 1.0),
        at_two_noise_reference=bool(ratio <= 2.0),
    )
    return result


def pair_record(p, q, i, j, labels, *, sigma=.005, power_floor=1e-9):
    result = dict(labels=[int(labels[i]), int(labels[j])], channels={})
    for channel, values in (("P_kw", p), ("Q_kvar", q)):
        directions = {
            "i_from_j": pair_channel(values[:, i], values[:, j], sigma=sigma, power_floor=power_floor),
            "j_from_i": pair_channel(values[:, j], values[:, i], sigma=sigma, power_floor=power_floor),
        }
        flags = {}
        for level, key in ((1, "at_one_noise_reference"), (2, "at_two_noise_reference")):
            values_at_level = [entry[key] for entry in directions.values()]
            flags[str(level)] = (
                all(values_at_level) if all(value is not None for value in values_at_level) else None
            )
        result["channels"][channel] = dict(directions=directions, descriptive_flags=flags)
    result["both_PQ_flags"] = {}
    for level in ("1", "2"):
        flags = [result["channels"][c]["descriptive_flags"][level] for c in ("P_kw", "Q_kvar")]
        result["both_PQ_flags"][level] = all(flags) if all(v is not None for v in flags) else None
    return result


def window_masks(days):
    hours = np.tile(np.arange(SLOTS)*.25, days)
    windows = [("all_day", np.ones(len(hours), dtype=bool))]
    windows += [
        (f"hours_{start:02d}_{start+4:02d}", (hours >= start) & (hours < start+4))
        for start in range(0, 24, 4)
    ]
    return windows


def empty_counts():
    flags = lambda: Counter(at_or_below=0, above=0, unresolved=0)
    return dict(pair_count=0, both_PQ={"1": flags(), "2": flags()}, channels={
        c: dict(statuses=Counter(), flags={"1": flags(), "2": flags()})
        for c in ("P_kw", "Q_kvar")
    })


def add_counts(counts, record):
    counts["pair_count"] += 1
    label = lambda value: "unresolved" if value is None else ("at_or_below" if value else "above")
    for level, value in record["both_PQ_flags"].items():
        counts["both_PQ"][level][label(value)] += 1
    for channel, item in record["channels"].items():
        for direction in item["directions"].values():
            counts["channels"][channel]["statuses"][direction["status"]] += 1
        for level, value in item["descriptive_flags"].items():
            counts["channels"][channel]["flags"][level][label(value)] += 1


def prepare(root, inventory_path, repo):
    root, repo, inventory_path = Path(root).resolve(), Path(repo).resolve(), Path(inventory_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/"protocol.json").exists():
        raise FileExistsError("history audit protocol already frozen")
    inventory = read(inventory_path)
    rows = select_metadata(inventory)
    for row in rows:
        for role, source in row["sources"].items():
            path = Path(source["path"]).resolve()
            if path.name != role + "_observed.npz":
                raise ValueError("role file name mismatch; never substitute test or true files")
            source["path"] = str(path)
            source["exists"] = path.is_file()
            source["bytes"] = path.stat().st_size if path.is_file() else None
            source["sha256"] = sha(path) if path.is_file() else None
            source["catalog_hash_matches"] = source["sha256"] is not None and source["sha256"] == source["catalog_sha256"]
        row["metadata_ready"] = all(s["catalog_hash_matches"] for s in row["sources"].values())
    source_paths = [Path(__file__).resolve()]
    test_source = repo/"research_experiments/tests/test_wzzt_dr_day_history_audit.py"
    if test_source.is_file():
        source_paths.append(test_source)
    for relative in (
        "research_experiments/wzzt_dr_selection/data_calibration/REPORT.md",
        "research_experiments/wzzt_dr_selection/data_calibration/protocol.json",
        "research_experiments/wzzt_dr_selection/data_calibration/protocol_control_training.json",
    ):
        source_paths.append(repo/relative)
    producer = Path(r"D:\0-github_workspace\Topo\outputs\wzzt_paper_20261029_full\source\rnj_wzzt\data")
    source_paths += [producer/"paper_style_case_bank.py", producer/"paper_style_terminal_lv.py"]
    protocol = dict(
        id="day_modules_history_input_audit_v1", status="frozen_metadata_only", created_utc=now(),
        repository=str(repo), source_inventory=dict(path=str(inventory_path), sha256=sha(inventory_path)),
        source_sha256={str(p): sha(p) for p in source_paths}, cases=rows,
        inclusion="All main normal-noise N288 reference 4 cases x 10 bundles; no outcome filtering",
        expected_cases=40, expected_observation_files=80,
        bounded_operations=dict(max_sources=40, max_observed_PQ_files=80,
            planned_pair_records=sum(r["terminal_count"]*(r["terminal_count"]-1)//2 for r in rows)*14,
            views=2, fixed_windows=7, directions=2, power_channels=2),
        physical_power_base_kw=POWER_BASE_KW,
        power_base_basis="Producer source declares base_mva=.2 for paper15 and the case bank; no graph read",
        parameters=dict(sigma_PQ=.005, numerical_power_floor_kw=1e-9,
            description_levels=[1., 2.], history_views=["training_3_days", "all_available_6_days"],
            windows=["all_day"] + [f"hours_{s:02d}_{s+4:02d}" for s in range(0,24,4)]),
        method="zero-intercept nonnegative proportional LS in both label directions; physical RMS and RMS-normalized residual",
        noise_reference="sigma*sqrt(mean(x_observed^2+alpha^2*y_observed^2)); observed plug-in, independent multiplicative error assumption",
        normalization_boundary="Numerical floor is not an empirical meter resolution. Constant/near-zero/nonpositive fits are explicitly unclassified; no clipping, interpolation, whitening or loss reweighting",
        threshold_boundary="1x/2x are descriptive counts, not confidence tests, structural certificates or a price objective",
        candidate_projection=dict(status="not_available", validation_selected_candidates_read=False,
            boundary="No verified train-only whitelist projection supplied; do not open existing result JSON. Output observed input-pair candidates only"),
        allowed_fields=["labels", "P", "Q"], forbidden_value_fields=["V", "root", "test", "network", "truth", "final_errors"],
        all_available_history_rule="Both training and previous validation are available past in the old contract; all-six-day diagnosis is primary for a purchasing gap, not only the first three days",
        development_scope=dict(
            synthetic="Existing 40 conditions are already-developed synthetic case-bank histories, not field or external validation",
            independent_units="10 original bundle blocks; cases paired within bundles, not 40 independent field days",
            iflex="No household file read. Existing train ends 2021-02-07; Feb9-22 and March tests have been seen; whole_day_evaluation_bank mixes splits",
            controls="Retain both b5 zero-gain controls and old flynn b7 without reopening their price batches"),
        counts=dict(NPZ_values_loaded=0, topology_truth_read=0, test_trajectories_read=0,
            household_files_read=0, LP_calls=0, QP_calls=0, identifier_calls=0, AC_calls=0),
        run_policy="Separate run command after review; retain every source, failure/unresolved and completed checkpoint; no automatic retry",
        public_data_policy="Only configuration/aggregates exported; P/Q and detailed pair metrics private",
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(), numpy=np.__version__),
    )
    write(root/"protocol.json", protocol)
    write(root/"protocol.sha256.json", dict(sha256=sha(root/"protocol.json")))
    receipt = dict(status="prepared_not_run", protocol_sha256=sha(root/"protocol.json"),
        sources=40, metadata_ready=sum(r["metadata_ready"] for r in rows),
        NPZ_value_reads=0, numerical_statistics_run=False, source_sha256=sha(__file__))
    write(root/"preparation_receipt.json", receipt)
    return receipt


def load_pq(source, terminal_count):
    path = Path(source["path"])
    if not source["catalog_hash_matches"] or sha(path) != source["sha256"]:
        raise ValueError("frozen observed source unavailable or hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"labels", "P", "Q", "V", "root"}:
            raise ValueError("unexpected observed-only archive schema")
        labels = archive["labels"]
        if labels.shape != (terminal_count,) or not np.isfinite(labels).all() or not np.array_equal(labels, labels.astype(int)):
            raise ValueError("invalid observed terminal labels")
        labels = labels.astype(int).tolist()
        if len(set(labels)) != terminal_count:
            raise ValueError("duplicate terminal labels")
        p, q = np.array(archive["P"], dtype=float), np.array(archive["Q"], dtype=float)
    expected = (3, SLOTS, terminal_count)
    if p.shape != expected or q.shape != expected or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("expected complete finite 3x96 observed P/Q; no row deletion")
    return labels, p*POWER_BASE_KW, q*POWER_BASE_KW


def run_case(root, row, protocol):
    case_dir = root/"cases"/row["job_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    detail = case_dir/"pair_diagnostics.jsonl.gz"
    partial = detail.with_suffix(detail.suffix + ".partial")
    if detail.exists() or partial.exists():
        raise FileExistsError("prior detailed attempt retained; do not rerun an interrupted case")
    labels, pt, qt = load_pq(row["sources"]["train"], row["terminal_count"])
    labels_v, pv, qv = load_pq(row["sources"]["validation"], row["terminal_count"])
    if labels_v != labels:
        raise ValueError("terminal order changes across historical roles")
    params = protocol["parameters"]
    windows = []
    with gzip.open(partial, "wt", encoding="utf-8") as stream:
        for view, p, q in (
            ("training_3_days", pt, qt),
            ("all_available_6_days", np.concatenate((pt,pv)), np.concatenate((qt,qv))),
        ):
            days = len(p)
            p, q = p.reshape(-1,len(labels)), q.reshape(-1,len(labels))
            for window, mask in window_masks(days):
                counts = empty_counts()
                for i in range(len(labels)):
                    for j in range(i+1,len(labels)):
                        record = pair_record(p[mask], q[mask], i, j, labels,
                            sigma=params["sigma_PQ"], power_floor=params["numerical_power_floor_kw"])
                        add_counts(counts, record)
                        stream.write(json.dumps(dict(history_view=view, window=window, **record),
                            separators=(",",":"), ensure_ascii=True, allow_nan=False) + "\n")
                windows.append(dict(history_view=view, window=window,
                    observed_rows=int(mask.sum()), counts=counts))
    partial.replace(detail)
    return dict(status="audited", job_id=row["job_id"], case=row["case"], bundle=row["bundle"],
        terminal_count=len(labels), terminal_labels=labels, windows=windows,
        historical_rows=dict(training=288, all_available=576),
        detail_path=str(detail), detail_sha256=sha(detail),
        candidate_association="observed input pairs only; no train-only structural projection supplied",
        existing_result_contents_read=False, topology_truth_read=False, test_values_read=False,
        household_files_read=False, P_Q_value_files_read=2,
        physical_units="kW/kvar on producer 200 kVA base; negative net injections retained")


def run(root, *, resume=False):
    root = Path(root).resolve()
    protocol_path = root/"protocol.json"
    protocol = read(protocol_path)
    expected = read(root/"protocol.sha256.json")["sha256"]
    if sha(protocol_path) != expected:
        raise ValueError("protocol changed")
    for name, digest in protocol["source_sha256"].items():
        if sha(name) != digest:
            raise ValueError("source drift: " + name)
    inventory = protocol["source_inventory"]
    if sha(inventory["path"]) != inventory["sha256"]:
        raise ValueError("source inventory changed")
    for row in protocol["cases"]:
        for source in row["sources"].values():
            if source["catalog_hash_matches"]:
                path = Path(source["path"])
                if not path.is_file() or path.stat().st_size != source["bytes"] or sha(path) != source["sha256"]:
                    raise ValueError("frozen observation bytes changed; completed records must not be silently reused")
    if (root/"ended.json").exists():
        raise FileExistsError("completed history audit retained; do not rerun")
    if (root/"progress.json").exists() and not resume:
        raise FileExistsError("progress exists; explicit --resume required")
    lock = root/"audit.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(started_utc=now(), protocol_sha256=expected), stream)
    records = []
    try:
        for row in protocol["cases"]:
            directory = root/"cases"/row["job_id"]
            marker = directory/"case_summary.json"
            if marker.exists():
                if not resume:
                    raise FileExistsError("case record exists; explicit --resume required")
                record = read(marker)
                if record["protocol_sha256"] != expected:
                    raise ValueError("old case protocol differs")
                if record["status"] == "audited" and sha(record["detail_path"]) != record["detail_sha256"]:
                    raise ValueError("completed detail changed")
            else:
                try:
                    record = run_case(root, row, protocol)
                except Exception as exc:
                    directory.mkdir(parents=True, exist_ok=True)
                    record = dict(status="failed_or_unresolved", job_id=row["job_id"],
                        case=row["case"], bundle=row["bundle"], exception_type=type(exc).__name__,
                        message=str(exc), traceback=traceback.format_exc(),
                        test_values_read=False, topology_truth_read=False,
                        failed_attempt_not_retried=True)
                record["protocol_sha256"] = expected
                write(marker, record)
            records.append(record)
            write(root/"progress.json", dict(protocol_sha256=expected, planned=40, records=records))
        result = dict(status="bounded_audit_finished", protocol_sha256=expected,
            source_sha256=sha(__file__), planned=40, recorded=len(records),
            audited=sum(r["status"] == "audited" for r in records),
            failed_or_unresolved=sum(r["status"] != "audited" for r in records),
            cases=records, no_case_or_price_selected=True,
            interpretation="Descriptive observed input diagnostics, not recovery or identifiability evidence",
            LP_calls=0, QP_calls=0, identifier_calls=0, AC_calls=0,
            test_values_read=0, topology_truth_read=0, household_files_read=0)
        write(root/"ended.json", result)
        return {key:result[key] for key in ("status","planned","recorded","audited","failed_or_unresolved")}
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--repo", type=Path, default=REPO)
    prep.add_argument("--inventory", type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("--root", type=Path, required=True)
    execute.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        inventory = args.inventory or args.repo/"research_experiments/wzzt_dr_selection/evidence/scenario_review_v1/source_inventory.json"
        result = prepare(args.root, inventory, args.repo)
    else:
        result = run(args.root, resume=args.resume)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
