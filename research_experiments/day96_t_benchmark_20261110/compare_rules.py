"""Compare saved historical selection/output contracts without new fitting.

Requires all native evaluations and the enriched DR bridge to be verified.
Selection is completed from saved validation losses before truth is read.
View 2 changes only the selection rule on the saved native output library;
view 3 changes the library, selection rule and topology map together and is
explicitly not a causal decomposition of those changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import candidates as contract
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth


ROOT = Path(__file__).resolve().parents[2]
VIEWS = ("native_strict24_refit", "same_library_band3", "enriched_band3_map16")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def positive_topology(point, n):
    return frozenset(tuple(C) for C, r, x in zip(point["supports"], point["r"], point["x"])
                     if 1 < len(C) < n and max(r, x) > 1e-9)


def selection_on_saved_output_library(rows, *, band_fraction):
    """Pick among the exact supplied fitted points; do not contract/refit.

    Count all supports in the supplied point, matching the declared neighbor
    style count. This is not the size of a pre-.24 raw path ancestor.
    """
    if not rows:
        raise ValueError("Saved native output library is empty")
    best = min(float(row["validation"]) for row in rows)
    eligible = [row for row in rows if float(row["validation"]) <= (1 + band_fraction) * best + 1e-12]
    if band_fraction == 0:
        # The native strict selector uses proper internal support count.
        return min(eligible, key=lambda row: (row["internal"], row["validation"], row["index"]))
    return min(eligible, key=lambda row: (len(row["point"]["supports"]), row["validation"], row["index"]))


def selection_on_enriched_library(export):
    rows = export["candidates"]
    best = min(float(row["validation_mae"]) for row in rows)
    eligible = [row for row in rows if float(row["validation_mae"]) <= 1.03 * best + 1e-12]
    chosen = min(eligible, key=lambda row: (len(row["point"]["supports"]), row["validation_mae"], row["index"]))
    declared = export["selections"]["neighbor_style_band3"]
    if chosen["candidate_id"] != declared["candidate_id"] or chosen["index"] != declared["index"]:
        raise AssertionError("Enriched pool neighbor-style selector does not replay")
    if export["output_mapping"]["resolution"] != .16 or export["output_mapping"]["refit"] is not False:
        raise AssertionError("View 3 requires the explicitly frozen .16/no-refit topology map")
    return chosen


def prepared_views(result, enriched):
    """Pure historical selection; this API has no truth or new-data input."""
    selected = result["selected"]
    rows = result["output_candidates"]
    native = selection_on_saved_output_library(rows, band_fraction=0.)
    if int(native["index"]) != int(selected["index"]):
        raise AssertionError("Native strict selection does not replay on its output library")
    canonical_native = contract._point(selected["point"], len(enriched["labels"]))
    if canonical_native != contract._point(native["point"], len(enriched["labels"])):
        raise AssertionError("Replayed native selection has different fitted coefficients")
    anchor = contract.digest(canonical_native)
    if enriched["mainline_selected_candidate_id"] != anchor:
        raise AssertionError("Enriched export anchor differs from the actual native selected coefficients")
    by_id = {row["candidate_id"]: row for row in enriched["candidates"]}
    if anchor not in by_id or contract._point(by_id[anchor]["point"], len(enriched["labels"])) != canonical_native:
        raise AssertionError("Enriched export does not contain the unchanged actual native predictor")
    band3 = selection_on_saved_output_library(rows, band_fraction=.03)
    other = selection_on_enriched_library(enriched)
    n = len(enriched["labels"])
    return [dict(view=VIEWS[0], point=selected["point"], selected_index=int(selected["index"]),
                 topology=positive_topology(selected["point"], n), validation=float(selected["validation"]),
                 selection_rule="strict validation MAE", library="saved native .24-contracted/refitted output library",
                 topology_map="saved .24/refitted point positive topology", changed_components="none"),
            dict(view=VIEWS[1], point=band3["point"], selected_index=int(band3["index"]),
                 topology=positive_topology(band3["point"], n), validation=float(band3["validation"]),
                 selection_rule="3% validation band; all supplied fitted supports, validation, original index",
                 library="same saved native .24-contracted/refitted output library",
                 topology_map="same saved fitted point; no additional contraction/refit",
                 changed_components="selection rule only"),
            dict(view=VIEWS[2], point=other["point"], selected_index=int(other["index"]),
                 topology=frozenset(tuple(C) for C in other["output_topology"]), validation=float(other["validation_mae"]),
                 selection_rule="enriched neighbor_style_band3",
                 library="enriched raw-path plus saved native fitted output models",
                 topology_map="frozen .16 threshold on selected saved coefficients; no refit",
                 changed_components="library + selection rule + output map; not a causal component decomposition")]


def compare(output):
    started = perf_counter()
    output = Path(output).resolve()
    protocol_file, catalog_file = output / "protocol.json", output / "catalog.json"
    protocol, catalog = read(protocol_file), read(catalog_file)
    methods, jobs = [row["id"] for row in protocol["methods"]], protocol["jobs"]
    planned = len(jobs) * len(methods)
    if planned != protocol["planned"] or planned != 144:
        raise ValueError("This comparison expects all 144 declared native evaluations")
    audit_file = output / "analysis/verification.json"
    audit = read(audit_file)
    if not (audit.get("passed") is True and audit.get("complete") is True and audit.get("verified") == planned):
        raise RuntimeError("Native independent post-analysis must verify the complete batch first")
    if audit["protocol_sha256"] != sha(protocol_file):
        raise AssertionError("Native audit references a different protocol")
    if audit["analyzer_sha256"] != sha(Path(__file__).with_name("analyze.py")):
        raise AssertionError("Current native analyzer differs from the version that verified this batch")
    if protocol.get("catalog_sha256") != sha(catalog_file):
        raise AssertionError("Catalog differs from the frozen protocol")
    for name, digest in protocol["source_hashes"].items():
        if sha(output / "source" / name) != digest:
            raise AssertionError("Frozen optimizer source changed: " + name)
    bridge_file = output / "bridge_candidates/summary.json"
    bridge = read(bridge_file)
    if (bridge.get("native_results_complete") is not True or bridge.get("exports_verified") != planned
            or bridge.get("requested_exports") != planned or bridge.get("anchor_missing_after") != 0):
        raise RuntimeError("All 144 enriched bridge exports with unchanged native anchors are required")
    expected_pairs = {(job, method) for job in jobs for method in methods}
    bridge_rows = {(row["job"], row["method"]): row for row in bridge["receipts"]}
    if set(bridge_rows) != expected_pairs or len(bridge["receipts"]) != planned:
        raise AssertionError("Bridge summary does not cover the unique expected job/method pairs")
    for name, digest in bridge["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise AssertionError("Bridge post-processing source changed: " + name)
    pending, provenance, missing_before, missing_after = [], {}, 0, 0
    # Finish all selection decisions before opening any true network.
    for job in jobs:
        meta = catalog[job]
        expected_hashes = protocol["input_hashes"][job]
        if {key: sha(meta[key]) for key in ("train", "validation", "test")} != expected_hashes:
            raise AssertionError("Frozen observation files changed: " + job)
        for method in methods:
            folder = output / "jobs" / job / method
            result_file, raw_file = folder / "result.json", folder / "candidates.json"
            enriched_file = output / "bridge_candidates" / job / (method + ".json")
            result, raw, enriched = read(result_file), read(raw_file), read(enriched_file)
            if result["input_hashes"] != {key: expected_hashes[key] for key in ("train", "validation")}:
                raise AssertionError("Native result inputs do not match frozen observed files")
            if result["candidate_export_sha256"] != sha(raw_file):
                raise AssertionError("Original worker candidate export changed")
            references = enriched["provenance"]["bridge_postprocessing"]
            expected_refs = dict(native_result_sha256=sha(result_file), native_raw_export_sha256=sha(raw_file),
                                 native_protocol_sha256=sha(protocol_file), native_analysis_receipt_sha256=sha(audit_file))
            if any(references[key] != value for key, value in expected_refs.items()):
                raise AssertionError("Enriched bridge references different native inputs")
            contract.verify_export(raw)
            contract.verify_export(enriched)
            if enriched["labels"] != meta["labels"] or enriched["labels"] != raw["labels"]:
                raise AssertionError("Terminal order differs between native and enriched models")
            if any(result.get(key) is not False for key in ("truth_used", "test_opened", "final_RNJ_fallback")):
                raise AssertionError("Native result has incompatible provenance")
            receipt = enriched["bridge_receipt"]
            anchor = enriched["mainline_selected_candidate_id"]
            old_ids = {row["candidate_id"] for row in raw["candidates"]}
            new_ids = {row["candidate_id"] for row in enriched["candidates"]}
            absent_before, absent_after = anchor not in old_ids, anchor not in new_ids
            if receipt["anchor_was_missing_from_raw_pool"] != absent_before or not receipt["anchor_in_pool"]:
                raise AssertionError("Bridge anchor missing-count receipt does not replay")
            if receipt["mainline_selected_candidate_id"] != anchor or not old_ids <= new_ids:
                raise AssertionError("Bridge changed its native anchor or discarded raw models")
            summary_receipt = bridge_rows[(job, method)]
            if any(summary_receipt[key] != receipt[key] for key in ("mainline_selected_candidate_id", "anchor_was_missing_from_raw_pool", "anchor_in_pool")):
                raise AssertionError("Bridge summary differs from the per-export anchor receipt")
            missing_before += int(absent_before)
            missing_after += int(absent_after)
            views = prepared_views(result, enriched)
            native_topology = views[0]["topology"]
            native_point = contract._point(views[0]["point"], len(meta["labels"]))
            native_parameters = contract.digest(native_point)
            native_r, native_x = contract.response_matrices(native_point, len(meta["labels"]))
            for view in views:
                point = contract._point(view["point"], len(meta["labels"]))
                r, x = contract.response_matrices(point, len(meta["labels"]))
                pending.append(dict(job=job, case=meta["case"], bundle=meta["bundle"],
                                    voltage_sigma=meta["voltage_sigma"], method=method, view=view["view"],
                                    topology=view["topology"], native_topology=native_topology,
                                    selected_index=view["selected_index"], historical_validation_MAE=view["validation"],
                                    support_count=len(point["supports"]), positive_internal_count=len(positive_topology(point, len(meta["labels"]))),
                                    output_topology_count=len(view["topology"]),
                                    topology_changed_vs_native=view["topology"] != native_topology,
                                    predictor_changed_vs_native=contract.digest(point) != native_parameters,
                                    response_matrices_changed_vs_native=not (np.array_equal(r, native_r) and np.array_equal(x, native_x)),
                                    candidate_parameter_sha256=contract.digest(point), native_anchor_id=anchor,
                                    selection_rule=view["selection_rule"], library=view["library"],
                                    topology_map=view["topology_map"], changed_components=view["changed_components"],
                                    selection_uses_truth=False, coefficients_refitted=False,
                                    native_method_seconds=result["total_seconds"],
                                    interpretation="post-analysis of historical selection/output rules; no DR response was simulated"))
            provenance[job + "/" + method] = dict(native_result_sha256=sha(result_file),
                                raw_export_sha256=sha(raw_file), enriched_export_sha256=sha(enriched_file))
    if (missing_before, missing_after) != (bridge["anchor_missing_before"], bridge["anchor_missing_after"]):
        raise AssertionError("Aggregate bridge anchor inclusion counts differ")
    if len(pending) != planned * len(VIEWS):
        raise AssertionError("Compatibility views lost their planned denominator")
    truth_by_job, truth_hashes = {}, {}
    for job in jobs:
        meta = catalog[job]
        prior = read(output / "analysis/jobs" / job / "verified.json")
        network = Path(meta["truth_network"])
        hashes = {name: sha(network / name) for name in ("metadata.json", "buses.csv", "branches.csv")}
        if hashes != prior["truth_source_hashes"]:
            raise AssertionError("Truth source differs from native audit")
        truth_hashes[job] = hashes
        physical = physical_truth(network, meta["labels"])
        index = {label: i for i, label in enumerate(meta["labels"])}
        truth_by_job[job] = frozenset(tuple(sorted(index[label] for label in C)) for C in physical["clades"])
    rows = []
    for row in pending:
        truth = truth_by_job[row["job"]]
        predicted, baseline = row.pop("topology"), row.pop("native_topology")
        tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
        exact, native_exact = predicted == truth, baseline == truth
        row.update(clade_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1., exact=exact,
                   true_positive=tp, false_positive=fp, false_negative=fn, true_count=len(truth),
                   native_exact=native_exact, new_exact=exact and not native_exact, lost_exact=native_exact and not exact,
                   predicted_topology=json.dumps(sorted(map(list, predicted))),
                   historical_predictions_reused=True)
        rows.append(row)
    frame = pd.DataFrame(rows)
    summaries = []
    for (method, view), group in frame.groupby(["method", "view"], sort=False):
        native = frame[(frame.method == method) & (frame.view == VIEWS[0])].set_index("job")
        paired = group.set_index("job").reindex(native.index)
        summaries.append(dict(method=method, view=view, planned=len(jobs), completed=len(group),
                              F1=float(group.clade_f1.mean()), exact=int(group.exact.sum()), recovery_rate=float(group.exact.mean()),
                              delta_F1_vs_native=float((paired.clade_f1 - native.clade_f1).mean()),
                              delta_exact_vs_native=int(paired.exact.sum() - native.exact.sum()),
                              changed_vs_native=int(group.topology_changed_vs_native.sum()),
                              predictor_changed_vs_native=int(group.predictor_changed_vs_native.sum()),
                              new_exact=int(group.new_exact.sum()), lost_exact=int(group.lost_exact.sum())))
    summary = pd.DataFrame(summaries)
    native_table = pd.read_csv(output / "analysis/per_model.csv")
    for row in frame[frame.view == VIEWS[0]].itertuples(index=False):
        match = native_table[(native_table.job == row.job) & (native_table.method == row.method)]
        if len(match) != 1 or abs(float(match.iloc[0].clade_f1) - row.clade_f1) > 1e-12 or bool(match.iloc[0].exact) != row.exact:
            raise AssertionError("Native compatibility anchor metrics do not reproduce the independent audit")
    target = output / "compatibility_analysis"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "per_model.csv", index=False, lineterminator="\n")
    summary.to_csv(target / "summary.csv", index=False, lineterminator="\n")
    verification = dict(passed=True, complete=True, native_evaluations=planned, view_rows=len(rows),
                        views=list(VIEWS), native_anchor_coefficients_unchanged=True,
                        native_anchor_metrics_reproduced=True, anchor_missing_before=missing_before, anchor_missing_after=missing_after,
                        all_enriched_pools_include_actual_native_selected_predictor=missing_after == 0,
                        native_audit_passed=True, frozen_source_hashes_verified=True, input_hashes_verified=True,
                        selection_uses_truth=False, truth_opened_only_after_all_historical_selections=True,
                        new_fit_calls=0, new_search_calls=0, new_optimizer_calls=0, new_DR_runs=0,
                        view2_changed_components="selection rule only; same saved .24/refit output library and coefficients",
                        view3_changed_components="enriched library +3% rule +.16/no-refit map; combined policy view, not component attribution",
                        native_protocol_sha256=sha(protocol_file), native_analysis_receipt_sha256=sha(audit_file),
                        bridge_summary_sha256=sha(bridge_file), script_sha256=sha(__file__),
                        contract_sha256=sha(Path(contract.__file__)), native_inputs=provenance,
                        truth_source_hashes=truth_hashes, seconds=perf_counter() - started)
    write(target / "verification.json", verification)
    print(summary.to_string(index=False))
    print(json.dumps({k: v for k, v in verification.items() if k not in {"native_inputs", "truth_source_hashes"}}, ensure_ascii=False))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "outputs/day96_t_benchmark_20261110")
    args = parser.parse_args()
    compare(args.output)
