"""Enrich completed worker candidate exports with already-fitted output models.

This is post-processing, not another optimizer. The native mainline applies
.24 contraction and a historical LP refit before selection; its selected model
need not occur on the raw forward path. Preserve both sets of fitted predictors
and explicitly anchor the actual selected one for downstream frozen-model DR.
Original worker files, method costs and frozen optimizer sources are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE_FILE = Path(__file__).with_name("candidates.py")
_spec = importlib.util.spec_from_file_location("day96_live_candidate_contract", MODULE_FILE)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def observed_arrays(path):
    """Read observed P/Q/V/root only; never open truth or test observations."""
    with np.load(Path(path), allow_pickle=False) as archive:
        labels = archive["labels"].astype(int).tolist()
        n = len(labels)
        p, q, v = (np.asarray(archive[key], float).reshape(-1, n) for key in ("P", "Q", "V"))
        root = np.asarray(archive["root"], float).reshape(-1)
    if p.shape != q.shape or p.shape != v.shape or len(root) != len(p):
        raise ValueError("Observed historical arrays do not align.")
    return labels, dict(p=p, q=q, y=root[:, None]**2 - v**2)


def enrich_candidates(result, raw_export, *, labels, train, validation,
                      provenance=None, root_label=None, base_kw=None):
    """Return a new full export and audit receipt using saved fitted weights.

    Provenance additions identify the saved result and original export. The raw
    pool's training-only assertions remain required by the candidate contract.
    Each output row and the native selected point are added without refitting.
    """
    contract.verify_export(raw_export, train=train, validation=validation)
    if list(labels) != raw_export["labels"]:
        raise ValueError("Historical labels differ from the original frozen pool.")
    for field in ("truth_used", "test_opened", "final_RNJ_fallback"):
        if result.get(field) is not False:
            raise ValueError(f"Native result must explicitly declare {field}=False.")
    selected = result["selected"]
    if selected.get("source") not in {"own_path", "singleton"}:
        raise ValueError("The native selected model is not from its own search library.")
    provenance_full = dict(raw_export["provenance"])
    if provenance is not None:
        provenance_full["bridge_postprocessing"] = dict(provenance)
    points, raw_parent = [], {}
    for row in raw_export["candidates"]:
        for origin in row["origins"]:
            wrapper = dict(origin, point=row["point"], parent_candidate=row["candidate_id"])
            points.append(wrapper)
            raw_parent[origin.get("path_index")] = row["candidate_id"]
    for index, point in enumerate(result["trajectory"]["path"]):
        points.append(dict(point=point, source="own_path", stage="native_raw_path",
                           path_index=index, parent_candidate=raw_parent.get(index)))
    outputs = result["output_candidates"]
    if not outputs:
        raise ValueError("A completed native result must contain its fitted output library.")
    for i, row in enumerate(outputs):
        index = row.get("index", i)
        source = row.get("source")
        if source not in {"own_path", "singleton"}:
            raise ValueError("An external model entered the native output library.")
        points.append(dict(point=row["point"], source=source,
                           stage="output_contract_refit_0.24", path_index=index,
                           parent_candidate=raw_parent.get(index), event_index=i))
    points.append(dict(point=selected["point"], source=selected["source"],
                       stage="mainline_selected_contract_refit_0.24",
                       path_index=selected.get("index"),
                       parent_candidate=raw_parent.get(selected.get("index"))))
    calibration = raw_export["output_mapping"]["calibration"]
    enriched = contract.export_candidates(points, labels=labels, train=train, validation=validation,
                     provenance=provenance_full, calibration=calibration,
                     root_label=raw_export["root_label"] if root_label is None else root_label,
                     base_kw=raw_export["base_kw"] if base_kw is None else base_kw,
                     contraction_resolution=.16, band_fraction=.03)
    anchor_parameters = contract._point(selected["point"], len(labels))
    anchor = contract.digest(anchor_parameters)
    by_id = {row["candidate_id"]: row for row in enriched["candidates"]}
    if anchor not in by_id:
        raise AssertionError("The actual native selected predictor is missing from the enriched pool.")
    old_ids = {row["candidate_id"] for row in raw_export["candidates"]}
    if not old_ids <= set(by_id):
        raise AssertionError("Enrichment must retain every original raw predictor.")
    anchor_row = by_id[anchor]
    native_validation = float(selected.get("validation", selected.get("validation_mae", np.nan)))
    if not np.isfinite(native_validation) or abs(native_validation-anchor_row["validation_mae"]) > 1e-10:
        raise ValueError("The native selected validation score does not replay.")
    for key in ("supports", "r", "x"):
        if anchor_parameters[key] != contract._point(anchor_row["point"], len(labels))[key]:
            raise AssertionError("The native anchor coefficients changed during export.")
    receipt = dict(original_raw_candidates=len(raw_export["candidates"]),
                   native_fitted_output_rows=len(outputs), enriched_candidates=len(enriched["candidates"]),
                   mainline_selected_candidate_id=anchor,
                   anchor_was_missing_from_raw_pool=anchor not in old_ids,
                   anchor_in_pool=True, all_original_raw_models_retained=True,
                   original_method_total_seconds=result.get("total_seconds"),
                   new_fit_calls=0, new_search_calls=0, truth_opened=False, test_opened=False,
                   native_selected_validation_mae=native_validation,
                   raw_pool_sha256=raw_export["pool_sha256"],
                   raw_export_selections=raw_export["selections"],
                   enriched_export_selections=enriched["selections"],
                   native_anchor_vs_export_rules={name: row["candidate_id"] == anchor
                                                  for name, row in enriched["selections"].items()},
                   selection_rules_equivalent_to_native_mainline=False,
                   semantics="Adds saved .24-contracted/refitted output predictors; .16 topology map and 3% export views remain separate from native selection.")
    enriched.update(export_kind="raw_and_native_output_history_fitted_models",
                    mainline_selected_candidate_id=anchor,
                    native_mainline_selection={key: value for key, value in selected.items() if key != "point"},
                    mainline_anchor_topology=anchor_row["positive_topology"],
                    native_mainline_output_rule=dict(resolution=.24, refit=True, selection="strict validation MAE"),
                    original_raw_pool_sha256=raw_export["pool_sha256"],
                    bridge_receipt=receipt)
    enriched["pool_sha256"] = contract.digest({k: v for k, v in enriched.items() if k != "pool_sha256"})
    verified = contract.verify_export(enriched, train=train, validation=validation)
    return enriched, receipt, verified


def enrich_completed(output, *, jobs=None, methods=None):
    output = Path(output).resolve()
    protocol, catalog = read(output/"protocol.json"), read(output/"catalog.json")
    all_methods = [m["id"] for m in protocol["methods"]]
    missing = [(job, method) for job in protocol["jobs"] for method in all_methods
               if not (output/"jobs"/job/method/"result.json").is_file()]
    if missing:
        raise RuntimeError(f"Bridge post-processing waits for all {protocol['planned']} native results; {len(missing)} missing.")
    verification_path = output/"analysis"/"verification.json"
    if not verification_path.is_file():
        raise RuntimeError("Bridge post-processing waits for completed native post-analysis verification.")
    native_verification = read(verification_path)
    if native_verification.get("passed") is not True or native_verification.get("complete") is not True:
        raise RuntimeError("Native post-analysis has not verified the complete planned result set.")
    chosen_jobs = protocol["jobs"] if jobs is None else jobs
    chosen_methods = all_methods if methods is None else methods
    if any(j not in protocol["jobs"] for j in chosen_jobs) or any(m not in all_methods for m in chosen_methods):
        raise ValueError("Unknown job or method filter.")
    started = perf_counter()
    receipts, written, reused = [], 0, 0
    source_hashes = {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                     str(MODULE_FILE.relative_to(ROOT)): sha(MODULE_FILE)}
    for job in chosen_jobs:
        meta = catalog[job]
        for method in chosen_methods:
            then = perf_counter()
            folder = output/"jobs"/job/method
            result_path, raw_path = folder/"result.json", folder/"candidates.json"
            target = output/"bridge_candidates"/job/(method+".json")
            result, raw = read(result_path), read(raw_path)
            references = dict(job=job, method=method, native_result_sha256=sha(result_path),
                              native_raw_export_sha256=sha(raw_path),
                              source_hashes=source_hashes, native_protocol_sha256=sha(output/"protocol.json"),
                              native_analysis_receipt_sha256=sha(verification_path))
            if result["candidate_export_sha256"] != references["native_raw_export_sha256"]:
                raise ValueError("The original worker candidate file changed.")
            labels, train = observed_arrays(meta["train"])
            val_labels, validation = observed_arrays(meta["validation"])
            if labels != val_labels or labels != meta["labels"]:
                raise ValueError("Catalog, train and validation terminal orders differ.")
            actual_hashes = {key: sha(meta[key]) for key in ("train", "validation")}
            if actual_hashes != result["input_hashes"] or actual_hashes != raw["provenance"]["input_hashes"]:
                raise ValueError("Historical input file hashes changed.")
            if target.exists():
                existing = read(target)
                if existing["provenance"].get("bridge_postprocessing") != references:
                    raise FileExistsError("Existing enriched export has different native inputs or bridge source.")
                contract.verify_export(existing, train=train, validation=validation)
                receipts.append(existing["bridge_receipt"])
                reused += 1
                continue
            enriched, receipt, verified = enrich_candidates(result, raw, labels=labels, train=train,
                              validation=validation, provenance=references,
                              root_label=meta["root_label"], base_kw=meta["base_kw"])
            receipt.update(job=job, method=method,
                           postprocessing_seconds_excluding_file_write=perf_counter()-then,
                           export_response_and_loss_verification={key: value for key, value in verified.items()
                                                                  if key != "complete_payload_sha256"})
            enriched["bridge_receipt"] = receipt
            enriched["pool_sha256"] = contract.digest({k: v for k, v in enriched.items() if k != "pool_sha256"})
            contract.verify_export(enriched, train=train, validation=validation)
            write(target, enriched)
            receipts.append(receipt)
            written += 1
    summary = dict(planned_native_evaluations=protocol["planned"], native_results_complete=True,
                   requested_exports=len(chosen_jobs)*len(chosen_methods), exports_verified=len(receipts),
                   written=written, reused=reused,
                   anchor_missing_before=sum(r["anchor_was_missing_from_raw_pool"] for r in receipts),
                   anchor_missing_after=sum(not r["anchor_in_pool"] for r in receipts),
                   original_raw_pool_size_sum=sum(r["original_raw_candidates"] for r in receipts),
                   enriched_pool_size_sum=sum(r["enriched_candidates"] for r in receipts),
                   newly_written_postprocessing_seconds=sum(r.get("postprocessing_seconds_excluding_file_write",0.) for r in receipts) if not reused else None,
                   total_bridge_wall_seconds=perf_counter()-started,
                   native_method_costs_modified=False, original_worker_exports_modified=False,
                   new_fit_calls=0, new_search_calls=0, truth_opened=False, test_opened=False,
                   source_hashes=source_hashes, receipts=receipts)
    write(output/"bridge_candidates"/"summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="receipts"}, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default=str(ROOT/"outputs/day96_t_benchmark_20261110"))
    parser.add_argument("--job", action="append")
    parser.add_argument("--method", action="append")
    args = parser.parse_args()
    enrich_completed(args.output, jobs=args.job, methods=args.method)


if __name__ == "__main__":
    main()
