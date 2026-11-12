"""Post-hoc candidate coverage and predictor-identity diagnostics; no fitting.

Truth is opened only after all 144 native-result and bridge hash chains pass.
Oracle maxima summarize an existing finite library and are not deployable
selectors. Matrix tolerances describe numerical proximity, not exact identity.
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
from research_experiments.rnj import audit_source_ac96_benchmark as truth_contract


ROOT = Path(__file__).resolve().parents[2]
POOLS = ("raw_positive", "native_output24", "raw_plus_output24")
TOLERANCES = (0., 1e-15, 1e-12, 1e-10, 1e-8, 1e-6)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8", newline="\n")


def topology(point, n):
    return frozenset(tuple(sorted(C)) for C, r, x in zip(point["supports"], point["r"], point["x"])
                     if 1 < len(C) < n and max(r, x) > 1e-9)


def f1(predicted, truth):
    denominator = len(predicted)+len(truth)
    return 2*len(predicted & truth)/denominator if denominator else 1.


def matrix_distance(R, X, native_R, native_X):
    dr, dx = np.asarray(R)-native_R, np.asarray(X)-native_X
    return dict(exact_matrix_equal=bool(np.array_equal(R, native_R) and np.array_equal(X, native_X)),
                joint_entry_maxabs=float(max(np.max(np.abs(dr)), np.max(np.abs(dx)))),
                R_entry_maxabs=float(np.max(np.abs(dr))), X_entry_maxabs=float(np.max(np.abs(dx))),
                joint_induced_inf=float(max(np.max(np.sum(np.abs(dr), axis=1)),
                                             np.max(np.sum(np.abs(dx), axis=1)))))


def pool_diagnostic(models, truth):
    union = frozenset().union(*models)
    return dict(models=len(models), distinct_topologies=len(set(models)),
                whole_true_tree_covered=any(model == truth for model in models),
                oracle_max_possible_F1=max(f1(model, truth) for model in models),
                true_connection_union_recall=len(union & truth)/len(truth) if truth else 1.,
                all_true_connections_in_union=truth <= union)


def diagnose(output):
    started = perf_counter(); output = Path(output).resolve()
    protocol_path, catalog_path = output/"protocol.json", output/"catalog.json"
    protocol, catalog = read(protocol_path), read(catalog_path)
    methods, jobs = [m["id"] for m in protocol["methods"]], protocol["jobs"]
    planned = len(methods)*len(jobs)
    if planned != 144 or len(jobs) != 48 or protocol["planned"] != planned:
        raise ValueError("This diagnostic requires the complete fixed 144-evaluation panel.")
    audit_path, bridge_path = output/"analysis/verification.json", output/"bridge_candidates/summary.json"
    audit, bridge = read(audit_path), read(bridge_path)
    if not (audit.get("passed") is True and audit.get("complete") is True and audit.get("verified") == planned):
        raise RuntimeError("All 144 native evaluations must pass independent numerical verification first.")
    protocol_hash, audit_hash = sha(protocol_path), sha(audit_path)
    if audit["protocol_sha256"] != protocol_hash or sha(catalog_path) != protocol["catalog_sha256"]:
        raise ValueError("Native audit or catalog references a different protocol.")
    if audit["analyzer_sha256"] != sha(Path(__file__).with_name("analyze.py")):
        raise ValueError("The current native analyzer is not the verified version.")
    if audit["exporter_sha256"] != sha(Path(contract.__file__)):
        raise ValueError("The export contract is not the independently verified version.")
    for name, expected in protocol["source_hashes"].items():
        if sha(output/"source"/name) != expected:
            raise ValueError(f"Frozen optimizer source changed: {name}")
    if not (bridge.get("native_results_complete") is True and bridge.get("exports_verified") == planned
            and bridge.get("requested_exports") == planned and bridge.get("anchor_missing_after") == 0):
        raise RuntimeError("All 144 bridge exports with actual native anchors must be verified first.")
    expected_pairs = {(j, m) for j in jobs for m in methods}
    bridge_rows = {(r["job"], r["method"]): r for r in bridge["receipts"]}
    if set(bridge_rows) != expected_pairs or len(bridge["receipts"]) != planned:
        raise ValueError("Bridge receipt pairs do not cover the fixed denominator uniquely.")
    for name, expected in bridge["source_hashes"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Bridge source changed: {name}")
    pending, provenance, distances, audits = [], {}, [], {}
    absent_before = absent_after = 0
    # All library/predictor references are frozen and checked before truth.
    for job in jobs:
        meta = catalog[job]; checked_path = output/"analysis/jobs"/job/"verified.json"
        checked = read(checked_path); audits[job] = checked
        if checked["errors"] or checked["protocol_sha256"] != protocol_hash or checked["analyzer_sha256"] != audit["analyzer_sha256"]:
            raise ValueError(f"Invalid per-input independent audit: {job}")
        if {k: sha(meta[k]) for k in ("train", "validation", "test")} != protocol["input_hashes"][job]:
            raise ValueError(f"Frozen observation hashes changed: {job}")
        for method in methods:
            folder = output/"jobs"/job/method
            native_path, raw_path = folder/"result.json", folder/"candidates.json"
            enriched_path = output/"bridge_candidates"/job/(method+".json")
            result, raw, enriched = read(native_path), read(raw_path), read(enriched_path)
            hashes = dict(native_result_sha256=sha(native_path), native_raw_export_sha256=sha(raw_path),
                          enriched_export_sha256=sha(enriched_path))
            if checked["raw_hashes"][method] != hashes["native_result_sha256"]:
                raise ValueError("A result changed after its independent numerical audit.")
            if result["candidate_export_sha256"] != hashes["native_raw_export_sha256"]:
                raise ValueError("A raw worker candidate file changed.")
            if result["input_hashes"] != {k: protocol["input_hashes"][job][k] for k in ("train", "validation")}:
                raise ValueError("Native input hashes differ from the frozen protocol.")
            references = enriched["provenance"]["bridge_postprocessing"]
            expected_refs = dict(native_result_sha256=hashes["native_result_sha256"],
                                 native_raw_export_sha256=hashes["native_raw_export_sha256"],
                                 native_protocol_sha256=protocol_hash, native_analysis_receipt_sha256=audit_hash)
            if any(references[k] != v for k, v in expected_refs.items()):
                raise ValueError("An enriched export references different verified native inputs.")
            contract.verify_export(raw); contract.verify_export(enriched)
            if raw["labels"] != meta["labels"] or enriched["labels"] != meta["labels"]:
                raise ValueError("Candidate terminal orders differ.")
            if any(result.get(k) is not False for k in ("truth_used", "test_opened", "final_RNJ_fallback")):
                raise ValueError("Native provenance is outside the own-search contract.")
            n = len(meta["labels"]); selected = result["selected"]["point"]
            anchor = contract.digest(contract._point(selected, n))
            raw_ids = {r["candidate_id"] for r in raw["candidates"]}
            enriched_by_id = {r["candidate_id"]: r for r in enriched["candidates"]}
            receipt = enriched["bridge_receipt"]
            missing = anchor not in raw_ids
            if (enriched["mainline_selected_candidate_id"] != anchor or anchor not in enriched_by_id
                    or not raw_ids <= set(enriched_by_id)):
                raise ValueError("Native anchor or original raw pool changed during enrichment.")
            if contract._point(enriched_by_id[anchor]["point"], n) != contract._point(selected, n):
                raise ValueError("The bridge anchor does not retain the exact selected coefficients.")
            if receipt["anchor_was_missing_from_raw_pool"] != missing or not receipt["anchor_in_pool"]:
                raise ValueError("Per-export anchor inclusion receipt does not replay.")
            if any(bridge_rows[(job, method)][k] != receipt[k] for k in
                   ("mainline_selected_candidate_id", "anchor_was_missing_from_raw_pool", "anchor_in_pool")):
                raise ValueError("Bridge summary disagrees with the per-export receipt.")
            absent_before += int(missing); absent_after += int(anchor not in enriched_by_id)
            native_R, native_X = contract.response_matrices(contract._point(selected, n), n)
            comparisons = []
            for raw_row in raw["candidates"]:
                row = dict(job=job, method=method, raw_candidate_id=raw_row["candidate_id"],
                           raw_index=raw_row["index"],
                           **matrix_distance(np.asarray(raw_row["R"]), np.asarray(raw_row["X"]), native_R, native_X))
                row["same_positive_topology_as_native"] = topology(raw_row["point"], n) == topology(selected, n)
                comparisons.append(row); distances.append(row)
            if not comparisons or not result["output_candidates"]:
                raise ValueError("An existing candidate library is empty.")
            nearest = min(comparisons, key=lambda r: (r["joint_entry_maxabs"], r["raw_index"]))
            raw_topologies = [topology(row["point"], n) for row in raw["candidates"]]
            output_topologies = [topology(row["point"], n) for row in result["output_candidates"]]
            proposal = result["exit_record"].get("raw_point")
            proposal_info = dict(exit_raw_available=isinstance(proposal, dict),
                                 exit_raw_accepted=bool(result["exit_record"].get("accepted", False)))
            if isinstance(proposal, dict):
                canonical_proposal = contract._point(proposal, n)
                proposal_R, proposal_X = contract.response_matrices(canonical_proposal, n)
                proposal_id = contract.digest(canonical_proposal)
                proposal_info.update(exit_raw_parameter_id=proposal_id,
                                     exit_raw_id_in_enriched=proposal_id in enriched_by_id,
                                     exit_raw_topology=topology(proposal, n),
                                     exit_raw_topology_in_existing_libraries=topology(proposal, n) in [*raw_topologies, *output_topologies],
                                     exit_raw_nearest_enriched_entry_maxabs=min(
                                         matrix_distance(row["R"], row["X"], proposal_R, proposal_X)["joint_entry_maxabs"]
                                         for row in enriched["candidates"]))
            if not any(contract._point(row["point"], n) == contract._point(selected, n)
                       for row in result["output_candidates"]):
                raise ValueError("Native selected point is absent from its saved output library.")
            pending.append(dict(job=job, method=method, case=meta["case"], bundle=meta["bundle"],
                                voltage_sigma=meta["voltage_sigma"], anchor_parameter_id_missing_from_raw=missing,
                                raw_any_exact_matrix_match=any(r["exact_matrix_equal"] for r in comparisons),
                                raw_exact_matrix_matches=sum(r["exact_matrix_equal"] for r in comparisons),
                                raw_nearest_entry_maxabs=nearest["joint_entry_maxabs"],
                                raw_nearest_induced_inf=nearest["joint_induced_inf"],
                                raw_nearest_candidate_id=nearest["raw_candidate_id"],
                                raw_nearest_same_positive_topology=nearest["same_positive_topology_as_native"],
                                raw_any_matrix_within_1e_minus10=nearest["joint_entry_maxabs"] <= 1e-10,
                                **proposal_info,
                                selected_topology=topology(selected, n),
                                pools=dict(raw_positive=raw_topologies, native_output24=output_topologies,
                                           raw_plus_output24=[*raw_topologies, *output_topologies])))
            provenance[job+"/"+method] = dict(**hashes, numerical_audit_sha256=sha(checked_path))
    if (absent_before, absent_after) != (bridge["anchor_missing_before"], bridge["anchor_missing_after"]):
        raise ValueError("Aggregate anchor-ID inclusion counts do not replay.")
    truth_by_job, truth_hashes = {}, {}
    for job in jobs:
        meta = catalog[job]; network = Path(meta["truth_network"])
        hashes = {name: sha(network/name) for name in ("metadata.json", "buses.csv", "branches.csv")}
        if hashes != audits[job]["truth_source_hashes"]:
            raise ValueError("Truth source changed since the independent native audit.")
        truth_hashes[job] = hashes
        physical = truth_contract.physical_truth(network, meta["labels"])
        index = {label: i for i, label in enumerate(meta["labels"])}
        truth_by_job[job] = frozenset(tuple(sorted(index[i] for i in C)) for C in physical["clades"])
    rows, pool_rows, proposal_rows = [], [], []
    for pending_row in pending:
        row = dict(pending_row); truth = truth_by_job[row["job"]]
        chosen, pools = row.pop("selected_topology"), row.pop("pools")
        row.update(chosen_F1=f1(chosen, truth), chosen_exact=chosen == truth, true_clades=len(truth))
        native_audit_row = next(r for r in audits[row["job"]]["rows"] if r["method"] == row["method"])
        if abs(row["chosen_F1"]-native_audit_row["clade_f1"]) > 1e-12 or row["chosen_exact"] != native_audit_row["exact"]:
            raise ValueError("Recomputed chosen metrics differ from the independent numerical audit.")
        for pool in POOLS:
            metrics = pool_diagnostic(pools[pool], truth)
            for key, value in metrics.items():
                row[pool+"_"+key] = value
            pool_rows.append(dict(job=row["job"], method=row["method"], pool=pool,
                                  chosen_F1=row["chosen_F1"], chosen_exact=row["chosen_exact"], **metrics))
        row["output24_has_true_but_chosen_wrong"] = row["native_output24_whole_true_tree_covered"] and not row["chosen_exact"]
        if row["exit_raw_available"]:
            proposed = row.pop("exit_raw_topology")
            raw_f1, raw_exact = f1(proposed, truth), proposed == truth
            existing_exact = row["raw_plus_output24_whole_true_tree_covered"]
            existing_oracle = row["raw_plus_output24_oracle_max_possible_F1"]
            row.update(exit_raw_F1=raw_f1, exit_raw_exact=raw_exact,
                       exit_raw_new_true_tree_beyond_existing=raw_exact and not existing_exact,
                       exit_raw_rejected_new_true_tree=raw_exact and not existing_exact and not row["exit_raw_accepted"],
                       exit_raw_rejected_oracle_F1_gain=(not row["exit_raw_accepted"]) and raw_f1 > existing_oracle+1e-12,
                       future_diagnostic_expanded_pool_true_tree=existing_exact or raw_exact,
                       future_diagnostic_expanded_oracle_F1=max(existing_oracle, raw_f1))
            proposal_rows.append({key: value for key, value in row.items()
                                  if key in {"job", "method", "case", "bundle", "voltage_sigma"}
                                  or key.startswith("exit_raw_") or key.startswith("future_diagnostic_")})
        rows.append(row)
    frame, pool_frame = pd.DataFrame(rows), pd.DataFrame(pool_rows)
    summary, matrix_summary, pool_summary, proposal_summary = [], [], [], []
    for method in methods:
        group = frame[frame.method == method]
        if len(group) != 48:
            raise ValueError("A method lost its 48-input denominator.")
        summary.append(dict(method=method, planned=48, completed=len(group), chosen_F1=float(group.chosen_F1.mean()),
                            chosen_exact=int(group.chosen_exact.sum()), output24_pool_true_tree=int(group.native_output24_whole_true_tree_covered.sum()),
                            output24_has_true_but_chosen_wrong=int(group.output24_has_true_but_chosen_wrong.sum()),
                            raw_pool_true_tree=int(group.raw_positive_whole_true_tree_covered.sum()),
                            union_pool_true_tree=int(group.raw_plus_output24_whole_true_tree_covered.sum()),
                            raw_oracle_max_F1_mean=float(group.raw_positive_oracle_max_possible_F1.mean()),
                            output24_oracle_max_F1_mean=float(group.native_output24_oracle_max_possible_F1.mean()),
                            union_oracle_max_F1_mean=float(group.raw_plus_output24_oracle_max_possible_F1.mean()),
                            raw_any_exact_matrix_match=int(group.raw_any_exact_matrix_match.sum()),
                            raw_any_matrix_within_1e_minus10=int(group.raw_any_matrix_within_1e_minus10.sum())))
        proposals = pd.DataFrame(proposal_rows)
        g = proposals[proposals.method == method] if len(proposals) else pd.DataFrame()
        proposal_summary.append(dict(method=method, planned=48, exit_raw_available=len(g),
            exit_raw_rejected=int((~g.exit_raw_accepted).sum()) if len(g) else 0,
            exit_raw_true_tree=int(g.exit_raw_exact.sum()) if len(g) else 0,
            exit_raw_new_true_tree_beyond_existing=int(g.exit_raw_new_true_tree_beyond_existing.sum()) if len(g) else 0,
            exit_raw_rejected_new_true_tree=int(g.exit_raw_rejected_new_true_tree.sum()) if len(g) else 0,
            exit_raw_rejected_oracle_F1_gain=int(g.exit_raw_rejected_oracle_F1_gain.sum()) if len(g) else 0,
            future_diagnostic_expanded_pool_true_tree=int(g.future_diagnostic_expanded_pool_true_tree.sum()) if len(g) else 0,
            future_diagnostic_expanded_oracle_F1_mean=float(g.future_diagnostic_expanded_oracle_F1.mean()) if len(g) else None))
        for pool in POOLS:
            g = pool_frame[(pool_frame.method == method) & (pool_frame.pool == pool)]
            pool_summary.append(dict(method=method, pool=pool, planned=48, complete=len(g),
                                     whole_true_tree_covered=int(g.whole_true_tree_covered.sum()),
                                     whole_true_tree_coverage_rate=float(g.whole_true_tree_covered.mean()),
                                     oracle_max_possible_F1_mean=float(g.oracle_max_possible_F1.mean()),
                                     oracle_F1_above_chosen_count=int((g.oracle_max_possible_F1 > g.chosen_F1+1e-12).sum()),
                                     all_true_connections_in_union=int(g.all_true_connections_in_union.sum()),
                                     connection_union_recall_mean=float(g.true_connection_union_recall.mean())))
    for method in [*methods, "ALL"]:
        group = frame if method == "ALL" else frame[frame.method == method]
        values = group.raw_nearest_entry_maxabs.to_numpy()
        record = dict(method=method, planned=len(group), anchor_parameter_id_missing=int(group.anchor_parameter_id_missing_from_raw.sum()),
                      exact_matrix_match=int(group.raw_any_exact_matrix_match.sum()),
                      minimum=float(np.min(values)), q25=float(np.quantile(values, .25)),
                      median=float(np.median(values)), q75=float(np.quantile(values, .75)),
                      p90=float(np.quantile(values, .90)), maximum=float(np.max(values)))
        for tolerance in TOLERANCES:
            record[f"entry_maxabs_le_{tolerance:g}"] = int(np.count_nonzero(values <= tolerance))
        matrix_summary.append(record)
    target = output/"candidate_diagnostics"; target.mkdir(parents=True, exist_ok=True)
    artifacts = {"per_model.csv": frame, "summary.csv": pd.DataFrame(summary),
                 "pool_summary.csv": pd.DataFrame(pool_summary), "matrix_summary.csv": pd.DataFrame(matrix_summary),
                 "raw_matrix_distances.csv": pd.DataFrame(distances),
                 "proposal_diagnostics.csv": pd.DataFrame(proposal_rows),
                 "proposal_summary.csv": pd.DataFrame(proposal_summary)}
    for name, data in artifacts.items():
        data.to_csv(target/name, index=False, lineterminator="\n")
    verification = dict(passed=True, complete=True, native_evaluations=planned, per_method_denominator=48,
                        bridge_exports_verified=planned, raw_matrix_comparisons=len(distances),
                        anchor_parameter_id_missing_before=absent_before, anchor_parameter_id_missing_after=absent_after,
                        raw_exact_matrix_matches=int(frame.raw_any_exact_matrix_match.sum()),
                        raw_matrix_within_1e_minus10=int(frame.raw_any_matrix_within_1e_minus10.sum()),
                        entry_norm="max(max_ij |R_raw-R_native|, max_ij |X_raw-X_native|)",
                        induced_inf_norm="also saved: max of the two matrix maximum absolute row sums; nearest chosen by entry norm",
                        near_equality_threshold=1e-10, exact_matrix_equality="numpy.array_equal on saved raw R/X and canonical native R/X; no tolerance",
                        ID_absence_is_not_functional_predictor_absence=True,
                        topology_definition="proper internal supports with max(r,x)>1e-9; native output uses its saved .24-contracted/refitted model",
                        union_library_definition="existing complete raw models plus existing complete output24 models; no connection recombination",
                        connection_union_does_not_imply_whole_tree_coverage=True,
                        oracle_max_F1_is_diagnostic_not_executable=True,
                        exit_raw_proposals_scope="saved E raw_point only; diagnostics include accepted and rejected proposals, never add them to the actual path or bridge pool",
                        chosen_metrics_match_native_audit=True,
                        selection_or_parameters_changed=False, truth_opened_only_after_all_native_and_bridge_hash_guards=True,
                        new_fit_calls=0, new_solver_calls=0, new_candidates_generated=0,
                        native_protocol_sha256=protocol_hash, native_analysis_receipt_sha256=audit_hash,
                        bridge_summary_sha256=sha(bridge_path), script_sha256=sha(__file__),
                        contract_sha256=sha(Path(contract.__file__)), truth_helper_sha256=sha(Path(truth_contract.__file__)),
                        native_inputs=provenance, truth_source_hashes=truth_hashes,
                        result_hashes={name: sha(target/name) for name in artifacts}, seconds=perf_counter()-started)
    write(target/"verification.json", verification)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(matrix_summary).to_string(index=False))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT/"outputs/day96_t_benchmark_20261110")
    diagnose(parser.parse_args().output)
