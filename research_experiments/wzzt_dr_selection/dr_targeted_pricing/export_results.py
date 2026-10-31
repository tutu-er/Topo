"""Whitelist export of terminal targeted-price results; no scoring or solving.

This exporter reads only saved scalar/topology results.  It does not import an
identifier, optimizer, physical simulator, or production scoring routine.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick(data, keys):
    return {k: data[k] for k in keys if k in data}


def bound(ref):
    p = Path(ref["path"])
    if sha(p) != ref["sha256"]:
        raise ValueError("Referenced receipt changed: " + p.name)
    return p


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


MODEL_KEYS = (
    "full_errors", "full_recovery", "missing", "extra", "predicted_blocks",
    "truth_blocks", "prediction_definition", "truth_definition", "selected_origin",
    "train_mae", "validation_mae", "before_contraction_errors",
)
PRICE_KEYS = (
    "index", "plan_index", "module_id", "price_indices", "prices",
    "resource_day_probabilities", "resource_probabilities",
    "training_pair_probabilities", "validation_pair_probabilities",
    "pair_state_probabilities", "risk_lower", "risk_upper", "expected_fee",
    "worst_fee", "training_Gamma", "validation_Gamma",
)
TRACE_KEYS = (
    "accepted_updates", "accepted_cross_updates", "removed_blocks", "search_stops",
    "wall_limited", "final_source", "pre_contraction_errors", "final_errors",
)
MECHANISM_KEYS = (
    "record_id", "status", "target_in_pool", "target_ever_in_accepted_path",
    "best_validation_with_target", "best_validation_without_target",
    "no_target_exclusion_margin", "selected_raw_target", "selected_final_target",
    "selected_origin", "final_errors", "raw_errors",
)


def prices(result):
    out = pick(result, (
        "status", "module_ids", "state_order", "paid_events", "price_vector_order",
        "energy_per_paid_event_kwh", "hard_budget", "seed", "loss_table_exact",
        "uncertain_cells", "finite_domain_lower_bound", "finite_domain_gap_upper",
        "feasible_price_vectors_per_plan", "all_price_vectors_per_plan",
        "solver_receipts", "variables", "scope", "missing_semantics",
        "lookup_equivalence", "AC_calls", "identifier_calls",
        "independent_upper_optimality_check", "independent_lex_fee_check",
        "risk_table_exact", "selection_basis",
    ))
    for k in ("selected", "upper_candidate"):
        out[k] = None if result.get(k) is None else pick(result[k], PRICE_KEYS)
    for k in ("feasible_columns", "all_exact_lex_ties", "possible_risk_optimal_columns",
              "upper_risk_lex_ties"):
        if k in result:
            out[k] = [pick(v, PRICE_KEYS) for v in result[k]]
    return out


def phase(root, name):
    folder = root / name
    scored, frozen, mechanisms = [read(folder / n) for n in (
        "scored.json", "fits_frozen.json", "target_mechanisms.json")]
    observations = read(folder / "observations.json")
    if scored["status"] != "complete" or frozen["status"] != "frozen":
        raise RuntimeError(name + " has not reached terminal scoring/freeze")
    if any(folder.glob("*.lock")):
        raise RuntimeError(name + " still has an execution lock")
    if scored["fits_frozen_sha256"] != sha(folder / "fits_frozen.json"):
        raise RuntimeError(name + " scored/frozen hash mismatch")
    if frozen["observation_ledger_sha256"] != sha(folder / "observations.json"):
        raise RuntimeError(name + " frozen/observations hash mismatch")
    if len(scored["rows"]) != scored["planned"]:
        raise RuntimeError(name + " record count mismatch")
    job_keys = [j["key"] for j in frozen["jobs"]]
    if len(set(job_keys)) != scored["unique_jobs"]:
        raise RuntimeError(name + " unique-job count mismatch")
    rows, receipt_rows = [], []
    for r in scored["rows"]:
        out = pick(r, (
            "record_id", "policy", "module", "phase", "path", "train_state",
            "validation_state", "prices", "status", "physical_status",
            "actual_payment", "worst_payment", "operation_checks", "payment_scope",
            "exception_type", "message", "fit_key",
        ))
        # Exact complete-input identities are digests, not observation arrays.
        out["models"] = {k: pick(v, MODEL_KEYS)
                         for k, v in r.get("models", {}).items()}
        out["trace"] = pick(r.get("trace", {}), TRACE_KEYS)
        rows.append(out)
        if "physical_receipt" not in r:
            receipt_rows.append({"record_id": r["record_id"], "status": "missing"})
            continue
        p = bound(r["physical_receipt"])
        rec = read(p)
        events = rec.get("event_records", [])
        receipt_rows.append(dict(
            record_id=r["record_id"], receipt_sha256=sha(p),
            **pick(rec, ("status", "physical_events_expected", "physical_events_complete",
                         "physical_event_rows_expected", "all_operation_checks_pass",
                         "budget_pass", "quote_qualification_pass", "actual_payment",
                         "committed_payment", "worst_payment", "incremental_electricity_bill")),
            event_statuses=[e.get("status") for e in events],
            AC_converged_samples=sum(e.get("checks", {}).get("ac_converged_samples", 0)
                                     for e in events),
            event_checks=[dict(event=e.get("event"),
                               resource_feasible=e.get("resource_feasible"),
                               independent_resource_billing_pass=e.get("independent_resource_billing_pass"),
                               increment_only_bridge_pass=e.get("increment_only_bridge_pass"),
                               **pick(e.get("checks", {}), (
                                   "electrical_equations_pass", "voltage_limits_pass",
                                   "ac_all_96_converged", "branch_ampacity_checked")))
                          for e in events],
        ))
    summary = pick(scored, ("status", "planned", "unique_jobs", "unique_complete",
                            "unique_failed", "actual_LP_calls"))
    summary.update(
        record_status_counts=dict(Counter(r["status"] for r in rows)),
        unique_status_counts=dict(Counter(j["status"] for j in frozen["jobs"])),
        unique_elapsed_seconds=sum(j.get("elapsed_seconds") or 0 for j in frozen["jobs"]),
        unique_fixed_LP_calls=sum(j.get("LP_solves") or 0 for j in frozen["jobs"]),
        physical_events_complete=sum(r.get("physical_events_complete", 0) for r in receipt_rows),
        AC_converged_samples=sum(r.get("AC_converged_samples", 0) for r in receipt_rows),
        all_saved_operation_checks_pass=all(r.get("all_operation_checks_pass") is True
                                            for r in receipt_rows),
        all_saved_budget_checks_pass=all(r.get("budget_pass") is True for r in receipt_rows),
        record_wall_limited=sum(bool(r["trace"].get("wall_limited")) for r in rows),
        independent_paths=sorted(set(r["path"] for r in rows)),
    )
    return dict(summary=summary, records=rows,
                target_mechanisms=[pick(r, MECHANISM_KEYS) for r in mechanisms["rows"]],
                physical_checks=receipt_rows)


def export(root, destination):
    root, destination = Path(root), Path(destination)
    # summary is written only after both completed score/freeze stages.
    full = read(root / "summary.json")
    if full["status"] != "completed":
        raise RuntimeError("Final aggregate is not terminal")
    base, decision, oracle = [read(root / p) for p in (
        "protocol.json", "decision_protocol.json", "oracle_selected.json")]
    if oracle["protocol_sha256"] != sha(root / "protocol.json"):
        raise RuntimeError("Oracle base protocol mismatch")
    if oracle["decision_protocol_sha256"] != sha(root / "decision_protocol.json"):
        raise RuntimeError("Oracle decision protocol mismatch")
    for key, file in (("planning_scored_sha256", "planning/scored.json"),
                      ("loss_table_sha256", "planning/loss_table.json"),
                      ("price_optimization_sha256", "event_price_optimization.json")):
        if oracle[key] != sha(root / file):
            raise RuntimeError("Oracle input mismatch: " + key)
    phases = {name: phase(root, name) for name in ("planning", "test")}
    if len(phases["test"]["summary"]["independent_paths"]) != len(base["test_paths"]):
        raise RuntimeError("Test-path denominator mismatch")
    opt = read(root / "event_price_optimization.json")
    candidates = {k: prices(v) for k, v in opt.items()}
    if len(candidates["global_result"]["feasible_columns"]) != 100:
        raise RuntimeError("Declared hundred-column domain mismatch")
    baseline = read(root / "baseline_evaluation.json")
    base_summary = pick(baseline, (
        "status", "created_utc", "protocol_sha256", "rows", "root_label", "labels",
        "target_support_labels", "full_truth_blocks", "LP_calls", "current_target_missing",
        "current_has_any_full_errors", "old_failure_not_assumed", "raw_selected_support_labels",
        "final_support_labels", "selected_raw_index", "selected_raw_config", "selected_raw_origin",
        "best_validation_mae", "validation_cutoff", "selected_raw_validation",
        "selected_validation_ratio_to_best", "selected_distance_inside_validation_band",
        "truth_definition", "solver_or_AC_calls_in_evaluation",
    ))
    base_summary["scores"] = {k: pick(v, MODEL_KEYS) for k, v in baseline["scores"].items()}
    policy_groups = {}
    for name, p in oracle["test_policies"].items():
        key = json.dumps([p["module_id"], p["prices"]], sort_keys=True)
        policy_groups.setdefault(key, []).append(name)
    scope = dict(
        design="known-error-block targeting and known-generating-model offline reference",
        planning_natural_measurement_backgrounds=1,
        behavior_states_per_complete_plan=16,
        independent_test_paths=len(base["test_paths"]),
        repeated_policy_observations_are_not_independent=True,
        policies_with_identical_plan_and_quotes=list(policy_groups.values()),
        parameter_fits="all continuous R/X reestimated from zero; original identifier unchanged",
        physical_constraint_scope="saved resource, budget, AC equations and voltage checks; ampacity remains unchecked",
        no_population_or_continuous_domain_optimality=True,
        expected_fees_are_distinct_from_nominal_precompute_receipts=True,
        no_new_score_optimizer_identifier_or_AC_calls_in_export=True,
    )
    final_summary = pick(full, (
        "status", "protocol_sha256", "planning_requests", "planning_unique", "test_requests",
        "test_unique", "completed_unique", "failed_unique", "LP_solves", "source", "scope",
        "no_population_or_full_continuous_space_optimality_claim",
    ))
    # Already computed aggregate scalar rows; no topology or payment is recomputed here.
    final_summary["test"] = []
    for r in full["test"]:
        public = pick(r, (
            "policy", "module_id", "prices", "completed", "denominator",
            "full_error_sum", "mean_full_errors", "recoveries", "actual_fee_sum",
        ))
        public["by_path"] = [pick(q, ("path", "status", "error", "fee"))
                             for q in r.get("by_path", [])]
        final_summary["test"].append(public)
    final_summary["baseline"] = base_summary
    final_summary["phase_counts"] = {k: v["summary"] for k, v in phases.items()}
    final_summary["scope_detail"] = scope
    oracle_public = pick(oracle, (
        "status", "created_utc", "protocol_sha256", "decision_protocol_sha256",
        "planning_scored_sha256", "loss_table_sha256", "price_optimization_sha256",
        "test_policies", "no_test_data_yet", "scope",
    ))
    products = dict(
        actual_loss_table=read(root / "planning/loss_table.json"),
        actual_price_summary=candidates,
        actual_oracle=oracle_public,
        actual_summary=final_summary,
        actual_record_ledger={k: dict(records=v["records"], physical_checks=v["physical_checks"])
                              for k, v in phases.items()},
        actual_target_mechanisms={k: v["target_mechanisms"] for k, v in phases.items()},
    )
    destination.mkdir(parents=True, exist_ok=True)
    exported = {}
    for name, obj in products.items():
        p = destination / (name + ".json")
        write(p, obj)
        exported[p.name] = dict(sha256=sha(p), bytes=p.stat().st_size)
    source_files = ["protocol.json", "decision_protocol.json", "event_price_decision_lock.json",
                    "oracle_selected.json", "event_price_optimization.json", "summary.json",
                    "baseline_evaluation.json", "planning/loss_table.json"]
    source_files += [f"{phase_name}/{name}.json" for phase_name in ("planning", "test")
                     for name in ("observations", "fits_frozen", "scored", "target_mechanisms")]
    manifest = dict(
        status="exported", created_utc=datetime.now(timezone.utc).isoformat(),
        exporter_sha256=sha(__file__), source_input_sha256={p: sha(root / p) for p in source_files},
        frozen_source_sha256=base["source_sha256"],
        frozen_decision_source_sha256=decision["source_sha256"],
        files=exported, scope=scope,
        whitelist="scalar scores, discrete supports, state loss tables, quotes, solver receipts and checks only",
        excluded="household data; P/Q/V/AC trajectories; continuous fitted R/X; raw results and payload packets",
        earlier_evidence_preserved=True,
    )
    # Preserve the earlier preparation manifest instead of silently replacing it.
    old_manifest = destination / "manifest.json"
    if old_manifest.exists() and not (destination / "preparation_manifest.json").exists():
        previous = read(old_manifest)
        if previous.get("exporter_sha256") != sha(__file__):
            write(destination / "preparation_manifest.json", previous)
    write(old_manifest, manifest)
    print(json.dumps(dict(status="exported", files=list(exported),
                          destination=str(destination), independent_paths=len(base["test_paths"])),
                     ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    export(a.root, a.output)


if __name__ == "__main__":
    main()
