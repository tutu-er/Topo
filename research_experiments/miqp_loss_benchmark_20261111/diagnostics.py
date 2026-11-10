"""Small descriptive diagnostics after the complete MIQP/L1 audit passes.

No observations, truth files, fitting code, or native solver are imported.
The input is the 192 SHA-verified saved models and their existing audit rows.
Run only after all 96 new and 96 paired parent models have passed analysis::

    python -m research_experiments.miqp_loss_benchmark_20261111.diagnostics OUTPUT

Writes OUTPUT/analysis/diagnostics/{summary.json,solver_calls.csv,
selector_changes.csv,t_effects.csv,interactions.csv,seed_changes.csv,coverage.csv}.
summary.json defines every contrast and records input/output hashes.  The CSVs
contain individual saved calls/models; summary arrays contain descriptive
counts and means.  A changed saved-point index is kept separate from a changed
parameter fit, support family, or positive proper-support topology.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
LOSSES = ("mae", "scaled_mse")
METHODS = ("noT", "T003")
VIEWS = ("primary_mae", "secondary_mse")
TIE = 1e-10


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                          encoding="utf-8", newline="\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("An expected complete diagnostics table is empty: "+str(path))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def canonical(supports):
    return tuple(sorted({tuple(map(int, C)) for C in supports}, key=lambda C: (len(C), C)))


def topology(point, n):
    return frozenset(tuple(C) for C, r, x in zip(point["supports"], point["r"], point["x"])
                     if 1 < len(C) < n and max(r, x) > 1e-9)


def fingerprint(point):
    return tuple(sorted((tuple(C), float(r), float(x)) for C, r, x in
                        zip(point["supports"], point["r"], point["x"])))


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def gaps(upper, lower):
    """Native minimization gap, with an explicit zero-incumbent convention."""
    upper, lower = finite(upper), finite(lower)
    if upper is None or lower is None:
        return dict(incumbent=upper, bound=lower, absolute_gap=None, relative_gap=None,
                    relative_gap_kind="unavailable")
    if lower > upper+2e-8:
        raise ValueError("A saved solver lower bound exceeds its incumbent.")
    absolute = max(0., upper-lower)
    relative = absolute/abs(upper) if upper != 0. else (0. if absolute == 0. else None)
    return dict(incumbent=upper, bound=lower, absolute_gap=absolute, relative_gap=relative,
                relative_gap_kind="finite" if relative is not None else "infinite_zero_incumbent")


def calls(result):
    for event_index, event in enumerate(result["trajectory"].get("attempts", [])):
        for trial_index, call in enumerate(event.get("trials", [event])):
            if call.get("status") is not None:
                yield event.get("kind", "unknown"), event_index, trial_index, call
    event = result.get("exit_event", {})
    if event.get("status") is not None:
        yield "final_exit", None, None, event


def average(values):
    values = [float(v) for v in values if v is not None]
    return sum(values)/len(values) if values else None


def contrast_summary(rows, keys, field):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in groups.items():
        item = dict(zip(keys, key)); values = [float(r[field]) for r in group]
        item.update(paired=len(group), mean=average(values), total=sum(values),
                    wins=sum(v > TIE for v in values), ties=sum(abs(v) <= TIE for v in values),
                    losses=sum(v < -TIE for v in values))
        output.append(item)
    return output


def guard(output):
    protocol, verification = read(output/"protocol.json"), read(output/"analysis/verification.json")
    if (protocol["planned"] != 96 or len(protocol["jobs"]) != 48
            or tuple(m["id"] for m in protocol["methods"]) != METHODS):
        raise ValueError("The fixed 48-input / 96-new-model protocol is required.")
    if (not verification["passed"] or not verification["complete"]
            or verification["new_native_verified"] != 96 or verification["reused_parent_verified"] != 96):
        raise RuntimeError("Diagnostics require all 96 new and 96 paired parent models to pass independent analysis.")
    if any(verification[k] != 0 for k in ("new_native_failed", "new_native_invalid", "new_native_pending")):
        raise ValueError("A nonzero unresolved new-model denominator remains.")
    if verification["protocol_sha256"] != sha(output/"protocol.json") or verification["catalog_sha256"] != sha(output/"catalog.json"):
        raise ValueError("The verified new protocol or catalog changed.")
    parent = Path(protocol["parent_output"])
    parent_verification = read(parent/"analysis/verification.json")
    if (not parent_verification["passed"] or not parent_verification["complete"]
            or parent_verification["planned"] != 144 or parent_verification["verified"] != 144):
        raise ValueError("Complete passed parent L1 verification is required.")
    if verification["parent_analysis_verification_sha256"] != sha(parent/"analysis/verification.json"):
        raise ValueError("The reused parent analysis changed.")
    if protocol["parent_protocol_sha256"] != sha(parent/"protocol.json") or protocol["parent_catalog_sha256"] != sha(parent/"catalog.json"):
        raise ValueError("The reused parent protocol or catalog changed.")
    for base, receipt in ((output, verification), (parent, parent_verification)):
        if sha(base/"analysis_source/analyze.py") != receipt["analyzer_sha256"]:
            raise ValueError("An authoritative frozen post-analysis source changed.")
    for base, hashes in ((output, protocol["source_hashes"]), (parent, protocol["parent_source_hashes"])):
        for name, digest in hashes.items():
            if sha(base/"source"/name) != digest:
                raise ValueError("A frozen native source changed: "+name)
    return protocol, parent, verification


def diagnostics(output):
    output = Path(output).resolve(); started = perf_counter()
    # This guard runs before any saved metrics are read or output is created.
    protocol, parent, verification = guard(output)
    catalog = read(output/"catalog.json")
    models, view_rows, model_rows, source_hashes = {}, {}, {}, {}
    verification_hashes = {}
    for job in protocol["jobs"]:
        audit_file = output/"analysis/jobs"/job/"verified.json"
        audit = read(audit_file)
        if audit["analyzer_sha256"] != verification["analyzer_sha256"] or audit["protocol_sha256"] != verification["protocol_sha256"]:
            raise ValueError("A per-input audit refers to a different analysis or protocol.")
        verification_hashes[job] = sha(audit_file)
        for row in audit["rows"]:
            key = (job, row["fit_loss"], row["method"])
            if row["status"] != "verified" or key in model_rows:
                raise ValueError("Missing, repeated, or unverified model row.")
            model_rows[key] = row
        for row in audit["views"]:
            key = (job, row["fit_loss"], row["method"], row["view"])
            if row["status"] != "verified" or key in view_rows:
                raise ValueError("Missing, repeated, or unverified selector-view row.")
            view_rows[key] = row
        for loss_name, base in (("mae", parent), ("scaled_mse", output)):
            for method in METHODS:
                file = base/"jobs"/job/method/"result.json"; measured = sha(file)
                if audit["raw_hashes"][loss_name+"/"+method] != measured:
                    raise ValueError("A saved model differs from the current independent audit.")
                if loss_name == "mae":
                    provenance = audit["parent_analysis"][method]
                    if (provenance["result_sha256"] != measured
                            or provenance["job_verification_sha256"] != sha(parent/"analysis/jobs"/job/"verified.json")
                            or provenance["analysis_verification_sha256"] != sha(parent/"analysis/verification.json")):
                        raise ValueError("A reused parent per-input audit changed.")
                source_hashes[job+"/"+loss_name+"/"+method] = measured
                models[job, loss_name, method] = read(file)
    expected_models = {(j, loss, method) for j in protocol["jobs"] for loss in LOSSES for method in METHODS}
    expected_views = {(*key, view) for key in expected_models for view in VIEWS}
    if set(models) != set(model_rows) or set(models) != expected_models or set(view_rows) != expected_views:
        raise ValueError("The verified 192-model / 384-view denominator changed.")

    solver_rows, selectors, seed_rows, coverage_rows = [], [], [], []
    for (job, loss_name, method), result in models.items():
        meta = catalog[job]
        header = dict(job=job, case=meta["case"], seed_index=meta["seed_index"],
                      voltage_sigma=meta["voltage_sigma"], fit_loss=loss_name, method=method)
        for index, (stage, event_index, trial_index, call) in enumerate(calls(result)):
            upper = call.get("objective", call.get("final_objective"))
            lower = call.get("dual_bound", call.get("bound", call.get("final_bound")))
            gap = gaps(upper, lower)
            constant = float(call.get("protected_count_cost", call.get("fixed_count_cost", 0.)))
            regularized = gaps(None if gap["incumbent"] is None else gap["incumbent"]+constant,
                               None if gap["bound"] is None else gap["bound"]+constant)
            solver_rows.append(dict(**header, call_index=index, stage=stage, event_index=event_index,
                trial_index=trial_index, status=int(call["status"]), seconds=finite(call.get("seconds")),
                saved_relative_gap=finite(call.get("relative_gap", call.get("gap"))),
                requested_relative_gap=finite(call.get("requested_relative_gap")), **gap,
                protected_count_constant=constant, regularized_absolute_gap=regularized["absolute_gap"],
                regularized_relative_gap=regularized["relative_gap"],
                gap_scope="native model objective; protected count constant separately added when recorded"))
        by_index = {int(row["index"]): row for row in result["output_candidates"]}
        primary = view_rows[job, loss_name, method, "primary_mae"]
        secondary = view_rows[job, loss_name, method, "secondary_mse"]
        a, b = by_index[int(primary["selected_index"])], by_index[int(secondary["selected_index"])]
        if int(result["selected"]["index"]) != int(a["index"]) or fingerprint(result["selected"]["point"]) != fingerprint(a["point"]):
            raise ValueError("The primary view does not anchor the actual selected full model.")
        if "selected_mse_view" in result and (int(result["selected_mse_view"]["index"]) != int(b["index"])
                or fingerprint(result["selected_mse_view"]["point"]) != fingerprint(b["point"])):
            raise ValueError("The secondary view does not anchor its saved full model.")
        n = len(meta["labels"])
        selectors.append(dict(**header, output_library_size=len(by_index), primary_index=a["index"], secondary_index=b["index"],
            saved_point_changed=a["index"] != b["index"], parameter_fit_changed=fingerprint(a["point"]) != fingerprint(b["point"]),
            support_family_changed=canonical(a["point"]["supports"]) != canonical(b["point"]["supports"]),
            topology_changed=topology(a["point"], n) != topology(b["point"], n),
            delta_F1=secondary["clade_f1"]-primary["clade_f1"], delta_exact=int(secondary["exact"])-int(primary["exact"]),
            primary_F1=primary["clade_f1"], secondary_F1=secondary["clade_f1"],
            primary_exact=bool(primary["exact"]), secondary_exact=bool(secondary["exact"]),
            primary_validation_mae=primary["validation_mae"], secondary_validation_mae=secondary["validation_mae"],
            primary_validation_mse=primary["validation_mse"], secondary_validation_mse=secondary["validation_mse"]))
        row = model_rows[job, loss_name, method]
        raw, mapped = bool(row["raw_pool_has_true_tree"]), bool(row["output_pool_has_true_tree"])
        united = bool(row["own_pool_union_has_true_connections"])
        coverage_rows.append(dict(**header, raw_pool_has_true_tree=raw, output_pool_has_true_tree=mapped,
            own_pool_union_has_true_connections=united, own_pool_union_true_recall=row["own_pool_union_true_recall"],
            truth_lost_by_output_mapping=raw and not mapped, truth_created_by_output_mapping=not raw and mapped,
            primary_misses_available_true_tree=mapped and not primary["exact"],
            secondary_misses_available_true_tree=mapped and not secondary["exact"],
            union_connections_without_raw_true_tree=united and not raw,
            primary_exact=bool(primary["exact"]), secondary_exact=bool(secondary["exact"])))

    for job in protocol["jobs"]:
        for method in METHODS:
            old, new = (models[job, loss_name, method]["trajectory"]["seeding"] for loss_name in LOSSES)
            a, b = canonical(old["seeds"]), canonical(new["seeds"])
            seed_rows.append(dict(job=job, case=catalog[job]["case"], seed_index=catalog[job]["seed_index"],
                voltage_sigma=catalog[job]["voltage_sigma"], method=method, seed_family_changed=a != b,
                L1_seed_count=len(a), MSE_seed_count=len(b), removed_seed_count=len(set(a)-set(b)),
                added_seed_count=len(set(b)-set(a)), L1_ranking_complete=bool(old["ranking_complete"]),
                MSE_ranking_complete=bool(new["ranking_complete"]),
                L1_seeds=json.dumps(a), MSE_seeds=json.dumps(b)))

    t_rows, interaction_rows = [], []
    within_loss_seed_changes = []
    for job in protocol["jobs"]:
        for loss_name in LOSSES:
            seeds = [canonical(models[job, loss_name, method]["trajectory"]["seeding"]["seeds"]) for method in METHODS]
            if seeds[0] != seeds[1]:
                within_loss_seed_changes.append(dict(job=job, fit_loss=loss_name, noT_seeds=seeds[0], T003_seeds=seeds[1]))
            for view in VIEWS:
                a, b = (view_rows[job, loss_name, method, view] for method in METHODS)
                t_rows.append(dict(job=job, case=catalog[job]["case"], seed_index=catalog[job]["seed_index"],
                    voltage_sigma=catalog[job]["voltage_sigma"], fit_loss=loss_name, view=view,
                    delta_F1=b["clade_f1"]-a["clade_f1"], delta_exact=int(b["exact"])-int(a["exact"]),
                    delta_seconds=b["total_seconds"]-a["total_seconds"],
                    new_exact=bool(b["exact"] and not a["exact"]), lost_exact=bool(a["exact"] and not b["exact"]),
                    same_initial_seed_family=seeds[0] == seeds[1]))
        for view in VIEWS:
            cells = {(loss_name, method): view_rows[job, loss_name, method, view]
                     for loss_name in LOSSES for method in METHODS}
            def interaction(field):
                return ((float(cells["scaled_mse", "T003"][field])-float(cells["scaled_mse", "noT"][field]))
                       -(float(cells["mae", "T003"][field])-float(cells["mae", "noT"][field])))
            interaction_rows.append(dict(job=job, case=catalog[job]["case"], seed_index=catalog[job]["seed_index"],
                voltage_sigma=catalog[job]["voltage_sigma"], view=view, F1_interaction=interaction("clade_f1"),
                exact_interaction=interaction("exact"), seconds_interaction=interaction("total_seconds")))

    solver_summary = []
    for loss_name in LOSSES:
        for method in METHODS:
            group = [r for r in solver_rows if r["fit_loss"] == loss_name and r["method"] == method]
            for stage in ["all", *sorted({r["stage"] for r in group})]:
                subset = group if stage == "all" else [r for r in group if r["stage"] == stage]
                gap_values = [r["relative_gap"] for r in subset if r["relative_gap"] is not None]
                solver_summary.append(dict(fit_loss=loss_name, method=method, stage=stage, calls=len(subset),
                    status2=sum(r["status"] == 2 for r in subset), status9=sum(r["status"] == 9 for r in subset),
                    other_statuses={str(s): sum(r["status"] == s for r in subset) for s in sorted({r["status"] for r in subset}-{2, 9})},
                    computable_gaps=len(gap_values), max_relative_gap=max(gap_values) if gap_values else None,
                    mean_relative_gap=average(gap_values), solver_seconds=sum(r["seconds"] or 0. for r in subset)))
    selector_summary, seed_summary, coverage_summary = [], [], []
    for loss_name in LOSSES:
        for method in METHODS:
            subset = [r for r in selectors if r["fit_loss"] == loss_name and r["method"] == method]
            selector_summary.append(dict(fit_loss=loss_name, method=method, paired=len(subset),
                **{name: sum(bool(r[name]) for r in subset) for name in
                   ("saved_point_changed", "parameter_fit_changed", "support_family_changed", "topology_changed")},
                delta_F1=average(r["delta_F1"] for r in subset), delta_exact=sum(r["delta_exact"] for r in subset),
                new_exact=sum(r["secondary_exact"] and not r["primary_exact"] for r in subset),
                lost_exact=sum(r["primary_exact"] and not r["secondary_exact"] for r in subset)))
            subset = [r for r in coverage_rows if r["fit_loss"] == loss_name and r["method"] == method]
            coverage_summary.append(dict(fit_loss=loss_name, method=method, models=len(subset),
                **{name: sum(bool(r[name]) for r in subset) for name in
                   ("raw_pool_has_true_tree", "output_pool_has_true_tree", "own_pool_union_has_true_connections",
                    "truth_lost_by_output_mapping", "truth_created_by_output_mapping",
                    "primary_misses_available_true_tree", "secondary_misses_available_true_tree",
                    "union_connections_without_raw_true_tree", "primary_exact", "secondary_exact")}))
    for method in METHODS:
        subset = [r for r in seed_rows if r["method"] == method]
        seed_summary.append(dict(method=method, paired=len(subset), changed=sum(r["seed_family_changed"] for r in subset),
            L1_incomplete_rankings=sum(not r["L1_ranking_complete"] for r in subset),
            MSE_incomplete_rankings=sum(not r["MSE_ranking_complete"] for r in subset)))

    destination = output/"analysis/diagnostics"; destination.mkdir(parents=True, exist_ok=True)
    tables = dict(solver_calls=solver_rows, selector_changes=selectors, t_effects=t_rows,
                  interactions=interaction_rows, seed_changes=seed_rows, coverage=coverage_rows)
    for name, rows in tables.items():
        write_csv(destination/(name+".csv"), rows)
    summary = dict(passed=True, complete=True, new_native_verified=96, reused_parent_verified=96,
        logical_models=192, selector_views=384, new_fit_calls=0, new_native_optimizer_calls=0,
        observations_opened=False, truth_files_opened=False, script_sha256=sha(__file__),
        protocol_sha256=sha(output/"protocol.json"), catalog_sha256=sha(output/"catalog.json"),
        native_analysis_verification_sha256=sha(output/"analysis/verification.json"),
        native_analyzer_sha256=verification["analyzer_sha256"],
        parent_analysis_verification_sha256=sha(parent/"analysis/verification.json"),
        per_input_verification_sha256=verification_hashes, native_result_sha256=source_hashes,
        output_sha256={name+".csv": sha(destination/(name+".csv")) for name in tables},
        solver=solver_summary, selectors=selector_summary, across_loss_seeds=seed_summary,
        within_loss_T_seed_changes=within_loss_seed_changes, coverage=coverage_summary,
        T_F1=contrast_summary(t_rows, ("fit_loss", "view"), "delta_F1"),
        T_exact=contrast_summary(t_rows, ("fit_loss", "view"), "delta_exact"),
        T_seconds=contrast_summary(t_rows, ("fit_loss", "view"), "delta_seconds"),
        F1_interaction=contrast_summary(interaction_rows, ("view",), "F1_interaction"),
        exact_interaction=contrast_summary(interaction_rows, ("view",), "exact_interaction"),
        field_definitions=dict(
            passed="all descriptive inputs remain SHA-anchored to complete native verification; structural status9 is reported rather than hidden",
            solver_gap="max(0,incumbent-bound)/abs(incumbent); native objective units; E included; zero-incumbent nonzero gap flagged separately",
            saved_point_changed="the selected output-library index differs; topology may remain identical",
            parameter_fit_changed="exact saved support/r/x tuple differs; no refit",
            topology_changed="positive (>1e-9) proper internal support family differs, matching audited F1",
            T_delta="T003 minus noT within the same training loss and validation view",
            interaction="(MSE:T003-noT) minus (L1:T003-noT), computed per fixed input",
            seed_change="RNJ selected small-seed family differs across training losses; rankings are data-only but loss-dependent",
            raw_pool="saved own raw trajectory models only, not the full feasible search domain",
            output_pool="same own path after .24 contraction and training refit",
            union_coverage="all true connections occur somewhere across saved raw/output models; this does not assert a single true full model",
            output_mapping="both contraction and subsequent refitting; not attributed to contraction alone",
            misses_available_truth="an exact model is in the fitted output pool but that validation view selects a different topology",
            timing="successful algorithm wall plus per-method charged common costs; parent infrastructure charges remain in native analysis; solver_seconds is a subset"),
        interpretation="descriptive paired diagnostics on four fixed networks and four globally shared seed groups; no tuning or independent causal identification",
        elapsed_seconds=perf_counter()-started)
    write(destination/"summary.json", summary)
    print(json.dumps(dict(passed=True, complete=True, destination=str(destination),
                         solver_calls=len(solver_rows), logical_models=192, new_native_optimizer_calls=0)))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    diagnostics(parser.parse_args().output)
