"""Summarize completed, saved post-identification audit; no fitting or solvers."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "audit_v1.json"
OUT = ROOT / "mechanism_summary_v1.json"
REPORT = ROOT / "RESULTS.md"
METHODS = (
    "current_rebuilt_rnj", "current_root_augmented_rnj", "current_pool_wzzt",
    "all_legal_family_reference",
)
NOISES = ("voltage_only", "power_only", "combined_reference")


def summarize(a):
    rows = a["evaluations"]
    tables = [r for r in a["groups"] if r["method"] in METHODS]
    comparisons = []
    for noise in NOISES:
        for policy in ("zero_sham", "pair_random"):
            relevant = [r for r in a["same_input_method_comparisons"]
                        if r["noise_id"] == noise and r["policy"] == policy
                        and r["method"] == "current_pool_wzzt"
                        and r["reference"] == "current_root_augmented_rnj"]
            differences = [r["final_error_difference"] for r in relevant if r["comparable"]]
            comparisons.append(dict(noise_id=noise, policy=policy,
                                    planned_named_cases=len(relevant), comparable=len(differences),
                                    better=sum(x < 0 for x in differences),
                                    same=sum(x == 0 for x in differences),
                                    worse=sum(x > 0 for x in differences)))
    details = []
    selection = []
    for noise in NOISES:
        for method in METHODS:
            rs = [r for r in rows if r["noise_id"] == noise and r["policy"] == "pair_random"]
            selection.append(dict(
                noise_id=noise, method=method,
                missing_block_stage_counts=dict(Counter(
                    t["stage"] for r in rs for t in r["methods"][method].get("trace", []))),
                missing_block_selection_detail_counts=dict(Counter(
                    t["selection_detail"] for r in rs for t in r["methods"][method].get("trace", [])))))
    for row in rows:
        if row["policy"] != "pair_random":
            continue
        method = row["methods"]["current_root_augmented_rnj"]
        if method["true_removed_by_contract"] or method["false_removed_by_contract"]:
            details.append(dict(case_id=row["case_id"], noise_id=row["noise_id"],
                                raw=method["raw"], final=method["final"],
                                true_removed=method["true_removed_by_contract"],
                                false_removed=method["false_removed_by_contract"]))
    return dict(
        source_path=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        independent_audit_status=a["status"], independent_audit_checks=a["checks"],
        unique_snapshots=a["unique_snapshots"], evaluator_mappings=len(rows),
        case_denominators=a["case_denominators"], tables=tables,
        W_vs_formal_RNJ_root=comparisons,
        DR_vs_zero=[r for r in a["DR_vs_zero_summary"] if r["method"] in METHODS],
        missing_block_stage_and_selection=selection,
        formal_RNJ_root_contract_changes=details,
        expanded_same_as_current_pool_counts=dict(Counter(str(r["expanded_equals_current_pool"]) for r in rows)),
        discovery_status_counts=dict(Counter(r["discovery"]["status"] for r in rows)),
        actual_payment_counts=dict(Counter(str(r["actual_payment"]) for r in rows)),
        new_solver_calls=0,
        interpretations=[
            "Counts are named conditional models with one shared random stream, not estimated recovery probabilities.",
            "Each noise regime has 12 named cases but 10 distinct generating R/X models; c=0 has three duplicate names.",
            "Formal root-augmented RNJ matches W final support error casewise in power_only and combined_reference DR.",
            "Voltage_only formal RNJ+root loses true blocks at contraction; power_only contraction also correctly removes false roots.",
            "Saved exact-support points not selected indicate selection among saved fits, not global identifiability or optimality conclusions.",
            "All discovery outputs remain unresolved because the predeclared observation error model is unclosed; this is not a tested noisy candidate claim.",
            "Neither comparison proves optimized prices beat simple exploration; the pair-random policy was fixed before the batch.",
        ])


def render(s):
    lines = [
        "# Independent reference-noise mechanism audit", "",
        "Completed saved-output audit only. No new fitting, LP, QP, or AC calls.", "",
        f"{s['unique_snapshots']} snapshots; {s['evaluator_mappings']} truth-policy mappings; "
        f"{s['independent_audit_checks']} independent checks passed. "
        "Every regime has 12 named cases but 10 distinct generating R/X models. "
        "One shared random stream was used, so the following are conditional case counts, not recovery probabilities.", "",
        "## Raw and final support results", "",
        "Errors are complete missed-plus-extra effective blocks; full denotes zero errors. "
        "The same current calibration, 3% validation selection, and .16 contraction were replayed.", "",
        "| Noise | Policy | Method | Raw errors | Final errors | Raw full | Final full |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in s["tables"]:
        lines.append(f"| {r['noise_id']} | {r['policy']} | {r['method']} | "
                     f"{r['raw_errors']} | {r['final_errors']} | {r['raw_full']}/{r['planned']} | "
                     f"{r['final_full']}/{r['planned']} |")
    lines += ["", "## WZZT versus same-information formal RNJ plus optional root", "",
              "| Noise | Policy | W better | Same | W worse |",
              "|---|---|---:|---:|---:|"]
    for r in s["W_vs_formal_RNJ_root"]:
        lines.append(f"| {r['noise_id']} | {r['policy']} | {r['better']} | {r['same']} | {r['worse']} |")
    lines += ["", "The old RNJ result without an optional common root cannot isolate a WZZT advantage. "
              "Under power-only and combined reference noise, the formal RNJ+root and WZZT final errors "
              "are equal in every DR case. Under voltage-only noise, neither method dominates.", "",
              "## Where the missing blocks were lost", "",
              "In voltage-only DR, current WZZT had a saved exact-support family in eight named cases "
              "where final output missed a true block: five lost to the 3% simplicity rule; "
              "three duplicate c=0 cases selected another eligible equal-size family by validation/index. "
              "Two other missing blocks were positive in some saved fit but no complete true family was saved. "
              "One additional true root block was removed at contraction. These are stored-path observations, "
              "not a proof that the omitted truth is globally excluded.", "",
              "For formal RNJ+root in voltage-only DR, raw complete recovery was 8/12 and final 3/12. "
              "Contraction removed eight true blocks: seven full-root blocks (including the three duplicate "
              "c=0 names) and one strong internal block. The same rule helped under power-only DR: "
              "raw 6/12 became final 9/12 because three false roots in the b=0 cases were removed. "
              "Uniformly reducing the contraction threshold is therefore unsupported.", "",
              "For combined reference-noise DR, current WZZT missed eight true blocks: six after "
              "3% simplicity chose a smaller saved family, one positive in a saved fit but not in the "
              "raw selected point, and one absent from the fitted/pool support slots. "
              "The latter describes only this candidate source, not mathematical impossibility.", "",
              "## Scope", "",
              "All 72 mappings had unresolved direct-control discovery and unchanged expanded/current pools. "
              "The source requires a predeclared response/model-error band and none was supplied in these "
              "regimes. Thus this run evaluates raw randomized DR input and current identifiers; it does "
              "not validate a noisy code-derived candidate rule or a new price optimizer.", "",
              "The data establish a useful conditional DR effect versus the matched no-reward observations "
              "in this deliberately synchronized input example. They do not establish a WZZT-specific "
              "price-design benefit, multi-seed recovery probability, or general network performance.", "",
              "Machine-readable per-case block traces and replay checks remain in audit_v1.json. "
              "mechanism_summary_v1.json contains the compact aggregate and formal-root contraction cases.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("Keep prior version; choose a new version before rewriting evidence.")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source["status"] != "passed" or source["failed_checks"]:
        raise ValueError("A failed audit cannot be interpreted as verified.")
    result = summarize(source)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render(result), encoding="utf-8")
    print(json.dumps(dict(status="saved", out=str(OUT), report=str(REPORT), new_solver_calls=0)))
