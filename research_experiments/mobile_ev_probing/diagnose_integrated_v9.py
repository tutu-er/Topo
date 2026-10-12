"""Post-hoc v9 diagnostics: fixed-bank action evidence and AC deletion witnesses.

No LP/NLP is solved. Simulation truth is used only for offline diagnosis, never
for changing an action, candidate bank, measurement model, or reported bound.
Numerical AC witnesses are not interval existence proofs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import ac_general_v9 as ac
from audit_integrated_v9 import independent_nodal_ac


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
        else v.item() if isinstance(v, np.generic) else str(v))+"\n", encoding="utf-8")


def lower(record):
    exact = record.get("exact_lower")
    if exact is not None:
        return Fraction(exact)
    value = record.get("lower")
    return None if value is None else Fraction(float(value))


def _score_for(design, key):
    kind, *sites = key.split("_")
    sites = tuple(map(int, sites))
    same_kind = [s for s in design.get("scores", []) if s["kind"] == kind]
    item = next((s for s in same_kind if tuple(s["sites"]) == sites), None)
    if item is None:
        return None
    value = float(item["score"])
    return dict(score=value, rank_strict=1+sum(float(s["score"]) > value for s in same_kind),
                rank_last_tie=sum(float(s["score"]) >= value for s in same_kind),
                action_grid_count=len(same_kind), exact_score_ties=True)


def fixed_bank_diagnostics(result):
    """Compare recorded bounds only on the SAME pre-EV candidate ID set."""
    initial = {r["id"] for r in result["proposals"]["candidates"]}
    truth_ids = {key for key in initial
                 if result["candidate_truth_metrics"][key]["exact"]}
    wrong_ids = initial-truth_ids
    by_action = {}
    for key, models in result["records"].items():
        by_action[key] = {}
        for model, outcome in models.items():
            missing = initial-set(outcome["records"])
            if missing:
                raise ValueError(f"Initial bank missing from {key}/{model}: {missing}")
            values = {h: lower(outcome["records"][h]) for h in initial}
            excluded = {h for h, v in values.items() if v is not None and v > 0}
            statuses = {h for h in initial if outcome["records"][h]["status"] == "excluded"}
            if statuses != excluded:
                raise ValueError("Recorded exclusion differs from positive verified lower bound")
            known_wrong = [values[h] for h in wrong_ids if values[h] is not None]
            by_action[key][model] = dict(
                initial_bank_count=len(initial), excluded_count=len(excluded),
                retained_count=len(initial-excluded), excluded_ids=sorted(excluded),
                retained_ids=sorted(initial-excluded), true_ids_excluded=sorted(truth_ids & excluded),
                unknown_bound_ids=sorted(h for h, v in values.items() if v is None),
                min_wrong_lower=None if len(known_wrong) != len(wrong_ids) or not known_wrong
                    else float(min(known_wrong)),
                wrong_candidate_lower={h: None if values[h] is None else float(values[h])
                                       for h in sorted(wrong_ids)},
                updated_bank_count=len(outcome["records"]),
                design=_score_for(result["design"], key))

    policies = {label: dict(action=value["action"], evidence=by_action[value["action"]])
                for label, value in result["policies"].items()}
    comparisons = []
    for first, second in (("selected_single", "fixed_single"),
                          ("selected_single", "matched_constant"),
                          ("selected_mobile", "fixed_mobile"),
                          ("selected_single", "no_EV"),
                          ("selected_mobile", "no_EV"),
                          ("selected_mobile", "selected_single")):
        if first not in policies or second not in policies:
            continue
        for model in policies[first]["evidence"]:
            left, right = (policies[label]["evidence"][model] for label in (first, second))
            le, re = set(left["excluded_ids"]), set(right["excluded_ids"])
            lm, rm = left["min_wrong_lower"], right["min_wrong_lower"]
            ls, rs = left["design"], right["design"]
            comparisons.append(dict(
                policy_a=first, policy_b=second, model=model,
                same_action=policies[first]["action"] == policies[second]["action"],
                a_minus_b_exclusion_count=len(le)-len(re),
                excluded_only_by_a=sorted(le-re), excluded_only_by_b=sorted(re-le),
                a_strictly_dominated_by_b_on_exclusion_set=le < re,
                a_excludes_fewer=len(le) < len(re),
                a_minus_b_min_wrong_lower=None if lm is None or rm is None else lm-rm,
                score_a=None if ls is None else ls["score"],
                score_b=None if rs is None else rs["score"],
                score_prefers_a_but_a_excludes_fewer=bool(ls and rs and
                    ls["score"] > rs["score"] and len(le) < len(re)),
                score_tied_but_evidence_counts_differ=bool(ls and rs and
                    ls["score"] == rs["score"] and len(le) != len(re)),
                resource_note="All EV actions charge 27 kWh; mobile also consumes 2 kWh travel."
                    if "mobile" in first+second else "Same site/energy for matched constant; no EV has zero EV energy."))

    # Spearman is intentionally omitted: only a chosen subset of actions was
    # executed, and multiple policy labels may refer to the same physical action.
    return dict(comparison_domain="fixed initial candidate ID bank, never varying updated banks",
                initial_ids=sorted(initial), true_ids_in_initial_bank=sorted(truth_ids),
                truth_in_initial_bank=bool(truth_ids), actions=by_action, policies=policies,
                comparisons=comparisons,
                selection_scope="Scores cover the full declared action grid; actual evidence covers only saved executed/comparison actions.",
                used_nonwitness_references=result["design"].get("used_nonwitness_references"),
                references_verified_in_shared_gain_domain=result["design"].get("references_verified_in_shared_gain_domain", False))


def _box(candidate):
    """Frozen v9 declared domain; checked against the driver source hash."""
    return np.array([(.002, .014) if len(s) == candidate.n else
                     ((.03, .10) if len(s) == 1 else (1e-6, .04))
                     for s in candidate.supports])


def deletion_witness(observed, truth, remove=(0, 1)):
    """Remove one physical pair edge, keep every other parameter and latent PQ.

    Required jitter is evaluated at saved gains and at an explicit refitted
    gain interval. These are evaluation-only nuisance choices, not information
    available to the identification rule.
    Failing this construction is NOT proof that no other AC witness exists.
    """
    original = truth["candidate"]
    n = int(original["n_terminal"])
    supports = [tuple(s) for s in original["supports"]]
    if remove not in supports:
        return dict(status="deleted_edge_not_present", removed_support=remove)
    alternative = ac.RadialCandidate.from_supports("posthoc_delete_pair_0_1", n,
                       [s for s in supports if s != remove])
    mapping = dict(zip(supports, truth["theta"]))
    theta = np.array([mapping[s] for s in alternative.supports])
    p, q = np.asarray(truth["P"]), np.asarray(truth["Q"])
    source = np.asarray(truth["source_p"])
    settings = truth["meter_settings"]
    gains = np.asarray(truth["meter_gains"])
    measured = np.asarray(truth["voltage_measured"])
    root_measured = np.asarray(truth["root_voltage_measured"])
    info = dict(removed_support=remove, removed_theta=mapping[remove],
        candidate=alternative.metadata(), theta=theta.tolist(),
        source_q_rule="source_q=.3*source_p, matching the frozen generator",
        evaluation_only=True, pq_choice="saved physical true PQ, verified inside observed PQ boxes",
        gain_choice="Both saved gains and a concrete constant-gain refit are checked below.",
        numerical_AC_equation_tolerance=1e-11,
        no_witness_meaning="Only this explicit deletion construction failed; no impossibility claim follows.")
    try:
        y_alt, voltage, residual = ac.ac_batch(alternative, theta, p, q, source, .3*source, kappa=.5)
        independent_y, independent_voltage, nodal_residual = independent_nodal_ac(
            alternative.metadata(), theta, p, q, source)
    except (RuntimeError, ValueError) as error:
        return dict(info, status="construction_numerical_AC_failed", error=str(error),
                    compatible=False)
    vm = abs(voltage[:, alternative.terminal_edges])
    required = measured-gains[None, 1:]*vm
    root_required = root_measured-gains[0]
    terminal_budget = np.broadcast_to(np.asarray(settings["terminal_jitter"]), (n,))
    root_budget = float(settings["root_jitter"])
    jitter_margin = terminal_budget[None, :]-abs(required)
    root_margin = root_budget-abs(root_required)
    bp, bq = np.asarray(observed["P_error"]), np.asarray(observed["Q_error"])
    p_margin = bp-abs(np.asarray(observed["P"])-p)
    q_margin = bq-abs(np.asarray(observed["Q"])-q)
    glo, ghi = settings["gain_bounds"]
    box = _box(alternative)
    checks = dict(
        topology_differs=tuple(alternative.supports) != tuple(supports),
        theta_in_declared_domain=bool(np.all((theta >= box[:, 0]) & (theta <= box[:, 1]))),
        gains_in_declared_domain=bool(np.all((gains >= glo) & (gains <= ghi))),
        source_p_in_domain=bool(np.all(abs(source) <= observed["source_p_abs_cap"])),
        source_q_in_domain=bool(np.all(abs(.3*source) <= observed["source_q_abs_cap"])),
        common_source_effective_in_domain=bool(np.all(abs(1.15*source-observed["source_nominal_eff"])
                                             <= observed["source_error_eff"])),
        PQ_inside_measurement_boxes=bool(np.min(p_margin) >= 0 and np.min(q_margin) >= 0),
        all_bus_voltage_inside_domain=bool(abs(voltage).min() >= .9 and abs(voltage).max() <= settings["vmax"]),
        root_jitter_inside_declared_budget=bool(np.min(root_margin) >= 0),
        terminal_jitter_inside_declared_budget=bool(np.min(jitter_margin) >= 0),
        numerical_AC_equations=bool(residual <= 1e-11),
        independent_nodal_equations=bool(nodal_residual <= 1e-8),
        independent_AC_agreement=bool(np.max(abs(independent_voltage-voltage)) <= 2e-11
                                     and np.max(abs(independent_y-y_alt)) <= 2e-11))
    hlen = int(truth["history_count"])
    segments = {}
    for name, subset in (("history", slice(0, hlen)), ("new", slice(hlen, None)), ("joint", slice(None))):
        margin = jitter_margin[subset]
        segments[name] = dict(terminal_jitter_margin_min=float(margin.min()),
            root_jitter_margin_min=float(root_margin[subset].min()),
            violating_terminal_samples=int(np.sum(margin < 0)),
            terminal_jitter_feasible=bool(margin.min() >= 0),
            max_voltage_change=float(np.max(abs(vm[subset]-np.asarray(truth["voltage_terminal"])[subset]))),
            max_squared_drop_change=float(np.max(abs(y_alt[subset]-np.asarray(truth["Y"])[subset]))))
    compatible = all(checks.values())
    fixed = dict(status="numerical_original_AC_competing_witness" if compatible else "explicit_witness_not_found",
                compatible=compatible, checks=checks,
                gains=gains.tolist(), segments=segments,
                min_joint_jitter_margin=float(min(jitter_margin.min(), root_margin.min())),
                near_jitter_boundary=bool(compatible and min(jitter_margin.min(), root_margin.min()) < 1e-10),
                required_terminal_jitter=required.tolist(), required_root_jitter=root_required.tolist(),
                gain_choice="saved physical true gains, unchanged over all rows")

    # For a fixed AC trajectory, each original magnitude equation is affine
    # in one constant gain. Intersect every time-sample interval, then build a
    # concrete witness. Empty intersections remain an explicit negative result.
    full_v = np.c_[np.ones(len(vm)), vm]
    full_measured = np.c_[root_measured, measured]
    budgets = np.r_[root_budget, terminal_budget]
    interval_lo = np.maximum(glo, np.max((full_measured-budgets)/full_v, axis=0))
    interval_hi = np.minimum(ghi, np.min((full_measured+budgets)/full_v, axis=0))
    intervals_nonempty = bool(np.all(interval_lo <= interval_hi))
    fitted_gain = (interval_lo+interval_hi)/2
    fitted_required = full_measured-full_v*fitted_gain
    fitted_margin = budgets[None, :]-abs(fitted_required)
    fitted_checks = dict(checks,
        constant_gain_intersections_nonempty=intervals_nonempty,
        gains_in_declared_domain=bool(np.all((fitted_gain >= glo) & (fitted_gain <= ghi))),
        root_jitter_inside_declared_budget=bool(fitted_margin[:, 0].min() >= 0),
        terminal_jitter_inside_declared_budget=bool(fitted_margin[:, 1:].min() >= 0))
    fitted_segments = {}
    for name, subset in (("history", slice(0, hlen)), ("new", slice(hlen, None)), ("joint", slice(None))):
        margin = fitted_margin[subset]
        fitted_segments[name] = dict(segments[name],
            terminal_jitter_margin_min=float(margin[:, 1:].min()),
            root_jitter_margin_min=float(margin[:, 0].min()),
            violating_terminal_samples=int(np.sum(margin[:, 1:] < 0)),
            terminal_jitter_feasible=bool(margin[:, 1:].min() >= 0))
    fitted_compatible = all(fitted_checks.values())
    refitted = dict(status="numerical_original_AC_competing_witness" if fitted_compatible else "explicit_witness_not_found",
        compatible=fitted_compatible, checks=fitted_checks,
        gains=fitted_gain.tolist(), gain_interval_lower=interval_lo.tolist(),
        gain_interval_upper=interval_hi.tolist(),
        min_gain_interval_width=float(np.min(interval_hi-interval_lo)),
        min_joint_jitter_margin=float(fitted_margin.min()),
        near_jitter_boundary=bool(fitted_compatible and fitted_margin.min() < 1e-10),
        narrow_nonconstant_gain_intersection=bool(ghi > glo and intervals_nonempty
                                                  and np.min(interval_hi-interval_lo) < 1e-10),
        segments=fitted_segments, required_terminal_jitter=fitted_required[:, 1:].tolist(),
        required_root_jitter=fitted_required[:, 0].tolist(),
        gain_choice="One constant gain per root/terminal, intersection across ALL historical and new magnitude equations.")
    return dict(info, status="numerical_original_AC_competing_witness" if compatible or fitted_compatible else "explicit_witness_not_found",
        compatible=compatible or fitted_compatible,
        fixed_gain_witness=fixed, refitted_gain_witness=refitted,
        min_voltage=float(abs(voltage).min()), max_voltage=float(abs(voltage).max()),
        AC_equation_residual=float(residual), independent_nodal_residual=float(nodal_residual),
        independent_nodal_source_sha256=sha(HERE/"audit_integrated_v9.py"),
        min_P_box_margin=float(p_margin.min()), min_Q_box_margin=float(q_margin.min()),
        alternative_voltage_magnitude=vm.tolist(),
        certificate_note="Numerical residual/domain checks only; not an interval-certified AC existence theorem.")


def diagnose_case(folder):
    result_file = folder/"result.json"
    result = read(result_file)
    witnesses = {}
    for key in result["records"]:
        npz, truth_file = folder/(key+".npz"), folder/(key+"_truth.json")
        if sha(npz) != result["inputs"][npz.name] or sha(truth_file) != result["truth_files"][truth_file.name]:
            raise ValueError(f"Recorded input hash mismatch: {folder.name}/{key}")
        with np.load(npz, allow_pickle=False) as data:
            observed = {name: data[name] for name in data.files}
        witnesses[key] = deletion_witness(observed, read(truth_file))
        alternative = witnesses[key].get("candidate")
        if alternative is not None:
            n = alternative["n_terminal"]
            target = {tuple(s) for s in alternative["supports"] if 1 < len(s) < n}
            for label, bank in (("initial", result["proposals"]),
                                ("updated", result.get("post_banks", {}).get(key))):
                witnesses[key][label+"_bank_contains_deleted_pair_alternative"] = (
                    None if bank is None else any(
                        {tuple(s) for s in row["supports"] if 1 < len(s) < n} == target
                        for row in bank["candidates"]))
    return dict(case=result["case"], config=result["config"], result_sha256=sha(result_file),
                comparison=fixed_bank_diagnostics(result), deletion_witnesses=witnesses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    protocol = read(args.directory/"protocol.json")
    expected = protocol.get("source_sha256", {})
    for relative, recorded in expected.items():
        # Protocol may include Windows separators when read on another platform.
        path = ROOT/relative.replace("\\", "/")
        if sha(path) != recorded:
            raise ValueError(f"Frozen source changed: {relative}")
    folders = sorted(path.parent for path in args.directory.glob("*/result.json"))
    complete = len(folders) == len(protocol["cases"])
    if args.require_complete and not complete:
        raise RuntimeError(f"Only {len(folders)}/{len(protocol['cases'])} cases complete")
    results = [diagnose_case(folder) for folder in folders]
    witness_counts = {}
    for result in results:
        group = "weak" if result["config"][1] else "ordinary"
        counts = witness_counts.setdefault(group, dict(physical_batches=0, compatible_batches=0,
            fixed_gain_compatible_batches=0, refitted_gain_compatible_batches=0,
            gain_refit_rescued_batches=0, episodes=0, episodes_with_any_witness=0))
        values = list(result["deletion_witnesses"].values())
        counts["physical_batches"] += len(values)
        counts["compatible_batches"] += sum(v.get("compatible", False) for v in values)
        counts["fixed_gain_compatible_batches"] += sum(v.get("fixed_gain_witness", {}).get("compatible", False) for v in values)
        counts["refitted_gain_compatible_batches"] += sum(v.get("refitted_gain_witness", {}).get("compatible", False) for v in values)
        counts["gain_refit_rescued_batches"] += sum(v.get("refitted_gain_witness", {}).get("compatible", False)
            and not v.get("fixed_gain_witness", {}).get("compatible", False) for v in values)
        counts["episodes"] += 1
        counts["episodes_with_any_witness"] += any(v.get("compatible", False) for v in values)
    output = dict(schema=9, complete=complete, completed_case_count=len(results),
        expected_case_count=len(protocol["cases"]), diagnostic_source_sha256=sha(__file__),
        protocol_sha256=sha(args.directory/"protocol.json"), results=results,
        deletion_witness_counts=witness_counts,
        purpose="Offline diagnosis only. No action/candidate/model modified; no LP or NLP solved.",
        counts_are_not_population_probabilities=True)
    target = args.directory/"diagnostics.json"
    write(target, output)
    print(json.dumps(dict(output=str(target), complete=complete, cases=len(results),
                         deletion_witness_counts=witness_counts)))


if __name__ == "__main__":
    main()
