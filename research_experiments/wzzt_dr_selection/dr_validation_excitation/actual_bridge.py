"""Freeze continuous validation-price plans and reuse the existing AC/RNJ+W chain.

This module does not implement physics or topology identification.  Freezing
reads the old *historical* case protocol/problem/plans and the new planning
receipt only.  It never opens an old/new actual observation or truth file.

The original linear-g fields required by the existing physical worker remain
in its compatibility problem/plan; they are explicitly obsolete diagnostics.
The primary new excitation event is post-training-refit 3% validation exclusion.
Actual RNJ+WZZT still estimates all training weights and freezes them on validation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import zipfile


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=_sha(path))


def _checked(item):
    path = Path(item["path"]).resolve()
    if _sha(path) != item["sha256"]:
        raise RuntimeError("frozen input hash changed: "+str(path))
    return path


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def _old_case_root(root):
    root = Path(root).resolve()
    # Both the old master and its explicitly named main case are accepted.
    if (root/"main/protocol.json").is_file():
        return root/"main"
    if (root/"protocol.json").is_file():
        return root
    raise FileNotFoundError("old main-case historical protocol unavailable")


def _setup(repo):
    repo = Path(repo).resolve()
    for folder in (repo, repo/"rnj_wzzt_core"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    return repo


def _target_labels(old):
    if old.get("target_labels"):
        return list(old["target_labels"])
    baseline = _read(_checked(old["historical_baseline_result"]))
    competition = old["competition"]
    coarse = baseline["selection_candidates"][competition["selected_index"]]
    fine = baseline["selection_candidates"][competition["alternative_index"]]
    family = lambda p: p.get("full_fit", {}).get("supports", p.get("native_family", p["supports"]))
    added = set(map(tuple, family(fine)))-set(map(tuple, family(coarse)))
    if len(added) != 1:
        return []
    support = next(iter(added))
    return sorted(old["labels"][i] for i in support)


def freeze_actual(old_source_root, planning_root, selected_amplitudes16,
                  frozen_amplitudes16_or_None, output_root, extra_sources=()):
    """Freeze a bounded same-case independent actual pilot, without running it.

    ``planning_root/protocol.json`` must already be frozen and ``summary.json``
    must exist.  A summary field ``selected_refit_sample90_feasible: true`` can
    record a verified planning-sample gate; its absence/false value labels this
    actual pilot ``diagnostic_phaseI_not90_feasible``.  This flag never claims
    field90%, nor 90% topology recovery.

    At most four policy names are emitted: zero, rotation,
    continuous_validation and (if genuinely distinct) historical_frozen.  A
    duplicate frozen-price plan is recorded as an alias and not rerun.
    """
    old_root = _old_case_root(old_source_root)
    planning_root = Path(planning_root).resolve()
    output = Path(output_root).resolve()
    old = _read(old_root/"protocol.json")
    repo = _setup(old["repository"])
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test.prepare import (
        _plan, _sources,
    )
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem
    import numpy as np

    planning_protocol_path = planning_root/"protocol.json"
    planning_summary_path = planning_root/"summary.json"
    planning_protocol = _read(planning_protocol_path)
    planning_summary = _read(planning_summary_path)
    if planning_protocol.get("status") != "frozen":
        raise ValueError("freeze the planning protocol before actual prices")
    if old.get("status") != "frozen" or old.get("case") != "flynn16":
        raise ValueError("this bounded bridge reuses the frozen flynn16 case")
    if old.get("hard_four_event_budget") != 1.6:
        raise ValueError("preserve the original four-event budget1.6")
    if old.get("original_history_rows") != [288, 288] or old.get("future_rows") != [480, 480]:
        raise ValueError("preserve complete288/288 history and480/480 combined data")
    _checked(old["history_packet"])
    _checked(old["source_metadata"])
    _checked(old["historical_baseline_result"])
    problem_path = _checked(old["problem"])
    problem = PriceProblem.from_dict(_read(problem_path))
    if problem.n_contracts != 16 or problem.budget != 1.6:
        raise ValueError("the exact old16-contract/1.6 resource domain is required")
    zero = _read(_checked(old["plans"]["zero"]))
    rotation = _read(_checked(old["plans"]["rotation"]))
    resources = zero["resources_by_event"][0]
    modules = zero["modules_by_event"][0]
    local_d = zero["response_factors_by_event"][0]
    if len(modules) != 8 or len(local_d) != 8:
        raise ValueError("old canonical resource/type declarations differ")

    def vector(value, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (16,) or not np.isfinite(result).all() or (result < 0).any():
            raise ValueError(name+" must be16 finite nonnegative expected amplitudes")
        return result

    selected = vector(selected_amplitudes16, "selected_amplitudes16")
    vectors = dict(zero=np.zeros(16),
                   rotation=vector(rotation["global_expected_amplitudes"], "old_rotation"),
                   continuous_validation=selected)
    aliases = {}
    if frozen_amplitudes16_or_None is not None:
        frozen = vector(frozen_amplitudes16_or_None, "frozen_amplitudes16")
        alias = next((name for name, value in vectors.items()
                      if np.allclose(frozen, value, rtol=0., atol=1e-12)), None)
        if alias is None:
            vectors["historical_frozen"] = frozen
        else:
            aliases["historical_frozen"] = alias

    # Validate every price before creating a new actual root.  _plan reuses all
    # old convex caps/device/exclusivity checks but does not run the optimizer.
    payloads, checks = {}, {}
    for policy, amplitudes in vectors.items():
        if any(np.count_nonzero(amplitudes[list(group)] > 0.) > 1
               for group in problem.exclusive_groups):
            raise ValueError("worker requires exactly zero inactive contract types")
        payload, check = _plan(problem, resources, modules, local_d,
                               amplitudes, policy, threshold=0.)
        if check["hard_budget_bound"] > 1.6+1e-8:
            raise ValueError("worker hard-budget tolerance exceeded")
        payload.update(
            primary_excitation="post-training-refit3pct_validation_exclusion",
            legacy_linear_excitation_obsolete=True,
            legacy_excitation_scope="compatibility receipt only; not new design objective",
        )
        payloads[policy], checks[policy] = payload, check

    output.mkdir(parents=True, exist_ok=False)
    started = dict(status="started", created_utc=_now(),
                   old_case_protocol=_ref(old_root/"protocol.json"),
                   planning_protocol=_ref(planning_protocol_path),
                   planning_summary=_ref(planning_summary_path),
                   graph_or_future_or_truth_parsed=False, new_AC=0, new_identifier=0)
    _write(output/"freeze_receipt.json", started)
    try:
        extra = {str(Path(p).resolve()) for p in extra_sources}
        extra.update(str(p.resolve()) for p in Path(__file__).parent.glob("*.py"))
        extra.update(str(p.resolve()) for p in
                     (repo/"research_experiments/wzzt_dr_selection/dr_validation_excitation").glob("*.py"))
        extra.add(str(Path(__file__).resolve()))
        sources, algorithm = _sources(repo, [dict(source_root=old["source_root"])], sorted(extra))
        if algorithm != old["algorithm_source_sha256"]:
            raise RuntimeError("original identifier source changed; do not call this an unchanged-algorithm pilot")
        plan_refs, details = {}, {}
        for policy, payload in payloads.items():
            file = output/"plans"/(policy+".json")
            _write(file, payload)
            plan_refs[policy] = _ref(file)
            check = checks[policy]
            details[policy] = dict(
                expected_payment=check["expected_cost"],
                hard_budget_bound=check["hard_budget_bound"],
                positive_contract_ids=[problem.contract_ids[j] for j, a in enumerate(vectors[policy]) if a > 1e-6],
                legacy_linear_reliable_benefit=check["reliable_benefit"],
            )
        source_snapshot = output/"source_snapshot.zip"
        with zipfile.ZipFile(source_snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
            for index, file in enumerate(sorted(sources)):
                archive.write(file, arcname=f"{index:03d}_"+Path(file).name)
        sample90 = planning_summary.get("selected_refit_sample90_feasible") is True
        quote_status = ("planning_sample90_feasible_pending_independent_effects" if sample90
                        else "diagnostic_phaseI_not90_feasible")
        c = deepcopy(old)
        c.update(
            id="continuous_validation_price_actual_v1", created_utc=_now(),
            repository=str(repo), plans=plan_refs, policies=list(vectors), policy_details=details,
            old_case_protocol=_ref(old_root/"protocol.json"),
            planning_protocol=_ref(planning_protocol_path), planning_summary=_ref(planning_summary_path),
            source_sha256=sources, algorithm_source_sha256=algorithm,
            source_snapshot=_ref(source_snapshot),
            test_paths=[1200, 1201], response_seed=2026110133, replicate_base=470000,
            event_profiles=[0, 2, 0, 2], workers=2,
            maximum_unique_fits=len(vectors)*2, maximum_requested_records=len(vectors)*2,
            fit_worker_wall_seconds=3600, excitation_threshold=0.,
            target_labels=_target_labels(old),
            quote_feasibility_status=quote_status,
            selected_refit_sample90_feasible=sample90,
            diagnostic_actual_pilot_allowed_if_sample90_unmet=True,
            probability_scope="90pct post-training-refit validation-exclusion excitation event only; not topology recovery",
            legacy_linear_excitation_obsolete=True,
            legacy_physical_excitation_scope="oldlinear_g compatibility diagnostics only; not primary pass/fail",
            parameter_protocol="all original identifier weights refit on full training, fixed on validation; historical_frozen names the price-planning variant only",
            case_selection="same already-declared main case; no new case/impedance/noise/resource screening",
            no_truth_or_actual_future_in_planning=True, all_available_history_used=True,
            policy_aliases=aliases,
        )
        _write(output/"protocol.json", c)
        freeze = dict(status="frozen", created_utc=_now(),
                      protocol=_ref(output/"protocol.json"), plans=plan_refs,
                      planning_protocol=_ref(planning_protocol_path),
                      planning_summary=_ref(planning_summary_path),
                      source_snapshot=_ref(source_snapshot),
                      selected_amplitudes=selected.tolist(), policies=list(vectors),
                      policy_aliases=aliases, quote_feasibility_status=quote_status,
                      prices_frozen_before_actual=True, graph_or_future_or_truth_parsed=False,
                      actual_test_paths=[1200, 1201], new_AC=0, new_identifier=0)
        _write(output/"price_freeze.json", freeze)
        _write(output/"freeze_receipt.json", dict(started, status="complete", completed_utc=_now(),
                   price_freeze=_ref(output/"price_freeze.json"),
                   source_snapshot=_ref(source_snapshot), quote_feasibility_status=quote_status))
        return freeze
    except Exception as error:
        _write(output/"freeze_receipt.json", dict(started, status="failed", completed_utc=_now(),
                   error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
                   failure_retained=True))
        raise


def run_actual(root, phase, worker=0):
    """Delegate actual stages; no physics, fitting or selection is reimplemented."""
    root = Path(root).resolve()
    c = _read(root/"protocol.json")
    _setup(c["repository"])
    freeze = _read(root/"price_freeze.json")
    _checked(freeze["protocol"])
    _checked(freeze["planning_protocol"])
    _checked(freeze["planning_summary"])
    _checked(freeze["source_snapshot"])
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test import driver
    if phase == "observe":
        return driver.observe(root)
    if phase == "fit":
        if worker not in (0, 1):
            raise ValueError("fit worker must be0 or1")
        return driver.fit(root, worker)
    if phase == "finish":
        return driver.finish(root)
    if phase != "aggregate":
        raise ValueError("phase must be observe, fit, finish or aggregate")
    receipt = root/"aggregate_description_receipt.json"
    if receipt.exists():
        raise FileExistsError("aggregate adjustment already completed; keep original receipt")
    original = driver.aggregate(root)
    original_path = root/"summary_original_aggregate.json"
    if original_path.exists():
        raise FileExistsError("original aggregate already retained")
    # Only descriptive fields are changed.  Raw actual results/denominators,
    # error/cost tables and paired comparisons remain byte-value identical.
    _write(original_path, original)
    adjusted = dict(original,
        planner_scope="bounded continuous prices for complete-family joint-training-refit3pct validation exclusion; frozen historical-parameter guidance is a named comparison",
        confidence_scope="90pct excitation requirement only; two independent actual paths do not certify90pct or topology recovery",
        legacy_linear_excitation_scope="compatibility physical receipt only, obsolete as design/effect criterion",
        quote_feasibility_status=c["quote_feasibility_status"],
        selected_refit_sample90_feasible=c["selected_refit_sample90_feasible"],
        policy_aliases=c["policy_aliases"],
    )
    changed = {key for key in adjusted if adjusted.get(key) != original.get(key)}
    allowed = {"planner_scope", "confidence_scope", "legacy_linear_excitation_scope",
               "quote_feasibility_status", "selected_refit_sample90_feasible", "policy_aliases"}
    if not changed <= allowed:
        raise AssertionError("actual result fields changed")
    _write(root/"summary.json", adjusted)
    _write(receipt, dict(status="complete", original=_ref(original_path),
                         adjusted=_ref(root/"summary.json"), changed_descriptive_fields=sorted(changed),
                         new_solver_calls=0,new_AC=0))
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("observe", "fit", "finish", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    run_actual(args.root, args.phase, args.worker)


if __name__ == "__main__":
    main()
