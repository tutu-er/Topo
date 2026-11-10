"""Export explicit aggregate-only fields from the completed full-price run.

Private schedules, forecasts, P/Q/V, observations, and per-path scores are
deliberately excluded. This does not run an optimizer, simulator, or sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root, destination):
    root, destination = Path(root), Path(destination)
    evaluation = read(root/"evaluation/results.json")
    finished = read(root/"evaluation/complete.json")
    audit = read(root/"evaluation/audit/summary.json")
    optimization = read(root/"optimization_restore/summary.json")
    allocation = read(root/"optimization_restore/allocation_audit.json")
    formulas = read(root/"formula_checks/run_v1/summary.json")
    assert finished["status"] == "complete"
    assert optimization["qualified"] and audit["passed"]
    assert audit["records"] == audit["requested"] == 160
    assert audit["matched_plan_fee_error"] < 1e-8
    assert formulas["checks"] == formulas["passed"] == 89
    assert formulas["failed"] == 0
    stages = []
    for stage in ("optimization", "optimization_continue", "optimization_scaled_v2", "optimization_restore"):
        summary = read(root/stage/"summary.json")
        search = summary.get("search")
        if search is None:
            search = {key:summary[key] for key in ("statuses", "seconds", "price_evaluations")}
        stages.append(dict(stage=stage, search=search,
                           best_probability=summary["refined_probability"],
                           fee=summary["best_fee"], qualified=summary["qualified"]))
    model = {}
    for mode in ("conditional", "independent_full"):
        item = evaluation[mode]
        model[mode] = {key:item[key] for key in
                       ("requested", "completed", "seed", "policies", "paired", "seconds")}
        assert item["completed"] == item["requested"]
    predictions = {name:dict(probability=item["probability"],
                            numerical_pass=item["numerical_pass"],
                            analytic_tail_bound=item["analytic_tail_bound"],
                            certified_error=item["certified_error"])
                   for name, item in evaluation["conditional_predictions"].items()}
    ac = evaluation["actual"]
    ac_public = {key:ac[key] for key in (
        "natural_days", "requested_execution_paths", "requested_policy_observations",
        "observed_policy_records", "complete_policy_records", "failed_or_unresolved_records",
        "seconds", "threshold_micro", "response_seed", "meter_seed", "thermal_limits_checked",
        "policies", "paired")}
    references = [root/"evaluation/complete.json", root/"evaluation/audit/summary.json",
                  root/"optimization_restore/quote.pkl", root/"optimization_restore/request.json",
                  root/"optimization_restore/allocation_audit.json", root/"formula_checks/run_v1/checks.json"]
    output = dict(case="paper15_b2_v0005", status="complete", frozen_parameters=True,
        new_parameter_fits=0, new_W_search=0, full_history_train_rows=288,
        full_history_validation_rows=288, new_day_weight=2, common_validation_band=.03,
        output_contraction=.16, target_probability=.90,
        target_scope="fixed forecast mean and nominal delivery; independent residual voltage innovations",
        optimization_stages=stages,
        quote=dict(probability=optimization["refined_probability"], fee=optimization["best_fee"],
                   qualified=optimization["qualified"], global_optimal=False,
                   refined=optimization["refined"], allocation=allocation["summary"],
                   by_type=allocation["by_type"]),
        conditional_predictions=predictions, model_sampling=model, actual_AC=ac_public,
        independent_audit=audit, formula_verification=formulas,
        private_evidence_sha256={str(path.relative_to(root)):digest(path) for path in references},
        excluded_fields="all quote arrays, forecasts, P/Q/V, natural curves, AC trajectories and per-path observations")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(destination), sha256=digest(destination), records=ac_public["observed_policy_records"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.private_root, args.out)
