"""Audit frozen passive-first inputs/certificates without generating new truth."""
from fractions import Fraction as F
import argparse
import json
from pathlib import Path

import numpy as np

import passive_first_v8 as p


def audit(folder):
    report = json.loads((folder / "passive_results.json").read_text(encoding="utf-8"))
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)
    check("source_hash", p.d.sha(p.__file__) == report["protocol"]["source_hash"])
    check("v7_hash", p.d.sha(p.d.__file__) == report["protocol"]["v7_hash"])
    check("v6_hash", p.d.sha(p.v6.__file__) == report["protocol"]["v6_hash"])
    lp_count, qp_count = 0, 0
    max_upper = 0.
    for ep in report["episodes"]:
        name = ep["name"]
        hi = ep["history_input"]
        check(name + ":history_hash", p.d.sha(folder / hi["path"]) == hi["sha256"])
        with np.load(folder / hi["path"]) as archive:
            history = {k: archive[k].copy() for k in archive.files}
        check(name + ":no_truth_in_history", set(history) == {"P_total", "Y_drop", "source_nominal", "source_error_bound", "drop_error_bound", "batch_id"})
        check(name + ":history_pattern", np.array_equal(history["P_total"], p.history_inputs(ep["history_mode"])))
        first_truth = None
        for inp in ep["inputs"]:
            check(name + ":observations_hash:" + inp["policy"], p.d.sha(folder / inp["observations"]) == inp["observations_hash"])
            check(name + ":truth_hash:" + inp["policy"], p.d.sha(folder / inp["truth"]) == inp["truth_hash"])
            with np.load(folder / inp["observations"]) as archive:
                obs = {k: archive[k].copy() for k in archive.files}
            truth = json.loads((folder / inp["truth"]).read_text(encoding="utf-8"))
            check(name + ":matched_history:" + inp["policy"], all(np.array_equal(history[k], obs[k][:12]) for k in history))
            if first_truth is not None:
                check(name + ":paired_source_noise:" + inp["policy"], truth["source_delta"] == first_truth["source_delta"] and truth["noise_draw"] == first_truth["noise_draw"])
            first_truth = truth
            exact_error = []
            theta = tuple(map(p.d.f, truth["theta"]))
            for t, pt in enumerate(obs["P_total"]):
                for i in range(3):
                    exact = sum((c*v for c, v in zip(p.d.prediction_coeff(pt, obs["source_nominal"][t], truth["topology"], i), theta)), F())
                    exact += theta[0] * p.d.f(truth["source_delta"][t])
                    exact_error.append(abs(p.d.f(obs["Y_drop"][t, i])-exact) <= p.d.f(obs["drop_error_bound"][t, i]))
            check(name + ":exact_truth_error_coverage:" + inp["policy"], all(exact_error))
            check(name + ":source_coverage:" + inp["policy"], all(abs(p.d.f(v)) <= p.d.f(b) for v, b in zip(truth["source_delta"], obs["source_error_bound"])))
            check(name + ":truth_parameter_coverage:" + inp["policy"], all(p.d.f(l) <= x <= p.d.f(u) for x, (l, u) in zip(theta, p.d.BOX)))
            check(name + ":device:" + inp["policy"], all(truth["checks"].values()))
            check(name + ":energy:" + inp["policy"], truth["energy_kwh"] == (0 if inp["policy"] == "none" else 9))
        screen = ep["decision"]["screening"]
        check(name + ":truth_retained", ep["topology"] in screen["retained"])
        lp_records = [r["result"] for r in screen["records"].values()] + [r["result"] for r in ep["evaluation"]]
        for k, r in enumerate(lp_records):
            lp_count += 1
            lo, hi = F(r["exact_dual_lower"]), F(r["exact_primal_upper"])
            check(name + f":lp_certificate:{k}", r["exact_primal_verified"] and r["exact_box_corrected_dual_verified"] and
                  F(r["reliable_lower_bound"]) <= lo <= hi <= F(r["verified_upper_bound"]) and hi < 1)
            max_upper = max(max_upper, r["verified_upper_bound"])
        for h, rec in screen["records"].items():
            if "feasibility_witness" in rec:
                witness = rec["feasibility_witness"]
                poly = p.d.build_history_polytope(history, h)
                check(name + ":exact_singleton_witness:" + h,
                      witness.get("exact_feasible_verified") and poly.exact_feasible(witness["x"]))
            if rec["classification"] == "excluded":
                check(name + ":history_exclusion_monotone:" + h,
                      all(r["result"]["reliable_lower_bound"] > 0 for r in ep["evaluation"] if r["candidate"] == h))
        design = ep["decision"].get("design", {})
        if design.get("state") == "probe_selected":
            retained = set(screen["retained"])
            check(name + ":design_only_retained", set(design["representatives"]) == retained and
                  all(r["reference"] in retained and r["candidate"] in retained for r in design["qp_records"]))
            for k, r in enumerate(design["qp_records"]):
                fit = r["result"]
                qp_count += 1
                check(name + f":qp_certificate:{k}", fit["exact_feasible_verified"] and fit["exact_tangent_dual_verified"] and
                      F(fit["lower"]) <= F(fit["exact_lower"]) <= F(fit["exact_upper"]) <= F(fit["upper"]))
        if ep["decision"]["state"] == "singleton_verified_skip_ev":
            check(name + ":zero_energy_on_resolved_history", ep["decision"]["selected_action"] is None and ep["summary"]["deployed_energy_kwh"] == 0 and len(screen["retained"]) == 1)
        check(name + ":future_truth_retained", not any(x["truth_excluded"] for x in ep["summary"]["policies"].values()))
    replays = []
    for mode in p.HISTORY_MODES:
        for profile in p.d.PROFILES:
            ep = next(e for e in report["episodes"] if e["history_mode"] == mode and e["profile"] == profile)
            inp = next(i for i in ep["inputs"] if i["policy"] == "none")
            with np.load(folder / inp["observations"]) as archive:
                obs = {k: archive[k].copy() for k in archive.files}
            for h in p.d.NAMES:
                old = next(r["result"] for r in ep["evaluation"] if r["candidate"] == h and r["policy"] == "none")
                fresh = p.v6.solve_lp(obs, h, parameter_box=p.d.BOX)
                check(ep["name"] + ":replay:" + h, p.classify_fit(old) == p.classify_fit(fresh) and
                      fresh["reliable_lower_bound"] <= old["verified_upper_bound"] + 1e-9 and
                      old["reliable_lower_bound"] <= fresh["verified_upper_bound"] + 1e-9)
                replays.append(dict(episode=ep["name"], candidate=h, result=fresh))
    result = dict(report_hash=p.d.sha(folder / "passive_results.json"), audit_source_hash=p.d.sha(__file__),
                  checks_count=len(checks), all_passed=all(x["passed"] for x in checks),
                  lp_count=lp_count, design_qp_count=qp_count, max_verified_lp_upper_bound=max_upper,
                  all_lp_caps_inactive=max_upper < 1, replay_lp_count=len(replays), replays=replays, checks=checks)
    p.d.dump(folder / "passive_artifact_audit.json", result)
    print({k: v for k, v in result.items() if k not in {"checks", "replays"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    audit(parser.parse_args().folder)
