"""Independent artifact consistency audit for the small nonlinear-AC study.

This checks recorded exact bound intervals and physical observations. It does
not rederive LP dual certificates because those multipliers/matrices are not
stored by the runner. Independent tests exercise their implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ac_shared_v8 as ac


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(folder, *, require_current_source=True):
    folder = Path(folder)
    protocol = json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    data = json.loads((folder/"results.json").read_text(encoding="utf-8"))
    issues, hashes, counts, summaries = [], {}, Counter(), []
    max_gap = 0.
    observation_fields = {"P", "Q", "Y_drop", "drop_error_bound", "source_nominal_eff",
                          "source_error_eff", "source_p_abs_cap", "source_q_abs_cap", "batch_id"}
    default_actions = ["site1_none"]+[f"site{s}_{p}" for s in (1, 2, 3) for p in ("constant", "endpoint")]
    candidates = tuple(ac.v6.CANDIDATES)
    box = np.asarray(protocol["box"])

    def check(condition, message):
        if not condition:
            issues.append(message)

    def remember(path):
        hashes[str(path.relative_to(folder))] = sha(path)

    def certificate(result, name):
        nonlocal max_gap
        counts["LP_records"] += 1
        check(result.get("lower") is not None, name+": no lower bound")
        if result.get("lower") is None:
            return
        lo, hi = F(result["exact_lower"]), F(result["exact_upper"])
        check(0 <= lo <= hi, name+": invalid exact interval")
        check(F(result["lower"]) <= lo and F(result["upper"]) >= hi,
              name+": non-outward float interval")
        check(F(result["gap"]) >= hi-lo, name+": non-outward gap")
        check(result["status"] == ("excluded" if lo > 0 else "unresolved"), name+": status mismatch")
        check(result.get("exact_primal_verified") is True and result.get("exact_box_dual_verified") is True,
              name+": verification flags missing")
        check(result.get("solver_status") == 0, name+": unexpected solver status")
        max_gap = max(max_gap, result["gap"])

    check(data["protocol"] == protocol, "aggregate protocol differs from frozen protocol")
    if require_current_source:
        check(protocol["source_sha256"] == sha(ac.__file__), "core source changed after protocol freeze")
        check(data.get("source_hash_unchanged") is True, "runner did not confirm unchanged source")
    for name, digest in data.get("input_sha256", {}).items():
        check(sha(folder/name) == digest, "input hash mismatch: "+name)
    dependencies = protocol.get("dependency_sha256", {})
    for name, digest in dependencies.items():
        path = Path(__file__).parent / name
        check(path.exists() and sha(path) == digest, "dependency mismatch: "+name)
    if protocol["mode"] == "formal":
        check(set(dependencies) >= {"design_v7.py", "local_shared_source_v6.py"},
              "formal protocol lacks dependency hashes")
        expected_cases = {(h, kind, profile, seed) for h in candidates
                          for kind, profile in [("flat", "symmetric"), ("rich", "symmetric"), ("flat", "weak")]
                          for seed in (8201, 8202)}
        check(set(map(tuple, protocol["cases"])) == expected_cases, "formal planned case set changed")
    case_keys = [(e["truth"], e["history"], e["profile"], e["seed"]) for e in data["episodes"]]
    check(set(case_keys) == set(map(tuple, protocol["cases"])) and len(set(case_keys)) == len(case_keys),
          "planned case coverage or uniqueness failed")

    for episode in data["episodes"]:
        h, kind, profile, seed = episode["truth"], episode["history"], episode["profile"], episode["seed"]
        prefix = f"{h}_{kind}_{profile}_{seed}"
        actions = list(dict.fromkeys(r["action"] for r in episode["records"]))
        planned_actions = protocol.get("actions", default_actions)
        check(set(actions) == set(planned_actions), prefix+": action library differs from protocol")
        actions = ["site1_none"]+[action for action in actions if action != "site1_none"]
        epfile = folder/(prefix+"_results.json")
        check(json.loads(epfile.read_text(encoding="utf-8")) == episode, prefix+": aggregate/episode mismatch")
        remember(epfile)
        baseline, base_truth = None, None
        historical = None
        for action in actions:
            obsfile, truthfile = folder/(prefix+"_"+action+".npz"), folder/(prefix+"_"+action+"_truth.json")
            remember(obsfile); remember(truthfile)
            with np.load(obsfile) as handle:
                obs = {k: handle[k] for k in handle.files}
            truth = json.loads(truthfile.read_text(encoding="utf-8"))
            counts["observation_batches"] += 1
            check(set(obs) == observation_fields, prefix+action+": estimator file contains unexpected fields")
            check(truth["topology"] == h and truth["history"] == kind and truth["seed"] == seed,
                  prefix+action+": truth identity mismatch")
            theta, source = np.asarray(truth["theta"]), np.asarray(truth["source_p"])
            check(np.all(theta >= box[:, 0]) and np.all(theta <= box[:, 1]), prefix+action+": theta out of domain")
            check(all(truth["checks"].values()), prefix+action+": recorded physical domain failure")
            check(np.all(abs(source-.12) <= .1+1e-15), prefix+action+": source budget violated")
            check(np.all(abs(source) <= obs["source_p_abs_cap"]+1e-15) and
                  np.all(abs(.3*source) <= obs["source_q_abs_cap"]+1e-15), prefix+action+": source current cap violated")
            prediction, voltage, residual = ac.ac_batch(h, theta, obs["P"], obs["Q"], source)
            check(residual < 1e-11 and truth["max_distflow_error"] < 1e-11, prefix+action+": AC residual failure")
            check(np.max(abs(prediction-np.asarray(truth["Y_true"]))) < 1e-11,
                  prefix+action+": stored true response differs from recomputed AC")
            check(abs(voltage).min() >= .9 and abs(voltage).max() <= 1.1, prefix+action+": AC voltage out of domain")
            check(abs(float(abs(voltage).min())-truth["min_voltage"]) < 1e-11 and
                  abs(float(abs(voltage).max())-truth["max_voltage"]) < 1e-11,
                  prefix+action+": stored voltage extrema mismatch")
            noise = np.asarray(truth["actual_noise"])
            check(np.max(abs(obs["Y_drop"]-np.asarray(truth["Y_true"])-noise)) < 3e-16,
                  prefix+action+": observed response mismatch")
            check(np.all(abs(noise) <= obs["drop_error_bound"]), prefix+action+": measurement box violated")
            ev = np.asarray(truth["EV_P_kw"])
            if "EV_bus_index_by_slot" in truth:
                sites = np.asarray(truth["EV_bus_index_by_slot"], int)
            else:
                sites = np.full(24, int(action[4])-1)
            check(sites.shape == (24,) and np.all((sites >= -1) & (sites <= 2)), prefix+action+": EV route invalid")
            check(np.all((sites >= 0) | (ev == 0)), prefix+action+": disconnected EV has injection")
            physical_background = obs["P"].copy()
            for slot, site in enumerate(sites):
                if site >= 0:
                    physical_background[12+slot, site] -= ev[slot]/20
            check(np.max(abs(obs["Q"]-.3*physical_background)) < 1e-15,
                  prefix+action+": Q includes unintended EV command")
            if action.endswith("none"):
                baseline, base_truth, historical = obs, truth, ac.sliced(obs, slice(0, 12))
                check(np.all(ev == 0), prefix+action+": passive baseline has EV")
            else:
                check(np.all((ev == 0) | ((ev >= 4.2-1e-12) & (ev <= 6+1e-12))), prefix+action+": EV power domain")
                check(abs(.25*ev.sum()-27) < 1e-12 and abs(truth["energy_kwh"]-27) < 1e-12,
                      prefix+action+": equal charging energy failed")
                travel = float(truth.get("travel_kwh", 0.))
                travel_slot_key = next((k for k in ("travel_by_slot_kwh", "travel_kwh_by_slot") if k in truth), None)
                if travel_slot_key:
                    travel_slots = np.asarray(truth[travel_slot_key])
                    check(travel_slots.shape == (24,) and abs(travel_slots.sum()-travel) < 1e-12,
                          prefix+action+": travel-energy inconsistency")
                else:
                    travel_slots = np.zeros(24); travel_slots[0] = travel
                expected_soc = .4 + (.95*.25*np.cumsum(ev)-np.cumsum(travel_slots))/60
                check(abs(expected_soc[-1]-(.8275-travel/60)) < 1e-12 and expected_soc.min() >= .2 and expected_soc.max() <= .9,
                      prefix+action+": SOC failed")
                check(abs(truth["soc_max"]-max(.4, expected_soc.max())) < 1e-12 and
                      abs(truth["soc_min"]-min(.4, expected_soc.min())) < 1e-12,
                      prefix+action+": recorded SOC extrema mismatch")
                if "mobile" in action:
                    check(travel == 2. and np.count_nonzero(sites == -1) == 2 and len(set(sites[sites >= 0])) == 2,
                          prefix+action+": mobile route/travel protocol mismatch")
                check(np.max(abs(physical_background-baseline["P"])) < 1e-15,
                      prefix+action+": unmatched background")
            if baseline is not None:
                check(all(np.array_equal(obs[k][:12], baseline[k][:12]) for k in observation_fields),
                      prefix+action+": history differs across actions")
                check(truth["source_p"] == base_truth["source_p"] and truth["actual_noise"] == base_truth["actual_noise"],
                      prefix+action+": unmatched source/noise across actions")

        retained = [j for j, result in episode["history_results"].items() if result["status"] != "excluded"]
        check(set(episode["history_results"]) == set(candidates), prefix+": history candidate coverage failed")
        check(retained == episode["retained"], prefix+": history retained list inconsistent")
        check(h in retained, prefix+": history falsely excludes truth")
        for j, result in episode["history_results"].items():
            certificate(result, prefix+":history:"+j)
        design = episode["design"]
        refs = design.get("references", {})
        if design.get("witness"):
            refs = {**refs, design["witness"]["candidate"]: design["witness"]}
        for j, ref in refs.items():
            check(j in retained, prefix+": reference not retained")
            ref_theta = np.asarray(ref["theta"])
            check(np.all(ref_theta >= box[:, 0]) and np.all(ref_theta <= box[:, 1]),
                  prefix+": reference parameters outside fitting domain")
            prediction, voltage, residual = ac.ac_batch(j, ref["theta"], historical["P"], historical["Q"], ref["source_p"])
            witness = bool(np.all(abs(prediction-historical["Y_drop"])+1e-11 <= historical["drop_error_bound"])
                           and abs(voltage).min() >= .9 and abs(voltage).max() <= 1.1)
            check(witness == ref["historical_numeric_witness"], prefix+": numeric witness flag inconsistent")
            check(np.all(np.asarray(ref["source_p"]) >= .02) and np.all(np.asarray(ref["source_p"]) <= .22),
                  prefix+": reference source outside fitting domain")
        if len(retained) == 1 and "singleton" in design["status"] and "skip" in design["status"]:
            check(retained[0] in refs and refs[retained[0]]["historical_numeric_witness"],
                  prefix+": singleton reported without numerical AC witness")
        for j in design.get("used_references", []):
            check(j in refs and refs[j]["historical_numeric_witness"], prefix+": invalid design reference")
        for rec in design.get("records", []):
            certificate(rec["result"], prefix+":design:"+rec["action"]+":"+rec["candidate"])
        indexed = {(r["action"], r["candidate"], r["sharing"]): r["result"] for r in episode["records"]}
        check(len(indexed) == len(actions)*9 and len(episode["records"]) == len(actions)*9,
              prefix+": action/candidate ablation coverage failed")
        stats = {}
        for action in actions:
            stats[action] = {}
            for sharing in ("path", "output", "invalid_linear"):
                wrong = 0
                for j in candidates:
                    result = indexed[(action, j, sharing)]
                    certificate(result, prefix+":"+action+":"+j+":"+sharing)
                    if j == h and result["status"] == "excluded":
                        counts[sharing+"_false_exclusions"] += 1
                        if sharing != "invalid_linear":
                            issues.append(prefix+":"+action+": true AC candidate excluded")
                    wrong += int(j != h and result["status"] == "excluded")
                stats[action][sharing+"_wrong_excluded"] = wrong
            for j in candidates:
                path, output = indexed[(action, j, "path")], indexed[(action, j, "output")]
                if path.get("upper") is not None and output.get("lower") is not None:
                    check(path["upper"]+1e-12 >= output["lower"], prefix+action+j+": relaxation order contradiction")
                    counts["strict_path_strengthening"] += int(path["lower"] > output["upper"]+1e-10)
        selected = design.get("selection", {}).get("selected") if design.get("selection") else None
        summary = dict(case=prefix, truth=h, history=kind, profile=profile, seed=seed,
                       history_wrong_excluded=3-len(retained), design_status=design["status"],
                       selected=selected, actionstats=stats)
        if selected:
            summary["selected_path_wrong_excluded"] = stats[selected]["path_wrong_excluded"]
            constant = selected.replace("endpoint", "constant")
            if constant in stats:
                summary["same_site_constant_path_wrong_excluded"] = stats[constant]["path_wrong_excluded"]
        if design.get("mobile_selection"):
            mobile = design["mobile_selection"]["selected"]
            summary["selected_mobile"] = mobile
            summary["selected_mobile_path_wrong_excluded"] = stats[mobile]["path_wrong_excluded"]
        summaries.append(summary)
    for file in (folder/"protocol.json", folder/"results.json"):
        remember(file)
    replay_log = []
    if protocol["mode"] == "formal":
        # Rebuild nine predetermined LPs from raw observations; this is a
        # replay through the inspected implementation, not a second solver.
        for h in candidates:
            episode = next(e for e in data["episodes"] if (e["truth"], e["history"], e["profile"], e["seed"])
                           == (h, "flat", "symmetric", 8201))
            filename = f"{h}_flat_symmetric_8201_mobile12.npz"
            with np.load(folder/filename) as handle:
                obs = {k:handle[k] for k in handle.files}
            for candidate in candidates:
                recorded = next(r["result"] for r in episode["records"]
                                if (r["action"], r["candidate"], r["sharing"]) == ("mobile12", candidate, "path"))
                began = perf_counter()
                rebuilt = ac.solve_outer(obs, candidate)
                complete = rebuilt.get("lower") is not None and recorded.get("lower") is not None
                overlap = bool(complete and
                    max(F(rebuilt["exact_lower"]), F(recorded["exact_lower"])) <=
                    min(F(rebuilt["exact_upper"]), F(recorded["exact_upper"])))
                classification = rebuilt["status"] == recorded["status"]
                truth_retained = candidate != h or rebuilt["status"] != "excluded"
                check(overlap and classification and truth_retained, filename+candidate+": LP replay mismatch")
                replay_log.append(dict(observations_file=filename, input_sha256=sha(folder/filename),
                                       truth=h, candidate=candidate, exact_intervals_overlap=overlap,
                                       same_classification=classification, true_candidate_retained=truth_retained,
                                       replayed_result=rebuilt, elapsed_seconds=perf_counter()-began))
        counts["replayed_LPs"] = len(replay_log)
    return dict(passed=not issues, issues=issues, source_sha256=sha(ac.__file__),
                audit_source_sha256=sha(__file__), hashes=hashes, counts=dict(counts),
                max_recorded_certificate_gap=max_gap, summaries=summaries,
                replayed_LPs=replay_log,
                scope="stored exact-interval/physical-input audit; does not independently rederive saved dual multipliers",
                replay_scope="nine predetermined raw-data model rebuilds with the inspected solver; not an independent optimization implementation",
                historical_AC_witness_semantics="numerical residual check, not interval existence proof")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--allow-source-change", action="store_true")
    args = parser.parse_args()
    result = audit(args.folder, require_current_source=not args.allow_source_change)
    target = args.folder/"artifact_audit.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "issues", "counts", "max_recorded_certificate_gap")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
