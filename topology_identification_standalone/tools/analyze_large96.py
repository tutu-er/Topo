"""Independent post-seal audit of single-day 96-point benchmark panels.

NumPy and the standard library only. No algorithm, fitter or solver is imported.
The sole observed file is replayed as 64 training / 32 selection rows. Physical
truth is opened only after all planned terminal records and frozen models have
been checked. Resampling units are (case, replicate), stratified by fixed case.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations, permutations
import json
import math
from pathlib import Path
import traceback

import numpy as np


FACTORIAL = ("000", "100", "010", "001", "110", "101", "011", "111")
TERMINAL = {"fitted", "complete", "failed", "interrupted", "budget_exceeded", "common_failed"}
KAPPAS = (0., .003, .01, .03, .1)
BOOTSTRAP_SEED = 20261112


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, saved, message, *, atol=1e-12, rtol=1e-7):
    require(np.isfinite(actual) and np.isfinite(saved) and np.isclose(actual, saved, atol=atol, rtol=rtol), message)


def local(root, relative):
    root = Path(root).resolve(); path = (root/relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root), "Path escapes the declared folder")
    return path


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def public(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = [public(r) for r in rows]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict)) else v
                             for k, v in row.items()})


def load96(path):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"P", "Q", "V", "root", "labels"}, "The sole NPZ has unexpected fields")
        labels = tuple(map(int, data["labels"])); n = len(labels)
        require(n > 0 and len(set(labels)) == n, "Invalid terminal labels")
        values = [np.asarray(data[key], float) for key in ("P", "Q", "V")]
        require(all(a.shape in ((96, n), (1, 96, n)) for a in values), "Input is not exactly one 96-point day")
        require(len({a.shape for a in values}) == 1, "Observed shapes differ")
        p, q, voltage = [a.reshape(96, n) for a in values]
        root = np.asarray(data["root"], float).reshape(-1)
    require(root.shape == (96,), "Root sample count differs")
    y = root[:, None]**2-voltage**2
    require(all(np.isfinite(a).all() for a in (p, q, y)), "Nonfinite observations")
    return labels, (p[:64], q[:64], y[:64]), (p[64:], q[64:], y[64:])


def split_fingerprint(labels, rows):
    digest = hashlib.sha256(repr(tuple(labels)).encode())
    for value in rows:
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(repr(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def canonical(family, n, *, laminar=True):
    result = []
    for C in family:
        row = tuple(int(i) for i in C)
        require(row and len(set(row)) == len(row) and all(float(i) == j for i, j in zip(C, row)), "Invalid support entries")
        require(min(row) >= 0 and max(row) < n, "Support is outside terminal range")
        result.append(tuple(sorted(row)))
    require(len(set(result)) == len(result), "Repeated support")
    if laminar:
        for C, D in combinations(map(set, result), 2):
            require(not C & D or C <= D or D <= C, "Crossing supports in a fitted model")
    return tuple(sorted(result, key=lambda C: (len(C), C)))


def active(point, n, zero):
    return frozenset(tuple(sorted(C)) for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True)
                     if 1 < len(C) < n and max(r, x) > zero)


def internal_count(point, n):
    return sum(1 < len(C) < n for C in point["supports"])


def model_hash(point):
    return signature(sorted((tuple(C), float(r).hex(), float(x).hex())
                            for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True)))


def score_model(point, train, selection, reference, bound, cache):
    """Recompute feasible predictions and a convex fixed-family box certificate."""
    key = model_hash(point)
    if key in cache:
        return cache[key]
    n = train[0].shape[1]; canonical(point["supports"], n)
    r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
    require(r.shape == x.shape == (len(point["supports"]),), "Coefficient dimensions differ")
    require(np.isfinite(r).all() and np.isfinite(x).all(), "Nonfinite coefficients")
    violation = max(float(np.max(-r, initial=0.)), float(np.max(-x, initial=0.)),
                    float(np.max(r-bound, initial=0.)), float(np.max(x-bound, initial=0.)))
    require(violation <= 1e-7, "Response coefficient bound violation")
    R, X = np.zeros((n, n)), np.zeros((n, n))
    features = []
    for C, rr, xx in zip(point["supports"], r, x, strict=True):
        z = np.zeros(n); z[list(C)] = 1.; outer = np.outer(z, z)
        R += rr*outer; X += xx*outer
    errors = [y-p@R.T-q@X.T for p, q, y in (train, selection)]
    training_mse, selection_mse = [float(np.mean(e**2)) for e in errors]
    close(training_mse/reference, point["objective"], "Saved training objective does not replay", atol=1e-9)
    if "mse" in point:
        close(training_mse, point["mse"], "Saved physical training MSE does not replay", atol=1e-13)
    for power in train[:2]:
        for C in point["supports"]:
            z = np.zeros(n); z[list(C)] = 1.
            features.append(np.multiply.outer(power@z, z).ravel())
    F = np.column_stack(features)/math.sqrt(reference)
    residual = F.astype(np.longdouble)@np.r_[r, x].astype(np.longdouble)-train[2].ravel().astype(np.longdouble)/math.sqrt(reference)
    objective = float(np.mean(residual**2))
    gradient = np.asarray(2*F.astype(np.longdouble).T@residual/len(residual), float)
    theta = np.r_[r, x]
    lower = max(0., objective+float(np.minimum(-gradient*theta, gradient*(bound-theta)).sum()))
    certificate_gap = max(0., objective-lower)
    record = dict(model_sha256=key, train_mse=training_mse, selection_mse=selection_mse,
                  selection_scaled_mse=selection_mse/reference, internal=internal_count(point, n),
                  bound_violation=violation, qp_box_gap=certificate_gap,
                  qp_box_certificate_passed=certificate_gap <= 1e-6*(1+objective)+1e-10)
    cache[key] = record
    return record


def choose(candidates, tie):
    best = min(c["selection_scaled_mse"] for c in candidates)
    return min((c for c in candidates if c["selection_scaled_mse"] <= best+tie),
               key=lambda c: (c["internal"], c["selection_scaled_mse"], c["index"]))


def near_pair(primary, alternative, kappa, tie):
    if alternative is None:
        return False
    a, b = primary["selection_scaled_mse"], alternative["selection_scaled_mse"]
    return bool(b <= a+max(tie, kappa*b))


def compare_topology(predicted, real):
    tp, fp, fn = len(predicted & real), len(predicted-real), len(real-predicted)
    return dict(F1=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1., exact=predicted == real,
                TP=tp, FP=fp, FN=fn, predicted_clades=len(predicted), true_clades=len(real))


def audit_result(wrapper, common_wrapper, train, selection, labels, config, method, cache):
    result = wrapper["result"]; n = len(labels); zero, tie, alpha = [config[k] for k in ("zero_tol", "validation_tie", "alpha")]
    common = common_wrapper.get("common", common_wrapper)
    reference = float(common["reference_mse"])
    require(np.isfinite(reference) and reference > 0, "Invalid training reference")
    close(reference, result["reference_mse"], "Reference mismatch", atol=1e-16)
    require(result["method"] == method and result["selection_metric"] == "validation_mse", "Wrong method or selection metric")
    require(result.get("truth_used") is False and result.get("test_used") is False, "Unexpected information flags")
    require(result.get("complete_RNJ_output_fallback") is False and result.get("output_contraction") is None, "Unexpected final-tree fallback/contraction")
    effective = wrapper.get("effective_config", result["config"])
    require(result["config"] == effective, "Effective config differs from actual method config")
    require(all(effective[k] == v for k, v in config.items() if k != "total_seconds"), "A structural setting changed")
    require(0 < effective["total_seconds"] <= config["total_seconds"]+.05, "Invalid effective time allowance")
    if "configured_config" in wrapper:
        require(wrapper["configured_config"] == config, "Configured settings differ from protocol")
    close(wrapper["total_seconds"], wrapper["fresh_seconds"]+wrapper["common_seconds"], "Runtime accounting differs", atol=1e-7)
    if wrapper["status"] in {"fitted", "complete"}:
        require(not result.get("total_cap_exceeded") and wrapper["total_seconds"] <= config["total_seconds"]+.05, "Successful label exceeds budget")
    bank = []
    stages = ["baseline" if method in {"rnj", "nj"} else "initial"]+[e["stage"] for e in result["events"] if e.get("accepted")]
    for index, candidate in enumerate(result["frozen_candidates"]):
        require(candidate["index"] == index and tuple(candidate["labels"]) == labels, "Frozen candidate order or labels changed")
        scored = score_model(candidate["point"], train, selection, reference, config["response_bound"], cache)
        close(scored["selection_mse"], candidate["validation_mse"], "Frozen candidate selection MSE differs", atol=1e-13)
        close(scored["selection_scaled_mse"], candidate["validation_scaled_mse"], "Frozen candidate normalized MSE differs")
        bank.append(dict(index=index, **scored, _active=active(candidate["point"], n, zero),
                         source_stage=stages[index] if index < len(stages) else "singleton_protection"))
    require(bank, "Empty frozen model library")
    primary = choose(bank, tie)
    require(primary["index"] == result["selected"]["index"] == result["selected_mse_view"]["index"], "Actual selection does not follow MSE rule")
    require(model_hash(result["selected"]["point"]) == primary["model_sha256"], "Actual selected coefficients changed")
    alternatives = [b for b in bank if b["_active"] != primary["_active"]]
    alternate = choose(alternatives, tie) if alternatives else None
    saved = result["competition"]; ambiguous = near_pair(primary, alternate, config["ambiguity_kappa"], tie)
    indices = [primary["index"]]+([alternate["index"]] if ambiguous else [])
    require(saved["primary_index"] == primary["index"] and saved["secondary_index"] == (alternate["index"] if ambiguous else None), "Conditional output does not replay")
    require(saved["best_distinct_index"] == (alternate["index"] if alternate else None), "Best distinct candidate differs")
    require(saved["output_count"] == len(indices) and [m["index"] for m in saved["models"]] == indices, "Returned candidate set differs")
    require(saved["dr_candidate_indices"] == (indices if ambiguous else []), "DR pair differs from returned candidates")
    require(saved["selection_rule"] == "relative_selection_mse_ambiguity_v1", "Unknown conditional-output rule")
    close(saved["ambiguity_kappa"], config["ambiguity_kappa"], "Output kappa changed")
    path = result["path"]; require(path, "Empty accepted path")
    require(len(stages) == len(path), "Accepted-event/path lengths differ")
    path_values = [score_model(p, train, selection, reference, config["response_bound"], cache) for p in path]
    current = path[0]; cursor = 0; events = []
    guidance = common.get("guidance", {}).get("pool", []) if len(method) == 3 and method[0] == "1" else []
    materialized = {tuple(C) for C in guidance if 1 < len(C) < n}
    for candidate in result["frozen_candidates"]:
        materialized.update(tuple(sorted(C)) for C in candidate["point"]["supports"] if 1 < len(C) < n)
    for number, event in enumerate(result["events"]):
        stage = event["stage"]; require(stage in {"W", "T", "E"}, "Unexpected search stage")
        row = dict(event=number, stage=stage, accepted=bool(event.get("accepted")), skipped=event.get("skipped"),
                   zero_releases=len(event.get("zero_events", [])), _before=active(current, n, zero))
        for C in event.get("pool", []):
            if 1 < len(C) < n:
                materialized.add(tuple(sorted(C)))
        proposal = event.get("point")
        if proposal is not None:
            old = score_model(current, train, selection, reference, config["response_bound"], cache)
            new = score_model(proposal, train, selection, reference, config["response_bound"], cache)
            before = old["train_mse"]/reference+alpha*old["internal"]
            after = new["train_mse"]/reference+alpha*new["internal"]
            close(before, event["before_score"], "Event before-J mismatch", atol=1e-8)
            close(after, event["after_score"], "Event proposal-J mismatch", atol=1e-8)
            close(before-after, event["gain"], "Event J gain mismatch", atol=1e-8)
            old_weights = {tuple(C): max(r, x) for C, r, x in zip(current["supports"], current["r"], current["x"], strict=True)}
            new_family = set(map(tuple, proposal["supports"])); removed = set(old_weights)-new_family
            zero_only = bool(removed) and new_family <= set(old_weights) and all(old_weights[C] <= zero for C in removed)
            zero_only = zero_only and active(current, n, zero) == active(proposal, n, zero) and new["train_mse"]/reference <= old["train_mse"]/reference+config["gain_tol"]
            required = config["gain_tol"] if zero_only else max(config["gain_tol"], config["relative_gain"]*before)
            close(required, event["required_gain"], "Event acceptance threshold differs")
            expected = event["gain"] > required and not event.get("repeated_family", False)
            require(bool(event["accepted"]) == expected, "Event acceptance bypassed declared gate")
            row.update(before_J=before, proposal_J=after, J_gain=before-after, fit_gain_normalized=(old["train_mse"]-new["train_mse"])/reference,
                       count_gain=-alpha*(new["internal"]-old["internal"]), before_selection_mse=old["selection_mse"],
                       proposal_selection_mse=new["selection_mse"], _proposal=active(proposal, n, zero))
            if event["accepted"]:
                cursor += 1; require(cursor < len(path) and model_hash(proposal) == model_hash(path[cursor]), "Accepted proposal/path mismatch")
                current = path[cursor]
        row["_after"] = active(current, n, zero); events.append(row)
    require(cursor == len(path)-1, "Accepted path has unmatched models")
    require(all(b["train_mse"]/reference+alpha*b["internal"] <= a["train_mse"]/reference+alpha*a["internal"]+1e-8
                for a, b in zip(path_values, path_values[1:])), "Accepted J path increased")
    native = result.get("MIQP", [])
    for call in native:
        if "bound_consistent_with_fixed_family_qp" in call:
            lower, feasible = call.get("bound"), call.get("fixed_family_qp_objective")
            require(lower is not None and feasible is not None, "Bound consistency flag lacks its compared values")
            expected = lower <= feasible+call["bound_consistency_tolerance"]
            require(call["bound_consistent_with_fixed_family_qp"] == expected, "Native-bound conflict flag differs")
    return dict(primary=primary, alternate=alternate, bank=bank, events=events, materialized=materialized,
                reference=reference, decision=saved["decision"], native=native,
                QP=result.get("QP", []), stop_reason=result.get("stop_reason"),
                selected_origin=primary["source_stage"], cached_models=len(cache))


def grouped(rows, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in fields), []).append(row)
    output = []
    for key, group in groups.items():
        valid = [r for r in group if r["status"] == "verified"]
        known = [r for r in group if r.get("total_seconds") is not None]
        exact = sum(bool(r["exact"]) for r in valid)
        output.append(dict(zip(fields, key), planned=len(group), verified=len(valid),
            **{status: sum(r["status"] == status for r in group) for status in ("failed", "interrupted", "budget_exceeded", "invalid")},
            F1_verified_mean=float(np.mean([r["F1"] for r in valid])) if valid else None,
            F1_failure_zero_mean=sum(r["F1"] for r in valid)/len(group), exact=exact,
            exact_rate_planned=exact/len(group), exact_rate_verified=exact/len(valid) if valid else None,
            total_seconds_recorded=sum(r["total_seconds"] for r in known), timing_unknown=len(group)-len(known),
            native_timeouts=sum(r.get("native_timeouts", 0) for r in group),
            native_bound_conflicts=sum(r.get("native_bound_conflicts", 0) for r in group),
            single=sum(r.get("decision", "").startswith("single_") for r in valid),
            double=sum(r.get("decision") == "two_near_tied" for r in valid)))
    return output


def dr_summary(rows):
    output = []
    for method in dict.fromkeys(r["method"] for r in rows):
        planned = [r for r in rows if r["method"] == method]
        eligible = [r for r in planned if r["status"] == "verified"]
        for kappa in KAPPAS:
            decisions = []
            for row in eligible:
                trigger = near_pair(row["_primary"], row["_alternate"], kappa, row["_tie"])
                wrong = not row["exact"]
                covered = row["exact"] or (trigger and row["best_distinct_exact"])
                decisions.append((row, trigger, wrong, covered))
            triggers = sum(t for _, t, _, _ in decisions); wrong = sum(w for _, _, w, _ in decisions)
            caught = sum(t and w for _, t, w, _ in decisions)
            triggered_coverage = sum(t and covered for _, t, _, covered in decisions)
            wrong_pair_truth = sum(t and w and covered for _, t, w, covered in decisions)
            output.append(dict(method=method, kappa=kappa, planned=len(planned), eligible=len(eligible),
                triggers=triggers, trigger_rate_eligible=triggers/len(eligible) if eligible else None,
                wrong_primary=wrong, wrong_caught=caught, wrong_capture_rate=caught/wrong if wrong else None,
                returned_truth=sum(c for _, _, _, c in decisions),
                returned_truth_rate=sum(c for _, _, _, c in decisions)/len(eligible) if eligible else None,
                triggered_truth_covered=triggered_coverage, triggered_truth_coverage_rate=triggered_coverage/triggers if triggers else None,
                wrong_triggered_truth_available=wrong_pair_truth,
                wrong_truth_pair_rate=wrong_pair_truth/wrong if wrong else None,
                triggered_wrong_truth_pair_rate=wrong_pair_truth/caught if caught else None,
                triggered_neither_truth=triggers-triggered_coverage,
                correct_primary_triggers=sum(t and not w for _, t, w, _ in decisions),
                unnecessary_triggers=sum(t and not w for _, t, w, _ in decisions),
                wrong_single_separated=sum(w and not t and r["_alternate"] is not None for r, t, w, _ in decisions),
                wrong_single_no_alternative=sum(w and r["_alternate"] is None for r, _, w, _ in decisions)))
    return output


def factorial_statistics(rows, methods):
    available = set(methods); jobs = list(dict.fromkeys(r["job"] for r in rows)); bank = {(r["job"], r["method"]): r for r in rows}
    effects, interactions, orders, shapley = [], [], [], []
    for job in jobs:
        marginals = {}
        for bit, module in enumerate("RTE"):
            terms = []
            for off in FACTORIAL:
                on = off[:bit]+"1"+off[bit+1:]
                if off[bit] != "0" or off not in available or on not in available:
                    continue
                a, b = bank[job, off], bank[job, on]; valid = a["status"] == b["status"] == "verified"
                row = dict(job=job, case=a["case"], replicate=a["replicate"], module=module, off=off, on=on, valid=valid,
                           delta_F1=b["F1"]-a["F1"] if valid else None,
                           delta_exact=int(b["exact"])-int(a["exact"]) if valid else None,
                           delta_seconds=b["total_seconds"]-a["total_seconds"] if valid else None)
                effects.append(row); marginals[off, on] = row
                if valid:
                    terms.append((1/6 if off.count("1") == 1 else 1/3, row))
            shapley.append(dict(job=job, module=module, valid=len(terms) == 4,
                **{k: sum(w*r[k] for w, r in terms) if len(terms) == 4 else None for k in ("delta_F1", "delta_exact", "delta_seconds")}))
        if set(FACTORIAL) <= available:
            for order in permutations("RTE"):
                off = "000"
                for step, module in enumerate(order, 1):
                    bit = "RTE".index(module); on = off[:bit]+"1"+off[bit+1:]
                    orders.append(dict(order="".join(order), step=step, **marginals[off, on])); off = on
        for a, b in combinations(range(3), 2):
            condition = next(i for i in range(3) if i not in (a, b))
            for value in (0, 1):
                terms = []
                for aa, bb, sign in ((0, 0, 1), (1, 0, -1), (0, 1, -1), (1, 1, 1)):
                    bits = ["0"]*3; bits[a], bits[b], bits[condition] = str(aa), str(bb), str(value)
                    terms.append(("".join(bits), sign))
                if not all(m in available for m, _ in terms):
                    continue
                valid = all(bank[job, m]["status"] == "verified" for m, _ in terms)
                interactions.append(dict(job=job, interaction="RTE"[a]+"RTE"[b], condition_module="RTE"[condition], condition_value=value, valid=valid,
                    **{key: sum(sign*float(bank[job, m][source]) for m, sign in terms) if valid else None
                       for key, source in (("delta_F1", "F1"), ("delta_exact", "exact"), ("delta_seconds", "total_seconds"))}))
        if set(FACTORIAL) <= available:
            valid = all(bank[job, m]["status"] == "verified" for m in FACTORIAL)
            interactions.append(dict(job=job, interaction="RTE", condition_module=None, condition_value=None, valid=valid,
                **{key: sum((1 if m.count("1") % 2 else -1)*float(bank[job, m][source]) for m in FACTORIAL) if valid else None
                   for key, source in (("delta_F1", "F1"), ("delta_exact", "exact"), ("delta_seconds", "total_seconds"))}))
    return effects, interactions, orders, shapley


def delta_summary(rows, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in fields), []).append(row)
    result = []
    for key, group in groups.items():
        valid = [r for r in group if r["valid"]]
        result.append(dict(zip(fields, key), planned=len(group), paired=len(valid),
            **{metric+"_mean": float(np.mean([r[metric] for r in valid])) if valid else None
               for metric in ("delta_F1", "delta_exact", "delta_seconds")},
            F1_positive=sum(r["delta_F1"] > 1e-12 for r in valid), F1_negative=sum(r["delta_F1"] < -1e-12 for r in valid)))
    return result


def cluster_intervals(rows, methods, draws=2000):
    """Fixed-case stratified paired bootstrap, keeping a day's noise siblings together."""
    rng = np.random.default_rng(BOOTSTRAP_SEED); bank = {(r["job"], r["method"]): r for r in rows}
    jobs = list(dict.fromkeys(r["job"] for r in rows)); meta = {r["job"]: r for r in rows}
    groups = {}
    for job in jobs:
        row = meta[job]; groups.setdefault(row["case"], {}).setdefault(row["replicate"], []).append(job)
    samples = []
    for _ in range(draws):
        selected = []
        for case in sorted(groups):
            keys = sorted(groups[case]); picked = rng.integers(0, len(keys), len(keys))
            selected.extend(job for i in picked for job in groups[case][keys[i]])
        samples.append(selected)
    output = []
    for first, second in [(None, m) for m in methods]+list(combinations(methods, 2)):
        eligible = [j for j in jobs if bank[j, second]["status"] == "verified"
                    and (first is None or bank[j, first]["status"] == "verified")]
        for metric in ("F1", "exact", "total_seconds"):
            delta = {j: float(bank[j, second][metric])-(float(bank[j, first][metric]) if first else 0.) for j in eligible}
            values = [float(np.mean([delta[j] for j in sampled if j in delta])) for sampled in samples if any(j in delta for j in sampled)]
            low, high = np.percentile(values, (2.5, 97.5)) if values else (None, None)
            output.append(dict(old=first, new=second, metric=metric, planned=len(jobs), paired=len(eligible),
                independent_case_day_clusters=sum(len(v) for v in groups.values()), draws=draws,
                estimate=float(np.mean(list(delta.values()))) if delta else None,
                interval_low=float(low) if low is not None else None, interval_high=float(high) if high is not None else None,
                scope="fixed-case stratified case/day paired descriptive bootstrap; no topology probability"))
    return output


def analyze(output, *, bootstrap=2000, destination=None):
    out = Path(output).resolve(); runtime = out/"runtime"; folder = Path(destination).resolve() if destination else out/"analysis"
    protocol, seal, completion = [read(out/n) for n in ("protocol.json", "selection_seal.json", "completion.json")]
    require(seal.get("sealed") and seal.get("all_planned_terminal") and completion.get("all_planned_terminal"), "All planned tasks must be terminal and sealed")
    require(seal["protocol_sha256"] == sha(out/"protocol.json") == sha(runtime/"protocol.json"), "Protocol seal mismatch")
    if "selection_seal_sha256" in completion:
        require(completion["selection_seal_sha256"] == sha(out/"selection_seal.json"), "Completion seal mismatch")
    require(protocol["split"] == dict(total=96, train=[0, 64], selection=[64, 96], shuffle=False), "Unexpected data split")
    methods, inputs, config = protocol["methods"], protocol["inputs"], protocol["config"]
    expected = {f"runs/{j}/{m}/result.json" for j in inputs for m in methods}
    require(set(seal["result_sha256"]) == expected, "Sealed result denominator differs")
    consumed = {"protocol.json": sha(out/"protocol.json"), "selection_seal.json": sha(out/"selection_seal.json"), "completion.json": sha(out/"completion.json")}
    for name, checksum in protocol["source_sha256"].items():
        require(sha(local(runtime, name)) == checksum, "Frozen source changed: "+name)
        consumed["runtime/"+name] = checksum
    for info in inputs.values():
        if info.get("path"):
            require(sha(local(runtime, info["path"])) == info["sha256"], "A sole input file changed")
            consumed["runtime/"+info["path"]] = info["sha256"]
    runtime_expected = set(protocol["source_sha256"]) | {"protocol.json"}
    runtime_expected.update(info["path"] for info in inputs.values() if info.get("path"))
    runtime_actual = {p.relative_to(runtime).as_posix() for p in runtime.rglob("*") if p.is_file()}
    require(runtime_actual == runtime_expected, "Unexpected files in worker runtime")
    for name, checksum in seal["result_sha256"].items():
        require(sha(local(out, name)) == checksum, "A sealed result changed")
        require(read(local(out, name))["status"] in TERMINAL, "A planned result is not terminal")
        consumed[name] = checksum
    for name, checksum in seal.get("common_sha256", {}).items():
        require(sha(local(out, name)) == checksum, "A sealed common record changed")
        consumed[name] = checksum
    evaluation_map = read(out/"evaluation_map.json"); consumed["evaluation_map.json"] = sha(out/"evaluation_map.json")
    if protocol.get("evaluation_map_sha256"):
        require(consumed["evaluation_map.json"] == protocol["evaluation_map_sha256"], "Evaluation map changed")
    require(set(evaluation_map) == set(inputs), "Evaluation metadata denominator differs")
    rows, candidates, events, errors, solvers, box_certificates = [], [], [], [], [], []
    common_costs = {}; audits = {}
    for job, info in inputs.items():
        meta = evaluation_map[job]; cache = {}; data = None
        common_path = out/"common"/job/"common.json"
        common = read(common_path) if common_path.exists() else {}
        if common_path.exists():
            consumed[common_path.relative_to(out).as_posix()] = sha(common_path)
            common_costs[job] = float(common.get("total_seconds", common.get("seconds", 0.)))
        for method in methods:
            path = out/"runs"/job/method/"result.json"; wrapper = read(path)
            base = dict(job=job, method=method, **{k: meta.get(k) for k in ("case", "replicate", "voltage_sigma", "profile", "level", "cohort")},
                        status=wrapper["status"], raw_status=wrapper["status"], total_seconds=wrapper.get("total_seconds"),
                        fresh_seconds=wrapper.get("fresh_seconds"), common_seconds=wrapper.get("common_seconds"),
                        failure_stage=wrapper.get("failure_stage"), error=wrapper.get("error"), F1=None, exact=None)
            rows.append(base)
            if not wrapper.get("result"):
                continue
            try:
                require(not info.get("generation_failed"), "A failed-generation input has a model")
                if data is None:
                    file = local(runtime, info["path"]); require(sha(file) == info["sha256"], "Sole input changed")
                    consumed["runtime/"+info["path"]] = info["sha256"]
                    labels, train, selection = load96(file)
                    require(list(labels) == info["labels"], "Frozen labels differ")
                    require(split_fingerprint(labels, train) == info["training_sha256"] and split_fingerprint(labels, selection) == info["selection_sha256"], "Chronological split fingerprint differs")
                    data = labels, train, selection
                labels, train, selection = data
                for name in ("source_sha256_before", "source_sha256_after"):
                    if name in wrapper:
                        require(wrapper[name] == protocol["source_sha256"], "Method source provenance differs")
                if wrapper.get("common_sha256"):
                    require(wrapper["common_sha256"] == sha(common_path), "Method used changed common preprocessing")
                if "observed_npz_opened" in wrapper:
                    require(set(wrapper["observed_npz_opened"]) == {str(local(runtime, info["path"]))}, "Method read another observed file or lacks its sole input receipt")
                audit = audit_result(wrapper, common, train, selection, labels, config, method, cache)
                audits[job, method] = audit
                if wrapper["status"] in {"fitted", "complete"}:
                    base["status"] = "verified"
                base.update(model_audit_passed=True, selected_index=audit["primary"]["index"], selected_origin=audit["selected_origin"],
                    train_mse=audit["primary"]["train_mse"], selection_mse=audit["primary"]["selection_mse"],
                    decision=audit["decision"], stop_reason=audit["stop_reason"],
                    saved_candidates=len(audit["bank"]), native_calls=len(audit["native"]),
                    native_timeouts=sum(call.get("status") == 9 for call in audit["native"]),
                    native_bound_conflicts=sum(call.get("bound_consistent_with_fixed_family_qp") is False for call in audit["native"]),
                    native_bound_unchecked=sum("bound_consistent_with_fixed_family_qp" not in call for call in audit["native"]),
                    MIQP_seconds=sum(float(call.get("seconds", 0.) or 0.) for call in audit["native"]),
                    QP_seconds=sum(float(call.get("seconds", 0.) or 0.) for call in audit["QP"]),
                    zero_releases=sum(e["zero_releases"] for e in audit["events"]),
                    _primary=audit["primary"], _alternate=audit["alternate"], _tie=config["validation_tie"])
                for candidate in audit["bank"]:
                    candidates.append(dict(job=job, method=method, **candidate))
                for event in audit["events"]:
                    events.append(dict(job=job, method=method, **event))
                for number, call in enumerate(audit["native"]):
                    solvers.append(dict(job=job, method=method, index=number, **call))
            except Exception:
                base.update(status="invalid" if wrapper["status"] in {"fitted", "complete"} else wrapper["status"],
                            model_audit_passed=False, audit_error=traceback.format_exc())
        for certificate in cache.values():
            box_certificates.append(dict(job=job, **certificate))
    # This receipt precedes the first opening of an evaluation truth file.
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/"pre_truth_audit.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        all_planned_terminal=True, planned=len(rows), model_audits=sum(r.get("model_audit_passed", False) for r in rows),
        invalid=sum(r["status"] == "invalid" for r in rows), observations_per_input=96, train_points=64, selection_points=32,
        observed_file_sha256={k: v for k, v in consumed.items() if k.endswith(".npz")}, truth_opened=False))
    truth_cache = {}
    for row in rows:
        key = row["job"], row["method"]
        if key not in audits:
            continue
        audit = audits[key]; job = row["job"]
        if job not in truth_cache:
            truth_path = out/"evaluation"/(job+".json"); truth = read(truth_path)
            consumed[truth_path.relative_to(out).as_posix()] = sha(truth_path)
            require(truth["labels"] == inputs[job]["labels"], "Truth terminal labels differ")
            n = len(truth["labels"]); real = frozenset(canonical(truth["proper_clades"], n))
            require(all(1 < len(C) < n for C in real), "Truth contains a nonproper clade")
            truth_cache[job] = real, n
        real, n = truth_cache[job]; first, second = audit["primary"], audit["alternate"]
        row.update(compare_topology(first["_active"], real),
            best_distinct_exact=second["_active"] == real if second else False,
            library_truth_present=any(c["_active"] == real for c in audit["bank"]),
            support_union_covers_truth=real <= audit["materialized"],
            materialized_true_clades=len(real & audit["materialized"]), true_clades=len(real),
            best_distinct_index=second["index"] if second else None,
            relative_mse_gap=max(0., (second["selection_mse"]-first["selection_mse"])/second["selection_mse"]) if second and second["selection_mse"] else 0. if second else None)
        triggered = near_pair(first, second, config["ambiguity_kappa"], config["validation_tie"])
        row["returned_truth"] = row["exact"] or (triggered and row["best_distinct_exact"])
        for kind, clades in (("FP", first["_active"]-real), ("FN", real-first["_active"])):
            for C in sorted(clades):
                errors.append(dict(job=job, method=row["method"], case=row["case"], kind=kind, size=len(C), size_fraction=len(C)/n,
                    size_band="small_2_3" if len(C) <= 3 else "large_over_half" if len(C) > n/2 else "medium_to_half", support=list(C)))
    for candidate in candidates:
        real, _ = truth_cache[candidate["job"]]
        candidate.update(compare_topology(candidate["_active"], real), active_clades=[list(C) for C in sorted(candidate["_active"])])
    for event in events:
        real, _ = truth_cache[event["job"]]
        for prefix, field in (("before", "_before"), ("proposal", "_proposal"), ("accepted", "_after")):
            if field in event:
                event.update({prefix+"_"+k: v for k, v in compare_topology(event[field], real).items() if k in {"F1", "exact", "FP", "FN"}})
    for name, checksum in consumed.items():
        require(sha(local(out, name)) == checksum, "An input/source/result changed during analysis")
    summaries = grouped(rows, ["method"]); effects, interactions, orders, shapley = factorial_statistics(rows, methods)
    physical = sum(float(r.get("fresh_seconds") or 0.) for r in rows)+sum(common_costs.values())
    timing = dict(logical_method_seconds=sum(float(r.get("total_seconds") or 0.) for r in rows),
        physical_common_once_plus_fresh_seconds=physical, generation_seconds=protocol.get("generation_seconds"),
        manager_elapsed_seconds_all_invocations=completion.get("manager_elapsed_seconds_all_invocations", completion.get("calendar_seconds")),
        unknown_method_timings=sum(r.get("total_seconds") is None for r in rows),
        boundary="logical costs charge common once to every method; physical counts each common once; generation/manager overhead separate")
    verification = dict(verified=True, created_utc=datetime.now(timezone.utc).isoformat(), planned=len(rows),
        verified_methods=sum(r["status"] == "verified" for r in rows), invalid_methods=sum(r["status"] == "invalid" for r in rows),
        all_fitted_records_valid=not any(r["status"] == "invalid" for r in rows), candidates=len(candidates), events=len(events),
        source_and_input_sha256=consumed, qp_box_certificates=len(box_certificates),
        qp_box_certificate_failures=sum(not c["qp_box_certificate_passed"] for c in box_certificates),
        analysis_source_sha256=sha(Path(__file__)), truth_opened_only_after_pre_truth_audit=True,
        information_scope="sole96 per input; train64 coefficients; selection32 frozen ranking; no extra day/test observations",
        support_union_scope="materialized guidance/pools/path only; unknown-support W domain can contain other clades")
    summary = dict(methods=summaries, timing=timing, planned_inputs=len(inputs), methods_per_input=len(methods),
        bound_checks=dict(native_calls=len(solvers), contradicted=sum(s.get("bound_consistent_with_fixed_family_qp") is False for s in solvers),
                          unchecked=sum("bound_consistent_with_fixed_family_qp" not in s for s in solvers),
                          reported_time_limits=sum(s.get("status") == 9 for s in solvers)),
        scope="fresh simulated single days; no DR executed; conditional pair is finite-library diagnostic, not calibrated confidence")
    artifacts = {"results.csv": rows, "candidates.csv": candidates, "events.csv": events, "error_clades.csv": errors,
        "native_calls.csv": solvers, "qp_box_certificates.csv": box_certificates,
        "by_case_noise.csv": grouped(rows, ["case", "voltage_sigma", "method"]),
        "by_profile.csv": grouped(rows, ["case", "profile", "level", "method"]),
        "dr_curve.csv": dr_summary(rows), "ablation_case_effects.csv": effects,
        "ablation_effects.csv": delta_summary(effects, ["module", "off", "on"]),
        "interactions.csv": delta_summary(interactions, ["interaction", "condition_module", "condition_value"]),
        "module_addition_orders.csv": orders, "shapley.csv": shapley,
        "cluster_intervals.csv": cluster_intervals(rows, methods, bootstrap)}
    for name, content in artifacts.items():
        write_csv(folder/name, content)
    write_json(folder/"summary.json", summary); write_json(folder/"verification.json", verification)
    digest = sha(Path(__file__)); archive = folder/"source"/digest/Path(__file__).name
    archive.parent.mkdir(parents=True, exist_ok=True); archive.write_bytes(Path(__file__).read_bytes())
    write_json(folder/"artifact_manifest.json", dict(source_sha256=digest,
        files={name: sha(folder/name) for name in [*artifacts, "summary.json", "verification.json", "pre_truth_audit.json"]}))
    print(json.dumps(dict(planned=len(rows), methods=summaries, timing=timing, invalid=verification["invalid_methods"]), indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    analyze(args.output, bootstrap=args.bootstrap, destination=args.destination)
