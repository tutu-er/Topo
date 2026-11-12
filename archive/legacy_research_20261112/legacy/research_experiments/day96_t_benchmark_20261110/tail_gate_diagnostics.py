"""Replay the saved gates in the three L1 T-vs-noT F1 regressions.

This is a post-hoc explanation of saved training decisions. It performs no
optimization, fitting, parameter tuning, candidate generation or reselection.
E raw proposals, retained E endpoints and final validated outputs are reported
separately. Stage locations use exact control-derived path indices.
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
from . import stage_effects
from research_experiments.rnj import audit_source_ac96_benchmark as truth_contract


ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8", newline="\n")


def family(point):
    return frozenset(tuple(sorted(map(int, C))) for C in point["supports"])


def positive(point, n):
    return frozenset(tuple(sorted(C)) for C, r, x in zip(point["supports"], point["r"], point["x"])
                     if 1 < len(C) < n and max(r, x) > 1e-9)


def proper_count(point, n):
    return sum(1 < len(C) < n for C in point["supports"])


def f1(clades, truth):
    return 2*len(clades & truth)/(len(clades)+len(truth)) if clades or truth else 1.


def matrices(point, n):
    return contract.response_matrices(contract._point(point, n), n)


def physical_loss(point, arrays, n):
    p, q, y = arrays; R, X = matrices(point, n)
    return float(np.mean(np.abs(y-p@R.T-q@X.T)))


def endpoint(point, *, arrays, truth, n, penalty):
    L = physical_loss(point, arrays, n)
    if abs(L-point["objective"]) > 1e-10:
        raise ValueError("A saved endpoint's training MAE does not replay.")
    clades = positive(point, n); count = proper_count(point, n)
    return dict(L=L, retained_proper_count=count, J=L+penalty*count,
                positive_internal_count=len(clades), F1=f1(clades, truth), exact=clades == truth,
                false_positive=len(clades-truth), false_negative=len(truth-clades),
                positive_topology=json.dumps(sorted(map(list, clades))),
                parameter_sha256=contract.digest(contract._point(point, n)))


def feasible_in_exit(point, event, n):
    """Constructive family feasibility for the saved fixed-column E domain.

    Put a=1 on its supplied proper supports and zero on all other columns;
    singleton/full coefficients can be zero-padded and residual e=|Y-Yhat|.
    Laminarity and the subset-cardinality cuts are checked directly. No fit is
    needed to establish feasibility of the already supplied coefficients.
    """
    contract._point(point, n)  # validates laminarity and the [0,2] bounds
    supplied, pool = family(point), {tuple(C) for C in event["pool"]}
    basis = {(i,) for i in range(n)} | {tuple(range(n))}
    hard, forbidden = {tuple(C) for C in event.get("hard", [])}, {tuple(C) for C in event.get("forbidden", [])}
    supported = supplied <= pool | basis
    hard_ok = hard <= supplied
    forbidden_ok = not (supplied & forbidden)
    selected = {C for C in supplied if 1 < len(C) < n}
    subset_cuts_ok = all(sum(set(C) < set(W) for C in selected) <= len(W)-2
                         for W in pool | {tuple(range(n))} if len(W) >= 2)
    return dict(feasible=bool(supported and hard_ok and forbidden_ok and subset_cuts_ok),
                supports_in_pool_or_basis=supported, hard_present=hard_ok,
                forbidden_absent=forbidden_ok, laminar_and_bounds=True, subset_cuts=subset_cuts_ok)


def diagnostics(output):
    started = perf_counter(); output = Path(output).resolve()
    protocol_path, catalog_path = output/"protocol.json", output/"catalog.json"
    native_path, table_path = output/"analysis/verification.json", output/"analysis/per_model.csv"
    stage_path, locations_path = output/"stage_effects/verification.json", output/"stage_effects/locations.json"
    protocol, catalog, native, stage = map(read, (protocol_path, catalog_path, native_path, stage_path))
    locations, frame = read(locations_path), pd.read_csv(table_path)
    protocol_hash, native_hash = sha(protocol_path), sha(native_path)
    if not (protocol["planned"] == 144 and native.get("passed") is True and native.get("complete") is True
            and native.get("verified") == 144 and native.get("all_MIP_gaps_met") is True):
        raise RuntimeError("The complete independently verified L1 benchmark is required.")
    if native["protocol_sha256"] != protocol_hash or protocol["catalog_sha256"] != sha(catalog_path):
        raise ValueError("Native verification or catalog refers to a different protocol.")
    if native["analyzer_sha256"] != sha(Path(__file__).with_name("analyze.py")):
        raise ValueError("Current native analyzer differs from the verified version.")
    if not (stage.get("passed") is True and stage.get("native_results_complete") is True
            and stage.get("planned_native_results") == 144 and stage["native_verification_sha256"] == native_hash
            and stage["native_protocol_sha256"] == protocol_hash
            and stage["stage_analysis_sha256"] == sha(Path(stage_effects.__file__))):
        raise ValueError("Stage locations are not from the verified native results/source.")
    for name, expected in protocol["source_hashes"].items():
        if sha(output/"source"/name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    methods = [m["id"] for m in protocol["methods"]]
    if methods != ["noT", "T003", "T010"] or len(frame) != 144:
        raise ValueError("Expected the fixed noT/T003/T010 panel.")
    audited, result_hashes = {}, {}
    for job in protocol["jobs"]:
        audit_file = output/"analysis/jobs"/job/"verified.json"
        audit = read(audit_file); audited[job] = audit
        if audit["errors"] or audit["protocol_sha256"] != protocol_hash:
            raise ValueError("A per-input native audit is invalid.")
        if stage["native_per_input_audit_sha256"][job] != sha(audit_file):
            raise ValueError("Per-input numerical audit changed after stage analysis.")
        for method in methods:
            result_file = output/"jobs"/job/method/"result.json"
            measured = sha(result_file); key = job+"/"+method
            if audit["raw_hashes"][method] != measured or stage["result_hashes"][key] != measured:
                raise ValueError("A native result changed after numerical/stage verification.")
            result_hashes[key] = measured
            matching = frame[(frame.job == job) & (frame.method == method)]
            row = next(r for r in audit["rows"] if r["method"] == method)
            if len(matching) != 1 or abs(float(matching.iloc[0].clade_f1)-row["clade_f1"]) > TOL:
                raise ValueError("The per-model table does not reproduce the verified audit.")
    control = frame[frame.method == "noT"].set_index("job")
    harmful = []
    for row in frame[frame.method != "noT"].itertuples(index=False):
        if row.clade_f1 < float(control.loc[row.job, "clade_f1"])-TOL:
            harmful.append(dict(job=row.job, method=row.method, delta_output_F1=float(row.clade_f1-control.loc[row.job, "clade_f1"])))
    if len(harmful) != 3 or len({r["job"] for r in harmful}) != 2:
        raise ValueError("The observed three regressions on two inputs changed; review diagnostic scope.")
    jobs = [j for j in protocol["jobs"] if j in {r["job"] for r in harmful}]
    models, E_rows, T_rows, receipts, truth_hashes = {}, [], [], {}, {}
    for job in jobs:
        meta = catalog[job]; n = len(meta["labels"])
        if sha(meta["train"]) != protocol["input_hashes"][job]["train"]:
            raise ValueError("Frozen training input changed.")
        labels, arrays = stage_effects.arrays(meta["train"])
        if labels != meta["labels"]:
            raise ValueError("Training terminal order changed.")
        network = Path(meta["truth_network"])
        hashes = {name: sha(network/name) for name in ("metadata.json", "buses.csv", "branches.csv")}
        if hashes != audited[job]["truth_source_hashes"]:
            raise ValueError("Saved physical truth source changed.")
        truth_hashes[job] = hashes
        physical = truth_contract.physical_truth(network, labels); index = {label: i for i, label in enumerate(labels)}
        truth = frozenset(tuple(sorted(index[i] for i in C)) for C in physical["clades"])
        for method in methods:
            key = job+"/"+method; result = read(output/"jobs"/job/method/"result.json")
            located = stage_effects.locate_stages(result)
            if located != locations[key] or located["E"]["status"] != "resolved":
                raise ValueError("Stored control-derived stage locations do not replay exactly.")
            path, er, ee = result["trajectory"]["path"], result["exit_record"], result["exit_event"]
            before = path[located["prefix_end_index"]]; raw = er["raw_point"]
            n_penalty = .003*result["reference"]
            if abs(n_penalty-result["trajectory"]["penalty"]) > TOL:
                raise ValueError("E penalty changed from its shared fixed reference.")
            before_metrics = endpoint(before, arrays=arrays, truth=truth, n=n, penalty=n_penalty)
            raw_metrics = endpoint(raw, arrays=arrays, truth=truth, n=n, penalty=n_penalty)
            kept_metrics = endpoint(er["point"], arrays=arrays, truth=truth, n=n, penalty=n_penalty)
            final_metrics = endpoint(result["trajectory"]["point"], arrays=arrays, truth=truth, n=n, penalty=n_penalty)
            if abs(before_metrics["J"]-er["before_score"]) > TOL or abs(raw_metrics["J"]-er["raw_score"]) > TOL:
                raise ValueError("The saved E scores do not replay from their actual fitted weights/counts.")
            gain, gate = float(er["before_score"]-er["raw_score"]), 1e-10+.001*float(before["objective"])
            if bool(er["accepted"]) != (gain > gate):
                raise ValueError("A target E outcome is not explained by its gain gate; inspect timing conditions.")
            bound, incumbent = ee.get("final_bound"), ee.get("final_objective")
            relative = None if bound is None or incumbent is None else max(0., incumbent-bound)/max(abs(incumbent), 1e-300)
            if ee.get("status") != 2 or relative is None or relative > .0001+1e-9:
                raise ValueError("A target E solve failed to meet the declared gap criterion.")
            pool = frozenset(tuple(C) for C in ee["pool"])
            row = dict(job=job, method=method, prefix_end_index=located["prefix_end_index"],
                       E_accepted=bool(er["accepted"]), gain=gain, gate=gate, gain_minus_gate=gain-gate,
                       native_before_J=er["before_score"], native_raw_score=er["raw_score"],
                       native_retained_score=er["score"],
                       gain_over_gate=gain/gate, acceptance_matches_gain_gate=True,
                       native_status=ee["status"], native_final_bound=bound, native_final_incumbent=incumbent,
                       native_record_bound=er.get("bound"), derived_native_relative_gap=relative,
                       gap_met=True, timeout=False, pool_count=len(pool),
                       pool_sha256=contract.digest(sorted(map(list, pool))), penalty=n_penalty,
                       hard=json.dumps(ee.get("hard", [])), forbidden=json.dumps(ee.get("forbidden", [])),
                       raw_proposal_is_retained_endpoint=contract._point(raw, n) == contract._point(er["point"], n),
                       retained_J_minus_raw_J=kept_metrics["J"]-raw_metrics["J"],
                       native_output_F1=float(frame[(frame.job == job) & (frame.method == method)].iloc[0].clade_f1))
            for name, metrics in (("before", before_metrics), ("raw_proposal", raw_metrics),
                                  ("retained", kept_metrics), ("final_path", final_metrics)):
                row.update({name+"_"+k: v for k, v in metrics.items()})
            E_rows.append(row)
            models[key] = dict(result=result, pool=pool, E_row=row, n=n)
            receipts[key] = dict(result_sha256=result_hashes[key], stage_location=located)
            for call in located["T_calls"]:
                event = result["trajectory"]["attempts"][call["attempt_index"]]
                point = path[call["before_index"]]
                tail_penalty = float(event["penalty"])
                b = endpoint(point, arrays=arrays, truth=truth, n=n, penalty=tail_penalty)
                if abs(b["J"]-event["before_score"]) > TOL:
                    raise ValueError("The saved T-before score does not replay.")
                tgain = float(event["before_score"]-event["decoded_score"])
                tgate = 1e-10+.001*float(point["objective"])
                if abs(tgain-event["gain"]) > TOL or abs(tgate-event["threshold"]) > TOL or bool(event["accepted"]) != (tgain > tgate):
                    raise ValueError("The saved T gate does not replay.")
                decoded_count_float = (float(event["decoded_score"])-float(event["decoded_objective"]))/tail_penalty
                decoded_count = round(decoded_count_float)
                if abs(decoded_count_float-decoded_count) > 1e-7:
                    raise ValueError("The decoded T score does not identify an integral retained support count.")
                trow = dict(job=job, method=method, attempt_index=call["attempt_index"],
                            before_index=call["before_index"], after_index=call["after_index"],
                            accepted=bool(event["accepted"]), before_L=b["L"], before_count=b["retained_proper_count"],
                            before_J=b["J"], decoded_L=event["decoded_objective"], decoded_count=decoded_count,
                            decoded_J=event["decoded_score"], gain=tgain, gate=tgate, gain_minus_gate=tgain-tgate,
                            penalty=tail_penalty, candidate_count=event["candidate_count"], native_status=event.get("status"),
                            native_dual_bound=event.get("dual_bound"), regularized_bound=event.get("regularized_bound"),
                            gain_gate_replayed=True, decoded_count_basis="(decoded_score-decoded_objective)/penalty",
                            full_decoded_weights_saved=call["accepted_proposal_index"] is not None)
                if call["accepted_proposal_index"] is not None:
                    decoded = endpoint(path[call["accepted_proposal_index"]], arrays=arrays, truth=truth, n=n, penalty=tail_penalty)
                    if abs(decoded["J"]-event["decoded_score"]) > TOL or decoded["retained_proper_count"] != decoded_count:
                        raise ValueError("An accepted T decoded point does not replay its score/count.")
                    trow.update(decoded_F1=decoded["F1"], decoded_false_positive=decoded["false_positive"],
                                decoded_positive_topology=decoded["positive_topology"])
                T_rows.append(trow)
    input_domains = []
    for job in jobs:
        group = [models[job+"/"+method] for method in methods]
        reference_pool = group[0]["pool"]
        same_pools = all(model["pool"] == reference_pool for model in group)
        same_penalties = len({model["E_row"]["penalty"] for model in group}) == 1
        same_hard = len({json.dumps(model["result"]["exit_event"].get("hard", [])) for model in group}) == 1
        same_forbidden = len({json.dumps(model["result"]["exit_event"].get("forbidden", [])) for model in group}) == 1
        if not (same_pools and same_penalties and same_hard and same_forbidden):
            raise ValueError("The three methods do not share exactly the same target E domain.")
        lower = max(model["E_row"]["native_final_bound"] for model in group)
        upper = min(model["E_row"]["native_final_incumbent"] for model in group)
        raw_scores = [model["E_row"]["native_raw_score"] for model in group]
        input_domains.append(dict(job=job, methods=json.dumps(methods), pool_count=len(reference_pool),
                                  all_three_pool_sets_equal=True, all_three_E_domains_equal=True,
                                  strongest_common_lower_bound=lower, best_common_incumbent=upper,
                                  common_absolute_gap=max(0., upper-lower),
                                  common_relative_gap=max(0., upper-lower)/max(abs(upper), 1e-300),
                                  raw_score_range=max(raw_scores)-min(raw_scores),
                                  bound_scope="same finite saved E domain and objective; bounds transferable across these three solves"))
    pairs = []
    for harm in harmful:
        job, method = harm["job"], harm["method"]
        left, right = models[job+"/noT"], models[job+"/"+method]
        lresult, rresult = left["result"], right["result"]
        le, re = left["E_row"], right["E_row"]
        same_pool = left["pool"] == right["pool"]
        same_domain = same_pool and lresult["exit_event"].get("hard", []) == rresult["exit_event"].get("hard", []) and lresult["exit_event"].get("forbidden", []) == rresult["exit_event"].get("forbidden", []) and le["penalty"] == re["penalty"]
        l_in_r = feasible_in_exit(lresult["trajectory"]["point"], rresult["exit_event"], left["n"])
        r_in_l = feasible_in_exit(rresult["trajectory"]["point"], lresult["exit_event"], left["n"])
        if not same_domain or not (l_in_r["feasible"] and r_in_l["feasible"]):
            raise ValueError("Target pair does not share the same E domain/cross-feasible endpoints.")
        lR, lX = matrices(lresult["exit_record"]["raw_point"], left["n"])
        rR, rX = matrices(rresult["exit_record"]["raw_point"], left["n"])
        pairs.append(dict(**harm, same_E_pool_set=same_pool, same_E_domain=True, pool_count=le["pool_count"],
                          noT_final_family_feasible_in_T_E_domain=l_in_r["feasible"],
                          T_final_family_feasible_in_noT_E_domain=r_in_l["feasible"],
                          E_raw_J_difference_T_minus_noT=re["raw_proposal_J"]-le["raw_proposal_J"],
                          E_raw_response_maxabs=float(max(np.max(abs(rR-lR)), np.max(abs(rX-lX)))),
                          E_raw_positive_topology_equal=le["raw_proposal_positive_topology"] == re["raw_proposal_positive_topology"],
                          E_retained_J_difference_T_minus_noT=re["retained_J"]-le["retained_J"],
                          noT_E_accepted=le["E_accepted"], T_E_accepted=re["E_accepted"],
                          noT_gain=le["gain"], noT_gate=le["gate"], T_gain=re["gain"], T_gate=re["gate"],
                          both_E_status_2=le["native_status"] == re["native_status"] == 2,
                          both_E_gap_met=le["gap_met"] and re["gap_met"],
                          strongest_shared_domain_lower_bound=max(le["native_final_bound"], re["native_final_bound"]),
                          best_shared_domain_incumbent=min(le["native_final_incumbent"], re["native_final_incumbent"])))
    target = output/"tail_gate_diagnostics"; target.mkdir(parents=True, exist_ok=True)
    artifacts = {"pairs.csv": pd.DataFrame(pairs), "exit_gates.csv": pd.DataFrame(E_rows),
                 "tail_calls.csv": pd.DataFrame(T_rows), "input_domains.csv": pd.DataFrame(input_domains)}
    for name, data in artifacts.items():
        data.to_csv(target/name, index=False, lineterminator="\n")
    verification = dict(passed=True, complete=True, native_verified=144, observed_harmful_pairs=len(harmful),
                        distinct_inputs=len(jobs), E_rows=len(E_rows), T_calls=len(T_rows),
                        harmful_pair_selection="post-hoc verified output F1(T)<F1(noT), descriptive diagnostics only",
                        before_L_location="saved stage_effects/locations.json prefix_end_index, replayed from exact path append control metadata",
                        score_definition="L=mean absolute training residual; J=L+penalty*retained proper internal support count",
                        E_penalty=".003*f0_MAE", T_penalty="method tail_coefficient*f0_MAE",
                        gate_definition="1e-10+.001*L_before", gain_definition="J_before-J_raw_proposal for E; J_before-J_decoded for T",
                        raw_proposal_and_retained_endpoint_separate=True,
                        same_E_domain_and_cross_feasible_final_families=True,
                        target_E_status_2_and_gap_met=True,
                        optimality_scope="saved finite E domains; same incumbent training objectives within numerical precision and recorded bounds; no whole-tree global claim",
                        cross_feasibility_basis="fixed-column E model: laminar supports, [0,2] weights, membership, hard/forbidden and cardinality cuts; residual epigraph can be set to absolute residual",
                        fit_calls=0, solver_calls=0, parameters_changed=False, candidate_generation_calls=0,
                        test_observations_opened=False, truth_used_only_for_postanalysis=True,
                        script_sha256=sha(__file__), contract_sha256=sha(Path(contract.__file__)),
                        stage_source_sha256=sha(Path(stage_effects.__file__)), truth_helper_sha256=sha(Path(truth_contract.__file__)),
                        native_verification_sha256=native_hash, protocol_sha256=protocol_hash,
                        per_model_csv_sha256=sha(table_path), stage_verification_sha256=sha(stage_path),
                        stage_locations_sha256=sha(locations_path), checked_native_result_hashes=result_hashes,
                        targeted_results=receipts, truth_source_hashes=truth_hashes,
                        result_hashes={name: sha(target/name) for name in artifacts}, seconds=perf_counter()-started)
    write(target/"verification.json", verification)
    print(pd.DataFrame(pairs).to_string(index=False))
    print(pd.DataFrame(E_rows)[["job", "method", "E_accepted", "before_L", "gain", "gate", "raw_proposal_J", "retained_J"]].to_string(index=False))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT/"outputs/day96_t_benchmark_20261110")
    diagnostics(parser.parse_args().output)
