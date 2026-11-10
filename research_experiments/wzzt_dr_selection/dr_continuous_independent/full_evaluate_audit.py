"""Read-only independent arithmetic audit of the completed five-contract run.

No AC simulation, fitting, search, or price update is performed. Observation
hashes and the frozen selections are checked before consulting network truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

import run as bounded
import actual_compare as actual


def require(condition, label):
    if not condition:
        raise ArithmeticError(label)


def resource_check(device, delivered):
    """Independently accumulate every device state and its declared limits."""
    values = np.asarray(delivered, float)
    power_error = max(float(np.maximum(device.delta_lower_kw-values, 0).max()),
                      float(np.maximum(values-device.delta_upper_kw, 0).max()))
    energy = np.zeros(device.R)
    energy_error = deadline_error = 0.
    for t in range(device.T):
        energy[device.reset[:, t]] = 0.
        for r in range(device.R):
            action = values[r, t]
            efficiency = device.eta[r] if action >= 0 else 1./device.eta[r]
            energy[r] += device.dt*efficiency*action
        energy_error = max(energy_error, float(np.max(np.maximum(device.energy_low-energy, 0))),
                           float(np.max(np.maximum(energy-device.energy_high, 0))))
        due = [r for r in range(device.R)
               if device.kind[r] != 1 and (t == 95 or device.kind[r] == 2 and t == 31)]
        if due:
            deadline_error = max(deadline_error, float(np.max(np.abs(energy[due]))))
    return dict(power_violation_kw=power_error, energy_violation_kwh=energy_error,
                deadline_violation_kwh=deadline_error,
                passed=max(power_error, energy_error, deadline_error)<1e-7,
                absolute_DR_kwh=float(device.dt*np.abs(values).sum()),
                net_DR_kwh=float(device.dt*values.sum()))


def independent_score(scene, pool):
    """Compute squared-voltage residuals and the unchanged w=2/common3% rule."""
    p, q, terminal = [np.asarray(scene[key], float)
                      for key in ("P_terminal", "Q_terminal", "V_terminal")]
    root = np.asarray(scene["V_root_observed"], float).reshape(-1, 1)
    require(p.shape == q.shape == terminal.shape == (96, len(pool["labels"])), "Channel shape")
    observed = root*root-terminal*terminal
    order = list(pool["selection_order"])
    models = {row["candidate_id"]:row for row in pool["models"]}
    require(set(models) == {"coarse", "refined"}, "Declared frozen candidate pair")
    losses = {}
    combined = {}
    for key in order:
        model = models[key]
        predicted = p@np.asarray(model["R"]).T+q@np.asarray(model["X"]).T
        losses[key] = float(np.mean(np.abs(observed-predicted)))
        combined[key] = (288*model["historical_validation_mae"]+192*losses[key])/480.
    history_min = min(model["historical_validation_mae"] for model in models.values())
    historical = [k for k in order if models[k]["historical_validation_mae"]<=1.03*history_min+1e-12]
    band = 1.03*min(combined[k] for k in historical)+1e-12
    eligible = [k for k in historical if combined[k]<=band]
    selected = min(eligible, key=lambda k:(len(models[k]["point"]["supports"]), combined[k], order.index(k)))
    gap = 1e6*(losses["coarse"]-1.03*losses["refined"])
    hc = 1e6*models["coarse"]["historical_validation_mae"]
    hf = 1e6*models["refined"]["historical_validation_mae"]
    threshold = 1.5*(1.03*hf-hc)+2.5e-6
    left = 1e6*(combined["coarse"]-1.03*combined["refined"]-1e-12)
    right = .4*(gap-threshold)
    return dict(selected=selected, excluded="coarse" not in eligible,
                D_micro=gap, h_micro=threshold, J_margin_micro=left,
                margin_identity_error=abs(left-right),
                J_C_micro=1e6*combined["coarse"], J_F_micro=1e6*combined["refined"],
                L_C_micro=1e6*losses["coarse"], L_F_micro=1e6*losses["refined"])


def main(args):
    began = time.perf_counter()
    folder = args.evaluation.resolve()
    destination = folder/"audit"
    destination.mkdir(exist_ok=True)
    completed = bounded.read(folder/"complete.json")
    require(completed["status"] in ("complete", "bounded_partial"), "Evaluation final checkpoint")
    request = bounded.read(bounded.checked(completed["request"]))
    bounded.checked(completed["results"])
    frozen = bounded.read(folder/"policy_freeze.json")
    contracts = bounded.unpickle(bounded.checked(frozen["contracts"]))
    problem = bounded.build_problem(args.archive.resolve(), args.public.resolve())
    device = problem["device"]
    metadata = bounded.read(args.archive/bounded.CASE/"inputs.json")
    pool = bounded.read(bounded.checked(metadata["pool.json"]))
    condition = bounded.read(bounded.checked(metadata["case_protocol.json"]))
    prediction_path = folder/"actual/predictions_frozen.json"
    prediction_freeze = bounded.read(prediction_path)
    require(prediction_freeze["truth_used"] is False, "Selection freeze before truth")
    scored = bounded.read(folder/"actual/evaluated_rows.json")
    by_key = {(r["path"],r["policy"]):r for r in scored}
    require(len(by_key) == len(scored), "Unique policy/path records")
    order = request["policy_order"]
    source_files = [Path(__file__), bounded.HERE/"run.py", bounded.HERE/"actual_compare.py"]
    audit_request = dict(evaluation=bounded.reference(folder/"complete.json"),
                         selection_freeze=bounded.reference(prediction_path),
                         scored_records=bounded.reference(folder/"actual/evaluated_rows.json"),
                         frozen_contracts=frozen["contracts"], pool=metadata["pool.json"],
                         sources={str(p.relative_to(bounded.TASK)):bounded.sha(p) for p in source_files},
                         python=platform.python_version(), numpy=np.__version__, read_only=True,
                         no_AC=True, no_fitting=True, no_price_update=True)
    bounded.write(destination/"request.json", audit_request)
    rows = []
    hashes = []
    response_children = np.random.SeedSequence(request["response_seed"]).spawn(request["AC_paths"])
    for original in prediction_freeze["rows"]:
        key = original["path"],original["policy"]
        require(key in by_key, "Scored record missing")
        old = by_key[key]
        row = dict(path=key[0], policy=key[1], status=old["status"])
        if "observation" not in old:
            row["observation_missing"] = True
            rows.append(row)
            continue
        path = bounded.checked(old["observation"])
        hashes.append(dict(path=key[0], policy=key[1], sha256=bounded.sha(path)))
        packet = bounded.unpickle(path)
        if old["status"] != "complete":
            rows.append(row)
            continue
        computed = independent_score(packet["scene"], pool)
        require(computed["selected"] == original["selected"] == old["selected"], "Frozen selection mismatch")
        require(computed["excluded"] == original["excluded"] == old["excluded"], "Frozen exclusion mismatch")
        row.update(computed)
        row["D_replay_error"] = abs(computed["D_micro"]-old["D_micro"])
        require(row["D_replay_error"]<1e-8, "Observed D mismatch")
        require(row["margin_identity_error"]<1e-8, "J/D/h identity mismatch")
        nominal = np.asarray(contracts[key[1]]["mean_u"])
        require(np.array_equal(nominal,packet["private"]["nominal"]), "Frozen nominal contract mismatch")
        delivered = np.asarray(packet["private"]["delivered"])
        rng = np.random.default_rng(response_children[key[0]])
        multiplier = 1.+device.relative_sd*rng.standard_normal((device.R,device.T))
        row["multiplier_replay_error"] = float(np.max(np.abs(multiplier-packet["private"]["multiplier"])))
        row["delivery_replay_error"] = float(np.max(np.abs(device.project(nominal*multiplier)-delivered)))
        require(row["multiplier_replay_error"]<1e-12 and row["delivery_replay_error"]<1e-12, "Response stream/controller replay")
        audit = resource_check(device,delivered)
        require(audit["passed"], "Delivered resource constraint violation")
        row["resource"] = audit
        row["energy_replay_error"] = abs(audit["absolute_DR_kwh"]-old["delivered_absolute_kwh"])
        require(row["energy_replay_error"]<1e-8, "Delivered energy mismatch")
        fee = float(device.fees(nominal))
        require(abs(fee-old["fee"])<1e-8, "Fixed planned contract fee mismatch")
        headroom = np.where(nominal>=0,device.delta_upper_kw,-device.delta_lower_kw)
        reward = np.divide(device.cost_full_activation*np.abs(nominal),headroom,
                           out=np.zeros_like(nominal),where=headroom>0)
        nominal_settlement = float(device.dt*np.sum(reward*np.abs(nominal)))
        require(abs(nominal_settlement-fee)<1e-8, "Frozen per-kWh reward does not replay plan fee")
        row.update(fixed_plan_contract_fee=fee,
                   frozen_reward_delivered_settlement=float(device.dt*np.sum(reward*np.abs(delivered))),
                   delivered_wear_proxy=float(device.fees(delivered)),
                   stored_wear_proxy=packet["summary"]["actual_resource_wear_proxy_fee"])
        require(abs(row["delivered_wear_proxy"]-row["stored_wear_proxy"])<1e-8, "Wear proxy mismatch")
        rows.append(row)
    # All arithmetic selections above are fixed before this truth-only call.
    runtime = actual.original.open_runtime(condition["source_root"],condition["source_metadata"],
        repo_root=bounded.REPO,protocol_ref=actual.original.ref(args.archive/"protocol.json"),
        labels=condition["labels"],case="paper15",v_sigma=condition["voltage_sigma"],pq_sigma=condition["pq_sigma"])
    truth = {tuple(sorted(block)) for block in runtime["legacy"].evaluate_supports(
             [],runtime["context"].net)["missed_blocks"]}
    models = {m["candidate_id"]:m for m in pool["models"]}
    for row in rows:
        if row["status"] != "complete":
            continue
        support = {tuple(sorted(block)) for block in models[row["selected"]]["frozen_output_support_labels"]}
        row["errors"] = len(support^truth)
        original = by_key[row["path"],row["policy"]]
        require(row["errors"]==original["errors"], "Post-freeze full support error mismatch")
        require((row["errors"]==0)==original["complete_recovery"], "Post-freeze complete recovery mismatch")
    results = {}
    for policy in order:
        complete_rows = [row for row in rows if row["policy"]==policy and row["status"]=="complete"]
        def mean(name):
            return float(np.mean([r[name] for r in complete_rows])) if complete_rows else None
        results[policy] = dict(completed=len(complete_rows), requested=request["AC_paths"],
            full_error_count=sum(r["errors"] for r in complete_rows),
            complete_recoveries=sum(r["errors"]==0 for r in complete_rows),
            coarse_exclusions=sum(r["excluded"] for r in complete_rows),
            fixed_plan_contract_fee=contracts[policy]["fee"],
            mean_frozen_reward_delivered_settlement=mean("frozen_reward_delivered_settlement"),
            mean_delivered_wear_proxy=mean("delivered_wear_proxy"),
            mean_D_micro=mean("D_micro"),mean_J_margin_micro=mean("J_margin_micro"))
    # Recount every saved independent-model path; no new random draw is made.
    sampling_checks = {}
    saved_results = bounded.read(folder/"results.json")
    for mode, spec in request["specs"].items():
        gap_parts, valid_parts = [], []
        for file in sorted((folder/mode).glob("batch_*.npz")):
            with np.load(file, allow_pickle=False) as batch:
                gap_parts.append(batch["gap"])
                valid_parts.append(batch["valid"])
        if not gap_parts:
            sampling_checks[mode] = dict(completed=0,requested=spec["samples"],policies={})
            continue
        gap = np.concatenate(gap_parts, axis=1)
        valid = np.concatenate(valid_parts, axis=1)
        passed = gap>problem["packet"]["threshold"]
        count = gap.shape[1]
        require(count==saved_results[mode]["completed"], "Saved MC completed count mismatch")
        sampling_checks[mode] = dict(completed=count, requested=spec["samples"],policies={})
        for i, policy in enumerate(order):
            successes = int(np.sum(passed[i]&valid[i]))
            n = int(np.sum(valid[i]))
            record = saved_results[mode]["policies"][policy]
            require(successes==record["successes"] and n==record["valid"], "Saved MC success/valid count mismatch")
            require(n==0 or abs(successes/n-record["probability"])<1e-14, "Saved MC fraction mismatch")
            sampling_checks[mode]["policies"][policy] = dict(successes=successes, valid=n,
                failures=count-n, missing=spec["samples"]-count,probability=successes/n if n else None)
    maxima = {}
    for name in ("D_replay_error","margin_identity_error","multiplier_replay_error","delivery_replay_error","energy_replay_error"):
        maxima[name] = max((row.get(name,0.) for row in rows),default=0.)
    bounded.write(destination/"rows_private.json",rows)
    bounded.write(destination/"observation_hashes.json",hashes)
    summary = dict(passed=True,records=len(rows),verified_observation_hashes=len(hashes),
                   complete=sum(r["status"]=="complete" for r in rows),
                   failures_or_unresolved=sum(r["status"]!="complete" for r in rows),
                   requested=request["AC_paths"]*len(order),policies=results,max_errors=maxima,
                   sampling_recount=sampling_checks,
                   matched_plan_fee_error=abs(contracts["new"]["fee"]-contracts["rotation"]["fee"]),
                   contract_fee_meaning="fixed nominal planned contract fee, constant across execution paths",
                   delivered_settlement_meaning="posthoc sum(dt * frozen per-kWh reward * absolute delivered power)",
                   wear_proxy_meaning="original fees(delivered); quadratic usage proxy, not an actual payment",
                   identity="J_C_micro-1.03*J_F_micro-1e-6=.4*(D_micro-h_micro)",
                   seconds=time.perf_counter()-began,no_AC_rerun=True,no_price_update=True)
    bounded.write(destination/"summary.json",summary)
    bounded.write(destination/"complete.json",dict(summary=bounded.reference(destination/"summary.json"),
                  request=bounded.reference(destination/"request.json"),all_checks_passed=True))
    print(json.dumps(bounded.plain(summary),ensure_ascii=True),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","public","evaluation"):
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
