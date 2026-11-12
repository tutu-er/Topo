"""Freeze all available policies, then independently check model and AC paths."""
from __future__ import annotations

import os
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from copy import copy
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from ... import io
from ...dr.pipeline import load_resources
from ...observations import load
from .history_scan import BETAS, ROOT, load_request, require, sha, utc
from .posterior import add_observations, log_weights, retained
from .prospective import ForecastLaw, binomial_interval, failure_bound, simulate_target, spectral_moment, stream
from .physical import execute, generate_background, open_runtime


def rotation(original, desired_fee):
    """Original heterogeneous pilot's fixed two-hour exploration and scalar fee matching."""
    x = np.zeros((original.R, 24))
    phase = np.arange(original.R) % 4
    for r in range(original.R):
        x[r] = np.where((np.arange(24)//2+phase[r]) % 2 == 0, 1., -1.)
        if original.kind[r] == 1:
            x[r] = np.minimum(x[r], 0.)
    if desired_fee == 0:
        return np.zeros_like(original.delta_lower_kw), dict(matched=True, fee=0., ratio=None, scale=0.)
    full = original.nominal(x); full_fee = original.fees(full)
    if full_fee < desired_fee:
        return full, dict(matched=False, fee=full_fee, ratio=full_fee/desired_fee, scale=1.,
                          reason="fixed rotation at full scale is below quoted fee")
    left, right = 0., 1.
    for _ in range(50):
        middle = (left+right)/2
        delivery = original.nominal(middle*x); fee = original.fees(delivery)
        if abs(fee-desired_fee) <= .001*desired_fee:
            break
        if fee < desired_fee:
            left = middle
        else:
            right = middle
    require(original.audit(delivery)["passed"], "Rotation left original resource domain")
    return delivery, dict(matched=abs(fee-desired_fee) <= .01*desired_fee,
                          fee=fee, ratio=fee/desired_fee, scale=middle)


def prepare(run):
    require((run/"planning/complete.json").exists(), "Finish every bounded pricing attempt before opening confirmation")
    require(io.read(run/"physical/historical_bridge_verification.json")["status"] == "passed", "Physical bridge not verified")
    out = run/"confirmation"; require(not out.exists(), "Existing confirmation must be preserved")
    out.mkdir()
    planning = io.read(run/"planning/protocol.json")
    requests = {r["id"]: r for r in io.read(run/"requests.json")}
    policies = []
    for item in planning["selected"]:
        request = requests[item["id"]]
        scored = io.read(run/"scores"/(item["id"]+".json"))
        original, _, _ = load_resources(request["case"], labels=scored["labels"])
        for target in item["targets"]:
            priced = run/"planning/targets"/(item["id"]+f"__target{target}")
            result = io.read(priced/"summary.json")
            entry = dict(id=item["id"], target=target, role=item["role"], case=item["case"],
                target_topology_id=scored["models"][target]["topology_id"],
                quote_summary=io.ref(priced/"summary.json"), quote_status=result["status"], policies={})
            means = {"no_dr": np.zeros_like(original.delta_lower_kw)}
            if result["qualified"]:
                with np.load(priced/"quote.npz", allow_pickle=False) as data:
                    means["optimized"] = np.array(data["mean_u"], copy=True)
                fee = original.fees(means["optimized"])
                means["rotation"], rotation_receipt = rotation(original, fee)
                entry["rotation_matching"] = rotation_receipt
                entry["quote_npz"] = io.ref(priced/"quote.npz")
            for name in ("no_dr", "rotation", "optimized"):
                if name in means:
                    require(original.audit(means[name])["passed"], "Frozen policy violates original devices")
                    path = out/"policies"/(item["id"]+f"__target{target}__{name}.npz")
                    path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path, mean_u=means[name])
                    entry["policies"][name] = dict(available=True, nominal=io.ref(path),
                        contract_fee=original.fees(means[name]), delivery_sha256=io.contract_hash(means[name]))
                else:
                    entry["policies"][name] = dict(available=False, reason="No qualified optimized quote; no fee-matched comparison defined")
            policies.append(entry)
    sources = {}
    for path in [*sorted(Path(__file__).parent.glob("*.py")), *sorted((ROOT/"rnj_wzzt_dr/dr").glob("*.py"))]:
        relative = path.relative_to(ROOT)
        destination = out/"source_snapshot"/relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
        sources[relative.as_posix()] = io.ref(path)
    protocol = dict(created_utc=utc(), policies=policies, sources=sources,
        historical_selection_seal=io.ref(run/"selection_seal.json"), planning_protocol=io.ref(run/"planning/protocol.json"),
        planning_completion=io.ref(run/"planning/complete.json"), bridge=io.ref(run/"physical/historical_bridge_verification.json"),
        beta_star=planning["beta_star"], betas=BETAS, eta=.9, model_paths=2048,
        actual_replicates=list(range(20000, 20032)), actual_study_seed=2026111201,
        actual_namespace="posterior_dr_confirm_20261112; original day96-v1 generator at independent study_seed/replicate",
        unavailable_policy="not executed; planning failure, not a zero action or an observed AC failure",
        fee_rule="original fixed quoted contract fee; delivered-resource fee is a wear proxy, not extra payment",
        original_delivery="Devices.draw full correlated horizon then Devices.project(nominal_mean*multiplier)",
        actual_future_opened=False, controls_and_thresholds_frozen=True)
    io.write(out/"protocol.json", protocol)
    io.write(out/"policy_seal.json", dict(created_utc=utc(), protocol_sha256=sha(out/"protocol.json"),
        policy_files={p["nominal"]["path"]: p["nominal"]["sha256"] for row in policies for p in row["policies"].values() if p["available"]}))
    return out, protocol


def policies_checked(entry):
    result = {}
    for name, policy in entry["policies"].items():
        if policy["available"]:
            require(sha(policy["nominal"]["path"]) == policy["nominal"]["sha256"], "Frozen policy changed")
            with np.load(policy["nominal"]["path"], allow_pickle=False) as data:
                result[name] = np.array(data["mean_u"], copy=True)
    return result


def confirm_model(run, out, protocol):
    requests = {r["id"]: r for r in io.read(run/"requests.json")}
    rows = []; conflicts = []
    for entry in protocol["policies"]:
        request = requests[entry["id"]]
        _, train, valid = load_request(request)
        forecast = np.column_stack((np.vstack((train.p, valid.p)), np.vstack((train.q, valid.q))))
        law = ForecastLaw(forecast)
        scored = io.read(run/"scores"/(entry["id"]+".json"))
        H = np.asarray([p["H"] for p in scored["parameters"]])
        lp = np.array([m["log_probability"] for m in scored["models"]])
        original, _, base = load_resources(request["case"], labels=train.labels)
        policies = policies_checked(entry); cache = {}; sample_arrays = {}
        for name, mean_u in policies.items():
            identity = io.contract_hash(mean_u)
            if identity not in cache:
                dp, dq = original.aggregate(mean_u)
                controlled = copy(law)
                controlled.mean = law.mean+np.column_stack((dp, dq))/base
                logp = simulate_target(H, lp, entry["target"], controlled, scored["sigma2"],
                    stream("posterior_dr_model_confirm_20261112", entry["id"], entry["target"]), protocol["model_paths"])
                moments = {j: spectral_moment(H[entry["target"]]-H[j], controlled, scored["sigma2"])
                           for j in range(len(H)) if j != entry["target"]}
                cache[identity] = logp, moments
            logp, moments = cache[identity]; sample_arrays[name] = logp
            gates = {}
            for beta in BETAS:
                bound = failure_bound(lp, moments, entry["target"], beta)
                success = int(np.sum(logp >= np.log(beta)))
                gates[str(beta)] = dict(success_lower=bound["success_lower"], successes=success,
                    paths=len(logp), probability=success/len(logp), interval95=binomial_interval(success, len(logp)))
                if gates[str(beta)]["interval95"][1]+1e-12 < bound["success_lower"]:
                    conflicts.append(dict(id=entry["id"], target=entry["target"], strategy=name, beta=beta,
                                          reason="model interval upper below analytical lower; investigate before physical paths"))
            rows.append(dict(id=entry["id"], target=entry["target"], strategy=name, gates=gates))
        np.savez_compressed(out/(entry["id"]+f"__target{entry['target']}_model_confirmation.npz"), **sample_arrays)
        print("Independent model confirmation: "+entry["id"]+" target "+str(entry["target"]), flush=True)
    io.write(out/"model_results.json", rows)
    io.write(out/"model_complete.json", dict(status="complete", created_utc=utc(), rows=len(rows),
        protocol_sha256=sha(out/"protocol.json"), results_sha256=sha(out/"model_results.json"),
        interval_conflicts=conflicts, consistency_passed=not conflicts))


def score_actual(scored, observed, truth_id, target_topology_id):
    H = np.asarray([p["H"] for p in scored["parameters"]])
    history = np.asarray([p["history_sse"] for p in scored["parameters"]])
    power = np.column_stack((observed.p, observed.q))
    calculated = add_observations(history, H, power, observed.y, scored["sigma2"])
    logp = calculated["log_probability"]
    best = int(np.argmax(logp)); ids = [m["topology_id"] for m in scored["models"]]
    true_index = next((i for i, identity in enumerate(ids) if identity == truth_id), None)
    target_index = next((i for i, identity in enumerate(ids) if identity == target_topology_id), None)
    p_target = float(np.exp(logp[target_index])) if target_index is not None else None
    target_pair = None
    if target_index is not None and len(ids) > 1:
        other = max(logp[j] for j in range(len(ids)) if j != target_index)
        target_pair = float(np.exp(logp[target_index]-np.logaddexp(logp[target_index], other)))
    decisions = {}
    for beta in BETAS:
        kept = retained(logp, beta); contains = true_index in kept if true_index is not None else False
        decisions[str(beta)] = dict(size=len(kept), correct_unique=len(kept) == 1 and contains,
            wrong_unique=len(kept) == 1 and not contains, truth_retained=contains,
            target_pass=p_target >= beta if p_target is not None else None,
            pair_pass_but_full_fail=bool(target_pair is not None and target_pair >= beta and p_target < beta))
    residual = observed.y[None]-np.einsum("tc,knc->ktn", power, H)
    fresh_slot = np.sum(residual**2, axis=2).T
    prefix_sse = np.vstack((history, history[None]+np.cumsum(fresh_slot, axis=0)))
    prefix = log_weights(prefix_sse, scored["sigma2"])
    return dict(p_target=p_target, p_true=float(np.exp(logp[true_index])) if true_index is not None else None,
        p_max=float(np.exp(logp[best])), map_correct=ids[best] == truth_id,
        selected_topology_id=ids[best], selected_clades=scored["models"][best]["active_clades"],
        target_pair_weight=target_pair, truth_covered=true_index is not None,
        decisions=decisions, log_probability=logp, prefix_log_probability=prefix,
        target_index=target_index, true_index=true_index)


def execute_actual(run, out, protocol):
    require((out/"model_complete.json").exists(), "Independent model check must precede physical paths")
    require(io.read(out/"model_complete.json")["consistency_passed"], "Model-bound consistency needs investigation")
    require(not (out/"actual_started.json").exists(), "Existing actual sample attempt must be preserved")
    io.write(out/"actual_started.json", dict(created_utc=utc(), protocol_sha256=sha(out/"protocol.json")))
    requests = {r["id"]: r for r in io.read(run/"requests.json")}
    evaluation = {r["id"]: r for r in io.read(run/"evaluation.json")}
    selection = io.read(run/"planning/protocol.json")["selected"]
    modules = {}; rows = []; observations_written = 0; actual_ac_days = 0
    for selected in selection:
        identity = selected["id"]; request = requests[identity]
        if request["cohort"] not in modules:
            modules[request["cohort"]] = open_runtime(run, request["cohort"])
        module, batch, original_protocol = modules[request["cohort"]]
        old_metadata = io.read(batch/"generation.json")[request["job"]]["metadata"]
        parameters = {k: old_metadata[k] for k in ("case", "replicate", "voltage_sigma", "pq_sigma", "root_sigma", "study_seed", "profile", "level")}
        entries = [entry for entry in protocol["policies"] if entry["id"] == identity]
        score_banks = {method: io.read(run/"scores"/(identity.rsplit("__", 1)[0]+"__"+method+".json")) for method in ("101", "111")}
        original, _, _ = load_resources(request["case"], labels=score_banks["101"]["labels"])
        nominal_policies = {(entry["target"], name): mean for entry in entries for name, mean in policies_checked(entry).items()}
        truth = evaluation[identity]
        for replicate in protocol["actual_replicates"]:
            parameters.update(replicate=replicate, study_seed=protocol["actual_study_seed"])
            trial_rows = []
            try:
                background = generate_background(module, parameters); actual_ac_days += 1
                require(background["metadata"]["physical_branch_sha256"] == old_metadata["physical_branch_sha256"], "Actual pressure network differs from selected version")
                response_seed = int(stream("posterior_dr_confirm_20261112_devices", identity, replicate).integers(0, 2**31-1))
                multiplier = original.draw(response_seed, 1)[0]
                executions = {}
                for entry in entries:
                    for name in ("no_dr", "rotation", "optimized"):
                        if not entry["policies"][name]["available"]:
                            for method in ("101", "111"):
                                trial_rows.append(dict(id=identity, case=request["case"], target=entry["target"], method=method,
                                    strategy=name, replicate=replicate, role=selected["role"], status="not_run_no_quote"))
                            continue
                        mean = nominal_policies[entry["target"], name]; key = io.contract_hash(mean)
                        if key not in executions:
                            observed, execution, private = execute(module, background, original, mean, multiplier)
                            actual_ac_days += execution["ac_day_calls"]
                            path = out/"actual_observations"/identity/(str(replicate)+"__"+key[:16]+".npz")
                            path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(path, P=observed.p, Q=observed.q, V=observed.voltage,
                                root=observed.root, labels=observed.labels)
                            private_path = out/"actual_execution"/identity/(str(replicate)+"__"+key[:16]+".npz")
                            private_path.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(private_path, **private)
                            observations_written += 1
                            executions[key] = observed, execution, io.ref(path)
                        observed, execution, reference = executions[key]
                        for method, bank in score_banks.items():
                            decision = score_actual(bank, observed, truth["truth_id"], entry["target_topology_id"])
                            prefix = decision.pop("prefix_log_probability"); logp = decision.pop("log_probability")
                            prefix_file = out/"prefix"/(identity+f"__target{entry['target']}__{name}__{method}__{replicate}.npz")
                            prefix_file.parent.mkdir(exist_ok=True)
                            np.savez_compressed(prefix_file, prefix_log_probability=prefix, final_log_probability=logp)
                            real = set(map(tuple, truth["truth_clades"])); predicted = set(map(tuple, decision.pop("selected_clades")))
                            trial_rows.append(dict(id=identity, case=request["case"], target=entry["target"], method=method,
                                strategy=name, replicate=replicate, role=selected["role"], status="complete",
                                target_correct=entry["target_topology_id"] == truth["truth_id"],
                                missing=len(real-predicted), extra=len(predicted-real), observation=reference,
                                execution=execution, prefix=io.ref(prefix_file), **decision))
            except Exception as exc:
                trial_rows = []
                for entry in entries:
                    for name in ("no_dr", "rotation", "optimized"):
                        for method in ("101", "111"):
                            trial_rows.append(dict(id=identity, case=request["case"], target=entry["target"], method=method,
                                strategy=name, replicate=replicate, role=selected["role"],
                                status="failed" if entry["policies"][name]["available"] else "not_run_no_quote",
                                error=repr(exc), traceback=traceback.format_exc()))
            rows.extend(trial_rows)
            io.write(out/"actual_rows_checkpoint.json", rows)
        print("Physical 32 paired paths completed: "+identity, flush=True)
    io.write(out/"actual_results.json", rows)
    io.write(out/"actual_complete.json", dict(status="complete", created_utc=utc(),
        results_sha256=sha(out/"actual_results.json"), protocol_sha256=sha(out/"protocol.json"),
        distinct_observation_files=observations_written, actual_ac_day_calls=actual_ac_days,
        source_bfs_time_slots=96*actual_ac_days, physical_inputs=len(selection),
        identifier_refits=0, parameter_refits=0, interpretation="Only available frozen policies were executed; missing quotes stay not_run."))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(); run = args.run.resolve()
    out, protocol = prepare(run)
    confirm_model(run, out, protocol)
    execute_actual(run, out, protocol)


if __name__ == "__main__":
    main()
