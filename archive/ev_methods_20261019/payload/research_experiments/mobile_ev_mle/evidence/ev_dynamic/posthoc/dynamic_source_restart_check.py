"""Posthoc source-restart diagnostic; frozen policies and raw trials stay intact.

Select the two complete calibration histories with the smallest saved actual M
for design_v2g_mobile (seed breaks ties).  Refit only their saved validation and
noisy future observations, fixing each candidate's original training R/X.
This is an outcome-selected computational diagnostic, not an independent
strategy evaluation, calibration replacement, or global optimization proof.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))

import numpy as np
import scipy
from research_experiments.mobile_ev_mle import likelihood as lk
from research_experiments.mobile_ev_mle import gradient_price_study as io
from research_experiments.mobile_ev_mle.ev_fixed_train_fit import fit_fixed_inputs
from research_experiments.mobile_ev_mle.ev_fixed_train_review import review_fixed_fit

HERE = STAGE / "research_experiments" / "mobile_ev_mle"
METHOD = "design_v2g_mobile"
SEED_OFFSET = 9_000_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith(".gz") else data)


def saved_candidate(row, n):
    supports = tuple(tuple(s) for s in row["supports"])
    candidate = lk.RadialCandidate.from_supports(row["name"], n,
        tuple(s for s in supports if 1 < len(s) < n))
    if candidate.supports != supports:
        raise ValueError("candidate support/edge order changed")
    return candidate


def observations(row):
    return lk.Observations(**row)


def meter(row):
    return lk.MeasurementModel(**{k: np.asarray(v, float) if isinstance(v, list)
                                 else v for k, v in row.items()})


def margin(costs, true_name):
    return float(min(v for k, v in costs.items() if k != true_name)-costs[true_name])


def frozen_source_paths(freeze):
    return {name: (HERE.parents[1]/name if name.startswith("rnj_wzzt_core/")
                   else HERE/name) for name in freeze["source_sha256"]}


def source_audit(freeze, snapshot):
    if snapshot["source_sha256"] != freeze["source_sha256"]:
        raise ValueError("snapshot and freeze manifests differ")
    current = {}
    for name, path in frozen_source_paths(freeze).items():
        expected = freeze["source_sha256"][name]
        payload = base64.b64decode(snapshot["source_base64"][name], validate=True)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("frozen snapshot hash mismatch: "+name)
        current[name] = sha(path)
        if current[name] != expected:
            raise ValueError("current fitter/dependencies differ from freeze: "+name)
    return current


def trial_freeze_audit(trial, freeze):
    checks = dict(policy_id_matches=trial.get("policy_id")==freeze["policy_id"],
        source_manifest_matches=trial.get("source_sha256")==freeze["source_sha256"],
        source_unchanged_declared=trial.get("source_unchanged") is True,
        config_matches=trial.get("config")==freeze["config"])
    if not all(checks.values()):
        raise ValueError("trial/freeze provenance mismatch: "+str(trial.get("seed"))+" "+str(checks))
    return dict(seed=trial["seed"], **checks)


def refit_case(path, trial, *, counters):
    started = perf_counter()
    method = next(m for m in trial["methods"] if m["name"] == METHOD)
    training = {r["candidate"]: r for r in trial["training_result"]["fits"]}
    candidates = [saved_candidate(c, len(trial["training"]["p"][0]))
                  for c in trial["candidates"]]
    if len(candidates) != 4 or not method["noisy"]["all_complete"]:
        raise ValueError("diagnostic requires the saved four-candidate complete fit")
    options = {**trial["config"]["profile"], **trial["config"]["source"]}
    if options["multistart"] != 2:
        raise ValueError("unexpected original restart count")
    options["multistart"] = 8
    result = dict(seed=trial["seed"], true_name=trial["true_name"],
        input_path=str(path.resolve()), input_sha256=sha(path), original_M=method["M"],
        options=options, resolved_unchanged_defaults=dict(r_bounds=[1e-6,.1],
            x_bounds=[1e-6,.1], voltage_bounds=[.85,1.15]), fits=[], failures=[])
    original = {}; new = {}; retained = {}
    splits = (
        ("validation", trial["validation"], trial["validation_meter"], trial["validation_result"]),
        ("future_noisy", method["observations"], trial["future_meter"], method["noisy"]),
    )
    for split_index, (label, obs_row, meter_row, old_bank) in enumerate(splits):
        obs, model = observations(obs_row), meter(meter_row)
        by_name = {r["candidate"]: r for r in old_bank["fits"]}
        original[label] = {}; new[label] = {}; retained[label] = {}
        for candidate_index, candidate in enumerate(candidates):
            name = candidate.name
            old_row, train = by_name[name], training[name]
            r, x = np.asarray(train["r"],float), np.asarray(train["x"],float)
            if not old_row["success"] or not np.array_equal(old_row["r"],r) or not np.array_equal(old_row["x"],x):
                raise ValueError("original source fit does not retain training impedance")
            old_review = review_fixed_fit(candidate, obs, model, old_row, trained_r=r, trained_x=x)
            counters["independent_replay_AC_batches"] += 1
            counters["independent_replay_AC_snapshots"] += obs.t
            original[label][name] = float(old_row["loss"])
            retained[label][name] = float(old_row["loss"])
            fit_seed = int(trial["seed"])+SEED_OFFSET+100*split_index+candidate_index
            row = dict(split=label, candidate=name, restart_seed=fit_seed,
                original_SSE=old_row["loss"], original_success=old_row["success"],
                original_optimality=old_row["optimality"],
                original_restart_losses=[r.get("loss") for r in old_row["diagnostics"]["restarts"]],
                original_source_p=old_row["source_p"], original_source_q=old_row["source_q"],
                fixed_training_r=r.tolist(), fixed_training_x=x.tolist(),
                old_independent_review=old_review)
            before = dict(counters); began = perf_counter()
            try:
                fit = fit_fixed_inputs(candidate, obs, model,
                    initial=SimpleNamespace(r=r, x=x, source_p=old_row["source_p"], source_q=old_row["source_q"]),
                    fixed_r=r, fixed_x=x, seed=fit_seed, call_counter=counters, **options)
                fitted = io.serial(io.fit_record(candidate, fit))
                check = review_fixed_fit(candidate, obs, model, fitted, trained_r=r, trained_x=x)
                counters["independent_replay_AC_batches"] += 1
                counters["independent_replay_AC_snapshots"] += obs.t
                row.update(new_fit=fitted, new_independent_review=check,
                    new_success=bool(fit.success), new_SSE=float(fit.loss),
                    attained_SSE_improvement=float(old_row["loss"]-fit.loss),
                    new_optimality=float(fit.optimality),
                    new_restart_losses=[r.get("loss") for r in fit.diagnostics["restarts"]],
                    restart_successes=sum(bool(r.get("success")) for r in fit.diagnostics["restarts"]))
                if fit.success:
                    new[label][name] = float(fit.loss)
                    retained[label][name] = min(float(old_row["loss"]), float(fit.loss))
                else:
                    result["failures"].append(dict(split=label,candidate=name,error="source_fit_nonconverged"))
            except (ValueError, RuntimeError, AssertionError, FloatingPointError, np.linalg.LinAlgError) as error:
                row.update(new_success=False, error=type(error).__name__+": "+str(error))
                result["failures"].append(dict(split=label,candidate=name,error=row["error"]))
            row["seconds"] = perf_counter()-began
            row["new_source_fit_AC_batches"] = counters.get("ac_calls",0)-before.get("ac_calls",0)
            row["new_source_fit_AC_snapshots"] = row["new_source_fit_AC_batches"]*obs.t
            counters["AC_snapshots"] += row["new_source_fit_AC_snapshots"]
            row["retained_SSE"] = retained[label][name]
            row["retained_SSE_improvement"] = float(old_row["loss"]-row["retained_SSE"])
            result["fits"].append(row)
            print(f"seed={trial['seed']} split={label} candidate={name} old={old_row['loss']:.12g} new={row.get('new_SSE')} success={row['new_success']}",flush=True)
    def combined(values):
        return {c.name: values["validation"][c.name]+values["future_noisy"][c.name] for c in candidates}
    original_combined = combined(original)
    if not np.isclose(margin(original_combined,trial["true_name"]), method["M"], atol=2e-7,rtol=5e-8):
        raise ValueError("original actual M arithmetic mismatch")
    retained_combined = combined(retained)
    result.update(original_loss_by_candidate=original_combined,
        retained_loss_by_candidate=retained_combined,
        retained_M=margin(retained_combined,trial["true_name"]),
        delta_M=margin(retained_combined,trial["true_name"])-method["M"],
        all_new_fits_complete=not result["failures"],
        retained_rule="minimum of original independently replayed feasible SSE and successful new attained SSE for each split/candidate",
        original_validation_SSE=original["validation"], original_future_SSE=original["future_noisy"],
        retained_validation_SSE=retained["validation"], retained_future_SSE=retained["future_noisy"],
        seconds=perf_counter()-started)
    if not result["failures"]:
        result["new_only_M"] = margin(combined(new),trial["true_name"])
    return result


def run(directory):
    began=perf_counter(); directory=Path(directory).resolve()
    freeze_path=directory/"freeze.json"; snapshot_path=directory/"source_snapshot.json.gz"
    freeze, snapshot=read(freeze_path),read(snapshot_path)
    sources_before=source_audit(freeze,snapshot)
    paths=sorted(directory.glob("trial_*.json.gz")); ranking=[]; input_hashes={}; provenance=[]
    if len(paths)!=36: raise ValueError("expected all 36 calibration histories")
    for path in paths:
        trial=read(path); input_hashes[path.name]=sha(path)
        provenance.append(trial_freeze_audit(trial,freeze))
        selected=next(m for m in trial["methods"] if m["name"]==METHOD)
        value=selected.get("M")
        eligible=(trial["status"]=="complete" and selected["status"]=="complete" and value is not None and np.isfinite(value))
        ranking.append(dict(seed=trial["seed"],M=value,eligible=bool(eligible),path=str(path.resolve())))
    eligible=sorted((r for r in ranking if r["eligible"]),key=lambda r:(r["M"],r["seed"]))
    if len(eligible)<2: raise ValueError("fewer than two eligible histories")
    counters=dict(fit_calls=0,solver_calls=0,ac_calls=0,AC_snapshots=0,
        independent_replay_AC_batches=0,independent_replay_AC_snapshots=0)
    selected=[]
    for choice in eligible[:2]:
        path=Path(choice["path"])
        selected.append(refit_case(path,read(path),counters=counters))
    unchanged=all(sha(directory/name)==value for name,value in input_hashes.items())
    source_unchanged=source_audit(freeze,snapshot)==sources_before
    if counters["fit_calls"]!=16:raise ValueError("unexpected source-fit denominator")
    return io.serial(dict(schema="dynamic_source_restart_posthoc_v1",directory=str(directory),
        method=METHOD,selection_rule="among all 36 complete saved calibration histories: ascending actual M, then ascending seed; take first 2",
        posthoc=True,independent_strategy_evaluation=False,calibration_replaced=False,
        strategy_modified=False,source_box_modified=False,training_impedance_modified=False,
        future_noise_modified=False,global_optimality_certified=False,
        selection_population=ranking,selection_denominator=len(paths),eligible_denominator=len(eligible),
        selected_seeds=[r["seed"] for r in eligible[:2]],cases=selected,counters=counters,
        trial_freeze_provenance=provenance,
        fit_denominator=16,restart_denominator=128,
        attempted_restarts=sum(len(r.get("new_fit",{}).get("diagnostics",{}).get("restarts",[])) for c in selected for r in c["fits"]),
        attained_restarts=sum("attained_state" in restart for c in selected for r in c["fits"]
            for restart in r.get("new_fit",{}).get("diagnostics",{}).get("restarts",[])),
        successful_restarts=sum(r.get("restart_successes",0) for c in selected for r in c["fits"]),
        failed_source_fits=sum(len(c["failures"]) for c in selected),
        source_fits_AC_batches=counters["ac_calls"],source_fits_AC_snapshots=counters["AC_snapshots"],
        total_AC_batches=counters["ac_calls"]+counters["independent_replay_AC_batches"],
        total_AC_snapshots=counters["AC_snapshots"]+counters["independent_replay_AC_snapshots"],
        all_raw_inputs_unchanged=unchanged,all_frozen_sources_unchanged=source_unchanged,
        raw_input_sha256=input_hashes,freeze_sha256=sha(freeze_path),snapshot_sha256=sha(snapshot_path),
        frozen_source_sha256=sources_before,diagnostic_script_sha256=sha(__file__),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        process_threads=1,seed_rule="history seed + 9000000 + 100*split_index + candidate_index; split validation=0,future_noisy=1",
        restart_initialization="8 total starts: saved attained source warm start plus 7 independent Normal(0,.01pu) source perturbations, clipped using the original source box",
        interpretation="This diagnoses saved current-source local fits only. Better training-parameter or source-bound models and other AC equilibria are outside scope; retained local values are not global MLEs.",
        seconds=perf_counter()-began))


def audit_existing(output):
    """Add source/trial provenance and precise counts without any new fitting."""
    began=perf_counter(); result=read(output); directory=Path(result["directory"])
    freeze=read(directory/"freeze.json");snapshot=read(directory/"source_snapshot.json.gz")
    source_audit(freeze,snapshot)
    provenance=[]; ranking=[]
    for name,expected in result["raw_input_sha256"].items():
        path=directory/name
        if sha(path)!=expected: raise ValueError("raw changed since source fit: "+name)
        trial=read(path); provenance.append(trial_freeze_audit(trial,freeze))
        method=next(m for m in trial["methods"] if m["name"]==METHOD)
        if trial["status"]!="complete" or method["status"]!="complete" or method.get("M") is None:
            raise ValueError("postrun audit expected all 36 complete calibration histories")
        ranking.append((method["M"],trial["seed"]))
    if len(provenance)!=36 or [s for _,s in sorted(ranking)[:2]]!=result["selected_seeds"]:
        raise ValueError("postrun selection denominator/ranking differs")
    if "source_fit_execution_script_base64" in result:
        payload=base64.b64decode(result["source_fit_execution_script_base64"],validate=True)
        if hashlib.sha256(payload).hexdigest()!=result["source_fit_execution_script_sha256"]:
            raise ValueError("saved executed diagnostic script bytes/hash differ")
    restarts=[restart for case in result["cases"] for row in case["fits"]
              for restart in row.get("new_fit",{}).get("diagnostics",{}).get("restarts",[])]
    result.update(trial_freeze_provenance=provenance,
        attempted_restarts=len(restarts), attained_restarts=sum("attained_state" in r for r in restarts),
        restart_initialization="8 total starts: saved attained source warm start plus 7 independent Normal(0,.01pu) source perturbations, clipped using the original source box",
        diagnostic_script_sha256=sha(__file__), postrun_audit_script_sha256=sha(__file__),
        postrun_provenance_audit_seconds=perf_counter()-began,
        postrun_provenance_audit_new_fit_calls=0,
        postrun_script_revision="Only explicit trial/freeze provenance verification and restart labels were added after the 16 source fits; executed script bytes/hash are retained in this JSON. No fit was repeated.")
    io.write_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=HERE/"outputs"/"dynamic_calibration_v2")
    parser.add_argument("--output",type=Path,default=Path(__file__).with_suffix(".json"))
    parser.add_argument("--audit-existing",action="store_true",help="only verify provenance/count labels of the existing result; perform no source fits")
    args=parser.parse_args()
    if args.audit_existing:
        result=audit_existing(args.output)
        print(json.dumps(dict(output=str(args.output.resolve()),verified_trial_provenance=len(result["trial_freeze_provenance"]),
            attempted_restarts=result["attempted_restarts"],attained_restarts=result["attained_restarts"],new_fit_calls=0)),flush=True)
        return
    if args.output.exists(): raise ValueError("diagnostic output already exists; keep it and choose a new output")
    try:
        from threadpoolctl import threadpool_limits
        limit=threadpool_limits(limits=1)
    except ImportError:
        limit=nullcontext()
    with limit: result=run(args.directory)
    io.write_json(args.output,result)
    print(json.dumps(dict(output=str(args.output.resolve()),selected_seeds=result["selected_seeds"],
        case_M=[dict(seed=c["seed"],old=c["original_M"],retained=c["retained_M"],delta=c["delta_M"]) for c in result["cases"]],
        counters=result["counters"],seconds=result["seconds"]),ensure_ascii=False),flush=True)


if __name__=="__main__": main()
