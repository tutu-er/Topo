"""Post-selection development diagnostics of frozen-model extrapolation.

Consumes saved actual curves only. Oracle parameters are offline diagnostics;
no action is chosen or changed. Open-R/X joint fits use the same measured P/Q
and output likelihood as the new experiment and do not reuse any threshold.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from . import gradient_price_study as io
from . import likelihood as lk
from .ev_fixed_train_fit import fit_fixed_inputs, _FixedInputProblem
from .ev_fixed_train_review import review_fixed_fit
from .information_context import meter_from_record

HERE = Path(__file__).resolve().parent


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bank(record):
    return [lk.RadialCandidate.from_supports(row["name"], 3, row["supports"]) for row in record["candidates"]]


def _observations(record):
    return lk.Observations(**record)


def _by_name(rows):
    return {row["candidate"]: row for row in rows}


def _margin(losses, true_name):
    if true_name not in losses or len(losses) < 2 or any(v is None or not np.isfinite(v) for v in losses.values()):
        return None
    return float(min(v for k, v in losses.items() if k != true_name)-losses[true_name])


def _join_models(models, observations):
    changes = {}
    for k in ("p", "q", "v", "master_p", "master_q"):
        changes[k+"_sigma"] = np.concatenate([lk._array(getattr(m, k+"_sigma"),
            (o.t, o.n) if k in ("p", "q", "v") else (o.t,), k+"_sigma") for m, o in zip(models, observations)])
    for k in ("p", "q", "v"):
        changes["bias_"+k] = np.concatenate([lk._array(getattr(m, "bias_"+k), (o.t, o.n), "bias_"+k)
            for m, o in zip(models, observations)])
    return replace(models[0], **changes)


def _input_basis(observations, model):
    x = np.c_[observations.p-lk._array(model.bias_p, (observations.t, observations.n), "bias_p"),
              observations.q-lk._array(model.bias_q, (observations.t, observations.n), "bias_q")]
    centered = x-x.mean(axis=0)
    _, values, right = np.linalg.svd(centered, full_matrices=False)
    fraction = values**2/max(float(values@values), np.finfo(float).tiny)
    rank99 = int(np.searchsorted(np.cumsum(fraction), .99)+1)
    covariance = centered.T@centered/max(observations.t-1, 1)
    inverse = np.linalg.pinv(covariance, rcond=1e-10)
    return dict(x=x, center=x.mean(axis=0), singular=values, right=right, rank99=rank99,
        fraction=fraction, inverse=inverse, centered_rms=float(np.linalg.norm(centered)/np.sqrt(observations.t)),
        offspace_rms=float(np.linalg.norm(centered-centered@right[:rank99].T@right[:rank99])/np.sqrt(observations.t)))


def _input_metrics(observations, model, basis):
    x = np.c_[observations.p-lk._array(model.bias_p, (observations.t, observations.n), "bias_p"),
              observations.q-lk._array(model.bias_q, (observations.t, observations.n), "bias_q")]
    delta = x-basis["center"]
    main = basis["right"][:basis["rank99"]]
    off = delta-delta@main.T@main
    leverages = np.einsum("ti,ij,tj->t", delta, basis["inverse"], delta)
    return dict(times=observations.t, centered_norm_by_time=np.linalg.norm(delta, axis=1).tolist(),
        off_training_99percent_space_norm_by_time=np.linalg.norm(off, axis=1).tolist(),
        Mahalanobis_leverage_by_time=leverages.tolist(), max_Mahalanobis_leverage=float(max(leverages)),
        mean_Mahalanobis_leverage=float(np.mean(leverages)),
        input_rms_relative_training=float(np.linalg.norm(delta)/np.sqrt(observations.t)/max(basis["centered_rms"], 1e-30)),
        offspace_rms_relative_training=float(np.linalg.norm(off)/np.sqrt(observations.t)/max(basis["offspace_rms"], 1e-30)),
        input_min=x.min(axis=0).tolist(), input_max=x.max(axis=0).tolist(),
        scope="descriptive observed-input leverage, no confidence-region or independence claim")


def _line_geometry(candidate, observations, model, row, options, truth=None, counter=None):
    problem = _FixedInputProblem(candidate, observations, model, initial=row,
        r_bounds=options["r_bounds"], x_bounds=options["x_bounds"],
        source_p_bounds=options["source_p_bounds"], source_q_bounds=options["source_q_bounds"], call_counter=counter)
    z = np.r_[row["r"], row["x"], np.c_[row["source_p"], row["source_q"]].ravel()]
    jac = problem.jacobian(z)
    line, source = jac[:, :2*candidate.m], jac[:, 2*candidate.m:]
    left, singular_source, _ = np.linalg.svd(source, full_matrices=False)
    source_rank = int(np.count_nonzero(singular_source > singular_source[0]*1e-10))
    line = line-left[:, :source_rank]@(left[:, :source_rank].T@line)
    widths = problem.upper[:2*candidate.m]-problem.lower[:2*candidate.m]
    _, values, right = np.linalg.svd(line*widths[None, :], full_matrices=False)
    theta = z[:2*candidate.m]
    relative = (theta-problem.lower[:2*candidate.m])/widths
    near = (relative <= 1e-5) | (relative >= 1-1e-5)
    diagnostic = dict(scaled_line_profile_singular_values=values.tolist(),
        scaled_line_profile_rank_1e7=int(np.count_nonzero(values > values[0]*1e-7)),
        scaled_line_profile_condition=float(values[0]/values[-1]) if values[-1] > 0 else None,
        line_scaling="full declared per-edge R/X box width", source_rank=source_rank,
        source_profile_projection="unbounded local tangent only; source-active-bound information omitted",
        near_line_bound_count=int(np.count_nonzero(near)), near_line_bound_mask=near.tolist(),
        line_relative_box_coordinates=relative.tolist(), residual_rows=jac.shape[0])
    if truth is not None:
        error = theta-np.r_[truth["r"], truth["x"]]
        normalized_error = error/widths
        diagnostic.update(R_relative_error=float(np.linalg.norm(error[:candidate.m])/np.linalg.norm(truth["r"])),
            X_relative_error=float(np.linalg.norm(error[candidate.m:])/np.linalg.norm(truth["x"])),
            line_error_in_box_width_units=normalized_error.tolist(),
            error_coordinates_in_profile_right_singular_basis=(right@normalized_error).tolist(),
            linearized_source_profile_output_error_norm=float(np.linalg.norm(line@error)),
            linearized_source_profile_output_error_RMS=float(np.linalg.norm(line@error)/np.sqrt(jac.shape[0])),
            linearization_scope="local derivative times finite parameter error; actual oracle replay reported separately")
    return diagnostic


def _do_fit(candidate, obs, meter, initial, options, counter, *, fixed_r=None, fixed_x=None, seed):
    fit = fit_fixed_inputs(candidate, obs, meter, initial=initial, fixed_r=fixed_r, fixed_x=fixed_x,
        call_counter=counter, seed=seed, **options)
    row = io.serial(io.fit_record(candidate, fit))
    review = review_fixed_fit(candidate, obs, meter, row, trained_r=fixed_r, trained_x=fixed_x)
    return fit, row, review


def diagnose_history(path, *, run_open=True):
    before = _hash(path)
    record = io.read_json(path)
    if not record["training_result"]["all_complete"] or not record["validation_result"]["all_complete"]:
        raise ValueError("post-selection diagnostic requires saved training/validation fits; method failures are retained")
    started = perf_counter()
    bank = _bank(record)
    true_name = record["true_name"]
    true_candidate = next(c for c in bank if c.name == true_name)
    train = _observations(record["training"])
    validation = _observations(record["validation"])
    tm, vm, fm = (meter_from_record(record[k]) for k in ("training_meter", "validation_meter", "future_meter"))
    trained = _by_name(record["training_result"]["fits"])
    validated = _by_name(record["validation_result"]["fits"])
    options = {**record["config"]["source"], **{k:record["config"]["training"][k] for k in ("r_bounds", "x_bounds")},
        "multistart": 2, "max_nfev": 350, "tol": 1e-10}
    oracle_options = dict(options, max_nfev=200)
    geometry_counter, oracle_counter, open_counter = {}, {}, {}
    basis = _input_basis(train, tm)
    # Truth is read only here, after the action has already been saved.
    truth = record["truth"]
    true_x = np.c_[np.asarray(truth["p"])[record["training_indices"]],
                   np.asarray(truth["q"])[record["training_indices"]]]
    _, true_singular, _ = np.linalg.svd(true_x-true_x.mean(axis=0), full_matrices=False)
    true_fraction = true_singular**2/float(true_singular@true_singular)
    geometry = dict(observed_training_input_singular_values=basis["singular"].tolist(),
        observed_training_input_energy_fraction=basis["fraction"].tolist(), observed_training_99percent_rank=basis["rank99"],
        true_training_input_singular_values=true_singular.tolist(), true_training_input_energy_fraction=true_fraction.tolist(),
        truth_input_scope="offline generation-truth diagnostic; never used by selector",
        training_inputs=_input_metrics(train, tm, basis), validation_inputs=_input_metrics(validation, vm, basis),
        candidates={c.name:_line_geometry(c, train, tm, trained[c.name], options,
            truth=truth if c.name == true_name else None, counter=geometry_counter) for c in bank})
    oracle_fits, oracle_reviews = {}, {}
    for part, obs, meter, initial in (("training", train, tm, trained[true_name]),
                                       ("validation", validation, vm, validated[true_name])):
        _, fitrow, review = _do_fit(true_candidate, obs, meter, initial, oracle_options, oracle_counter,
            fixed_r=truth["r"], fixed_x=truth["x"], seed=record["seed"]+12000)
        oracle_fits[part], oracle_reviews[part] = fitrow, review
    cache, method_rows = {}, []
    for method in record["methods"]:
        row = dict(name=method["name"], original_status=method["status"], original_heldout_M=method.get("M"))
        if method.get("power_kw") is None or not method.get("noisy", {}).get("all_complete", False):
            row.update(status="original_failure_retained", oracle_result=None, open_result=None)
            method_rows.append(row)
            continue
        power = np.asarray(method["power_kw"], float)
        key = np.round(power, 10).astype("<f8").tobytes().hex()
        row["curve_key"] = hashlib.sha256(bytes.fromhex(key)).hexdigest()
        if key not in cache:
            future = _observations(method["observations"])
            fixed_future = _by_name(method["noisy"]["fits"])
            _, oracle_row, oracle_review = _do_fit(true_candidate, future, fm, fixed_future[true_name],
                oracle_options, oracle_counter, fixed_r=truth["r"], fixed_x=truth["x"], seed=record["seed"]+13000)
            fixed_heldout = {name:validated[name]["loss"]+fixed_future[name]["loss"] for name in trained}
            fixed_all = {name:trained[name]["loss"]+fixed_heldout[name] for name in trained}
            oracle_heldout = dict(fixed_heldout, **{true_name:oracle_fits["validation"]["loss"]+oracle_row["loss"]})
            oracle_all = dict(fixed_all, **{true_name:oracle_fits["training"]["loss"]+oracle_heldout[true_name]})
            data = dict(power_kw=power, future_input_geometry=_input_metrics(future, fm, basis),
                future_true_candidate_line_geometry=_line_geometry(true_candidate, future, fm,
                    fixed_future[true_name], options, truth=truth, counter=geometry_counter),
                original_fixed=dict(heldout_losses=fixed_heldout, all_losses=fixed_all,
                    heldout_M=_margin(fixed_heldout,true_name), all_M=_margin(fixed_all,true_name)),
                oracle_true_theta=dict(future_fit=oracle_row, future_review=oracle_review,
                    heldout_losses=oracle_heldout, all_losses=oracle_all,
                    heldout_M=_margin(oracle_heldout,true_name), all_M=_margin(oracle_all,true_name),
                    deployed_policy=False, wrong_candidates_keep_original_training_theta=True))
            if abs(data["original_fixed"]["heldout_M"]-float(method["M"])) > 1e-8:
                raise ValueError("saved heldout margin does not reproduce")
            if run_open:
                joint = lk.concat_observations(train, validation, future)
                jm = _join_models((tm, vm, fm), (train, validation, future))
                open_rows, open_reviews = [], []
                for i, candidate in enumerate(bank):
                    name = candidate.name
                    initial = dict(r=trained[name]["r"], x=trained[name]["x"],
                        source_p=np.r_[trained[name]["source_p"],validated[name]["source_p"],fixed_future[name]["source_p"]],
                        source_q=np.r_[trained[name]["source_q"],validated[name]["source_q"],fixed_future[name]["source_q"]])
                    _, fitrow, review = _do_fit(candidate, joint, jm, initial, options, open_counter,
                        seed=record["seed"]+14000+i)
                    open_rows.append(fitrow);open_reviews.append(review)
                losses = {r["candidate"]:r["loss"] for r in open_rows}
                data["open_joint_theta"] = dict(fits=open_rows, independent_review=open_reviews,
                    all_complete=all(r["success"] for r in open_rows), losses=losses, all_M=_margin(losses,true_name),
                    original_fixed_all_M=data["original_fixed"]["all_M"],
                    source_initial="saved fixed-theta source states, same records and same train R/X warm start",
                    objective="training+validation+actual-future SSE; same measured P/Q inputs",
                    parameters_refitted_after_action=True, deployed_fixed_policy_result=False,
                    old_threshold_applied=False, global_MLE_certified=False,
                    all_candidate_loss_nonincrease_vs_fixed=all(losses[n]<=fixed_all[n]+1e-6 for n in losses))
            cache[key] = io.serial(data)
        data = cache[key]
        row.update(status="complete", cache_hit=sum(r.get("curve_key")==row["curve_key"] for r in method_rows)>0,
            oracle_heldout_M=data["oracle_true_theta"]["heldout_M"], original_fixed_all_M=data["original_fixed"]["all_M"],
            oracle_all_M=data["oracle_true_theta"]["all_M"],
            open_all_M=data.get("open_joint_theta",{}).get("all_M"),
            future_input_rms_relative_training=data["future_input_geometry"]["input_rms_relative_training"],
            future_offspace_rms_relative_training=data["future_input_geometry"]["offspace_rms_relative_training"])
        method_rows.append(row)
    result = dict(schema="fixed_train_post_selection_extrapolation_v1", seed=record["seed"], true_name=true_name,
        input_path=str(Path(path).resolve()), input_sha256=before, input_policy_id=record["policy_id"],
        geometry=geometry, oracle_training_validation_fits=oracle_fits, oracle_training_validation_reviews=oracle_reviews,
        methods=method_rows, curves={hashlib.sha256(bytes.fromhex(k)).hexdigest():v for k,v in cache.items()},
        all_methods_preserved=len(method_rows)==len(record["methods"]), unique_actual_curves=len(cache),
        costs=dict(geometry=geometry_counter, oracle=oracle_counter, open_joint=open_counter),
        seconds=perf_counter()-started, new_action_selection_calls=0, new_future_physical_executions=0,
        diagnostic_only=True, oracle_enters_selector=False, old_threshold_applied=False,
        source_interface=1, P_Q_inputs_fixed=True, global_MLE_certified=False,
        input_unchanged=before==_hash(path))
    if not result["input_unchanged"]:
        raise ValueError("source development trial changed while diagnosing")
    return io.serial(result)


def summarize(results):
    names = [r["name"] for r in results[0]["methods"]]
    rows = []
    for name in names:
        methods = [next(m for m in result["methods"] if m["name"]==name) for result in results]
        complete = [m for m in methods if m["status"]=="complete"]
        rows.append(dict(name=name, total_histories=len(methods), complete=len(complete), original_failures=len(methods)-len(complete),
            **{k:float(np.mean([m[k] for m in complete])) if complete and all(m[k] is not None for m in complete) else None
               for k in ("original_heldout_M","oracle_heldout_M","original_fixed_all_M","oracle_all_M","open_all_M")}))
    return dict(histories=len(results), independent_history_ids=[r["seed"] for r in results], methods=rows,
        unique_curves=sum(r["unique_actual_curves"] for r in results),
        original_failures_retained=sum(m["status"]!="complete" for r in results for m in r["methods"]),
        costs={kind:{k:sum(r["costs"][kind].get(k,0) for r in results) for k in ("fit_calls","solver_calls","ac_calls")}
               for kind in ("geometry","oracle","open_joint")},
        all_input_bytes_unchanged=all(r["input_unchanged"] for r in results),
        new_action_selection_calls=0, old_threshold_applied=False,
        scope="development-only post-selection parameter/oracle diagnostics; no policy performance or statistical risk recalibration")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--oracle-only",action="store_true")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=sorted(args.input.glob("trial_*.json.gz"))
    if not files:raise ValueError("no saved development trials")
    sources={k:_hash(HERE/k) for k in ("ev_fixed_train_extrapolation.py","ev_fixed_train_fit.py","ev_fixed_train_review.py")}
    protocol=dict(input_trial_sha256={p.name:_hash(p) for p in files},source_sha256=sources,
        ordering="A oracle true R/X held-out/source profile first; B open shared R/X joint all-record fit second",
        selection_calls=0, threshold_reuse=False, new_future_noise=False,
        truth_use="offline oracle A and parameter/input diagnostics only",
        options=dict(multistart=2,max_nfev=350,tol=1e-10),oracle_max_nfev=200,
        methods="every saved method, original failures retained, identical actual curves cached",
        geometry_99percent="descriptive centered training input energy threshold, not coverage",
        source_interface=1,PQ_measured_inputs_exact=True,whole_research_complete=False)
    protocol_path=args.output/"protocol.json"
    if protocol_path.exists():
        if io.read_json(protocol_path)!=protocol:raise ValueError("existing protocol differs")
    else:io.write_json(protocol_path,protocol)
    results=[]
    for path in files:
        target=args.output/(path.stem.removesuffix(".json")+"_diagnostic.json.gz")
        if target.exists():
            result=io.read_json(target)
            if result["input_sha256"]!=_hash(path):raise ValueError("cached input hash differs")
        else:
            result=diagnose_history(path,run_open=not args.oracle_only)
            io.write_json(target,result,compressed=True)
        results.append(result)
        print(f"diagnosed {path.name}: {result['unique_actual_curves']} actual curves",flush=True)
    final=summarize(results)
    final["source_unchanged"] = all(_hash(HERE/k)==v for k,v in sources.items())
    io.write_json(args.output/"summary.json",final)
    print(f"histories={len(results)} curves={final['unique_curves']} preserved failures={final['original_failures_retained']}",flush=True)


if __name__=="__main__":main()
