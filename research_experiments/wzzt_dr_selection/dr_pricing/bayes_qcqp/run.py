"""Plan, freeze and independently sample a two-candidate quadratic design.

Inputs are the existing portable case and an explicit Bayesian settings JSON.
No historical fitter, physical future, true topology or AC simulator is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from scipy.special import expit, logit

if __package__:
    from ..problem import load_problem
    from ..solve import reference, write_arrays, write_json
    from .core import build_expected, build_separation, posterior_pair, separation_threshold, event_probability
    from .solver import solve
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from problem import load_problem
    from solve import reference, write_arrays, write_json
    from bayes_qcqp.core import build_expected, build_separation, posterior_pair, separation_threshold, event_probability
    from bayes_qcqp.solver import solve


def load_inputs(case, settings_path):
    problem = load_problem(case)
    settings_path = Path(settings_path).resolve()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if settings.get("likelihood") != "fixed_common_diagonal_gaussian":
        raise ValueError("Only an explicitly fixed common diagonal Gaussian likelihood is implemented")
    if not .5 < settings["posterior_threshold"] < 1:
        raise ValueError("posterior_threshold must lie in (0.5,1)")
    prior = np.asarray(settings["base_prior"], float)
    if (prior.shape != (2,) or not np.isfinite(prior).all() or np.any(prior<=0)
            or not np.isclose(prior.sum(),1.,atol=1e-12)):
        raise ValueError("Declare two positive normalized priors in C,F order")
    path = settings_path.parent / settings["arrays_file"]
    with np.load(path,allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    sd = np.asarray(arrays["likelihood_sd"], float)
    if sd.shape != problem.sigma.shape or np.any(sd<=0) or not np.isfinite(sd).all():
        raise ValueError("Likelihood standard deviations must be positive and match all observations")
    history = float(settings["history_log_bayes_factor"])
    if not np.isfinite(history):
        raise ValueError("Historical Gaussian evidence must be finite")
    if not isinstance(settings["history_rows"],int) or settings["history_rows"]<0:
        raise ValueError("Historical row count must be a nonnegative integer")
    if settings["history_rows"]:
        receipt_reference = settings["history_receipt"]
        receipt_path = Path(receipt_reference["path"])
        if not receipt_path.is_absolute(): receipt_path = settings_path.parent/receipt_path
        if reference(receipt_path)["sha256"] != receipt_reference["sha256"]:
            raise ValueError("Historical evidence receipt changed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt["history_rows"] != settings["history_rows"]
                or receipt["method"] != "common_diagonal_gaussian_squared_residuals"
                or not np.isclose(receipt["log_bayes_factor"],history,atol=1e-10,rtol=0)):
            raise ValueError("Historical evidence differs from the declared Gaussian receipt")
    elif history != 0:
        raise ValueError("Zero historical rows require zero historical evidence")
    expected = build_expected(problem.base["C"],problem.base["F"],problem.effect["C"],problem.effect["F"],sd,
        response_relative_sd=arrays.get("response_relative_sd",0.),
        background_residual_variance_C=arrays.get("background_residual_variance_C",0.),
        background_residual_variance_F=arrays.get("background_residual_variance_F",0.))
    separation = build_separation(problem.base["C"],problem.base["F"],problem.effect["C"],problem.effect["F"],sd)
    return problem,settings,arrays,expected,separation,float(np.log(prior[1]/prior[0])),path


def scalar_summary(result):
    return {key:value for key,value in result.items() if key not in ("x","mean_u","unit_reward")}


def _plan(case, settings_path, mode, out):
    """Run the specified mathematical model; all scientific choices are inputs."""
    out = Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/"request.json").exists():
        raise ValueError("A previous attempt is preserved; use a new output directory")
    problem,settings,arrays,expected,separation,prior,array_path = load_inputs(case,settings_path)
    initial_odds = prior + float(settings["history_log_bayes_factor"])
    eta = float(settings["posterior_threshold"])
    beta = float(settings["event_probability_target"])
    threshold_odds = float(logit(eta))
    source = Path(__file__).resolve().parent
    files = [source/name for name in ("core.py","solver.py","run.py","PROTOCOL.json")]
    files += [source.parent/"problem.py"]
    request = dict(mode=mode,settings=settings,case=reference(problem.case_path),
        case_arrays=reference(problem.arrays_path),settings_input=reference(settings_path),
        bayes_arrays=reference(array_path),sources={str(p):reference(p)["sha256"] for p in files},
        python=platform.python_version(),no_actual_future_or_truth_access=True,
        history_count=settings["history_rows"],historical_log_bayes_factor=settings["history_log_bayes_factor"],
        new_day_weight=1,legacy_new_day_weight=problem.new_weight,
        prior_log_odds=prior,initial_log_odds=initial_odds)
    write_json(out/"request.json",request)
    with zipfile.ZipFile(out/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as z:
        for index,p in enumerate(files): z.write(p,f"{index:02d}_{p.name}")
    write_arrays(out/"quadratics.npz",expected_Q=expected.Q,expected_linear=expected.linear,
        expected_constant=expected.constant,separation_Q=separation.Q,
        separation_linear=separation.linear,separation_constant=separation.constant)
    controls = settings["solver"]
    started = time.perf_counter()
    results = {}
    if mode == "expected_either":
        zero_odds = initial_odds + expected.value(problem.device.nominal(np.zeros(problem.shape)))
        branches = ["F"] if zero_odds>=threshold_odds else ["C"] if zero_odds<=-threshold_odds else ["F","C"]
        jobs = [(state,expected.scaled(1. if state=="F" else -1.),
                 threshold_odds-initial_odds if state=="F" else threshold_odds+initial_odds)
                for state in branches]
    elif mode == "pair_event_nominal":
        jobs = [("both",separation,separation_threshold(eta,beta,initial_odds))]
    else:
        raise ValueError("Unknown mode")
    for state,evidence,threshold in jobs:
        results[state] = solve(problem,evidence,threshold,out/state,**controls)
    valid = [(key,row) for key,row in results.items() if row["qualified"]]
    best = min(valid,key=lambda pair:pair[1]["objective"]) if valid else None
    bounds=[]
    for row in results.values():
        if row["solver_status"]==3:
            continue
        bounds.append(max(0.,row["objbound"]) if row["objbound"] is not None else 0.)
    lower = min(bounds) if bounds else None
    summary = dict(mode=mode,qualified=bool(best),posterior_threshold=eta,
        event_probability_target=beta if mode=="pair_event_nominal" else None,
        history_log_bayes_factor=settings["history_log_bayes_factor"],initial_log_odds=initial_odds,
        branch_results={key:scalar_summary(row) for key,row in results.items()},
        lower_bound=lower,seconds=time.perf_counter()-started,
        expectation_is_not_probability=True,
        scope=("absolute expected log posterior odds under the historical reference, including declared Gaussian response moments"
               if mode=="expected_either" else
               "nominal deterministic design; correct-candidate confidence under either of two closed Gaussian hypotheses"))
    if best:
        key,row=best
        u=row["mean_u"]
        expected_odds=initial_odds+expected.value(u)
        S=max(0.,separation.value(u))
        upper=float(row["objective"])
        summary.update(selected_branch=key,fee=upper,upper_bound=upper,
            cost_gap=None if lower is None else max(0.,upper-lower)/max(upper,1e-12),
            analytic_zero_cost_optimum=bool(upper==0 and row["solver_status"]=="CERTIFIED_COST_ZERO"),
            expected_log_odds=expected_odds,
            posterior_at_expected_log_odds=float(expit(abs(expected_odds))),
            posterior_at_expected_log_odds_is_not_expected_posterior=True,
            nominal_separation=S,
            ideal_pair_probability_F=event_probability(S,eta,initial_odds,"F"),
            ideal_pair_probability_C=event_probability(S,eta,initial_odds,"C"),
            nominal_device=problem.audit_delivery(u))
        write_arrays(out/"quote.npz",x=row["x"],mean_u=u,unit_reward=row["unit_reward"])
        write_json(out/"policy_freeze.json",dict(status="frozen",quote=reference(out/"quote.npz"),
            request=reference(out/"request.json"),selected_branch=key,
            no_actual_future_or_truth_access=True))
    write_json(out/"summary.json",summary)
    write_json(out/"complete.json",dict(status="complete",qualified=bool(best),summary=reference(out/"summary.json")))
    return summary


def _sample(case,settings_path,quote,out,*,count=32768,seed=2026111401):
    """Post-freeze independent Gaussian diagnostics, never used by the solver."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/"request.json").exists(): raise ValueError("Existing sampling attempt preserved")
    problem,settings,arrays,expected,separation,prior,array_path=load_inputs(case,settings_path)
    if not isinstance(count,int) or count<=1:
        raise ValueError("Independent diagnostics require count > 1")
    freeze_path=Path(quote).resolve().parent/"policy_freeze.json"
    if not freeze_path.exists(): raise ValueError("Sampling requires a frozen planning quote")
    freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status")!="frozen" or freeze["quote"]!=reference(quote):
        raise ValueError("Frozen quote reference changed")
    request_path=Path(freeze["request"]["path"])
    if reference(request_path)!=freeze["request"]:
        raise ValueError("Planning request changed after freeze")
    frozen_inputs=json.loads(request_path.read_text(encoding="utf-8"))
    for path,digest in frozen_inputs["sources"].items():
        if reference(path)["sha256"]!=digest:
            raise ValueError("Source changed after planning freeze: "+path)
    for key,path in (("case",problem.case_path),("case_arrays",problem.arrays_path),
                     ("settings_input",settings_path),("bayes_arrays",array_path)):
        if frozen_inputs[key]!=reference(path):
            raise ValueError("Sampling input differs from frozen planning input: "+key)
    with np.load(quote,allow_pickle=False) as z:
        x,u=z["x"].copy(),z["mean_u"].copy()
    replay=problem.device.nominal(x)
    if not np.allclose(replay,u,atol=1e-10,rtol=0): raise ValueError("Frozen quote does not replay")
    eta=float(settings["posterior_threshold"]);cut=float(logit(eta))
    history=float(settings["history_log_bayes_factor"]);initial=prior+history
    rho=np.broadcast_to(arrays.get("response_relative_sd",0.),u.shape)
    sd=arrays["likelihood_sd"]
    if np.any(arrays.get("background_residual_variance_C",0.)) or np.any(arrays.get("background_residual_variance_F",0.)):
        raise ValueError("Sampling needs joint background factors, not only marginal variance inputs")
    write_json(out/"request.json",dict(quote=reference(quote),case=reference(problem.case_path),
        settings=reference(settings_path),arrays=reference(array_path),count=count,seed=seed,
        scope="unclipped Gaussian response and shared residual voltage; no actual AC or topology truth"))
    rng=np.random.default_rng(seed);odds=[];f_success=c_success=either=0
    for first in range(0,count,256):
        batch=min(256,count-first)
        delivered=u[None]+rho[None]*u[None]*rng.standard_normal((batch,*u.shape))
        voltage=sd[None]*rng.standard_normal((batch,*sd.shape))
        rc=problem.base["C"][None]+np.einsum("nr,brt->btn",problem.effect["C"],delivered)+voltage
        rf=problem.base["F"][None]+np.einsum("nr,brt->btn",problem.effect["F"],delivered)+voltage
        result=posterior_pair(rc,rf,sd,prior,history)
        value=np.asarray(result["log_odds"])
        odds.append(value);f_success+=int(np.sum(value>=cut));c_success+=int(np.sum(value<=-cut))
        either+=int(np.sum(np.abs(value)>=cut))
    values=np.concatenate(odds);analytic=initial+expected.value(u)
    sem=float(np.std(values,ddof=1)/np.sqrt(count))
    S=max(0.,separation.value(u));noise=rng.standard_normal(count)
    ideal_F=initial+.5*S+np.sqrt(S)*noise
    ideal_C=initial-.5*S+np.sqrt(S)*noise
    summary=dict(count=count,seed=seed,reference_F_threshold_successes=f_success,
        reference_C_threshold_successes=c_success,reference_either_successes=either,
        reference_either_fraction=either/count,reference_mean_log_odds=float(values.mean()),
        analytic_expected_log_odds=analytic,mean_error=float(values.mean()-analytic),mean_standard_error=sem,
        ideal_F_correct_confidence_successes=int(np.sum(ideal_F>=cut)),
        ideal_C_correct_confidence_successes=int(np.sum(ideal_C<=-cut)),
        ideal_F_analytic_probability=event_probability(S,eta,initial,"F"),
        ideal_C_analytic_probability=event_probability(S,eta,initial,"C"),
        no_model_or_quote_changes=True)
    np.savez_compressed(out/"log_odds_private.npz",reference=values,ideal_F=ideal_F,ideal_C=ideal_C)
    write_json(out/"summary.json",summary);write_json(out/"complete.json",dict(status="complete"))
    return summary


def _guard(out,operation):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    lock=out/"running.lock"
    with lock.open("x",encoding="utf-8") as stream: stream.write(str(os.getpid()))
    try:
        return operation()
    except Exception as exc:
        if not (out/"complete.json").exists():
            write_json(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


def plan(case,settings_path,mode,out):
    return _guard(out,lambda:_plan(case,settings_path,mode,out))


def sample(case,settings_path,quote,out,*,count=32768,seed=2026111401):
    return _guard(out,lambda:_sample(case,settings_path,quote,out,count=count,seed=seed))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["plan","sample"])
    parser.add_argument("--case",type=Path,required=True)
    parser.add_argument("--settings",type=Path,required=True)
    parser.add_argument("--mode",choices=["expected_either","pair_event_nominal"],default="expected_either")
    parser.add_argument("--quote",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=plan(args.case,args.settings,args.mode,args.out) if args.action=="plan" else sample(
        args.case,args.settings,args.quote,args.out)
    print(json.dumps(result,ensure_ascii=False,default=lambda v:v.tolist(),allow_nan=False))
