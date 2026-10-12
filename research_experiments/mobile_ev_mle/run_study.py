"""Reproducible synthetic local-AC likelihood/EV study (no oracle selector)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from dataclasses import asdict
import numpy as np
import scipy
from scipy.stats import beta
from .candidates import propose, normalize_family
from .likelihood import fit_candidate, concat_observations
from .design import EVAction, EVConstraints, enumerate_actions, check_action, select_action
from .simulation import generate, observe, model, truth_metadata, BASE_KW, ZBASE_OHM

HERE = Path(__file__).resolve().parent
METHODS = ("none", "constant_fixed", "fixed_pulse", "random_action", "designed")


def encode(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def finite_json(value):
    if isinstance(value,np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value,np.generic):
        return finite_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    if isinstance(value,dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_json(v) for v in value]
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(value), ensure_ascii=False, indent=2,
        default=encode, allow_nan=False)+"\n", encoding="utf-8")


def fingerprint(*arrays):
    d = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        d.update(str(a.shape).encode())
        d.update(a.astype("<f8").tobytes())
    return d.hexdigest()


def fit_record(candidate, fit):
    return dict(name=candidate.name, supports=list(map(list,candidate.supports)),
        success=bool(fit.success), loss=float(fit.loss), loglik=float(fit.loglik),
        r_pu=fit.r, x_pu=fit.x, r_ohm=fit.r*ZBASE_OHM, x_ohm=fit.x*ZBASE_OHM,
        diagnostics=fit.diagnostics, global_optimum_certified=False)


def fit_bank(candidates, obs, meter, *, initial=None, seed=0,
             multistart=2, max_nfev=120):
    fitted, failures = {}, {}
    for i, c in enumerate(candidates):
        try:
            f = fit_candidate(c, obs, meter, initial=None if initial is None else initial.get(c.name),
                seed=seed+i, multistart=multistart, max_nfev=max_nfev)
            fitted[c.name] = f
            if not f.success:
                failures[c.name] = "optimizer_nonconverged"
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
            failures[c.name] = type(e).__name__+":"+str(e)
    return fitted, failures


def one_trial(seed, case, *, n=3, run_wzzt=True, multistart=2,
              max_nfev=120, output=None):
    start = perf_counter()
    path = generate(seed, case, n=n,
        truth_family=((0,1),) if case=="missing_truth" else None)
    meter = model(n)
    history, hp = observe(path, slice(0,path.history_count))
    candidates, proposal = propose(history, run_wzzt=run_wzzt)
    if case == "missing_truth":
        # Predeclared candidate-library defect, NOT a selector truth argument.
        candidates = [c for c in candidates if normalize_family(c.supports,n) != ((0,1),)]
    histfits, histfail = fit_bank(candidates,history,meter,seed=seed+300,
        multistart=multistart,max_nfev=max_nfev)
    availability = np.ones((8,n),bool)
    availability[:2,-1] = False
    constraints = EVConstraints(p_max=.3,energy=.3,base_kw=BASE_KW,
        availability=availability,travel_slots=np.ones((n,n),int)-np.eye(n,dtype=int),
        travel_energy=.01*(np.ones((n,n))-np.eye(n)),soc_min=.1,initial_site=0)
    actions, rejected = enumerate_actions(8,n,constraints,max_actions=11)
    design = select_action(candidates,history,histfits,meter,actions,
        constraints=constraints,multistart=1,max_nfev=max_nfev,seed=seed+700)
    zero = EVAction("none",np.zeros((8,n)),np.zeros((8,n)),
                    tuple([-1]*8),"none")
    fixed = next(a for a in actions if a.family=="constant" and np.all(np.asarray(a.sites)==0))
    pulse = next(a for a in actions if a.family=="fixed_pulse" and
                 all(j in (-1,0) for j in a.sites))
    random = actions[np.random.default_rng(seed+90000).integers(len(actions))]
    selected = next((a for a in actions if a.name==design["selected"]),None)
    choices = dict(none=zero,constant_fixed=fixed,fixed_pulse=pulse,
                   random_action=random,designed=selected)
    truth_family = normalize_family(path.candidate.supports,n)
    true_name = next((c.name for c in candidates if normalize_family(c.supports,n)==truth_family),None)
    records = []
    historical_rows = [fit_record(c,histfits[c.name]) for c in candidates if c.name in histfits]
    actual_npz = dict(history_p=history.p,history_q=history.q,history_v=history.v,
                      history_master_p=history.master_p,history_master_q=history.master_q)
    for method, a in choices.items():
        if a is None:
            records.append(dict(method=method,status="selection_failure",true_covered=true_name is not None,
                seed=seed,case=case))
            continue
        method_start=perf_counter()
        future, pred=observe(path,slice(path.history_count,None),a)
        observed=concat_observations(history,future)
        fits, failed=fit_bank(candidates,observed,meter,initial=histfits,seed=seed+1000,
                             multistart=multistart,max_nfev=max_nfev)
        valid={name:f for name,f in fits.items() if f.success}
        best=min(valid,key=lambda name:(valid[name].loss,name)) if valid else None
        gaps={name:float(f.loss-valid[best].loss) for name,f in valid.items()} if best else {}
        row=dict(seed=seed,case=case,n=n,method=method,status="complete" if len(valid)==len(candidates) else "optimization_failure",
            action=a.name,action_family=a.family,action_feasibility=check_action(a,constraints) if method!="none" else
                dict(feasible=True,energy_kwh=0.,equal_energy_control=False),
            fits=[fit_record(c,fits[c.name]) for c in candidates if c.name in fits],
            fit_failures=failed,best=best,deviance_gaps=gaps,
            true_name=true_name,true_covered=true_name is not None,
            true_deviance=gaps.get(true_name),true_loss=float(valid[true_name].loss) if true_name in valid else None,
            best_loss=float(valid[best].loss) if best else None,
            wrong_winner=best!=true_name if true_name is not None and best else None,
            voltage_min=float(pred["v"].min()),voltage_max=float(pred["v"].max()),
            operational_voltage_ok=bool(pred["v"].min()>=.9 and pred["v"].max()<=1.1),
            observation_hash=fingerprint(observed.p,observed.q,observed.v,observed.master_p,observed.master_q),
            seconds=perf_counter()-method_start)
        if true_name in fits:
            f=fits[true_name]
            row["impedance_relative_rmse"]=float(np.sqrt(np.mean(
                np.r_[((f.r-path.r)/path.r)**2,((f.x-path.x)/path.x)**2])))
            row["impedance_absolute_rmse_pu"]=float(np.sqrt(np.mean(
                np.r_[(f.r-path.r)**2,(f.x-path.x)**2])))
            row["source_rmse_pu"]=float(np.sqrt(np.mean(np.r_[
                (f.source_p-path.source_p)**2,(f.source_q-path.source_q)**2])))
        records.append(row)
        actual_npz.update({method+"_p":future.p,method+"_q":future.q,
                           method+"_v":future.v,method+"_master_p":future.master_p,
                           method+"_master_q":future.master_q})
    paired_complete = all(r["status"]=="complete" for r in records)
    for row in records:
        row["paired_trial_complete"] = paired_complete
    result=dict(seed=seed,case=case,n=n,proposal=proposal,historical_fits=historical_rows,
        historical_failures=histfail,design=design,library_rejections=rejected,
        records=records,truth=truth_metadata(path),seconds=perf_counter()-start,
        selection_uses_truth=False,selection_uses_future_noise=False)
    if output is not None:
        dump(Path(output)/f"{case}_{seed}.json",result)
        np.savez_compressed(Path(output)/f"{case}_{seed}_observations.npz",**actual_npz)
    return result


def threshold(values,alpha):
    values=sorted(float(v) for v in values)
    if not values:
        return None
    k=int(np.ceil((len(values)+1)*(1-alpha)))
    return values[k-1] if k<=len(values) else None


def calibrate(trials,alpha):
    by_case={}
    for case in sorted({t["case"] for t in trials}):
        subset=[t for t in trials if t["case"]==case]
        ds,ls=[],[]
        for t in subset:
            rows=t["records"]
            if len(rows)==len(METHODS) and all(r.get("status")=="complete" and
                r.get("true_deviance") is not None for r in rows):
                ds.append(max(r["true_deviance"] for r in rows))
                ls.append(max(r["true_loss"] for r in rows))
        by_case[case]=dict(deviance_threshold=threshold(ds,alpha),
            adequacy_loss_threshold=threshold(ls,alpha),valid_count=len(ds),
            failed_count=len(subset)-len(ds),max_across_methods=True,
            calibration_truth_deviances=ds,calibration_truth_losses=ls,
            alpha=alpha,scope="simulation mixture; conditional calibrated meters and declared optimizer; not topology probability")
    return by_case


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.] if n else None


def summarize(trials,thresholds):
    out=[]
    for case in sorted({t["case"] for t in trials}):
        tcase=thresholds.get(case,thresholds.get("strong",{}))
        dcrit,lcrit=tcase.get("deviance_threshold"),tcase.get("adequacy_loss_threshold")
        for method in METHODS:
            rows=[r for t in trials if t["case"]==case for r in t["records"] if r["method"]==method]
            usable=[r for r in rows if r["status"]=="complete" and r.get("paired_trial_complete",True) and dcrit is not None]
            covered=[r for r in usable if r["true_covered"]]
            false=sum(r["true_deviance"]>dcrit for r in covered)
            complete_elimination=0;wrong_removed=0;wrong_total=0;unresolved=0;empty=0;inadequate=0
            for r in usable:
                retained=[name for name,gap in r["deviance_gaps"].items() if gap<=dcrit]
                r["retained_candidates"]=retained
                r["simulation_screen_deviance_threshold"]=dcrit
                r["model_inadequacy"]=bool(lcrit is not None and r["best_loss"]>lcrit)
                inadequate+=r["model_inadequacy"]
                if r["true_covered"]:
                    wrong=[name for name in r["deviance_gaps"] if name!=r["true_name"]]
                    wrong_total+=len(wrong)
                    wrong_removed+=sum(name not in retained for name in wrong)
                    complete_elimination+=bool(retained==[r["true_name"]])
                    unresolved+=len(retained)>1
                empty+=not retained
            rmse=[r["impedance_absolute_rmse_pu"] for r in usable if "impedance_absolute_rmse_pu" in r]
            out.append(dict(case=case,method=method,runs=len(rows),valid_count=len(usable),
                computational_failures=sum(r["status"]!="complete" for r in rows),
                paired_trial_failures=sum(not r.get("paired_trial_complete",True) for r in rows),
                true_coverage=sum(r.get("true_covered",False) for r in rows),
                false_rejections=false,false_rejection_denominator=len(covered),
                false_rejection_95pct_interval=interval(false,len(covered)),
                all_wrong_removed=complete_elimination,wrong_candidates_removed=wrong_removed,
                wrong_candidates_total=wrong_total,unresolved=unresolved,empty_retained_set=empty,
                absolute_model_inadequacy=inadequate,
                wrong_winners=sum(bool(r.get("wrong_winner")) for r in covered),
                median_impedance_absolute_rmse_pu=float(np.median(rmse)) if rmse else None,
                mean_trial_method_seconds=float(np.mean([r.get("seconds",0) for r in rows])),
                operational_voltage_failures=sum(not r.get("operational_voltage_ok",True) for r in rows)))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("pilot","formal","stress","scale"),default="pilot")
    parser.add_argument("--output",default=str(HERE/"outputs"/"pilot"))
    parser.add_argument("--calibration-count",type=int,default=20)
    parser.add_argument("--evaluation-count",type=int,default=20)
    parser.add_argument("--max-nfev",type=int,default=120)
    parser.add_argument("--multistart",type=int,default=2)
    parser.add_argument("--skip-wzzt",action="store_true")
    parser.add_argument("--thresholds")
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy()
    config.update(alpha=.1,methods=METHODS,history_count=12,future_count=8,
        base_kw=BASE_KW,source_point="same common junction node1",r_and_x_independent=True,
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        interpretation="local attained loglikelihoods; no global MLE/topology probability")
    def run(seeds,cases,n=3):
        output_trials=[]
        for case in cases:
            for seed in seeds:
                t=one_trial(seed,case,n=n,run_wzzt=not args.skip_wzzt,
                    multistart=args.multistart,max_nfev=args.max_nfev,output=output)
                output_trials.append(t)
                print(f"{case} {seed}: {t['seconds']:.2f}s design={t['design']['status']} "+
                      f"covered={t['records'][0].get('true_covered')} ",flush=True)
        return output_trials
    if args.phase=="pilot":
        config["seeds"]=[31001]
        dump(output/"config.json",config)
        trials=run([31001],["strong","weak"])
        dump(output/"results.json",trials)
    elif args.phase=="formal":
        config["calibration_seeds"]=list(range(41000,41000+args.calibration_count))
        config["evaluation_seeds"]=list(range(51000,51000+args.evaluation_count))
        dump(output/"config.json",config)
        calibration=run(config["calibration_seeds"],["strong","weak"])
        thresholds=calibrate(calibration,.1);dump(output/"thresholds.json",thresholds)
        evaluation=run(config["evaluation_seeds"],["strong","weak"])
        summary=summarize(evaluation,thresholds)
        dump(output/"summary.json",dict(config=config,thresholds=thresholds,summary=summary,
            calibration_total_seconds=sum(t["seconds"] for t in calibration),
            evaluation_total_seconds=sum(t["seconds"] for t in evaluation),
            design_mean_seconds=float(np.mean([t["design"]["elapsed_seconds"] for t in evaluation]))))
        dump(output/"results.json",dict(calibration=calibration,evaluation=evaluation))
    else:
        seeds=list(range(61000,61004)) if args.phase=="stress" else [71001]
        config["seeds"]=seeds
        dump(output/"config.json",config)
        trials=run(seeds,["missing_truth","bias_misspec"] if args.phase=="stress" else ["strong"],n=3 if args.phase=="stress" else 5)
        thresholds=json.loads(Path(args.thresholds).read_text(encoding="utf-8")) if args.thresholds else {}
        dump(output/"summary.json",dict(config=config,summary=summarize(trials,thresholds)))
        dump(output/"results.json",trials)


if __name__=="__main__":
    main()




