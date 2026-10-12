"""Ordered small AC audits of symmetric distance, power and topology EIG."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
from dataclasses import asdict
from time import perf_counter
import numpy as np
import scipy

from .audit_common import (build_bank,unvectorize,evaluation_atom,changed_prior,array_hash,atom_mean)
from .symmetric import symmetric_scores
from .power import (power_scores,calibrate_rank_nulls,candidate_pvalues,evaluation_summary,historical_losses)
from .bayes import posterior_weights,eig_scores
from .likelihood import concat_observations,fit_candidate
from .run_study import dump


HERE=Path(__file__).resolve().parent


def best_name(scores,names,key):
    if not names:
        raise ValueError("empty feasible design domain")
    return min(names,key=lambda name:(-float(scores[name][key]),name))


def fit_wider_family(bank,yh,yf,name,seed,max_nfev):
    """Observed-data continuous AC check, separate from finite-null inference."""
    obs=concat_observations(unvectorize(yh,bank.meter,bank.history_count),
                            unvectorize(yf,bank.meter,bank.future_count))
    loss_h=historical_losses(yh,bank.history_means)
    finite_loss=loss_h+np.sum((bank.future_means[name]-yf)**2,axis=1)
    results=[]
    for h,candidate in enumerate(bank.candidates):
        ids=np.flatnonzero(bank.labels==h)
        i=int(ids[np.argmin(finite_loss[ids])])
        atom=bank.atoms[i]
        p,q=atom["p"].copy(),atom["q"].copy()
        p[bank.history_count:]+=bank.actions[name].p
        q[bank.history_count:]+=bank.actions[name].q
        start=dict(r=atom["r"],x=atom["x"],p=p,q=q,source_p=atom["source_p"],source_q=atom["source_q"])
        fitted=fit_candidate(candidate,obs,bank.meter,initial=start,multistart=2,
                             max_nfev=max_nfev,seed=seed+h)
        row=fitted.summary()
        row.update(finite_best_loss=float(np.min(finite_loss[ids])),
                   initial_atom=i,source_p=fitted.source_p.tolist(),source_q=fitted.source_q.tolist(),
                   target="wider continuous observed-data AC model",finite_threshold_applied=False)
        initial_loss=float(fitted.diagnostics.get("initial_loss",np.nan))
        row["initial_loss_matches_finite_seed"]=bool(np.isclose(initial_loss,row["finite_best_loss"],rtol=2e-8,atol=2e-6))
        if not row["initial_loss_matches_finite_seed"]:
            raise RuntimeError("continuous initial loss does not match independently evaluated finite seed")
        if fitted.success and fitted.loss>initial_loss+2e-6:
            raise RuntimeError("successful continuous fit increased its feasible seed loss")
        results.append(row)
        print(f"continuous {seed} {name} {candidate.name}: success={fitted.success}, loss={fitted.loss:.5f}",flush=True)
    return results


def one_history(bank,seed,strength,topology,args,stress=None):
    began=perf_counter()
    # All uses of generator/noise below are in this evaluation driver.
    true_atom=evaluation_atom(bank,seed,strength,topology)
    generator=bank.atoms[true_atom]
    generated_future={name:means[true_atom] for name,means in bank.future_means.items()}
    generated_history=bank.history_means[true_atom]
    if stress=="off_grid":
        generator={**generator,"r":generator["r"]*1.08,"x":generator["x"]*.94,
                   "source_p":generator["source_p"]+.009,"source_q":generator["source_q"]-.006}
        generated_history=atom_mean(bank.candidates[topology],generator,bank.meter,slice(0,12))[0]
        generated_future={name:atom_mean(bank.candidates[topology],generator,bank.meter,slice(12,None),action)[0]
                          for name,action in bank.actions.items()}
    if stress=="bias_misspec":
        def shifted(mean,count):
            value=mean.copy()
            value[2*count*3:3*count*3]+=np.tile([-4.,0.,4.],count)
            return value
        generated_history=shifted(generated_history,12)
        generated_future={name:shifted(value,8) for name,value in generated_future.items()}
    yh=generated_history+np.random.default_rng(seed+100).normal(size=bank.history_means.shape[1])
    hm,fm,labels=bank.history_means,bank.future_means,bank.labels
    eligible=[n for n in bank.actions if n!="none"]
    constants=[n for n,a in bank.actions.items() if a.family=="constant"]
    # Direction 1 first.
    first=perf_counter()
    sym=symmetric_scores(yh,hm,fm,labels,cutoff=12.)
    sym_scores=sym["actions"]
    sym_key="distance_squared"
    symmetric=best_name(sym_scores,eligible,sym_key)
    constant_sym=best_name(sym_scores,constants,sym_key)
    support_sensitivity={str(c):symmetric_scores(yh,hm,fm,labels,cutoff=c) for c in (0.,4.,36.)}
    sym_seconds=perf_counter()-first
    print(f"history {seed} direction1: {symmetric}",flush=True)
    # Direction 2 only after the first criterion has completed.
    first=perf_counter()
    power=power_scores(yh,hm,fm,labels,cutoff=12.,alpha=.1,
        training_count=args.training,calibration_count=args.calibration,seed=seed+2000)
    powered=best_name(power,eligible,"score")
    constant_power=best_name(power,constants,"score")
    final_cal=calibrate_rank_nulls(yh,hm,fm,labels,B=args.calibration,seed=seed+3000)
    power_seconds=perf_counter()-first
    print(f"history {seed} direction2: {powered}",flush=True)
    # Direction 3: update a previously fixed proper prior, no hard cutoff.
    first=perf_counter()
    posterior=posterior_weights(yh,hm,labels,bank.prior)
    eig_runs={}
    for count in args.eig_counts:
        eig_runs[str(count)]=[eig_scores(posterior,fm,samples_per_topology=count,seed=seed+4000+j)
                              for j in range(args.integration_repeats)]
    eig=eig_runs[str(max(args.eig_counts))][0]
    informed=best_name(eig,eligible,"eig")
    constant_eig=best_name(eig,constants,"eig")
    sensitivity={}
    for tm,wm in ((.2,1/3),(.8,1/3),(.5,.1),(.5,.7)):
        pp=posterior_weights(yh,hm,labels,changed_prior(bank,tm,wm))
        ee=eig_scores(pp,fm,samples_per_topology=max(args.eig_counts),seed=seed+4000)
        sensitivity[f"topology0={tm},weak={wm}"]=dict(posterior=pp.summary(),
            selected=best_name(ee,eligible,"eig"),scores=ee)
    eig_seconds=perf_counter()-first
    print(f"history {seed} direction3: {informed}",flush=True)
    fixed="constant_site_0"
    pulse=next(n for n,a in bank.actions.items() if a.family=="fixed_pulse" and
               all(j in (-1,0) for j in a.sites))
    random_name=eligible[int(np.random.default_rng(seed+5000).integers(len(eligible)))]
    choices=dict(none="none",constant_fixed=fixed,fixed_pulse=pulse,random_action=random_name,
        symmetric=symmetric,power=powered,bayes=informed,
        best_constant_symmetric=constant_sym,best_constant_power=constant_power,best_constant_bayes=constant_eig)
    # One noise array for all controls. It was unavailable to all selectors.
    noise=np.random.default_rng(seed+6000).normal(size=(args.evaluation,fm["none"].shape[1]))
    methods={}
    for method,name in choices.items():
        ys=generated_future[name]+noise
        pv=candidate_pvalues(yh,hm,fm[name],labels,ys,final_cal)
        metrics=evaluation_summary(pv,topology,np.unique(labels),alpha=.1)
        metrics.update(action=name,paired_future_noise_hash=array_hash(noise),
                       finite_scope=stress is None,failure_count=0)
        methods[method]=metrics
    gains={}
    for method,score_map,key,selected,const in (
        ("symmetric",sym_scores,sym_key,symmetric,constant_sym),
        ("power",power,"score",powered,constant_power),("bayes",eig,"eig",informed,constant_eig)):
        gains[method]=dict(selected=selected,score=float(score_map[selected][key]),
            versus_none=float(score_map[selected][key]-score_map["none"][key]),
            versus_fixed_constant=float(score_map[selected][key]-score_map[fixed][key]),
            versus_best_constant=float(score_map[selected][key]-score_map[const][key]),
            best_constant=const)
    continuous={}
    if args.continuous:
        for name in sorted(set(choices.values())):
            continuous[name]=fit_wider_family(bank,yh,generated_future[name]+noise[0],name,seed+7000,args.max_nfev)
            for row in continuous[name]:
                if row["candidate"]==bank.candidates[topology].name and row["success"]:
                    atom=generator
                    estimated=np.r_[row["r"],row["x"]]; actual=np.r_[atom["r"],atom["x"]]
                    row["impedance_relative_rmse"]=float(np.linalg.norm(estimated-actual)/np.linalg.norm(actual))
                    row["source_rmse"]=float(np.sqrt(np.mean((np.asarray(row["source_p"])-atom["source_p"])**2+
                                                              (np.asarray(row["source_q"])-atom["source_q"])**2)))
    return dict(seed=seed,strength=strength,true_topology=topology,true_atom=true_atom,stress=stress,
        true_atom_covered=stress is None,evaluation_only_truth=True,
        history_hash=array_hash(yh),history_y=yh,choices=choices,symmetric=sym,
        support_sensitivity=support_sensitivity,power=power,eig_runs=eig_runs,
        prior_sensitivity=sensitivity,posterior=posterior.summary(),gains=gains,methods=methods,
        continuous=continuous,calibration_hash=array_hash(final_cal["values"]),
        calibration_seed=seed+3000,training_seed=seed+2000,integration_seed=seed+4000,
        evaluation_noise_seed=seed+6000,seconds=perf_counter()-began,
        selector_seconds=dict(symmetric=sym_seconds,power=power_seconds,bayes=eig_seconds)),noise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"outputs"/"three_directions")
    parser.add_argument("--histories-per-case",type=int,default=3)
    parser.add_argument("--training",type=int,default=64)
    parser.add_argument("--calibration",type=int,default=99)
    parser.add_argument("--evaluation",type=int,default=512)
    parser.add_argument("--eig-counts",type=int,nargs="+",default=[128,512,2048])
    parser.add_argument("--integration-repeats",type=int,default=3)
    parser.add_argument("--continuous",action="store_true")
    parser.add_argument("--max-nfev",type=int,default=150)
    parser.add_argument("--stress",action="store_true")
    args=parser.parse_args()
    if not 1<=args.histories_per_case<=3:
        parser.error("frozen first stage permits 1..3 histories per strong/weak case")
    args.output.mkdir(parents=True,exist_ok=True)
    bank=build_bank()
    dependencies=[HERE/n for n in ("audit_common.py","symmetric.py","power.py","bayes.py","run_three_directions.py",
                                  "likelihood.py","simulation.py","design.py","candidates.py")]
    dependencies += [HERE.parent/"mobile_ev_probing"/n for n in ("ac_general_v9.py","passive_bridge_v9.py")]
    config={**vars(args),"output":str(args.output),"alpha":.1,"scope":"finite AC composite audit plus optional wider-model check",
        "prior_external":True,"continuous_global_mle":False,"python":platform.python_version(),
        "numpy":np.__version__,"scipy":scipy.__version__,"meter":asdict(bank.meter),
        "bank_mean_hash":array_hash(bank.history_means,*bank.future_means.values()),
        "source_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}}
    dump(args.output/"config.json",config)
    dump(args.output/"bank_diagnostics.json",bank.diagnostics)
    records=[]
    for strength,base in ((2,66001),(0,67001)):
        for i in range(args.histories_per_case):
            record,noise=one_history(bank,base+i,strength,i%2,args)
            dump(args.output/f"history_{base+i}.json",record)
            np.savez_compressed(args.output/f"history_{base+i}_observations.npz",history_y=record["history_y"],future_noise=noise)
            records.append(record)
    if args.stress:
        for seed,stress in ((68001,"off_grid"),(68002,"bias_misspec")):
            record,noise=one_history(bank,seed,2,0,args,stress=stress)
            dump(args.output/f"history_{seed}.json",record)
            np.savez_compressed(args.output/f"history_{seed}_observations.npz",history_y=record["history_y"],future_noise=noise)
            records.append(record)
    summary=dict(config=config,bank=bank.diagnostics,history_count=len(records),
        records=[{k:r[k] for k in ("seed","strength","true_topology","true_atom","true_atom_covered","stress","choices","gains","methods", "posterior","selector_seconds")}
                 for r in records],
        total_seconds=float(sum(r["seconds"] for r in records)),
        continuous_fit_count=sum(len(rows) for r in records for rows in r["continuous"].values()),
        continuous_failure_count=sum(not row["success"] for r in records for rows in r["continuous"].values() for row in rows),
        prior_scope="two topologies, 24 atoms each; not continuous-domain posterior/calibration",
        expansion_status="not justified by first-stage finite-family results alone")
    dump(args.output/"summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("history_count","total_seconds","continuous_fit_count","continuous_failure_count")}),flush=True)


if __name__=="__main__":
    main()
