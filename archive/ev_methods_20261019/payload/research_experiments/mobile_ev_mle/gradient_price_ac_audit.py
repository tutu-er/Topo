"""Independent AC path and joint-profile derivative audit, no policy tuning."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
from .simulation import generate, observe, model
from .candidates import propose
from .likelihood import fit_candidate
from .gradient_price import GradientPriceEvaluator
from .gradient_price_search import SearchSettings, optimize_price_curves, search_record
from .single_site_milp import SingleSiteConstraints
from .run_study import dump


def audit(seed=810001, case="strong"):
    start=perf_counter()
    path=generate(seed,case,history_count=8,future_count=4,truth_family=((0,1),))
    history,_=observe(path,slice(0,8))
    meter=model()
    candidates,_=propose(history,run_wzzt=False)
    fitted={c.name:fit_candidate(c,history,meter,multistart=2,max_nfev=250,
        tol=1e-11,linear_solver="exact",seed=seed+index) for index,c in enumerate(candidates)}
    if not all(f.success for f in fitted.values()):
        return dict(seed=seed,case=case,success=False,kind="historical_fit",
                    fits=[f.summary() for f in fitted.values()],seconds=perf_counter()-start)
    ranked=sorted(candidates,key=lambda c:(fitted[c.name].loss,c.name))
    selected=ranked[:2]
    options=dict(multistart=2,max_nfev=250,tol=1e-11,linear_solver="exact")
    evaluator=GradientPriceEvaluator(history,selected,fitted,meter,horizon=4,
                                    fit_options=options,seed=seed+100)
    checks=[]
    for site in range(3):
        u=np.array([0.,5.,4.,3.])
        direction=np.array([0.,-1.,1.,0.])
        baseline=evaluator.evaluate(site,u)
        if not baseline["success"]:
            checks.append(dict(site=site,kind="baseline_fit",success=False,
                               failures=baseline["failures"]));continue
        estimates=[]
        for step in (.02,.005):
            plus=evaluator.evaluate(site,u+step*direction)
            minus=evaluator.evaluate(site,u-step*direction)
            bank_plus={p["pair_id"]:p for p in plus["pairs"]}
            bank_minus={p["pair_id"]:p for p in minus["pairs"]}
            for pair in baseline["pairs"]:
                identifier=pair["pair_id"]
                valid=plus["success"] and minus["success"]
                analytical=float(pair["gradient_kw"]@direction)
                finite=(bank_plus[identifier]["score"]-bank_minus[identifier]["score"])/(2*step) if valid else None
                error=None if finite is None else abs(analytical-finite)
                # Complete refits are numerical and local; preserve the errors
                # and restart diagnostics as well as the declared tolerance.
                passed=valid and error <= 5e-4+5e-3*abs(analytical)
                estimates.append(dict(pair=identifier,step_kw=step,analytical=analytical,
                    finite_difference=finite,absolute_error=error,success=bool(passed),
                    pq_only=float(pair["pq_only_gradient_kw"]@direction),
                    voltage=float(pair["gradient_voltage"]@direction),
                    master=float(pair["gradient_master"]@direction),
                    plus_failures=plus["failures"],minus_failures=minus["failures"],
                    plus_fit_records=plus["fit_records"],minus_fit_records=minus["fit_records"]))
        for reference in selected:
            response=evaluator.future_response(reference.name,site,u)
            raw_checks=[]
            h=1e-4
            p=evaluator.future_response(reference.name,site,u+h*direction)["observations"]
            m=evaluator.future_response(reference.name,site,u-h*direction)["observations"]
            blocks=[]
            for channel in evaluator.channels:
                raw=(getattr(p,channel)-getattr(m,channel))/(2*h)
                sigma=np.broadcast_to(getattr(evaluator.future_model,channel+"_sigma"),raw.shape)
                blocks.append((raw/sigma).reshape(4,-1))
            finite=np.concatenate(blocks,axis=1).ravel()
            analytic=response["jacobian"]@direction
            relative=np.linalg.norm(finite-analytic)/max(1.,np.linalg.norm(analytic))
            raw_checks.append(dict(reference=reference.name,relative_error=float(relative),success=bool(relative<1e-6)))
            checks.append(dict(site=site,kind="fixed_AC_response",checks=raw_checks,
                               success=all(c["success"] for c in raw_checks)))
        checks.append(dict(site=site,kind="reoptimized_profile",checks=estimates,
                           success=all(c["success"] for c in estimates)))
    # One actual search closes the physics -> optimization -> refit loop.
    constraints=SingleSiteConstraints(energy=.15,initial_site=0,
        travel_slots=np.ones((3,3),int)-np.eye(3,dtype=int),
        travel_energy=.01*(np.ones((3,3))-np.eye(3)))
    search=optimize_price_curves(evaluator,constraints,sites=[0,1],
        settings=SearchSettings(max_iterations=2,max_starts=3))
    return dict(seed=seed,case=case,success=bool(all(c["success"] for c in checks) and search["success"]),
        reference_candidates=[c.name for c in selected],checks=checks,search=search_record(search),
        fit_calls=evaluator.fit_calls,total_ac_calls=evaluator.ac_evaluations+evaluator.response_ac_calls,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=perf_counter()-start,
        global_design_optimality_certified=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="research_experiments/mobile_ev_mle/evidence/gradient_price/ac_audit.json")
    args=parser.parse_args()
    names=("gradient_price_ac_audit.py","gradient_price.py","gradient_price_search.py",
           "single_site_milp.py","likelihood.py","simulation.py","candidates.py")
    here=Path(__file__).resolve().parent
    hashes={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names}
    records=[audit(810001,"strong"),audit(810002,"weak")]
    result=dict(success=all(r["success"] for r in records),independent_histories=2,records=records,
                scope="finite_difference_local_AC_profile_and_feasible_search",source_sha256=hashes,
                config=dict(history_points=8,future_points=4,steps_kw=[.02,.005],
                            power_kw=[0.,5.,4.,3.],direction=[0.,-1.,1.,0.],
                            fit_multistart=2,max_nfev=250,tol=1e-11,linear_solver="exact"))
    dump(args.output,result)
    import gzip
    target=Path(args.output)
    target.with_suffix(target.suffix+".gz").write_bytes(gzip.compress(target.read_bytes(),mtime=0))
    print({"success":result["success"],"histories":2,"seconds":sum(r["seconds"] for r in records),
           "total_ac_calls":sum(r.get("total_ac_calls",0) for r in records)})


if __name__=="__main__":
    main()