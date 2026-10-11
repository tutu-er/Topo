"""Audit saved v7 formal results without solving/tuning design again."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json

import numpy as np

import design_v7 as d


def audit(folder):
    report=json.loads((folder/"formal_results.json").read_text(encoding="utf-8"))
    checks=dict(complete_episode_table=len(report["episodes"])==27,
                source_hash_matches=d.sha(d.__file__)==report["protocol"]["source_hash"],
                all_saved_input_hashes=True,paired_history=True,paired_source=True,
                paired_noise=True,same_energy=True,all_generation_domains=True,
                reference_theta_original_history_feasible=True,qp_float_bounds_outward=True,
                qp_exact_intervals_ordered=True,qp_certificate_flags=True,
                lp_certificate_flags=True,truth_never_excluded=True)
    qp_count=0; lp_count=0; seen=[]
    for ep in report["episodes"]:
        seen.append((ep["topology"],ep["profile"],ep["seed"]))
        firstobs=firsttruth=None
        for entry in ep["inputs"]:
            ofile,tfile=folder/entry["observations"],folder/entry["truth"]
            checks["all_saved_input_hashes"] &= d.sha(ofile)==entry["observations_hash"] and d.sha(tfile)==entry["truth_hash"]
            with np.load(ofile) as z:
                obs={k:z[k] for k in z.files}
            truth=json.loads(tfile.read_text(encoding="utf-8"))
            if firstobs is None:
                firstobs,firsttruth=obs,truth
            for k in obs:
                checks["paired_history"] &= np.array_equal(obs[k][:12],firstobs[k][:12])
            checks["paired_source"] &= truth["source_delta"]==firsttruth["source_delta"]
            # actual_error includes round-to-float simulation conversion, so
            # compare the intended generated meter error with arithmetic slack.
            checks["paired_noise"] &= np.max(abs(np.asarray(truth["actual_error"])-np.asarray(firsttruth["actual_error"])))<2e-17
            checks["same_energy"] &= abs(truth["energy_kwh"]-9)<1e-12
            checks["all_generation_domains"] &= all(truth["checks"].values())
        history=d.observation_slice(firstobs,slice(0,12))
        for h,ref in ep["design"]["representatives"].items():
            poly=d.build_history_polytope(history,h)
            # Check rounded theta as actually passed to reference predictions.
            xf=[(d.f(v)-l)/w for v,l,w in zip(ref["theta"],poly.offset,poly.width)]
            checks["reference_theta_original_history_feasible"] &= all(0<=x<=1 for x in xf) and all(
                sum((a*x for a,x in zip(row,xf)),F())<=b for row,b in zip(poly.exact_a,poly.exact_b))
        qps=[r["result"] for r in ep["design"]["qp_records"]]+list(ep["design"]["representatives"].values())
        for q in qps:
            qp_count+=1
            lo,hi=F(q["exact_lower"]),F(q["exact_upper"])
            checks["qp_exact_intervals_ordered"] &= 0<=lo<=hi
            checks["qp_float_bounds_outward"] &= d.f(q["lower"])<=lo and d.f(q["upper"])>=hi
            checks["qp_certificate_flags"] &= q["exact_feasible_verified"] and q["exact_tangent_dual_verified"]
        for r in ep["evaluation"]:
            lp_count+=1; q=r["result"]
            checks["lp_certificate_flags"] &= q["exact_primal_verified"] and q["exact_box_corrected_dual_verified"]
            if r["candidate"]==ep["topology"]:
                checks["truth_never_excluded"] &= q["status"]!="excluded"
    checks["complete_episode_table"] &= sorted(seen)==sorted((h,p,s) for h in d.NAMES for p in d.PROFILES for s in d.FORMAL_SEEDS)
    result=dict(checks={k:bool(v) for k,v in checks.items()},all_pass=all(checks.values()),
                qp_count_including_references=qp_count,final_lp_count=lp_count,
                formal_results_hash=d.sha(folder/"formal_results.json"),
                source_hash=d.sha(d.__file__),audit_source_hash=d.sha(__file__))
    d.dump(folder/"artifact_audit.json",result)
    print(json.dumps(result,indent=2),flush=True)
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("folder",type=Path)
    audit(p.parse_args().folder)
