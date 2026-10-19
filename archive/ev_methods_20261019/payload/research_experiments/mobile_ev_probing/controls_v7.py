"""Prespecified small controls and post-hoc diagnostics; no policy tuning."""
from pathlib import Path
import argparse
import json

import numpy as np

import design_v7 as d
import local_shared_source_v6 as v6


def run_controls(output):
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(design_source_hash=d.sha(d.__file__),control_source_hash=d.sha(__file__),
                  cases=[dict(name="source_perfectly_correlated",seed=7301,epsilon=1e-5),
                         dict(name="large_measurement_box",seed=7302,epsilon=.002),
                         dict(name="informative_history_alone",seed=7303,epsilon=1e-5),
                         dict(name="box_and_reset",seed=7304,epsilon=1e-5)],
                  extra_final_lp_count=57,truth_profile="symmetric",truth_topology="cherry12")
    d.dump(output/"controls_protocol.json",protocol)
    episodes=[]
    for name,seed,eps,corr in (("source_perfectly_correlated",7301,1e-5,True),
                               ("large_measurement_box",7302,.002,False)):
        ep=d.evaluate_actions("cherry12",d.PROFILES["symmetric"],seed,epsilon=eps,
                              correlated=corr,output=output,prefix=name)
        ep.update(name=name,topology="cherry12",seed=seed)
        episodes.append(ep); d.dump(output/f"{name}_results.json",ep)
        print(name,ep["summary"]["selected"],flush=True)
    records=[]
    obs,truth=d.simulate("cherry12",d.PROFILES["symmetric"],7303,0,"endpoint",rich_history=True)
    hist=d.observation_slice(obs,slice(0,12))
    np.savez_compressed(output/"informative_history_observations.npz",**hist)
    d.dump(output/"informative_history_evaluation_truth.json",truth)
    for h in d.NAMES:
        records.append(dict(case="informative_history_alone",candidate=h,
                            result=v6.solve_lp(hist,h,parameter_box=d.BOX)))
    for policy in ("constant","endpoint"):
        obs,truth=d.simulate("cherry12",d.PROFILES["symmetric"],7304,0,policy)
        np.savez_compressed(output/f"box_reset_{policy}_observations.npz",**obs)
        d.dump(output/f"box_reset_{policy}_evaluation_truth.json",truth)
        for source,sharing in (("shared","joint"),("box","joint"),("shared","reset")):
            for h in d.NAMES:
                records.append(dict(case="box_and_reset",policy=policy,source_model=source,
                    parameter_sharing=sharing,candidate=h,
                    result=v6.solve_lp(obs,h,source_model=source,parameter_sharing=sharing,parameter_box=d.BOX)))
    all_lp=[r["result"] for e in episodes for r in e["evaluation"]]+[r["result"] for r in records]
    report=dict(protocol=protocol,episodes=episodes,other_records=records,
                lp_count=len(all_lp),all_certified=all(r.get("exact_primal_verified") and r.get("exact_box_corrected_dual_verified") for r in all_lp),
                max_lp_gap=max(r.get("exact_gap",float("inf")) for r in all_lp),
                max_verified_lp_upper_bound=max(r["verified_upper_bound"] for r in all_lp),
                all_lp_upper_bounds_below_tau_cap=all(r["verified_upper_bound"]<1. for r in all_lp),
                design_source_hash_unchanged=d.sha(d.__file__)==protocol["design_source_hash"],
                control_source_hash_unchanged=d.sha(__file__)==protocol["control_source_hash"])
    d.dump(output/"controls_results.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("protocol","episodes","other_records")},indent=2),flush=True)


def posthoc(formal,output):
    """Uses future evaluation only for reporting, never changes design decisions."""
    report=json.loads(Path(formal).read_text(encoding="utf-8"))
    rows=[]
    for ep in report["episodes"]:
        stats=ep["summary"]["actionstats"]
        oracle=ep["summary"]["oracle_margin_sites"]["possible"]
        for ref,scores in ep["design"]["scores_by_reference"].items():
            selection=d.rank_intervals({a:s for a,s in scores.items() if a.endswith("endpoint")})
            chosen=selection["selected"]
            rows.append(dict(episode=ep["name"],truth=ep["topology"],profile=ep["profile"],
                reference=ref,correct_reference=ref==ep["topology"],selected=chosen,
                unique=selection["unique_resolved"],possible=selection["possible"],
                excluded_wrong=stats[chosen]["excluded_wrong"],
                margin_lower=stats[chosen]["closest_wrong_lower"],
                margin_upper=stats[chosen]["closest_wrong_upper"],oracle_hit=chosen in oracle))
    byref={}
    for correct in (True,False):
        sub=[r for r in rows if r["correct_reference"]==correct]
        byref[str(correct)]=dict(count=len(sub),both_wrong_excluded=sum(r["excluded_wrong"]==2 for r in sub),
            oracle_hits=sum(r["oracle_hit"] for r in sub),mean_margin_lower=float(np.mean([r["margin_lower"] for r in sub])))
    byprofile={}
    for profile in d.PROFILES:
        es=[e for e in report["episodes"] if e["profile"]==profile]
        byprofile[profile]=d.aggregate(es)
    result=dict(formal_file=str(formal),formal_hash=d.sha(formal),rows=rows,
                by_reference_correctness=byref,by_profile=byprofile,
                semantics="post-hoc sensitivity only; wrong references were not selected using truth")
    d.dump(output,result)
    print(json.dumps(byref,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--formal",type=Path)
    args=p.parse_args()
    if args.formal:
        posthoc(args.formal,args.output)
    else:
        run_controls(args.output)
