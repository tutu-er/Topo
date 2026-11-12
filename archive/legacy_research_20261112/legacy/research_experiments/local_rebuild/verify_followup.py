"""Check the final LP enumeration and normalized MILP evidence, without solves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import ROOT, from_observed, sha, write
from .verify import independent_loss
from .run import clade_score
from .model import build_model
import numpy as np


def verify(directory):
    read=lambda name:json.loads((directory/name).read_text(encoding="utf-8"))
    exact=read("region_enumeration/results.json")
    normalized=read("scaled_MILP/results.json")
    initial=read("real_matched.json")
    unscaled=read("numerical_recheck/real.json")
    checks=0
    first_bad, second_bad=0,0
    embedded_violations=[]
    for record in exact:
        assert record["status"]=="completed" and record["enumerated"]==236
        assert abs(record["absolute_gap"])<=1e-8
        assert abs(min(r["dual_bound"] for r in record["LP_certificates"])-record["global_lower_bound"])<=1e-8
        for certificate in record["LP_certificates"]:
            assert certificate["status"]==0 and certificate["valid"]
            assert max(certificate["absolute_gap"],certificate["constraint_violation"],
                       certificate["dual_stationarity"],certificate["dual_sign_violation"],
                       certificate["reproduction_error"])<=1e-7
            checks+=1
        parent=ROOT/"outputs/wzzt_paper_20261029_full/jobs"/record["id"]
        payload=json.loads((parent/"payload.json").read_text(encoding="utf-8"))
        source=next(r for r in initial if r["case"]==record["case"])
        search=source["searches"][record["region_index"]]
        train=from_observed(payload["train"],48)
        model,meta=build_model(train,search["single"]["point"]["supports"],search["region"],
                               search["blocks"],hard=search["mandatory"])
        exact_point=record["joint"]
        lookup={tuple(s):(r,x) for s,r,x in zip(exact_point["supports"],exact_point["r"],exact_point["x"])}
        values=np.zeros(model.variables.size)
        prediction=np.zeros_like(train.target[0])
        for k,s in enumerate(meta["all_supports"]):
            values[model.r.start+k],values[model.x.start+k]=lookup.get(s,(0.,0.))
        for k,s in enumerate(meta["candidates"]):values[meta["selection"].start+k]=float(s in lookup)
        for s,(r,x) in lookup.items():
            z=np.array([float(i in s) for i in range(train.n)])
            prediction+=((train.p[0]@z)*r+(train.q[0]@z)*x)[:,None]*z
        values[model.absolute]=abs(train.target[0]-prediction).ravel()
        constraints=model.rows.build(model.variables.size);activity=constraints.A@values
        violation=float(max(0.,np.max(activity-constraints.ub),np.max(constraints.lb-activity),
                            np.max(-values),np.max(values-np.asarray(model.variables.upper))))
        assert violation<=1e-8
        assert abs(np.asarray(model.variables.objective)@values-record["global_upper_bound"])<=1e-8
        embedded_violations.append(violation);checks+=1
        before=source["searches"][record["region_index"]]["joint"]["diagnostics"]["dual_bound"]
        after=next(r for r in unscaled if r["case"]==record["case"] and r["region_index"]==record["region_index"])["joint"]["diagnostics"]["dual_bound"]
        first_bad+=before is not None and before>record["global_upper_bound"]+1e-8
        second_bad+=after is not None and after>record["global_upper_bound"]+1e-8
        scaled=next(r for r in normalized if r["case"]==record["case"] and r["region_index"]==record["region_index"])
        assert scaled["matches_enumeration"] and scaled["native"]["audit"]["valid"] and scaled["native"]["certified"]
        assert scaled["physical_dual_bound"]<=record["global_upper_bound"]+1e-8
        assert record["unlocked"]==(record["gain"]>record["threshold"])
        for split in ("train","validation","test"):
            data=from_observed(payload[split],48 if split=="train" else None)
            assert sha(payload[split])==source["data_sha256"][split]
            for method in ("single","joint"):
                point=record[method]
                value=independent_loss(data.p[0],data.q[0],data.target[0],point)
                assert abs(value-record["scores"][method][split])<=1e-8
                checks+=1
            if split=="train":
                point=scaled["physical_point"]
                value=independent_loss(data.p[0],data.q[0],data.target[0],point)
                assert abs(value-record["global_upper_bound"])<=1e-8
                assert abs(scaled["native"]["diagnostics"]["objective"]/scaled["measurement_scale"]-value)<=1e-8
                assert clade_score(point,data.n,record["scores"]["joint"]["truth"])["f1"]==record["scores"]["joint"]["f1"]
                checks+=3
            for policy in record["policies"].values():
                point=policy["choice"]["point"]
                value=independent_loss(data.p[0],data.q[0],data.target[0],point)
                assert abs(value-policy["score"][split])<=1e-8
                checks+=1
        assert abs(record["single"]["objective"]-record["joint"]["objective"]-record["gain"])<=1e-8
    summary=dict(passed=True,checks=checks,regions=len(exact),enumerated_families=sum(r["enumerated"] for r in exact),
                 unlocked=sum(r["unlocked"] for r in exact),
                 normalized_MILP_verified=len(normalized),
                 original_scale_false_lower_bounds=first_bad,
                 original_scale_no_presolve_false_lower_bounds=second_bad,
                 enumerated_optima_embedded_in_full_MILPs=len(embedded_violations),
                 maximum_embedding_violation=max(embedded_violations),
                 source_sha256=sha(__file__),
                 interpretation="final LP enumeration supersedes provisional unscaled MILP exclusion claims")
    write(directory/"final_verification.json",summary);print(summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path)
    verify(parser.parse_args().directory)
