"""Equivalent measurement-unit normalization, checked against exact LP banks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import ROOT, core, from_observed, sha, write
from .model import prediction_loss, solve_rebuild


def normalized_solve(scenarios, family, region, blocks, *, hard=(), incumbent=None,
                     scale=1000., seconds=8.):
    """Scale P/Q/Y together; coefficients, domain and L1 minimizer are unchanged.

    Raw diagnostics remain in scaled units. Physical values are returned
    separately; no reported dual bound or failed audit is overwritten.
    """
    original=core._prepare_scenarios(scenarios)
    scale=core._require_positive_finite("scale",scale)
    scaled=core._PreparedScenarios(original.labels,tuple(p*scale for p in original.p),
                                   tuple(q*scale for q in original.q),tuple(y*scale for y in original.target))
    native=solve_rebuild(scaled,family,region,blocks,hard=hard,incumbent=incumbent,
                         presolve=False,seconds=seconds)
    physical=None
    if native["point"] is not None:
        point=native["point"]
        value=prediction_loss(original,point["supports"],point["r"],point["x"])
        physical={**point,"objective":value}
        assert abs(value-point["objective"]/scale)<=1e-8
    bound=native["diagnostics"]["dual_bound"]
    return dict(measurement_scale=scale,physical_point=physical,
                physical_dual_bound=None if bound is None else bound/scale,native=native)


def recheck(directory):
    output=directory/"scaled_MILP";output.mkdir(exist_ok=False)
    write(output/"manifest.json",dict(scale=1000.,seconds_per_region=8.,presolve=False,
        source_sha256=sha(__file__),model_sha256=sha(Path(__file__).with_name("model.py")),
        enumeration_sha256=sha(directory/"region_enumeration/results.json"),
        purpose="same eight fixed domains; scale all P/Q/Y by 1000, evaluate in original units"))
    inputs=json.loads((directory/"real_matched.json").read_text(encoding="utf-8"))
    exact=json.loads((directory/"region_enumeration/results.json").read_text(encoding="utf-8"))
    results=[]
    for row in exact:
        source=next(r for r in inputs if r["case"]==row["case"])
        search=source["searches"][row["region_index"]]
        parent=ROOT/"outputs/wzzt_paper_20261029_full/jobs"/row["id"]
        payload=json.loads((parent/"payload.json").read_text(encoding="utf-8"))
        train=from_observed(payload["train"],48)
        single=search["single"]["point"]
        result=normalized_solve(train,single["supports"],search["region"],search["blocks"],
                                hard=search["mandatory"],incumbent=single)
        point=result["physical_point"];bound=result["physical_dual_bound"]
        result.update(case=row["case"],region_index=row["region_index"],
            enumerated_optimum=row["joint"]["objective"],
            incumbent_difference=None if point is None else point["objective"]-row["joint"]["objective"],
            lower_bound_contradiction=bound is not None and bound>row["joint"]["objective"]+1e-8)
        result["matches_enumeration"]=bool(point is not None and abs(result["incumbent_difference"])<=1e-8 and
                                           not result["lower_bound_contradiction"] and result["native"]["audit"]["valid"])
        results.append(result);write(output/"results.json",results)
        print(row["case"],row["region_index"],result["native"]["diagnostics"]["status"],
              'matches',result["matches_enumeration"],'difference',result["incumbent_difference"],flush=True)
    summary=dict(planned=8,returned=len(results),matches=sum(r["matches_enumeration"] for r in results),
                 certified=sum(r["native"]["certified"] for r in results),
                 valid=sum(r["native"]["audit"]["valid"] for r in results),
                 timeouts=sum(r["native"]["diagnostics"]["status"]==1 for r in results),
                 bound_contradictions=sum(r["lower_bound_contradiction"] for r in results))
    write(output/"summary.json",summary);print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path)
    recheck(parser.parse_args().directory)
