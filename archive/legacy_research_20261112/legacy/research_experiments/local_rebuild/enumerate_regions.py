"""Independent 236-hierarchy LP certificates for the eight fixed AC regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .run import ROOT, from_observed, sha, write
from .model import canonical
from .matched import FiniteLP, all_hierarchies, gate, single_moves
from .followup import scores
from .diagnose import select_points


def enumerate_regions(directory):
    output=directory/"region_enumeration";output.mkdir(exist_ok=False)
    write(output/"manifest.json",dict(source_sha256=sha(__file__),
        model_sha256=sha(Path(__file__).with_name("matched.py")),
        phase1_input_sha256=sha(directory/"real_matched.json"),
        hierarchies_per_region=236,region_seconds=30.,LP_seconds=3.,gate_relative=.001,
        purpose="independently resolve MILP numerical certificates on the same eight domains"))
    rows=json.loads((directory/"real_matched.json").read_text(encoding="utf-8"))
    results=[]
    for row in rows:
        parent=ROOT/"outputs/wzzt_paper_20261029_full/jobs"/row["id"]
        payload=json.loads((parent/"payload.json").read_text(encoding="utf-8"))
        saved=json.loads((parent/"result.json").read_text(encoding="utf-8"))
        splits=[from_observed(payload[s],48 if s=="train" else None) for s in ("train","validation","test")]
        for index,search in enumerate(row["searches"]):
            started=perf_counter()
            record=dict(case=row["case"],id=row["id"],region_index=index,region=search["region"],
                        blocks=search["blocks"],status="failed")
            bank=FiniteLP(splits[0],search["domain"],mandatory=search["mandatory"],seconds=3.)
            points=[]
            try:
                for family in all_hierarchies(5):
                    if perf_counter()-started>30.:
                        raise TimeoutError("30-second independent region enumeration budget")
                    expanded=[tuple(sorted(i for a in indices for i in search["blocks"][a])) for indices in family]
                    points.append(bank.fit(expanded))
                best=min(points,key=lambda p:(p["objective"],len(p["supports"]),p["supports"]))
                lower=min(c["dual_bound"] for c in bank.solves)
                start=bank.fit(search["single"]["point"]["supports"])
                moves=single_moves(bank,start)
                assert max(m["gain"] for m in moves)<=gate(start["objective"])+1e-8
                gain=start["objective"]-best["objective"]
                unlocked=gain>gate(start["objective"])
                path=[search["start"],*search["single"]["path"][1:]]
                paths={"single":path,"with_joint":path+([best] if unlocked else [])}
                pool=canonical(s for path in paths.values() for point in path for s in point["supports"])
                refit=FiniteLP(splits[0],pool,mandatory=[(i,) for i in range(splits[0].n)])
                policies={}
                for method,path in paths.items():
                    for order in ("pre","post_refit"):
                        decision=select_points(path,splits[1],saved["calibration"],order=order,refit_bank=refit)
                        policies[method+"_"+order]=dict(choice=decision,score=scores(decision["point"],splits,row["truth"]))
                record.update(status="completed",enumerated=len(points),global_lower_bound=lower,
                    global_upper_bound=best["objective"],absolute_gap=best["objective"]-lower,
                    single=start,joint=best,gain=gain,threshold=gate(start["objective"]),unlocked=unlocked,
                    single_best_gain=max(m["gain"] for m in moves),
                    scores=dict(single=scores(start,splits,row["truth"]),joint=scores(best,splits,row["truth"])),
                    policies=policies,refit_LP_certificates=refit.solves)
            except Exception as exc:
                record.update(error=f"{type(exc).__name__}: {exc}",enumerated=len(points))
            record.update(seconds=perf_counter()-started,LP_certificates=bank.solves)
            results.append(record);write(output/"results.json",results)
            print(row["case"],index,record["status"],len(points),'gain',record.get("gain"),
                  'unlocked',record.get("unlocked"),'seconds',record["seconds"],flush=True)
    summary=dict(planned=8,returned=len(results),completed=sum(r["status"]=="completed" for r in results),
                 unlocked=sum(r.get("unlocked",False) for r in results),
                 all_absolute_gaps=[r.get("absolute_gap") for r in results],
                 seconds=sum(r["seconds"] for r in results))
    write(output/"summary.json",summary);print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path)
    enumerate_regions(parser.parse_args().directory)
