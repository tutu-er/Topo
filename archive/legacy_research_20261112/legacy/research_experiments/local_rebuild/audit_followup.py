"""Independent output checks and no-presolve rechecks of the matched pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import ROOT, scenes, from_observed, sha, write
from .verify import independent_loss
from .model import canonical, solve_rebuild, prediction_loss
from .matched import FiniteLP, gate, single_moves
from .followup import scores
from .diagnose import select_points
import numpy as np


def audit(directory):
    output=directory/"numerical_recheck"
    output.mkdir(exist_ok=False)
    manifest=dict(sources={n:sha(Path(__file__).with_name(n)) for n in
                  ("audit_followup.py","model.py","matched.py","diagnose.py","followup.py","verify.py")},
                  phase1_sources={n:sha(p) for n,p in ((p.name,p) for p in (directory/"source_phase1").glob('*.py'))},
                  inputs={n:sha(directory/n) for n in ("landscape.json","real_matched.json","protocol_all.json")},
                  policy=dict(presolve=False,MILP_seconds=8.,absolute_gate=1e-10,relative_gate=.001,
                              known_incumbent="verified in full MILP rows, not a solver MIP start"))
    write(output/"manifest.json",manifest)
    checks=0
    protocol=json.loads((directory/"protocol_all.json").read_text(encoding="utf-8"))
    for name,digest in protocol["sources"].items():
        assert sha(directory/"source_phase1"/name)==digest
        checks+=1
    landscape=json.loads((directory/"landscape.json").read_text(encoding="utf-8"))
    for row in landscape:
        assert row["status"]=="completed" and row["hierarchy_count"]==236
        assert abs(row["MILP_vs_enumeration"])<=1e-8
        with np.load(directory/(row["id"]+".npz")) as data:
            for entry in row["starts"]:
                p=entry["endpoint"]
                value=independent_loss(data["train_P"],data["train_Q"],data["train_Y"],p)
                assert abs(value-p["objective"])<=1e-8
                checks+=1
        for record in row["LP_certificates"]:
            assert record["status"]==0 and record["valid"]
            assert max(record["absolute_gap"],record["constraint_violation"],record["dual_stationarity"],
                       record["dual_sign_violation"],record["reproduction_error"])<=1e-7
            checks+=1
    # The first witness is selected by dataset order and training stationarity,
    # without considering topology scores or independent-split errors.
    source=next((row for row in landscape if row["first_witness"]),None)
    witness=None
    if source:
        previous=source["first_witness"];point=previous["endpoint"]
        with np.load(directory/(source["id"]+".npz")) as data:
            splits=[scenes(data[s+"_P"],data[s+"_Q"],data[s+"_Y"]) for s in ("train","validation","test")]
        n=splits[0].n
        mandatory=[(i,) for i in range(n)]+[tuple(range(n))]
        bank=FiniteLP(splits[0],source["domain"],mandatory=mandatory)
        start=bank.fit(point["supports"])
        moves=single_moves(bank,start)
        assert all(m["gain"]<=gate(start["objective"]) for m in moves)
        joint=solve_rebuild(splits[0],start["supports"],range(n),tuple((i,) for i in range(n)),
                            hard=mandatory,presolve=False,incumbent=start,seconds=8.)
        assert joint["point"] is not None and joint["audit"]["valid"] and joint["certified"]
        assert start["objective"]-joint["point"]["objective"]>gate(start["objective"])
        assert abs(joint["point"]["objective"]-source["enumeration_best"]["objective"])<=1e-8
        witness=dict(id=source["id"],single=start,
            moves=[dict(candidate=m["candidate"],gain=m["gain"],objective=m["point"]["objective"]) for m in moves],
            best_single_gain=max(m["gain"] for m in moves),threshold=gate(start["objective"]),
            joint=joint,comparison=dict(single=scores(start,splits,source["truth"]),
                                        joint=scores(joint["point"],splits,source["truth"])),
            LP_certificates=bank.solves)
        checks+=len(moves)+3
        write(output/"witness.json",witness)
        print('verified witness',source["id"],witness["best_single_gain"],flush=True)
    rows=json.loads((directory/"real_matched.json").read_text(encoding="utf-8"))
    rechecks=[]
    for row in rows:
        assert row["status"]=="completed"
        parent=ROOT/"outputs/wzzt_paper_20261029_full/jobs"/row["id"]
        payload=json.loads((parent/"payload.json").read_text(encoding="utf-8"))
        saved=json.loads((parent/"result.json").read_text(encoding="utf-8"))
        splits=[from_observed(payload[s],48 if s=="train" else None) for s in ("train","validation","test")]
        for split in ("train","validation","test"):
            assert sha(payload[split])==row["data_sha256"][split]
            checks+=1
        for i,search in enumerate(row["searches"]):
            single=search["single"]["point"]
            train=splits[0]
            value=independent_loss(train.p[0],train.q[0],train.target[0],single)
            assert abs(value-single["objective"])<=1e-8
            assert search["single"]["stationary"]
            joint=solve_rebuild(train,single["supports"],search["region"],search["blocks"],
                hard=search["mandatory"],presolve=False,incumbent=single,seconds=8.)
            point=joint["point"]
            gain=None if point is None else single["objective"]-point["objective"]
            threshold=gate(single["objective"])
            lower=joint["diagnostics"]["dual_bound"]
            bound_consistent=lower is not None and lower<=single["objective"]+1e-8
            gain_upper=None if lower is None else single["objective"]-lower
            accepted=bool(gain is not None and gain>threshold and joint["audit"]["valid"])
            excludes=bool(bound_consistent and gain_upper<=threshold-1e-8)
            old_lower=search["joint"]["diagnostics"]["dual_bound"]
            old_contradiction=old_lower is not None and old_lower>single["objective"]+1e-8
            points=[search["start"],*search["single"]["path"][1:]]
            paths={"single":points,"with_joint":points+([point] if accepted else [])}
            pool=canonical(s for path in paths.values() for p in path for s in p["supports"])
            refit=FiniteLP(train,pool,mandatory=[(j,) for j in range(train.n)])
            policies={}
            for method,path in paths.items():
                for order in ("pre","post_refit"):
                    choice=select_points(path,splits[1],saved["calibration"],order=order,refit_bank=refit)
                    policies[method+"_"+order]=dict(choice=choice,score=scores(choice["point"],splits,row["truth"]))
            item=dict(case=row["case"],id=row["id"],region_index=i,region=search["region"],blocks=search["blocks"],
                      single=single,joint=joint,joint_gain=gain,threshold=threshold,
                      unlocked=accepted,gain_upper_bound=gain_upper,bound_consistent=bound_consistent,
                      excludes_above_gate_improvement=excludes,
                      prior_presolve_bound_contradicted_witness=old_contradiction,
                      scores=dict(single=scores(single,splits,row["truth"]),
                                  joint=None if point is None else scores(point,splits,row["truth"])),
                      policies=policies,refit_LP_certificates=refit.solves)
            rechecks.append(item);write(output/"real.json",rechecks)
            checks+=3
            print(row["case"],i,'gain',gain,'unlocked',accepted,'excluded',excludes,
                  'audit',joint["audit"]["valid"],'status',joint["diagnostics"]["status"],flush=True)
    # Assert the independent held-out metrics match the saved initial evaluator.
    for row in rechecks:
        for policy in row["policies"].values():
            point=policy["choice"]["point"]
            parent=ROOT/"outputs/wzzt_paper_20261029_full/jobs"/row["id"]
            payload=json.loads((parent/"payload.json").read_text(encoding="utf-8"))
            for split in ("validation","test"):
                data=from_observed(payload[split])
                actual=independent_loss(data.p[0],data.q[0],data.target[0],point)
                assert abs(actual-policy["score"][split])<=1e-8
                checks+=1
    summary=dict(passed=True,checks=checks,small_datasets=len(landscape),
                 enumerated_starts=sum(r["hierarchy_count"] for r in landscape),
                 trap_starts=sum(r["stationary_trap_starts"] for r in landscape),
                 trap_datasets=sum(r["stationary_trap_starts"]>0 for r in landscape),
                 witness_rechecked=witness is not None,real_regions=len(rechecks),
                 real_unlocked=sum(r["unlocked"] for r in rechecks),
                 real_excluded_above_gate=sum(r["excludes_above_gate_improvement"] for r in rechecks),
                 real_valid_incumbents=sum(r["joint"]["audit"]["valid"] for r in rechecks),
                 real_strict_certificates=sum(r["joint"]["certified"] for r in rechecks),
                 prior_presolve_contradictions=sum(r["prior_presolve_bound_contradicted_witness"] for r in rechecks))
    write(output/"verification.json",summary);print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path)
    audit(parser.parse_args().directory)
