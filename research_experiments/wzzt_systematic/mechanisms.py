"""Posthoc structural obstructions and path-selection diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .common import effective_directory, read, write


def diagnose(output,destination):
    import numpy as np
    import pandas as pd
    output=Path(output).resolve()
    destination=Path(destination).resolve()
    sys.path.insert(0,str(output/"source"))
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics
    from research_experiments.rnj.rx75_edge_floor import classify_rx75_atoms
    rows=[]
    detailed=[]
    protocol=read(output/"protocol.json")
    for job in protocol["jobs"]:
        directory=effective_directory(output,job)
        if not (directory/"result.json").exists():
            continue
        payload=read(directory/"payload.json")
        if payload["method"]=="NJ":
            continue
        model=read(directory/"result.json")
        labels=model["labels"]
        truth=physical_truth(output/"data"/payload["case"]/payload["condition"]/"truth_network",labels)["clades"]
        def family(p,*,positive=False,effective=False):
            keep=range(len(p["supports"]))
            if effective:
                keep=classify_rx75_atoms(p["supports"],p["r"],p["x"],model["calibration"])["retained_indices"]
            return {frozenset(labels[i] for i in p["supports"][j]) for j in keep
                    if 2<=len(p["supports"][j])<len(labels) and
                    (not positive or max(p["r"][j],p["x"][j])>1e-9)}
        selected=model.get("selected_index")
        selected=len(model["path"])-1 if selected is None else selected
        chosen,last=model["path"][selected],model["path"][-1]
        final_family=family(last)
        positive=family(last,positive=True)
        effective=family(last,effective=True)
        def crosses(a,b):
            return bool(a&b) and not (a<=b or b<=a)
        missing=truth-final_family
        blockers=[dict(missing_true=sorted(t),crossing_atoms=[sorted(a) for a in positive if crosses(a,t)])
                  for t in missing if any(crosses(a,t) for a in positive)]
        initial_groups=[frozenset(labels[i] for i in s) for s in model["groups"]]
        group_exclusion=[sorted(t) for t in truth if any(0<len(t&g)<len(g) for g in initial_groups)]
        first_block=None
        for index,point in enumerate(model["path"]):
            f=family(point)
            pos=family(point,positive=True)
            if any(any(crosses(a,t) for a in pos) for t in truth-f):
                first_block=index
                break
        chosen_f=family(chosen,effective=True)
        chosen_score=clade_metrics(chosen_f,truth,labels)["clade_f1"]
        last_score=clade_metrics(effective,truth,labels)["clade_f1"]
        record=dict(job_id=job,case=payload["case"],condition=payload["condition"],N=payload["N"],method=payload["method"],
            status=model["status"],stop_reason=model["stop_reason"],selected_index=selected,path_points=len(model["path"]),
            selected_f1=chosen_score,last_f1=last_score,AC_selection_f1_change=chosen_score-last_score,
            training_loss_decrease_after_selected=chosen["search_objective"]-last["search_objective"],
            selected_false_clades=len(chosen_f-truth),last_false_clades=len(effective-truth),
            final_crossing_positive_atoms=sum(any(crosses(a,t) for t in truth) for a in positive-truth),
            missing_true_blocked_by_positive=len(blockers),
            missing_true_blocked_by_family=sum(any(crosses(a,t) for a in final_family) for t in missing),
            initial_group_excluded_true=len(group_exclusion),first_positive_obstruction_path_index=first_block,
            final_refinement_false_clades=sum(any(a<t for t in truth) for a in effective-truth))
        rows.append(record)
        detailed.append(dict(**record,positive_obstructions=blockers,initial_group_excluded_true_clades=group_exclusion))
    frame=pd.DataFrame(rows)
    frame.to_csv(destination/"mechanisms.csv",index=False,encoding="utf-8-sig")
    write(destination/"mechanisms.json",dict(scope="posthoc physical truth audit; crossing atom is a structural obstruction while kept; no counterfactual rerun",
                                               jobs=detailed))
    summary=dict(evaluated=len(frame),positive_obstruction_jobs=int((frame.missing_true_blocked_by_positive>0).sum()),
        initial_group_exclusion_jobs=int((frame.initial_group_excluded_true>0).sum()),
        ac_selected_earlier=int((frame.selected_index<frame.path_points-1).sum()),
        ac_selection_improved_f1=int((frame.AC_selection_f1_change>1e-12).sum()),
        ac_selection_reduced_f1=int((frame.AC_selection_f1_change< -1e-12).sum()),
        incomplete=int((frame.status!="complete").sum()))
    write(destination/"mechanism_summary.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    diagnose(args.output,args.destination)
