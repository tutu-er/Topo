"""Distinguish wrong grouping from harmless-to-compatibility over-resolution."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from research_experiments.joint_five.run import ROOT, read, write
from research_experiments.joint_five.analyze import supports
from research_experiments.local_rebuild.model import compatible
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth

PARENT = ROOT/'outputs/joint_five_rnj_wzzt_20261031'


def error_type(support, truth):
    if support in truth:
        return 'true', []
    blocked = [t for t in truth if not compatible(support, t)]
    return ('crosses_truth' if blocked else 'compatible_overresolution'), sorted(blocked)


def diagnose(output):
    destination = Path(output)/'diagnosis'; destination.mkdir(parents=True, exist_ok=True)
    catalog = read(PARENT/'catalog.json'); protocol = read(PARENT/'protocol.json')
    records = []; cases = []; examples = {}
    for job in protocol['evaluation']:
        meta = catalog[job]; common = read(PARENT/'common'/job/'result.json'); labels = common['labels']; n = len(labels)
        truth = physical_truth(Path(meta['truth_network']), labels); pos = {v:i for i,v in enumerate(labels)}
        true = {tuple(sorted(pos[v] for v in t)) for t in truth['clades']}
        r = read(PARENT/'evaluation'/job/'baseline'/'result.json'); trajectory = r['trajectory']
        raw = trajectory['states']['rnj']; initial = supports(raw,n,True); prior = set(initial)
        endpoint = supports(trajectory['point'],n,True); final = supports(r['selected']['point'],n)
        if not all(compatible(s,t) for i,s in enumerate(initial) for t in list(initial)[i+1:]):
            raise AssertionError('RNJ output is not itself laminar')
        initial_cross = 0; later_cross = 0
        def append(s, stage, point, iteration):
            kind, blocked = error_type(s,true)
            if kind == 'true':
                return kind
            index = list(map(tuple,point['supports'])).index(s)
            record = dict(job=job,panel=meta['panel'],case=meta['case'],stage=stage,iteration=iteration,
                support=s,labels=[labels[i] for i in s],terminal_size=len(s),size='small' if len(s)<=5 else 'large',
                error_type=kind,blocked_true=blocked,blocked_count=len(blocked),
                positive=max(point['r'][index],point['x'][index])>1e-9,
                survives_endpoint=s in endpoint,survives_final=s in final,selected_origin=r['selected_origin'])
            records.append(record)
            if kind=='crosses_truth':
                examples.setdefault(stage,record)
            return kind
        for s in sorted(initial):
            initial_cross += append(s,'rnj_input',raw,0)=='crosses_truth'
        for k,point in enumerate(trajectory['path'][1:],1):
            current = supports(point,n,True)
            for s in sorted(current-prior):
                later_cross += append(s,'wzzt_added',point,k)=='crosses_truth'
            prior = current
        other_bank = final-initial-supports(trajectory['point'],n,True)
        for s in sorted(other_bank):
            append(s,'other_rnj_bank',r['selected']['point'],-1)
        cases.append(dict(job=job,panel=meta['panel'],case=meta['case'],initial_cross=initial_cross,later_cross=later_cross,
                          initial_false=len(initial-true),final_origin=r['selected_origin']))
    frame = pd.DataFrame(records); frame.to_csv(destination/'error_ledger.csv',index=False)
    frame.groupby(['panel','stage','size','error_type']).agg(occurrences=('job','count'),positive=('positive','sum'),
        survives_endpoint=('survives_endpoint','sum'),survives_final=('survives_final','sum')).reset_index().to_csv(destination/'error_summary.csv',index=False)
    case_frame = pd.DataFrame(cases); case_frame.to_csv(destination/'case_origins.csv',index=False)
    main = case_frame[case_frame.panel=='main']
    receipt = dict(scope='offline diagnosis of the original full-RNJ + WZZT baseline; no inference/tuning oracle',
        RNJ_internal_crossings=0,main_inputs=len(main),initial_any_false=int((main.initial_false>0).sum()),
        initial_cross_cases=int((main.initial_cross>0).sum()),later_cross_cases=int((main.later_cross>0).sum()),
        both_cross_cases=int(((main.initial_cross>0)&(main.later_cross>0)).sum()),
        clean_input_later_cross=int(((main.initial_cross==0)&(main.later_cross>0)).sum()),examples=examples)
    write(destination/'verification.json',receipt)
    print(frame[frame.panel=='main'].groupby(['stage','error_type']).size().to_string(),flush=True)
    print({k:v for k,v in receipt.items() if k!='examples'},flush=True)
