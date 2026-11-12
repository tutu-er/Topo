"""Choose penalties using only disjoint development predictive observations."""
from __future__ import annotations

from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

from .run import read,write,sha,config_id,result_choice
from research_experiments.rebuild_benchmark.engine import prepare,loss
from research_experiments.wzzt_paper.common import observed


def choose(rows):
    """Paired one-SE rule is a selection heuristic, not a coverage theorem."""
    best=min(rows,key=lambda r:(r['mean_log_ratio'],r['mean_internal'],r['coefficient']))
    eligible=[]
    for row in rows:
        differences=np.array(row['group_log_ratios'])-np.array(best['group_log_ratios'])
        se=float(np.std(differences,ddof=1)/np.sqrt(len(differences))) if len(differences)>1 else 0.
        gap=float(np.mean(differences));threshold=max(np.log(1.001),se)
        row.update(paired_gap_to_best=gap,paired_SE=se,eligible=gap<=threshold+1e-12)
        if row['eligible']:
            eligible.append(row)
    selected=min(eligible,key=lambda r:(r['mean_internal'],r['mean_log_ratio'],r['coefficient']))
    return selected,best


def select(output):
    output=Path(output);protocol=read(output/'protocol.json');catalog=read(output/'catalog.json')
    completion=read(output/'development_completion.json')
    if completion['returned']!=completion['planned']:
        raise RuntimeError('all development attempts must be retained and resolved before freezing a parameter')
    if (output/'frozen.json').exists():
        raise FileExistsError('parameters already frozen')
    data={j:prepare(observed(catalog[j]['test'])) for j in protocol['development']}
    baseline={j:read(output/'development'/j/config_id(0.)/'result.json') for j in protocol['development']}
    base_loss={j:loss(data[j],baseline[j]['selected_plain']['bank']['point']) for j in data}
    grid=[];records=[];selected={};strict={}
    for mode in ('search_only','selection_only','both'):
        mode_rows=[]
        for coefficient in protocol['grid']:
            fits=[]
            for job in protocol['development']:
                cid=config_id(0. if mode=='selection_only' else coefficient)
                file=output/'development'/job/cid/'result.json';r=read(file)
                assert not r['truth_used'] and not r['test_opened']
                q=result_choice(r,mode,coefficient)['bank'];value=loss(data[job],q['point'])
                logratio=float(np.log(max(value,1e-12)/max(base_loss[job],1e-12)))
                meta=catalog[job]
                fits.append(dict(job=job,case=meta['case'],bundle=meta['bundle'],coefficient=coefficient,mode=mode,
                    tuning_MAE=value,log_ratio=logratio,internal=q['internal'],result_sha256=sha(file)))
            grouped=pd.DataFrame(fits).groupby(['case','bundle']).log_ratio.mean()
            row=dict(mode=mode,coefficient=coefficient,mean_log_ratio=float(np.mean([q['log_ratio'] for q in fits])),
                     mean_internal=float(np.mean([q['internal'] for q in fits])),group_log_ratios=grouped.tolist(),inputs=len(fits))
            mode_rows.append(row);records+=fits
        picked,minimum=choose(mode_rows);selected[mode]=dict(picked);strict[mode]=dict(minimum);grid+=mode_rows
    coefficients=sorted({0.,*(q['coefficient'] for q in selected.values()),strict['both']['coefficient']})
    frozen=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,strict_min=strict,
        evaluation_settings={config_id(v):dict(coefficient=v) for v in coefficients},
        methods=dict(joint_unregularized=dict(config_id=config_id(0.),mode='search_only',coefficient=0.),
            count_search=dict(config_id=config_id(selected['search_only']['coefficient']),mode='search_only',coefficient=selected['search_only']['coefficient']),
            count_selection=dict(config_id=config_id(0.),mode='selection_only',coefficient=selected['selection_only']['coefficient']),
            count_both=dict(config_id=config_id(selected['both']['coefficient']),mode='both',coefficient=selected['both']['coefficient']),
            count_both_strict=dict(config_id=config_id(strict['both']['coefficient']),mode='both',coefficient=strict['both']['coefficient'])),
        protocol_sha256=sha(output/'protocol.json'),truth_used=False,evaluation_test_used=False,
        development_tuning_observations_used=True,development_outer_hashes={j:sha(catalog[j]['test']) for j in protocol['development']},
        rule='paired one-SE or 0.1% near-minimum predictive error, then fewer lines; strict minimum both-stage control retained')
    write(output/'frozen.json',frozen);write(output/'tuning_grid.json',grid)
    pd.DataFrame([{k:v for k,v in q.items() if k!='group_log_ratios'} for q in grid]).to_csv(output/'tuning_grid.csv',index=False)
    pd.DataFrame(records).to_csv(output/'tuning_records.csv',index=False)
    print('SELECTED',{k:q['coefficient'] for k,q in selected.items()},'strict_both',strict['both']['coefficient'],
          'evaluation searches',len(coefficients)*len(protocol['evaluation']),flush=True)
