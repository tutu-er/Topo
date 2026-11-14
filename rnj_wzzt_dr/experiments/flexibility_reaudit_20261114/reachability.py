"""Post-run analytic power-box bounds for the frozen absolute-recovery QCQP.

Read sealed forecasts and historical validation only. No truth, new-day
observations, fitting, optimizer, or AC solver enters this calculation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .resources import read, reference, checked
from ...observations import load
from ..reactive_flex_20261114.domain import PQIncrementCase
from ..absolute_identification_20261114.solver import SymmetricRecovery


def checked_forecast(folder):
    seal_ref=reference(Path(folder)/'seal.json')
    seal=read(seal_ref['path'])
    planner_ref=checked(seal['planner'])
    with np.load(planner_ref['path'],allow_pickle=False) as z:
        forecast=z['forecast'].copy()
    return forecast,dict(seal=seal_ref,planner=planner_ref,score=checked(seal['score']),
        protocol=checked(seal['protocol']))


def run(directory,base_kw=200.):
    directory=Path(directory).resolve()
    complete_ref=reference(directory/'complete.json')
    complete=read(complete_ref['path'])
    if complete['status']!='complete' or not complete['source_unchanged']:
        raise ValueError('Expected a complete source-unchanged run')
    protocol_ref=reference(directory/'protocol.json');protocol=read(protocol_ref['path'])
    inputs_ref=checked(protocol['inputs']);entries=read(inputs_ref['path'])
    finished=read(directory/'complete_fraction0.10.json')
    if finished['failed_tasks'] or finished['completed_forecasts']!=finished['total_forecasts']:
        raise ValueError('The requested full forecast batch has not completed')
    original_sources=[checked(r) for r in protocol['sources']]
    previous_ref=checked(protocol['previous_protocol']);previous=read(previous_ref['path'])
    previous_inputs_ref=checked(previous['inputs']);previous_entries=read(previous_inputs_ref['path'])
    if previous['seed']!=protocol['seed']:
        raise ValueError('Cannot claim same forecast streams for different seeds')
    prior_by_id={e['original_id']:e for e in previous_entries if e['resource_variant']=='eligible_sites'}
    prior_root=Path(protocol['previous_run']['path'])
    rows=[];identity_error=0.
    for entry in entries:
        score_ref=checked(entry['score']);context_ref=checked(entry['resource_context'])
        validation_ref=checked(entry['files']['validation.npz'])
        score=read(score_ref['path']);context=read(context_ref['path']);validation=load(validation_ref['path'])
        previous_entry=prior_by_id[entry['original_id']]
        previous_score_ref=checked(previous_entry['score']);previous_score=read(previous_score_ref['path'])
        if score != previous_score:
            raise ValueError('Prior forecast comparison uses a different frozen score')
        target=int(score['map_index'])
        nearest=min([i for i in range(len(score['parameters'])) if i!=target],
                    key=lambda i:(score['parameters'][i]['history_sse'],i))
        for draw in range(protocol['forecasts']):
            folder=directory/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            forecast,sealed=checked_forecast(folder)
            if sealed['score']!=score_ref:
                raise ValueError('Entry and sealed forecast score sources differ')
            previous_folder=prior_root/'fraction0.10'/previous_entry['id']/f'forecast{draw:02d}'
            old_forecast,old_sealed=checked_forecast(previous_folder)
            if not np.array_equal(forecast,old_forecast):
                raise ValueError('New sealed forecast differs from the cited prior forecast')
            case=PQIncrementCase(context,score['labels'],forecast,base_kw,q_resources=entry['q_resources'],fraction=.1)
            model=SymmetricRecovery(case,score,validation,forecast,base_kw,mode='symmetric_guard')
            pair_rows=[]
            # Independent centre-radius formula verifies the existing lo/hi bound.
            center=np.repeat((case.lower+case.upper)/2,case.repeat,axis=0)
            radius=np.repeat((case.upper-case.lower)/2,case.repeat,axis=0)
            for k,(i,j) in enumerate(model.pairs):
                D=model.H[i]-model.H[j]
                effect=D@case.B/base_kw
                middle=forecast@D.T+center@effect.T
                bound=float(np.square(np.abs(middle)+radius@np.abs(effect).T).sum()/(2*model.sigma2))
                identity_error=max(identity_error,abs(bound-model.upper[k]))
                if not np.isclose(bound,model.upper[k],rtol=1e-12,atol=1e-10):
                    raise ArithmeticError('Centre-radius bound differs from existing upper')
                h=float(model.h[k])
                required=(model.z+math.sqrt(model.z**2-2*(h-model.tie)))**2/2 if h<model.tie else None
                pair_rows.append(dict(pair_representative=[i,j],
                    pair_native=[int(score['models'][i]['index']),int(score['models'][j]['index'])],
                    historical_offset=h,normal_quantile=float(model.z),tie=float(model.tie),
                    J_at_zero=float(model.c[k]),J_upper=float(model.upper[k]),independent_J_upper=bound,
                    J_required_when_h_below_tie=required,
                    upper_over_required=None if required is None else float(model.upper[k]/required),
                    nominal_constraints_infeasible_by_upper=bool(required is not None and model.upper[k]<required-1e-7)))
            near=next(row for row in pair_rows if set(row['pair_representative'])=={target,nearest})
            rows.append(dict(id=entry['id'],variant=entry['resource_variant'],forecast=draw,
                sealed_sources=sealed,score=score_ref,registration=context_ref,validation=validation_ref,
                prior_forecast_sources=old_sealed,prior_score=previous_score_ref,
                prior_forecast_bitwise_equal=True,old_forecast_fallback_used=False,
                targets=model.targets,per_competitor_alpha=float(model.alpha),base_kw=base_kw,
                resource_count=case.R,reactive_resource_count=case.Q_R,
                near_pair=near,all_pairs=pair_rows,
                any_nominal_pair_proved_infeasible=any(p['nominal_constraints_infeasible_by_upper'] for p in pair_rows)))
    summaries=[]
    for variant in sorted({r['variant'] for r in rows}):
        selected=[r for r in rows if r['variant']==variant]
        summaries.append(dict(variant=variant,forecast_problems=len(selected),
            near_J_upper_range=[min(r['near_pair']['J_upper'] for r in selected),max(r['near_pair']['J_upper'] for r in selected)],
            near_J_required=selected[0]['near_pair']['J_required_when_h_below_tie'],
            near_upper_ratio_range=[min(r['near_pair']['upper_over_required'] for r in selected),max(r['near_pair']['upper_over_required'] for r in selected)],
            proved_infeasible=sum(r['any_nominal_pair_proved_infeasible'] for r in selected),
            prior_forecast_matches=sum(r['prior_forecast_bitwise_equal'] for r in selected)))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),source=reference(__file__),
        run=str(directory),protocol=protocol_ref,completion=complete_ref,inputs=inputs_ref,
        prior_protocol=previous_ref,prior_inputs=previous_inputs_ref,original_sources=original_sources,
        conditions=dict(no_truth=True,no_new_day_observations=True,no_optimization=True,no_AC=True,no_refit=True,
            historical_validation='Read only to instantiate the unchanged historical mean guards; guards are relaxed in this upper bound',
            bound='All P/Q resource power boxes; relax energy/time coupling and mean guards; each output may take its own favourable endpoint',
            scalar_necessity='For h<tie, h+J-z*sqrt(2J)>=tie requires J>=(z+sqrt(z*z-2*(h-tie)))**2/2',
            scope='Infeasibility of the declared frozen representative-model QCQP requirement; no physical impossibility or full-bank recovery claim',
            if_upper_sufficient='Passing an upper-bound check is only necessary and does not establish joint feasibility',
            solver_status='Post-hoc analytic finding does not rewrite original timeout/undecided solver statuses'),
        maximum_upper_identity_error=identity_error,summary=summaries,rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.run)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=reference(args.out),summary=result['summary'],
        maximum_upper_identity_error=result['maximum_upper_identity_error'])))


if __name__=='__main__':
    main()
