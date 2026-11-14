"""Offline R/X-response errors of the original 16 historical native winners.

No best-true-model substitution, fitting, optimization, new randomness, or AC
calculation. Original physical lines are used only as an offline reference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .resources import checked, read, reference
from ..measured_pq_flex_20261114.analyze import physical_response
from ..reactive_flex_20261114.diagnose import physical_coefficients


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),minimum=float(a.min()),median=float(np.median(a)),
        mean=float(a.mean()),maximum=float(a.max())) if len(a) else dict(n=0)


def error(matrix,physical):
    den=float(np.linalg.norm(physical)); num=float(np.linalg.norm(matrix-physical))
    return dict(fitted_frobenius=float(np.linalg.norm(matrix)),physical_frobenius=den,
        error_frobenius=num,relative_frobenius_error=num/den if den else None)


def historical_observation(source):
    path=checked(source)['path']
    with np.load(path,allow_pickle=False) as z:
        return dict(P=z['P'].copy(),Q=z['Q'].copy(),Y=z['Y'].copy(),labels=list(map(int,z['labels'])))


def run(main_run,base_kw=200.):
    main_run=Path(main_run).resolve()
    entries_ref=reference(main_run/'inputs.json')
    entries=read(entries_ref['path'])
    if len(entries)!=16 or len({e['id'] for e in entries})!=16:
        raise ValueError('This audit requires the original 16 distinct inputs')
    rows=[]; maximum_MSE_replay_error=0.
    for entry in entries:
        score_ref=checked(entry['score']); native_ref=checked(entry['native_copy'])
        score=read(score_ref['path']);native=read(native_ref['path'])
        winner=int(native['selected']['index'])
        if winner!=int(score['native_index']):
            raise ValueError('Historical winner differs between original records')
        H=np.asarray(score['native_bank'][winner]['H'],float)
        labels=list(map(int,score['labels'])); n=len(labels)
        physical=physical_response(entry['case'],labels,entry['physical_cache']['network_hash'])
        coefficients=physical_coefficients(entry['case'],labels)
        truth={tuple(sorted(labels[i] for i in group)) for group,value in coefficients.items()
               if 1<len(group)<n and np.max(value)>0}
        selected=set(map(tuple,score['native_bank'][winner]['active_clades']))
        training=historical_observation(entry['files']['train.npz'])
        validation=historical_observation(entry['files']['validation.npz'])
        if training['labels']!=labels or validation['labels']!=labels:
            raise ValueError('Historical terminal order changed')
        computed={}
        for name,obs in [('training',training),('validation',validation)]:
            power=np.c_[obs['P'],obs['Q']]
            fitted=power@H.T;linear=power@physical.T
            computed[name]=dict(points=len(power),cells=obs['Y'].size,
                frozen_winner_MSE=float(np.mean((obs['Y']-fitted)**2)),
                physical_linear_MSE_on_same_noisy_observations=float(np.mean((obs['Y']-linear)**2)),
                fitted_minus_physical_prediction_RMS=float(np.sqrt(np.mean((fitted-linear)**2))))
        frozen_row=next(m for m in score['all_frozen_points'] if m['index']==winner)
        for a,b in [(computed['training']['frozen_winner_MSE'],frozen_row['train_mse']),
                    (computed['validation']['frozen_winner_MSE'],frozen_row['validation_mse'])]:
            maximum_MSE_replay_error=max(maximum_MSE_replay_error,abs(a-b))
            if not np.isclose(a,b,rtol=1e-10,atol=1e-18):
                raise ArithmeticError('Historical SSE replay differs from frozen score')
        q=training['Q']*base_kw
        columns=[]
        for k,label in enumerate(labels):
            pcol,qcol=error(H[:,k],physical[:,k]),error(H[:,n+k],physical[:,n+k])
            columns.append(dict(node_label=label,P_column=pcol,Q_column=qcol,
                training_Q_mean_kvar=float(q[:,k].mean()),training_Q_std_kvar=float(q[:,k].std(ddof=1)),
                training_Q_RMS_kvar=float(np.sqrt(np.mean(q[:,k]**2))),
                training_Q_max_abs_kvar=float(np.abs(q[:,k]).max())))
        rows.append(dict(id=entry['id'],case=entry['case'],start_date=entry['start_date'],
            score=score_ref,native=native_ref,training=checked(entry['files']['train.npz']),
            validation=checked(entry['files']['validation.npz']),physical_network_hash=entry['physical_cache']['network_hash'],
            historical_native_winner=winner,no_oracle_model_substitution=True,
            internal_topology_correct=selected==truth,
            missing_internal_supports=[list(c) for c in sorted(truth-selected)],
            extra_internal_supports=[list(c) for c in sorted(selected-truth)],
            R_response=error(H[:,:n],physical[:,:n]),X_response=error(H[:,n:],physical[:,n:]),
            historical_scoring=computed,node_columns=columns))
    summaries={}
    for name,group in [('all',rows),('internal_topology_correct',[r for r in rows if r['internal_topology_correct']]),
                       ('internal_topology_wrong',[r for r in rows if not r['internal_topology_correct']])]:
        summaries[name]=dict(inputs=len(group),
            R_relative_frobenius_error=stats([r['R_response']['relative_frobenius_error'] for r in group]),
            X_relative_frobenius_error=stats([r['X_response']['relative_frobenius_error'] for r in group]),
            Q_columns_relative_error=stats([c['Q_column']['relative_frobenius_error'] for r in group for c in r['node_columns']]),
            training_MSE=stats([r['historical_scoring']['training']['frozen_winner_MSE'] for r in group]),
            validation_MSE=stats([r['historical_scoring']['validation']['frozen_winner_MSE'] for r in group]))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),source=reference(__file__),
        source_dependencies=[reference(Path(__file__).parents[1]/'measured_pq_flex_20261114'/'analyze.py'),
            reference(Path(__file__).parents[1]/'reactive_flex_20261114'/'diagnose.py')],
        inputs=entries_ref,maximum_historical_MSE_replay_error=maximum_MSE_replay_error,
        conditions=dict(original_16_historical_winners_only=True,no_oracle_selection=True,no_refit=True,
            no_optimizer=True,no_new_AC=True,no_new_random_draws=True,
            physical_lines_used_only_offline=True,no_new_day_observations_read=True,
            matrix_units='Squared-voltage P/Q response in the common per-unit system; not individual line ohms',
            noise='Training and validation MSE use saved noisy historical observations',
            limitation='Response errors quantify fitted prediction parameters; they do not by themselves isolate lack of excitation, measurement error, regularization, or model mismatch'),
        summary=summaries,rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run',type=Path,default=Path('outputs/measured_pq_dfs_main_20261114_v1'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.main_run)
    args.out.mkdir(parents=True,exist_ok=False)
    for name,data in [('population.json',result),('SUMMARY.json',{k:v for k,v in result.items() if k!='rows'})]:
        with (args.out/name).open('x',encoding='utf-8') as stream:
            json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=reference(args.out/'population.json'),summary=result['summary'],
        maximum_historical_MSE_replay_error=result['maximum_historical_MSE_replay_error'])))


if __name__=='__main__':
    main()
