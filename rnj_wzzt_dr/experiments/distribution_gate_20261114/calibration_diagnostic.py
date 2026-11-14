"""Two fixed historical P/Q scale fits and retrospective clean-score checks.

The fit stage opens only frozen scoring matrices and historical validation.
The evaluation stage requires the sealed coefficient receipt before opening
old executions. No procurement, AC, topology fit, or hyperparameter selection.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np

from ... import io
from ...observations import load
from ..flexibility_sweep_20261113.fitting import calibrated_means

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_OUT=ROOT/'outputs/distribution_gate_pq_calibration_diagnostic_20261114_v1'
RUNS={
    'old_confirmation': ('flexibility_stability_confirmation_20261114_v1', ('no_dfs','anchor','capacity'),64),
    'rx_anchor': ('rx_mean_guard_formal_20261114_v1', ('no_dfs','anchor'),32),
    'line': ('distribution_gate_line_formal_20261114_v1', ('no_dfs','anchor','normal_fixed'),32),
}


def checked(ref):
    if io.ref(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('Changed sealed input: '+ref['path'])
    return Path(ref['path'])


def entry_for(run):
    return next(e for e in io.read(run/'inputs.json') if e['resource_variant']=='original_p10_q36')


def fit(out):
    out=Path(out)
    if (out/'coefficients_sealed.json').exists():
        raise FileExistsError('Preserve sealed coefficients')
    source=ROOT/'outputs'/RUNS['rx_anchor'][0]
    entry=entry_for(source)
    score=io.read(checked(entry['score']));val=load(checked(entry['files']['validation.npz']))
    H=np.asarray([r['H'] for r in score['parameters']],float)
    K,T,N=len(H),val.rows,val.n
    sigma2=float(score['sigma2'])
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        task='Single historical-validation comparison of two fixed P/Q row-scaling recipes',
        sources=[io.ref(__file__)],inputs=[entry['score'],entry['files']['validation.npz']],
        rule_A='One aP,aQ per representative; stack all time/terminal cells; ridge lambda=T*N*sigma_score2',
        rule_B='One aP,aQ per representative/output row; ridge lambda=T*sigma_score2',
        formula='a=(Z.T@Z+lambda*I)^(-1)@(Z.T@y+lambda*ones), Z=[P@R.T,Q@X.T]',
        intercept=False,clipping=False,hyperparameter_search=False,
        fitting_data='Same 96 historical validation points only; no future or network parameter inputs',
        scoring='All native H, historical SSE, score sigma2 and final selection unchanged',
        controls='All predeclared old confirmation anchor/capacity, old RX anchor and completed line controls; no current distribution QCQP run',
        evaluation_runs={key:dict(directory=str(ROOT/'outputs'/value[0]),policies=list(value[1]),repeats=value[2],forecasts=4)
                         for key,value in RUNS.items()},
        primary_error='Mean prediction minus clean score using the same realized clean P/Q, isolating mean-response error',
        secondary_error='Same mean prediction on requested forecast P/Q versus stored clean score, includes forecast/delivery error',
        gain_error='Predicted change from paired zero action minus actual clean score change',
        calibration_voltage_MSE_scope='In-sample historical validation fit diagnostic, not a generalization score',
        optimizer_calls=0,AC_calls=0,topology_or_impedance_refits=0)
    io.write(out/'protocol.json',protocol)
    Ppart=np.einsum('tc,knc->ktn',val.p,H[:,:,:N])
    Qpart=np.einsum('tc,knc->ktn',val.q,H[:,:,N:])
    shared=np.empty((K,2));row=np.empty((K,N,2));checks=[]
    for i in range(K):
        Z=np.c_[Ppart[i].ravel(),Qpart[i].ravel()];y=val.y.ravel()
        lam=T*N*sigma2
        shared[i]=np.linalg.solve(Z.T@Z+lam*np.eye(2),Z.T@y+lam*np.ones(2))
        residual=Z.T@(Z@shared[i]-y)+lam*(shared[i]-1)
        checks.append(float(np.linalg.norm(residual)/max(1.,np.linalg.norm(Z.T@y))))
        for n in range(N):
            Z=np.c_[Ppart[i,:,n],Qpart[i,:,n]];y=val.y[:,n];lam=T*sigma2
            row[i,n]=np.linalg.solve(Z.T@Z+lam*np.eye(2),Z.T@y+lam*np.ones(2))
            residual=Z.T@(Z@row[i,n]-y)+lam*(row[i,n]-1)
            checks.append(float(np.linalg.norm(residual)/max(1.,np.linalg.norm(Z.T@y))))
    A=H.copy();A[:,:,:N]*=shared[:,0,None,None];A[:,:,N:]*=shared[:,1,None,None]
    B=H.copy();B[:,:,:N]*=row[:,:,0,None];B[:,:,N:]*=row[:,:,1,None]
    old=calibrated_means(score,val)
    means=np.stack([H,old,A,B])
    variants=['frozen_H','old_scalar','shared_PQ','row_PQ']
    if not np.isfinite(means).all():raise ArithmeticError('Nonfinite calibration')
    powers=np.c_[val.p,val.q]
    prediction=np.einsum('tc,vknc->vktn',powers,means)
    fit_mse=np.mean(np.square(prediction-val.y[None,None]),axis=(2,3))
    native_indices=np.array([p['index'] for p in score['parameters']],int)
    np.savez_compressed(out/'coefficients.npz',shared=shared,row=row,means=means,
        variant_names=np.asarray(variants),native_indices=native_indices,labels=np.asarray(val.labels),
        calibration_mse=fit_mse)
    io.write(out/'coefficients_sealed.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol=io.ref(out/'protocol.json'),coefficients=io.ref(out/'coefficients.npz'),
        score=entry['score'],validation=entry['files']['validation.npz'],native_indices=native_indices,
        variants=variants,maximum_normal_equation_residual=max(checks),
        future_or_network_opened=False,source_sha256=io.ref(__file__)['sha256']))
    print(dict(status='coefficients_sealed',variants=variants,maximum_normal_equation_residual=max(checks)))


def score_difference(y,power,H_i,H_j,h,sigma2):
    return float(h+(np.square(y-power@H_j.T).sum()-np.square(y-power@H_i.T).sum())/(2*sigma2))


def predicted_difference(power,K,H_i,H_j,h,sigma2):
    return score_difference(power@K.T,power,H_i,H_j,h,sigma2)


def action_map(entry,labels):
    B=np.zeros((2*len(labels),len(entry['p_resources'])+len(entry['q_resources'])))
    for k,r in enumerate(entry['p_resources']):B[labels.index(r['node_label']),k]=1.
    for k,r in enumerate(entry['q_resources'],len(entry['p_resources'])):B[len(labels)+labels.index(r['node_label']),k]=1.
    return B


def evaluate(out):
    out=Path(out)
    if (out/'complete.json').exists():raise FileExistsError('Preserve completed diagnostic')
    seal=io.read(out/'coefficients_sealed.json');checked(seal['protocol']);checked(seal['coefficients'])
    if io.ref(__file__)['sha256']!=seal['source_sha256']:raise ValueError('Diagnostic source changed after fitting')
    score=io.read(checked(seal['score']));sigma2=float(score['sigma2'])
    native=np.asarray([p['H'] for p in score['native_bank']],float)
    hist=np.array([p['history_sse'] for p in score['native_bank']])
    with np.load(checked(seal['coefficients']),allow_pickle=False) as z:
        means=z['means'].copy();variants=z['variant_names'].tolist();indices=z['native_indices'].tolist();labels=z['labels'].tolist()
    references=[io.ref(out/'coefficients_sealed.json')];line_checks=[];records=[];forecast_predictions=[]
    max_score_identity=0.;began=perf_counter()
    # These target indices are the frozen historical target set. No true model
    # is used to choose either calibration coefficients or evaluation actions.
    targets=[2,4]
    for group,(run_name,policies,repeats) in RUNS.items():
        run=ROOT/'outputs'/run_name;complete=io.read(run/'complete.json')
        if complete['status']!='complete':raise ValueError('Old execution must be complete')
        entry=entry_for(run)
        if entry['score']['sha256']!=seal['score']['sha256']:raise ValueError('Different frozen score')
        references.extend([io.ref(run/'complete.json'),io.ref(run/'inputs.json')])
        Bmap=action_map(entry,labels)
        for draw in range(4):
            d=run/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            planned=io.read(d/'seal.json');planner=checked(planned['planner'])
            references.extend([io.ref(d/'seal.json'),planned['planner']])
            with np.load(planner,allow_pickle=False) as z:
                forecast=z['forecast'].copy();controls={p:z[p].copy() for p in policies}
            if group=='line':
                source=ROOT/'outputs/distribution_gate_line_polish_diagnostic_20261114_v1/candidate_points.npz'
                with np.load(source,allow_pickle=False) as z:
                    expected=z[f'forecast{draw:02d}_normal_fixed']
                    error=float(np.max(abs(expected-controls['normal_fixed'])))
                if error!=0:raise ArithmeticError('Executed line point differs from independently saved diagnostic')
                line_checks.append(dict(draw=draw,maximum_absolute_action_difference=error))
                references.append(io.ref(source))
            for policy,u in controls.items():
                pforecast=forecast+np.repeat(u@Bmap.T/200.,4,axis=0)
                for v,name in enumerate(variants):
                    for i in targets:
                        j=4 if i==2 else 2;r=indices.index(i);h=(hist[j]-hist[i])/(2*sigma2)
                        forecast_predictions.append(dict(group=group,draw=draw,policy=policy,variant=name,
                            target_native=i,competitor_native=j,
                            predicted_L=predicted_difference(pforecast,means[v,r],native[i],native[j],h,sigma2)))
            for repeat in range(repeats):
                base_path=d/f'execution{repeat:02d}'/'no_dfs'/'observations.npz'
                references.append(io.ref(base_path))
                with np.load(base_path,allow_pickle=False) as z:
                    x0=np.c_[z['true_P'],z['true_Q']];y0=z['true_root'][:,None]**2-z['true_V']**2
                for policy in policies:
                    path=d/f'execution{repeat:02d}'/policy/'observations.npz';references.append(io.ref(path))
                    with np.load(path,allow_pickle=False) as z:
                        x=np.c_[z['true_P'],z['true_Q']];y=z['true_root'][:,None]**2-z['true_V']**2
                        requested=z['requested_action'].copy()
                    if not np.array_equal(requested,controls[policy]):raise ArithmeticError('Execution request differs from sealed plan')
                    xf=forecast+np.repeat(requested@Bmap.T/200.,4,axis=0)
                    for i in targets:
                        j=4 if i==2 else 2;r=indices.index(i);h=(hist[j]-hist[i])/(2*sigma2)
                        actual=score_difference(y,x,native[i],native[j],h,sigma2)
                        actual0=score_difference(y0,x0,native[i],native[j],h,sigma2)
                        D=native[i]-native[j]
                        for v,name in enumerate(variants):
                            K=means[v,r];pred=predicted_difference(x,K,native[i],native[j],h,sigma2)
                            pred0=predicted_difference(x0,K,native[i],native[j],h,sigma2)
                            pred_forecast=predicted_difference(xf,K,native[i],native[j],h,sigma2)
                            projected=float(np.sum((x@K.T-y)*(x@D.T))/sigma2)
                            max_score_identity=max(max_score_identity,abs((pred-actual)-projected))
                            records.append(dict(group=group,draw=draw,repeat=repeat,policy=policy,variant=name,
                                target_native=i,competitor_native=j,actual_clean_L=actual,
                                predicted_L_given_realized_PQ=pred,score_error=pred-actual,
                                predicted_L_given_requested_forecast=pred_forecast,forecast_score_error=pred_forecast-actual,
                                actual_clean_gain=actual-actual0,predicted_gain=pred-pred0,
                                gain_error=(pred-pred0)-(actual-actual0),
                                mean_voltage_response_MSE=float(np.mean(np.square(x@K.T-y))),
                                action_voltage_response_MSE=float(np.mean(np.square((x-x0)@K.T-(y-y0))))))
    if max_score_identity>1e-8:raise ArithmeticError('Independent full-SSE/projection identity failed')
    summary=[]
    for group,(run_name,policies,repeats) in RUNS.items():
        for policy in policies:
            for i in targets:
                for variant in variants:
                    rows=[r for r in records if (r['group'],r['policy'],r['target_native'],r['variant'])==(group,policy,i,variant)]
                    summary.append(dict(group=group,policy=policy,target_native=i,variant=variant,paths=len(rows),
                        mean_actual_clean_L=float(np.mean([r['actual_clean_L'] for r in rows])),
                        mean_predicted_L=float(np.mean([r['predicted_L_given_realized_PQ'] for r in rows])),
                        mean_score_error=float(np.mean([r['score_error'] for r in rows])),
                        mean_absolute_score_error=float(np.mean([abs(r['score_error']) for r in rows])),
                        mean_gain_error=float(np.mean([r['gain_error'] for r in rows])),
                        mean_absolute_gain_error=float(np.mean([abs(r['gain_error']) for r in rows])),
                        mean_forecast_score_error=float(np.mean([r['forecast_score_error'] for r in rows])),
                        mean_absolute_forecast_score_error=float(np.mean([abs(r['forecast_score_error']) for r in rows])),
                        mean_voltage_response_MSE=float(np.mean([r['mean_voltage_response_MSE'] for r in rows])),
                        action_voltage_response_MSE=float(np.mean([r['action_voltage_response_MSE'] for r in rows])),
                        clean_target_beats_near_count=sum(r['actual_clean_L']>0 for r in rows),
                        forecast_prediction_beats_near_count=sum(r['predicted_L_given_requested_forecast']>0 for r in rows)))
    references=list({r['path']:r for r in references}.values())
    io.write(out/'evaluation_manifest.json',references)
    io.write(out/'results.json',dict(summary=summary,rows=records,forecast_predictions=forecast_predictions,
        maximum_score_projection_identity_error=max_score_identity,line_action_checks=line_checks,
        interpretation='Retrospective calibration diagnostic under unchanged frozen native scores; no new planning or final-selection change',
        independent_new_day_claimed=False))
    io.write(out/'complete.json',dict(status='complete',coefficients_seal=io.ref(out/'coefficients_sealed.json'),
        results=io.ref(out/'results.json'),manifest=io.ref(out/'evaluation_manifest.json'),seconds=perf_counter()-began,
        source_unchanged=io.ref(__file__)['sha256']==seal['source_sha256'],
        input_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in references),
        optimizer_calls=0,AC_calls=0,parameter_fits='Only two declared historical scalar scale recipes; frozen H unchanged'))
    for row in summary:
        if row['target_native']==4 and row['policy']!='no_dfs':print(row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('fit','evaluate'),required=True)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    a=p.parse_args();(fit if a.stage=='fit' else evaluate)(a.out.resolve())


if __name__=='__main__':main()
