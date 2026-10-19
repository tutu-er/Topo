"""Independent complete-AC residual replay of the two physical histories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .likelihood import Observations,RadialCandidate,ac_predict
from .information_context import meter_from_record
from .ev_physical_spatial_fit_audit import read,write,serial

HERE=Path(__file__).resolve().parent


def replay_fit(record,fit,observations,meter):
    candidate=RadialCandidate.from_supports(fit['name'],observations.n,fit['supports'])
    if [list(s) for s in candidate.supports]!=fit['supports']:raise ValueError('line order changed')
    physical=ac_predict(candidate,fit['r'],fit['x'],fit['p'],fit['q'],fit['source_p'],fit['source_q'],observations.root_v)
    costs=np.zeros(observations.t);normalizer=0.;pred_errors={}
    for channel in ('p','q','v','master_p','master_q'):
        observed=getattr(observations,channel);sigma=np.broadcast_to(np.asarray(getattr(meter,channel+'_sigma')),np.asarray(observed).shape)
        bias=getattr(meter,'bias_'+channel) if channel in ('p','q','v') else 0.
        mean=physical[channel]+bias
        error=(mean-observed)/sigma
        costs+=np.sum(error.reshape(observations.t,-1)**2,axis=1)
        normalizer+=float(np.sum(np.log(2*np.pi*sigma*sigma)))
        pred_errors[channel]=float(np.max(abs(mean-np.asarray(fit['predicted'][channel]))))
    total=float(costs.sum());loglik=-.5*(total+normalizer)
    h=record['config']['history_count']
    restart=fit['diagnostics']['restarts']
    hfit=next(f for f in record['historical_fits'] if f['name']==fit['name'])
    return dict(name=fit['name'],full_AC_cost=total,saved_cost=float(fit['loss']),
        cost_replay_abs_error=abs(total-float(fit['loss'])),loglik_replay_abs_error=abs(loglik-float(fit['loglik'])),
        history_cost=float(costs[:h].sum()),future_cost=float(costs[h:].sum()),
        H_cost_minus_original_local=float(costs[:h].sum()-float(hfit['loss'])),
        predicted_mean_replay_max_abs=max(pred_errors.values()),predicted_channel_errors=pred_errors,
        ac_equation_residual=physical['ac_equation_residual'],power_balance_residual=physical['power_balance_residual'],
        saved_fit_success=fit['success'],restarts=len(restart),converged_restarts=sum(r['success'] for r in restart),
        failed_restarts=[r for r in restart if not r['success']],
        shared_rank=fit['diagnostics']['local_shared_rank'],shared_count=fit['diagnostics']['shared_parameter_count'],
        active_bounds=fit['diagnostics']['active_bound_count'],source_interface=fit['diagnostics']['source_node'])


def review_history(path):
    before=hashlib.sha256(Path(path).read_bytes()).hexdigest();record=read(path);meter=meter_from_record(record['meter'])
    rows=[]
    hobs=Observations(**record['history'])
    for fit in record['historical_fits']:rows.append(dict(scenario='H',**replay_fit(record,fit,hobs,meter)))
    for row in record['rows']:
        observations=Observations(**row['joint_observations'])
        for fit in row['fits']:rows.append(dict(scenario=row['scenario'],**replay_fit(record,fit,observations,meter)))
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=before:raise ValueError('frozen history modified')
    source=[]
    for name,expected in record['source_sha256'].items():
        target=(HERE/name).resolve();actual=hashlib.sha256(target.read_bytes()).hexdigest()
        source.append(dict(name=name,expected=expected,actual=actual,match=expected==actual))
    return dict(seed=record['seed'],file=Path(path).name,input_sha256=before,input_bytes_unchanged=True,
        rows=rows,source_checks=source,all_sources_match=all(s['match'] for s in source))


def summarize(trials):
    rows=[r for t in trials for r in t['rows']]
    return dict(independent_histories=len(trials),saved_candidate_fits=len(rows),replayed_AC_calls=len(rows),
        successful_saved_fits=sum(r['saved_fit_success'] for r in rows),candidate_fit_denominator=len(rows),
        optimizer_restarts=sum(r['restarts'] for r in rows),converged_restarts=sum(r['converged_restarts'] for r in rows),
        failed_restarts=sum(r['restarts']-r['converged_restarts'] for r in rows),
        max_cost_replay_error=max(r['cost_replay_abs_error'] for r in rows),
        max_loglik_replay_error=max(r['loglik_replay_abs_error'] for r in rows),
        max_prediction_replay_error=max(r['predicted_mean_replay_max_abs'] for r in rows),
        max_AC_equation_residual=max(r['ac_equation_residual'] for r in rows),
        max_power_balance_residual=max(r['power_balance_residual'] for r in rows),
        source_interfaces=sorted({r['source_interface'] for r in rows}),
        joint_H_cost_below_attained_H_rows=sum(r['scenario']!='H' and r['H_cost_minus_original_local']<-1e-6 for r in rows),
        max_shared_rank_deficiency=max(r['shared_count']-r['shared_rank'] for r in rows),
        all_source_hashes_match=all(t['all_sources_match'] for t in trials),
        new_fits=0,new_selectors=0,formal_risk_calibration=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();began=perf_counter()
    trials=[review_history(p) for p in sorted(args.directory.glob('history_*.json.gz'))]
    if not trials:raise ValueError('no frozen physical histories')
    result=dict(schema='independent_complete_AC_physical_spatial_replay_v1',summary=summarize(trials),
        trials=trials,seconds=perf_counter()-began,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(args.output,result);print(json.dumps(serial(result['summary']),ensure_ascii=False))


if __name__=='__main__':main()
