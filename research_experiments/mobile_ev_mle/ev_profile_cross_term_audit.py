"""Post-hoc four-corner diagnostic at saved nonlinear competitor witnesses.

No fitting, action selection, simulation of noise, or modification of frozen
records occurs. The audit explains a fixed proxy at saved refitted states; it
does not establish causation or a global nonlinear optimum.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .likelihood import Observations, RadialCandidate
from .information_context import meter_from_record
from .symmetric_ac_local import mean_and_jacobian, whiten_observations

HERE = Path(__file__).resolve().parent
STATE = ('p', 'q', 'source_p', 'source_q')


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as handle:
        return json.load(handle)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name):
    result = np.asarray(value, float)
    if not np.isfinite(result).all():
        raise ValueError(name + ' contains nonfinite values')
    return result


def state(record, time_slice=slice(None)):
    return {key: finite(record[key], key)[time_slice] for key in STATE}


def state_vector(values):
    return np.column_stack([values[key] for key in STATE]).ravel()


def decomposition(m00, m10, m01, m11, jtheta, jstate, dtheta, dstate):
    """Exact algebraic split of the first-order Taylor remainder.

    R_theta and R_state each vary one coordinate block. R_cross measures
    their interaction. Norms can cancel and are not additive contributions.
    """
    m00, m10, m01, m11 = [finite(v, 'mean') for v in (m00, m10, m01, m11)]
    if not (m00.ndim == 1 and m00.shape == m10.shape == m01.shape == m11.shape):
        raise ValueError('four matching mean vectors required')
    jt, jw = finite(jtheta, 'line Jacobian'), finite(jstate, 'state Jacobian')
    dt, dw = finite(dtheta, 'line step'), finite(dstate, 'state step')
    if jt.shape != (len(m00), len(dt)) or jw.shape != (len(m00), len(dw)):
        raise ValueError('Jacobian and step dimensions differ')
    terms = dict(theta=m10-m00-jt@dt, state=m01-m00-jw@dw,
                 cross=m11-m10-m01+m00)
    total = m11-m00-jt@dt-jw@dw
    terms['total'] = total
    closure = total-sum(terms[key] for key in ('theta', 'state', 'cross'))
    return dict(vectors=terms, norms={key:float(np.linalg.norm(v)) for key,v in terms.items()},
        pairwise_inner_products={a+'__'+b:float(terms[a]@terms[b])
            for a,b in (('theta','state'),('theta','cross'),('state','cross'))},
        closure_norm=float(np.linalg.norm(closure)),
        closure_max_abs=float(np.max(np.abs(closure), initial=0.)),
        norm_sum=float(sum(np.linalg.norm(terms[key]) for key in ('theta','state','cross'))))


def _corners(candidate, theta0, theta1, w0, w1, root_v, meter, channels, geometry, counter):
    def forward(theta, w):
        counter['ac_forward_attempts'] += 1
        answer = mean_and_jacobian(candidate, theta[:candidate.m], theta[candidate.m:],
            w['p'], w['q'], w['source_p'], w['source_q'], root_v, meter,
            channels=channels, with_jacobian=False)
        counter['ac_forward_completed'] += 1
        return answer['mean']
    m00, m10, m01, m11 = [forward(theta,w) for theta,w in
        ((theta0,w0),(theta1,w0),(theta0,w1),(theta1,w1))]
    value = decomposition(m00,m10,m01,m11,geometry['line_jacobian'],geometry['state_jacobian'],
        theta1-theta0,state_vector(w1)-state_vector(w0))
    value['center_mean_replay_max_abs'] = float(np.max(abs(m00-finite(geometry['mean'],'saved mean'))))
    value['actual_response_norm'] = float(np.linalg.norm(m11-m00))
    value['state_step_norm'] = float(np.linalg.norm(state_vector(w1)-state_vector(w0)))
    return value, m11


def audit_row(record, audit, model, counter):
    by_name = {f['name']:f for f in record['historical_fits']}
    candidates = {f['name']:f for f in audit.get('predicted_joint_fits',[])}
    if model['b'] not in candidates:
        raise ValueError('missing saved nonlinear competitor fit')
    fit0, fit1 = by_name[model['b']], candidates[model['b']]
    n = len(record['history']['p'][0]); h = len(record['history']['p'])
    candidate = RadialCandidate.from_supports(model['b'],n,fit0['supports'])
    if [list(s) for s in candidate.supports] != fit0['supports'] or fit1['supports'] != fit0['supports']:
        raise ValueError('saved line order differs from reconstructed candidate')
    theta0 = np.r_[fit0['r'],fit0['x']]; theta1 = np.r_[fit1['r'],fit1['x']]
    channels = tuple(model['channels'])
    wh0, wh1 = state(fit0), state(fit1,slice(0,h))
    wn0, wn1 = state(model['forecast_b']), state(fit1,slice(h,None))
    u = finite(next(r['u_kw'] for r in record['rows'] if r['method']==audit['method']), 'curve')
    hist, hmean = _corners(candidate,theta0,theta1,wh0,wh1,record['history']['root_v'],
        meter_from_record(record['measurement_models']['history']),channels,model['geometry_b']['history'],counter)
    future, nmean = _corners(candidate,theta0,theta1,wn0,wn1,model['forecast_b']['root_v'],
        meter_from_record(record['measurement_models']['future']),channels,model['geometry_b']['future'],counter)
    hobs = whiten_observations(Observations(**record['history']),
        meter_from_record(record['measurement_models']['history']),channels)
    nobs = whiten_observations(Observations(**audit['predicted_observations']),
        meter_from_record(record['measurement_models']['future']),channels)
    center = np.r_[theta0,state_vector(wh0),state_vector(wn0)]
    witness = np.r_[theta1,state_vector(wh1),state_vector(wn1)]
    full = model['full']; delta = witness-center
    if not np.allclose(center,full['parameter_center'],rtol=0,atol=1e-12):
        raise ValueError('reconstructed parameter center differs from saved proxy')
    lin = finite(full['c'],'c')+finite(full['B'],'B')@u+finite(full['C'],'C')@delta
    exact = np.r_[hmean-hobs,nobs-nmean]
    source_affine = finite(model['geometry_a']['future']['mean'],'source mean')+finite(full['B'],'B')[len(hobs):]@(u-finite(model['reference_kw'],'reference'))
    source_error = nobs-source_affine
    remainder = np.r_[hist['vectors']['total'],source_error-future['vectors']['total']]
    closure = exact-lin-remainder
    loss = float(exact@exact); saved_loss=float(fit1['loss'])
    lower=finite(full['lower'],'lower'); upper=finite(full['upper'],'upper')
    lo=center+lower; hi=center+upper; line_count=len(theta0)
    norms=dict(theta_step=float(np.linalg.norm(theta1-theta0)),
        theta_step_relative_to_center=float(np.linalg.norm(theta1-theta0)/max(np.linalg.norm(theta0),1e-15)),
        theta_max_abs_step=float(np.max(abs(theta1-theta0),initial=0.)),
        state_step=float(np.linalg.norm(delta[line_count:])))
    # Keep residual vector evidence so cancellation and squared-cost identities
    # can be replayed without another nonlinear call.
    return dict(status='complete',method=audit['method'],a=model['a'],b=model['b'],u_kw=u.tolist(),
        saved_competitor_fit_success=bool(fit1['success']),
        saved_joint_failures=audit.get('predicted_joint_failures',{}),
        history=hist,future=future,parameter_step=norms,
        theta_center=theta0.tolist(),theta_witness=theta1.tolist(),theta_delta=(theta1-theta0).tolist(),
        theta_absolute_lower=lo[:line_count].tolist(),theta_absolute_upper=hi[:line_count].tolist(),
        theta_active_lower=np.flatnonzero(abs(witness[:line_count]-lo[:line_count])<=1e-8).tolist(),
        theta_active_upper=np.flatnonzero(abs(witness[:line_count]-hi[:line_count])<=1e-8).tolist(),
        full_bound_violation=float(max(0.,np.max(lower-delta),np.max(delta-upper))),
        original_source_forward_error_norm=audit.get('source_forward_linear_error_norm'),
        source_forward_response_norm=audit.get('source_forward_response_norm'),
        source_forward_relative_error=audit.get('source_forward_relative_linear_error'),
        source_forward_error_norm=float(np.linalg.norm(source_error)),
        nonlinear_competitor_cost_at_saved_witness=loss,saved_nonlinear_competitor_cost=saved_loss,
        nonlinear_cost_replay_abs_error=abs(loss-saved_loss),
        linear_competitor_cost_at_same_witness=float(lin@lin),
        bounded_local_primal=audit.get('bounded',{}).get('primal_upper'),
        bounded_local_lower=audit.get('bounded',{}).get('tangent_lower_numeric'),
        free_local_cost=audit.get('bounded',{}).get('free_value'),
        residual_closure_norm=float(np.linalg.norm(closure)),
        residual_closure_max_abs=float(np.max(abs(closure),initial=0.)),
        residual_remainder_norm=float(np.linalg.norm(remainder)),
        linear_residual=lin.tolist(),nonlinear_residual=exact.tolist(),source_forward_error=source_error.tolist(),
        squared_cost_difference=float(loss-lin@lin),
        squared_cost_difference_from_remainder=float(2*lin@remainder+remainder@remainder),
        nonlinear_cost_is_global_minimum=False)


def audit_trial(path):
    path=Path(path); before=sha(path); record=read_json(path); counter=dict(ac_forward_attempts=0,ac_forward_completed=0)
    context=record.get('proxy_models') or {}; models={(m['a'],m['b']):m for m in context.get('models',[])}
    rows=[]; skipped=[]
    for item in record.get('audit',[]):
        identity={k:item[k] for k in ('method','a','b')}
        if 'predicted_joint_fits' not in item:
            skipped.append(dict(**identity,reason='no_saved_nonlinear_predicted_fit'));continue
        try:
            result=audit_row(record,item,models[(item['a'],item['b'])],counter)
            rows.append(result)
        except (ValueError,RuntimeError,FloatingPointError,KeyError,np.linalg.LinAlgError) as error:
            rows.append(dict(**identity,status='audit_failure',error=type(error).__name__+': '+str(error)))
    after=sha(path)
    if before!=after:raise ValueError('input trial bytes changed during read-only audit')
    source_check=[]
    for name,expected in record['source_sha256'].items():
        target=(HERE/name).resolve()
        actual=sha(target) if target.is_file() else None
        source_check.append(dict(name=name,expected_sha256=expected,actual_sha256=actual,match=actual==expected))
    return dict(file=path.name,file_sha256=before,seed=record['seed'],truth_r=record['truth_r'],fit_floor=record['fit_floor'],
        truth_in_parameter_domain=record['truth_in_parameter_domain'],
        original_source_sha256=record['source_sha256'],source_current_check=source_check,
        source_current_all_match=all(x['match'] for x in source_check),rows=rows,skipped=skipped,**counter,
        fitting_calls=0,optimizer_calls=0,selector_calls=0,input_bytes_unchanged=True)


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:serial(v) for key,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def summarize(trials):
    rows=[r for t in trials for r in t['rows'] if r['status']=='complete']
    norms=lambda scope,key:[r[scope]['norms'][key] for r in rows]
    grouped={}
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        grouped[method]=dict(rows=len(group),
            future_median_norms={key:float(np.median([r['future']['norms'][key] for r in group]))
                for key in ('theta','state','cross','total')},
            median_linear_cost_at_nonlinear_witness=float(np.median([r['linear_competitor_cost_at_same_witness'] for r in group])),
            median_nonlinear_cost_at_same_witness=float(np.median([r['nonlinear_competitor_cost_at_saved_witness'] for r in group])),
            median_source_relative_error=(float(np.median([r['source_forward_relative_error'] for r in group if r['source_forward_relative_error'] is not None]))
                if any(r['source_forward_relative_error'] is not None for r in group) else None))
    return dict(trials=len(trials),rows=len(rows),failures=sum(r['status']!='complete' for t in trials for r in t['rows']),
        skipped_rows=sum(len(t['skipped']) for t in trials),
        ac_forward_attempts=sum(t['ac_forward_attempts'] for t in trials),
        ac_forward_completed=sum(t['ac_forward_completed'] for t in trials),
        new_fitting_calls=0,new_optimizer_calls=0,new_selector_calls=0,
        max_decomposition_closure=max((r[s]['closure_max_abs'] for r in rows for s in ('history','future')),default=0.),
        max_residual_closure=max((r['residual_closure_max_abs'] for r in rows),default=0.),
        max_cost_replay_abs_error=max((r['nonlinear_cost_replay_abs_error'] for r in rows),default=0.),
        median_future_cross_norm=float(np.median(norms('future','cross'))) if rows else None,
        median_future_theta_norm=float(np.median(norms('future','theta'))) if rows else None,
        median_future_state_norm=float(np.median(norms('future','state'))) if rows else None,
        median_source_forward_error=float(np.median([r['source_forward_error_norm'] for r in rows])) if rows else None,
        rows_future_cross_larger_than_both_pure_remainders=sum(r['future']['norms']['cross']>max(r['future']['norms']['theta'],r['future']['norms']['state']) for r in rows),
        median_theta_relative_step=float(np.median([r['parameter_step']['theta_step_relative_to_center'] for r in rows])) if rows else None,
        max_theta_relative_step=max((r['parameter_step']['theta_step_relative_to_center'] for r in rows),default=0.),
        max_parameter_bound_violation=max((r['full_bound_violation'] for r in rows),default=0.),
        by_method=grouped,
        all_original_source_hashes_match=all(t['source_current_all_match'] for t in trials))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=HERE/'evidence'/'ev_profile_milp'/'v1')
    parser.add_argument('--pattern',default='trial_986101_*.json.gz')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--output',type=Path,default=HERE/'evidence'/'ev_profile_milp'/'cross_term_diagnostic.json')
    args=parser.parse_args(); began=perf_counter()
    paths=sorted(args.directory.glob(args.pattern))
    if args.limit is not None:paths=paths[:args.limit]
    if not paths:raise ValueError('no matching saved trials')
    trials=[audit_trial(path) for path in paths]
    result=dict(schema='post_hoc_four_corner_competitor_diagnostic_v1',declared_after_first_results=True,
        scope='saved nonlinear predicted-mean fits only; no new optimization; not predeclared validation',
        identities=dict(total='m11-m00-Jtheta*dtheta-Jstate*dstate',
            theta='m10-m00-Jtheta*dtheta',state='m01-m00-Jstate*dstate',cross='m11-m10-m01+m00',
            cost='r_AC=r_linear+[R_history,source_forward_error-R_future]'),
        norms_are_not_additive_contribution_fractions=True,summary=summarize(trials),trials=trials,
        mean_and_residual_units='dimensionless; each meter divided by its declared standard deviation',
        audit_source_sha256={name:sha(HERE/name) for name in ('ev_profile_cross_term_audit.py','symmetric_ac_local.py','likelihood.py','information_context.py')},
        seconds=perf_counter()-began)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(serial(result),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':main()
