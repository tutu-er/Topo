"""Independent original-domain, same-witness and cross-action AC replays.

This reads saved predicted-mean fits only. No new fits, future noise, truth-
dependent selection, candidate removal or formal strategy evaluation occurs.
"""
from __future__ import annotations
import argparse,gzip,json,hashlib
import platform,sys,scipy
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from .likelihood import Observations,RadialCandidate
from .information_context import meter_from_record
from .ev_ac_moving_cross import build_from_saved,state_vector
from .ev_original_ac_check import (make_ac_context,evaluate_ac_witness,recenter_at_absolute,
    original_model_qp,fixed_absolute_coefficients,polynomial_residual,product_structure)
from .ev_original_domain_milp import (from_moving_model,evaluate_original_pair,
    fixed_absolute_witness,polynomial_residual as master_polynomial)

HERE=Path(__file__).resolve().parent
STATE=('p','q','source_p','source_q')


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path,'rt',encoding='utf-8-sig') as f:return json.load(f)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def fit_object(fit):
    return SimpleNamespace(**{k:np.asarray(fit[k],float) if k in ('r','x',*STATE) else fit[k]
        for k in ('r','x',*STATE,'loss','success')})


def physical_witness(fit,h):
    return np.r_[fit['r'],fit['x'],
        state_vector({k:np.asarray(fit[k])[:h] for k in STATE}),
        state_vector({k:np.asarray(fit[k])[h:] for k in STATE})]


def runtime(record):
    history=Observations(**record['history']);records={x['name']:x for x in record['historical_fits']}
    candidates=[RadialCandidate.from_supports(x['name'],history.n,x['supports']) for x in records.values()]
    return history,candidates,{k:fit_object(v) for k,v in records.items()},*(meter_from_record(record['measurement_models'][k]) for k in ('history','future'))


def normalize_model(model):
    result=dict(model)
    for k in ('c','B','A0','At','reference_kw','reference_residual','C_at_reference','future_root_v'):
        result[k]=np.asarray(result[k],float)
    result['domain']={k:np.asarray(v,float) if isinstance(v,list) else v for k,v in result['domain'].items()}
    result['future_background']={k:np.asarray(v,float) for k,v in result['future_background'].items()}
    return result


def compact(check):
    return {k:v for k,v in check.items() if k not in ('complete_AC_residual','proxy_residual','proxy_minus_AC','reference_future_state')}


def compare_proxy(model,u,xi,context,saved_cost):
    # Saved BVLS fits can differ from a bound by <1e-15. Record this explicit
    # numerical projection; reject any material excursion instead of hiding it.
    lo=np.asarray(model['domain']['physical_lower']);hi=np.asarray(model['domain']['physical_upper'])
    projected=np.clip(xi,lo,hi);projection=float(np.max(abs(projected-xi)))
    if projection>1e-9:raise ValueError('saved witness materially outside original box')
    xi=projected
    ac=evaluate_ac_witness(model,u,xi,context)
    qp=original_model_qp(model,u);original=qp['original'];outer=qp['outer']
    pair=from_moving_model(model);direct=evaluate_original_pair(pair,u)
    poly=fixed_absolute_coefficients(model,xi);other=fixed_absolute_witness(pair,xi)
    px=polynomial_residual(poly,u);mx=master_polynomial(other,u)
    return dict(AC=compact(ac),saved_fit_cost=float(saved_cost) if saved_cost is not None else None,
        AC_saved_cost_abs_error=abs(ac['complete_AC_cost']-saved_cost) if saved_cost is not None else None,
        explicit_box_roundoff_projection_max_abs=projection,
        polynomial_max_abs=float(np.max(abs(px-ac['proxy_residual']))),
        independent_polynomial_max_abs=float(np.max(abs(px-mx))),
        QP_original={k:original[k] for k in ('success','primal_upper','tangent_lower_numeric','physical_bound_violation')},
        QP_outer={k:outer[k] for k in ('success','primal_upper','tangent_lower_numeric','physical_bound_violation')},
        QP_absolute={k:direct[k] for k in ('success','primal_upper','tangent_lower_numeric','physical_bound_violation')},
        QP_absolute_vs_moving_abs_error=abs(direct['primal_upper']-original['primal_upper']),
        original_lower_minus_outer_upper=qp['original_lower_minus_outer_upper'],
        outer_primal_minus_original_primal=outer['primal_upper']-original['primal_upper'],
        fixed_legal_witness_proxy_cost_minus_original_QP=ac['proxy_cost']-original['primal_upper'])


def audit_v2(path):
    before=sha(path);record=read(path);rt=runtime(record);history=rt[0]
    guard=record['selection']['ac_guard'];baseline=guard['baseline'];proposal=guard['candidate']
    # Freeze baseline-only explanations before evaluating the distinct proposal.
    baseref={r['reference']:r for r in baseline['records']};rows=[];calls=0
    same=np.array_equal(baseline['u'],proposal['u'])
    for raw in record['moving_models']:
        model=normalize_model(raw);context=make_ac_context(model,*rt)
        anchor=next(f for f in baseref[model['a']]['fits'] if f['name']==model['b'])
        xi0=physical_witness(anchor,history.t);u0=np.asarray(baseline['u'],float)
        centered=recenter_at_absolute(model,u0,xi0,context);calls+=centered['expansion_ac_calls']
        for label,action in (('baseline',baseline),('proposal',proposal)):
            row=dict(action=label,a=model['a'],b=model['b'],witness_origin='saved_AC_predicted_joint_fit',
                is_distinct_holdout=bool(label=='proposal' and not same))
            try:
                fit=next(f for r in action['records'] if r['reference']==model['a'] for f in r['fits'] if f['name']==model['b'])
                xi=physical_witness(fit,history.t);u=np.asarray(action['u'],float)
                checked=compare_proxy(model,u,xi,context,fit['loss']);calls+=3
                recentered=evaluate_ac_witness(centered,u,xi,context);calls+=3
                row.update(status='complete',u_kw=u,saved_fit_success=fit['success'],original=checked,
                    baseline_centered_AC=compact(recentered),
                    center_H_cost=centered['expansion_history_cost'],
                    center_H_cost_minus_attained=centered['expansion_history_cost_minus_attained'],
                    original_box_unchanged=centered['original_physical_domain_unchanged'])
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,KeyError,StopIteration) as e:
                row.update(status='failure',failure=type(e).__name__+':'+str(e))
            rows.append(row)
        # These two saved legal proxy-QP explanations give nonconstant actions
        # even when the old selector reverted its proposal to constant. They
        # are point diagnostics, kept separate from saved nonlinear fit rows.
        for label in ('quality_lp','fixed_mle_milp'):
            u=np.asarray(next(r['u_kw'] for r in record['rows'] if r['method']==label),float)
            savedqp=next(r['original'] for r in record['selection']['curves_audit'][label]
                if r['a']==model['a'] and r['b']==model['b'])
            row=dict(action=label,a=model['a'],b=model['b'],witness_origin='saved_original_proxy_QP',
                is_distinct_holdout=not np.array_equal(u,u0))
            try:
                xi=np.asarray(savedqp['physical_parameter']);checked=compare_proxy(model,u,xi,context,None);calls+=3
                again=evaluate_ac_witness(centered,u,xi,context);calls+=3
                row.update(status='complete',u_kw=u,original=checked,baseline_centered_AC=compact(again),
                    center_H_cost=centered['expansion_history_cost'],
                    center_H_cost_minus_attained=centered['expansion_history_cost_minus_attained'],
                    original_box_unchanged=centered['original_physical_domain_unchanged'])
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,KeyError,StopIteration) as e:
                row.update(status='failure',failure=type(e).__name__+':'+str(e))
            rows.append(row)
    if sha(path)!=before:raise RuntimeError('frozen input changed')
    return dict(file=path.name,sha256=before,base_seed=record['base_seed'],quality=record['quality'],
        input_unchanged=True,rows=rows,AC_calls=calls,new_fits=0,new_selectors=0)


def audit_original54(audit_path,directory):
    saved=read(audit_path);trials=[];calls=0
    for oldtrial in saved['trials']:
        path=directory/oldtrial['file'];before=sha(path);record=read(path);rt=runtime(record);history=rt[0]
        models={};contexts={};centers={};rows=[]
        for raw in record['proxy_models']['models']:
            c=next(c for c in rt[1] if c.name==raw['b']);model=build_from_saved(raw,c,rt[-1]);calls+=2
            key=(raw['a'],raw['b']);models[key]=model;contexts[key]=make_ac_context(model,*rt)
            item=next(x for x in record['audit'] if x['method']=='constant' and (x['a'],x['b'])==key)
            fit=next(f for f in item['predicted_joint_fits'] if f['name']==raw['b'])
            u0=np.asarray(next(x['u_kw'] for x in record['rows'] if x['method']=='constant'))
            centers[key]=recenter_at_absolute(model,u0,physical_witness(fit,history.t),contexts[key]);calls+=centers[key]['expansion_ac_calls']
        oldrows={(x['method'],x['a'],x['b']):x for x in oldtrial['rows']}
        for item in record['audit']:
            if 'predicted_joint_fits' not in item:continue
            key=(item['a'],item['b']);old=oldrows[(item['method'],*key)]
            row=dict(method=item['method'],a=key[0],b=key[1],
                original54_future_gate_pass=old['metrics']['moving_cross']['future_channel_RMS_gate_pass'],
                original54_future_max_RMS=max(old['metrics']['moving_cross']['future_channel_RMS'].values()),
                is_distinct_holdout=item['method']!='constant')
            try:
                fit=next(f for f in item['predicted_joint_fits'] if f['name']==key[1]);xi=physical_witness(fit,history.t)
                u=np.asarray(next(x['u_kw'] for x in record['rows'] if x['method']==item['method']))
                checked=compare_proxy(models[key],u,xi,contexts[key],fit['loss']);calls+=3
                again=evaluate_ac_witness(centers[key],u,xi,contexts[key]);calls+=3
                row.update(status='complete',u_kw=u,original=checked,baseline_centered_AC=compact(again),
                    center_H_cost=centers[key]['expansion_history_cost'],
                    center_H_cost_minus_attained=centers[key]['expansion_history_cost_minus_attained'])
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,KeyError,StopIteration) as e:
                row.update(status='failure',failure=type(e).__name__+':'+str(e))
            rows.append(row)
        if before!=sha(path):raise RuntimeError('frozen input changed')
        trials.append(dict(file=path.name,sha256=before,input_unchanged=True,rows=rows))
    return dict(source_audit=str(audit_path.name),source_audit_sha256=sha(audit_path),
        original54_saved_pass=48,original54_saved_total=54,original54_saved_failure=6,
        trials=trials,AC_calls=calls,new_fits=0,new_selectors=0)


def summary(rows):
    complete=[r for r in rows if r['status']=='complete'];heldout=[r for r in complete if r['is_distinct_holdout']]
    def group(rs):
        return dict(rows=len(rs),original_strict_pass=sum(r['original']['AC']['strict_RMS_025_pass'] for r in rs),
            recentered_strict_pass=sum(r['baseline_centered_AC']['strict_RMS_025_pass'] for r in rs),
            median_original_max_channel_RMS=float(np.median([r['original']['AC']['max_channel_RMS'] for r in rs])) if rs else None,
            median_recentered_max_channel_RMS=float(np.median([r['baseline_centered_AC']['max_channel_RMS'] for r in rs])) if rs else None,
            max_original_channel_RMS=max((r['original']['AC']['max_channel_RMS'] for r in rs),default=0.),
            max_recentered_channel_RMS=max((r['baseline_centered_AC']['max_channel_RMS'] for r in rs),default=0.),
            all_AC_voltage_feasible=all(r['original']['AC']['complete_AC_voltage_feasible'] for r in rs),
            all_physical_box_feasible=all(r['original']['AC']['physical_box_feasible'] for r in rs))
    return dict(total=len(rows),complete=len(complete),failed=len(rows)-len(complete),all=group(complete),distinct_holdout=group(heldout),
        max_AC_cost_replay_abs_error=max((r['original']['AC_saved_cost_abs_error'] for r in complete if r['original']['AC_saved_cost_abs_error'] is not None),default=0.),
        max_explicit_box_roundoff_projection=max((r['original']['explicit_box_roundoff_projection_max_abs'] for r in complete),default=0.),
        max_polynomial_identity_abs_error=max((r['original']['polynomial_max_abs'] for r in complete),default=0.),
        max_independent_polynomial_abs_error=max((r['original']['independent_polynomial_max_abs'] for r in complete),default=0.),
        max_absolute_vs_moving_QP_abs_error=max((r['original']['QP_absolute_vs_moving_abs_error'] for r in complete),default=0.),
        max_outer_primal_minus_original=max((r['original']['outer_primal_minus_original_primal'] for r in complete),default=0.),
        max_outer_physical_violation=max((r['original']['QP_outer']['physical_bound_violation'] for r in complete),default=0.),
        max_original_lower_minus_outer_upper=max((r['original']['original_lower_minus_outer_upper'] for r in complete),default=0.),
        minimum_fixed_witness_cost_minus_QP=min((r['original']['fixed_legal_witness_proxy_cost_minus_original_QP'] for r in complete),default=0.))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy54',action='store_true');args=parser.parse_args();began=perf_counter()
    names=('ev_original_ac_check.py','ev_original_ac_audit.py','ev_original_domain_milp.py',
        'ev_ac_moving_cross.py','symmetric_ac_local.py','likelihood.py','gradient_price_study.py')
    sources={name:dict(sha256=sha(HERE/name),text=(HERE/name).read_text(encoding='utf-8')) for name in names}
    paths=[HERE/'evidence'/'ev_cross_all'/'v2'/f'trial_{seed}_{q}.json.gz' for seed in (987101,987102) for q in ('uniform','temporal')]
    trials=[audit_v2(path) for path in paths]
    out=dict(schema='original_absolute_AC_independent_audit_v1',scope='saved predicted means only; no fits, action choices or formal evaluation',
        predeclared_v2_seeds=[987101,987102],predeclared_conditions=['uniform','temporal'],trials=trials,
        v2_summary=summary([r for t in trials for r in t['rows']]),new_fits=0,new_selectors=0,
        prior54_status_retained=dict(pass_rows=48,failed_rows=6,total=54))
    if args.legacy54:
        out['original54']=audit_original54(HERE/'evidence'/'ev_ac_moving_cross'/'moving_cross_audit_v2.json.gz',HERE/'evidence'/'ev_profile_milp'/'v1')
        rows=[r for t in out['original54']['trials'] for r in t['rows']]
        out['original54_summary']=summary(rows)
        out['original54_summary']['prior_failed_rows_retained']=sum(not r['original54_future_gate_pass'] for r in rows)
    if any(sha(HERE/name)!=s['sha256'] for name,s in sources.items()):raise RuntimeError('source changed during audit; rerun stable source')
    out.update(seconds=perf_counter()-began,source_sha256={name:s['sha256'] for name,s in sources.items()},
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.output,'wt',encoding='utf-8') as f:json.dump(serial(out),f,indent=2,allow_nan=False)
    with gzip.open(args.output.with_name('source_snapshot.json.gz'),'wt',encoding='utf-8') as f:json.dump(sources,f,indent=2,allow_nan=False)
    print(json.dumps(serial({k:v for k,v in out.items() if k.endswith('summary')}),allow_nan=False))


if __name__=='__main__':main()
