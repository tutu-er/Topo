"""Distribution-based frozen-score procurement; no actual-network inputs."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import norm

from ..absolute_identification_20261114.solver import seed_items
from ..flexibility_extended_solve_20261114.solver import _Recorder, _number, _reference
from .evidence import DistributionEvidence

BUFFER = 1e-8


def quadratic_box_upper(Q, a, c, lower, upper):
    """Interval upper bound, including negative diagonals; no domain shrinking."""
    Q, a = np.asarray(Q, np.longdouble), np.asarray(a, np.longdouble)
    lower, upper = np.asarray(lower, np.longdouble), np.asarray(upper, np.longdouble)
    if Q.ndim == 2:
        Q = np.broadcast_to(Q, (len(lower), *Q.shape))
    value, absolute = np.longdouble(c), abs(np.longdouble(c))
    for h in range(len(lower)):
        lo, hi = lower[h], upper[h]
        linear = np.maximum(2*a[h]*lo, 2*a[h]*hi)
        square_lo = np.where((lo <= 0) & (hi >= 0), 0, np.minimum(lo*lo, hi*hi))
        square_hi = np.maximum(lo*lo, hi*hi)
        diagonal = np.maximum(np.diag(Q[h])*square_lo, np.diag(Q[h])*square_hi)
        value += linear.sum()+diagonal.sum()
        absolute += abs(linear).sum()+abs(diagonal).sum()
        for i in range(len(lo)):
            for j in range(i+1, len(lo)):
                products = np.array([lo[i]*lo[j], lo[i]*hi[j], hi[i]*lo[j], hi[i]*hi[j]])
                term = max((Q[h,i,j]+Q[h,j,i])*products)
                value += term; absolute += abs(term)
    pad = 1e-10*max(1., float(absolute))
    return dict(upper=float(value)+pad, roundoff_padding=pad,
                scope='Componentwise interval outer bound on original resource amplitude box; energy constraints relaxed')


def direct_audit(case, evidence, u):
    data = evidence.verify_expansion(u)
    domain = case.audit(u)
    margins = np.asarray(data['margins'], float)
    return dict(resource_audit=domain, evidence=data,
        passed_acceptance=bool(domain['passed'] and data['feasible']),
        minimum_buffered_margin=float(np.min(margins-BUFFER)) if len(margins) else None,
        strict_buffered_nonnegative=bool(np.all(margins >= BUFFER)),
        acceptance_scope='Original direct probability margins >= -1e-7, failure bound <= allocated alpha+1e-9, retained scalar guards and original physical resource tolerance; encoded buffer 1e-8')


def _finish(recorder, receipt, initial=None, final=None):
    receipt.update(native_attributes=recorder.native_attributes,
        native_attribute_unavailable=recorder.unavailable_attributes,
        native_attribute_invalid=recorder.invalid_attributes,
        initial_buffered_direct_audit=initial, buffered_direct_audit=final,
        solver_source=_reference(__file__),
        evidence_source=_reference(Path(__file__).with_name('evidence.py')),
        raw_bound_scope='Activation cost for this exact encoded buffered distribution QCQP and retained scalar guards, subject to native numerical tolerances; no original chance-problem/AC/total-bill bound',
        global_optimality_scope='Only encoded fixed-law procurement; no AC reliability or true parameter coverage guarantee')
    recorder.event(dict(event='finish', wall_seconds=perf_counter()-recorder.began,
        status=receipt.get('status'), source=receipt.get('source'), cost_yuan=receipt.get('cost_yuan'),
        lower_bound_yuan=receipt.get('lower_bound_yuan')))
    if recorder.stream is not None:
        recorder.stream.close(); recorder.stream = None
    files = {}
    for name, path in (('progress', recorder.trace_path), ('native_log', recorder.native_path)):
        if path is not None and path.exists(): files[name] = _reference(path)
    receipt.update(record_directory=None if recorder.directory is None else str(recorder.directory),
        diagnostic_files=files, trace_rows=recorder.trace_rows, logging_errors=recorder.errors,
        native_log_configured=recorder.native_log_configured,
        trace_status='not_requested' if recorder.directory is None else 'recording_errors' if recorder.errors else 'complete',
        seconds=perf_counter()-recorder.began)
    if recorder.directory is not None:
        try:
            path = recorder.directory/'diagnostics.json'
            with path.open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False,
                    default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())
            receipt['diagnostics_receipt'] = _reference(path)
        except Exception as exc:
            recorder.error('write_diagnostics', exc)
            receipt.update(trace_status='recording_errors', logging_errors=recorder.errors)
    receipt['seconds'] = perf_counter()-recorder.began
    return receipt


def _expression(u, Q, a, c, hours):
    Q = np.asarray(Q)
    return sum(u[h]@(Q if Q.ndim == 2 else Q[h])@u[h] for h in range(hours))+2*(a*u).sum()+float(c)


def plan(case, score, validation, forecast, base_kw, *, mode='mgf_forecast',
         seconds=600., env, seed_points=None, target_mass=.99, record_directory=None):
    if seconds <= 0 or not np.isfinite(seconds):
        raise ValueError('Positive finite total quote budget required')
    began = perf_counter(); recorder = _Recorder(record_directory, began)
    evidence = DistributionEvidence(case,score,validation,forecast,base_kw,mode=mode,target_mass=target_mass)
    zero = np.zeros(case.shape); baseline = evidence.verify_expansion(zero)
    common = dict(mode=mode, objective_kind='activation', objective_cost_object='activation_yuan',
        bound_cost_object='activation_yuan', targets=evidence.targets, gate_rows=evidence.rows,
        gate_count=len(evidence.rows), guard_count=len(evidence.guards), baseline=baseline,
        target_mass=target_mass, numerical_gate_buffer=BUFFER,
        constrained_bank='Chosen frozen representative versus all original native points in other internal-topology families',
        original_nominal_gates_retained=False, original_scalar_guards_retained=True,
        time_scope='Evidence construction, seed checks, model construction, solve and receipts included; old common preparation reported by caller',
        confidence_scope='Conditional on fixed historical calibrated mean and declared fixed Gaussian law; nominal delivery, no actual-AC or random-delivery guarantee')
    if case.audit(zero)['passed'] and baseline['feasible']:
        receipt=dict(common,feasible=True,status='zero_sufficient',source='zero',cost_yuan=0.,
            lower_bound_yuan=0.,relative_gap=0.,global_optimality_verified=True,optimizer_invoked=False)
        return _finish(recorder,receipt,final=direct_audit(case,evidence,zero)),zero
    box=[]
    if mode != 'normal_fixed':
        for k in range(len(evidence.rows)):
            bound=quadratic_box_upper(evidence.gate_q[k],evidence.gate_a[k],evidence.gate_c[k],case.lower,case.upper)
            box.append(dict(row=k,**bound,impossible=bool(bound['upper'] < -1e-7)))
    common['resource_box_checks']=box
    if any(r['impossible'] for r in box):
        receipt=dict(common,feasible=False,status='infeasible_by_probability_box_bound',optimizer_invoked=False,
            native_status=None,infeasibility_proved=True,cost_yuan=None,lower_bound_yuan=None,
            infeasibility_scope='Only this fixed-lambda sufficient QCQP; not the original probability event or alternate lambda',
            global_optimality_verified=False)
        return _finish(recorder,receipt),None
    point=warm=None;chosen=None;merit=-np.inf;records=[];tick=perf_counter()
    for name,value in seed_items(seed_points):
        try:
            u=case.array(value);domain=case.audit(u);info=evidence.verify_expansion(u);cost=float(case.fee(u))
            if not np.isfinite(cost) or cost < -1e-9:raise ValueError('Invalid activation fee')
            valid=bool(domain['passed'] and info['feasible'])
            margin=min([*np.asarray(info['margins']).tolist(),*np.asarray(info['guard_gains']).tolist()],default=0.)
            records.append(dict(name=name,resource_feasible=bool(domain['passed']),qcqp_feasible=valid,
                cost_yuan=cost,minimum_margin=margin))
            if valid and (point is None or cost<case.fee(point)):point,chosen=u.copy(),name
            if domain['passed'] and margin>merit:warm,merit=u.copy(),margin
        except Exception as exc:records.append(dict(name=name,error=repr(exc),qcqp_feasible=False))
    if point is not None:warm=point.copy()
    seed=None if point is None else point.copy()
    initial=dict(source='same_forecast_frozen_common_pool',feasible=seed is not None,name=chosen,
        cost_yuan=None if seed is None else float(case.fee(seed)),audit_seconds=perf_counter()-tick,candidates=records)
    initial_audit=None if seed is None else direct_audit(case,evidence,seed)
    import gurobipy as gp
    with gp.Model('frozen_distribution_selection_'+mode,env=env) as model:
        for key,value in dict(OutputFlag=0,Threads=1,Seed=0,NonConvex=2,
                FeasibilityTol=1e-9,NumericFocus=1,MIPGap=1e-4).items():model.setParam(key,value)
        recorder.configure(model);u,fee=case.attach(model)
        for k,row in enumerate(evidence.rows):
            if mode=='normal_fixed':
                variance=_expression(u,evidence.variance_q[k],evidence.variance_a[k],evidence.variance_c[k],case.H)
                mu=_expression(u,evidence.mu_q[k],evidence.mu_a[k],evidence.mu_c[k],case.H)
                vb=quadratic_box_upper(evidence.variance_q[k],evidence.variance_a[k],evidence.variance_c[k],case.lower,case.upper)
                s=model.addVar(lb=0.,ub=np.sqrt(max(0.,vb['upper']))+1e-8,name=f'std_{k}')
                model.addConstr(variance<=s*s,name=f'variance_{k}')
                model.addConstr(mu>=float(norm.isf(row['alpha']))*s+BUFFER,name=f'normal_{k}')
                if warm is not None:
                    value=(np.einsum('hi,hij,hj->',warm,evidence.variance_q[k],warm)
                           +2*np.sum(evidence.variance_a[k]*warm)+evidence.variance_c[k])
                    s.Start=float(np.sqrt(max(0.,value)))
            else:
                value=_expression(u,evidence.gate_q[k],evidence.gate_a[k],evidence.gate_c[k],case.H)
                model.addConstr(value>=BUFFER,name=f'mgf_{k}')
        for k,(_,_,_,Q,a) in enumerate(evidence.guards):
            model.addConstr(_expression(u,Q,a,0.,case.H)>=0.,name=f'history_direction_{k}')
        model.setObjective(fee)
        if warm is not None:u.Start=warm
        model.Params.TimeLimit=max(.001,seconds-(perf_counter()-began))
        model.optimize(recorder.callback) if recorder.directory is not None else model.optimize()
        recorder.capture(model)
        try:
            bound=_number(model.ObjBound,gp.GRB.INFINITY)
            reason=None if bound is not None else 'nonfinite_or_solver_infinity'
        except (AttributeError,gp.GurobiError):bound,reason=None,'native_ObjBound_unavailable'
        source='verified_initial_retained' if seed is not None else None;native_audit=None
        if model.SolCount:
            candidate=np.asarray(u.X);native_audit=direct_audit(case,evidence,candidate)
            if native_audit['passed_acceptance'] and (point is None or case.fee(candidate)<case.fee(point)-1e-9):
                point=candidate;source='native_found' if seed is None else 'native_improved_initial'
        common.update(optimizer_invoked=True,native_returned_point_audit=native_audit)
        if point is None:
            receipt=dict(common,feasible=False,status='native_infeasible' if model.Status==3 else 'unresolved',
                native_status=int(model.Status),initial=initial,infeasibility_proved=model.Status==3,
                cost_yuan=None,lower_bound_yuan=bound,relative_gap=None,global_optimality_verified=False,
                native_bound_unavailable_reason=reason)
        else:
            cost=float(case.fee(point));conflict=bound is not None and bound>cost+1e-7*max(1.,cost)
            lower=0. if bound is None or conflict else max(0.,min(cost,bound));gap=(cost-lower)/max(cost,1e-12)
            receipt=dict(common,feasible=True,status='feasible',source=source,native_status=int(model.Status),
                initial=initial,cost_yuan=cost,lower_bound_yuan=lower,relative_gap=gap,
                lower_bound_withdrawn=conflict,native_infeasibility_withdrawn=model.Status==3,
                lower_bound_withdrawal_reason='conflicts_verified_incumbent' if conflict else None,
                native_bound_unavailable_reason=reason,
                lower_bound_source='nonnegative_cost_universal_bound' if bound is None or conflict or bound<0 else 'native_ObjBound',
                global_optimality_verified=gap<=1e-4,
                improvement_from_verified_initial_yuan=None if seed is None else float(case.fee(seed)-cost),
                final=evidence.verify_expansion(point),resource_audit=case.audit(point))
    final=None if point is None else direct_audit(case,evidence,point)
    return _finish(recorder,receipt,initial_audit,final),point
