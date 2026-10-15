"""Topology-free squared-power precision baseline on an EV operation MILP.

Q(u)=sum(normalized_inverse_voltage_variance*u**2), in weighted kW squared.
The shared-square master bounds this exact quadratic objective. No AC model,
training parameters, load forecast, true topology or future error is read.
Callers supply only predeclared topology-free initial curves and parent plans.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np

from .ev_original_domain_milp import (OriginalDomainPair, exchange_original_domain,
    _operation, _complete_operation)

DEFAULT_EXCHANGE = dict(max_iterations=6,control_segments=4,residual_segments=4,
    max_segments=16,solver_seconds=2.,total_seconds=35.,absolute_gap=.01,
    relative_gap=.01,witness_policy='affine_preferred',backend='gurobi',
    master_form='shared_control_square')


def precision_weights(future_meter,shape):
    """Same inverse-V-variance normalization as the existing linear baseline."""
    if len(shape)!=2 or any(int(v)!=v or v<1 for v in shape):
        raise ValueError('a nonempty (time,site) control shape is required')
    sigma=(future_meter.get('v_sigma') if isinstance(future_meter,Mapping)
        else getattr(future_meter,'v_sigma',None))
    try:sigma=np.broadcast_to(np.asarray(sigma,float),shape).copy()
    except (TypeError,ValueError) as exc:raise ValueError('V sigma must broadcast to control shape') from exc
    if not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('V sigma must be positive and finite')
    # Equivalent to (1/sigma**2)/max(1,max(1/sigma**2)), without overflow.
    least=float(sigma.min())
    return (least/sigma)**2 if least<=1. else (1./sigma)**2


def squared_precision_pair(weights):
    weights=np.asarray(weights,float)
    if weights.ndim!=2 or not weights.size or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('finite nonnegative time/site weights required')
    n=weights.size
    return OriginalDomainPair('squared_precision',np.zeros(n),np.diag(np.sqrt(weights.ravel())),
        np.zeros((n,0)),np.zeros((n,n,0)),np.zeros(0),np.zeros(0),np.zeros(0),np.zeros((0,n)))


def _initial_list(initial_powers,shape):
    if isinstance(initial_powers,Mapping):items=list(initial_powers.items())
    elif isinstance(initial_powers,np.ndarray) and initial_powers.shape in (shape,(int(np.prod(shape)),)):
        items=[('initial_0',initial_powers)]
    else:items=[('initial_'+str(i),value) for i,value in enumerate(initial_powers)]
    result=[]
    for name,value in items:
        if not isinstance(name,str) or not name:raise ValueError('initial curve names must be nonempty strings')
        if value is None:result.append((name,None));continue
        u=np.asarray(value,float)
        if u.shape==(int(np.prod(shape)),):u=u.reshape(shape)
        if u.shape!=shape or not np.isfinite(u).all():
            raise ValueError('initial curve must be finite with matching time/site dimensions: '+name)
        result.append((name,u.copy()))
    return result


def solve_squared_precision(domain,future_meter,initial_powers=(),exchange_options=None):
    """Return the best physically checked curve under its own exact Q score.

    Initial curves may be a sequence or a name->power mapping; unavailable and
    infeasible parents are recorded. Operation completion and exchange share the
    total 35-second default wall budget. The main master has the same 2-second
    default limit as discrimination. The caller audits initial-plan provenance.
    A feasible incumbent can be returned when the requested gap is unfinished.
    """
    began=perf_counter();shape=tuple(np.asarray(domain.indices.u).shape)
    weights=precision_weights(future_meter,shape);pair=squared_precision_pair(weights)
    options=dict(DEFAULT_EXCHANGE);options.update(exchange_options or {})
    owned={'operation','constraints','offsets','weights','aggregation','score_scale','score_caps',
        'initial_u','initial_controls','initial_operation_points','zero_product_pairs'}
    forbidden=owned&set(exchange_options or {})
    if forbidden:raise ValueError('squared precision owns exchange fields '+repr(sorted(forbidden)))
    if options.get('master_form')!='shared_control_square':
        raise ValueError('squared precision requires shared_control_square')
    total=float(options['total_seconds']);single=float(options['solver_seconds'])
    tolerance=float(options.get('feasibility_tolerance',1e-7))
    if not np.isfinite([total,single,tolerance]).all() or min(total,single)<=0 or tolerance<0:
        raise ValueError('positive finite budgets and nonnegative feasibility tolerance required')
    op_data=domain.as_dict()
    low=np.asarray(op_data['control_lower'],float);high=np.asarray(op_data['control_upper'],float)
    op=_operation(pair.control_dimension,low,high,None,op_data)
    initial_records=[];failures=[];candidates=[];points=[];seen=set();completion_calls=0
    def keep(x,origin):
        x=np.asarray(x,float).copy();audit=domain.check(x,tolerance=tolerance)
        if not audit.get('feasible'):raise ValueError('operation audit failed: '+origin)
        u=np.asarray(domain.decode(x)['u_kw'],float)
        if u.shape!=shape or not np.isfinite(u).all():raise ValueError('decoded control shape/nonfiniteness')
        if not np.allclose(op['U']@x,u.ravel(),rtol=0.,atol=1e-12):
            raise ValueError('domain decode and operation control map disagree')
        Q=float(np.sum(weights*u*u))
        candidate=dict(origin=origin,power_kw=u.copy(),Q=Q,operation=dict(success=True,x=x,audit=audit))
        candidates.append(candidate);return candidate
    for name,u in _initial_list(initial_powers,shape):
        if u is None:
            initial_records.append(dict(origin=name,success=False,reason='unavailable_parent'));continue
        key=np.ascontiguousarray(u,dtype='<f8').tobytes()
        if key in seen:
            initial_records.append(dict(origin=name,success=None,reason='duplicate_control'));continue
        seen.add(key);remaining=total-(perf_counter()-began)
        if remaining<=0:
            initial_records.append(dict(origin=name,success=False,reason='total_budget_exhausted'));continue
        try:
            realized=_complete_operation(u.ravel(),op,time_limit=min(single,remaining),
                tolerance=tolerance,backend=options['backend'])
            completion_calls+=realized.get('solver_calls',0)
            record=dict(origin=name,realized=realized,success=bool(realized['success']))
            if realized['success']:
                candidate=keep(realized['x'],'initial:'+name);points.append(candidate['operation']['x'])
                record.update(Q=candidate['Q'],power_kw=candidate['power_kw'],
                    max_requested_power_difference_kw=float(np.max(abs(candidate['power_kw']-u))))
            else:record['reason']='operation_completion_failed'
            initial_records.append(record)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            record=dict(origin=name,success=False,reason=type(exc).__name__+':'+str(exc))
            initial_records.append(record);failures.append(dict(kind='initial_operation_exception',**record))
    exchange=None;remaining=total-(perf_counter()-began)
    if remaining>0:
        try:
            exchange_kwargs=dict(options);exchange_kwargs['total_seconds']=remaining
            exchange=exchange_original_domain([pair],low,high,operation=op_data,
                offsets=np.zeros(1),weights=np.ones(1),aggregation='mean',score_scale=1.,
                initial_operation_points=points,**exchange_kwargs)
            # Exact Q has no source minimization. Compare every saved feasible
            # operation, rather than selecting by a chord incumbent surrogate.
            for i,row in enumerate(exchange.get('evaluations',[])):
                try:keep(row['operation_x'],'exchange_evaluation:'+str(i))
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    failures.append(dict(kind='exchange_operation_rejected',index=i,reason=type(exc).__name__+':'+str(exc)))
            if exchange.get('operation_x') is not None:
                try:keep(exchange['operation_x'],'exchange_selected')
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    failures.append(dict(kind='exchange_selected_rejected',reason=type(exc).__name__+':'+str(exc)))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures.append(dict(kind='exchange_exception',reason=type(exc).__name__+':'+str(exc)))
    else:failures.append(dict(kind='exchange_skipped_total_budget'))
    winner=max(candidates,key=lambda row:row['Q']) if candidates else None
    lower=upper=gap=None;completed=False
    if winner is not None:
        guard=256*np.finfo(float).eps*(1.+abs(winner['Q']))
        lower=max(0.,winner['Q']-guard)
        if exchange is not None:
            old_lower=exchange.get('objective_lower');old_upper=exchange.get('objective_upper')
            if old_lower is not None and np.isfinite(old_lower) and old_lower>winner['Q']+max(guard,1e-8):
                failures.append(dict(kind='backend_lower_above_checked_incumbent',backend_lower=float(old_lower),Q=winner['Q']))
            if old_upper is not None and np.isfinite(old_upper):
                upper=float(old_upper)
                if upper<lower:
                    failures.append(dict(kind='numerical_upper_below_attained_lower',upper=upper,lower=lower));upper=None
            if upper is not None:
                gap=upper-lower
                target=float(options['absolute_gap'])+float(options['relative_gap'])*max(1e-12,abs(upper),abs(lower))
                completed=bool(exchange.get('completed') and gap<=target)
    best_initial=max((row['Q'] for row in initial_records if row.get('success') is True and 'Q' in row),default=None)
    return dict(success=winner is not None,power_kw=None if winner is None else winner['power_kw'],
        Q=None if winner is None else winner['Q'],selected_origin=None if winner is None else winner['origin'],
        operation=dict(success=False,audit=dict(feasible=False)) if winner is None else winner['operation'],
        weights=weights,initials=initial_records,best_initial_Q=best_initial,
        inheritance_protected=winner is not None and (best_initial is None or winner['Q']>=best_initial),
        objective_lower=lower,objective_upper=upper,gap=gap,completed=completed,unfinished=not completed,
        status=exchange.get('status') if exchange is not None else 'exchange_unavailable',
        exchange=exchange,exchange_configuration=options,exchange_seconds_available=remaining,
        initial_operation_completion_calls=completion_calls,failures=failures,seconds=perf_counter()-began,
        objective='sum normalized inverse V variance times net grid kW squared; score_scale=1',
        scope='Exact quadratic precision objective and original operation MILP; shared-square numerical bounds, no AC distance.',
        information_inputs=['public voltage precision','operation domain','supplied topology-free initial curves'],
        uses_AC=False,uses_line_parameters=False,uses_forecast=False,uses_truth=False,uses_future_noise=False,
        source_coordinates=0,full_curve_enumeration=False)
