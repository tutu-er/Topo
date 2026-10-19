"""Independent single-EV, exact time-block shared power-square upper master.

No frozen source is changed. Numerical floating bounds, not interval arithmetic.
"""
from __future__ import annotations
import hashlib,json,math
import numpy as np


def require_single_ev(fleet_size=1):
    if fleet_size!=1:raise ValueError('only the single-EV operations model is supported')


def certify_time_block_support(B,n_slots,n_sites,*,fleet_size=1):
    """Sufficient exact-zero support certificate; never threshold a near-zero."""
    require_single_ev(fleet_size)
    B=np.asarray(B,float)
    if B.ndim!=2 or B.shape[1]!=n_slots*n_sites or not np.isfinite(B).all():
        raise ValueError('finite time-major B with T*n_sites columns required')
    row_times=[]
    for row in B:
        support=np.flatnonzero(row!=0.0)
        times=np.unique(support//n_sites)
        if len(times)>1:raise ValueError('non-block time support: compression rejected without deleting coefficients')
        row_times.append(int(times[0]) if len(times) else None)
    return {'certificate':'each residual row has exact nonzero support in at most one time block',
            'row_times':row_times,'zero_rows':sum(t is None for t in row_times),
            'd':np.sum(B*B,axis=0),'n_slots':int(n_slots),'n_sites':int(n_sites),
            'near_zero_coefficients_removed':False}


def zero_knots(lower,upper,segments):
    if int(segments)!=segments or segments<1:raise ValueError('positive integer segments required')
    if not(math.isfinite(lower) and math.isfinite(upper)) or lower>0 or upper<0 or lower>upper:
        raise ValueError('finite bounds must contain zero')
    m=int(segments)
    if lower==upper:return np.array([0.0])
    if lower<0<upper:
        if m<2 or m%2:raise ValueError('even segments required for signed bounds containing zero')
        return np.r_[np.linspace(lower,0,m//2+1),np.linspace(0,upper,m//2+1)[1:]]
    return np.linspace(lower,upper,m+1)


def secant_value(u,knots):
    nodes=np.asarray(knots,float)
    if len(nodes)==1:
        if u!=nodes[0]:raise ValueError('fixed power differs from bound')
        return float(u*u)
    if u<nodes[0] or u>nodes[-1]:raise ValueError('power outside knot domain')
    k=min(max(int(np.searchsorted(nodes,u,side='right'))-1,0),len(nodes)-2)
    a,b=nodes[k:k+2]
    return float((a+b)*u-a*b)


def coefficients(pair,z,n_slots,n_sites,*,fleet_size=1):
    audit=certify_time_block_support(pair.B,n_slots,n_sites,fleet_size=fleet_size)
    z=np.asarray(z,float)
    if z.shape!=pair.lower.shape or not np.isfinite(z).all() or np.any(z<pair.lower) or np.any(z>pair.upper):
        raise ValueError('fixed feasible nuisance witness required')
    r=pair.c+pair.A@z
    constant=float(r@r+np.sum((pair.F@z)**2))
    linear=2*pair.B.T@r
    return constant,linear,audit['d'],audit


def check_control(u,n_slots,n_sites,*,fleet_size=1):
    require_single_ev(fleet_size)
    u=np.asarray(u,float).reshape(n_slots,n_sites)
    if not np.isfinite(u).all() or np.any(np.count_nonzero(u!=0.0,axis=1)>1):
        raise ValueError('single EV requires at most one nonzero site per time')
    return u.ravel()


def decomposed_cost(pair,z,u,n_slots,n_sites,*,fleet_size=1):
    flat=check_control(u,n_slots,n_sites,fleet_size=fleet_size)
    c,l,d,_=coefficients(pair,z,n_slots,n_sites,fleet_size=fleet_size)
    return float(c+l@flat+d@(flat*flat))


def secant_error_bounds(d,knots,n_slots,n_sites):
    widths=np.array([max(np.diff(k),default=0.0) for k in knots])
    coordinate=np.asarray(d)*widths**2/4
    return {'general':float(coordinate.sum()),'single_ev_physical':float(coordinate.reshape(n_slots,n_sites).max(axis=1).sum())}


def _backend():
    # Caller controls which frozen checkout is on sys.path; no source mutation.
    from research_experiments.mobile_ev_mle.ev_operations_milp import operation_model,decode_plan
    from research_experiments.mobile_ev_mle.ev_information_milp import control_box,proxy_box_upper
    return operation_model,decode_plan,control_box,proxy_box_upper


def _shared_graph(model,u_indices,lower,upper,segments):
    square=model.add_variables('shared_power_square',len(u_indices),0,np.maximum(lower*lower,upper*upper))
    knots=[];graph_variables=0
    for j,(uid,lo,hi) in enumerate(zip(u_indices,lower,upper)):
        nodes=zero_knots(float(lo),float(hi),segments);knots.append(nodes)
        sid=int(square[j])
        if len(nodes)==1:continue
        m=len(nodes)-1
        weights=model.add_variables('shared_power_weights_'+str(j),(m,2),0,1)
        binary=model.add_variables('shared_power_segment_'+str(j),m,0,1,integer=True)
        model.add_row({int(i):1 for i in binary},1,1)
        model.add_row({int(i):1 for i in weights.ravel()},1,1)
        power={int(uid):-1.0};sq={sid:-1.0}
        for k in range(m):
            model.add_row({int(weights[k,0]):1,int(weights[k,1]):1,int(binary[k]):-1},upper=0)
            for endpoint in range(2):
                index=int(weights[k,endpoint]);node=float(nodes[k+endpoint])
                power[index]=node;sq[index]=node*node
        model.add_row(power,0,0);model.add_row(sq,0,0)
        graph_variables+=3*m
    return square,knots,graph_variables+len(square)


def _json_number(v):
    return float(v) if math.isfinite(v) else '+inf' if v>0 else '-inf'


def model_snapshot(model,objective,metadata):
    snapshot={'schema':'shared_power_square_linear_milp_v1','cmax':True,
              'lower':[_json_number(v) for v in model.lower],'upper':[_json_number(v) for v in model.upper],
              'integer':list(model.integer),'rows':[{str(k):v for k,v in sorted(row.items())} for row in model.rows],
              'row_lower':[_json_number(v) for v in model.row_lower],'row_upper':[_json_number(v) for v in model.row_upper],
              'objective':{str(k):v for k,v in objective.items()},'metadata':metadata}
    encoded=json.dumps(snapshot,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return snapshot,hashlib.sha256(encoded).hexdigest()


def build_shared_upper_model(config,pairs,witnesses,*,segments=4,fleet_size=1):
    require_single_ev(fleet_size)
    pairs=list(pairs)
    if not pairs or len(pairs)!=len(witnesses) or any(not group for group in witnesses):
        raise ValueError('nonempty feasible witness group for every pair required')
    T,n=config.n_slots,config.n_sites
    # Certify every B before building any shared-square constraints.
    for pair in pairs:certify_time_block_support(pair.B,T,n,fleet_size=fleet_size)
    operation_model,_,control_box,proxy_box_upper=_backend()
    model,indices=operation_model(config);lower,upper=control_box(config)
    for lo,hi in zip(lower,upper):zero_knots(float(lo),float(hi),segments)
    qbar=proxy_box_upper(pairs,lower,upper,witnesses)
    if not math.isfinite(qbar) or qbar<0:raise ValueError('finite nonnegative qbar required')
    scale=max(1.0,qbar)
    objective_index=int(model.add_variables('shared_squared_score_scaled',(),0,qbar/scale))
    operation_variables=model.n_variables-1
    square,knots,shared_variables=_shared_graph(model,indices.u_index.ravel(),lower,upper,segments)
    errors=[];structure=[];maxpower=np.maximum(abs(lower),abs(upper))
    for k,pair in enumerate(pairs):
        for s,z in enumerate(witnesses[k]):
            constant,linear,d,audit=coefficients(pair,z,T,n,fleet_size=fleet_size)
            # Numerical roundoff guard only; not an interval-arithmetic proof.
            guard=128*np.finfo(float).eps*(1+abs(constant)+np.sum(abs(linear)*maxpower)+np.sum(d*maxpower**2))
            row={objective_index:1.0}
            row.update({int(i):-float(v)/scale for i,v in zip(indices.u_index.ravel(),linear) if v})
            row.update({int(i):-float(v)/scale for i,v in zip(square,d) if v})
            model.add_row(row,upper=(constant+guard)/scale)
            errors.append({'pair':pair.name,'witness':s,'numeric_rhs_guard':guard,**secant_error_bounds(d,knots,T,n)})
            if s==0:structure.append({'pair':pair.name,'exact_block_support_verified':True,'zero_rows':audit['zero_rows'],'d':d.tolist()})
    metadata={'n_slots':T,'n_sites':n,'segments':int(segments),'fleet_size':1,'qbar':qbar,'scale':scale,
              'objective_index':objective_index,'control_indices':indices.u_index.tolist(),'square_indices':square.tolist(),
              'knots':[k.tolist() for k in knots],'structure':structure,'secant_errors':errors,
              'operation_variables':operation_variables,'shared_graph_variables':shared_variables}
    snapshot,digest=model_snapshot(model,{objective_index:1.0},metadata)
    return dict(model=model,indices=indices,objective_index=objective_index,scale=scale,qbar=qbar,
                square_indices=square,knots=knots,secant_errors=errors,n_variables=model.n_variables,n_rows=len(model.rows),
                operation_variables=operation_variables,shared_graph_variables=shared_variables,snapshot=snapshot,snapshot_sha256=digest)


def score_from_result(result,scale):
    incumbent=None if result.objective_value is None else float(result.objective_value*scale)
    bound=None if result.objective_bound is None else float(result.objective_bound*scale)
    if bound is not None and not math.isfinite(bound):bound=None
    return {'q_incumbent':incumbent,'q_proxy_upper':bound}


def upper_master_shared(config,pairs,witnesses,*,segments=4,time_limit=30,mip_rel_gap=1e-8,fleet_size=1):
    built=build_shared_upper_model(config,pairs,witnesses,segments=segments,fleet_size=fleet_size)
    result=built['model'].solve({built['objective_index']:1.0},time_limit=time_limit,mip_rel_gap=mip_rel_gap)
    _,decode_plan,_,_=_backend()
    plan=None if result.x is None else decode_plan(result,built['indices'])
    return {**built,**score_from_result(result,built['scale']),'result':result,'plan':plan,
            'bound_scope':'full frozen affine proxy numerical upper, from maximization global best bound; no nonlinear AC certificate',
            'incumbent_is_full_proxy_upper':False}
