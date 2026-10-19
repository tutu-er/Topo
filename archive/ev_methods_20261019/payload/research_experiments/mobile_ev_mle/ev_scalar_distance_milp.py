"""Scalar source-profile L2 chord MILPs for frozen affine two-model geometry.

Bounds apply to the frozen affine residual, never nonlinear AC.  Three nodes
reuse the original charge/discharge variables with no new binary variables.
Finer grids use adjacent SOS2 weights linked to actual connection arc binaries.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds,LinearConstraint
from scipy.sparse import csr_matrix,hstack

from .ev_original_domain_milp import OriginalDomainPair,evaluate_original_pair,_solve_mip
from . import ev_fixed_train_design as fixed


def problem_identity(context,domain):
    """Bind every table/certificate to its frozen residual and operation model."""
    digest=sha256(repr(context['control_shape']).encode())
    for pair in context['pairs']:
        digest.update(pair.name.encode())
        for key in ('c','B','A0','At','physical_lower','physical_upper','base_center','shift'):
            value=np.asarray(getattr(pair,key),dtype='<f8')
            digest.update(repr(value.shape).encode());digest.update(value.tobytes())
    for name,line in sorted(context.get('lines',{}).items()):
        digest.update(name.encode())
        for key in ('r','x'):digest.update(np.asarray(line[key],dtype='<f8').tobytes())
    op=domain.as_dict()
    for key in ('lower','upper','integrality','u_map'):
        value=np.asarray(op[key],dtype='<f8')
        digest.update(repr(value.shape).encode());digest.update(value.tobytes())
    for con in op['constraints']:
        matrix=csr_matrix(con.A)
        for value in (matrix.data,matrix.indices,matrix.indptr,np.asarray(con.lb),np.asarray(con.ub)):
            arr=np.asarray(value,dtype='<f8');digest.update(repr(arr.shape).encode());digest.update(arr.tobytes())
    return digest.hexdigest()


def scalar_profile(c,b,G,lower,upper,center,p,*,power_range=None):
    """Feasible primal upper and arbitrary-residual Fenchel dual lower.

    Residual is c+b*p+G*(s-center), q=.5 min ||residual||².  The dual affine
    function remains a lower bound for every p; its residual need not be optimal.
    """
    c,b,G=(np.asarray(a,float) for a in (c,b,G))
    lower,upper,center=(np.asarray(a,float) for a in (lower,upper,center))
    m,k=G.shape
    pair=OriginalDomainPair('scalar',c,b[:,None],G,np.zeros((1,m,k)),
        lower,upper,center,np.zeros((k,1)))
    qp=evaluate_original_pair(pair,np.array([float(p)]))
    residual=qp['residual'];d=c-G@center;coefficient=G.T@residual
    support_min=float(coefficient@np.where(coefficient>=0,lower,upper))
    intercept=float(residual@d-.5*(residual@residual)+support_min)
    slope=float(b@residual)
    pscale=max(abs(float(p)),*(abs(float(v)) for v in ((p,p) if power_range is None else power_range)))
    guard=512*np.finfo(float).eps*(1+abs(residual@d)+.5*abs(residual@residual)+
        float(abs(coefficient)@np.maximum(abs(lower),abs(upper)))+abs(slope)*pscale)
    primal=.5*float(qp['primal_upper']);dual=max(0.,intercept+slope*p-guard)
    if dual>primal+1e-9*(1+abs(primal)):raise RuntimeError('numerical scalar dual exceeds primal')
    return dict(success=bool(qp['success']),p=float(p),primal_upper=primal,dual_lower=dual,
        dual_intercept=intercept-guard,dual_slope=slope,dual_numeric_guard=guard,
        primal_dual_gap=max(0.,primal-dual),physical_source=qp['physical_parameter'],
        residual=residual,source_QP=qp,curvature_upper=float(b@b),
        scope='half-SSE source-profile frozen affine residual; numerical guard recorded')


def _time_parts(context):
    if not context.get('success') or len(context.get('bank',()))!=2 or len(context.get('pairs',()))!=2:
        raise ValueError('successful two-candidate fixed-affine context required')
    T,n=context['control_shape'];m=T*(n+2);parts=[]
    for pair in context['pairs']:
        if (pair.c.shape!=(m,) or pair.B.shape!=(m,T*n) or pair.A0.shape!=(m,2*T)
                or np.any(pair.At) or np.any(pair.shift)):
            raise ValueError('At=shift=0 and independent two-source coordinates required')
        blocks=[]
        for t in range(T):
            rows=np.r_[np.arange(t*n,(t+1)*n),T*n+t,T*n+T+t]
            controls=np.arange(t*n,(t+1)*n);sources=np.arange(2*t,2*t+2)
            other_u=np.setdiff1d(np.arange(T*n),controls)
            other_s=np.setdiff1d(np.arange(2*T),sources)
            if np.any(pair.B[np.ix_(rows,other_u)]) or np.any(pair.A0[np.ix_(rows,other_s)]):
                raise ValueError('cross-time electrical coupling prevents scalar decomposition')
            blocks.append(dict(c=pair.c[rows],B=pair.B[np.ix_(rows,controls)],
                G=pair.A0[np.ix_(rows,sources)],lower=pair.lower[sources],upper=pair.upper[sources],
                center=pair.base_center[sources]))
        parts.append(blocks)
    return parts


def build_scalar_tables(context,domain,*,grid_points=3,cache=None):
    began=perf_counter()
    if grid_points not in (3,5,9,17):raise ValueError('declared grids are 3/5/9/17 nodes')
    T,n=context['control_shape']
    if domain.indices.u.shape!=(T,n):raise ValueError('operation/control shape mismatch')
    parts=_time_parts(context);cache={} if cache is None else cache
    op=domain.as_dict();lower=op['control_lower'].reshape(T,n);upper=op['control_upper'].reshape(T,n)
    if np.any(lower>0) or np.any(upper<0):raise ValueError('power domains must contain zero')
    calls=hits=0
    def node(t,j,p):
        nonlocal calls,hits
        records=[]
        for k,blocks in enumerate(parts):
            row=blocks[t];b=row['B'][:,j]
            # Zero injections are physically identical at every EV site.
            bkey=np.zeros_like(b) if p==0 else b
            digest=sha256()
            for value in (row['c'],bkey,row['G'],row['lower'],row['upper'],row['center'],
                          np.array([p,lower[t,j],upper[t,j]])):
                digest.update(np.asarray(value,dtype='<f8').tobytes())
            key=digest.hexdigest()
            if key not in cache:
                cache[key]=scalar_profile(row['c'],b,row['G'],row['lower'],row['upper'],row['center'],p,
                    power_range=(lower[t,j],upper[t,j]));calls+=1
            else:hits+=1
            records.append(cache[key])
        return dict(p=float(p),primal_upper=float(np.mean([r['primal_upper'] for r in records])),
            dual_lower=float(np.mean([r['dual_lower'] for r in records])),
            gap=float(np.mean([r['primal_dual_gap'] for r in records])),pair_records=records)
    background=[];tables=[];errors=[]
    for t in range(T):
        # Use one reference site's zero QP for every on/off alternative.
        zero=node(t,0,0.);background.append(zero);rowtables=[];rowerrors=[zero['gap']]
        for j in range(n):
            lo,hi=lower[t,j],upper[t,j]
            grid=np.unique(np.r_[lo,0.,hi] if grid_points==3 else np.r_[np.linspace(lo,hi,grid_points),0.])
            values=[zero if p==0 else node(t,j,float(p)) for p in grid]
            L=float(np.mean([block[t]['B'][:,j]@block[t]['B'][:,j] for block in parts]))
            h=float(np.max(np.diff(grid),initial=0.))
            error=L*h*h/8+max(v['gap'] for v in values)
            rowtables.append(dict(t=t,j=j,grid=grid,actual_node_count=len(grid),values=values,
                primal_upper=np.array([v['primal_upper'] for v in values]),
                dual_lower=np.array([v['dual_lower'] for v in values]),
                curvature_upper=L,max_spacing=h,curvature_chord_error=L*h*h/8,
                endpoint_gap=max(v['gap'] for v in values),total_error_upper=error))
            rowerrors.append(error)
        tables.append(rowtables);errors.append(max(rowerrors))
    return dict(grid_points=grid_points,problem_id=problem_identity(context,domain),background=background,tables=tables,
        frozen_profile_error_upper=float(sum(errors)),time_error_upper=np.array(errors),
        seconds=perf_counter()-began,QP_calls=calls,cache_hits=hits,
        expected_directed_pair_count=2,score_scale='mean over two directed half-SSE distances',
        bound_scope='frozen affine profile only; full AC is independently checked')


def action_profile_bounds(context,power_kw):
    began=perf_counter();u=np.asarray(power_kw,float)
    results=[evaluate_original_pair(pair,u.ravel()) for pair in context['pairs']]
    return dict(success=all(r['success'] for r in results),
        dual_lower=float(np.mean([.5*r['tangent_lower_numeric'] for r in results])),
        primal_upper=float(np.mean([.5*r['primal_upper'] for r in results])),
        pairs=results,seconds=perf_counter()-began,
        lower_scope='fixed feasible action; source-QP dual lower, not source primal')


def solve_scalar_distance_milp(context,domain,*,grid_points=3,tables=None,cache=None,
        initial_powers=None,solver_seconds=2.,backend='gurobi',mip_rel_gap=1e-8):
    began=perf_counter()
    table=build_scalar_tables(context,domain,grid_points=grid_points,cache=cache) if tables is None else tables
    if table['grid_points']!=grid_points:raise ValueError('table grid differs')
    if table.get('problem_id')!=problem_identity(context,domain):
        raise ValueError('table belongs to a different frozen context/operation domain')
    T,n=context['control_shape'];op=domain.as_dict();N=len(op['lower'])
    objective=np.zeros(N);constant=float(sum(row['primal_upper'] for row in table['background']))
    constraints=list(op['constraints']);added_binary=added_continuous=0
    fullstart=None
    for power in (initial_powers or []):
        realized=fixed._realize(domain,power)
        if realized['success']:fullstart=realized['x'];break
    if grid_points==3:
        # Original C/D exclusivity gives the negative/positive disjunction.
        # Background distance remains even while driving or disconnected.
        for t in range(T):
            q0=table['background'][t]['primal_upper']
            for j in range(n):
                node=table['tables'][t][j];grid=node['grid'];values=node['primal_upper']
                if grid[-1]>0:objective[domain.indices.charge[t,j]]+=(values[-1]-q0)/grid[-1]
                if grid[0]<0:objective[domain.indices.discharge[t,j]]+=(values[0]-q0)/(-grid[0])
        lower,upper,integer=op['lower'],op['upper'],op['integrality']
    else:
        indices={};size=N
        for t in range(T):
            for j in range(n):
                K=len(table['tables'][t][j]['grid'])
                lam=np.arange(size,size+K);size+=K
                seg=np.arange(size,size+max(K-1,0));size+=len(seg)
                indices[t,j]=(lam,seg);added_continuous+=K;added_binary+=len(seg)
        width=size-N;objective=np.r_[objective,np.zeros(width)]
        constraints=[LinearConstraint(hstack((csr_matrix(c.A),csr_matrix((c.A.shape[0],width))),format='csr'),c.lb,c.ub)
            for c in constraints]
        lower=np.r_[op['lower'],np.zeros(width)];upper=np.r_[op['upper'],np.ones(width)]
        integer=np.r_[op['integrality'],np.zeros(width,int)]
        for t in range(T):
            q0=table['background'][t]['primal_upper']
            for j in range(n):
                node=table['tables'][t][j];grid=node['grid'];lam,seg=indices[t,j]
                y=int(domain.indices.connected_arc[t,j]);objective[y]-=q0
                objective[lam]+=node['primal_upper'];integer[seg]=1
                row=np.zeros(size);row[lam]=1.;row[y]-=1.
                constraints.append(LinearConstraint(row,0.,0.))
                row=np.zeros(size);row[int(domain.indices.u[t,j])]=1.;row[lam]-=grid
                constraints.append(LinearConstraint(row,0.,0.))
                if len(seg):
                    row=np.zeros(size);row[seg]=1.;row[y]-=1.
                    constraints.append(LinearConstraint(row,0.,0.))
                    for k,index in enumerate(lam):
                        row=np.zeros(size);row[index]=1.
                        if k>0:row[seg[k-1]]-=1.
                        if k<len(seg):row[seg[k]]-=1.
                        constraints.append(LinearConstraint(row,-np.inf,0.))
        # A full original operation point alone is insufficient to specify SOS2
        # weights. Leave these starts unspecified; no incorrect warm start.
        fullstart=None
    solving=perf_counter()
    result=_solve_mip(-objective,integer,Bounds(lower,upper),constraints,
        time_limit=solver_seconds,mip_rel_gap=mip_rel_gap,backend=backend,start=fullstart)
    solver_time=perf_counter()-solving
    x=None if result.x is None else np.asarray(result.x[:N],float)
    audit=domain.check(x) if x is not None else dict(feasible=False)
    power=None if x is None else np.asarray(op['u_map']@x).reshape(T,n)
    bound=getattr(result,'mip_dual_bound',None)
    upper_bound=None if bound is None or not np.isfinite(bound) else constant-float(bound)
    action=None if power is None or not audit['feasible'] else action_profile_bounds(context,power)
    incumbent=None if result.fun is None else constant-float(result.fun)
    lower_value=None if action is None else action['dual_lower']
    consistent=bool(upper_bound is None or lower_value is None or
        upper_bound+1e-8*(1+abs(upper_bound))>=lower_value)
    return dict(success=bool(action is not None and action['success']),power_kw=power,operation=audit,
        operation_x=x,grid_points=grid_points,table=table,status=int(result.status),message=str(result.message),
        solver_proxy_incumbent=incumbent,solver_proxy_best_bound=upper_bound,
        solver_proxy_mip_gap=getattr(result,'mip_gap',None),frozen_objective_lower=lower_value,
        frozen_objective_upper=upper_bound,
        frozen_absolute_gap=None if upper_bound is None or lower_value is None or not consistent else max(0.,upper_bound-lower_value),
        bound_consistent=consistent,
        action_QP=action,added_binary_count=added_binary,added_continuous_count=added_continuous,
        original_binary_count=int(np.count_nonzero(op['integrality'])),
        bound_scope='global frozen affine L2 optimum bounded by MILP best-bound and feasible-action QP dual',
        full_AC_global_gap=None,solver_seconds=solver_time,seconds=perf_counter()-began,
        method='original_C_D_endpoint_chords' if grid_points==3 else 'connected_arc_adjacent_SOS2_chords')
