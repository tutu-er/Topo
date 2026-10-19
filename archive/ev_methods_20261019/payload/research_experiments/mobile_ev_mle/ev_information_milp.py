"""Adaptive MILP bounds for fixed affine, soft-history AC design proxies.

All bounds concern the declared frozen proxy and finite pair collection.
The selector never receives truth or future measurement noise. Floating
convex/solver bounds are numerical certificates, not interval arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import hashlib
import numpy as np
from scipy.optimize import lsq_linear

try:
    from .ev_operations_milp import EVConfig, operation_model, decode_plan, check_plan
except ImportError:
    from ev_operations_milp import EVConfig, operation_model, decode_plan, check_plan


@dataclass(frozen=True)
class FrozenPair:
    name: str
    c: np.ndarray
    B: np.ndarray
    A: np.ndarray
    F: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scale: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    _solver_cache: dict = field(default_factory=dict,init=False,repr=False,compare=False)

    def __post_init__(self):
        for key in ('c', 'B', 'A', 'F', 'lower', 'upper'):
            a = np.array(getattr(self, key), dtype=float, copy=True)
            if not np.isfinite(a).all():
                raise ValueError('nonfinite frozen matrix or bounds: '+key)
            a.setflags(write=False)
            object.__setattr__(self, key, a)
        m, p = self.A.shape if self.A.ndim == 2 else (-1, -1)
        if (self.c.shape != (m,) or self.B.ndim != 2 or self.B.shape[0] != m
                or self.F.ndim != 2 or self.F.shape[1] != p
                or self.lower.shape != (p,) or self.upper.shape != (p,)
                or np.any(self.lower > self.upper)):
            raise ValueError('inconsistent frozen pair dimensions or bounds')
        scale = np.ones(p) if self.scale is None else np.asarray(self.scale, float)
        if scale.shape != (p,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('positive finite coordinate scale required')
        scale=scale.copy();scale.setflags(write=False)
        object.__setattr__(self, 'scale', scale)

    @property
    def control_dimension(self):
        return self.B.shape[1]

    @property
    def matrix(self):
        return np.vstack((self.A, self.F))

    def residual(self, u, z):
        u = np.asarray(u, float).ravel()
        z = np.asarray(z, float)
        if u.shape != (self.control_dimension,) or z.shape != self.lower.shape:
            raise ValueError('control or nuisance dimension mismatch')
        return np.r_[self.c+self.B@u+self.A@z, self.F@z]

    def fingerprint(self):
        h = hashlib.sha256(self.name.encode())
        for a in (self.c,self.B,self.A,self.F,self.lower,self.upper,self.scale):
            h.update(str(a.shape).encode());h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def solve_pair(pair: FrozenPair, u, *, tolerance=1e-10, max_iter=1000):
    """Primal upper and convex-tangent lower numerical bounds, including fixed z."""
    u = np.asarray(u, float).ravel()
    if u.shape != (pair.control_dimension,) or not np.isfinite(u).all():
        raise ValueError('finite control vector required')
    C = pair.matrix
    d = np.r_[pair.c+pair.B@u, np.zeros(len(pair.F))]
    fixed = pair.lower == pair.upper
    z = np.clip(np.zeros(len(pair.lower)), pair.lower, pair.upper)
    z[fixed] = pair.lower[fixed]
    free = ~fixed
    from .history_qp_acceleration import accelerate_history_qp
    accelerated=accelerate_history_qp(pair,C,d,z,free,tolerance,max_iter) if free.any() else None
    if accelerated is not None:
        z=accelerated["z"]
        status,message,success,optimality=(accelerated["status"],accelerated["message"],True,accelerated["optimality"])
    elif free.any():
        scaled = C[:,free]*pair.scale[free]
        adjusted = d+C[:,fixed]@z[fixed]
        fit = lsq_linear(scaled, -adjusted,
            bounds=(pair.lower[free]/pair.scale[free],pair.upper[free]/pair.scale[free]),
            method='bvls',lsq_solver='exact',tol=tolerance,max_iter=max_iter)
        z[free] = np.clip(fit.x*pair.scale[free],pair.lower[free],pair.upper[free])
        status, message, success = int(fit.status), str(fit.message), bool(fit.success)
        optimality = float(fit.optimality)
    else:
        status,message,success,optimality=3,'all coordinates fixed',True,0.0
    residual = d+C@z
    primal = float(residual@residual)
    gradient = 2*C.T@residual
    correction = np.minimum(gradient*(pair.lower-z),gradient*(pair.upper-z))
    guard = 128*np.finfo(float).eps*(1+abs(primal)+np.sum(abs(correction)))
    lower = max(0.0,primal+float(np.sum(correction))-guard)
    return dict(success=success,status=status,message=message,z=z,
        residual=residual,q_upper=primal,q_lower=lower,numeric_gap=primal-lower,
        optimality=optimality,roundoff_guard=guard,gradient=gradient,
        acceleration='history_QR' if accelerated is not None else 'full_BVLS',
        semantics='frozen convex box QP; primal is upper; tangent is numerical lower')


@dataclass(frozen=True)
class DirectionCut:
    intercept: float
    slope: np.ndarray
    direction: np.ndarray
    roundoff_guard: float = 0.0

    def value(self,u):
        return float(self.intercept+self.slope@np.asarray(u).ravel())


def support_cut(pair: FrozenPair, direction, control_lower=None, control_upper=None):
    """Exact box support formula (with downward floating guard), not QP witness."""
    v = np.asarray(direction,float)
    if v.shape != (len(pair.c)+len(pair.F),) or not np.isfinite(v).all():
        raise ValueError('direction shape/nonfinite')
    norm = float(np.linalg.norm(v))
    if norm > 1+1e-12:
        raise ValueError('direction must have norm at most one')
    if norm > 1:
        v=v/norm
    vn,vh=v[:len(pair.c)],v[len(pair.c):]
    coefficient=vn@pair.A+vh@pair.F
    terms=np.minimum(coefficient*pair.lower,coefficient*pair.upper)
    intercept=float(vn@pair.c+terms.sum())
    slope=vn@pair.B
    magnitude=1+float(abs(vn)@abs(pair.c)+abs(terms).sum())
    if control_lower is not None:
        cl,cu=np.asarray(control_lower,float),np.asarray(control_upper,float)
        if (cl.shape!=(pair.control_dimension,) or cu.shape!=cl.shape
                or not np.isfinite(cl).all() or not np.isfinite(cu).all()
                or np.any(cl>cu)):
            raise ValueError('finite ordered control box required')
        extent=np.maximum(abs(cl),abs(cu))
        magnitude+=float(abs(slope)@extent)
    elif control_upper is not None:
        raise ValueError('provide both control box bounds')
    guard=128*np.finfo(float).eps*magnitude if norm else 0.0
    return DirectionCut(intercept-guard,slope.copy(),v.copy(),guard)


def cut_from_solution(pair,u,solution=None,control_lower=None,control_upper=None):
    sol=solve_pair(pair,u) if solution is None else solution
    residual=sol['residual']; norm=float(np.linalg.norm(residual))
    direction=np.zeros_like(residual) if norm == 0 else residual/norm
    return support_cut(pair,direction,control_lower,control_upper)


def affine_range(intercept,slope,lower,upper):
    """An enclosing range over the outer control box, with rounding padding."""
    slope=np.asarray(slope,float)
    a=float(intercept+np.minimum(slope*lower,slope*upper).sum())
    b=float(intercept+np.maximum(slope*lower,slope*upper).sum())
    pad=128*np.finfo(float).eps*(1+abs(a)+abs(b))
    return a-pad,b+pad


def control_box(config):
    _,cap,discharge,_,_,_=config.arrays()
    return (np.tile(-discharge,config.n_slots),np.tile(cap,config.n_slots))


def _quadratic_box_bound(pair,z,lower,upper):
    base=pair.c+pair.A@z
    a=base+np.minimum(pair.B*lower,pair.B*upper).sum(axis=1)
    b=base+np.maximum(pair.B*lower,pair.B*upper).sum(axis=1)
    bound=float(np.maximum(a*a,b*b).sum()+np.sum((pair.F@z)**2))
    return bound+1e-10*max(1,bound)


def proxy_box_upper(pairs,lower,upper,witnesses=None):
    vals=[]
    for k,pair in enumerate(pairs):
        zs=([np.clip(np.zeros(len(pair.lower)),pair.lower,pair.upper)]
            if witnesses is None else witnesses[k])
        if not zs:
            raise ValueError('at least one feasible witness per pair required')
        bounds=[]
        for z in zs:
            z=np.asarray(z,float)
            if z.shape != pair.lower.shape or not np.isfinite(z).all() or np.any(z<pair.lower) or np.any(z>pair.upper):
                raise ValueError('infeasible fixed upper-bound witness')
            bounds.append(_quadratic_box_bound(pair,z,lower,upper))
        vals.append(min(bounds))
    return min(vals)


def lower_master(config,pairs,cuts,*,time_limit=30,mip_rel_gap=1e-8):
    model,indices=operation_model(config)
    lower,upper=control_box(config)
    qbar=proxy_box_upper(pairs,lower,upper)
    scale=max(1.0,np.sqrt(qbar))
    t=int(model.add_variables('distance_scaled',(),0,np.sqrt(qbar)/scale))
    big_m=[]
    for k,pair in enumerate(pairs):
        group=cuts[k]
        if not group:
            raise ValueError('missing cuts including zero direction')
        binary=model.add_variables('direction_'+str(k),len(group),0,1,integer=True)
        model.add_row({int(j):1 for j in binary},1,1)
        for l,cut in enumerate(group):
            minimum,_=affine_range(cut.intercept,cut.slope,lower,upper)
            M=max(0,np.sqrt(qbar)-minimum)+1e-9*scale
            row={t:1,int(binary[l]):M/scale}
            row.update({int(i):-float(s)/scale for i,s in zip(indices.u_index.ravel(),cut.slope) if s})
            model.add_row(row,upper=(cut.intercept+M)/scale)
            big_m.append(M)
    result=model.solve({t:1},time_limit=time_limit,mip_rel_gap=mip_rel_gap)
    plan=None if result.x is None else decode_plan(result,indices)
    incumbent=None if result.objective_value is None else float(result.objective_value*scale)
    bound=None if result.objective_bound is None else float(result.objective_bound*scale)
    return dict(result=result,plan=plan,distance_incumbent=incumbent,
        distance_master_upper=bound,scale=scale,big_m_max=max(big_m,default=0),
        n_variables=model.n_variables,n_rows=len(model.rows),
        bound_scope='finite-direction lower MILP; its upper bound is not full proxy upper')


def _secant_sum(model,u_index,base,B,constant,*,segments,tag,scale):
    """SOS2 via interval binaries; return row weights and constant for fbar."""
    lower=np.array([model.lower[int(j)] for j in u_index])
    upper=np.array([model.upper[int(j)] for j in u_index])
    terms={}; value=float(constant); max_error=0.0; count=0
    for row,(intercept,slope) in enumerate(zip(base,B)):
        if not np.any(slope):
            value+=float(intercept**2);continue
        a,b=affine_range(intercept,slope,lower,upper)
        if b-a<1e-12:
            value+=float(max(a*a,b*b));continue
        nodes=np.linspace(a,b,int(segments)+1)
        weights=model.add_variables(f'{tag}_weight_{row}',(int(segments),2),0,1)
        binary=model.add_variables(f'{tag}_segment_{row}',int(segments),0,1,integer=True)
        model.add_row({int(j):1 for j in binary},1,1)
        model.add_row({int(j):1 for j in weights.ravel()},1,1)
        linear={int(j):-float(s) for j,s in zip(u_index,slope) if s}
        for seg in range(int(segments)):
            model.add_row({int(weights[seg,0]):1,int(weights[seg,1]):1,int(binary[seg]):-1},upper=0)
            for endpoint in range(2):
                node=float(nodes[seg+endpoint]); idx=int(weights[seg,endpoint])
                linear[idx]=node
                terms[idx]=node*node/scale
        model.add_row(linear,float(intercept),float(intercept))
        max_error+=(b-a)**2/(4*int(segments)**2)
        count+=1
    return terms,value/scale,max_error,count


def upper_master(config,pairs,witnesses,*,segments=4,time_limit=30,mip_rel_gap=1e-8):
    if int(segments)!=segments or segments<1:
        raise ValueError('positive integer segments required')
    model,indices=operation_model(config)
    lower,upper=control_box(config)
    qbar=proxy_box_upper(pairs,lower,upper,witnesses)
    scale=max(1.0,qbar)
    T=int(model.add_variables('squared_score_scaled',(),0,qbar/scale))
    errors=[]; variable_rows=0
    for k,pair in enumerate(pairs):
        for s,z in enumerate(witnesses[k]):
            z=np.asarray(z,float)
            terms,const,error,count=_secant_sum(model,indices.u_index.ravel(),
                pair.c+pair.A@z,pair.B,float(np.sum((pair.F@z)**2)),
                segments=segments,tag=f'upper_{k}_{s}',scale=scale)
            constraint={T:1};constraint.update({i:-v for i,v in terms.items()})
            model.add_row(constraint,upper=const)
            errors.append(dict(pair=pair.name,witness=s,secant_error_upper=error))
            variable_rows+=count
    result=model.solve({T:1},time_limit=time_limit,mip_rel_gap=mip_rel_gap)
    plan=None if result.x is None else decode_plan(result,indices)
    incumbent=None if result.objective_value is None else float(result.objective_value*scale)
    # The global best bound (not incumbent) upper-bounds the full frozen proxy.
    bound=None if result.objective_bound is None else float(result.objective_bound*scale)
    if bound is not None and not np.isfinite(bound):
        bound=None
    return dict(result=result,plan=plan,q_incumbent=incumbent,q_proxy_upper=bound,
        scale=scale,secant_errors=errors,variable_residual_rows=variable_rows,
        n_variables=model.n_variables,n_rows=len(model.rows),
        bound_scope='full frozen proxy upper via fixed feasible witnesses and square secants')


def _solver_record(master,kind):
    result=master['result']
    return dict(kind=kind,status=int(result.status),success=bool(result.success),
        message=str(result.message),incumbent=master.get('q_incumbent',master.get('distance_incumbent')),
        bound=master.get('q_proxy_upper',master.get('distance_master_upper')),
        mip_gap=None if getattr(result,'mip_gap',None) is None else float(result.mip_gap),
        mip_nodes=None if getattr(result,'mip_node_count',None) is None else int(result.mip_node_count),
        variables=master['n_variables'],rows=master['n_rows'])


def adaptive_search(config,pairs,*,initial_us=(),max_iterations=8,segments=2,
                    max_segments=16,solver_seconds=10,total_seconds=180,
                    absolute_gap=1e-4,relative_gap=1e-3,seed=0):
    """Generate actions through MILPs; no full route/action enumeration.

    One upper candidate and one direction-lower candidate per iteration are
    evaluated by convex nuisance QPs. Source/profile failures abort rather
    than silently deleting a difficult pair. seed is recorded for policy IDs;
    the current optimization itself is deterministic.
    """
    pairs=list(pairs)
    if not pairs or len({p.name for p in pairs})!=len(pairs):
        raise ValueError('nonempty uniquely named pair list required')
    dimension=config.n_slots*config.n_sites
    if any(p.control_dimension!=dimension for p in pairs):
        raise ValueError('EV and frozen control dimensions differ')
    began=perf_counter(); lower,upper=control_box(config)
    cuts=[[support_cut(p,np.zeros(len(p.c)+len(p.F)))] for p in pairs]
    witnesses=[[] for _ in pairs]
    records=[]; solver_records=[]; failures=[]; cache={}
    best=None; LB=0.0; UB=proxy_box_upper(pairs,lower,upper); qp_count=0

    def evaluate(u,origin,plan=None):
        nonlocal best,LB,qp_count
        u=np.asarray(u,float).reshape(config.n_slots,config.n_sites)
        # Distinct controls must never share a purported lower bound merely
        # because they round to the same decimal grid; B can amplify that error.
        key=np.ascontiguousarray(u,dtype='<f8').tobytes()
        if plan is not None:
            audit=check_plan(plan,config,tolerance=2e-6)
            if not audit['feasible']:
                raise RuntimeError('MILP physical validation failed: '+str(audit['issues']))
        if key in cache:
            row=cache[key]
            if plan is not None and (best is None or row['q_lower']>LB):
                best=plan;LB=max(LB,row['q_lower'])
            return row
        solved=[solve_pair(p,u) for p in pairs];qp_count+=len(solved)
        if not all(s['success'] for s in solved):
            raise RuntimeError('inner nuisance QP failed; no pair removed')
        qlo=min(s['q_lower'] for s in solved);qhi=min(s['q_upper'] for s in solved)
        row=dict(origin=origin,u_kw=u.tolist(),q_lower=qlo,q_upper=qhi,
            pair_values=[dict(name=p.name,q_lower=s['q_lower'],q_upper=s['q_upper'],
                numeric_gap=s['numeric_gap'],optimality=s['optimality'],acceleration=s['acceleration']) for p,s in zip(pairs,solved)])
        records.append(row);cache[key]=row
        if plan is not None and (best is None or qlo>LB):
            best=plan;LB=max(LB,qlo)
        for k,(p,s) in enumerate(zip(pairs,solved)):
            cut=cut_from_solution(p,u,s,lower,upper)
            if not any(np.linalg.norm(cut.direction-c.direction)<1e-7 for c in cuts[k]):
                cuts[k].append(cut)
            if not any(np.linalg.norm(s['z']-z)<1e-8 for z in witnesses[k]):
                witnesses[k].append(s['z'].copy())
        return row

    # Obtain a genuinely operations-feasible seed; arbitrary initial vectors
    # can supply cuts/witnesses but cannot become a feasible outer incumbent.
    operations,idx=operation_model(config)
    feasible=operations.solve(time_limit=min(solver_seconds,total_seconds))
    if feasible.x is None:
        return dict(success=False,status='operations_infeasible_or_failed',failures=[str(feasible.message)],
                    plan=None,records=[],qp_count=0,seconds=perf_counter()-began)
    seed_plan=decode_plan(feasible,idx)
    try:
        evaluate(seed_plan.u_kw,'operations_seed',seed_plan)
        for u in initial_us:
            evaluate(u,'declared_initial_vector')
        for iteration in range(int(max_iterations)):
            remaining=total_seconds-(perf_counter()-began)
            if remaining<=0:
                break
            low=lower_master(config,pairs,cuts,time_limit=min(solver_seconds,remaining))
            solver_records.append(_solver_record(low,'direction_lower'))
            if low['plan'] is not None:
                evaluate(low['plan'].u_kw,'lower_'+str(iteration),low['plan'])
            remaining=total_seconds-(perf_counter()-began)
            if remaining<=0:
                break
            high=upper_master(config,pairs,witnesses,segments=segments,
                time_limit=min(solver_seconds,remaining))
            solver_records.append(_solver_record(high,'secant_upper'))
            if high['q_proxy_upper'] is not None:
                UB=min(UB,high['q_proxy_upper'])
            if high['plan'] is not None:
                evaluate(high['plan'].u_kw,'upper_'+str(iteration),high['plan'])
            if UB+1e-6*max(1,UB)<LB:
                raise RuntimeError('inconsistent numerical proxy bounds')
            if UB-LB<=absolute_gap+relative_gap*max(1,abs(LB)):
                break
            segments=min(int(max_segments),2*int(segments))
    except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
        failures.append(str(error))
    gap=max(0.0,UB-LB)
    return dict(success=best is not None and not failures,
        status=('failure_preserved' if failures else
            'proxy_gap_met' if gap<=absolute_gap+relative_gap*max(1,abs(LB)) else 'budget_or_iteration_limit'),
        plan=best,proxy_lower=LB,proxy_upper=UB,absolute_gap=gap,
        relative_gap=gap/max(1,abs(LB)),records=records,solver_records=solver_records,
        failures=failures,qp_count=qp_count,cut_counts=[len(c) for c in cuts],
        witness_counts=[len(w) for w in witnesses],final_segments=segments,
        pair_hashes={p.name:p.fingerprint() for p in pairs},seconds=perf_counter()-began,
        seed=seed,enumerated_complete_actions=0,
        bound_scope='declared frozen linearized proxy only; not complete nonlinear AC',
        selector_uses_truth=False,selector_uses_future_noise=False)
