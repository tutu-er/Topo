"""Compact witness upper MILP with one shared square chord per control.

The original absolute-domain QPs are unchanged.  This module is used only
when every retained witness is co-moving and every nonzero off-diagonal
Gram term multiplies a zero product proved by the complete operation MILP.
Otherwise the caller falls back to the generic residual/product master.
"""
from __future__ import annotations

from time import perf_counter
from hashlib import sha256
import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix, csr_matrix


class SharedSquareUnsupported(ValueError):
    """An exact compact representation has not been certified."""


_PROOF_CACHE={}


def _operation_hash(op,zero_pairs):
    digest=sha256()
    for name in ('lower','upper','integrality','U'):
        value=np.ascontiguousarray(op[name]);digest.update(str(value.shape).encode());digest.update(value.tobytes())
    for constraint in op['constraints']:
        matrix=csr_matrix(constraint.A,copy=True);matrix.sort_indices()
        for value in (matrix.data,matrix.indices,matrix.indptr,np.atleast_1d(constraint.lb),np.atleast_1d(constraint.ub)):
            array=np.ascontiguousarray(value);digest.update(str(array.shape).encode());digest.update(array.tobytes())
    digest.update(repr(sorted(zero_pairs)).encode());return digest.hexdigest()


def _rows(op):
    answer = []
    for constraint in op['constraints']:
        matrix = csr_matrix(constraint.A)
        lo = np.broadcast_to(constraint.lb, (matrix.shape[0],))
        hi = np.broadcast_to(constraint.ub, (matrix.shape[0],))
        for i in range(matrix.shape[0]):
            row = matrix.getrow(i)
            values = {int(j): float(v) for j, v in zip(row.indices, row.data)
                      if v and not (op['lower'][j] == op['upper'][j] == 0.)}
            answer.append((values, float(lo[i]), float(hi[i])))
    return answer


def certify_operation_zero_products(op, zero_pairs):
    """Verify zero products from power links and binary exclusion, not labels.

    An equality expresses u as a sum of nonnegative power components.  Each
    component is bounded by M*y for the same binary occupancy y.  Thus y=0
    implies u=0 even with signed net power.  A dual LP upper <2 for y+y'
    then proves that the two occupancies cannot both equal one.
    """
    from .ev_bilinear_score_exchange_v2 import _lp_form, _linear_dual_lower
    zero_pairs = sorted(set(zero_pairs))
    operation_key=_operation_hash(op,zero_pairs)
    if operation_key in _PROOF_CACHE:return dict(_PROOF_CACHE[operation_key],cache_hit=True,operation_sha256=operation_key)
    if not zero_pairs:
        return dict(valid=True, method='no_zero_products_needed', pairs=[])
    rows = _rows(op)
    binaries = set(np.flatnonzero((op['integrality'] == 1)
                                 & (op['lower'] >= 0.) & (op['upper'] <= 1.)))
    # component -> binary supports, using exact sparse row identities.
    supports = {};signed_links={}
    for values, lo, hi in rows:
        variants = [(values, hi)]
        if np.isfinite(lo): variants.append(({j: -v for j, v in values.items()}, -lo))
        for row, rhs in variants:
            if rhs != 0. or len(row) != 2: continue
            for v, coefficient in row.items():
                y, negative = next((j,x) for j,x in row.items() if j!=v)
                if y in binaries and negative<0.:
                    signed_links.setdefault((v,y),set()).add(int(np.sign(coefficient)))
            for v, coefficient in row.items():
                if coefficient <= 0. or op['lower'][v] < 0.: continue
                others = [(j, x) for j, x in row.items() if j != v]
                y, negative = others[0]
                if y in binaries and negative < 0.:
                    supports.setdefault(v, set()).add(y)
    control_support = {}
    fixed_zero = set()
    for k, mapping in enumerate(op['U']):
        ids = np.flatnonzero(mapping)
        if len(ids) != 1 or mapping[ids[0]] != 1.: continue
        v = int(ids[0])
        if op['lower'][v] == op['upper'][v] == 0.:
            fixed_zero.add(k); continue
        choices = set(supports.get(v, ()))
        choices.update(y for (variable,y),signs in signed_links.items()
                       if variable==v and signs=={-1,1})
        for values, lo, hi in rows:
            if lo != 0. or hi != 0. or v not in values: continue
            components = [j for j in values if j != v]
            if not components: continue
            if all(op['lower'][j] >= 0. and j in supports for j in components):
                common = set.intersection(*(supports[j] for j in components))
                choices.update(common)
        if choices: control_support[k] = choices
    if not np.isfinite(op['lower']).all() or not np.isfinite(op['upper']).all():
        raise SharedSquareUnsupported('zero-product proof needs finite complete operation bounds')
    form = _lp_form(op['lower'], op['upper'], op['constraints'])
    audits = []
    cache = {}
    for k, l in zero_pairs:
        if k in fixed_zero or l in fixed_zero:
            audits.append(dict(controls=[k,l], method='one_control_fixed_zero', valid=True)); continue
        valid = None
        for y in sorted(control_support.get(k, ())):
            for z in sorted(control_support.get(l, ())):
                if y == z: continue
                binary_key = tuple(sorted((y,z)))
                if binary_key not in cache:
                    objective = np.zeros(len(op['lower'])); objective[y] = objective[z] = -1.
                    lower, audit = _linear_dual_lower(objective,op['lower'],op['upper'],form)
                    cache[binary_key] = (-float(lower), audit)
                upper, audit = cache[binary_key]
                if np.isfinite(upper) and upper < 2.:
                    valid = dict(controls=[k,l], occupancy_variables=[y,z],
                        occupancy_sum_lp_upper=upper, LP_audit=audit,
                        method='exact_power_links_and_binary_exclusion_LP', valid=True)
                    break
            if valid: break
        if valid is None:
            raise SharedSquareUnsupported('unproved zero product for controls '+str((k,l)))
        audits.append(valid)
    answer=dict(valid=True, method='complete_operation_power_links_and_binary_exclusion',
                pairs=audits, labels_used_as_proof=False,cache_hit=False,operation_sha256=operation_key)
    if len(_PROOF_CACHE)>=32:_PROOF_CACHE.pop(next(iter(_PROOF_CACHE)))
    _PROOF_CACHE[operation_key]=answer
    return answer


def compile_shared_square_witness(pair, poly, low, high, certified_zero_pairs):
    """Return kappa+ell*u+d*u² plus an upward floating-point guard.

    A significant unsupported cross Gram term triggers the generic master.
    Tiny computed cross dots are enclosed, never silently set to zero.
    """
    slope = poly['physical_slope']
    if not np.array_equal(slope, pair.shift) or poly['products']:
        raise SharedSquareUnsupported('witness is not exactly co-moving affine')
    a, C = np.asarray(poly['a']), np.asarray(poly['C'])
    m, K = C.shape; eps = np.finfo(float).eps
    n = max(2*m+8, 2*len(pair.lower)+8)
    if n*eps >= 1.: raise SharedSquareUnsupported('roundoff factor is undefined')
    gamma = n*eps/(1.-n*eps)
    ld = np.longdouble
    aa, CC = a.astype(ld), C.astype(ld)
    kappa = float(aa@aa); ell = np.asarray(2*(CC.T@aa), float)
    gram = np.asarray(CC.T@CC, float); d = np.diag(gram).copy()
    ubar = np.maximum(abs(low),abs(high))
    absgram = np.asarray(abs(CC).T@abs(CC),float)
    err_k = gamma*float(abs(aa)@abs(aa))
    err_ell = 2*gamma*np.asarray(abs(CC).T@abs(aa),float)
    err_d = gamma*np.diag(absgram)
    coef_guard = float(err_k+err_ell@ubar+err_d@(ubar**2))
    cross_guard = 0.; crossed = []
    for k in range(K):
        for l in range(k+1,K):
            if (k,l) in certified_zero_pairs or ubar[k] == 0. or ubar[l] == 0.: continue
            err = gamma*absgram[k,l]
            if abs(gram[k,l]) > err:
                raise SharedSquareUnsupported('nonzero unproved cross-control Gram term '+str((k,l)))
            cost = 2*(abs(gram[k,l])+err)*ubar[k]*ubar[l]
            cross_guard += cost
            if cost: crossed.append(dict(controls=[k,l], dot=float(gram[k,l]),
                                        dot_error=float(err), upward_cost_guard=float(cost)))
    # Error in forming a=c+A0(b-xi0), C=B+At(b-xi0).
    intercept = np.asarray(poly['physical_intercept'])
    delta = intercept-pair.base_center
    delta_error = 2*eps*(abs(intercept)+abs(pair.base_center))
    ea = gamma*(abs(pair.c)+abs(pair.A0)@abs(delta))+abs(pair.A0)@delta_error
    eC = gamma*(abs(pair.B)+np.einsum('kmn,n->mk',abs(pair.At),abs(delta)))
    eC += np.einsum('kmn,n->mk',abs(pair.At),delta_error)
    e = ea+eC@ubar; magnitude = abs(a)+abs(C)@ubar
    form_guard = float(2*magnitude@e+e@e)
    # Floating squares at chord nodes and the final cut coefficients.
    chord_round_guard = float(8*eps*np.sum(abs(d)*ubar**2))
    guard = float(coef_guard+cross_guard+form_guard+chord_round_guard)
    return dict(kappa=kappa,ell=ell,diagonal=d,upward_cost_guard=guard,
        coefficient_guard=coef_guard,cross_numeric_guard=float(cross_guard),
        formation_guard=form_guard,chord_round_guard=chord_round_guard,
        enclosed_cross_dots=crossed,original_rows=m,gram=gram,
        gram_factor=gamma,products_removed_by_proof=sorted(certified_zero_pairs),
        witness_validation=poly['witness_validation'])


def solve_shared_square_master(pairs,witnesses,control_lower,control_upper,constraints=None,*,
        offsets=None,weights=None,aggregation='mean',score_scale=.5,operation=None,
        zero_product_pairs=None,control_segments=4,residual_segments=4,time_limit=3.,mip_rel_gap=1e-8,
        attained_lower=None,feasibility_tolerance=1e-7,backend='scipy',warm_start_operation=None,
        score_caps=None):
    """Compact master. Unsupported structure raises for explicit safe fallback."""
    from . import ev_original_domain_milp as od
    began=perf_counter()
    pairs,low,high,off,w=od._validate(pairs,control_lower,control_upper,offsets,weights,aggregation,score_scale)
    caps=od._caps(score_caps,pairs)
    if isinstance(control_segments,bool) or int(control_segments)!=control_segments or control_segments<1:
        raise ValueError('control_segments must be positive integer')
    if not np.isfinite(time_limit) or time_limit<=0 or not np.isfinite(mip_rel_gap) or mip_rel_gap<0:
        raise ValueError('positive budget/nonnegative gap required')
    if backend not in ('scipy','gurobi'):raise ValueError('backend must be scipy or gurobi')
    op=od._operation(len(low),low,high,constraints,operation)
    zeros=od._zero_products(len(low),op,zero_product_pairs)
    proof=certify_operation_zero_products(op,zeros)
    groups=od._witness_group(pairs,witnesses,low,high,op)
    compiled=[];pair_upper=[];errors=[];ubar=np.maximum(abs(low),abs(high))
    for k,(pair,polys) in enumerate(zip(pairs,groups)):
        records=[]
        for i,poly in enumerate(polys):
            rec=compile_shared_square_witness(pair,poly,low,high,zeros)
            maximum=rec['kappa']+float(np.maximum(rec['ell']*low,rec['ell']*high).sum())
            maximum+=float(rec['diagonal']@(ubar**2))+rec['upward_cost_guard']
            error=float(rec['diagonal']@(((high-low)/control_segments)**2/4))
            rec.update(pair=k,witness=i,name=pair.name,
                score_box_upper=float(score_scale*(maximum-off[k])),
                square_chord_cost_error_upper=error,
                pwl_score_error_upper=float(score_scale*(error+rec['upward_cost_guard'])))
            records.append(rec)
        compiled.extend(records)
        pair_upper.append(min(rec['score_box_upper'] for rec in records))
        errors.append(max(rec['pwl_score_error_upper'] for rec in records))
    pair_upper=np.minimum(pair_upper,caps)
    upper=od._aggregate(pair_upper,w,aggregation)
    if attained_lower is not None and (not np.isfinite(attained_lower) or upper<attained_lower):
        return dict(u=None,operation_x=None,objective_upper=upper,objective_lower=None,
            surrogate_incumbent=None,bound_consistent=False,status=4,backend=backend,
            master_form='shared_control_square',message='compact box upper is below attained original-QP lower',
            seconds=perf_counter()-began,shared_square_structure=proof)
    scale=max(1e-12,abs(upper),0. if attained_lower is None else abs(attained_lower),
              0. if attained_lower is None else abs(upper-attained_lower))
    guard=1024*np.finfo(float).eps*(1+abs(upper)+float(np.max(abs(off),initial=0.))*score_scale)
    lbs=list(op['lower']);ubs=list(op['upper']);integer=list(op['integrality'])
    rows=[];los=[];his=[]
    def row(values,lo=-np.inf,hi=np.inf):
        rows.append({int(j):float(v) for j,v in values.items() if v});los.append(float(lo));his.append(float(hi))
    def variables(lo,hi,kind=0):
        lo,hi=np.atleast_1d(lo),np.atleast_1d(hi);start=len(lbs)
        lbs.extend(lo.tolist());ubs.extend(hi.tolist());integer.extend([kind]*len(lo));return np.arange(start,start+len(lo))
    for constraint in op['constraints']:
        matrix=csr_matrix(constraint.A)
        ll=np.broadcast_to(constraint.lb,(matrix.shape[0],));hh=np.broadcast_to(constraint.ub,(matrix.shape[0],))
        for i in range(matrix.shape[0]):
            rr=matrix.getrow(i);row(dict(zip(rr.indices,rr.data)),ll[i],hh[i])
    uid=variables(low,high)
    for k,index in enumerate(uid):
        eq={int(index):1.}
        eq.update({int(j):-float(v) for j,v in enumerate(op['U'][k]) if v})
        row(eq,0.,0.)
    square_lower=np.where((low<=0)&(high>=0),0.,np.minimum(low**2,high**2))
    sid=variables(square_lower,ubar**2);square_binary=0;selectors={}
    for k,(left,right) in enumerate(zip(low,high)):
        if left==right:continue
        nodes=np.linspace(left,right,int(control_segments)+1)
        if np.any(np.diff(nodes)<=0):raise ValueError('control range cannot support segments')
        delta=variables(np.zeros(control_segments),np.ones(control_segments),1)
        fraction=variables(np.zeros(control_segments),np.ones(control_segments));square_binary+=control_segments
        row(dict(zip(delta,np.ones(control_segments))),1.,1.)
        eu={int(uid[k]):1.};es={int(sid[k]):1.}
        for l,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
            row({int(fraction[l]):1.,int(delta[l]):-1.},hi=0.)
            eu[int(delta[l])]=-a;eu[int(fraction[l])]=-(b-a)
            es[int(delta[l])]=-a*a;es[int(fraction[l])]=-(b*b-a*a)
        row(eu,0.,0.);row(es,0.,0.);selectors[k]=(delta,fraction,nodes)
    gamma_lower=np.minimum(-score_scale*off,caps)
    gid=variables((gamma_lower-guard)/scale,(pair_upper+guard)/scale)
    target=None
    if aggregation=='worst':
        target=int(variables([(min(gamma_lower)-guard)/scale],[(upper+guard)/scale])[0])
        for g in gid:row({target:1.,int(g):-1.},hi=0.)
    if attained_lower is not None:
        row({int(g):float(v) for g,v in zip(gid,w)} if aggregation=='mean' else {target:1.},
            lo=(attained_lower-guard)/scale)
    for rec in compiled:
        cut={int(gid[rec['pair']]):1.}
        for j,v in zip(uid,rec['ell']):
            if v:cut[int(j)]=-score_scale*v/scale
        for j,v in zip(sid,rec['diagonal']):
            if v:cut[int(j)]=-score_scale*v/scale
        row(cut,hi=(score_scale*(rec['kappa']+rec['upward_cost_guard']-off[rec['pair']])+guard)/scale)
    rr=[];cc=[];vv=[]
    for i,r in enumerate(rows):
        for j,value in r.items():rr.append(i);cc.append(j);vv.append(value)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(rows),len(lbs))).tocsc()
    objective=np.zeros(len(lbs))
    if aggregation=='mean':objective[gid]=-w
    else:objective[target]=-1.
    start=None;start_audit=dict(requested=warm_start_operation is not None,provided_to_solver=False,
        filled_variables=0,backend_supports_start=backend=='gurobi')
    if warm_start_operation is not None:
        sx=np.asarray(warm_start_operation,float);sa=od._check_operation(sx,op,tolerance=feasibility_tolerance)
        start_audit['operation_validation']=sa
        if sa['feasible']:
            su=op['U']@sx;zero_violation=max((abs(su[k]*su[l]) for k,l in zeros),default=0.)
            start_audit['zero_product_violation']=float(zero_violation)
            if zero_violation<=feasibility_tolerance:
                start=np.full(len(lbs),np.nan);start[:len(op['lower'])]=sx;start[uid]=su
                square=su**2
                for k,(delta,fraction,nodes) in selectors.items():
                    j=min(int(np.searchsorted(nodes,su[k],side='right')-1),len(nodes)-2);j=max(j,0)
                    alpha=float(np.clip((su[k]-nodes[j])/(nodes[j+1]-nodes[j]),0.,1.))
                    start[delta]=0.;start[fraction]=0.;start[delta[j]]=1.;start[fraction[j]]=alpha
                    square[k]=(1-alpha)*nodes[j]**2+alpha*nodes[j+1]**2
                start[sid]=square
                start_audit.update(filled_variables=int(np.isfinite(start).sum()),provided_to_solver=backend=='gurobi')
            else:start_audit['reason']='declared_zero_product_violated'
        else:start_audit['reason']='infeasible_operation_start_ignored'
    result=od._solve_mip(objective,np.asarray(integer),Bounds(lbs,ubs),LinearConstraint(matrix,los,his),
        time_limit=time_limit,mip_rel_gap=mip_rel_gap,backend=backend,start=start)
    x=None if result.x is None else np.asarray(result.x[:len(op['lower'])],float)
    audit=od._check_operation(x,op,tolerance=feasibility_tolerance) if x is not None else dict(feasible=False)
    u=None if x is None else op['U']@x
    if u is not None:
        violation=max((abs(u[k]*u[l]) for k,l in zeros),default=0.)
        if violation>feasibility_tolerance:audit.update(feasible=False,zero_product_violation=violation)
    if not audit['feasible']:u=None;x=None
    incumbent=None if result.fun is None else float(-result.fun*scale)
    dual=getattr(result,'mip_dual_bound',None)
    if dual is None and result.status==0 and not any(integer):dual=result.fun
    solver_upper=None if dual is None or not np.isfinite(dual) else float(-dual*scale+guard)
    proposed=min(upper+guard,solver_upper) if solver_upper is not None else upper+guard
    consistent=bool((incumbent is None or proposed>=incumbent) and (attained_lower is None or proposed>=attained_lower))
    summaries=[{key:rec[key] for key in rec if key not in ('ell','diagonal','gram')} for rec in compiled]
    return dict(u=u,operation_x=x,objective_upper=float(proposed if consistent else upper+guard),objective_lower=None,
        surrogate_incumbent=incumbent,solver_best_bound=solver_upper,coarse_box_upper=float(upper+guard),
        bound_consistent=consistent,rejected_solver_upper=None if consistent else proposed,
        status=int(result.status),message=str(result.message),solver_success=bool(result.success),
        solver_mip_gap=getattr(result,'mip_gap',None),solver_nodes=getattr(result,'mip_node_count',None),
        operation_validation=audit,backend=backend,warm_start=start_audit,
        backend_details={key:getattr(result,key,None) for key in ('backend_requested','backend_used','fallback_used',
            'fallback_reason','start_supplied','start_provided_count','start_used','gurobi_status','gurobi_runtime',
            'gurobi_version','threads','seed','objective_sense','objective_transform','backend_seconds')},
        master_form='shared_control_square',scale=scale,numeric_guard=guard,
        n_variables=len(lbs),n_rows=len(rows),n_binary=int(sum(integer)),
        n_operation_binary=int(sum(op['integrality'])),n_product_binary=0,n_residual_binary=0,
        n_shared_square_binary=square_binary,n_shared_square_controls=len(selectors),
        product_keys=[],zero_product_pairs=sorted(zeros),zero_product_proof=op['zero_product_proof'],
        shared_square_structure=proof,shared_square_witnesses=summaries,compressions=[],
        score_caps=None if score_caps is None else dict(zip([p.name for p in pairs],caps.tolist())),
        pwl_error_upper_against_finite_witness_objective=float(w@np.asarray(errors)) if aggregation=='mean' else float(max(errors)),
        control_segments=int(control_segments),residual_segments=0,seconds=perf_counter()-began,
        full_curve_enumeration=False,incumbent_is_true_score_lower=False,
        scope='shared control-square witness upper for original-box bilinear proxy; not full AC')
