"""Original-absolute-domain profile design with shared product-lift MILP.

All bounds concern the declared moving-center bilinear residual and its
fixed physical parameter box. Fixed physical witnesses are feasible for
every action. Product graph OUTER envelopes and residual-square chords
give a master upper; full original-domain QPs give attained action lowers.
Power controls can be a linear selection from a complete operation MILP.
No truth, future noise, route library or nonlinear AC certificate is used.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,lsq_linear,milp
from scipy.sparse import coo_matrix,csr_matrix,issparse
from .ev_profile_chord_milp import _constraints


@dataclass(frozen=True)
class OriginalDomainPair:
    name: str
    c: np.ndarray
    B: np.ndarray
    A0: np.ndarray
    At: np.ndarray
    physical_lower: np.ndarray
    physical_upper: np.ndarray
    base_center: np.ndarray
    shift: np.ndarray

    def __post_init__(self):
        for key in ('c','B','A0','At','physical_lower','physical_upper','base_center','shift'):
            value=np.array(getattr(self,key),float,copy=True)
            if not np.isfinite(value).all():raise ValueError('finite original-domain arrays required')
            value.setflags(write=False);object.__setattr__(self,key,value)
        m,n=self.A0.shape if self.A0.ndim==2 else (0,0)
        t=self.B.shape[1] if self.B.ndim==2 else 0
        if (not isinstance(self.name,str) or not self.name or self.c.shape!=(m,) or m<1 or t<1
                or self.B.shape!=(m,t) or self.At.shape!=(t,m,n) or self.shift.shape!=(n,t)
                or self.base_center.shape!=(n,) or self.physical_lower.shape!=(n,)
                or self.physical_upper.shape!=(n,) or np.any(self.physical_lower>self.physical_upper)):
            raise ValueError('inconsistent original-domain dimensions or bounds')

    @property
    def control_dimension(self):return self.B.shape[1]
    @property
    def lower(self):return self.physical_lower
    @property
    def upper(self):return self.physical_upper
    @property
    def xi0(self):return self.base_center
    @property
    def S(self):return self.shift
    def matrix(self,u):return self.A0+np.einsum('t,tmn->mn',np.asarray(u,float),self.At)
    def residual(self,u,xi):
        u=np.asarray(u,float);xi=np.asarray(xi,float)
        return self.c+self.B@u+self.matrix(u)@(xi-self.base_center-self.shift@u)


OriginalPair=OriginalDomainPair


def from_moving_model(model):
    if 'control_curvature_difference' in model:
        raise ValueError('control-curvature diagnostic requires explicit additional polynomial terms; not silently ignored')
    d=model['domain']
    return OriginalDomainPair(model['name'],model['c'],model['B'],model['A0'],model['At'],
        d['physical_lower'],d['physical_upper'],d['base_center'],d['shift'])


def evaluate_original_pair(pair,u,*,tolerance=1e-10,max_iter=2000):
    """Fixed-u convex LS directly in absolute xi, including zero-width bounds."""
    began=perf_counter();u=np.asarray(u,float)
    if u.shape!=(pair.control_dimension,) or not np.isfinite(u).all():raise ValueError('finite matching control required')
    A=pair.matrix(u);d=pair.c+pair.B@u-A@(pair.base_center+pair.shift@u)
    lo,hi=pair.lower,pair.upper;fixed=lo==hi;free=~fixed
    xi=np.clip(np.zeros(len(lo)),lo,hi)
    if np.any(free):
        norm=np.linalg.norm(A[:,free],axis=0);scale=np.ones(len(norm));scale[norm>0]=1/norm[norm>0]
        result=lsq_linear(A[:,free]*scale,-(d+A[:,fixed]@xi[fixed]),
            bounds=(lo[free]/scale,hi[free]/scale),method='bvls',tol=tolerance,max_iter=max_iter)
        xi[free]=np.clip(result.x*scale,lo[free],hi[free])
        success,status,message,optimality=bool(result.success),int(result.status),str(result.message),float(result.optimality)
    else:success,status,message,optimality=True,3,'all physical coordinates fixed',0.
    r=d+A@xi;primal=float(r@r);g=2*A.T@r
    correction=float(np.minimum(g*(lo-xi),g*(hi-xi)).sum())
    guard=256*np.finfo(float).eps*(1+abs(primal)+abs(correction)+float(abs(g)@(abs(lo)+abs(hi)+2*abs(xi))))
    tangent=max(0.,min(primal,primal+correction-guard))
    z=xi-pair.base_center-pair.shift@u
    return dict(success=success,status=status,message=message,optimality=optimality,
        physical_parameter=xi,xi=xi,z=z,residual=r,primal_upper=primal,
        tangent_lower_numeric=float(tangent),q_primal_upper=primal,q_tangent_lower=float(tangent),
        numerical_gap=float(primal-tangent),numeric_guard=guard,
        original_action_lower=lo-pair.base_center-pair.shift@u,
        original_action_upper=hi-pair.base_center-pair.shift@u,
        physical_bound_violation=float(max(0.,np.max(lo-xi,initial=0.),np.max(xi-hi,initial=0.))),
        active_lower=np.flatnonzero(abs(xi-lo)<=1e-8),active_upper=np.flatnonzero(abs(xi-hi)<=1e-8),
        fixed_indices=np.flatnonzero(fixed),seconds=perf_counter()-began,
        domain='original_absolute_box',scope='fixed-action declared bilinear proxy QP; no full AC certificate')


def fixed_absolute_witness(pair,xi):
    """Residual a+C u+sum_(t<=s) Q_ts*u_t*u_s; no cross term is dropped."""
    xi=np.asarray(xi,float)
    if xi.shape!=pair.lower.shape or not np.isfinite(xi).all() or np.any(xi<pair.lower) or np.any(xi>pair.upper):
        raise ValueError('absolute witness must lie in the full physical box')
    delta=xi-pair.base_center
    a=pair.c+pair.A0@delta
    C=pair.B+np.einsum('tmn,n->mt',pair.At,delta)-pair.A0@pair.shift
    ts=np.einsum('tmn,ns->mts',pair.At,pair.shift)
    terms={}
    for t in range(pair.control_dimension):
        for s in range(t,pair.control_dimension):
            value=-ts[:,t,s] if s==t else -(ts[:,t,s]+ts[:,s,t])
            if np.any(value!=0.):terms[(t,s)]=value.copy()
    return dict(a=a,C=C,products=terms,physical_parameter=xi.copy(),
        cross_coefficients={key:value for key,value in terms.items() if key[0]!=key[1]},
        diagonal_coefficients={key[0]:value for key,value in terms.items() if key[0]==key[1]},
        cross_terms=[key for key in terms if key[0]!=key[1]],
        diagonal_terms=[key for key in terms if key[0]==key[1]])


def polynomial_residual(polynomial,u):
    u=np.asarray(u,float);r=polynomial['a']+polynomial['C']@u
    for (t,s),coefficient in polynomial['products'].items():r=r+coefficient*u[t]*u[s]
    return r


def physical_witness_polynomial(pair,witness,control_lower,control_upper,*,operation=None):
    """A fixed or affine absolute explanation, globally checked before use.

    A dict uses physical_intercept + physical_slope @ u. Every enclosure is
    checked over the complete control box, or an operation LP relaxation if
    that box test is inconclusive. A failed candidate is never a witness.
    """
    if not isinstance(witness,Mapping):
        answer=fixed_absolute_witness(pair,witness)
        answer.update(physical_intercept=np.asarray(witness,float),physical_slope=np.zeros_like(pair.shift),
            witness_validation=dict(method='fixed_physical_box',valid=True))
        return answer
    intercept=np.asarray(witness['physical_intercept'],float)
    slope=np.asarray(witness.get('physical_slope',np.zeros_like(pair.shift)),float)
    if (intercept.shape!=pair.lower.shape or slope.shape!=pair.shift.shape
            or not np.isfinite(intercept).all() or not np.isfinite(slope).all()):
        raise ValueError('finite affine physical witness dimensions required')
    low,high=np.asarray(control_lower,float),np.asarray(control_upper,float)
    wl=intercept+np.minimum(slope*low,slope*high).sum(axis=1)
    wh=intercept+np.maximum(slope*low,slope*high).sum(axis=1)
    bad=np.flatnonzero((wl<pair.lower)|(wh>pair.upper));method='complete_control_box'
    lp_audits=[]
    if len(bad) and operation is not None and np.isfinite(operation['lower']).all() and np.isfinite(operation['upper']).all():
        from .ev_bilinear_score_exchange_v2 import _lp_form,_linear_dual_lower
        form=_lp_form(operation['lower'],operation['upper'],operation['constraints'])
        for j in bad:
            objective=slope[j]@operation['U']
            lo,la=_linear_dual_lower(objective,operation['lower'],operation['upper'],form)
            hi,ha=_linear_dual_lower(-objective,operation['lower'],operation['upper'],form)
            wl[j]=max(wl[j],intercept[j]+lo);wh[j]=min(wh[j],intercept[j]-hi)
            lp_audits.append(dict(coordinate=int(j),lower=la,upper=ha))
        method='control_box_and_operation_LP_outer_relaxation'
    if np.any(wl<pair.lower) or np.any(wh>pair.upper):
        raise ValueError('affine physical witness is not certified feasible over the complete operation domain')
    delta=intercept-pair.base_center;D=slope-pair.shift
    a=pair.c+pair.A0@delta
    C=pair.B+np.einsum('tmn,n->mt',pair.At,delta)+pair.A0@D
    ts=np.einsum('tmn,ns->mts',pair.At,D);terms={}
    for t in range(pair.control_dimension):
        for s in range(t,pair.control_dimension):
            value=ts[:,t,s] if t==s else ts[:,t,s]+ts[:,s,t]
            if np.any(value!=0.):terms[(t,s)]=value.copy()
    return dict(a=a,C=C,products=terms,physical_intercept=intercept.copy(),physical_slope=slope.copy(),
        cross_coefficients={key:value for key,value in terms.items() if key[0]!=key[1]},
        diagonal_coefficients={key[0]:value for key,value in terms.items() if key[0]==key[1]},
        cross_terms=[key for key in terms if key[0]!=key[1]],diagonal_terms=[key for key in terms if key[0]==key[1]],
        witness_validation=dict(valid=True,method=method,physical_enclosure_lower=wl,
            physical_enclosure_upper=wh,LP_audits=lp_audits))


def _validate(pairs,low,high,offsets,weights,aggregation,score_scale):
    pairs=tuple(pairs);low=np.asarray(low,float);high=np.asarray(high,float)
    if (not pairs or low.ndim!=1 or not len(low) or high.shape!=low.shape or not np.isfinite([low,high]).all()
            or np.any(low>high) or any(not isinstance(p,OriginalDomainPair) or p.control_dimension!=len(low) for p in pairs)
            or len({p.name for p in pairs})!=len(pairs)):
        raise ValueError('unique compatible original-domain pairs and finite control box required')
    names=[p.name for p in pairs]
    offsets={name:0. for name in names} if offsets is None else offsets
    if isinstance(offsets,Mapping):
        if set(offsets)!=set(names):raise ValueError('every required pair needs one offset')
        off=np.asarray([offsets[name] for name in names],float)
    else:off=np.asarray(offsets,float)
    if weights is None:w=np.full(len(pairs),1/len(pairs))
    elif isinstance(weights,Mapping):
        if set(weights)!=set(names):raise ValueError('every required pair needs one task weight')
        w=np.asarray([weights[name] for name in names],float)
    else:w=np.asarray(weights,float)
    if (off.shape!=(len(pairs),) or w.shape!=off.shape or not np.isfinite([off,w]).all()
            or np.any(w<0) or abs(float(w.sum())-1.)>1e-10 or not np.isfinite(score_scale) or score_scale<=0
            or aggregation not in ('mean','worst')):
        raise ValueError('finite offsets, normalized nonnegative task weights, positive scale and mean/worst required')
    return pairs,low,high,off,w


def _operation(n,low,high,constraints,operation):
    if operation is None:
        return dict(lower=low.copy(),upper=high.copy(),integrality=np.zeros(n,int),U=np.eye(n),
            constraints=_constraints(constraints,n),identity=True,zero_product_pairs=(),zero_product_proof=None)
    lo,hi=np.asarray(operation['lower'],float),np.asarray(operation['upper'],float)
    integer=np.asarray(operation.get('integrality',np.zeros(len(lo))),int)
    if (lo.ndim!=1 or hi.shape!=lo.shape or integer.shape!=lo.shape or np.isnan(lo).any() or np.isnan(hi).any()
            or np.any(lo>hi) or not np.isin(integer,[0,1]).all()):raise ValueError('invalid complete operation bounds/integrality')
    if 'u_map' in operation:
        U=operation['u_map'].toarray() if issparse(operation['u_map']) else np.asarray(operation['u_map'],float)
    else:
        ids=np.asarray(operation['control_indices'],int)
        if ids.shape!=(n,) or np.any(ids<0) or np.any(ids>=len(lo)):raise ValueError('invalid operation control indices')
        U=np.zeros((n,len(lo)));U[np.arange(n),ids]=1.
    if U.shape!=(n,len(lo)) or not np.isfinite(U).all():raise ValueError('operation u_map dimension or values invalid')
    groups=_constraints(operation.get('constraints'),len(lo))
    for c in _constraints(constraints,n):groups.append(LinearConstraint(coo_matrix(c.A)@csr_matrix(U),c.lb,c.ub))
    groups.append(LinearConstraint(U,low,high))
    return dict(lower=lo,upper=hi,integrality=integer,U=U,constraints=groups,identity=False,
        zero_product_pairs=operation.get('zero_product_pairs',()),zero_product_proof=operation.get('zero_product_proof'))


def _check_operation(x,op,*,tolerance=1e-7):
    x=np.asarray(x,float)
    if x.shape!=op['lower'].shape or not np.isfinite(x).all():return dict(feasible=False,reason='operation_shape_or_nonfinite')
    box=float(max(0.,np.max(op['lower']-x,initial=0.),np.max(x-op['upper'],initial=0.)))
    linear=0.
    for c in op['constraints']:
        value=np.asarray(c.A@x).ravel()
        linear=max(linear,float(np.max(c.lb-value,initial=0.)),float(np.max(value-c.ub,initial=0.)))
    discrete=float(np.max(abs(x[op['integrality']==1]-np.rint(x[op['integrality']==1])),initial=0.))
    return dict(feasible=max(box,linear,discrete)<=tolerance,box_violation=box,
        linear_violation=linear,integer_violation=discrete,tolerance=float(tolerance))


def _solve_mip(objective,integrality,bounds,constraints,*,time_limit,mip_rel_gap=1e-8,
        backend='scipy',start=None):
    if backend=='scipy':
        return milp(objective,integrality=integrality,bounds=bounds,constraints=constraints,
            options=dict(time_limit=float(time_limit),mip_rel_gap=float(mip_rel_gap),presolve=True))
    if backend!='gurobi':raise ValueError('backend must be scipy or gurobi')
    from .ev_original_mip_backend import solve_linear_mip
    return solve_linear_mip(objective,integrality,bounds,constraints,time_limit=float(time_limit),
        mip_rel_gap=float(mip_rel_gap),start=start,backend=backend)


def _complete_operation(u,op,*,time_limit=3.,tolerance=1e-7,backend='scipy'):
    if op['identity']:
        x=np.asarray(u,float).copy();audit=_check_operation(x,op,tolerance=tolerance)
        return dict(success=audit['feasible'],x=x if audit['feasible'] else None,audit=audit,solver_calls=0)
    groups=list(op['constraints'])
    if u is not None:groups.append(LinearConstraint(op['U'],u,u))
    result=_solve_mip(np.zeros(len(op['lower'])),op['integrality'],Bounds(op['lower'],op['upper']),
        groups,time_limit=time_limit,backend=backend)
    audit=_check_operation(result.x,op,tolerance=tolerance) if result.x is not None else dict(feasible=False)
    return dict(success=audit['feasible'],x=None if not audit['feasible'] else result.x,
        audit=audit,solver_calls=1,status=int(result.status),message=str(result.message),backend=backend)


def _zero_products(n,op,raw):
    requested=op['zero_product_pairs'] if raw is None else raw
    answer=set()
    for pair in requested:
        t,s=sorted(map(int,pair))
        if not 0<=t<s<n:raise ValueError('declared zero products must be distinct valid control pairs')
        answer.add((t,s))
    return answer


def _aggregate(values,w,aggregation):
    return float(w@np.asarray(values)) if aggregation=='mean' else float(np.min(values))


def _product_range(t,s,low,high):
    if t==s:
        return (0. if low[t]<=0<=high[t] else min(low[t]**2,high[t]**2),max(low[t]**2,high[t]**2))
    corners=[low[t]*low[s],low[t]*high[s],high[t]*low[s],high[t]*high[s]]
    return min(corners),max(corners)


def _witness_group(pairs,witnesses,low,high,op=None):
    groups=[witnesses[p.name] for p in pairs] if isinstance(witnesses,Mapping) else list(witnesses)
    if len(groups)!=len(pairs) or any(not len(g) for g in groups):raise ValueError('at least one witness for every required pair')
    return [[physical_witness_polynomial(p,xi,low,high,operation=op) for xi in group] for p,group in zip(pairs,groups)]


def _compress(poly,keys,feature_low,feature_high):
    n=poly['C'].shape[1]
    F=np.column_stack((poly['C'],np.column_stack([poly['products'].get(key,np.zeros(len(poly['a']))) for key in keys]))) if keys else poly['C'].copy()
    fixed=feature_low==feature_high;active=~fixed
    a=poly['a']+F[:,fixed]@feature_low[fixed]
    joined=np.column_stack((F[:,active],a));_,R=np.linalg.qr(joined,mode='reduced')
    Fsmall=np.zeros((len(R),len(feature_low)));Fsmall[:,active]=R[:,:-1];d=R[:,-1]
    gram=joined.T@joined;delta=R.T@R-gram
    norm=float(np.max(abs(delta).sum(axis=1),initial=0.))
    round_guard=128*np.finfo(float).eps*(1+float(np.max(abs(gram).sum(axis=1),initial=0.)))
    guard=(norm+round_guard)*(1+float(np.maximum(feature_low[active]**2,feature_high[active]**2).sum()))
    return dict(d=d,F=Fsmall,qr_upward_cost_guard=float(guard),original_rows=len(poly['a']),
        compressed_rows=len(d),active_feature_columns=int(sum(active)),constant_column_retained=True,
        gram_max_abs_error=float(np.max(abs(delta),initial=0.)))


def _interval(d,F,low,high):
    ld=np.longdouble;FL=F.astype(ld);lo=low.astype(ld);hi=high.astype(ld)
    a=d.astype(ld)+np.minimum(FL*lo,FL*hi).sum(axis=1)
    b=d.astype(ld)+np.maximum(FL*lo,FL*hi).sum(axis=1)
    magnitude=1+abs(d.astype(ld))+(abs(FL)*np.maximum(abs(lo),abs(hi))).sum(axis=1)
    variable=np.any((F!=0)&(low!=high),axis=1)
    pad=np.where(variable,64*np.finfo(float).eps*magnitude,0.)
    return np.asarray(a-pad,float),np.asarray(b+pad,float),variable


def _lift(op,low,high,keys,zeros,segments):
    n=len(low);N=len(op['lower']);lbs=list(op['lower']);ubs=list(op['upper']);integer=list(op['integrality'])
    rows=[];los=[];his=[]
    def variables(lo,hi,kind=0):
        lo,hi=np.atleast_1d(lo),np.atleast_1d(hi);start=len(lbs)
        lbs.extend(lo.tolist());ubs.extend(hi.tolist());integer.extend([kind]*len(lo));return np.arange(start,start+len(lo))
    def row(values,lo=-np.inf,hi=np.inf):
        rows.append({int(k):float(v) for k,v in values.items() if v!=0});los.append(float(lo));his.append(float(hi))
    for c in op['constraints']:
        A=c.A.tocsr() if issparse(c.A) else c.A
        for j in range(A.shape[0]):
            if issparse(A):r=A.getrow(j);values=dict(zip(r.indices,r.data))
            else:values=dict(enumerate(A[j]))
            row(values,c.lb[j],c.ub[j])
    uid=variables(low,high)
    for t in range(n):row({int(uid[t]):1.,**{j:-float(v) for j,v in enumerate(op['U'][t]) if v}},0.,0.)
    p_low=[];p_high=[]
    for key in keys:
        a,b=(0.,0.) if key in zeros else _product_range(*key,low,high)
        p_low.append(a);p_high.append(b)
    vid=variables(p_low,p_high) if keys else np.array([],int)
    needed={t for key in keys if key not in zeros and low[key[0]]!=high[key[0]] and low[key[1]]!=high[key[1]] for t in key}
    selectors={}
    for t in sorted(needed):
        nodes=np.linspace(low[t],high[t],segments+1)
        if np.any(np.diff(nodes)<=0):raise ValueError('control interval cannot support requested segments')
        delta=variables(np.zeros(segments),np.ones(segments),1)
        fraction=variables(np.zeros(segments),np.ones(segments))
        row(dict(zip(delta,np.ones(segments))),1.,1.)
        values={int(uid[t]):1.}
        for j,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
            row({int(fraction[j]):1.,int(delta[j]):-1.},hi=0.)
            values[int(delta[j])]=-a;values[int(fraction[j])]=-(b-a)
        row(values,0.,0.);selectors[t]=(nodes,delta,fraction)
    errors=[];Ms=[]
    for j,(t,s) in enumerate(keys):
        v=int(vid[j]);error=0.
        if (t,s) in zeros:errors.append(0.);continue
        if low[t]==high[t]:row({v:1.,int(uid[s]):-low[t]},0.,0.)
        elif low[s]==high[s]:row({v:1.,int(uid[t]):-low[s]},0.,0.)
        elif t==s:
            nodes,delta,fraction=selectors[t]
            upper={v:1.};left={v:1.};right={v:1.}
            for k,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
                width=b-a
                upper[int(delta[k])]=-a*a;upper[int(fraction[k])]=-(b*b-a*a)
                left[int(delta[k])]=-a*a;left[int(fraction[k])]=-2*a*width
                right[int(delta[k])]=-(2*b*a-b*b);right[int(fraction[k])]=-2*b*width
            row(upper,hi=0.);row(left,lo=0.);row(right,lo=0.)
            error=float(np.max(np.diff(nodes))**2/4)
        else:
            nt,dt,ft=selectors[t];ns,ds,fs=selectors[s]
            for k,(a,b) in enumerate(zip(nt[:-1],nt[1:])):
                for l,(c,d) in enumerate(zip(ns[:-1],ns[1:])):
                    # lower w>=a*y+c*x-a*c, and w>=b*y+d*x-b*d;
                    # upper w<=b*y+c*x-b*c, and w<=a*y+d*x-a*d.
                    for cx,cy,rhs,is_lower in [(c,a,-a*c,True),(d,b,-b*d,True),
                                                (c,b,-b*c,False),(d,a,-a*d,False)]:
                        values={v:1.,int(uid[t]):-cx,int(uid[s]):-cy}
                        minexpr=p_low[j]+min(-cx*low[t],-cx*high[t])+min(-cy*low[s],-cy*high[s])
                        maxexpr=p_high[j]+max(-cx*low[t],-cx*high[t])+max(-cy*low[s],-cy*high[s])
                        M=max(0.,rhs-minexpr if is_lower else maxexpr-rhs)
                        M+=128*np.finfo(float).eps*(1+abs(rhs)+abs(minexpr)+abs(maxexpr));Ms.append(M)
                        if is_lower:
                            values[int(dt[k])]=-M;values[int(ds[l])]=-M;row(values,lo=rhs-2*M)
                        else:
                            values[int(dt[k])]=M;values[int(ds[l])]=M;row(values,hi=rhs+2*M)
            error=float(np.max(np.diff(nt))*np.max(np.diff(ns))/4)
        errors.append(error)
    return dict(lower=lbs,upper=ubs,integrality=integer,rows=rows,los=los,his=his,
        operation_dimension=N,uid=uid,vid=vid,product_keys=keys,
        feature_low=np.r_[low,p_low],feature_high=np.r_[high,p_high],
        product_abs_error=np.asarray(errors),product_big_m_max=max(Ms,default=0.),
        product_selectors=selectors,operation_binary=int(sum(op['integrality'])),
        product_binary=int(sum(integer)-sum(op['integrality'])))


def _coarse(pairs,groups,low,high,zeros,off,w,aggregation,score_scale):
    ups=[]
    for p,polys,o in zip(pairs,groups,off):
        candidates=[]
        for poly in polys:
            keys=sorted(poly['products']);bounds=[(0.,0.) if key in zeros else _product_range(*key,low,high) for key in keys]
            fl=np.r_[low,[a for a,b in bounds]];fh=np.r_[high,[b for a,b in bounds]]
            F=np.column_stack((poly['C'],np.column_stack([poly['products'][key] for key in keys]))) if keys else poly['C']
            a,b,var=_interval(poly['a'],F,fl,fh)
            cost=float(np.sum(np.maximum(a*a,b*b).astype(np.longdouble)))
            guard=256*np.finfo(float).eps*(1+abs(cost)+abs(o))
            candidates.append(score_scale*(cost-o+guard))
        ups.append(min(candidates))
    return _aggregate(ups,w,aggregation),np.asarray(ups)


def solve_original_master(pairs,witnesses,control_lower,control_upper,constraints=None,*,
        offsets=None,weights=None,aggregation='mean',score_scale=.5,operation=None,
        zero_product_pairs=None,control_segments=4,residual_segments=4,time_limit=3.,mip_rel_gap=1e-8,
        attained_lower=None,feasibility_tolerance=1e-7,backend='scipy',warm_start_operation=None):
    """MILP proposal and global upper. No returned incumbent is a true-score lower."""
    began=perf_counter();pairs,low,high,off,w=_validate(pairs,control_lower,control_upper,offsets,weights,aggregation,score_scale)
    for name,value in [('control_segments',control_segments),('residual_segments',residual_segments)]:
        if isinstance(value,bool) or int(value)!=value or value<1:raise ValueError(name+' must be positive integer')
    if not np.isfinite(time_limit) or time_limit<=0 or not np.isfinite(mip_rel_gap) or mip_rel_gap<0:raise ValueError('positive budget/nonnegative gap required')
    if backend not in ('scipy','gurobi'):raise ValueError('backend must be scipy or gurobi')
    op=_operation(len(low),low,high,constraints,operation);zeros=_zero_products(len(low),op,zero_product_pairs)
    groups=_witness_group(pairs,witnesses,low,high,op)
    keys=sorted(set().union(*(set(poly['products']) for group in groups for poly in group)))
    lift=_lift(op,low,high,keys,zeros,int(control_segments))
    compiled=[];pair_upper=[];errors=[]
    for k,(p,polys,o) in enumerate(zip(pairs,groups,off)):
        records=[]
        for i,poly in enumerate(polys):
            comp=_compress(poly,keys,lift['feature_low'],lift['feature_high'])
            a,b,var=_interval(comp['d'],comp['F'],lift['feature_low'],lift['feature_high'])
            maximum=float(np.sum(np.maximum(a*a,b*b).astype(np.longdouble)))
            upper=score_scale*(maximum-o+comp['qr_upward_cost_guard'])
            dq=abs(comp['F'][:,len(low):])@lift['product_abs_error'] if keys else np.zeros(len(a))
            product_error=float(2*np.sqrt(maximum)*np.linalg.norm(dq)+dq@dq)
            residual_error=float(np.sum(((b-a)/residual_segments)**2/4*var))
            records.append(dict(pair=k,witness=i,name=p.name,**comp,a=a,b=b,variable=var,
                score_box_upper=upper,product_cost_error_upper=product_error,
                residual_square_error_upper=residual_error,
                pwl_score_error_upper=score_scale*(product_error+residual_error+2*comp['qr_upward_cost_guard'])))
        pair_upper.append(min(r['score_box_upper'] for r in records));compiled.extend(records)
        errors.append(max(r['pwl_score_error_upper'] for r in records))
    upper=_aggregate(pair_upper,w,aggregation)
    if attained_lower is not None and (not np.isfinite(attained_lower) or upper<attained_lower):
        return dict(u=None,operation_x=None,objective_upper=upper,objective_lower=None,
            surrogate_incumbent=None,bound_consistent=False,status=4,
            backend=backend,message='finite lift-box upper is below attained original-QP lower',seconds=perf_counter()-began)
    numerical_scale=max(1e-12,abs(upper),0. if attained_lower is None else abs(attained_lower),
                        0. if attained_lower is None else abs(upper-attained_lower))
    guard=512*np.finfo(float).eps*(1+abs(upper)+float(np.max(abs(off),initial=0.))*score_scale)
    lbs,ubs,integer=lift['lower'],lift['upper'],lift['integrality']
    rows,los,his=lift['rows'],lift['los'],lift['his']
    def variables(lo,hi,kind=0):
        lo,hi=np.atleast_1d(lo),np.atleast_1d(hi);start=len(lbs)
        lbs.extend(lo.tolist());ubs.extend(hi.tolist());integer.extend([kind]*len(lo));return np.arange(start,start+len(lo))
    def row(values,lo=-np.inf,hi=np.inf):
        rows.append({int(k):float(v) for k,v in values.items() if v!=0});los.append(float(lo));his.append(float(hi))
    gid=variables((-score_scale*off-guard)/numerical_scale,(np.asarray(pair_upper)+guard)/numerical_scale)
    target=None
    if aggregation=='worst':
        target=int(variables([min(-score_scale*off)-guard],[upper+guard])[0])
        # target variable is normalized like every pair gamma.
        lbs[target]=(min(-score_scale*off)-guard)/numerical_scale;ubs[target]=(upper+guard)/numerical_scale
        for g in gid:row({target:1.,int(g):-1.},hi=0.)
    if attained_lower is not None:
        row({int(g):float(v) for g,v in zip(gid,w)} if aggregation=='mean' else {target:1.},
            lo=(attained_lower-guard)/numerical_scale)
    feature_ids=np.r_[lift['uid'],lift['vid']]
    residual_binary=0;compression_audits=[]
    for rec in compiled:
        score={int(gid[rec['pair']]):1.}
        constant=np.longdouble(-off[rec['pair']]+rec['qr_upward_cost_guard'])
        for j,(a,b,changes) in enumerate(zip(rec['a'],rec['b'],rec['variable'])):
            if not changes:constant+=np.longdouble(a)**2;continue
            nodes=np.linspace(a,b,int(residual_segments)+1)
            if np.any(np.diff(nodes)<=0):raise ValueError('residual range cannot support segments')
            delta=variables(np.zeros(residual_segments),np.ones(residual_segments),1)
            fraction=variables(np.zeros(residual_segments),np.ones(residual_segments));residual_binary+=residual_segments
            row(dict(zip(delta,np.ones(residual_segments))),1.,1.)
            expression={int(index):float(value) for index,value in zip(feature_ids,rec['F'][j]) if value}
            for l,(left,right) in enumerate(zip(nodes[:-1],nodes[1:])):
                row({int(fraction[l]):1.,int(delta[l]):-1.},hi=0.)
                expression[int(delta[l])]=-left;expression[int(fraction[l])]=-(right-left)
                score[int(delta[l])]=-score_scale*left*left/numerical_scale
                score[int(fraction[l])]=-score_scale*(right*right-left*left)/numerical_scale
            row(expression,-rec['d'][j],-rec['d'][j])
        row(score,hi=float((score_scale*constant+guard)/numerical_scale))
        compression_audits.append({key:rec[key] for key in ('name','witness','original_rows','compressed_rows',
            'active_feature_columns','constant_column_retained','gram_max_abs_error','qr_upward_cost_guard',
            'score_box_upper','product_cost_error_upper','residual_square_error_upper','pwl_score_error_upper')})
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
        sx=np.asarray(warm_start_operation,float)
        sa=_check_operation(sx,op,tolerance=feasibility_tolerance)
        start_audit['operation_validation']=sa
        if sa['feasible']:
            su=op['U']@sx
            violation=max((abs(su[t]*su[s]) for t,s in zeros),default=0.)
            start_audit['zero_product_violation']=float(violation)
            if violation<=feasibility_tolerance:
                start=np.full(len(lbs),np.nan);start[:lift['operation_dimension']]=sx
                start[lift['uid']]=su
                start[lift['vid']]=[0. if (t,s) in zeros else su[t]*su[s] for t,s in keys]
                start_audit.update(filled_variables=int(np.isfinite(start).sum()),
                    provided_to_solver=backend=='gurobi')
            else:start_audit['reason']='declared_zero_product_violated'
        else:start_audit['reason']='infeasible_operation_start_ignored'
    result=_solve_mip(objective,np.asarray(integer),Bounds(lbs,ubs),LinearConstraint(matrix,los,his),
        time_limit=time_limit,mip_rel_gap=mip_rel_gap,backend=backend,start=start)
    x=None if result.x is None else np.asarray(result.x[:lift['operation_dimension']],float)
    audit=_check_operation(x,op,tolerance=feasibility_tolerance) if x is not None else dict(feasible=False)
    u=None if x is None else op['U']@x
    zero_violation=0. if u is None else max((abs(u[t]*u[s]) for t,s in zeros),default=0.)
    if zero_violation>feasibility_tolerance:audit['feasible']=False;audit['zero_product_violation']=zero_violation
    if not audit['feasible']:u=None;x=None
    incumbent=None if result.fun is None else float(-result.fun*numerical_scale)
    dual=getattr(result,'mip_dual_bound',None)
    if dual is None and result.status==0 and not any(integer):dual=result.fun
    solver_upper=None if dual is None or not np.isfinite(dual) else float(-dual*numerical_scale+guard)
    proposed=min(upper+guard,solver_upper) if solver_upper is not None else upper+guard
    consistent=bool((incumbent is None or proposed>=incumbent) and (attained_lower is None or proposed>=attained_lower))
    preserved=proposed if consistent else upper+guard
    return dict(u=u,operation_x=x,objective_upper=float(preserved),objective_lower=None,
        surrogate_incumbent=incumbent,solver_best_bound=solver_upper,coarse_box_upper=float(upper+guard),
        bound_consistent=consistent,rejected_solver_upper=None if consistent else proposed,
        status=int(result.status),message=str(result.message),solver_success=bool(result.success),
        solver_mip_gap=getattr(result,'mip_gap',None),solver_nodes=getattr(result,'mip_node_count',None),
        operation_validation=audit,backend=backend,warm_start=start_audit,
        backend_details={key:getattr(result,key,None) for key in ('backend_requested','backend_used',
            'fallback_used','fallback_reason','start_supplied','start_provided_count','start_used',
            'gurobi_status','gurobi_runtime','gurobi_version','threads','seed','objective_sense',
            'objective_transform','backend_seconds')},
        scale=numerical_scale,numeric_guard=guard,
        n_variables=len(lbs),n_rows=len(rows),n_binary=int(sum(integer)),
        n_operation_binary=lift['operation_binary'],n_product_binary=lift['product_binary'],n_residual_binary=residual_binary,
        product_keys=keys,zero_product_pairs=sorted(zeros),zero_product_proof=op['zero_product_proof'],
        zero_products_are_operation_assumptions=bool(zeros),product_abs_error=lift['product_abs_error'],
        product_big_m_max=lift['product_big_m_max'],compressions=compression_audits,
        pwl_error_upper_against_finite_witness_objective=float(w@np.asarray(errors)) if aggregation=='mean' else float(max(errors)),
        control_segments=int(control_segments),residual_segments=int(residual_segments),seconds=perf_counter()-began,
        full_curve_enumeration=False,incumbent_is_true_score_lower=False,
        scope='shared product outer lift and square chords for original-box bilinear proxy witnesses; not full AC')


def exchange_original_domain(pairs,control_lower,control_upper,constraints=None,*,offsets=None,
        weights=None,aggregation='mean',score_scale=.5,operation=None,zero_product_pairs=None,
        initial_u=None,initial_controls=None,initial_operation_points=None,max_iterations=4,
        control_segments=4,residual_segments=4,max_segments=16,solver_seconds=3.,total_seconds=30.,
        absolute_gap=.01,relative_gap=.01,mip_rel_gap=1e-8,feasibility_tolerance=1e-7,
        witness_atol=1e-10,witness_rtol=1e-9,add_comoving_witness=True,witness_policy='both',backend='scipy'):
    """Mean-task original-domain exchange with actual-QP lower bounds."""
    began=perf_counter();pairs,low,high,off,w=_validate(pairs,control_lower,control_upper,offsets,weights,aggregation,score_scale)
    if any(not np.isfinite(v) or v<0 for v in (absolute_gap,relative_gap,mip_rel_gap,feasibility_tolerance,witness_atol,witness_rtol)):
        raise ValueError('nonnegative finite tolerances required')
    if any(not np.isfinite(v) or v<=0 for v in (solver_seconds,total_seconds)):raise ValueError('positive finite budgets required')
    for key,value in [('max_iterations',max_iterations),('control_segments',control_segments),('residual_segments',residual_segments),('max_segments',max_segments)]:
        if isinstance(value,bool) or int(value)!=value or value<1:raise ValueError(key+' must be positive integer')
    if max_segments<max(control_segments,residual_segments):raise ValueError('max_segments must cover starting segments')
    if witness_policy not in ('both','affine_preferred','fixed_only'):
        raise ValueError('witness_policy must be both, affine_preferred or fixed_only')
    if backend not in ('scipy','gurobi'):raise ValueError('backend must be scipy or gurobi')
    effective_policy=witness_policy if add_comoving_witness else 'fixed_only'
    op=_operation(len(low),low,high,constraints,operation);zeros=_zero_products(len(low),op,zero_product_pairs)
    witnesses=[[] for p in pairs];evaluations=[];iterations=[];failures=[];qp_calls=0;completion_calls=0
    lower=None;upper=None;best=None
    def add_witness(k,intercept,slope,origin):
        witness=dict(physical_intercept=np.asarray(intercept,float).copy(),physical_slope=np.asarray(slope,float).copy())
        if any(np.allclose(witness['physical_intercept'],old['physical_intercept'],rtol=witness_rtol,atol=witness_atol)
               and np.allclose(witness['physical_slope'],old['physical_slope'],rtol=witness_rtol,atol=witness_atol)
               for old in witnesses[k]):return False,True
        try:poly=physical_witness_polynomial(pairs[k],witness,low,high,operation=op)
        except ValueError as error:
            failures.append(dict(kind='affine_witness_rejected',pair=pairs[k].name,origin=origin,message=str(error)))
            return False,False
        witness['validation']=poly['witness_validation'];witness['origin']=origin
        witnesses[k].append(witness);return True,True
    def evaluate(x,origin):
        nonlocal lower,best,qp_calls
        audit=_check_operation(x,op,tolerance=feasibility_tolerance)
        if not audit['feasible']:raise ValueError('full operation point is infeasible')
        x=np.asarray(x,float).copy();u=op['U']@x
        zero_violation=max((abs(u[t]*u[s]) for t,s in zeros),default=0.)
        if zero_violation>feasibility_tolerance:raise ValueError('declared zero product violated by operation point')
        solutions=[];added=[];added_fixed=[];added_comoving=[]
        for k,p in enumerate(pairs):
            qp_calls+=1
            try:s=evaluate_original_pair(p,u)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                xi=np.clip(np.zeros(len(p.lower)),p.lower,p.upper);res=p.residual(u,xi);cost=float(res@res)
                if not np.isfinite(cost):raise FloatingPointError('no finite original-domain QP fallback') from error
                s=dict(success=False,status=-1,message=type(error).__name__+':'+str(error),
                    physical_parameter=xi,xi=xi,z=xi-p.base_center-p.shift@u,residual=res,
                    primal_upper=cost,tangent_lower_numeric=0.,numerical_gap=cost,
                    physical_bound_violation=0.,domain='original_absolute_box',fallback=True)
                failures.append(dict(kind='inner_qp_exception',origin=origin,pair=p.name,message=s['message']))
            solutions.append(s);xi=s['physical_parameter']
            fixed=False;comoving=False
            if effective_policy=='affine_preferred':
                comoving,legal=add_witness(k,xi-p.shift@u,p.shift,'co_moving_'+origin)
                if not legal:fixed,_=add_witness(k,xi,np.zeros_like(p.shift),'fixed_absolute_fallback_'+origin)
            else:
                fixed,_=add_witness(k,xi,np.zeros_like(p.shift),'fixed_absolute_'+origin)
                if effective_policy=='both':comoving,_=add_witness(k,xi-p.shift@u,p.shift,'co_moving_'+origin)
            added_fixed.append(fixed);added_comoving.append(comoving);added.append(fixed or comoving)
        pair_lower=score_scale*(np.array([s['tangent_lower_numeric'] for s in solutions])-off)
        pair_upper=score_scale*(np.array([s['primal_upper'] for s in solutions])-off)
        mean_interval=[float(w@pair_lower),float(w@pair_upper)]
        worst_interval=[float(min(pair_lower)),float(min(pair_upper))]
        selected=mean_interval if aggregation=='mean' else worst_interval
        entry=dict(origin=origin,u=u,operation_x=x,score_lower=selected[0],score_upper=selected[1],
            mean_score_interval=mean_interval,worst_score_interval=worst_interval,
            pair_score_lower=pair_lower,pair_score_upper=pair_upper,qp_results=solutions,
            added_witness=added,added_fixed_witness=added_fixed,added_comoving_witness=added_comoving,
            operation_validation=audit,all_qp_converged=all(s['success'] for s in solutions))
        evaluations.append(entry)
        if lower is None or entry['score_lower']>lower:lower=entry['score_lower'];best=entry
        if not entry['all_qp_converged']:failures.append(dict(kind='inner_qp_not_converged',origin=origin,
            pairs=[p.name for p,s in zip(pairs,solutions) if not s['success']]))
        return entry
    controls=[] if initial_controls is None else list(initial_controls)
    if initial_u is not None:controls.insert(0,initial_u)
    for index,x in enumerate(initial_operation_points or []):
        try:evaluate(x,'initial_operation_'+str(index))
        except ValueError as error:failures.append(dict(kind='initial_operation_infeasible',message=str(error)))
    for index,u in enumerate(controls):
        u=np.asarray(u,float)
        if u.shape!=low.shape or not np.isfinite(u).all():raise ValueError('initial control shape/nonfiniteness')
        if any(np.array_equal(u,e['u']) for e in evaluations):continue
        found=_complete_operation(u,op,time_limit=solver_seconds,tolerance=feasibility_tolerance,backend=backend);completion_calls+=found['solver_calls']
        if found['success']:
            try:evaluate(found['x'],'initial_control_'+str(index))
            except ValueError as error:failures.append(dict(kind='initial_control_invalid_identity',message=str(error)))
        else:failures.append(dict(kind='initial_control_operation_failure',index=index,details=found))
    if not evaluations:
        if op['identity']:
            # Solve the operation LP/MILP even for the ordinary u-only case.
            finder=dict(op);finder['identity']=False
        else:finder=op
        found=_complete_operation(None,finder,time_limit=solver_seconds,tolerance=feasibility_tolerance,backend=backend);completion_calls+=found['solver_calls']
        if found['success']:evaluate(found['x'],'initial_feasible_operation')
    if not evaluations:
        return dict(u=None,operation_x=None,objective_lower=None,objective_upper=None,gap=None,
            status='no_valid_initial_operation',completed=False,unfinished=True,failures=failures,
            evaluations=[],iterations=[],qp_calls=qp_calls,operation_completion_calls=completion_calls,backend=backend,
            seconds=perf_counter()-began)
    groups=_witness_group(pairs,witnesses,low,high,op)
    upper,pair_coarse=_coarse(pairs,groups,low,high,zeros,off,w,aggregation,score_scale)
    initialization_seconds=perf_counter()-began
    initial_upper=float(upper);cs=int(control_segments);rs=int(residual_segments)
    def target():return float(absolute_gap+relative_gap*max(1e-12,abs(upper),abs(lower)))
    def reached():return upper>=lower and upper-lower<=target()
    status='gap_reached_initial_witness_bound' if reached() else 'iteration_limit'
    if upper<lower:
        status='initial_numerical_bound_conflict';failures.append(dict(kind=status,lower=lower,upper=upper))
    elif not reached():
        for it in range(int(max_iterations)):
            remaining=total_seconds-(perf_counter()-began)
            if remaining<=0:status='time_budget';break
            previous=upper
            master=solve_original_master(pairs,witnesses,low,high,constraints,offsets=off,weights=w,
                aggregation=aggregation,score_scale=score_scale,operation=operation,zero_product_pairs=zeros,
                control_segments=cs,residual_segments=rs,time_limit=min(solver_seconds,remaining),
                mip_rel_gap=mip_rel_gap,attained_lower=lower,feasibility_tolerance=feasibility_tolerance,
                backend=backend,warm_start_operation=best['operation_x'])
            record=dict(iteration=it,master=master,previous_upper=previous,global_upper=upper,
                global_lower=lower,witnesses_before=[len(z) for z in witnesses],upper_update_accepted=False,
                control_segments=cs,residual_segments=rs)
            iterations.append(record)
            if master['u'] is None:
                status='master_without_valid_operation';failures.append(dict(kind=status,iteration=it,message=master['message']));break
            repeated=any(np.allclose(master['u'],e['u'],rtol=0.,atol=1e-8) for e in evaluations)
            entry=evaluate(master['operation_x'],'master_'+str(it))
            proposed=min(previous,float(master['objective_upper']))
            if not master['bound_consistent'] or not np.isfinite(proposed) or proposed<lower:
                upper=previous;status='numerical_master_upper_rejected'
                record.update(global_upper=upper,global_lower=lower,rejected_upper=proposed,
                    gap=upper-lower,witnesses_after=[len(z) for z in witnesses])
                failures.append(dict(kind=status,iteration=it,candidate_upper=proposed,safe_previous_upper=previous,attained_lower=lower));break
            upper=proposed
            record.update(global_upper=upper,global_lower=lower,gap=upper-lower,target_gap=target(),
                upper_update_accepted=True,repeated_control=repeated,refinement_next=False,
                witnesses_after=[len(z) for z in witnesses])
            if reached():status='gap_reached';break
            if repeated or not any(entry['added_witness']):
                if cs<max_segments or rs<max_segments:
                    cs=min(2*cs,max_segments);rs=min(2*rs,max_segments);record['refinement_next']=True
                else:status='repeated_control_with_gap';break
    completed=status in ('gap_reached','gap_reached_initial_witness_bound') and best['all_qp_converged']
    if status in ('gap_reached','gap_reached_initial_witness_bound') and not completed:status='gap_reached_inner_qp_unfinished'
    return dict(u=best['u'],operation_x=best['operation_x'],objective_lower=float(lower),objective_upper=float(upper),
        raw_gap=float(upper-lower),gap=float(upper-lower),target_gap=target(),status=status,
        completed=completed,unfinished=not completed,initial_fixed_witness_upper=initial_upper,
        initial_pair_coarse_upper=pair_coarse,aggregation=aggregation,weights=dict(zip([p.name for p in pairs],w.tolist())),
        weight_meaning='predeclared task values, not topology truth probabilities',
        score_scale=float(score_scale),offsets=dict(zip([p.name for p in pairs],off.tolist())),
        required_pairs=[p.name for p in pairs],witnesses=witnesses,witness_counts=[len(z) for z in witnesses],
        evaluations=evaluations,iterations=iterations,qp_calls=qp_calls,
        inner_qp_failures=sum(not e['all_qp_converged'] for e in evaluations),failures=failures,
        operation_completion_calls=completion_calls,best_evaluation_index=next(i for i,e in enumerate(evaluations) if e is best),
        mean_score_interval=best['mean_score_interval'],worst_score_interval=best['worst_score_interval'],
        zero_product_pairs=sorted(zeros),zero_product_proof=op['zero_product_proof'],
        comoving_witnesses_enabled=bool(add_comoving_witness),
        witness_policy=effective_policy,backend=backend,
        final_control_segments=cs,final_residual_segments=rs,seconds=perf_counter()-began,
        initialization_seconds=initialization_seconds,optimization_seconds=perf_counter()-began-initialization_seconds,
        full_curve_enumeration=False,global_nonlinear_AC_optimum_certified=False,
        physical_domain_preserved=True,
        scope='numeric mean/worst profile bounds for original fixed physical box bilinear proxy; no full AC or statistical certificate')
