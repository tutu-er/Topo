"""Conservative Bhattacharyya-risk MILP for one frozen Gaussian AC proxy.

The original EV operation rows are copied unchanged.  Means are affine around
``anchor`` and covariances stay fixed.  Square chords are shifted down by a
uniform error and clipped at zero; the exponential uses a convex PWL epigraph
with no exponential segment binaries.  All bounds concern this frozen proxy,
not the optimal nonlinear AC Bayes risk.
"""
from __future__ import annotations

from itertools import combinations
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint
from scipy.special import logsumexp

from .ev_operations_milp import LinearMILP
from .ev_original_mip_backend import solve_linear_mip
from . import ev_mobile_v3_projection as operations
from . import ev_mobile_v4_master as squares


def _positive_integer(value, name):
    if isinstance(value,(bool,np.bool_)) or int(value)!=value or value<1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def _bank(bank, jacobian, anchor):
    means=np.asarray(bank['means'],float); cov=np.asarray(bank['covariances'],float)
    u=np.asarray(anchor,float); J=np.asarray(jacobian,float)
    if means.ndim!=3 or not all(means.shape) or means.shape[1]<2:
        raise ValueError('means require nonempty [scenario,candidate,output] with at least two candidates')
    L,K,d=means.shape
    if u.ndim!=2 or not u.size or cov.shape!=(L,K,d,d) or J.shape!=(L,K,d,u.size):
        raise ValueError('covariance, jacobian or anchor dimensions disagree')
    if not all(np.isfinite(x).all() for x in (means,cov,J,u)):
        raise ValueError('finite means, covariances, jacobian and anchor required')
    prior=np.asarray(bank['prior'],float); weights=np.asarray(bank['weights'],float)
    if prior.shape!=(K,) or weights.shape!=(L,) or not all(np.isfinite(x).all() for x in (prior,weights)):
        raise ValueError('matching finite prior and scenario weights required')
    if np.any(prior<0) or np.any(weights<0) or not np.isclose(prior.sum(),1.,atol=1e-10,rtol=1e-10) or not np.isclose(weights.sum(),1.,atol=1e-10,rtol=1e-10):
        raise ValueError('prior and scenario weights must be nonnegative and sum to one')
    log_key='log_prior' if 'log_prior' in bank else 'logprior' if 'logprior' in bank else None
    if log_key is not None:
        logprior=np.asarray(bank[log_key],float)
        if logprior.shape!=(K,) or np.isnan(logprior).any() or np.isposinf(logprior).any() or abs(float(logsumexp(logprior)))>1e-8:
            raise ValueError('log_prior must contain normalized log probabilities')
        if not np.allclose(np.exp(logprior),prior,atol=0.,rtol=1e-8):
            raise ValueError('prior and logprior disagree')
    else:
        with np.errstate(divide='ignore'):logprior=np.log(prior)
    logdet=np.empty((L,K))
    for l in range(L):
        for k in range(K):
            if not np.allclose(cov[l,k],cov[l,k].T,atol=1e-11,rtol=1e-10):
                raise ValueError('each covariance must be symmetric')
            try:chol=np.linalg.cholesky(cov[l,k])
            except np.linalg.LinAlgError as exc:raise ValueError('each covariance must be positive definite') from exc
            logdet[l,k]=2*np.log(np.diag(chol)).sum()
    return dict(means=means,covariances=cov,prior=prior,logprior=logprior,weights=weights,
                jacobian=J,anchor=u,logdet=logdet)


def _pairs(data):
    means,J,cov=data['means'],data['jacobian'],data['covariances']; L,K,_=means.shape
    result=[]
    for l in range(L):
        for i,j in combinations(range(K),2):
            bar=(cov[l,i]+cov[l,j])*.5; chol=np.linalg.cholesky(bar)
            covariance_term=.5*(2*np.log(np.diag(chol)).sum()-.5*(data['logdet'][l,i]+data['logdet'][l,j]))
            if covariance_term < -1e-9:raise ValueError('negative Gaussian covariance distance beyond roundoff')
            covariance_term=max(0.,float(covariance_term))
            # .5 ||c+B u||^2 = (mu_i-mu_j)^T Sbar^-1 (mu_i-mu_j)/8.
            B=.5*solve_triangular(chol,J[l,i]-J[l,j],lower=True)
            c=.5*solve_triangular(chol,means[l,i]-means[l,j],lower=True)-B@data['anchor'].ravel()
            logweight=(float(np.log(data['weights'][l])) if data['weights'][l]>0 else -np.inf)+.5*(data['logprior'][i]+data['logprior'][j])
            weight=float(np.exp(logweight))
            result.append(SimpleNamespace(a_label=str(l)+':'+str(i),b_label=str(l)+':'+str(j),
                B=B,c=c,covariance_term=covariance_term,weight=weight,logweight=float(logweight),
                scenario=l,i=i,j=j,control_dimension=data['anchor'].size,
                control_shape=data['anchor'].shape,reference_u=data['anchor']))
    return result


def frozen_bc_value(bank, jacobian, anchor, power_kw):
    """Accurate BC sum for the affine means/fixed covariances, without PWL."""
    data=_bank(bank,jacobian,anchor);u=np.asarray(power_kw,float)
    if u.shape!=data['anchor'].shape or not np.isfinite(u).all():raise ValueError('power shape differs from anchor')
    rows=[]
    for p in _pairs(data):
        residual=p.c+p.B@u.ravel(); B=float(.5*(residual@residual)+p.covariance_term)
        rows.append(dict(scenario=p.scenario,i=p.i,j=p.j,B=B,weight=p.weight,
            logweight=p.logweight,weighted_BC=float(p.weight*np.exp(-B))))
    return dict(C=float(sum(r['weighted_BC'] for r in rows)),pairs=rows,
                scope='affine means and frozen covariance Gaussian BC sum')


def _factor_grid(factor):
    return np.asarray(factor['grid'],float)


def _exp_grid(upper, segments):
    """More knots near zero; curvature becomes small in the far tail."""
    if upper<=0:return np.array([0.])
    transformed=np.linspace(1.,np.exp(-.5*upper),segments+1)
    grid=np.zeros(segments+1);grid[-1]=upper
    grid[1:-1]=-2*np.log(transformed[1:-1])
    return np.unique(grid)


def _exp_error(grid):
    """Exact maximum exp chord error, including very wide final intervals."""
    error=0.
    for lo,hi in zip(grid[:-1],grid[1:]):
        left,right=float(np.exp(-lo)),float(np.exp(-hi));slope=(right-left)/(hi-lo)
        if slope==0:continue
        point=float(np.clip(-np.log(-slope),lo,hi))
        error=max(error,left+slope*(point-lo)-float(np.exp(-point)))
    return max(0.,float(error))


def build_bc_master(domain, bank, jacobian, anchor, *, segments=4, exp_segments=8, power_radius=None):
    """Build a MIN MILP and its uniform one-sided approximation certificate."""
    began=perf_counter();segments=_positive_integer(segments,'segments')
    exp_segments=_positive_integer(exp_segments,'exp_segments');data=_bank(bank,jacobian,anchor)
    op=operations._operation(domain,data['anchor'].size)
    if power_radius is not None:
        radius=np.broadcast_to(np.asarray(power_radius,float),data['anchor'].shape).ravel()
        if not np.isfinite(radius).all() or np.any(radius<0):raise ValueError('nonnegative finite power radius required')
        lo=np.maximum(op['control_lower'],data['anchor'].ravel()-radius)
        hi=np.minimum(op['control_upper'],data['anchor'].ravel()+radius)
        if np.any(lo>hi):raise ValueError('power trust region is disjoint from operation bounds')
        op=dict(op,control_lower=lo,control_upper=hi,constraints=[*op['constraints'],LinearConstraint(op['U'],lo,hi)])
    pairs=_pairs(data);shared=squares._shared_structure(pairs,op,domain,'auto')
    model=LinearMILP();base=model.add_variables('operation',len(op['lower']),lower=op['lower'],upper=op['upper'],integer=op['integrality'])
    for con in op['constraints']:
        A=con.A.tocsr()
        for row in range(A.shape[0]):
            cut=slice(A.indptr[row],A.indptr[row+1])
            model.add_row(dict(zip(A.indices[cut],A.data[cut])),con.lb[row],con.ub[row])
    modes=[];n_square_binary=0;n_clip_binary=0

    def square_graph(factor,scale,name):
        nonlocal n_square_binary
        grid=_factor_grid(factor)
        if len(grid)==1:return {},scale*float(grid[0]**2)
        lam=model.add_variables(name+'_lambda',len(grid),upper=1.)
        model.add_row({int(i):1. for i in lam},1.,1.)
        coeff=np.asarray(factor['coefficient']@op['U']).ravel()
        equation={int(i):float(v) for i,v in enumerate(coeff) if v}
        equation.update({int(i):-float(g) for i,g in zip(lam,grid)})
        model.add_row(equation,0.,0.);select=[]
        if len(grid)>2:
            select=model.add_variables(name+'_segment',len(grid)-1,upper=1.,integer=True)
            n_square_binary+=len(select);model.add_row({int(i):1. for i in select},1.,1.)
            for k,index in enumerate(lam):
                row={int(index):1.}
                if k:row[int(select[k-1])]=-1.
                if k<len(select):row[int(select[k])]=-1.
                model.add_row(row,upper=0.)
        modes.append(dict(factor=factor,scale=scale,lambda_indices=lam,segment_indices=select))
        return {int(i):scale*float(g*g) for i,g in zip(lam,grid)},0.

    global_factors=None;global_squares=None
    if shared['used']:
        global_factors=squares._global_control_factors(pairs[0],op,segments,None)
        needed=np.any(np.vstack([r['diagonal']>0 for r in shared['records']]),axis=0)
        global_squares=[square_graph(f,1.,'u_'+str(i)) if need else ({},0.)
            for i,(f,need) in enumerate(zip(global_factors,needed))]
        shared['shared_nonzero_control_squares']=int(np.count_nonzero(needed))
    objective={};audits=[];errors=[]
    ubar=np.maximum(abs(op['control_lower']),abs(op['control_upper']))
    for index,pair in enumerate(pairs):
        if shared['used']:
            proof=shared['records'][index];diagonal=proof['diagonal'];factors=global_factors
            rounding=float(proof['upward_quadratic_guard'])
            chord_error=float(sum(d*f['square_error'] for d,f in zip(diagonal,factors)))
            square_coeff={};square_constant=0.
            for d,(coeff,constant) in zip(diagonal,global_squares):
                for j,value in coeff.items():square_coeff[j]=square_coeff.get(j,0.)+float(d*value)
                square_constant+=float(d*constant)
            form='shared_control_square'
        else:
            raw=squares._factor_data(pair,op,segments,None);factors=raw['factors'];diagonal=None
            rounding=float(raw['quadratic_rounding_upper']);chord_error=float(raw['chord_error'])
            square_coeff={};square_constant=0.
            for j,factor in enumerate(factors):
                coeff,constant=square_graph(factor,.5,'pair_'+str(index)+'_factor_'+str(j))
                square_coeff.update(coeff);square_constant+=constant
            form='generic_control_factor_square'
        linear=pair.B.T@pair.c;constant=float(.5*(pair.c@pair.c)+pair.covariance_term)
        numeric_guard=128*np.finfo(float).eps*(1.+abs(constant)+float(abs(linear)@ubar)+float(np.sum((abs(pair.B)@ubar)**2)))
        guard=rounding+numeric_guard;epsilon_B=chord_error+2*guard
        # q_chord >= B, q_chord-B <= epsilon_B, uniformly on the operation domain.
        Bcoeff=dict(square_coeff)
        for j,value in enumerate(np.asarray(linear@op['U']).ravel()):
            if value:Bcoeff[j]=Bcoeff.get(j,0.)+float(value)
        Bconstant=constant+square_constant+guard
        lo_r=pair.c+np.where(pair.B>=0,pair.B*op['control_lower'],pair.B*op['control_upper']).sum(axis=1)
        hi_r=pair.c+np.where(pair.B>=0,pair.B*op['control_upper'],pair.B*op['control_lower']).sum(axis=1)
        upper_B=float(pair.covariance_term+.5*(np.maximum(abs(lo_r),abs(hi_r))**2).sum())
        # q_chord-eps <= B, and B >= the Gaussian covariance contribution >= 0.
        low_v=float(pair.covariance_term-epsilon_B);high_v=max(low_v,upper_B)
        v=int(model.add_variables('pair_'+str(index)+'_shifted',lower=low_v,upper=high_v))
        equality={v:1.};equality.update({j:-value for j,value in Bcoeff.items()})
        model.add_row(equality,Bconstant-epsilon_B,Bconstant-epsilon_B)
        if low_v>=0:
            clip=v
        elif high_v<=0:
            clip=int(model.add_variables('pair_'+str(index)+'_clip',lower=0.,upper=0.))
        else:
            clip=int(model.add_variables('pair_'+str(index)+'_clip',upper=high_v))
            binary=int(model.add_variables('pair_'+str(index)+'_clip_on',upper=1.,integer=True));n_clip_binary+=1
            model.add_row({clip:1.,v:-1.},lower=0.)
            model.add_row({clip:1.,v:-1.,binary:-low_v},upper=-low_v)
            model.add_row({clip:1.,binary:-high_v},upper=0.)
        grid=_exp_grid(max(0.,high_v),exp_segments);values=np.exp(-grid)
        epi=int(model.add_variables('pair_'+str(index)+'_exp',lower=float(values[-1]),upper=1.))
        if len(grid)==1:model.add_row({epi:1.},1.,1.)
        else:
            for a,b,fa,fb in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
                slope=float((fb-fa)/(b-a));intercept=float(fa-slope*a)
                model.add_row({epi:1.,clip:-slope},lower=intercept)
        objective[epi]=pair.weight
        exp_error=_exp_error(grid);total_error=min(1.,epsilon_B+exp_error)
        errors.append(pair.weight*total_error)
        audits.append(dict(scenario=pair.scenario,i=pair.i,j=pair.j,weight=pair.weight,logweight=pair.logweight,
            covariance_term=pair.covariance_term,form=form,factors=factors,diagonal=diagonal,
            B_coefficient=Bcoeff,B_constant=Bconstant,linear=linear,constant=constant,
            rounding_guard=rounding,numeric_guard=numeric_guard,epsilon_B=epsilon_B,chord_error=chord_error,
            shifted_bounds=[low_v,high_v],clip_index=clip,shifted_index=v,exp_index=epi,
            exp_grid=grid,exp_chord_error=exp_error,weighted_uniform_error=pair.weight*total_error))
    arrays=model.arrays(objective,cmax=False)
    return dict(**arrays,model=model,operation=op,operation_dimension=len(base),data=data,pairs=pairs,
        pair_audits=audits,modes=modes,shared_control_square=shared,n_square_binaries=n_square_binary,
        n_clip_binaries=n_clip_binary,n_exp_binaries=0,proxy_uniform_error=float(sum(errors)),
        original_operation_constraints_preserved=True,power_radius=power_radius,
        build_seconds=perf_counter()-began,scope='frozen affine means and frozen covariance Gaussian BC upper-risk proxy')


def evaluate_proxy(master, power_kw):
    """Recompute minimal epigraph value; certificate requires an attainable EV power.

    This arithmetic helper checks the control box only.  Shared zero-product
    certificates apply to full feasible operation vectors, not arbitrary
    simultaneous injections at otherwise exclusive sites.
    """
    u=np.asarray(power_kw,float).ravel();op=master['operation']
    if u.shape!=(master['data']['anchor'].size,) or not np.isfinite(u).all():raise ValueError('matching finite power required')
    if np.any(u<op['control_lower']-1e-7) or np.any(u>op['control_upper']+1e-7):raise ValueError('power outside proxy control bounds')
    rows=[]
    for p,a in zip(master['pairs'],master['pair_audits']):
        if a['form']=='shared_control_square':
            quadratic=sum(float(d*np.interp(f['coefficient']@u,f['grid'],np.asarray(f['grid'])**2))
                for d,f in zip(a['diagonal'],a['factors']))
        else:
            quadratic=sum(float(.5*np.interp(f['coefficient']@u,f['grid'],np.asarray(f['grid'])**2)) for f in a['factors'])
        Bch=quadratic+a['linear']@u+a['constant']+a['rounding_guard']+a['numeric_guard']
        BL=max(0.,float(Bch-a['epsilon_B']))
        residual=p.c+p.B@u;exact_B=float(.5*(residual@residual)+p.covariance_term)
        approx=float(np.interp(BL,a['exp_grid'],np.exp(-a['exp_grid'])))
        rows.append(dict(scenario=p.scenario,i=p.i,j=p.j,B=exact_B,B_chord=float(Bch),B_lower=BL,
            epsilon_B=a['epsilon_B'],exact_BC=float(np.exp(-exact_B)),proxy_BC=approx,
            weight=p.weight,weighted_proxy=float(p.weight*approx),weighted_exact=float(p.weight*np.exp(-exact_B))))
    value=float(sum(r['weighted_proxy'] for r in rows));exact=float(sum(r['weighted_exact'] for r in rows))
    return dict(proxy=value,C=exact,actual_proxy_error=value-exact,uniform_error=master['proxy_uniform_error'],pairs=rows)


def _finite(value):
    if value is None:return None
    value=float(value)
    return value if np.isfinite(value) else None


def solve_bc_milp(domain, bank, jacobian, anchor, segments=4, exp_segments=8,
                  seconds=5, power_radius=None, *, backend='gurobi'):
    """Return one executable proposal plus frozen-proxy bounds, never original-risk optimality."""
    began=perf_counter()
    if not np.isfinite(seconds) or seconds<=0:raise ValueError('positive finite solver seconds required')
    master=build_bc_master(domain,bank,jacobian,anchor,segments=segments,exp_segments=exp_segments,power_radius=power_radius)
    solved=solve_linear_mip(master['c'],master['integrality'],master['bounds'],master['constraints'],
        time_limit=float(seconds),mip_rel_gap=1e-5,backend=backend)
    status=int(solved.status);raw_inc=_finite(getattr(solved,'fun',None))
    best=_finite(getattr(solved,'mip_dual_bound',None)) if status in (0,1) else None
    result=dict(success=False,status=status,stop_reason=str(solved.message),backend=getattr(solved,'backend_used',backend),
        solver_incumbent=raw_inc,solver_best_bound=best,solver_relative_gap=_finite(getattr(solved,'mip_gap',None)),
        operation_x=None,power_kw=None,physical_audit=None,operation_audit=None,master_audit=None,
        frozen_C=None,proxy_value=None,proxy_uniform_error=master['proxy_uniform_error'],
        frozen_C_anchor=frozen_bc_value(bank,jacobian,anchor,anchor)['C'],
        shared_control_square=master['shared_control_square'],n_square_binaries=master['n_square_binaries'],
        n_clip_binaries=master['n_clip_binaries'],n_exp_binaries=0,
        model_size=dict(variables=len(master['c']),rows=master['constraints'].A.shape[0],
            operation_variables=master['operation_dimension'],integer_variables=int(np.count_nonzero(master['integrality'])),
            pair_terms=len(master['pairs'])),original_operation_constraints_preserved=True,
        build_seconds=master['build_seconds'],solver_seconds=getattr(solved,'backend_seconds',None),
        full_AC_optimality_claim=False,scope=master['scope'],failure=None)
    if getattr(solved,'x',None) is None:
        result['failure']='no_solver_incumbent';result['seconds']=perf_counter()-began;return result
    full=np.asarray(solved.x,float)
    all_op=dict(lower=master['bounds'].lb,upper=master['bounds'].ub,integrality=master['integrality'],constraints=[master['constraints']])
    audit=operations._audit(full,all_op,1e-6);x=full[:master['operation_dimension']]
    op_audit=operations._audit(x,master['operation'],1e-6)
    physical=domain.check(x,tolerance=1e-6) if hasattr(domain,'check') else op_audit
    result.update(master_audit=audit,operation_audit=op_audit,physical_audit=physical)
    if not audit['feasible'] or not op_audit['feasible'] or not physical['feasible']:
        result['failure']='incumbent_constraint_audit_failed';result['seconds']=perf_counter()-began;return result
    u=np.asarray(master['operation']['U']@x).reshape(np.asarray(anchor).shape);value=evaluate_proxy(master,u)
    tolerance=1e-6*(1+abs(value['proxy']))
    if value['actual_proxy_error'] < -tolerance or value['actual_proxy_error']>master['proxy_uniform_error']+tolerance:
        result['failure']='proxy_uniform_error_audit_failed';result['seconds']=perf_counter()-began;return result
    if raw_inc is not None and raw_inc<value['proxy']-tolerance:
        result['failure']='solver_epigraph_objective_audit_failed';result['seconds']=perf_counter()-began;return result
    if best is not None and best>value['proxy']+tolerance:
        result['failure']='solver_best_bound_audit_failed';result['seconds']=perf_counter()-began;return result
    result.update(success=True,operation_x=x.copy(),power_kw=u.copy(),frozen_C=value['C'],proxy_value=value['proxy'],
        proxy_actual_error=value['actual_proxy_error'],pair_audits=value['pairs'],
        proxy_certificate=[max(0.,value['proxy']-master['proxy_uniform_error']),value['proxy']],
        frozen_C_optimum_lower=None if best is None else max(0.,best-master['proxy_uniform_error']),
        attained_frozen_risk_upper=value['proxy'],proxy_absolute_gap=None if best is None else max(0.,value['proxy']-best),
        error_audits=[{k:a[k] for k in ('scenario','i','j','weight','logweight','epsilon_B','chord_error','rounding_guard','numeric_guard','exp_chord_error','weighted_uniform_error','form')} for a in master['pair_audits']])
    result['seconds']=perf_counter()-began
    return result
