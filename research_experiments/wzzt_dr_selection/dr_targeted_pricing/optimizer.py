"""Exact finite pricing after complete train/validation loss precomputation.

The input loss table has shape (joint plans, 4 training pair states, 4
validation pair states), ordered 00,10,01,11. Each resource independently
accepts a complete paid-day task, and training/validation accepts are independent.
One paid training day and one paid validation day repeat the same quote. Other
appended days are natural observations already included in the supplied loss.

This module never imports an identifier, simulator, topology or study results.
Selecting one joint plan is mathematically equivalent to minimizing its finite
coefficient table. The MILP exposes the exact choices/budget and separates the
expensive observation-state cache from cheap price probability reweighting.
It is not an assertion of efficient full continuous-space DR optimization.
"""
from __future__ import annotations

from random import Random
import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

PAIR_STATES=((0,0),(1,0),(0,1),(1,1))


def pair_probabilities(p_i, p_j):
    values=np.asarray([p_i,p_j],dtype=float)
    if not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
        raise ValueError('participation probabilities must lie in [0,1]')
    return np.asarray([(p_i if i else 1-p_i)*(p_j if j else 1-p_j)
                       for i,j in PAIR_STATES],dtype=float)


def explicit_loss_intervals(losses, *, missing_lower, missing_upper):
    """Bound missing/nonfinite cells explicitly; do not impute a zero loss.

    The caller supplies both bounds. NaN/inf positions remain listed as missing;
    the returned lower endpoint is only a bound, never an observed outcome.
    """
    values=np.asarray(losses,dtype=float)
    if values.ndim!=3 or values.shape[1:]!=(4,4) or not values.shape[0]:
        raise ValueError('loss table must have shape (M,4,4), M>=1')
    a,b=float(missing_lower),float(missing_upper)
    if not math.isfinite(a) or not math.isfinite(b) or not 0<=a<=b:
        raise ValueError('explicit finite nonnegative missing-loss interval required')
    missing=~np.isfinite(values)
    if np.any(values[~missing]<0):raise ValueError('observed losses must be nonnegative')
    return dict(lower=np.where(missing,a,values),upper=np.where(missing,b,values),
                missing_cells=np.argwhere(missing).astype(int).tolist(),
                semantics='missing outcomes remain intervals; endpoints are not imputed outcomes')


def _losses(lower, upper):
    lo=np.asarray(lower,dtype=float)
    hi=lo.copy() if upper is None else np.asarray(upper,dtype=float)
    if lo.ndim!=3 or not lo.shape[0] or lo.shape[1:]!=(4,4) or hi.shape!=lo.shape:
        raise ValueError('aligned lower/upper loss tables of shape (M,4,4) required')
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError('NaN/failed outcomes need explicit loss intervals, not zero imputation')
    if np.any(lo<0) or np.any(lo>hi):raise ValueError('finite losses must satisfy 0<=lower<=upper')
    return lo,hi


def marginalized_gamma(loss_lower, loss_upper, validation_probabilities):
    """Gamma in the training pair after integrating the four validation states.

    Loss intervals are rectangular. Without shared observation identities this
    may be conservative; no correlation/coverage claim is made for missing loss.
    """
    lo=np.asarray(loss_lower,dtype=float);hi=np.asarray(loss_upper,dtype=float)
    p=np.asarray(validation_probabilities,dtype=float)
    if lo.shape!=(4,4) or hi.shape!=lo.shape or p.shape!=(4,):
        raise ValueError('one 4x4 table and four validation probabilities required')
    if (not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo<0)
            or np.any(lo>hi) or not np.isfinite(p).all() or np.any(p<0)
            or abs(float(p.sum())-1)>1e-10):
        raise ValueError('invalid finite loss intervals or validation law')
    a,b=lo@p,hi@p
    lower=float(a[3]+a[0]-b[1]-b[2]);upper=float(b[3]+b[0]-a[1]-a[2])
    return dict(training_state_order=['00','10','01','11'],
                validation_probabilities=p.tolist(),training_loss_lower=a.tolist(),
                training_loss_upper=b.tolist(),lower=lower,upper=upper,
                value=lower if lower==upper else None,
                scope='training-pair Gamma after selected-price validation marginalization')


def _finite(value):
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None


def _solve(objective, matrix, lower, upper, *, time_limit, tolerance):
    result=milp(np.asarray(objective,float),integrality=np.ones(len(objective),int),
        bounds=Bounds(np.zeros(len(objective)),np.ones(len(objective))),
        constraints=LinearConstraint(matrix,np.asarray(lower,float),np.asarray(upper,float)),
        options={'disp':False,'time_limit':float(time_limit),'mip_rel_gap':0.0})
    receipt=dict(status=int(result.status),message=str(result.message),
        solver_reported_optimal=int(result.status)==0,objective=_finite(result.fun),
        mip_dual_bound=_finite(getattr(result,'mip_dual_bound',None)),
        mip_gap=_finite(getattr(result,'mip_gap',None)),
        mip_node_count=_finite(getattr(result,'mip_node_count',None)),
        independent_primal_pass=False,primal_residuals=None)
    x=getattr(result,'x',None)
    if x is not None and np.isfinite(x).all():
        ax=np.asarray(matrix@x).reshape(-1)
        lo=np.asarray(lower,float);hi=np.asarray(upper,float)
        residuals=dict(lower_rows=float(np.maximum(lo[np.isfinite(lo)]-ax[np.isfinite(lo)],0).max(initial=0)),
            upper_rows=float(np.maximum(ax[np.isfinite(hi)]-hi[np.isfinite(hi)],0).max(initial=0)),
            variable_lower=float(np.maximum(-x,0).max(initial=0)),
            variable_upper=float(np.maximum(x-1,0).max(initial=0)),
            integrality=float(abs(x-np.rint(x)).max(initial=0)),
            objective_recalculation=abs(float(np.dot(objective,x))-float(result.fun)))
        receipt.update(primal_residuals=residuals,independent_primal_pass=max(residuals.values())<=tolerance)
    return result,receipt


def optimize_price_plan(loss_lower, loss_upper=None, *, module_ids=None,
        prices=(0,1,2), probabilities=(0.,.5,.75), energy_per_paid_event_kwh=.4,
        budget=1.6, seed=1031101, risk_tolerance=1e-9, fee_tolerance=1e-9,
        feasibility_tolerance=1e-7, time_limit_seconds=15.):
    """Optimize a declared finite full-plan/quote domain with two lex objectives.

    Joint-plan y and each resource's price one-hot u are binary. Product w is
    linked by all three upper inequalities and the matching lower inequality.
    The risk uses pi_train outer pi_validation, not a diagonal four-state table.
    Missing loss intervals yield bounds/possible optima. A deployable selection
    is returned only for an exact-risk table or conservative interval dominance.
    All exact lex ties are reported and resolved by a predeclared seed.
    """
    lo,hi=_losses(loss_lower,loss_upper);M=lo.shape[0]
    ids=[f'plan_{i}' for i in range(M)] if module_ids is None else list(module_ids)
    if len(ids)!=M or len(set(ids))!=M or any(not isinstance(v,str) or not v for v in ids):
        raise ValueError('unique nonempty module_ids must match the joint plans')
    quote=np.asarray(prices,dtype=float);prob=np.asarray(probabilities,dtype=float)
    if (quote.ndim!=1 or not len(quote) or prob.shape!=quote.shape
            or not np.isfinite(quote).all() or np.any(quote<0)
            or len(set(quote.tolist()))!=len(quote) or not np.isfinite(prob).all()
            or np.any(prob<0) or np.any(prob>1)):
        raise ValueError('finite unique nonnegative price menu and aligned probability menu required')
    E=float(energy_per_paid_event_kwh);B=float(budget)
    if not math.isfinite(E) or E<=0 or not math.isfinite(B) or B<0:
        raise ValueError('positive task energy and nonnegative finite budget required')
    if (not np.isfinite([risk_tolerance,fee_tolerance,feasibility_tolerance,time_limit_seconds]).all()
            or min(risk_tolerance,fee_tolerance)<0 or feasibility_tolerance<=0 or time_limit_seconds<=0):
        raise ValueError('invalid numeric tolerances/time limit')
    L=len(quote);offset=M+2*L;N=offset+M*L*L
    columns=[]
    for m in range(M):
        for i in range(L):
            for j in range(L):
                pi=pair_probabilities(prob[i],prob[j]);weights=np.outer(pi,pi)
                columns.append(dict(index=len(columns),plan_index=m,module_id=ids[m],
                    price_indices=[i,j],prices=[float(quote[i]),float(quote[j])],
                    resource_probabilities=[float(prob[i]),float(prob[j])],
                    pair_state_probabilities=pi.tolist(),
                    risk_lower=float(np.sum(weights*lo[m])),risk_upper=float(np.sum(weights*hi[m])),
                    expected_fee=float(2*E*(quote[i]*prob[i]+quote[j]*prob[j])),
                    worst_fee=float(2*E*(quote[i]+quote[j]))))
    feasible=[v for v in columns if v['worst_fee']<=B+feasibility_tolerance]
    lower_bound=min((v['risk_lower'] for v in feasible),default=None)
    rr=[];cc=[];dd=[];bl=[];bu=[]
    def row(entries,a,b):
        k=len(bl)
        for index,value in entries:
            if value:rr.append(k);cc.append(index);dd.append(float(value))
        bl.append(a);bu.append(b)
    row([(i,1) for i in range(M)],1,1)
    row([(M+i,1) for i in range(L)],1,1)
    row([(M+L+i,1) for i in range(L)],1,1)
    row([(offset+i,1) for i in range(len(columns))],1,1)
    for col in columns:
        k=offset+col['index'];m=col['plan_index'];i,j=col['price_indices']
        row([(k,1),(m,-1)],-np.inf,0)
        row([(k,1),(M+i,-1)],-np.inf,0)
        row([(k,1),(M+L+j,-1)],-np.inf,0)
        row([(k,1),(m,-1),(M+i,-1),(M+L+j,-1)],-2,np.inf)
    row([(offset+c['index'],c['worst_fee']) for c in columns],-np.inf,B)
    A=coo_matrix((dd,(rr,cc)),shape=(len(bl),N)).tocsc()
    risk=np.zeros(N);fee=np.zeros(N)
    for col in columns:risk[offset+col['index']]=col['risk_upper'];fee[offset+col['index']]=col['expected_fee']
    result,first=_solve(risk,A,bl,bu,time_limit=time_limit_seconds,tolerance=feasibility_tolerance)
    output=dict(status='optimizer_unresolved',selected=None,upper_candidate=None,
        module_ids=ids,state_order=['00','10','01','11'],paid_events=2,
        energy_per_paid_event_kwh=E,hard_budget=B,seed=int(seed),
        loss_table_exact=bool(np.array_equal(lo,hi)),
        uncertain_cells=np.argwhere(lo!=hi).astype(int).tolist(),
        finite_domain_lower_bound=lower_bound,finite_domain_gap_upper=None,
        feasible_columns=feasible,all_exact_lex_ties=None,possible_risk_optimal_columns=[],
        solver_receipts=[dict(stage='minimum_upper_risk',**first)],
        variables=dict(joint_plan=M,price_onehot=2*L,product=M*L*L,total=N),
        scope='declared finite joint-plan and repeated-quote domain; no full-parameter or statistical coverage',
        missing_semantics='loss endpoints are bounds, not imputed outcomes',
        lookup_equivalence=True,AC_calls=0,identifier_calls=0)
    if not feasible or not first['solver_reported_optimal'] or not first['independent_primal_pass']:
        return output
    enum_upper=min(v['risk_upper'] for v in feasible)
    risk_check=abs(float(result.fun)-enum_upper)
    output['independent_upper_optimality_check']=dict(value=enum_upper,absolute_difference=risk_check,
        pass_check=risk_check<=feasibility_tolerance)
    if risk_check>feasibility_tolerance:return output
    risk_best=float(result.fun)
    A2=vstack([A,coo_matrix(risk[None,:])],format='csc')
    result2,second=_solve(fee,A2,[*bl,-np.inf],[*bu,risk_best+risk_tolerance],
        time_limit=time_limit_seconds,tolerance=feasibility_tolerance)
    output['solver_receipts'].append(dict(stage='minimum_fee_among_upper_risk_ties',**second))
    eligible=[v for v in feasible if v['risk_upper']<=enum_upper+risk_tolerance]
    fee_best=min(v['expected_fee'] for v in eligible)
    finalists=[v for v in eligible if v['expected_fee']<=fee_best+fee_tolerance]
    second_check=abs(float(result2.fun)-fee_best) if result2.fun is not None else None
    output['independent_lex_fee_check']=dict(value=fee_best,absolute_difference=second_check,
        pass_check=second_check is not None and second_check<=feasibility_tolerance)
    if (not second['solver_reported_optimal'] or not second['independent_primal_pass']
            or second_check is None or second_check>feasibility_tolerance):return output
    finalists=sorted(finalists,key=lambda v:(v['module_id'],*v['prices']))
    upper=dict(Random(int(seed)).choice(finalists))
    output.update(status='interval_unresolved',upper_candidate=upper,
        finite_domain_gap_upper=max(0.,upper['risk_upper']-lower_bound),
        upper_risk_lex_ties=finalists,
        possible_risk_optimal_columns=[v for v in feasible if v['risk_lower']<=enum_upper+risk_tolerance])
    exact_risks=all(v['risk_lower']==v['risk_upper'] for v in feasible)
    output['risk_table_exact']=exact_risks
    if exact_risks:
        output.update(status='optimal',selected=upper,all_exact_lex_ties=finalists,
                      selection_basis='complete finite-risk table and verified two-stage MILP')
    else:
        # Sufficient, deliberately conservative proof of lex optimality for all
        # loss values in the declared intervals. Cheaper alternatives must be
        # excluded beyond the risk-tie band, not merely assigned a zero loss.
        certified=[]
        for q in feasible:
            other=[v for v in feasible if v['index']!=q['index']]
            risk_ok=all(q['risk_upper']<=v['risk_lower']+risk_tolerance for v in other)
            cheaper=[v for v in other if v['expected_fee']<q['expected_fee']-fee_tolerance]
            fee_ok=all(q['risk_upper']<v['risk_lower']-risk_tolerance for v in cheaper)
            if risk_ok and fee_ok:certified.append(q)
        output['interval_dominance_certified_columns']=certified
        if certified:
            certified=sorted(certified,key=lambda v:(v['module_id'],*v['prices']))
            chosen=dict(Random(int(seed)).choice(certified))
            output.update(status='interval_dominance_certified',selected=chosen,
                finite_domain_gap_upper=max(0.,chosen['risk_upper']-lower_bound),
                selection_basis='conservative finite interval lex dominance; unknown cells remain unknown')
    if output['selected'] is not None:
        selected=output['selected'];m=selected['plan_index']
        selected['training_Gamma']=marginalized_gamma(lo[m],hi[m],selected['pair_state_probabilities'])
        output['selected']=selected
    return output
