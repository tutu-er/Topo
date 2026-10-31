"""Finite restricted master with independent training/validation quote vectors.

For a complete joint plan m the supplied table is L[m, train_state, val_state]
with state order 00,10,01,11. Only one training day and one validation day have
paid tasks. Other appended natural days are already part of that final loss.
Each of the four resource-day offers has its own quote. State values are cached
once; pricing reweights them by pi_train outer pi_validation. No WZZT, physical
simulation or study-result imports occur in this module.

One binary column represents a complete joint plan and four-price vector.
This exact finite restricted master remains equivalent to minimizing a finite
coefficient table. It makes no full continuous-space/scalability assertion.
"""
from __future__ import annotations
import itertools
from random import Random
import math
import numpy as np
from scipy.sparse import coo_matrix, vstack
from .optimizer import _losses, _solve, pair_probabilities, marginalized_gamma


def optimize_event_prices(loss_lower, loss_upper=None, *, module_ids=None,
        prices=(0,1,2), probabilities=(0.,.5,.75), energy_per_paid_event_kwh=.4,
        budget=1.6, seed=10311134, risk_tolerance=1e-9, fee_tolerance=1e-9,
        feasibility_tolerance=1e-7, time_limit_seconds=15.):
    """Risk first, then expected fee, over complete-plan/four-offer columns.

    Returned prices have shape [[train_i,train_j],[val_i,val_j]] and integer
    entries. Missing outcomes must be supplied as explicit finite intervals;
    NaN is rejected. Intervals give risk bounds and possible optima, not imputed
    zero errors. Selection requires exact risks or conservative interval
    dominance. All exact lex ties are reported and seed-resolved.
    """
    lo,hi=_losses(loss_lower,loss_upper);M=lo.shape[0]
    ids=[f'plan_{i}' for i in range(M)] if module_ids is None else list(module_ids)
    if len(ids)!=M or len(set(ids))!=M or any(not isinstance(v,str) or not v for v in ids):
        raise ValueError('unique nonempty module_ids must match the joint plans')
    quote=np.asarray(prices,dtype=float);prob=np.asarray(probabilities,dtype=float)
    if (quote.ndim!=1 or not len(quote) or prob.shape!=quote.shape
            or not np.isfinite(quote).all() or np.any(quote<0)
            or np.any(quote!=np.rint(quote)) or len(set(quote.tolist()))!=len(quote)
            or not np.isfinite(prob).all() or np.any(prob<0) or np.any(prob>1)):
        raise ValueError('integer nonnegative price menu and aligned probabilities required')
    quote=quote.astype(int)
    E=float(energy_per_paid_event_kwh);B=float(budget)
    if not math.isfinite(E) or E<=0 or not math.isfinite(B) or B<0:
        raise ValueError('positive task energy and nonnegative finite budget required')
    if (not np.isfinite([risk_tolerance,fee_tolerance,feasibility_tolerance,time_limit_seconds]).all()
            or min(risk_tolerance,fee_tolerance)<0 or feasibility_tolerance<=0 or time_limit_seconds<=0):
        raise ValueError('invalid numeric tolerances/time limit')
    columns=[];all_vectors=list(itertools.product(range(len(quote)),repeat=4))
    for m in range(M):
        for levels in all_vectors:
            c=[int(quote[k]) for k in levels];p=[float(prob[k]) for k in levels]
            worst=E*sum(c)
            if worst>B+feasibility_tolerance:continue
            train=pair_probabilities(p[0],p[1]);valid=pair_probabilities(p[2],p[3])
            weights=np.outer(train,valid)
            columns.append(dict(index=len(columns),plan_index=m,module_id=ids[m],
                price_indices=list(levels),prices=[c[:2],c[2:]],
                resource_day_probabilities=[p[:2],p[2:]],
                training_pair_probabilities=train.tolist(),validation_pair_probabilities=valid.tolist(),
                risk_lower=float(np.sum(weights*lo[m])),risk_upper=float(np.sum(weights*hi[m])),
                expected_fee=float(E*sum(a*b for a,b in zip(c,p))),worst_fee=float(worst)))
    output=dict(status='optimizer_unresolved',selected=None,upper_candidate=None,
        module_ids=ids,state_order=['00','10','01','11'],paid_events=2,
        price_vector_order=['train_i','train_j','validation_i','validation_j'],
        energy_per_paid_event_kwh=E,hard_budget=B,seed=int(seed),
        loss_table_exact=bool(np.array_equal(lo,hi)),uncertain_cells=np.argwhere(lo!=hi).astype(int).tolist(),
        finite_domain_lower_bound=min((v['risk_lower'] for v in columns),default=None),
        finite_domain_gap_upper=None,feasible_columns=columns,
        feasible_price_vectors_per_plan=len(columns)//M,all_price_vectors_per_plan=len(all_vectors),
        all_exact_lex_ties=None,possible_risk_optimal_columns=[],solver_receipts=[],
        variables=dict(binary_complete_plan_price_columns=len(columns)),
        scope='declared finite restricted master; four offers are independent decisions; not continuous-space optimality',
        missing_semantics='endpoints bound unknown final outcomes; no zero imputation',
        lookup_equivalence=True,AC_calls=0,identifier_calls=0)
    if not columns:
        output.update(status='infeasible_finite_domain',finite_domain_empty=True,
            infeasibility_basis='all declared full-plan price vectors exceed the hard budget; no MILP was run')
        return output
    n=len(columns);risk=np.asarray([v['risk_upper'] for v in columns]);fees=np.asarray([v['expected_fee'] for v in columns])
    # Domain filtering is exact for the integer price menu. Keep the budget row
    # as an independent primal audit in addition to the one-hot selector.
    A=coo_matrix(np.vstack([np.ones(n),[v['worst_fee'] for v in columns]])).tocsc()
    lower=[1.,-np.inf];upper=[1.,B]
    first_result,first=_solve(risk,A,lower,upper,time_limit=time_limit_seconds,tolerance=feasibility_tolerance)
    output['solver_receipts'].append(dict(stage='minimum_upper_risk',**first))
    if not first['solver_reported_optimal'] or not first['independent_primal_pass']:return output
    enum_risk=min(v['risk_upper'] for v in columns)
    diff=abs(float(first_result.fun)-enum_risk)
    output['independent_upper_optimality_check']=dict(value=enum_risk,absolute_difference=diff,pass_check=diff<=feasibility_tolerance)
    if diff>feasibility_tolerance:return output
    A2=vstack([A,coo_matrix(risk[None,:])],format='csc')
    second_result,second=_solve(fees,A2,[*lower,-np.inf],[*upper,float(first_result.fun)+risk_tolerance],
        time_limit=time_limit_seconds,tolerance=feasibility_tolerance)
    output['solver_receipts'].append(dict(stage='minimum_fee_among_upper_risk_ties',**second))
    eligible=[v for v in columns if v['risk_upper']<=enum_risk+risk_tolerance]
    best_fee=min(v['expected_fee'] for v in eligible)
    fee_diff=abs(float(second_result.fun)-best_fee) if second_result.fun is not None else None
    output['independent_lex_fee_check']=dict(value=best_fee,absolute_difference=fee_diff,
        pass_check=fee_diff is not None and fee_diff<=feasibility_tolerance)
    if (not second['solver_reported_optimal'] or not second['independent_primal_pass']
            or fee_diff is None or fee_diff>feasibility_tolerance):return output
    key=lambda v:(v['module_id'],*v['prices'][0],*v['prices'][1])
    finalists=sorted([v for v in eligible if v['expected_fee']<=best_fee+fee_tolerance],key=key)
    proposed=dict(Random(int(seed)).choice(finalists))
    output.update(status='interval_unresolved',upper_candidate=proposed,
        finite_domain_gap_upper=max(0.,proposed['risk_upper']-output['finite_domain_lower_bound']),
        upper_risk_lex_ties=finalists,
        possible_risk_optimal_columns=[v for v in columns if v['risk_lower']<=enum_risk+risk_tolerance])
    exact=all(v['risk_lower']==v['risk_upper'] for v in columns);output['risk_table_exact']=exact
    if exact:
        output.update(status='optimal',selected=proposed,all_exact_lex_ties=finalists,
            selection_basis='complete finite risk table and independently verified lex MILP')
    else:
        certified=[]
        for q in columns:
            other=[v for v in columns if v['index']!=q['index']]
            if not all(q['risk_upper']<=v['risk_lower']+risk_tolerance for v in other):continue
            cheaper=[v for v in other if v['expected_fee']<q['expected_fee']-fee_tolerance]
            if all(q['risk_upper']<v['risk_lower']-risk_tolerance for v in cheaper):certified.append(q)
        output['interval_dominance_certified_columns']=certified
        if certified:
            chosen=dict(Random(int(seed)).choice(sorted(certified,key=key)))
            output.update(status='interval_dominance_certified',selected=chosen,
                finite_domain_gap_upper=max(0.,chosen['risk_upper']-output['finite_domain_lower_bound']),
                selection_basis='conservative finite interval lex dominance; unknown outcomes remain intervals')
    if output['selected'] is not None:
        chosen=output['selected'];m=chosen['plan_index']
        chosen['training_Gamma']=marginalized_gamma(lo[m],hi[m],chosen['validation_pair_probabilities'])
        reversed_gamma=marginalized_gamma(lo[m].T,hi[m].T,chosen['training_pair_probabilities'])
        chosen['validation_Gamma']=dict(validation_state_order=reversed_gamma['training_state_order'],
            training_probabilities=reversed_gamma['validation_probabilities'],
            validation_loss_lower=reversed_gamma['training_loss_lower'],
            validation_loss_upper=reversed_gamma['training_loss_upper'],
            lower=reversed_gamma['lower'],upper=reversed_gamma['upper'],value=reversed_gamma['value'],
            scope='validation-pair Gamma after selected-price training marginalization')
    return output
