"""One deterministic resource-removal pass on a verified optional initial point."""
from time import perf_counter
import numpy as np

from .line_seed import search_line
from .solver import direct_audit


def polish_resources(case, evidence, initial, *, minimum_improvement_yuan=1e-8):
    began=perf_counter()
    current=case.array(initial).copy()
    initial_audit=direct_audit(case,evidence,current)
    if not initial_audit['passed_acceptance']:
        raise ValueError('Polishing requires a directly verified initial control')
    fees=[]
    for r in range(case.R):
        isolated=np.zeros(case.shape);isolated[:,r]=current[:,r]
        fees.append(float(case.fee(isolated)))
    order=sorted(range(case.R),key=lambda r:(-fees[r],r))
    steps=[]
    for r in order:
        tick=perf_counter();endpoint=current.copy();endpoint[:,r]=0.
        before=float(case.fee(current));domain=case.audit(endpoint)
        resource_id=(case.resources[r]['id'] if hasattr(case,'resources') else str(r))
        step=dict(resource_index=r,resource_id=resource_id,ordering_activation_fee_yuan=fees[r],
            before_fee_yuan=before,endpoint_resource_audit=domain,accepted=False)
        if not domain['passed']:
            step.update(status='zeroed_endpoint_resource_infeasible',after_fee_yuan=before)
        elif np.array_equal(current,endpoint):
            step.update(status='resource_already_zero',after_fee_yuan=before)
        else:
            receipt,point=search_line(case,evidence,current,endpoint,name='remove_'+resource_id)
            step['line_search']=receipt
            if point is not None and before-case.fee(point)>minimum_improvement_yuan:
                current=point.copy();step.update(accepted=True,status='accepted_fee_reduction',
                    removal_fraction=float(receipt['selected_t']),after_fee_yuan=float(case.fee(current)))
            else:
                step.update(status='no_verified_fee_reduction',after_fee_yuan=before)
        step['seconds']=perf_counter()-tick;steps.append(step)
    final_audit=direct_audit(case,evidence,current)
    if not final_audit['passed_acceptance']:
        raise ArithmeticError('Polishing lost direct feasibility')
    return dict(feasible=True,initial_cost_yuan=float(case.fee(initial)),cost_yuan=float(case.fee(current)),
        improvement_yuan=float(case.fee(initial)-case.fee(current)),resource_order=order,steps=steps,
        accepted_steps=sum(step['accepted'] for step in steps),initial_audit=initial_audit,final_audit=final_audit,
        seconds=perf_counter()-began,optimizer_calls=0,passes=1,
        scope='One fixed fee-descending resource-column removal pass on previous planning-only initial points; no AC/outcome inputs',
        global_qcqp_optimality_claimed=False),current
