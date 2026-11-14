"""Two-part procurement with a fixed anchor-based activation budget.

Only one linear constraint is added to the existing capacity objective/QCQP.
Its bounds apply to this restricted problem, not to uncapped procurement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..absolute_identification_20261114.solver import SymmetricRecovery, seed_items
from ..flexibility_stability_20261114 import solver as economic


ALLOWANCE_YUAN = .01
ANCHOR_NAME = '__same_forecast_150s_anchor__'
BOUND_SCOPE = ('Activation plus requested daily capacity cost for the frozen representative '
    'QCQP WITH the recorded anchor-based activation budget. It is not a lower bound '
    'for procurement without that budget, nor an AC recovery guarantee.')


def action_hash(u):
    value = np.ascontiguousarray(u, dtype=np.float64)
    digest = hashlib.sha256(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


class CallBudgetCase(economic.ObjectiveCase):
    """Capacity objective plus a linear upper bound on activation compensation."""

    def __init__(self, domain, cap_yuan, *, kappaP=.5, kappaQ=.5):
        super().__init__(domain, 'capacity', kappaP=kappaP, kappaQ=kappaQ)
        self.cap_yuan = float(cap_yuan)
        if not np.isfinite(self.cap_yuan) or self.cap_yuan < 0:
            raise ValueError('Finite nonnegative activation cap required')

    def call_audit(self, u):
        activation = economic.activation_fee(self.domain, u)
        slack = self.cap_yuan-activation
        return dict(passed=bool(activation <= self.cap_yuan), cap_yuan=self.cap_yuan,
            achieved_activation_yuan=activation, slack_yuan=slack,
            maximum_violation_yuan=max(0., -slack), allowance_tolerance_yuan=0.,
            criterion='Direct physical activation fee <= cap; no resource or solver feasibility tolerance added to the cap')

    def audit(self, u, tolerance=1e-7):
        result = super().audit(u, tolerance=tolerance)
        physical_passed = bool(result['passed'])
        call = self.call_audit(u)
        result.update(passed=bool(physical_passed and call['passed']),
            physical_passed=physical_passed, call_budget=call,
            feasibility_scope='Original physical domain AND directly recomputed activation cap')
        return result

    def attach(self, model, name='economic_service'):
        # Preserve the physical domain's own absolute-value epigraph and prices.
        u, activation = self.domain.attach(model, name=name)
        model.addConstr(activation <= self.cap_yuan, name=name+'_activation_budget')
        reserve = model.addMVar(self.R, lb=0., ub=self.capacity_upper, name=name+'_daily_capacity')
        for h in range(self.H):
            model.addConstr(u[h] <= reserve, name=name+f'_reserve_positive_{h}')
            model.addConstr(-u[h] <= reserve, name=name+f'_reserve_negative_{h}')
        price = np.r_[np.full(self.P_R, self.kappaP), np.full(self.R-self.P_R, self.kappaQ)]
        return u, activation+price@reserve


def _record(directory, receipt):
    if directory is None:
        return
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'call_budget_receipt.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
            stream.write('\n')
        receipt['call_budget_receipt'] = economic.source_ref(path)
    except Exception as exc:
        receipt.setdefault('call_budget_recording_errors', []).append(repr(exc))


def plan(case, score, validation, forecast, base_kw, *, anchor, env,
         seed_points=None, seconds=300., record_directory=None,
         kappaP=.5, kappaQ=.5, allowance_yuan=ALLOWANCE_YUAN,
         anchor_source=None, mode='capacity_call_budget'):
    """Use caller's same-forecast portfolio and150s anchor; never load executions."""
    began = perf_counter()
    if mode not in ('capacity_call_budget', 'capacity'):
        raise ValueError('Only the capped total-procurement objective is implemented')
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive finite total quote budget required')
    if float(allowance_yuan) != ALLOWANCE_YUAN:
        raise ValueError('The locked activation allowance is exactly0.01 yuan')
    domain = economic.physical_case(case)
    sources = dict(wrapper=economic.source_ref(__file__),
        economic_wrapper=economic.source_ref(economic.__file__),
        frozen_engine=economic.source_ref(economic.engine.__file__))
    common = dict(mode='capacity_call_budget', objective_kind='capacity',
        evidence_mode='symmetric_guard', objective_cost_object='activation_plus_daily_reserved_capacity',
        bound_cost_object='activation_plus_daily_reserved_capacity',
        initial_and_final_cost_object='activation_plus_daily_reserved_capacity',
        bound_problem_scope=BOUND_SCOPE, native_objective_and_bound_scope=BOUND_SCOPE,
        global_optimality_scope=BOUND_SCOPE, infeasibility_scope=BOUND_SCOPE,
        lower_bound_valid_for_uncapped_problem=False, activation_allowance_yuan=ALLOWANCE_YUAN,
        anchor_source=anchor_source, anchor_selection='Caller-provided same-forecast150s activation anchor; never replaced by another reference',
        seed_scope='Only caller-provided same-forecast shared portfolio plus the supplied anchor; runner owns sealed-source provenance checks',
        generated_new_seeds=False, actual_future_outcomes_read=False,
        capacity_cost_in_objective=True, original_evidence_constraints_changed=False,
        additional_capacity_variables=domain.R, additional_linear_constraints=2*domain.H*domain.R+1,
        added_quadratic_constraints=0, kappaP_yuan_per_kw_day=float(kappaP),
        kappaQ_yuan_per_kvar_day=float(kappaQ), source_chain=sources,
        solver_source=sources['wrapper'], engine_source=sources['frozen_engine'],
        call_budget_recording_errors=[], total_requested_seconds=float(seconds),
        time_scope='Includes wrapper anchor/seed/direct audits, model, native solve and logging; caller separately accounts for prior shared-seed and150s anchor work')

    def finish(receipt, point):
        recording_errors = receipt.get('call_budget_recording_errors', [])
        receipt.update(common)
        receipt['call_budget_recording_errors'] = recording_errors
        receipt['seconds'] = perf_counter()-began
        _record(record_directory, receipt)
        receipt['seconds'] = perf_counter()-began
        return receipt, point

    try:
        if anchor is None:
            raise ValueError('No same-forecast150s anchor supplied')
        anchor = np.asarray(domain.array(anchor), float).copy()
        evidence = SymmetricRecovery(domain, score, validation, forecast, base_kw, 'symmetric_guard', .99)
        anchor_physical = domain.audit(anchor)
        anchor_direct = evidence.verify_expansion(anchor)
        anchor_activation = economic.activation_fee(domain, anchor)
        if not (anchor_physical['passed'] and anchor_direct['feasible']):
            raise ValueError('The fixed150s anchor fails original physical/evidence checks')
    except Exception as exc:
        return finish(dict(feasible=False, status='invalid_or_missing_anchor', error=repr(exc),
            optimizer_invoked=False, cost_yuan=None, lower_bound_yuan=None,
            global_optimality_verified=False, infeasibility_proved=False), None)
    cap = anchor_activation+ALLOWANCE_YUAN
    objective = CallBudgetCase(domain, cap, kappaP=kappaP, kappaQ=kappaQ)
    common.update(activation_cap_yuan=cap, anchor_activation_yuan=anchor_activation,
        anchor_total_yuan=objective.fee(anchor), anchor_action_sha256=action_hash(anchor),
        anchor_original_physical=anchor_physical, anchor_original_evidence=anchor_direct)
    accepted, records = {}, []
    for name, candidate in seed_items(seed_points):
        try:
            if name == ANCHOR_NAME:
                raise ValueError('Reserved anchor key in shared seed pool')
            u = np.asarray(domain.array(candidate), float).copy()
            check = objective.audit(u)
            valid = bool(check['passed'])
            direct = evidence.verify_expansion(u) if valid else None
            row = dict(name=name, action_sha256=action_hash(u), accepted_by_resource_and_call_cap=valid,
                resource_feasible=check['physical_passed'], call_budget=check['call_budget'],
                evidence_feasible=None if direct is None else bool(direct['feasible']))
            if valid:
                accepted[name] = u
            records.append(row)
        except Exception as exc:
            records.append(dict(name=name, accepted_by_resource_and_call_cap=False, error=repr(exc)))
    accepted[ANCHOR_NAME] = anchor.copy()
    common.update(seed_filter=records, seed_count_supplied=len(records),
        seed_count_accepted_without_anchor=len(accepted)-1, seed_count_with_anchor=len(accepted),
        anchor_added_to_initial_pool=True)
    remaining = seconds-(perf_counter()-began)
    if remaining > 0:
        try:
            receipt, point = economic.engine.plan(objective, score, validation, forecast, base_kw,
                mode='symmetric_guard', seconds=remaining, env=env, seed_points=accepted,
                target_mass=.99, record_directory=record_directory)
        except Exception as exc:
            receipt, point = dict(status='engine_exception_verified_seed_retained',
                engine_exception=repr(exc), optimizer_invoked=None, native_status=None,
                lower_bound_yuan=0., global_optimality_verified=False), None
    else:
        receipt, point = dict(status='budget_exhausted_before_engine', optimizer_invoked=False,
            native_status=None, lower_bound_yuan=0., global_optimality_verified=False), None
    receipt.update(engine_returned_action_sha256=None if point is None else action_hash(point),
        fallback_used=False, fallback_reason=None, fallback_seed_name=None)
    if point is not None and record_directory is not None:
        try:
            destination = Path(record_directory)/'engine_returned_action.npz'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            np.savez_compressed(destination, action=np.asarray(point))
            receipt['engine_returned_action'] = economic.source_ref(destination)
        except Exception as exc:
            receipt.setdefault('call_budget_recording_errors', []).append(repr(exc))
    # Independently check the engine's returned point, including an exact cap comparison.
    final_audit = None
    if point is not None:
        try:
            physical = domain.audit(point)
            original = evidence.verify_expansion(point)
            call = objective.call_audit(point)
            final_audit = dict(passed=bool(physical['passed'] and original['feasible'] and call['passed']),
                physical=physical, original_evidence=original, call_budget=call)
            if not final_audit['passed']:
                receipt['rejected_engine_point'] = final_audit
                point = None
        except Exception as exc:
            receipt['rejected_engine_point'] = dict(error=repr(exc))
            point = None
    if point is None or objective.fee(point) > objective.fee(anchor):
        reason = ('invalid_engine_return' if receipt.get('rejected_engine_point') else
                  'engine_exception' if receipt.get('engine_exception') else
                  'no_engine_point' if point is None else 'anchor_cheaper_than_engine_return')
        if point is not None:
            receipt['engine_point_more_expensive_than_anchor'] = True
        point = anchor.copy()
        receipt.update(feasible=True, engine_status=receipt.get('status'),
            status='anchor_retained_after_'+reason, source='same_forecast_anchor_retained', retained_seed_name=ANCHOR_NAME,
            fallback_used=True, fallback_reason=reason, fallback_seed_name=ANCHOR_NAME,
            infeasibility_proved=False)
        if receipt.get('native_status') == 3:
            receipt['native_infeasibility_withdrawn'] = True
    physical = domain.audit(point)
    original = evidence.verify_expansion(point)
    call = objective.call_audit(point)
    if not (physical['passed'] and original['feasible'] and call['passed']):
        raise ArithmeticError('Final anchor/seed fallback failed independent direct checks')
    costs = economic.cost_breakdown(domain, point, kappa_p=kappaP, kappa_q=kappaQ)
    cost = costs['planned_total_yuan']
    bound = receipt.get('lower_bound_yuan')
    bad_bound = bound is not None and (not np.isfinite(bound) or bound > cost+1e-7*max(1., cost))
    if bad_bound:
        receipt.update(lower_bound_withdrawn=True, lower_bound_withdrawal_reason='bound_conflicts_directly_verified_capped_incumbent')
    lower = 0. if bound is None or bad_bound else max(0., min(cost, float(bound)))
    receipt.update(feasible=True, cost_yuan=cost, lower_bound_yuan=lower,
        relative_gap=(cost-lower)/max(cost, 1e-12), cost_breakdown=costs,
        activation_cost_yuan=costs['activation_yuan'], availability_cost_yuan=costs['availability_yuan'],
        planned_total_yuan=cost, purchased_capacity_P_kw=costs['requested_peak_P_kw'],
        purchased_capacity_Q_kvar=costs['requested_peak_Q_kvar'],
        call_budget=call, final=original, resource_audit=objective.audit(point),
        final_independent_audit=dict(passed=True, physical=physical, original_evidence=original, call_budget=call),
        final_action_sha256=action_hash(point), improvement_from_anchor_total_yuan=objective.fee(anchor)-cost)
    if bad_bound or receipt.get('rejected_engine_point') or receipt.get('engine_exception'):
        receipt['global_optimality_verified'] = False
    if 'initial' in receipt:
        receipt['initial']['cost_object'] = common['objective_cost_object']
        name = receipt['initial'].get('name')
        if name in accepted and receipt['initial'].get('feasible'):
            receipt['initial']['cost_breakdown'] = economic.cost_breakdown(domain, accepted[name], kappa_p=kappaP, kappa_q=kappaQ)
    return finish(receipt, np.asarray(point).copy())
