"""Activation and two-part procurement objectives on the unchanged QCQP.

The physical domain, frozen candidates, absolute evidence and directional
guards are delegated to the existing engine. Capacity is a purchased daily
maximum adjustment, not a second charge for its positive and negative signs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..absolute_identification_20261114.solver import SymmetricRecovery, seed_items
from ..flexibility_extended_solve_20261114 import solver as engine


DEFAULT_KAPPA_P = .5  # yuan/(kW day)
DEFAULT_KAPPA_Q = .5  # yuan/(kvar day)


def source_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def physical_case(case):
    """Avoid charging an objective wrapper's reservation fee a second time."""
    while isinstance(case, ObjectiveCase):
        case = case.domain
    return case


def _prices(kappaP, kappaQ):
    values = np.asarray([kappaP, kappaQ], float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Daily capacity prices must be finite and nonnegative')
    return float(values[0]), float(values[1])


def _action(case, u):
    domain = physical_case(case)
    value = np.asarray(domain.array(u), float)
    if value.shape != (domain.H, domain.R) or not np.isfinite(value).all():
        raise ValueError('Expected a finite hour-by-resource action')
    if not 0 <= domain.P_R <= domain.R:
        raise ValueError('P_R must separate the original P and subsequent Q columns')
    return domain, value


def activation_fee(case, u):
    """Original activation compensation in yuan; never includes reservation."""
    domain, u = _action(case, u)
    return float(domain.fee(u))


def reservation_fee(case, u, kappaP=DEFAULT_KAPPA_P, kappaQ=DEFAULT_KAPPA_Q):
    """Daily capacity fee from the requested plan, paid once per resource."""
    domain, u = _action(case, u)
    kappaP, kappaQ = _prices(kappaP, kappaQ)
    peak = np.max(np.abs(u), axis=0)
    return float(kappaP*peak[:domain.P_R].sum()+kappaQ*peak[domain.P_R:].sum())


def planned_total(case, u, kappaP=DEFAULT_KAPPA_P, kappaQ=DEFAULT_KAPPA_Q):
    return cost_breakdown(case, u, kappa_p=kappaP, kappa_q=kappaQ)['total_yuan']


def settled_total(case, request, delivered, kappaP=DEFAULT_KAPPA_P, kappaQ=DEFAULT_KAPPA_Q):
    """Actual activation plus the capacity purchased BEFORE random delivery."""
    return cost_breakdown(case, request, delivered, kappa_p=kappaP, kappa_q=kappaQ)['total_yuan']


def cost_breakdown(case, requested, delivered=None, kappa_p=DEFAULT_KAPPA_P, kappa_q=DEFAULT_KAPPA_Q):
    domain, request = _action(case, requested)
    kappaP, kappaQ = _prices(kappa_p, kappa_q)
    peak = np.max(np.abs(request), axis=0)
    p_capacity = float(kappaP*peak[:domain.P_R].sum())
    q_capacity = float(kappaQ*peak[domain.P_R:].sum())
    requested_activation = activation_fee(domain, request)
    actual = request if delivered is None else _action(domain, delivered)[1]
    cells = domain.dt*(domain.fee_plus*np.maximum(actual, 0)+domain.fee_minus*np.maximum(-actual, 0))
    active_activation = float(cells[:, :domain.P_R].sum())
    reactive_activation = float(cells[:, domain.P_R:].sum())
    activation = activation_fee(domain, actual)
    if abs(active_activation+reactive_activation-activation) > 1e-8*max(1., activation):
        raise ArithmeticError('Domain activation fee does not match the stated P/Q unit prices')
    result = dict(activation_yuan=activation,
        active_activation_yuan=active_activation, reactive_activation_yuan=reactive_activation,
        availability_yuan=p_capacity+q_capacity,
        active_availability_yuan=p_capacity, reactive_availability_yuan=q_capacity,
        total_yuan=activation+p_capacity+q_capacity,
        planned_activation_yuan=requested_activation,
        planned_total_yuan=requested_activation+p_capacity+q_capacity,
        reserved_P_kw=float(peak[:domain.P_R].sum()), reserved_Q_kvar=float(peak[domain.P_R:].sum()),
        resource_reserved_capacity=peak.tolist(),
        requested_peak_P_kw=peak[:domain.P_R].tolist(),
        requested_peak_Q_kvar=peak[domain.P_R:].tolist(),
        requested_P_capacity_kw=float(peak[:domain.P_R].sum()),
        requested_Q_capacity_kvar=float(peak[domain.P_R:].sum()),
        kappaP_yuan_per_kw_day=kappaP, kappaQ_yuan_per_kvar_day=kappaQ,
        capacity_billing='Once per requested daily max-absolute adjustment, not twice for two signs',
        settlement_billing='Requested capacity remains paid; activation uses delivered actions')
    if delivered is not None:
        result.update(delivered_activation_yuan=activation,
                      settled_total_yuan=activation+p_capacity+q_capacity)
    return result


class ObjectiveCase:
    """Thin objective adapter; physical constraints remain in ``domain``."""

    def __init__(self, domain, mode='activation', *, kappaP=DEFAULT_KAPPA_P, kappaQ=DEFAULT_KAPPA_Q):
        if mode not in ('activation', 'capacity'):
            raise ValueError('Expected activation or capacity objective')
        self.domain = physical_case(domain)
        self.mode = mode
        self.kappaP, self.kappaQ = _prices(kappaP, kappaQ)
        if not 0 <= self.domain.P_R <= self.domain.R:
            raise ValueError('Expected explicit P_R and original-P-then-Q ordering')
        self.capacity_upper = np.max(np.maximum(np.abs(self.domain.lower), np.abs(self.domain.upper)), axis=0)
        if self.capacity_upper.shape != (self.domain.R,) or not np.isfinite(self.capacity_upper).all():
            raise ValueError('Finite resource-power bounds are required')
        self.objective_cost_object = ('activation_compensation' if mode == 'activation'
                                      else 'activation_plus_daily_reserved_capacity')

    def __getattr__(self, name):
        return getattr(self.domain, name)

    def fee(self, u):
        if self.mode == 'activation':
            return activation_fee(self.domain, u)
        return planned_total(self.domain, u, self.kappaP, self.kappaQ)

    def audit(self, u, tolerance=1e-7):
        result = dict(self.domain.audit(u, tolerance=tolerance))
        result.update(objective_mode=self.mode, objective_cost_object=self.objective_cost_object,
            objective_fee_yuan=self.fee(u),
            procurement_costs=cost_breakdown(self.domain, u, kappa_p=self.kappaP, kappa_q=self.kappaQ),
            service_fee_scope='Legacy service_fee_yuan, when present, remains physical-domain activation only',
            reservation_constraint_scope='For every physical action a feasible purchased peak exists within the unchanged offered power bounds')
        return result

    def attach(self, model, name='economic_service'):
        u, activation = self.domain.attach(model, name=name)
        if self.mode == 'activation':
            return u, activation
        reserve = model.addMVar(self.R, lb=0., ub=self.capacity_upper, name=name+'_daily_capacity')
        for h in range(self.H):
            model.addConstr(u[h] <= reserve, name=name+f'_reserve_positive_{h}')
            model.addConstr(-u[h] <= reserve, name=name+f'_reserve_negative_{h}')
        price = np.r_[np.full(self.P_R, self.kappaP), np.full(self.R-self.P_R, self.kappaQ)]
        return u, activation+price@reserve


def choose_anchor(case, score, validation, forecast, base_kw, *, seed_points=None):
    """Select the cheapest activation-fee seed passing original constraints.

    This is deterministic evaluation of the common supplied portfolio. It does
    not generate a new seed, call an optimizer or read actual future outcomes.
    """
    began = perf_counter(); domain = physical_case(case)
    evidence = SymmetricRecovery(domain, score, validation, forecast, base_kw, 'symmetric_guard', .99)
    point = None; name = None; selected_cost = None; rows = []
    for key, u in seed_items(seed_points):
        try:
            u = domain.array(u); physical = domain.audit(u); direct = evidence.verify_expansion(u)
            cost = activation_fee(domain, u)
            valid = bool(physical['passed'] and direct['feasible'])
            rows.append(dict(name=key, resource_feasible=physical['passed'], evidence_feasible=direct['feasible'],
                activation_cost_yuan=cost, minimum_pair_margin=direct['minimum_pair_margin'],
                minimum_guard_gain=direct['minimum_guard_gain']))
            if valid and (point is None or cost < selected_cost):
                point, name, selected_cost = np.asarray(u).copy(), key, cost
        except Exception as exc:
            rows.append(dict(name=key, error=repr(exc), resource_feasible=False, evidence_feasible=False))
    return dict(status='selected' if point is not None else 'no_original_feasible_anchor',
        name=name, activation_cost_yuan=selected_cost, candidates=rows,
        source='Original-evidence-feasible common seed with the lowest activation compensation; first supplied seed wins exact ties',
        generated_new_seeds=False, optimizer_invoked=False, future_outcomes_read=False,
        evidence_mode='symmetric_guard', target_mass=.99,
        seconds=perf_counter()-began, source_code=source_ref(__file__)), point


def _write_receipt(record_directory, receipt):
    """Preserve the engine's diagnostics; write separate objective semantics."""
    if record_directory is None:
        return
    destination = Path(record_directory)/'objective_receipt.json'
    try:
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
            stream.write('\n')
        receipt['objective_diagnostics_receipt'] = source_ref(destination)
    except Exception as exc:
        receipt.setdefault('objective_recording_errors', []).append(repr(exc))


def plan(case, score, validation, forecast, base_kw, *, mode='activation', seconds=300.,
         env, seed_points=None, record_directory=None, kappaP=DEFAULT_KAPPA_P, kappaQ=DEFAULT_KAPPA_Q):
    """Run the frozen exact-evidence solver with either procurement objective."""
    began = perf_counter()
    objective = ObjectiveCase(case, mode, kappaP=kappaP, kappaQ=kappaQ)
    receipt, point = engine.plan(objective, score, validation, forecast, base_kw,
        mode='symmetric_guard', seconds=max(.001, seconds-(perf_counter()-began)),
        env=env, seed_points=seed_points, target_mass=.99, record_directory=record_directory)
    engine_ref = receipt.get('solver_source', source_ref(engine.__file__))
    wrapper_ref = source_ref(__file__)
    receipt.update(mode=mode, objective_kind=mode, evidence_mode='symmetric_guard', objective_cost_object=objective.objective_cost_object,
        bound_cost_object=objective.objective_cost_object,
        initial_and_final_cost_object=objective.objective_cost_object,
        native_objective_and_bound_scope='Native ObjVal/ObjBound/ObjBoundC and reported cost/lower_bound concern '+objective.objective_cost_object,
        global_optimality_scope='Same frozen representative-model QCQP with the explicitly recorded procurement objective; actual recovery is evaluated separately',
        kappaP_yuan_per_kw_day=objective.kappaP, kappaQ_yuan_per_kvar_day=objective.kappaQ,
        capacity_cost_in_objective=mode == 'capacity',
        capacity_contract='Maximum offered capability is unchanged; selected s is the purchased daily adjustment amplitude. Q is bidirectional, P retains its original directional and availability limits.',
        additional_capacity_variables=objective.R if mode == 'capacity' else 0,
        capacity_variable_start='Original engine supplies u.Start; capacity epigraph variables are left for native partial-start completion' if mode == 'capacity' else 'not_applicable',
        additional_linear_constraints=2*objective.H*objective.R if mode == 'capacity' else 0,
        added_quadratic_constraints=0, original_evidence_constraints_changed=False,
        source_chain=dict(wrapper=wrapper_ref, frozen_engine=engine_ref),
        solver_source=wrapper_ref, objective_wrapper_source=wrapper_ref, engine_source=engine_ref,
        objective_recording_errors=[],
        time_scope='Objective wrapper, original evidence and seed auditing, unchanged engine solve and private diagnostics; caller separately charges common seed generation/anchor selection')
    if 'initial' in receipt:
        receipt['initial']['cost_object'] = objective.objective_cost_object
        initial_name = receipt['initial'].get('name')
        matches = [u for name, u in seed_items(seed_points) if name == initial_name]
        if receipt['initial'].get('feasible') and len(matches) == 1:
            receipt['initial']['cost_breakdown'] = cost_breakdown(case, matches[0], kappa_p=kappaP, kappa_q=kappaQ)
    if point is not None:
        costs = cost_breakdown(case, point, kappa_p=kappaP, kappa_q=kappaQ)
        independent_cost = costs['activation_yuan'] if mode == 'activation' else costs['planned_total_yuan']
        if abs(independent_cost-receipt['cost_yuan']) > 1e-8*max(1., independent_cost):
            raise ArithmeticError('Objective receipt and direct procurement fee disagree')
        receipt.update(cost_breakdown=costs, activation_cost_yuan=costs['activation_yuan'],
            availability_cost_yuan=costs['availability_yuan'], planned_total_yuan=costs['planned_total_yuan'],
            purchased_capacity_P_kw=costs['requested_peak_P_kw'], purchased_capacity_Q_kvar=costs['requested_peak_Q_kvar'])
    receipt['seconds'] = perf_counter()-began
    _write_receipt(record_directory, receipt)
    receipt['seconds'] = perf_counter()-began
    return receipt, point
