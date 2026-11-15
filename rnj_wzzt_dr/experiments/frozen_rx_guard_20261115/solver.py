"""Frozen distribution gates with an additional positive-semidefinite budget.

The original evidence, physical domain, prices and seed portfolio are reused.
Only the planning case is wrapped; physical execution uses the original case.
"""
from __future__ import annotations

import hashlib
import json
from math import fsum
from pathlib import Path
from time import perf_counter

import numpy as np

from ..distribution_gate_20261114 import solver as engine
from ..absolute_identification_20261114.solver import seed_items


GUARD_RELATIVE_BUFFER = 1e-8
BOUND_SCOPE = (
    'Activation cost for the original encoded distribution gates and scalar guards '
    'WITH the additional numerically buffered PSD budgets. This is not a bound '
    'for procurement without these budgets, the unbuffered boundary, AC recovery, '
    'or activation-plus-capacity settlement.')


def array_hash(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256(str(array.shape).encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


class PSDGuardCase:
    """Add u.flat(C).T Q u.flat(C) <= limit to an unchanged physical case."""

    def __init__(self, domain, guards):
        self.domain = domain
        self.shape = tuple(domain.shape)
        self.dimension = int(np.prod(self.shape))
        self.guards = []
        if not guards:
            raise ValueError('At least one explicit PSD guard is required')
        names = set()
        for number, specification in enumerate(guards):
            name = str(specification.get('id', f'guard_{number}'))
            if name in names:
                raise ValueError('Duplicate PSD guard id')
            names.add(name)
            Q = np.asarray(specification['Q'], dtype=float).copy()
            limit = float(specification['limit'])
            if Q.shape != (self.dimension, self.dimension) or not np.isfinite(Q).all():
                raise ValueError('Guard Q must be finite, full row-major action dimension')
            if not np.isfinite(limit) or limit < 0:
                raise ValueError('Guard limit must be finite and nonnegative')
            scale = float(np.max(np.abs(Q), initial=0.))
            asymmetry = float(np.max(np.abs(Q-Q.T), initial=0.))
            symmetry_tolerance = 64*np.finfo(float).eps*max(scale, np.finfo(float).tiny)
            if asymmetry > symmetry_tolerance:
                raise ValueError('Guard Q is not symmetric within arithmetic roundoff')
            original_hash = array_hash(Q)
            Q = (Q+Q.T)/2
            eigenvalues = np.linalg.eigvalsh(Q)
            spectral_scale = float(np.max(np.abs(eigenvalues), initial=0.))
            psd_tolerance = 64*np.finfo(float).eps*self.dimension*max(spectral_scale, np.finfo(float).tiny)
            if eigenvalues[0] < -psd_tolerance:
                raise ValueError('Guard Q is not positive semidefinite')
            # Symmetrization leaves the quadratic form unchanged. Never clip eigenvalues.
            encoded_Q = Q/limit if limit > 0 else Q.copy()
            if not np.isfinite(encoded_Q).all():
                raise ValueError('Guard normalization overflow')
            encoded_limit = 1.-GUARD_RELATIVE_BUFFER if limit > 0 else 0.
            Q.setflags(write=False); encoded_Q.setflags(write=False)
            self.guards.append(dict(id=name, Q=Q, encoded_Q=encoded_Q, limit=limit,
                encoded_limit=encoded_limit, encoded_limit_original_units=limit*encoded_limit if limit > 0 else 0.,
                source=specification.get('source'), units=specification.get('units', 'caller-declared squared-budget units'),
                original_matrix_sha256=original_hash, matrix_sha256=array_hash(Q),
                minimum_eigenvalue=float(eigenvalues[0]), psd_roundoff_tolerance=psd_tolerance,
                maximum_asymmetry=asymmetry, symmetrized=bool(asymmetry),
                eigenvalues_projected=False))

    def __getattr__(self, name):
        return getattr(self.domain, name)

    def metadata(self):
        return [dict(**{k:v for k,v in guard.items() if k not in ('Q','encoded_Q')},
            dimension=self.dimension, flatten_order='C: hour-major, original resource columns',
            relative_model_buffer=GUARD_RELATIVE_BUFFER if guard['limit'] > 0 else 0.) for guard in self.guards]

    def guard_audit(self, u):
        """Independent direct quadratic sum, without the encoded normalization."""
        flat = np.asarray(self.domain.array(u), float).ravel(order='C')
        if flat.size != self.dimension or not np.isfinite(flat).all():
            raise ValueError('Invalid action for PSD guard')
        rows=[]
        for guard in self.guards:
            terms = (flat[:, None]*guard['Q']*flat[None, :]).ravel()
            value = float(fsum(terms.tolist()))
            absolute_sum = float(fsum(np.abs(terms).tolist()))
            roundoff = 64*np.finfo(float).eps*max(absolute_sum, guard['limit'], np.finfo(float).tiny)
            limit = guard['limit']
            rows.append(dict(id=guard['id'], value=value, limit=limit, slack=limit-value,
                violation=max(0.,value-limit), roundoff_tolerance=roundoff,
                normalized_value=value/limit if limit > 0 else None,
                encoded_limit_original_units=guard['encoded_limit_original_units'],
                passed=bool(value <= limit+roundoff),
                encoded_budget_passed=bool(value <= guard['encoded_limit_original_units']+roundoff)))
        return dict(passed=all(r['passed'] for r in rows),
            encoded_budget_passed=all(r['encoded_budget_passed'] for r in rows), rows=rows,
            criterion='Direct original quadratic <= original limit plus arithmetic roundoff only; physical feasibility tolerance is not added')

    def audit(self, u, tolerance=1e-7):
        original = dict(self.domain.audit(u, tolerance=tolerance))
        physical_passed = bool(original['passed'])
        guard = self.guard_audit(u)
        original.update(passed=bool(physical_passed and guard['passed']),
            physical_passed=physical_passed, psd_guard=guard,
            feasibility_scope='Original physical domain AND additional PSD planning budgets; execution uses original physical case')
        return original

    def attach(self, model, name=None):
        u, fee = self.domain.attach(model) if name is None else self.domain.attach(model, name=name)
        flat = u.reshape(self.dimension)
        for index, guard in enumerate(self.guards):
            model.addConstr(flat@guard['encoded_Q']@flat <= guard['encoded_limit'],
                name=f'additional_psd_budget_{index}')
        return u, fee


def _record(directory, receipt):
    if directory is None:
        return
    try:
        path = Path(directory)/'guard_receipt.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False,
                default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
            stream.write('\n')
        receipt['guard_receipt'] = engine._reference(path)
    except Exception as exc:
        receipt.setdefault('guard_recording_errors', []).append(repr(exc))


def plan(case, score, validation, forecast, base_kw, *, guards,
         mode='mgf_forecast', seconds=600., env, seed_points=None, target_mass=.99,
         record_directory=None, meter_relative_sd=engine.METER_RELATIVE_SD,
         forecast_relative_sd=engine.FORECAST_RELATIVE_SD):
    """Return (receipt, point); failure returns point=None for caller zero fallback."""
    if mode not in ('normal_fixed','mgf_fixed','mgf_forecast'):
        raise ValueError('Use one of the three original distribution modes')
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive finite wrapper quote budget required')
    began = perf_counter()
    guarded = PSDGuardCase(case, guards)
    seed_records=[]
    # Diagnostic only: pass the identical original portfolio to the frozen engine.
    for name, value in seed_items(seed_points):
        try:
            action = case.array(value)
            physical = case.audit(action)
            budget = guarded.guard_audit(action)
            seed_records.append(dict(name=name, action_sha256=array_hash(action),
                physical_feasible=bool(physical['passed']), psd_guard=budget,
                physical_and_budget_feasible=bool(physical['passed'] and budget['passed'])))
        except Exception as exc:
            seed_records.append(dict(name=name, error=repr(exc), physical_and_budget_feasible=False))
    preparation = perf_counter()-began
    remaining = seconds-preparation
    if remaining <= 0:
        receipt, point = dict(feasible=False, status='budget_exhausted_before_engine',
            optimizer_invoked=False, native_status=None, infeasibility_proved=False,
            cost_yuan=None, lower_bound_yuan=0., relative_gap=None,
            global_optimality_verified=False), None
    else:
        try:
            receipt, point = engine.plan(guarded, score, validation, forecast, base_kw,
                mode=mode, seconds=remaining, env=env, seed_points=seed_points,
                target_mass=target_mass, record_directory=record_directory,
                meter_relative_sd=meter_relative_sd, forecast_relative_sd=forecast_relative_sd)
        except Exception as exc:
            receipt, point = dict(feasible=False, status='engine_exception', engine_exception=repr(exc),
                optimizer_invoked=None, native_status=None, infeasibility_proved=False,
                cost_yuan=None, lower_bound_yuan=0., relative_gap=None,
                global_optimality_verified=False), None
    receipt = dict(receipt)
    receipt['engine_seconds'] = receipt.get('seconds')
    receipt['engine_diagnostics_receipt'] = receipt.get('diagnostics_receipt')
    final = None
    if point is not None:
        final = guarded.audit(point)
        if not final['passed'] or not receipt.get('feasible'):
            receipt.update(engine_status=receipt.get('status'), status='rejected_final_guard_or_resource_audit',
                feasible=False, source=None, cost_yuan=None, relative_gap=None,
                global_optimality_verified=False, infeasibility_proved=False,
                rejected_point_sha256=array_hash(point))
            point = None
        else:
            point = np.asarray(point, float).copy()
            cost = float(case.fee(point))
            bound = receipt.get('lower_bound_yuan')
            invalid = bound is not None and (not np.isfinite(bound) or bound > cost+1e-7*max(1.,cost))
            outside_buffer = not final['psd_guard']['encoded_budget_passed']
            if invalid or outside_buffer:
                receipt.update(lower_bound_withdrawn=True,
                    lower_bound_withdrawal_reason='guard_bound_conflicts_verified_point' if invalid else 'verified_point_outside_numerically_buffered_guard',
                    lower_bound_yuan=0., lower_bound_source='nonnegative_cost_universal_bound',
                    global_optimality_verified=False)
            lower = receipt.get('lower_bound_yuan')
            lower = 0. if lower is None else max(0., min(cost, float(lower)))
            receipt.update(cost_yuan=cost, lower_bound_yuan=lower,
                relative_gap=(cost-lower)/max(cost,1e-12),
                final_action_sha256=array_hash(point))
    receipt.update(mode=mode, solver_source=engine._reference(__file__),
        engine_source=engine._reference(engine.__file__), psd_guards=guarded.metadata(),
        psd_guard_seed_audits=seed_records, final_guard_and_physical_audit=final,
        added_quadratic_constraints=len(guarded.guards), original_evidence_constraints_changed=False,
        original_objective_changed=False, seed_portfolio_modified=False, generated_new_seeds=False,
        returned_point_was_clipped=False, fallback_policy='No solver fallback to anchor; caller executes zero and records failure when point is None',
        new_guard_and_probability_feasible=bool(point is not None and receipt.get('feasible')),
        infeasibility_scope=BOUND_SCOPE, raw_bound_scope=BOUND_SCOPE, bound_problem_scope=BOUND_SCOPE,
        lower_bound_valid_for_unrestricted_problem=False,
        global_optimality_scope=BOUND_SCOPE, wrapper_preparation_seconds=preparation,
        engine_budget_seconds=max(0.,remaining), requested_wrapper_seconds=float(seconds),
        time_scope='Includes PSD checks, original portfolio diagnostics, unchanged evidence/solver, final direct budget check and logging; caller separately charges geometry preparation and sunk original seed work',
        seconds=perf_counter()-began)
    _record(record_directory,receipt)
    receipt['seconds']=perf_counter()-began
    return receipt, point
