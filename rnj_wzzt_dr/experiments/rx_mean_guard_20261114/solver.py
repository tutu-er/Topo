"""Frozen-score QCQP with additional historical full-matrix mean gates.

All original nominal gates and scalar-history directional guards remain.
Only already fitted historical means are accepted; no fitting or truth access.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import norm

from ..absolute_identification_20261114.solver import SymmetricRecovery, seed_items
from ..customer_flexibility_20261113.placement import terms
from ..flexibility_extended_solve_20261114.solver import (
    _Recorder, _number, _reference, buffered_direct_audit as old_buffered_audit)
from ..flexibility_reaudit_20261114.reduced_solver import BUFFER, reductions


class MeanRecovery(SymmetricRecovery):
    """One declared historical mean per frozen representative candidate."""

    def __init__(self, case, score, validation, forecast, base_kw, means, target_mass=.99):
        super().__init__(case, score, validation, forecast, base_kw,
                         mode='symmetric_guard', target_mass=target_mass)
        self.means = np.asarray(means, dtype=float).copy()
        if self.means.shape != self.H.shape or not np.isfinite(self.means).all():
            raise ValueError('means must be finite and aligned exactly with score.parameters H')
        self.mean_rows = []
        self.mean_q, self.mean_a, self.mean_c, self.mean_h = [], [], [], []
        targets = set(self.targets)
        for pair_index, (i, j) in enumerate(self.pairs):
            for left, right in ((i, j), (j, i)):
                if left not in targets:
                    continue
                mean = self.means[left]
                q, a = terms(case, self.H[left], self.H[right], mean,
                             self.forecast, self.sigma2, self.base_kw)
                d = self.forecast @ (self.H[left]-self.H[right]).T
                b = self.forecast @ (mean-self.H[left]).T
                self.mean_q.append(q)
                self.mean_a.append(a)
                self.mean_c.append(float(np.sum(d*d)/(2*self.sigma2)+np.sum(b*d)/self.sigma2))
                self.mean_h.append(float((self.history[right]-self.history[left])/(2*self.sigma2)))
                self.mean_rows.append(dict(pair_index=pair_index, target=left, competitor=right))
        for field in ('mean_q', 'mean_a', 'mean_c', 'mean_h'):
            setattr(self, field, np.asarray(getattr(self, field)))

    def direct(self, u):
        info = super().direct(u)
        original_feasible = bool(info['feasible'])
        power = self.forecast+self.case.shift_pq_pu(u, self.base_kw)
        mu, margin, failure, pair_J = [], [], [], []
        totals = {str(i): 0. for i in self.targets}
        for k, row in enumerate(self.mean_rows):
            i, j, p = row['target'], row['competitor'], row['pair_index']
            # Independent difference of full squared errors, not the expanded QCQP.
            expected = power @ self.means[i].T
            loss_i = np.sum((expected-power @ self.H[i].T)**2)
            loss_j = np.sum((expected-power @ self.H[j].T)**2)
            value = float(self.mean_h[k]+(loss_j-loss_i)/(2*self.sigma2))
            J = float(info['J'][p])
            residual = float(value-self.z*np.sqrt(2*J)-self.tie)
            fail = float(norm.cdf((self.tie-value)/np.sqrt(2*J))) if J > 0 else float(value <= self.tie)
            mu.append(value); margin.append(residual); failure.append(fail); pair_J.append(J)
            totals[str(i)] += fail
        mean_ok = bool(np.all(np.asarray(margin) >= -1e-7)
                       and np.all(np.asarray(failure) <= self.alpha+1e-9))
        info.update(original_feasible=original_feasible, mean_feasible=mean_ok,
            feasible=bool(original_feasible and mean_ok), mean_mu=np.asarray(mu),
            mean_pair_J=np.asarray(pair_J), mean_margins=np.asarray(margin),
            minimum_mean_margin=min(margin) if margin else None,
            mean_pair_failure=np.asarray(failure), mean_target_failure_sums=totals,
            mean_worst_recovery_lower=max(0., 1-max(totals.values(), default=0.)))
        return info

    def verify_expansion(self, u):
        u = self.case.array(u)
        info = super().verify_expansion(u)
        if self.mean_rows:
            expanded = (np.einsum('hi,kij,hj->k', u, self.mean_q, u)
                        +2*np.einsum('khi,hi->k', self.mean_a, u)
                        +self.mean_c+self.mean_h)
            if not np.allclose(expanded, info['mean_mu'], atol=1e-8, rtol=1e-8):
                raise ArithmeticError('Historical mean direct SSE difference and QCQP mismatch')
            info['maximum_mean_expansion_error'] = float(np.max(np.abs(expanded-info['mean_mu'])))
        else:
            info['maximum_mean_expansion_error'] = 0.
        return info


def buffered_direct_audit(case, evidence, u, reduction=None):
    reduction = reductions(evidence) if reduction is None else reduction
    audit = old_buffered_audit(case, evidence, u, reduction)
    info = evidence.verify_expansion(u)
    margins = np.asarray(info['mean_margins'])-BUFFER
    rows = []
    for k, row in enumerate(evidence.mean_rows):
        rows.append(dict(row, mu=float(info['mean_mu'][k]),
            J=float(info['mean_pair_J'][k]),
            original_margin=float(info['mean_margins'][k]),
            buffered_margin=float(margins[k]), failure=float(info['mean_pair_failure'][k])))
    audit.update(passed_original_acceptance=bool(audit['resource_audit']['passed'] and info['original_feasible']),
        passed_augmented_acceptance=bool(audit['resource_audit']['passed'] and info['feasible']),
        mean_rows=rows, minimum_mean_buffered_margin=float(margins.min()) if len(margins) else None,
        maximum_mean_buffered_violation=max(0., float(-margins.min())) if len(margins) else 0.,
        strict_buffered_mean_nonnegative=bool(np.all(margins >= 0)),
        mean_target_failure_sums=info['mean_target_failure_sums'],
        mean_worst_recovery_lower=info['mean_worst_recovery_lower'],
        maximum_mean_expansion_error=info['maximum_mean_expansion_error'],
        acceptance_scope='Original gates and scalar guards plus every historical-mean gate: unbuffered margin >= -1e-7 and failure <= alpha+1e-9; encoded gates retain original 1e-8 buffer')
    return audit


def _finish(recorder, receipt, initial=None, final=None):
    """Reuse log collection, but identify this augmented model in its receipt."""
    receipt.update(native_attributes=recorder.native_attributes,
        native_attribute_unavailable=recorder.unavailable_attributes,
        native_attribute_invalid=recorder.invalid_attributes,
        initial_buffered_direct_audit=initial, buffered_direct_audit=final,
        solver_source=None, recorder_source=None,
        raw_bound_scope='Activation-fee bound for augmented representative-bank buffered QCQP; original nominal gates, old scalar guards, and full historical mean gates all encoded; native numerical tolerances apply',
        global_optimality_scope='Only the encoded planning QCQP and its declared historical means; no actual-AC identification or parameter-estimation optimality guarantee')
    for key,path in (('solver_source',Path(__file__)),
                     ('recorder_source',Path(__file__).parents[1]/'flexibility_extended_solve_20261114/solver.py')):
        try:
            receipt[key]=_reference(path)
        except Exception as exc:
            recorder.error('hash_'+key,exc)
    recorder.event(dict(event='finish', wall_seconds=perf_counter()-recorder.began,
        status=receipt.get('status'), source=receipt.get('source'),
        initial_cost_yuan=receipt.get('initial', {}).get('cost_yuan'),
        final_cost_yuan=receipt.get('cost_yuan'), used_lower_bound_yuan=receipt.get('lower_bound_yuan'),
        lower_bound_withdrawn=receipt.get('lower_bound_withdrawn', False),
        logging_error_count=len(recorder.errors)))
    if recorder.stream is not None:
        try:
            recorder.stream.close()
        except Exception as exc:
            recorder.error('close_progress', exc)
        recorder.stream = None
    files = {}
    for name, path in (('progress', recorder.trace_path if recorder.trace_owned else None),
                       ('native_log', recorder.native_path if recorder.native_log_configured else None)):
        if path is not None:
            try:
                if path.exists(): files[name] = _reference(path)
            except Exception as exc:
                recorder.error('hash_'+name, exc)
    receipt.update(record_directory=None if recorder.directory is None else str(recorder.directory),
        diagnostic_files=files, native_log_configured=recorder.native_log_configured,
        trace_status='not_requested' if recorder.directory is None else 'recording_errors' if recorder.errors else 'complete',
        trace_rows=recorder.trace_rows, logging_errors=recorder.errors,
        seconds=perf_counter()-recorder.began)
    if recorder.directory is not None:
        try:
            destination = recorder.directory/'diagnostics.json'
            with destination.open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False,
                    default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
                stream.write('\n')
            receipt['diagnostics_receipt'] = _reference(destination)
        except Exception as exc:
            recorder.error('write_diagnostics', exc)
            receipt.update(trace_status='recording_errors', logging_errors=recorder.errors)
    receipt['seconds'] = perf_counter()-recorder.began
    return receipt


def plan(case, score, validation, forecast, base_kw, *, means, mean_source=None,
         mode='mean_absolute_guard', seconds=300., env, seed_points=None,
         target_mass=.99, record_directory=None):
    """Minimize the case's original activation fee; no feasible seed required."""
    if mode != 'mean_absolute_guard':
        raise ValueError(mode)
    if seconds <= 0 or not np.isfinite(seconds):
        raise ValueError('A positive finite computation budget is required')
    began = perf_counter()
    recorder = _Recorder(record_directory, began)
    evidence = MeanRecovery(case, score, validation, forecast, base_kw, means, target_mass)
    reduction = reductions(evidence)
    zero = np.zeros(case.shape)
    baseline = evidence.verify_expansion(zero)
    common = dict(mode=mode, objective_kind='activation', objective_cost_object='activation_yuan',
        bound_cost_object='activation_yuan', target_mass=target_mass, targets=evidence.targets,
        target_topology_ids=[score['models'][i]['topology_id'] for i in evidence.targets],
        historical_selected_mass=float(evidence.weights[evidence.targets].sum()),
        candidates=evidence.K, pair_count=len(evidence.pairs), guard_count=len(evidence.guards),
        mean_gate_count=len(evidence.mean_rows), mean_gate_directions=evidence.mean_rows,
        individual_comparison_risk=evidence.alpha, total_comparison_risk=.1, baseline=baseline,
        numerical_gate_buffer=BUFFER, mean_source=mean_source,
        means_shape=list(evidence.means.shape),
        means_sha256=hashlib.sha256(np.ascontiguousarray(evidence.means, dtype='<f8').tobytes()).hexdigest(),
        constrained_bank='frozen representative models; native-bank evaluation remains separate',
        original_gates_retained=True, original_scalar_guards_retained=True,
        confidence_scope='Conditional on selected target structure having exactly its declared historical K mean, given forecast/nominal delivery and iid Gaussian residual variance equal to frozen score variance; no bootstrap coverage, actual meter-correlation, input-noise, or AC guarantee',
        selection_scope='Historical cumulative weight target set with cutoff ties; selection does not establish true-structure coverage',
        time_scope='Includes historical-mean quadratic construction, seed auditing, diagnostics and native solve; caller adds historical fit and external common-seed cost',
        pair_reduction_for_audit=reduction)
    if baseline['feasible'] and case.audit(zero)['passed']:
        receipt = dict(common, feasible=True, status='zero_sufficient', source='zero', cost_yuan=0.,
            lower_bound_yuan=0., relative_gap=0., global_optimality_verified=True, optimizer_invoked=False)
        return _finish(recorder, receipt, final=buffered_direct_audit(case,evidence,zero,reduction)), zero

    box = []
    for k, row in enumerate(reduction):
        if row['kind'] != 'lower_bound': continue
        required, upper = row['lower_J'], float(evidence.upper[k])
        tolerance = 1e-7*max(1., required, upper)
        box.append(dict(pair=row['pair'], required_J=required, relaxed_power_box_upper_J=upper,
            comparison_margin=tolerance, impossible=bool(required > upper+tolerance)))
    common['resource_box_checks'] = box
    if any(row['impossible'] for row in box):
        receipt = dict(common, feasible=False, status='infeasible_by_resource_box_bound',
            optimizer_invoked=False, native_status=None, infeasibility_proved=True,
            infeasibility_scope='Retained nominal gate alone exceeds valid relaxed resource-box upper bound; hence augmented problem also infeasible',
            cost_yuan=None, lower_bound_yuan=None, global_optimality_verified=False)
        return _finish(recorder, receipt), None

    point, warm, chosen, merit = None, None, None, -np.inf
    records = []; tick = perf_counter()
    for name, value in seed_items(seed_points):
        try:
            candidate = case.array(value)
            domain = case.audit(candidate)
            info = evidence.verify_expansion(candidate)
            fee = float(case.fee(candidate))
            if not np.isfinite(fee) or fee < -1e-9:
                raise ValueError('Activation cost must be finite and nonnegative')
            valid = bool(domain['passed'] and info['feasible'])
            minimum = min((v for v in (info['minimum_pair_margin'], info['minimum_guard_gain'],
                                      info['minimum_mean_margin']) if v is not None), default=0.)
            records.append(dict(name=name, resource_feasible=bool(domain['passed']),
                original_qcqp_feasible=info['original_feasible'], mean_feasible=info['mean_feasible'],
                qcqp_feasible=info['feasible'], cost_yuan=fee, minimum_margin=minimum))
            if valid and (point is None or fee < case.fee(point)):
                point, chosen = candidate.copy(), name
            if domain['passed'] and minimum > merit:
                warm, merit = candidate.copy(), minimum
        except Exception as exc:
            records.append(dict(name=name, error=repr(exc), qcqp_feasible=False))
    if point is not None: warm = point.copy()
    seed = None if point is None else point.copy()
    initial = dict(source='caller_common_seed_pool', feasible=seed is not None, name=chosen,
        cost_yuan=None if seed is None else float(case.fee(seed)),
        audit_seconds=perf_counter()-tick, candidates=records)
    initial_audit = None if seed is None else buffered_direct_audit(case,evidence,seed,reduction)

    import gurobipy as gp
    with gp.Model('historical_full_mean_augmented_recovery', env=env) as model:
        for key, value in dict(OutputFlag=0, Threads=1, Seed=0, NonConvex=2,
                FeasibilityTol=1e-9, NumericFocus=1, MIPGap=1e-4).items():
            model.setParam(key, value)
        recorder.configure(model)
        u, fee = case.attach(model)
        s = []
        warm_info = None if warm is None else evidence.direct(warm)
        for k in range(len(evidence.pairs)):
            J = sum(u[h]@evidence.q[k]@u[h] for h in range(case.H))+2*(evidence.a[k]*u).sum()+evidence.c[k]
            auxiliary = model.addVar(lb=0., ub=float(np.sqrt(evidence.upper[k])*(1+1e-9)),
                                     name='pair_distance_'+str(k))
            s.append(auxiliary)
            model.addConstr(J <= auxiliary*auxiliary, name='distance_upper_aux_'+str(k))
            model.addConstr(evidence.h[k]+J >= evidence.tie+BUFFER+evidence.z*np.sqrt(2)*auxiliary,
                            name='retained_nominal_gate_'+str(k))
            if warm_info is not None:
                auxiliary.Start = float(np.sqrt(warm_info['J'][k]))
        for k, (left, right, mean, Q, a) in enumerate(evidence.guards):
            gain = sum(u[h]@Q@u[h] for h in range(case.H))+2*(a*u).sum()
            model.addConstr(gain >= 0., name='retained_scalar_guard_'+str(k))
        for k, row in enumerate(evidence.mean_rows):
            mu = (sum(u[h]@evidence.mean_q[k]@u[h] for h in range(case.H))
                  +2*(evidence.mean_a[k]*u).sum()+evidence.mean_c[k]+evidence.mean_h[k])
            model.addConstr(mu >= evidence.tie+BUFFER+evidence.z*np.sqrt(2)*s[row['pair_index']],
                            name='historical_mean_gate_'+str(k))
        model.setObjective(fee)
        if warm is not None: u.Start = warm
        model.Params.TimeLimit = max(.001, seconds-(perf_counter()-began))
        model.optimize(recorder.callback) if recorder.directory is not None else model.optimize()
        recorder.capture(model)
        reason = None
        try:
            bound = _number(model.ObjBound, gp.GRB.INFINITY)
            if bound is None: reason = 'nonfinite_or_Gurobi_infinity_sentinel'
        except (AttributeError, gp.GurobiError):
            bound, reason = None, 'native_ObjBound_unavailable'
        source = 'verified_initial_retained' if seed is not None else None
        native_audit = None
        if model.SolCount:
            candidate = np.asarray(u.X)
            native_audit = buffered_direct_audit(case,evidence,candidate,reduction)
            if native_audit['passed_augmented_acceptance']:
                if point is None or case.fee(candidate) < case.fee(point)-1e-9:
                    point = candidate
                    source = 'native_found' if seed is None else 'native_improved_initial'
        common.update(optimizer_invoked=True, native_returned_point_audit=native_audit)
        if point is None:
            receipt = dict(common, feasible=False, status='native_infeasible' if model.Status == 3 else 'unresolved',
                native_status=int(model.Status), initial=initial, infeasibility_proved=model.Status == 3,
                cost_yuan=None, lower_bound_yuan=bound, global_optimality_verified=False,
                lower_bound_withdrawn=False, lower_bound_withdrawal_reason=None,
                native_bound_unavailable_reason=reason)
        else:
            cost = float(case.fee(point))
            conflict = bound is not None and bound > cost+1e-7*max(1.,cost)
            lower = 0. if bound is None or conflict else max(0., min(cost,bound))
            gap = (cost-lower)/max(cost,1e-12)
            receipt = dict(common, feasible=True, status='feasible', source=source, native_status=int(model.Status),
                initial=initial, cost_yuan=cost, lower_bound_yuan=lower, relative_gap=gap,
                lower_bound_withdrawn=conflict, native_infeasibility_withdrawn=model.Status == 3,
                lower_bound_withdrawal_reason='bound_conflicts_verified_incumbent' if conflict else None,
                native_bound_unavailable_reason=reason,
                lower_bound_source='nonnegative_cost_universal_bound' if bound is None or conflict or bound < 0 else 'native_ObjBound',
                global_optimality_verified=gap <= 1e-4,
                improvement_from_verified_initial_yuan=None if seed is None else float(case.fee(seed)-cost),
                final=evidence.verify_expansion(point), resource_audit=case.audit(point))
    final_audit = None if point is None else buffered_direct_audit(case,evidence,point,reduction)
    return _finish(recorder,receipt,initial_audit,final_audit), point
