"""Unchanged exact distance reduction with optional private solve diagnostics.

The optimization and acceptance rules are copied from the reviewed reduced
solver. Logging failures are reported but never invalidate a solver result.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from ..absolute_identification_20261114.solver import (
    SymmetricRecovery, plan as original_plan, seed_items)
from ..flexibility_reaudit_20261114.reduced_solver import BUFFER, pair_reduction, reductions


def _number(value, infinity=1e100):
    """Gurobi's finite +/-1e100 sentinel is not a numerical bound."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and abs(value) < infinity / 2 else None


def _reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class _Recorder:
    def __init__(self, directory, began):
        self.directory = None if directory is None else Path(directory).resolve()
        self.began, self.errors, self.unavailable_attributes = began, [], []
        self.stream, self.trace_path, self.native_path = None, None, None
        self.last_runtime, self.trace_rows = None, 0
        self.native_attributes = None
        self.invalid_attributes = []
        self.native_log_configured = False
        self.trace_owned = False
        self.fallback = False
        if self.directory is not None:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                self.trace_path = self.directory / 'progress.jsonl'
                self.native_path = self.directory / 'native.log'
                if self.native_path.exists():
                    raise FileExistsError(self.native_path)
                self.stream = self.trace_path.open('x', encoding='utf-8', buffering=1)
                self.trace_owned = True
                self.event(dict(event='start', created_utc=datetime.now(timezone.utc).isoformat(),
                                wall_seconds=perf_counter()-began, interval_seconds=10.))
            except Exception as exc:
                self.error('prepare_recording', exc)

    def error(self, stage, exc):
        self.errors.append(dict(stage=stage, error=repr(exc), wall_seconds=perf_counter()-self.began))

    def event(self, row):
        if self.stream is None:
            return
        try:
            self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            self.stream.flush()
            self.trace_rows += 1
        except Exception as exc:
            self.error('write_progress', exc)
            try:
                self.stream.close()
            except Exception as close_exc:
                self.error('close_failed_progress', close_exc)
            self.stream = None

    def configure(self, model):
        if self.directory is None or self.native_path is None or self.errors:
            return
        try:
            model.Params.LogToConsole = 0
            model.Params.LogFile = str(self.native_path)
            model.Params.OutputFlag = 1
            self.native_log_configured = True
        except Exception as exc:
            self.error('configure_native_log', exc)
            try:
                model.Params.OutputFlag = 0
            except Exception as reset_exc:
                self.error('disable_failed_native_log', reset_exc)

    def callback(self, model, where):
        # Callback exceptions must never terminate the search.
        try:
            import gurobipy as gp
            if where != gp.GRB.Callback.MIP or self.stream is None:
                return
            runtime = _number(model.cbGet(gp.GRB.Callback.RUNTIME), gp.GRB.INFINITY)
            if runtime is None:
                return
            if self.last_runtime is not None and runtime-self.last_runtime < 10.:
                return
            self.last_runtime = runtime
            row = dict(event='mip', native_runtime_seconds=runtime,
                wall_seconds=perf_counter()-self.began,
                node_count=_number(model.cbGet(gp.GRB.Callback.MIP_NODCNT), gp.GRB.INFINITY),
                incumbent=_number(model.cbGet(gp.GRB.Callback.MIP_OBJBST), gp.GRB.INFINITY),
                bound=_number(model.cbGet(gp.GRB.Callback.MIP_OBJBND), gp.GRB.INFINITY))
            self.event(row)
        except Exception as exc:
            self.error('mip_callback', exc)

    def capture(self, model):
        import gurobipy as gp
        result = {}
        for name in ('Status', 'SolCount', 'ObjVal', 'ObjBound', 'ObjBoundC',
                     'Runtime', 'NodeCount', 'Work', 'MIPGap', 'MaxVio',
                     'BoundVio', 'ConstrVio', 'QConstrVio'):
            try:
                raw_value = model.getAttr(name)
                value = _number(raw_value, gp.GRB.INFINITY)
                if value is None:
                    self.invalid_attributes.append(dict(attribute=name,
                        reason='nonfinite_or_Gurobi_infinity_sentinel'))
                result[name] = int(value) if value is not None and name in ('Status', 'SolCount') else value
            except Exception as exc:
                result[name] = None
                self.unavailable_attributes.append(dict(attribute=name, reason=repr(exc)))
        self.native_attributes = result
        self.event(dict(event='native_end', wall_seconds=perf_counter()-self.began,
                        attributes=result, unavailable_attributes=self.unavailable_attributes))

    def finish(self, receipt, initial_audit=None, final_audit=None):
        receipt.update(native_attributes=self.native_attributes,
            native_attribute_unavailable=self.unavailable_attributes,
            native_attribute_invalid=self.invalid_attributes,
            initial_buffered_direct_audit=initial_audit,
            buffered_direct_audit=final_audit,
            diagnostics_scope=('Original solver fallback: no internal native-attribute or callback hook; '
                               'only outer timing and independent direct audits recorded'
                               if self.fallback else
                               'Same buffered QCQP and original acceptance tolerances; native attributes '
                               'are unmodified, reported lower_bound_yuan retains the original sanitization rule'),
            raw_bound_scope='Native bounds concern the encoded buffered representative-model QCQP, subject to solver numerical tolerances',
            solver_source=None)
        try:
            receipt['solver_source'] = _reference(__file__)
        except Exception as exc:
            self.error('hash_solver_source', exc)
        self.event(dict(event='finish', wall_seconds=perf_counter()-self.began,
            status=receipt.get('status'), source=receipt.get('source'),
            initial_cost_yuan=receipt.get('initial', {}).get('cost_yuan'),
            final_cost_yuan=receipt.get('cost_yuan'),
            used_lower_bound_yuan=receipt.get('lower_bound_yuan'),
            lower_bound_withdrawn=receipt.get('lower_bound_withdrawn', False),
            logging_error_count=len(self.errors)))
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception as exc:
                self.error('close_progress', exc)
            self.stream = None
        files = {}
        for name, path in (('progress', self.trace_path if self.trace_owned else None),
                           ('native_log', self.native_path if self.native_log_configured else None)):
            if path is not None:
                try:
                    if path.exists():
                        files[name] = _reference(path)
                except Exception as exc:
                    self.error('hash_'+name, exc)
        receipt.update(record_directory=None if self.directory is None else str(self.directory),
            diagnostic_files=files, native_log_configured=self.native_log_configured,
            trace_status=('not_requested' if self.directory is None else
                          'fallback_outer_only' if self.fallback and not self.errors else
                          'recording_errors' if self.errors else 'complete'),
            trace_rows=self.trace_rows, logging_errors=self.errors,
            seconds=perf_counter()-self.began)
        if self.directory is not None:
            try:
                destination = self.directory/'diagnostics.json'
                with destination.open('x', encoding='utf-8') as stream:
                    json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
                    stream.write('\n')
                receipt['diagnostics_receipt'] = _reference(destination)
            except Exception as exc:
                self.error('write_diagnostics', exc)
                receipt.update(trace_status='recording_errors', logging_errors=self.errors)
        receipt['seconds'] = perf_counter()-self.began
        return receipt


def buffered_direct_audit(case, evidence, u, reduction):
    """Record signed residuals; do not tighten or replace original acceptance."""
    u = case.array(u)
    info = evidence.verify_expansion(u)
    resource = case.audit(u)
    J = np.asarray(info['J'], float)
    margins = evidence.h+J-evidence.z*np.sqrt(2*J)-evidence.tie-BUFFER
    pairs = []
    for k, ((i, j), row) in enumerate(zip(evidence.pairs, reduction, strict=True)):
        pairs.append(dict(pair=[i, j], J=float(J[k]), historical_offset=float(evidence.h[k]),
            original_buffered_margin=float(margins[k]),
            original_auxiliary_s=float(np.sqrt(J[k])),
            original_auxiliary_s_upper=float(np.sqrt(evidence.upper[k])*(1+1e-9)),
            reduction_kind=row['kind'], required_J=row['lower_J'],
            J_minus_required=None if row['kind'] != 'lower_bound' else float(J[k]-row['lower_J'])))
    guards = [dict(pair=[left, right], gain=float(gain))
        for gain, (left, right, _, _, _) in zip(info['guard_gains'], evidence.guards, strict=True)]
    x = u.ravel()
    eq = case.A_eq@x-case.b_eq
    inequalities = case.A_ub@x-case.b_ub
    return dict(passed_original_acceptance=bool(resource['passed'] and info['feasible']),
        strict_buffered_pair_nonnegative=bool(np.all(margins >= 0)),
        minimum_original_buffered_margin=float(margins.min()) if len(margins) else None,
        maximum_original_buffered_violation=max(0.,float(-margins.min())) if len(margins) else 0.,
        minimum_J_minus_required=min((p['J_minus_required'] for p in pairs if p['J_minus_required'] is not None), default=None),
        pairs=pairs, guards=guards, resource_audit=resource,
        resource_residuals=dict(lower_minus_action=(case.lower-u).tolist(),
            action_minus_upper=(u-case.upper).tolist(), equality=eq.tolist(), inequality=inequalities.tolist()),
        acceptance_scope='Unchanged original direct check: pair margin >= -1e-7, pair failure <= alpha+1e-9, guard >= -1e-7; resource audit uses its unchanged tolerance',
        native_FeasibilityTol=1e-9, numerical_gate_buffer=BUFFER,
        cost_yuan=float(case.fee(u)))


def plan(case, score, validation, forecast, base_kw, *, mode='symmetric_guard',
         seconds=900., env, seed_points=None, target_mass=.99, record_directory=None):
    """Retain the reviewed reduced solver's model, starts, acceptance and bounds."""
    began = perf_counter()
    recorder = _Recorder(record_directory, began)
    evidence = SymmetricRecovery(case, score, validation, forecast, base_kw, mode, target_mass)
    reduction = reductions(evidence)
    if any(row['kind'] == 'two_intervals' for row in reduction):
        recorder.fallback = True
        recorder.event(dict(event='fallback', reason='two_intervals_use_original_solver',
                            wall_seconds=perf_counter()-began))
        receipt, point = original_plan(case, score, validation, forecast, base_kw,
            mode=mode, seconds=max(.001, seconds-(perf_counter()-began)), env=env,
            seed_points=seed_points, target_mass=target_mass)
        receipt.update(reformulation='original_solver_fallback_for_two_intervals',
            pair_reduction=reduction,
            time_scope='Includes reduction analysis and original solver; caller adds external seed generation')
        audit = None if point is None else buffered_direct_audit(case, evidence, point, reduction)
        return recorder.finish(receipt, final_audit=audit), point
    baseline = evidence.verify_expansion(np.zeros(case.shape))
    common = dict(mode=mode, target_mass=target_mass, targets=evidence.targets,
        target_topology_ids=[score['models'][i]['topology_id'] for i in evidence.targets],
        historical_selected_mass=float(evidence.weights[evidence.targets].sum()),
        candidates=evidence.K, pair_count=len(evidence.pairs), guard_count=len(evidence.guards),
        individual_comparison_risk=evidence.alpha, total_comparison_risk=.1, baseline=baseline,
        constrained_bank='frozen representative models; native-bank outcomes reported separately',
        reformulation='exact_auxiliary_elimination_no_domain_restriction', pair_reduction=reduction,
        numerical_gate_buffer=BUFFER, dropped_absolute_gates=sum(r['kind'] == 'all_J' for r in reduction),
        remaining_absolute_gates=sum(r['kind'] == 'lower_bound' for r in reduction),
        confidence_scope='Same conditional frozen-candidate iid Gaussian model as original solver',
        time_scope='Includes model, seed auditing, diagnostics and native solve; caller adds external seed loading/generation')
    if baseline['feasible']:
        point = np.zeros(case.shape)
        receipt = dict(common, feasible=True, status='zero_sufficient', source='zero', cost_yuan=0.,
            lower_bound_yuan=0., relative_gap=0., global_optimality_verified=True)
        return recorder.finish(receipt, final_audit=buffered_direct_audit(case, evidence, point, reduction)), point
    box_checks = []
    for k, row in enumerate(reduction):
        if row['kind'] != 'lower_bound':
            continue
        required, upper = row['lower_J'], float(evidence.upper[k])
        margin = 1e-7*max(1., required, upper)
        box_checks.append(dict(pair=row['pair'], required_J=required,
            relaxed_power_box_upper_J=upper, comparison_margin=margin,
            impossible=bool(required > upper+margin)))
    common['resource_box_checks'] = box_checks
    if any(row['impossible'] for row in box_checks):
        receipt = dict(common, feasible=False, status='infeasible_by_resource_box_bound',
            native_status=None, infeasibility_proved=True,
            infeasibility_source='analytic_relaxed_resource_power_box',
            infeasibility_scope='Declared buffered QCQP; a required single-high-branch distance exceeds a valid resource-box upper bound, even without energy coupling or mean guards',
            optimizer_invoked=False, cost_yuan=None, lower_bound_yuan=None, global_optimality_verified=False)
        return recorder.finish(receipt), None
    point, warm, chosen, merit = None, None, None, -np.inf
    initial_records = []; tick = perf_counter()
    for name, u in seed_items(seed_points):
        try:
            u = np.asarray(u, float); domain = case.audit(u); info = evidence.verify_expansion(u)
            valid = domain['passed'] and info['feasible']; fee = case.fee(u)
            value = min(v for v in (info['minimum_pair_margin'], info['minimum_guard_gain']) if v is not None)
            initial_records.append(dict(name=name, resource_feasible=domain['passed'],
                qcqp_feasible=info['feasible'], cost_yuan=fee, minimum_margin=value))
            if valid and (point is None or fee < case.fee(point)):
                point, chosen = u.copy(), name
            if domain['passed'] and value > merit:
                warm, merit = u.copy(), value
        except Exception as exc:
            initial_records.append(dict(name=name, error=repr(exc), qcqp_feasible=False))
    if point is not None: warm = point.copy()
    seed = None if point is None else point.copy()
    initial = dict(source='caller_common_seed_pool', feasible=seed is not None, name=chosen,
        cost_yuan=None if seed is None else case.fee(seed), audit_seconds=perf_counter()-tick,
        candidates=initial_records)
    initial_audit = None if seed is None else buffered_direct_audit(case, evidence, seed, reduction)
    import gurobipy as gp
    with gp.Model('symmetric_recovery_exact_distance_reduction', env=env) as model:
        for key, value in dict(OutputFlag=0, Threads=1, Seed=0, NonConvex=2,
                FeasibilityTol=1e-9, NumericFocus=1, MIPGap=1e-4).items():
            model.setParam(key, value)
        recorder.configure(model)
        u, fee = case.attach(model)
        for k, reduced in enumerate(reduction):
            if reduced['kind'] == 'lower_bound':
                J = sum(u[h]@evidence.q[k]@u[h] for h in range(case.H))+2*(evidence.a[k]*u).sum()+evidence.c[k]
                model.addConstr(J >= reduced['lower_J'])
        for left, right, K, Q, a in evidence.guards:
            gain = sum(u[h]@Q@u[h] for h in range(case.H))+2*(a*u).sum()
            model.addConstr(gain >= 0.)
        model.setObjective(fee)
        if warm is not None: u.Start = warm
        model.Params.TimeLimit = max(.001, seconds-(perf_counter()-began))
        model.optimize(recorder.callback) if recorder.directory is not None else model.optimize()
        recorder.capture(model)
        bound_unavailable_reason = None
        try:
            bound = _number(model.ObjBound, gp.GRB.INFINITY)
            if bound is None:
                bound_unavailable_reason = 'nonfinite_or_Gurobi_infinity_sentinel'
        except (AttributeError, gp.GurobiError):
            bound = None
            bound_unavailable_reason = 'native_ObjBound_unavailable'
        source = 'verified_initial_retained' if seed is not None else None
        if model.SolCount:
            candidate = np.asarray(u.X)
            if case.audit(candidate)['passed'] and evidence.verify_expansion(candidate)['feasible']:
                if point is None or case.fee(candidate) < case.fee(point)-1e-9:
                    point = candidate
                    source = 'native_found' if seed is None else 'native_improved_initial'
        if point is None:
            receipt = dict(common, feasible=False, status='native_infeasible' if model.Status == 3 else 'unresolved',
                native_status=int(model.Status), initial=initial, infeasibility_proved=model.Status == 3,
                cost_yuan=None, lower_bound_yuan=bound, global_optimality_verified=False,
                lower_bound_withdrawn=False, lower_bound_withdrawal_reason=None,
                native_bound_unavailable_reason=bound_unavailable_reason)
        else:
            cost = case.fee(point)
            conflict = bound is not None and bound > cost+1e-7*max(1., cost)
            lower = 0. if bound is None or conflict else max(0., min(cost, bound))
            gap = (cost-lower)/max(cost, 1e-12)
            receipt = dict(common, feasible=True, status='feasible', source=source, native_status=int(model.Status),
                initial=initial, cost_yuan=cost, lower_bound_yuan=lower, relative_gap=gap,
                lower_bound_withdrawn=conflict, native_infeasibility_withdrawn=model.Status == 3,
                lower_bound_withdrawal_reason='bound_conflicts_verified_incumbent' if conflict else None,
                native_bound_unavailable_reason=bound_unavailable_reason,
                lower_bound_source='nonnegative_cost_universal_bound' if bound is None or conflict or bound < 0 else 'native_ObjBound',
                global_optimality_verified=gap <= 1e-4,
                improvement_from_verified_initial_yuan=None if seed is None else case.fee(seed)-cost,
                final=evidence.verify_expansion(point), resource_audit=case.audit(point))
    final_audit = None if point is None else buffered_direct_audit(case, evidence, point, reduction)
    return recorder.finish(receipt, initial_audit, final_audit), point
