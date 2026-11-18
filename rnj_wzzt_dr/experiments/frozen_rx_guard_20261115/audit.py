"""Independent saved-array audit of frozen-R/X action-budget experiments.

The verifier does not import the new optimizer, resource wrapper or geometry
builder. Historical support designs and action-budget matrices are reconstructed
with NumPy; existing independent score/electrical verifiers are reused. No
identifier, optimization or AC solution is run by this module.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from math import fsum
from pathlib import Path
import shutil
import re
import sys
import traceback

import numpy as np

from ..distribution_panel_20261115 import audit as panel

base = panel.base
prob = panel.prob
ROOT = panel.ROOT


def policy_names(protocol):
    modes,kinds=protocol['modes'],protocol['kinds']
    if not modes or len(set(modes))!=len(modes) or not set(modes)<=set(panel.MODES):
        raise ValueError('Unknown, empty or repeated base probability modes')
    if not kinds or len(set(kinds))!=len(kinds) or not set(kinds)<={'pq','response'}:
        raise ValueError('Unknown, empty or repeated added-budget kinds')
    quotes=[m+'__'+k for m in modes for k in kinds]
    return quotes,quotes+[name+'_rotation' for name in quotes]


def array_hash(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256(str(value.shape).encode('ascii'))
    digest.update(value.tobytes())
    return digest.hexdigest()


def support_design(p, q, supports):
    """Time/output/parameter design; all R columns then all X columns."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim != 2 or p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Finite matching T by N historical P/Q required')
    t, n = p.shape
    answer = np.zeros((t, n, 2*len(supports)))
    for k, support in enumerate(supports):
        group = np.asarray(support, int)
        if not len(group) or len(set(group.tolist())) != len(group) or group.min() < 0 or group.max() >= n:
            raise ValueError('Invalid fixed support')
        answer[:, group, k] = p[:, group].sum(axis=1)[:, None]
        answer[:, group, k+len(supports)] = q[:, group].sum(axis=1)[:, None]
    return answer


def affine_history(p, q):
    """Observed training-only affine Q trend and residual RMS, with intercept."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim != 2 or p.shape != q.shape or p.shape[0] < 2:
        raise ValueError('Matching training arrays with at least two rows required')
    pc, qc = p-p.mean(axis=0), q-q.mean(axis=0)
    denom = np.square(pc).sum(axis=0)
    if np.any(denom <= 0):
        raise ValueError('Affine slope unidentified at a constant-P node')
    slope = (pc*qc).sum(axis=0)/denom
    intercept = q.mean(axis=0)-slope*p.mean(axis=0)
    residual = q-intercept-p*slope
    rms = np.sqrt(np.square(residual).mean(axis=0))
    return dict(slope=slope, intercept=intercept, residual_rms=rms,
                residual=residual, training_rows=len(p))


def parameter_sandwich(p, q, supports, covariance):
    """Unconstrained fixed-design covariance; no active-bound or support claim.

    A column-scaled SVD supplies a stable inverse of the full-rank Gram. The
    covariance sandwich itself is evaluated by whitening each output block,
    rather than importing the production geometry formula.
    """
    design = support_design(p, q, supports)
    flat = design.reshape(-1, design.shape[-1])
    scale = np.linalg.norm(flat, axis=0)
    if np.any(scale == 0):
        raise ValueError('Zero column in fixed-support historical design')
    normalized = flat/scale
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    tolerance = max(normalized.shape)*np.finfo(float).eps*singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank != flat.shape[1]:
        raise ValueError('Historical design is not full column rank')
    inverse = ((vt.T/np.square(singular))@vt)/np.outer(scale, scale)
    covariance = np.asarray(covariance, float)
    if covariance.shape != (p.shape[1], p.shape[1]):
        raise ValueError('Output noise covariance dimension differs')
    values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
    if values.min() < -1e-12*max(1., abs(values).max()):
        raise ValueError('Output covariance not PSD')
    root = np.sqrt(np.maximum(values, 0))[:, None]*vectors.T
    whitened = np.einsum('ab,tbk->tak', root, design).reshape(-1, flat.shape[1])
    factor = whitened@inverse
    result = factor.T@factor
    return dict(covariance=(result+result.T)/2, gram=flat.T@flat,
                design=design, rank=rank, parameters=flat.shape[1],
                singular_values=singular, column_norms=scale,
                rank_tolerance=tolerance, condition_column_scaled=float(singular[0]/singular[-1]))


def injection_maps(config, labels, base_kw):
    """One hourly action to nodal pu P/Q; no price or future arrays enter."""
    if not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError('Positive finite apparent-power base required')
    n, r = len(labels), len(config['resources'])
    p, q = np.zeros((n, r)), np.zeros((n, r))
    for col, (resource, unit) in enumerate(zip(config['resources'], config['action_units'], strict=True)):
        node = list(labels).index(int(resource['node_label']))
        if unit == 'kW':
            p[node, col] = 1/base_kw
            q[node, col] = float(resource.get('q_over_p', 0.))/base_kw
        elif unit == 'kvar':
            q[node, col] = 1/base_kw
        else:
            raise ValueError('Unknown resource action unit')
    return p, q


def increment_values(action, pmap, qmap, affine, covariance_rows, sigma2):
    """Direct physical-unit-free values; four observations/hour cancel in W."""
    action = np.asarray(action, float)
    dp, dq = action@pmap.T, action@qmap.T
    t, n = dp.shape
    residual = dq-dp*affine['slope']
    scales = np.asarray(affine['residual_rms'])
    if np.any(scales <= 0):
        raise ValueError('Positive nodewise historical residual scales required')
    values = {'pq': float(np.square(residual/scales).mean())}
    for native, item in covariance_rows.items():
        phi = support_design(dp, dq, item['supports'])
        value = np.einsum('tnk,kl,tnl->', phi, item['covariance'], phi, optimize=True)
        values['response_native'+str(native)] = float(value/(t*n*sigma2))
    return values


def quadratic_value(action, matrix, limit, relative_buffer=1e-8):
    """Original and encoded budgets are kept separate at roundoff precision."""
    vector = np.asarray(action, float).ravel(order='C')
    matrix = np.asarray(matrix, float)
    if matrix.shape != (vector.size, vector.size) or not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError('Invalid finite quadratic dimensions')
    if not np.isfinite(limit) or limit < 0:
        raise ValueError('Nonnegative finite budget required')
    terms = (vector[:, None]*matrix*vector[None]).ravel()
    value = float(fsum(terms.tolist()))
    absolute = float(fsum(np.abs(terms).tolist()))
    tolerance = 64*np.finfo(float).eps*max(absolute, limit, np.finfo(float).tiny)
    encoded = limit*(1-relative_buffer) if limit else 0.
    return dict(value=value, limit=float(limit), slack=float(limit-value),
                normalized_value=value/limit if limit else None,
                roundoff_tolerance=tolerance,
                encoded_limit_original_units=encoded,
                passed=bool(value <= limit+tolerance),
                encoded_budget_passed=bool(value <= encoded+tolerance))


def cost_certificate(quote, executed_activation_cost, original_feasible, buffered_feasible):
    """Validate the current point's bound object; retain contradictory raw logs."""
    accepted = bool(quote.get('new_guard_and_probability_feasible', quote.get('feasible')))
    fallback = bool(quote.get('zero_fallback', False))
    if accepted != bool(quote.get('feasible')):
        raise AssertionError('New gate acceptance and solver feasibility differ')
    if accepted and (fallback or not original_feasible):
        raise AssertionError('Fallback or invalid action presented as feasible quote')
    lower = quote.get('lower_bound_yuan')
    if lower is not None and (not np.isfinite(lower) or lower < 0):
        raise AssertionError('Invalid effective lower bound')
    if accepted:
        if not np.isclose(quote['cost_yuan'], executed_activation_cost, atol=1e-7, rtol=1e-10):
            raise AssertionError('Quote objective does not equal current activation fee')
        if lower is not None and lower > executed_activation_cost+1e-7*max(1.,executed_activation_cost):
            raise AssertionError('Effective lower bound exceeds the verified current point')
        if not buffered_feasible and lower not in (None, 0.):
            raise AssertionError('Buffered-domain lower bound paired with unbuffered incumbent')
        if quote.get('relative_gap') is not None and lower is not None:
            expected = (executed_activation_cost-lower)/max(executed_activation_cost,1e-12)
            if not np.isclose(quote['relative_gap'], expected, atol=1e-8):
                raise AssertionError('Reported gap uses a different cost or bound')
    elif quote.get('global_optimality_verified'):
        raise AssertionError('Failed quotation cannot claim a feasible global optimum')
    if quote.get('lower_bound_withdrawn') and lower not in (None, 0.):
        raise AssertionError('Withdrawn lower bound remains effective')
    return dict(accepted=accepted, zero_fallback=fallback,
                executed_activation_yuan=float(executed_activation_cost),
                feasible_upper_bound_yuan=float(executed_activation_cost) if accepted else None,
                lower_bound_yuan=lower, relative_gap=quote.get('relative_gap'),
                raw_native_attributes=quote.get('native_attributes'),
                lower_bound_withdrawn=bool(quote.get('lower_bound_withdrawn')),
                lower_bound_withdrawal_reason=quote.get('lower_bound_withdrawal_reason'),
                bound_scope=quote.get('bound_problem_scope'),
                original_domain_feasible=bool(original_feasible),
                encoded_domain_feasible=bool(buffered_feasible))


def full_native_diagnostic(ctx, values):
    """Offline truth-family margin with one history-fixed correct parameter point."""
    proper=lambda supports:{s for s in supports if 1<len(s)<ctx['n']}
    correct=[i for i,s in enumerate(ctx['active']) if proper(s)==proper(ctx['truth'])]
    foreign=[i for i in range(len(ctx['H'])) if i not in correct]
    result=dict(true_native_indices=correct,foreign_native_indices=foreign,
        target_rule='Smallest historical SSE among correct internal-family points, fixed before new-day comparison; truth is used only for this offline evaluation')
    if not correct or not foreign:
        result.update(available=False,reason='Missing correct or foreign family');return result
    target=min(correct,key=lambda i:(ctx['history'][i],i))
    result.update(available=True,true_history_fixed_target=target)
    for kind,key in (('clean','clean_sse'),('observed','observed_sse')):
        total=ctx['history']+values[key]
        rival=min(foreign,key=lambda j:(total[j],j))
        result[kind+'_strongest_foreign_native']=rival
        result[kind+'_worst_true_logodds']=float((total[rival]-total[target])/(2*ctx['sigma2']))
        result[kind+'_true_fixed_point_beats_all_foreign']=bool(total[target]<total[rival])
    return result


def native_bound_history(log_path, trace_path, verified_cost=None, buffered_feasible=False):
    """Preserve numerical bound reversals instead of trusting the terminal gap.

    Callback values retain full precision. Console values are separately parsed
    and compared with a 1e-5 allowance for their displayed decimal rounding.
    These are solver diagnostics, not new independently proved lower bounds.
    """
    callback=[];log_rows=[];final=None
    if trace_path is not None:
        with Path(trace_path).open(encoding='utf-8-sig') as stream:
            for number,line in enumerate(stream,1):
                if not line.strip():continue
                row=json.loads(line)
                if row.get('event')=='mip' and row.get('bound') is not None and np.isfinite(row['bound']):
                    callback.append(dict(line=number,bound=float(row['bound']),
                        seconds=row.get('native_runtime_seconds'),incumbent=row.get('incumbent')))
                if row.get('event')=='native_end':final=row.get('attributes')
    if log_path is not None:
        for number,line in enumerate(Path(log_path).read_text(encoding='utf-8-sig').splitlines(),1):
            tokens=line.split()
            for index,token in enumerate(tokens):
                if index>=2 and re.fullmatch(r'\d+(?:\.\d+)?%',token):
                    try:bound,incumbent=float(tokens[index-1]),float(tokens[index-2])
                    except ValueError:continue
                    if np.isfinite(bound):log_rows.append(dict(line=number,bound=bound,incumbent=incumbent,text=line.strip()))
    max_callback=max(callback,key=lambda r:r['bound']) if callback else None
    max_log=max(log_rows,key=lambda r:r['bound']) if log_rows else None
    tolerance=1e-7*max(1.,abs(verified_cost)) if verified_cost is not None else None
    conflicts=[] if verified_cost is None else [r for r in callback if r['bound']>verified_cost+tolerance]
    displayed=[] if verified_cost is None else [r for r in log_rows if r['bound']>verified_cost+max(tolerance,1e-5)]
    final_bound=(final or {}).get('ObjBound')
    reversal=(max_callback['bound']-final_bound if max_callback and final_bound is not None and np.isfinite(final_bound) else None)
    conflict=bool(conflicts or displayed)
    return dict(trace_source=None if trace_path is None else base.ref(trace_path),
        log_source=None if log_path is None else base.ref(log_path),
        callback_bound_count=len(callback),log_bound_row_count=len(log_rows),
        maximum_recorded_callback_lower_bound=max_callback,maximum_recorded_log_lower_bound=max_log,
        final_native_attributes=final,maximum_prior_to_final_bound_drop_yuan=reversal,
        verified_activation_cost=verified_cost,verified_point_within_buffered_constraints=bool(buffered_feasible),
        callback_prior_bounds_above_verified_point=len(conflicts),log_bounds_above_verified_point=len(displayed),
        first_callback_conflict=conflicts[0] if conflicts else None,last_callback_conflict=conflicts[-1] if conflicts else None,
        recorded_bound_conflict_with_verified_buffered_point=bool(conflict and buffered_feasible),
        recorded_bound_above_original_acceptance_point=conflict,
        interpretation='A historical solver bound above a subsequently verified point in the same buffered domain contradicts that bound. Terminal Status=2 or zero gap does not repair this numerical certificate inconsistency. Raw records remain unchanged.')


class Audit(panel.Audit):
    """Additional-budget audit; generic score and electrical helpers inherited."""

    def __init__(self):
        super().__init__()
        self.baseline_records = []
        self.geometry_records = []

    def geometry(self, entry, config, model, arrays, metadata, protocol):
        """Rebuild every action matrix from the original measured history."""
        train = base.native.arrays(self.checked(entry['files']['train.npz']))
        validation = base.native.arrays(self.checked(entry['files']['validation.npz']))
        labels = list(map(int, train['labels']))
        self.require(labels == model['labels'] == metadata['labels'], 'Geometry labels changed')
        self.require(metadata['training_rows'] == len(train['P']), 'Geometry training length changed')
        self.require(metadata['validation_rows'] == len(validation['P']), 'Geometry validation length changed')
        self.same('geometry_base', metadata['power_base_kva'], entry['base_kw'], atol=0, rtol=0)
        self.same('geometry_frozen_sigma', metadata['score_sigma2'], model['sigma2'], atol=0, rtol=0)
        self.same('geometry_meter_sd', metadata['meter_relative_sd'], model['meter_relative_sd'], atol=0, rtol=0)
        self.same('root_noise_variance', metadata['root_variance'], model['root_sd']**2, atol=1e-20)
        self.same('individual_noise_variance', metadata['individual_variance'], model['independent_sd']**2, atol=1e-20)
        target_native = [model['representatives'][t] for t in model['targets']]
        self.require(metadata['target_native_indices'] == target_native, 'Changed historical target set')
        self.require(metadata['target_representatives'] == model['targets'], 'Target representative map differs')
        self.same('historical_target_weights', metadata['representative_weights'], model['weights'], atol=1e-13)
        affine = affine_history(train['P'], train['Q'])
        self.same('training_affine_slope', arrays['pq_slope'], affine['slope'], atol=1e-13)
        self.same('training_affine_intercept', arrays['pq_intercept_pu'], affine['intercept'], atol=1e-15)
        self.same('training_affine_scale', arrays['pq_residual_rms_pu'], affine['residual_rms'], atol=1e-15)
        pmap, qmap = injection_maps(config, labels, entry['base_kw'])
        self.same('geometry_resource_map', arrays['resource_map_pu'], np.r_[pmap, qmap], atol=0, rtol=0)
        n, r, hours = len(labels), len(config['resources']), 24
        self.require(metadata['decision_shape'] == [hours,r] and metadata['repeats'] == 4,
                     'Action-hour or observation-repeat convention changed')
        if np.any(affine['residual_rms'] <= 0):
            raise ValueError('Nodewise residual scale is zero')
        departure = (qmap-affine['slope'][:,None]*pmap)/affine['residual_rms'][:,None]
        expected = {'pq': np.kron(np.eye(hours), departure.T@departure/(hours*n))}
        covariance_rows = {}
        for native in target_native:
            point = model['native_source']['frozen_candidates'][native]['point']
            result = parameter_sandwich(train['P'],train['Q'],point['supports'],model['covariance'])
            covariance_rows[native] = dict(supports=point['supports'], covariance=result['covariance'])
            self.same('parameter_covariance', arrays['covariance_native'+str(native)], result['covariance'], atol=1e-12,rtol=1e-9)
            self.same('historical_Gram', arrays['gram_native'+str(native)], result['gram'], atol=1e-13,rtol=1e-12)
            detail = next(t for t in metadata['targets'] if t['target_native'] == native)
            self.require(detail['supports'] == point['supports'] and detail['rank'] == result['rank'],
                         'Fixed support or historical design rank changed')
            self.require(detail['all_bound_parameters_retained'], 'Boundary coefficients omitted from covariance')
            # Independent basis-action construction, preserving the frozen resource order.
            phi = support_design(pmap.T,qmap.T,point['supports'])
            block = np.einsum('rnk,kl,snl->rs',phi,result['covariance'],phi,optimize=True)/(hours*n*model['sigma2'])
            expected['response_native'+str(native)] = np.kron(np.eye(hours),(block+block.T)/2)
        guard_rows = metadata['guards']
        self.require({g['matrix_key'] for g in guard_rows} == set(expected), 'Missing/extra action budgets')
        for row in guard_rows:
            key = row['matrix_key']; saved = np.asarray(arrays[key])
            self.same('independent_guard_matrix_'+key,saved,expected[key],atol=1e-10,rtol=1e-9)
            self.require(np.linalg.eigvalsh((saved+saved.T)/2)[0] >= -1e-10*max(1.,np.linalg.norm(saved,2)),
                         'Guard matrix not PSD')
            limit = protocol['pq_rho']**2 if row['kind']=='pq' else protocol['response_rms']**2
            self.same('predeclared_guard_limit',row['limit'],limit,atol=0,rtol=0)
        answer = dict(affine=affine,pmap=pmap,qmap=qmap,covariance_rows=covariance_rows,
                      matrices=arrays,metadata=metadata,sigma2=model['sigma2'])
        self.geometry_records.append(dict(id=entry['id'],training_rows=len(train['P']),
            target_native_indices=target_native,rank_full=True,score_frozen=True,
            guard_limits={g['id']:g['limit'] for g in guard_rows},
            interpretation='Action-increment restrictions only. W is fixed-support unconstrained design variance under the stated voltage noise; support selection, bounded-fit bias, P/Q input noise and total response bias are excluded.'))
        return answer

    def budget(self, geometry, action, kind=None):
        direct = increment_values(action,geometry['pmap'],geometry['qmap'],geometry['affine'],
                                  geometry['covariance_rows'],geometry['sigma2'])
        rows=[]
        for item in geometry['metadata']['guards']:
            if kind is not None and item['kind'] != kind:
                continue
            value = quadratic_value(action,geometry['matrices'][item['matrix_key']],item['limit'])
            self.same('direct_increment_'+item['id'],value['value'],direct[item['id']],atol=1e-10,rtol=1e-9)
            rows.append(dict(id=item['id'],kind=item['kind'],**value))
        return dict(passed=all(r['passed'] for r in rows),
                    encoded_budget_passed=all(r['encoded_budget_passed'] for r in rows),rows=rows)

    def guarded_quote(self, entry, name, quote, action, direct, budget, bill, seeds, seed_checks):
        self.references(quote)
        mode,kind = name.split('__')
        self.require(quote['mode']==mode and quote['guard_kind']==kind,'Quote identity changed')
        accepted=bool(quote['new_gate_feasible'])
        self.require(accepted==bool(quote['feasible']), 'Gate/solver feasibility flags differ')
        if accepted:
            self.require(direct['feasible'] and budget['passed'], 'Returned action violates original or added constraints')
            self.require(not quote['zero_fallback'], 'Accepted action is also marked fallback')
        else:
            self.require(quote['zero_fallback'], 'Failed new quote lacks explicit zero fallback')
            self.same('failed_quote_action_zero',action,np.zeros_like(action),atol=0,rtol=0)
        if 'executed_probability_check' in quote:
            self.require(bool(quote['executed_probability_check']['feasible'])==direct['feasible'],
                         'Executed probability flag differs from independent original formula')
        for item in budget['rows']:
            self.same('executed_added_budget',quote['executed_guard_values'][item['id']],item['value'],atol=1e-10,rtol=1e-9)
        initial=quote.get('initial') or {}
        if initial.get('feasible'):
            self.require(initial['name'] in seeds,'Claimed initial absent from the original sealed common pool')
            gate,guard,fee=seed_checks(initial['name'],mode,kind)
            self.require(gate['feasible'] and guard['passed'],'Claimed feasible initial fails direct constraints')
            self.same('initial_activation_fee',initial['cost_yuan'],fee,atol=1e-7)
            if accepted:self.require(bill['activation_yuan']<=fee+1e-6,'Accepted quote costs more than retained feasible seed')
        self.require(quote.get('seed_portfolio_modified',False) is False and quote.get('generated_new_seeds',False) is False,
                     'Method-specific initial portfolio modification')
        if quote.get('psd_guard_seed_audits'):
            self.require({s['name'] for s in quote['psd_guard_seed_audits']}==set(seeds),
                         'Incomplete common portfolio in guard seed audit')
        for seed in quote.get('psd_guard_seed_audits',[]):
            self.require(seed['name'] in seeds,'Seed receipt outside common portfolio')
            if 'action_sha256' in seed:
                self.require(seed['action_sha256']==array_hash(seeds[seed['name']]),'Seed action changed')
        if 'final_action_sha256' in quote:
            self.require(quote['final_action_sha256']==array_hash(action),'Quote final action differs from executed point')
        original_feasible=bool(direct['feasible'] and budget['passed'])
        minimum_buffer=min((r['margin']-1e-8 for r in direct['rows']),default=None)
        minimum_direction=min((r['gain'] for r in direct['guards']),default=None)
        buffered = (budget['encoded_budget_passed'] and all(r['margin']>=1e-8-1e-9 for r in direct['rows'])
                    and all(g['gain']>=-1e-9 for g in direct['guards']))
        certificate=cost_certificate(quote,bill['activation_yuan'],original_feasible,buffered)
        diagnostics=Path(self.where)/(name+'_diagnostics')
        log_path=diagnostics/'native.log';trace_path=diagnostics/'progress.jsonl'
        history=native_bound_history(self.file(log_path) if log_path.exists() else None,
            self.file(trace_path) if trace_path.exists() else None,
            bill['activation_yuan'] if accepted else None,buffered and accepted)
        certificate.update(native_bound_history=history,minimum_original_probability_buffer_margin=minimum_buffer,
            minimum_scalar_direction_margin=minimum_direction,buffer_margin_arithmetic_tolerance=1e-9,
            planner_reported_lower_bound_yuan=quote.get('lower_bound_yuan'),
            planner_reported_relative_gap=quote.get('relative_gap'),
            planner_claimed_global_optimality=quote.get('global_optimality_verified'),
            audit_withdrawal_due_to_bound_trajectory=False)
        if history['recorded_bound_conflict_with_verified_buffered_point']:
            certificate.update(lower_bound_yuan=0.,relative_gap=1. if bill['activation_yuan']>0 else 0.,
                audit_withdrawal_due_to_bound_trajectory=True,lower_bound_withdrawn=True,
                audit_global_optimality_supported=False,
                audit_lower_bound_source='Nonnegative activation cost; native trajectory contradicted a verified buffered feasible point')
        else:
            certificate['audit_global_optimality_supported']=bool(accepted and quote.get('global_optimality_verified')
                and buffered and not quote.get('lower_bound_withdrawn'))
        if quote.get('status')!='exception':
            self.require(quote['original_evidence_constraints_changed'] is False and quote['original_objective_changed'] is False,
                         'Base probability/objective changed')
            self.require(quote['lower_bound_valid_for_unrestricted_problem'] is False,'Added-domain bound claimed for unrestricted domain')
        for key in ('seconds','geometry_seconds','quote_wall_seconds','requested_quote_budget_seconds'):
            if quote.get(key) is not None:self.require(np.isfinite(quote[key]) and quote[key]>=0,'Invalid timing '+key)
        keys=('status','native_status','source','optimizer_invoked','initial','native_attributes','seconds',
              'geometry_seconds','shared_prediction_models_seconds','quote_wall_seconds','requested_quote_budget_seconds','engine_budget_seconds',
              'requested_wrapper_seconds','shared_preparation_seconds','charged_total_seconds','quote_budget_exceeded',
              'infeasibility_proved','bound_problem_scope','lower_bound_withdrawal_reason')
        self.quote_records.append(dict(**panel.identity(entry),forecast=0,policy=name,base_mode=mode,guard_kind=kind,
            **{k:quote.get(k) for k in keys},new_gate_feasible=accepted,zero_fallback=quote['zero_fallback'],
            executed_probability_feasible=direct['feasible'],executed_added_budget_feasible=budget['passed'],
            direct_gate=direct,additional_budget=budget,certificate=certificate,planned_cost=bill,
            nonzero_executed=bool(np.max(abs(action),initial=0)>1e-8)))

    def case(self, folder, entry, protocol, metadata, branches, buses):
        self.where=str(folder);ctx=self.context(entry);self.references(entry)
        seal=base.read(self.file(folder/'seal.json'));self.references(seal)
        self.require(seal['score']==entry['score'] and seal['validation']==entry['files']['validation.npz'],
                     'Frozen score or validation reference changed')
        prior=Path(entry['prior_directory']);oldseal=base.read(self.checked(entry['prior_seal']))
        self.require(oldseal['score']==entry['score'],'Old and new score source differs')
        source_planner=base.native.arrays(self.checked(oldseal['planner']))
        planner=base.native.arrays(self.checked(seal['planner']));forecast=planner['forecast']
        self.same('old_forecast_preserved',forecast,source_planner['forecast'],atol=0,rtol=0)
        names,policies=policy_names(protocol)
        self.require(set(planner)==set(policies+['forecast']),'Missing or extra new action')
        config=base.read(self.checked(seal['resource_case']))
        self.require(config==base.read(self.checked(oldseal['resource_case'])),'Original physical offers changed')
        pc=self.catalogue(entry,config,ctx);qc=len(config['resources'])-pc
        self.same('explicit_base',entry['base_kw'],ctx['base'],atol=0,rtol=0)
        noise=panel.noise_parameters(entry,protocol)
        expected=np.c_[ctx['baseline_P'],ctx['baseline_Q']]
        expected=expected*(1+noise['forecast_relative_sd']*base.native.random_stream(protocol,entry,0,0,991).normal(size=expected.shape))
        self.same('paired_forecast_random_stream',forecast,expected,atol=0,rtol=0)
        model=prob.build_model(ctx['score'],base.native.arrays(self.checked(entry['files']['validation.npz'])),ctx['native'],
            meter_relative_sd=noise['meter_relative_sd'],forecast_relative_sd=noise['forecast_relative_sd'])
        model['native_source']=ctx['native']
        geometry_meta=base.read(self.checked(seal['geometry']));self.references(geometry_meta)
        geometry_arrays=base.native.arrays(self.checked(geometry_meta['arrays']))
        geometry=self.geometry(entry,config,model,geometry_arrays,geometry_meta,protocol)
        self.input_records.append(self.coverage(entry,ctx,model))
        seeds=base.native.arrays(self.checked(entry['initial_points']))
        saved_seeds=base.native.arrays(self.file(folder/'shared_initial_points.npz'))
        self.require(set(seeds)==set(saved_seeds),'Common seed names changed')
        for key in seeds:self.same('common_seed_'+key,saved_seeds[key],seeds[key],atol=0,rtol=0)
        plans=base.read(self.checked(seal['plans']));self.require(set(plans)==set(names),'Incomplete quote set')
        saved_audits=base.read(self.checked(seal['prediction_audits']))
        kp,kq=float(protocol['reservation_prices']['P']),float(protocol['reservation_prices']['Q'])
        actions, forecasts, gates, budgets={}, {}, {}, {}
        for name in policies:
            u=planner[name]
            requested,violations,dp,dq=panel.resource_values(config,u,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
            self.same('independent_nodal_map',prob.shift(config,u,ctx['labels'],ctx['base']),np.c_[dp,dq],atol=1e-14)
            bill=base.procurement(config,u,u,kp,kq)['planned']
            actions[name]=dict(requested=requested,violations=violations,bill=bill)
            forecasts[name]=self.evaluate(ctx,forecast+np.c_[dp,dq])
            budgets[name]=self.budget(geometry,u)
            for mode in protocol['modes']:
                gates[mode,name]=self.gates(model,forecast,forecast+np.c_[dp,dq],mode,saved_audits[mode][name])
        cache={}
        def seed_check(name,mode,kind):
            key=name,mode,kind
            if key not in cache:
                u=seeds[name]
                _,_,dp,dq=panel.resource_values(config,u,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
                cache[key]=(self.gates(model,forecast,forecast+np.c_[dp,dq],mode),self.budget(geometry,u,kind),
                    base.procurement(config,u,u,kp,kq)['planned']['activation_yuan'])
            return cache[key]
        for name in names:
            self.require(plans[name]['initial_points']==entry['initial_points'],'Mode-specific seed source changed')
            saved=base.native.arrays(self.file(folder/(name+'_action.npz')))
            self.same('per_quote_frozen_action',saved['action'],planner[name],atol=0,rtol=0)
            self.require(base.read(self.file(folder/(name+'_plan.json')))==plans[name],'Per-mode quote changed after planning')
            mode,kind=name.split('__')
            self.guarded_quote(entry,name,plans[name],planner[name],gates[mode,name],
                               self.budget(geometry,planner[name],kind),actions[name]['bill'],seeds,seed_check)
        rotation=base.read(self.checked(seal['rotation']))
        self.require(set(rotation)==set(names),'Rotation identities incomplete')
        for name in names:
            a,b=actions[name]['bill']['total_yuan'],actions[name+'_rotation']['bill']['total_yuan']
            matched=abs(a-b)<=1e-7
            self.require(bool(rotation[name]['same_planned_fee'])==matched,'Same-fee declaration differs')
            self.same('rotation_target_fee',rotation[name]['target_total_yuan'],a,atol=1e-8)
            self.same('rotation_actual_fee',rotation[name]['rotation_total_yuan'],b,atol=1e-8)
            if not matched:self.fee_mismatches.append(dict(id=entry['id'],policy=name))
        done=base.read(self.file(folder/'complete.json'));self.references(done)
        rows=base.read(self.checked(done['results']))
        expected={(repeat,name) for repeat in range(protocol['repeats']) for name in policies}
        self.require(len(rows)==len(expected) and {(r['repeat'],r['strategy']) for r in rows}==expected,
                     'Execution denominator or identities differ')
        zero_forecast=self.evaluate(ctx,forecast)
        zero_by_repeat={}
        for row in rows:
            repeat,name=row['repeat'],row['strategy'];path=folder/f'execution{repeat:02d}'/name
            rec=dict(**panel.identity(entry),forecast=0,repeat=repeat,strategy=name,source=str(path),original_status=row['status'])
            try:
                self.require(base.read(self.file(path/'receipt.json'))==row,'Execution receipt differs from result ledger')
                if row['status']!='complete':
                    rec.update(audit_status='recorded_failure',error=row.get('error'),evaluation={'native_correct':False})
                    self.records.append(rec);continue
                self.references(row)
                parent=name.removesuffix('_rotation');quote=plans[parent]
                self.require(row['parent_strategy']==parent and row['mode']==quote['mode'] and row['guard_kind']==quote['guard_kind'],
                             'Executed policy identity differs')
                self.require(row['quote_zero_fallback']==quote['zero_fallback'] and
                             row['quote_new_constraints_feasible']==quote['new_gate_feasible'],'Executed quote flags differ')
                random=base.native.arrays(self.file(folder/f'execution{repeat:02d}'/'paired_randomness.npz'))
                original_random=base.native.arrays(self.file(prior/f'execution{repeat:02d}'/'paired_randomness.npz'))
                self.require(set(random)==set(original_random),'Paired random variables changed')
                for key in random:self.same('original_paired_'+key,random[key],original_random[key],atol=0,rtol=0)
                eta=np.r_[base.native.random_stream(protocol,entry,0,repeat,773).uniform(.9,1,pc),
                    base.native.random_stream(protocol,entry,0,repeat,774).uniform(.9,1,qc)]
                self.same('response_stream',random['eta'],eta,atol=0,rtol=0)
                shapes=(ctx['baseline_P'].shape,ctx['baseline_Q'].shape,ctx['baseline_P'].shape,ctx['baseline_root'].shape)
                for k,shape in enumerate(shapes):
                    self.same('noise_stream',random['noise'+str(k)],base.native.random_stream(protocol,entry,0,repeat,k+2).normal(size=shape),atol=0,rtol=0)
                obs=base.native.arrays(self.file(path/'observations.npz'));u=planner[name]
                self.same('sealed_request',obs['requested_action'],u,atol=0,rtol=0)
                delivery=u*eta[None]
                for col,resource in enumerate(config['resources'][:pc]):
                    if resource['kind']=='pv':
                        node=ctx['labels'].index(int(resource['node_label']))
                        cap=np.minimum(resource['contract_kw'],np.maximum(0,-ctx['baseline_P'][:,node]*ctx['base']).reshape(24,4).min(1))
                        delivery[:,col]=np.minimum(delivery[:,col],cap)
                self.same('paired_delivery',obs['delivered_action'],delivery,atol=1e-12,rtol=0)
                delivered,violations,dp,dq=panel.resource_values(config,delivery,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
                self.same('true_P',obs['true_P'],ctx['baseline_P']+dp,atol=1e-13)
                self.same('true_Q',obs['true_Q'],ctx['baseline_Q']+dq,atol=1e-13)
                for label,k,sd in (('P',0,noise['pq_relative_sd']),('Q',1,noise['pq_relative_sd']),
                                   ('V',2,noise['meter_relative_sd']),('root',3,noise['meter_relative_sd'])):
                    self.same('meter_'+label,obs[label],obs['true_'+label]*(1+sd*random['noise'+str(k)]),atol=1e-13)
                self.same('observed_Y',obs['Y'],obs['root'][:,None]**2-obs['V']**2,atol=1e-13)
                if 'reused_action_source' in row:
                    cached=base.native.arrays(self.checked(row['reused_action_source']))
                    for key in obs:self.same('cached_action_'+key,obs[key],cached[key],atol=0,rtol=0)
                bill=base.procurement(config,u,delivery,kp,kq)
                for when in ('planned','settled'):
                    self.costs(row[when+'_cost'],bill[when],when)
                    self.same(when+'_fee',row[when+'_fee_yuan'],bill[when]['total_yuan'],atol=1e-8)
                for field,value in (('requested_quantities',u),('delivered_quantities',delivery)):
                    self.same(field+'_P',row[field]['active_abs_kwh'],abs(value[:,:pc]).sum(),atol=1e-9)
                    self.same(field+'_Q',row[field]['reactive_abs_kvarh'],abs(value[:,pc:]).sum(),atol=1e-9)
                evaluation,values=self.score(ctx,obs,row)
                evaluation['full_native_diagnostic']=full_native_diagnostic(ctx,values)
                self.pair_file(self.file(path/'pair_evidence.npz'),ctx,values)
                if repeat not in zero_by_repeat:
                    oldfolder=prior/f'execution{repeat:02d}'/'no_dfs'
                    zero_obs=base.native.arrays(self.file(oldfolder/'observations.npz'))
                    zero_row=base.read(self.file(oldfolder/'receipt.json'))
                    zero_evaluation,zero_values=self.score(ctx,zero_obs,zero_row)
                    self.pair_file(self.file(oldfolder/'pair_evidence.npz'),ctx,zero_values)
                    zero_by_repeat[repeat]=(zero_obs,zero_evaluation,zero_values)
                zero_obs,zero_evaluation,zero_values=zero_by_repeat[repeat]
                self.reported_evidence(ctx,row['evidence'],values,zero_values,forecasts[name],zero_forecast)
                change=base.native.arrays(self.file(path/'pair_changes.npz'))
                pair_indices=list(combinations(range(len(ctx['H'])),2))
                for key in ('J','J_without_common','J_common','J_cross','bias','meter',
                            'observed_logodds','observed_logodds_without_common'):
                    differences=[self.directed(ctx,values,i,j)[key]-self.directed(ctx,zero_values,i,j)[key]
                                 for i,j in pair_indices]
                    self.same('canonical_pair_change_'+key,change[key],differences,atol=1e-8)
                if 'pair' in evaluation:
                    i,j=evaluation['pair'];evaluation['forecast_evidence']=self.directed(ctx,forecasts[name],i,j)
                    zero_pair=self.directed(ctx,zero_values,i,j)
                    evaluation['evidence_increment']={k:evaluation['evidence'][k]-zero_pair[k] for k in
                        ('J','bias','meter','clean_logodds','observed_logodds')}
                actual_budget=self.budget(geometry,delivery)
                for item in actual_budget['rows']:
                    self.same('delivered_increment_budget',row['delivered_budget_values'][item['id']],item['value'],atol=1e-9,rtol=1e-9)
                delta_power=np.c_[obs['true_P']-zero_obs['true_P'],obs['true_Q']-zero_obs['true_Q']]
                delta_y=(obs['true_root'][:,None]**2-obs['true_V']**2)-(zero_obs['true_root'][:,None]**2-zero_obs['true_V']**2)
                K_prediction=np.einsum('tc,knc->ktn',delta_power,model['means'])
                H_prediction=np.einsum('tc,knc->ktn',delta_power,model['H'][model['representatives']])
                errorK=np.square(delta_y[None]-K_prediction).mean(axis=(1,2))/model['sigma2']
                errorH=np.square(delta_y[None]-H_prediction).mean(axis=(1,2))/model['sigma2']
                self.same('actual_K_increment_error',row['actual_increment_response_error_K_over_sigma2_by_representative'],errorK,atol=1e-9)
                self.same('actual_H_increment_error',row['actual_increment_response_error_H_over_sigma2_by_representative'],errorH,atol=1e-9)
                physical=row['physical_state'];ac=base.native.arrays(self.checked(physical))
                if 'time_slice' in physical:
                    start,end=physical['time_slice'];length=len(ac['voltage_complex_pu'])
                    ac={k:v[start:end] if v.ndim and v.shape[0]==length else v for k,v in ac.items()}
                electrical,branch_values,bus_values=base.operation.electrical(self.operation,ac,obs,metadata,branches,buses)
                voltage_ok=electrical['cells_below_095']+electrical['cells_above_105']==0
                rec.update(audit_status='passed',requested=actions[name]['requested'],delivered=delivered,
                    requested_violations=actions[name]['violations'],delivered_violations=violations,
                    planned_cost=bill['planned'],settled_cost=bill['settled'],evaluation=evaluation,
                    electrical=electrical,branches=branch_values,buses=bus_values,voltage_screen_passed=voltage_ok,
                    recovered_and_voltage_screen_passed=bool(evaluation['native_correct'] and voltage_ok),
                    thermal_rating_status='unavailable_not_verified',quote_new_gate_feasible=quote['new_gate_feasible'],
                    quote_zero_fallback=quote['zero_fallback'],base_mode=quote['mode'],guard_kind=quote['guard_kind'],
                    is_rotation=name.endswith('_rotation'),delivered_budget=actual_budget,
                    actual_H_increment_error_over_sigma2=errorH.tolist(),actual_K_increment_error_over_sigma2=errorK.tolist(),
                    planning_budget_is_not_execution_resource_constraint=True,
                    reused_action_source=row.get('reused_action_source'),baseline_source=row['baseline_source'])
            except Exception as exc:
                rec.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
            self.records.append(rec)


def planning_boundary(path):
    """Check this adapter's seal-before-physical-context and narrow solve call."""
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    if not {'run_case','execute','context_planning','build_geometry'} <= set(functions):
        return dict(passed=False,reason='Missing expected driver entry points')
    case_calls=[n for n in ast.walk(functions['run_case']) if isinstance(n,ast.Call)]
    seals=[n for n in case_calls if ast.unparse(n.func)=='io.write' and n.args and
        isinstance(n.args[0],ast.BinOp) and isinstance(n.args[0].right,ast.Constant) and n.args[0].right.value=='seal.json']
    executions=[n for n in case_calls if ast.unparse(n.func)=='execute']
    solves=[n for n in case_calls if ast.unparse(n.func)=='solver.plan']
    contexts=[n for n in ast.walk(functions['execute']) if isinstance(n,ast.Call) and ast.unparse(n.func)=='old.context']
    forbidden=any(isinstance(n,ast.Call) and ast.unparse(n.func)=='old.context'
        for name in ('run_case','context_planning','build_geometry') for n in ast.walk(functions[name]))
    expected=['case',"planning['score']","planning['val']",'forecast',"planning['base']"]
    passed=(len(seals)==len(executions)==len(solves)==len(contexts)==1 and not forbidden
        and executions[0].lineno>seals[0].lineno>solves[0].lineno
        and [ast.unparse(a) for a in solves[0].args]==expected)
    return dict(passed=passed,source=base.ref(path),seal_line=seals[0].lineno if len(seals)==1 else None,
        execute_line=executions[0].lineno if len(executions)==1 else None,
        solver_arguments=[ast.unparse(a) for a in solves[0].args] if len(solves)==1 else [],
        scope='Narrow AST review: original future AC context is opened only by post-seal execute; solver positional inputs are original contracts, historical score/validation, forecast and base. Not a general information-flow proof.')


def run(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if not (source/'complete.json').exists():raise ValueError('Completed run required')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'audit.py')
    audit=Audit();p=base.read(audit.file(source/'protocol.json'))
    entries=base.read(audit.checked(p['inputs']));complete=base.read(audit.file(source/'complete.json'))
    audit.require(complete['source_unchanged'],'Run source changed during execution')
    policy_names(p)
    for item in p['sources']:
        relative=Path(item['path']).resolve().relative_to(ROOT)
        audit.file(source/'source_snapshot'/relative,item['sha256'])
    driver=source/'source_snapshot/rnj_wzzt_dr/experiments/frozen_rx_guard_20261115/run.py'
    boundary=planning_boundary(driver);audit.require(boundary['passed'],'Planning information boundary changed')
    original=base.read(audit.checked(p['source_protocol']))
    audit.require(p['seed']==original['seed'] and p['forecasts']==1,'Old random stream was replaced')
    source_audit=base.read(audit.checked(p['source_audit']))
    audit.require(source_audit['status']=='passed','Baseline audit did not pass')
    original_entries={e['id']:e for e in base.read(audit.checked(original['inputs']))}
    for entry in entries:
        old=original_entries[entry['id']]
        for key in ('score','native_copy','files','physical_cache','p_resources','q_resources','index','base_kw','meter'):
            audit.require(entry[key]==old[key],'Frozen original entry changed: '+key)
    selected={e['id'] for e in entries}
    with audit.checked(source_audit['records']).open(encoding='utf8') as stream:
        for line in stream:
            row=__import__('json').loads(line)
            if row['id'] in selected and row['repeat']<p['repeats']:
                audit.baseline_records.append(row)
    baseline_expected=len(entries)*p['repeats']*len(panel.POLICIES)
    audit.require(len(audit.baseline_records)==baseline_expected,'Referenced baseline denominator differs')
    audit.require(all(r['audit_status']=='passed' for r in audit.baseline_records),'Referenced baseline includes unverified execution')
    expected_rows=len(entries)*len(p['modes'])*len(p['kinds'])*2*p['repeats']
    audit.require(p['expected_rows']==expected_rows,'Protocol new-execution denominator differs')
    success={r['id'] for r in complete.get('receipts',[])}
    failures={r['id'] for r in complete.get('failed_tasks',[])}
    audit.require(not(success&failures) and success|failures==selected,'Task completion ledger missing/duplicated identities')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),
        source_protocol=base.ref(source/'protocol.json'),planning_boundary=boundary,
        expected_new_execution_rows=expected_rows,referenced_baseline_rows=baseline_expected,
        scope='Saved arrays only: frozen models, historical budget geometry, quote gates, same random paths, native scoring, cost accounting and electrical balance. No fitting/optimization/AC solves.',
        environment=dict(python=sys.version,numpy=np.__version__),
        helpers=[base.ref(m.__file__) for m in (panel,base,base.native,base.operation,prob)])
    base.write(out/'protocol.json',own)
    for entry in entries:
        if entry['id'] in failures:
            for repeat in range(p['repeats']):
                for mode in p['modes']:
                    for kind in p['kinds']:
                        for suffix in ('','_rotation'):
                            audit.records.append(dict(**panel.identity(entry),forecast=0,repeat=repeat,
                                strategy=mode+'__'+kind+suffix,audit_status='recorded_failure',
                                original_status='task_failed',evaluation={'native_correct':False}))
            continue
        try:
            casepath=source/'source_snapshot/topology_identification_standalone/cases'/entry['case']
            metadata=base.read(audit.file(casepath/'metadata.json'))
            with audit.file(casepath/'branches.csv').open(encoding='utf-8-sig',newline='') as f:branches=list(csv.DictReader(f))
            with audit.file(casepath/'buses.csv').open(encoding='utf-8-sig',newline='') as f:buses=list(csv.DictReader(f))
            for file in ('metadata.json','branches.csv','buses.csv'):
                audit.require(base.ref(casepath/file)['sha256']==base.ref(ROOT/'rnj_wzzt_dr/physical/cases'/entry['case']/file)['sha256'],
                    'Physical case table changed after snapshot')
            audit.case(source/'cases'/entry['id'],entry,p,metadata,branches,buses)
        except Exception as exc:
            audit.task_errors.append(dict(id=entry['id'],error=repr(exc),traceback=traceback.format_exc()))
        base.write(out/'progress.json',dict(rows=len(audit.records),expected=expected_rows,
            task_errors=len(audit.task_errors),row_errors=sum(r['audit_status']=='failed' for r in audit.records)))
    for path,sha in audit.hashes.items():
        if base.ref(path)['sha256']!=sha:audit.task_errors.append(dict(path=path,error='Input changed during independent audit'))
    passed=(len(audit.records)==expected_rows and not audit.task_errors and
        all(r['audit_status'] in ('passed','recorded_failure') for r in audit.records))
    for name,rows in (('records',audit.records),('quotes',audit.quote_records),('baseline_records',audit.baseline_records)):
        with (out/(name+'.jsonl')).open('w',encoding='utf8') as stream:
            for row in rows:panel.json_line(stream,row)
    result=dict(status='passed' if passed else 'failed',protocol=base.ref(out/'protocol.json'),
        source_protocol=base.ref(source/'protocol.json'),source_inputs=p['inputs'],
        expected_rows=expected_rows,audited_rows=len(audit.records),baseline_rows=len(audit.baseline_records),
        audit_status_counts=dict(Counter(r['audit_status'] for r in audit.records)),
        input_coverage=audit.input_records,geometry_checks=audit.geometry_records,task_errors=audit.task_errors,
        original_failed_tasks=complete.get('failed_tasks',[]),original_quote_exceptions=complete.get('quote_exceptions',0),
        original_execution_failed_rows=complete.get('failed_rows',0),
        records=base.ref(out/'records.jsonl'),quotes=base.ref(out/'quotes.jsonl'),baseline_records=base.ref(out/'baseline_records.jsonl'),
        rotation_unmatched_fees=audit.fee_mismatches,maximum_numeric_errors=dict(audit.maxima),
        maximum_electrical_errors=dict(audit.operation.maxima),counts=dict(audit.counts),
        evidence_scope='Conditional Gaussian gate and historical incremental design restriction are checked separately from actual AC recovery. Failed quotes and zero fallbacks remain in all executed-policy denominators.',
        thermal_scope='Branch current and root apparent power are reconstructed; equipment thermal ratings are unavailable.')
    base.write(out/'report.json',result);base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json')))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.run,args.out)
    print({k:result[k] for k in ('status','expected_rows','audited_rows','baseline_rows','task_errors')})
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
