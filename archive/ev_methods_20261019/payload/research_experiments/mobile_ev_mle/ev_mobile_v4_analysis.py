"""Read-only analysis of saved v4 plans and paired future observations.

No AC evaluation, parameter fit, action optimization, or policy selection runs
here.  Truth is used solely to describe already saved action rankings.  A
history is the independent unit; different future regimes are paired within
that history.  Missing records retain all declared method/candidate slots.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import statistics


DOMAINS = ('fixed_charge', 'fixed_bidir', 'mobile_charge', 'mobile_bidir')
PARENTS = {'fixed_charge': (), 'fixed_bidir': ('fixed_charge',),
           'mobile_charge': ('fixed_charge',), 'mobile_bidir': ('fixed_bidir', 'mobile_charge')}
FAMILIES = ('square', 'aligned', 'guarded')
METHODS = tuple(f+'__'+d for f in FAMILIES for d in DOMAINS)
PHASES = ('development', 'evaluation')
TOL = 1e-7


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def stats(values):
    values = [v for v in values if number(v) is not None]
    if not values: return dict(n=0, mean=None, median=None, min=None, max=None)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values), min=min(values), max=max(values))


def difference(a, b):
    return None if number(a) is None or number(b) is None else float(a-b)


def sign(value, tolerance=TOL):
    if number(value) is None: return None
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def minimum_set(values, tolerance=TOL):
    if not values or any(number(v) is None for v in values.values()): return None
    best = min(values.values())
    return sorted(k for k, value in values.items() if value <= best+tolerance*(1.+abs(best)))


def _training_rows(record):
    return {r.get('candidate', r.get('name')): r for r in record.get('training_result', {}).get('fits', [])
            if isinstance(r, dict) and isinstance(r.get('candidate', r.get('name')), str)}


def _nearest(record, expected=4):
    rows = record.get('nearest', [])
    if not isinstance(rows, list) or len(rows) != expected: return None
    result = {}
    for row in rows:
        label = row.get('a'); value = number(row.get('half_gap'))
        if not isinstance(label, str) or label in result or value is None: return None
        result[label] = value
    return result


def project_method(record, regime, name, *, candidate_count=4):
    """Project one method slot, keeping planning, physical, and fit failures apart."""
    family, domain = name.split('__')
    raw = record.get('methods', {}).get(name)
    present = isinstance(raw, dict); raw = raw if present else {}
    log = record.get('families', {}).get(family, {}).get('logs', {}).get(domain, {}) or {}
    key = raw.get('action_key')
    action = record.get('actual', {}).get(regime, {}).get('actions', {}).get(key, {}) if isinstance(key, str) else {}
    evaluation = action.get('evaluation', {}) or {}
    comparisons = evaluation.get('comparisons', {}) or {}
    losses = evaluation.get('losses', {}) or {}
    true = record.get('true_name'); training = _training_rows(record)
    expected_labels = set(training) if len(training) == candidate_count else None
    issues = []
    valid_losses = (isinstance(losses, dict) and len(losses) == candidate_count and true in losses
        and all(number(value) is not None for value in losses.values())
        and (expected_labels is None or set(losses) == expected_labels))
    valid_fits = (isinstance(comparisons, dict) and len(comparisons) == candidate_count
        and (expected_labels is None or set(comparisons) == expected_labels)
        and all(row.get('joint', {}).get('success') is True
                and number(row.get('joint', {}).get('loss')) is not None for row in comparisons.values()))
    if valid_losses and valid_fits and set(losses) != set(comparisons):
        valid_fits = False
    complete = evaluation.get('complete') is True and valid_losses and valid_fits
    if evaluation.get('complete') is True and not complete: issues.append('declared_complete_without_all_valid_candidate_fits')
    if valid_losses and valid_fits and any(abs(losses[k]-comparisons[k]['joint']['loss']) > TOL*(1.+abs(losses[k])) for k in losses):
        complete = False; issues.append('loss_map_disagrees_with_saved_candidate_joint_loss')
    feasible = raw.get('operation', {}).get('feasible')
    usable = complete and feasible is True
    M = min(v for k, v in losses.items() if k != true)-losses[true] if complete else None
    if complete and (number(evaluation.get('M')) is None or abs(evaluation['M']-M) > TOL*(1.+abs(M))):
        issues.append('saved_M_missing_or_inconsistent')
    nominal = raw.get('nominal', {}) or {}
    nearest = _nearest(nominal, candidate_count)
    nominal_success = nominal.get('success') is True and number(nominal.get('score')) is not None
    if nominal_success and nearest is None: issues.append('nominal_missing_complete_reference_nearest_values')
    phi = number(nominal.get('score')) if nominal_success else None
    true_ref = None if nearest is None else nearest.get(true)
    raw_argmin = minimum_set(losses) if usable else None
    dimensions = {k: 2*len(row['r']) for k, row in training.items()
                  if isinstance(row.get('r'), list) and row.get('r') and isinstance(row.get('x'), list)
                  and len(row['x']) == len(row['r'])}
    aic_scores = {k: losses[k]+2*dimensions[k] for k in losses} if usable and set(dimensions) == set(losses) else None
    aic_argmin = minimum_set(aic_scores) if aic_scores else None
    candidate_rows = list(comparisons.values()) if isinstance(comparisons, dict) else []
    fitted = sum(r.get('joint', {}).get('success') is True and number(r.get('joint', {}).get('loss')) is not None for r in candidate_rows)
    failures = evaluation.get('failures', {}) or {}
    seen = set(comparisons) | set(failures) if isinstance(comparisons, dict) and isinstance(failures, dict) else set(comparisons)
    if expected_labels is not None:
        if seen-expected_labels: issues.append('unexpected_candidate_label_in_evaluation')
        seen &= expected_labels
    failure = raw.get('planning_failure') or log.get('planning_failure')
    return dict(name=name, family=family, domain=domain, present=present, action_key=key,
        planning_plan_available=raw.get('scored_plan_available', raw.get('planning_success')),
        planning_success=raw.get('planning_success'), planning_failure=failure,
        planning_step_failures=[s.get('failure') for s in log.get('steps', []) if s.get('failure')],
        physical_status='success' if feasible is True else 'failed' if feasible is False else 'missing',
        fit_status='success' if complete else 'failed' if evaluation else 'missing',
        candidate_slots=dict(expected=candidate_count, present=len(seen), successful=fitted,
            failed=len(seen)-fitted, missing=max(0, candidate_count-len(seen))),
        candidate_fixed_source_failures=sum(row.get('fixed_source_success') is False for row in candidate_rows),
        evaluation_failures=failures, usable=bool(usable),
        M=float(M) if usable else None, diagnostic_M=float(M) if M is not None else None,
        actual_halfM=.5*M if usable else None,
        nominal_Phi=phi, nominal_true_ref_m=true_ref, nominal_nearest=nearest,
        nominal_status='success' if nominal_success else 'failed' if nominal else 'missing',
        raw_argmin=raw_argmin, raw_unique_correct=None if raw_argmin is None else raw_argmin == [true],
        raw_contains_truth=None if raw_argmin is None else true in raw_argmin,
        AIC_argmin=aic_argmin, AIC_unique_correct=None if aic_argmin is None else aic_argmin == [true],
        AIC_contains_truth=None if aic_argmin is None else true in aic_argmin,
        M_AIC=None if aic_scores is None else min(v for k, v in aic_scores.items() if k != true)-aic_scores[true],
        evaluation_seconds=number(evaluation.get('seconds')), issues=issues)


def parent_comparisons(methods, *, tolerance=TOL):
    """Compare every child to every direct parent; label the highest-Phi parent.

    Parent selection here describes saved deterministic plans and never
    changes their actions.  In MD, the other parent's per-reference floor is
    not a promised constraint, so it cannot count as a guard violation.
    """
    result = []
    for family in FAMILIES:
        for child, parents in PARENTS.items():
            if not parents: continue
            C = methods[family+'__'+child]
            parents_complete = all(number(methods[family+'__'+p]['nominal_Phi']) is not None for p in parents)
            chosen = None; ties = []
            if parents_complete:
                best = max(methods[family+'__'+p]['nominal_Phi'] for p in parents)
                ties = [p for p in parents if abs(methods[family+'__'+p]['nominal_Phi']-best) <= tolerance]
                # Match max(list, key=...) order: actual highest value first;
                # the declared parent order only resolves exact numeric ties.
                chosen = max(parents, key=lambda p: methods[family+'__'+p]['nominal_Phi'])
            for parent in parents:
                P = methods[family+'__'+parent]
                dp = difference(C['nominal_Phi'], P['nominal_Phi'])
                dr = difference(C['nominal_true_ref_m'], P['nominal_true_ref_m'])
                da = difference(C['actual_halfM'], P['actual_halfM'])
                before, after = P['nominal_nearest'], C['nominal_nearest']
                ref_slacks = ({a: after[a]-before[a] for a in before}
                              if before is not None and after is not None and set(before) == set(after) else None)
                violations = ([a for a, v in ref_slacks.items() if v < -tolerance*(1.+abs(before[a]))]
                              if ref_slacks is not None else None)
                metric_error = difference(dr, dp); mean_error = difference(da, dr)
                all_numeric = all(number(v) is not None for v in (dp, dr, da))
                selected_parent = parent == chosen if chosen is not None else None
                result.append(dict(family=family, child=child, parent=parent,
                    selected_highest_Phi_parent=chosen, parent_selection_complete=parents_complete,
                    highest_Phi_parent_ties=ties, is_selected_parent=selected_parent,
                    role='selected_highest_Phi_parent' if selected_parent else 'other_parent' if selected_parent is False else 'parent_selection_missing',
                    delta_nominal_Phi=dp, delta_nominal_true_ref_m=dr, delta_actual_halfM=da,
                    metric_aggregation_discrepancy=metric_error, prediction_actual_discrepancy=mean_error,
                    decomposition_error=None if not all_numeric else abs(da-(dp+metric_error+mean_error)),
                    average_metric_sacrifices_true_reference=sign(dp, tolerance) == 1 and sign(dr, tolerance) == -1,
                    predicted_true_reference_gain_actual_loss=sign(dr, tolerance) == 1 and sign(da, tolerance) == -1,
                    predicted_mean_gain_actual_loss=sign(dp, tolerance) == 1 and sign(da, tolerance) == -1,
                    nominal_per_reference_slacks=ref_slacks, reference_floor_violations=violations,
                    guard_violation=(bool(violations) if family == 'guarded' and selected_parent is True and violations is not None else None),
                    Phi_monotonicity_required=family != 'square',
                    Phi_monotonicity_violation=(sign(dp, tolerance) == -1 if family != 'square' and dp is not None else None)))
    return result


def rank_pairs(methods, *, tolerance=TOL):
    """Describe pair order on unique saved actions; duplicate methods add no pairs."""
    actions = {}; missing_keys = []
    for name, method in methods.items():
        key = method.get('action_key')
        if not isinstance(key, str) or not key:
            missing_keys.append(name); continue
        actions.setdefault(key, []).append(method)
    projected = []; issues = []
    for key, rows in actions.items():
        value = dict(action_key=key, methods=[r['name'] for r in rows])
        for metric in ('nominal_Phi', 'nominal_true_ref_m', 'actual_halfM'):
            finite = [r[metric] for r in rows if number(r[metric]) is not None]
            conflict = finite and max(finite)-min(finite) > tolerance*(1.+max(abs(v) for v in finite))
            value[metric] = None if conflict or not finite else finite[0]
            if conflict: issues.append(dict(action_key=key, metric=metric, issue='duplicate_action_conflicting_saved_values'))
        projected.append(value)
    rows = []
    for A, B in combinations(projected, 2):
        row = dict(action_a=A['action_key'], action_b=B['action_key'], methods_a=A['methods'], methods_b=B['methods'])
        row['delta_actual_halfM'] = difference(A['actual_halfM'], B['actual_halfM'])
        for metric in ('nominal_Phi', 'nominal_true_ref_m'):
            delta = difference(A[metric], B[metric]); actual = row['delta_actual_halfM']
            s, t = sign(delta, tolerance), sign(actual, tolerance)
            row[metric] = dict(delta=delta, complete=s is not None and t is not None,
                strict=s not in (None, 0) and t not in (None, 0),
                concordant=s not in (None, 0) and t not in (None, 0) and s == t,
                reversed=s not in (None, 0) and t not in (None, 0) and s != t,
                predicted_tie=s == 0 and t is not None, actual_tie=t == 0 and s is not None)
        rows.append(row)
    return dict(method_slots=len(methods), present_method_slots=sum(m['present'] for m in methods.values()),
        unique_actions=len(actions), duplicate_method_slots=sum(len(v)-1 for v in actions.values()),
        missing_action_keys=missing_keys, pair_denominator=len(rows), rows=rows, issues=issues,
        scope='descriptive action pairs within one history/regime; pairs are not independent histories')


def parameter_audit(record, competitor, z, tolerance=TOL):
    """Check saved delta explanations using original bounds, without fitting."""
    fit = _training_rows(record).get(competitor)
    cfg = record.get('config', {}); forecast = record.get('forecast', {}).get('source_by_candidate', {}).get(competitor)
    if not isinstance(fit, dict) or not isinstance(forecast, dict) or not isinstance(z, list):
        return dict(known=False, feasible=None, reason='missing saved domain inputs')
    r, x = fit.get('r'), fit.get('x'); hp, hq = fit.get('source_p'), fit.get('source_q')
    fp, fq = forecast.get('source_p'), forecast.get('source_q')
    source = cfg.get('source', {}); training = cfg.get('training', {})
    arrays = (r, x, hp, hq, fp, fq)
    if any(not isinstance(v, list) for v in arrays) or len(r) != len(x) or len(hp) != len(hq) or len(fp) != len(fq):
        return dict(known=False, feasible=None, reason='invalid saved domain vector shapes')
    m = len(r); center_source = [v for pair in zip(hp+fp, hq+fq) for v in pair]
    if len(z) != 2*m+len(center_source) or any(number(v) is None for v in list(z)+list(r)+list(x)+center_source):
        return dict(known=False, feasible=None, reason='parameter coordinate dimension or finite-value mismatch')
    names = ('R', 'X', 'source_P', 'source_Q')
    vectors = ( [v+w for v, w in zip(r, z[:m])], [v+w for v, w in zip(x, z[m:2*m])],
                [v+w for v, w in zip(center_source[::2], z[2*m::2])],
                [v+w for v, w in zip(center_source[1::2], z[2*m+1::2])] )
    bounds = (training.get('r_bounds'), training.get('x_bounds'), source.get('source_p_bounds'), source.get('source_q_bounds'))
    if any(not isinstance(v, (list, tuple)) or len(v) != 2 or any(number(t) is None for t in v) for v in bounds):
        return dict(known=False, feasible=None, reason='missing original physical bounds')
    violations = {name: max([0.]+[lo-v for v in values]+[v-hi for v in values])
                  for name, values, (lo, hi) in zip(names, vectors, bounds)}
    ratio = cfg.get('ratio_bounds')
    if not isinstance(ratio, (list, tuple)) or len(ratio) != 2:
        return dict(known=False, feasible=None, reason='missing original X/R bounds', box_violations=violations)
    violations['ratio_linear'] = max([0.]+[ratio[0]*R-X for R, X in zip(vectors[0], vectors[1])]
                                    +[X-ratio[1]*R for R, X in zip(vectors[0], vectors[1])])
    return dict(known=True, feasible=max(violations.values()) <= tolerance, violations=violations,
                line_failure=max(violations[k] for k in ('R', 'X', 'ratio_linear')) > tolerance,
                source_failure=max(violations[k] for k in ('source_P', 'source_Q')) > tolerance,
                tolerance=tolerance)


def _qp_record(record, value, path, competitor):
    return dict(path=path, a=value.get('a'), b=competitor,
        primal_upper=number(value.get('primal_upper')), dual_lower=number(value.get('dual_lower')),
        inner_gap=number(value.get('inner_gap')), completed=value.get('completed'),
        solver=value.get('solver'), failures=value.get('failures', []),
        saved_domain_audit=value.get('explanation_audit'),
        original_parameter_domain=parameter_audit(record, competitor, value.get('z')))


def _floor_audit(floor):
    if not isinstance(floor, dict): return dict(known=False, passes=None)
    before, slacks = floor.get('floor'), floor.get('slacks')
    tolerance = number(floor.get('tolerance'))
    labels = floor.get('expected_references')
    if (not isinstance(before, dict) or not isinstance(slacks, dict) or tolerance is None
            or not isinstance(labels, list) or len(labels) != 4 or len(set(labels)) != 4
            or set(before) != set(labels) or set(slacks) != set(labels)
            or any(number(v) is None for v in list(before.values())+list(slacks.values()))):
        return dict(known=False, passes=None, reason='incomplete per-reference floor evidence')
    violated = [a for a in labels if slacks[a] < -tolerance*(1.+abs(before[a]))]
    passes = floor.get('complete') is True and not violated
    return dict(known=True, passes=passes, recomputed_violated=violated,
                saved_success=floor.get('success'), agrees_with_saved_success=floor.get('success') is passes)


def planning_logs(record):
    """Flatten solver evidence once per historical plan, never per future regime."""
    rows = []; all_qp = []; all_masters = []; accepts = []; rejected = []; family_costs = []
    for family in FAMILIES:
        family_raw = record.get('families', {}).get(family, {}) or {}
        family_costs.append(dict(family=family, present=bool(family_raw),
            actual_seconds=number(family_raw.get('actual_seconds', family_raw.get('seconds'))),
            square_dependency_seconds=number(family_raw.get('square_dependency_seconds')),
            standalone_seconds=number(family_raw.get('standalone_planning_seconds', family_raw.get('seconds'))),
            budget_seconds=number(family_raw.get('budget_seconds')), engine_counters=family_raw.get('engine_counters')))
        for domain in DOMAINS:
            raw = family_raw.get('logs', {}).get(domain, {}) or {}
            row = dict(family=family, domain=domain, present=bool(raw),
                planning_failure=raw.get('planning_failure'),
                selected_score=number(raw.get('selected_score')), Q=number(raw.get('Q')),
                parent_Q=raw.get('parent_Q', []),
                saved_monotonicity_verified=raw.get('scalar_monotonicity_verified'),
                selected_origin=raw.get('selected_origin'), seconds=number(raw.get('seconds')),
                actual_family_seconds=number(family_raw.get('actual_seconds', family_raw.get('seconds'))),
                standalone_family_seconds=number(family_raw.get('standalone_planning_seconds')),
                initial_attempts=len(raw.get('initials', [])), steps=[], masters=[], QPs=[])
            for i, initial in enumerate(raw.get('initials', [])):
                event = dict(family=family, domain=domain, kind='initializer', index=i,
                    accepted=initial.get('accepted'), score=number(initial.get('score')),
                    action_key=initial.get('action_key'),
                    reference_floor=initial.get('reference_floor'),
                    floor_audit=_floor_audit(initial.get('reference_floor')))
                base = (event['reference_floor'] or {}).get('floor')
                event['mean_improves'] = (event['score'] > statistics.mean(base.values())+1e-9
                    if event['floor_audit']['known'] and event['score'] is not None else None)
                if initial.get('accepted'): accepts.append(event)
                elif family == 'guarded' and event['mean_improves'] is True and event['floor_audit']['passes'] is False:
                    rejected.append(event)
            for si, step in enumerate(raw.get('steps', [])):
                event = dict(family=family, domain=domain, kind='MILP_step', index=si,
                    accepted=step.get('accepted'), previous_AC_score=number(step.get('previous_AC_score')),
                    AC_score=number(step.get('AC_score')), mean_improves=step.get('mean_improves'),
                    rejection_reason=step.get('rejection_reason'), failure=step.get('failure'),
                    reference_floor=step.get('reference_floor'), proposed_action=step.get('proposed_action'),
                    floor_audit=_floor_audit(step.get('reference_floor')))
                row['steps'].append(event)
                if step.get('accepted'): accepts.append(event)
                if family == 'guarded' and step.get('rejection_reason') == 'reference_sacrifice': rejected.append(event)
                for pi, alignment in enumerate(step.get('alignment', {}).get('pairs', [])):
                    qp = _qp_record(record, alignment.get('parent_QP', {}), f'alignment[{si}].pairs[{pi}]', alignment.get('b'))
                    row['QPs'].append(qp)
                master = step.get('master', {}) or {}
                if master:
                    mrow = dict(path=f'steps[{si}].master', objective_lower=number(master.get('objective_lower')),
                        objective_upper=number(master.get('objective_upper')), gap=number(master.get('gap')),
                        completed=master.get('completed'), failures=master.get('failures', []), iterations=[])
                    for ii, iteration in enumerate(master.get('iterations', [])):
                        lower, upper = number(iteration.get('feasible_lower')), number(iteration.get('upper'))
                        comp = iteration.get('shared_control_square', {}) or {}
                        mrow['iterations'].append(dict(iteration=ii, solver_status=iteration.get('solver_status'),
                            lower=lower, upper=upper, gap=difference(upper, lower),
                            mip_best_bound=number(iteration.get('mip_best_bound')),
                            mip_incumbent_upper_model=number(iteration.get('mip_incumbent_upper_model')),
                            operation_audit=iteration.get('operation_incumbent_audit'),
                            master_audit=iteration.get('master_incumbent_audit'),
                            n_segment_binary=iteration.get('n_segment_binary'),
                            generic_segment_binary_estimate=iteration.get('generic_segment_binary_estimate'),
                            segment_binary_saving=iteration.get('segment_binary_saving'),
                            compression_used=comp.get('used'), compression_failure=comp.get('reason'),
                            zero_product_certificate=comp.get('zero_product_certificate'),
                            seconds=number(iteration.get('seconds'))))
                    row['masters'].append(mrow)
                    for ei, evaluation in enumerate(master.get('evaluations', [])):
                        for qi, solution in enumerate(evaluation.get('solutions', [])):
                            row['QPs'].append(_qp_record(record, solution,
                                f'steps[{si}].master.evaluations[{ei}].solutions[{qi}]', solution.get('b')))
            all_qp.extend(dict(family=family, domain=domain, **qp) for qp in row['QPs'])
            all_masters.extend(dict(family=family, domain=domain, **mrow) for mrow in row['masters'])
            rows.append(row)
    accepted_guarded = [a for a in accepts if a['family'] == 'guarded' and a.get('reference_floor') is not None]
    return dict(rows=rows, QPs=all_qp, masters=all_masters, family_costs=family_costs, accepted_events=accepts,
        guard_rejected_events=rejected,
        guarded_accepted_floor_checked=len(accepted_guarded),
        guarded_accepted_floor_violations=sum(a['floor_audit'].get('passes') is False for a in accepted_guarded),
        guarded_accepted_floor_unknown=sum(a['floor_audit'].get('known') is False for a in accepted_guarded),
        accepted_MILP_mean_violations=sum(a['kind'] == 'MILP_step' and a.get('mean_improves') is not True for a in accepts),
        counts_once_per_history=True)


def paired(method_rows, a, c):
    deltas = [difference(row['methods'][a]['M'], row['methods'][c]['M']) for row in method_rows]
    return dict(a=a, b=c, denominator=len(method_rows), complete=sum(v is not None for v in deltas),
                missing=sum(v is None for v in deltas), wins=sum(sign(v) == 1 for v in deltas),
                ties=sum(sign(v) == 0 for v in deltas), losses=sum(sign(v) == -1 for v in deltas),
                delta_M=stats(deltas), delta_halfM=stats(None if v is None else .5*v for v in deltas))


def aggregate_regimes(rows):
    result = dict(history_regime_denominator=len(rows),
        independent_history_count=len({(r['spec'].get('phase'), r['spec']['index']) for r in rows}),
        method_slot_denominator=len(rows)*len(METHODS),
        candidate_slot_denominator=len(rows)*len(METHODS)*4, methods={}, pairs={}, parent_comparisons={})
    for name in METHODS:
        values = [r['methods'][name] for r in rows]
        result['methods'][name] = dict(denominator=len(values), present=sum(v['present'] for v in values),
            usable=sum(v['usable'] for v in values), M=stats(v['M'] for v in values),
            M_AIC=stats(v['M_AIC'] for v in values), actual_halfM=stats(v['actual_halfM'] for v in values),
            planning_plan_available=dict(Counter(str(v['planning_plan_available']) for v in values)),
            planning_success=dict(Counter(str(v['planning_success']) for v in values)),
            planning_failure=sum(bool(v['planning_failure']) for v in values),
            step_failure=sum(bool(v['planning_step_failures']) for v in values),
            physical_status=dict(Counter(v['physical_status'] for v in values)),
            fit_status=dict(Counter(v['fit_status'] for v in values)),
            nominal_status=dict(Counter(v['nominal_status'] for v in values)),
            candidate_slots={k: sum(v['candidate_slots'][k] for v in values) for k in ('expected', 'present', 'successful', 'failed', 'missing')},
            actual_evaluation_exception_counts=dict(Counter(str(message) for v in values for message in v['evaluation_failures'].values())),
            candidate_fixed_source_failures=sum(v['candidate_fixed_source_failures'] for v in values),
            decisions={tag:dict(complete_subset=sum(v[tag+'_argmin'] is not None for v in values),
                unique_correct=sum(v[tag+'_unique_correct'] is True for v in values),
                contains_truth=sum(v[tag+'_contains_truth'] is True for v in values),
                ties=sum(v[tag+'_argmin'] is not None and len(v[tag+'_argmin']) > 1 for v in values))
                for tag in ('raw', 'AIC')},
            issue_count=sum(len(v['issues']) for v in values))
    comparisons = []
    for domain in DOMAINS:
        comparisons += [(family+'__'+domain, 'square__'+domain) for family in ('aligned', 'guarded')]
        comparisons += [('guarded__'+domain, 'aligned__'+domain)]
    for family in FAMILIES:
        comparisons += [(family+'__'+child, family+'__'+parent) for child, parents in PARENTS.items() for parent in parents]
    for a, c in comparisons: result['pairs'][a+'__vs__'+c] = paired(rows, a, c)
    for family in FAMILIES:
        selected = [p for row in rows for p in row['parent_comparisons']
                    if p['family'] == family and p['is_selected_parent'] is True]
        result['parent_comparisons'][family] = dict(expected_children=3*len(rows),
            selected_parent_rows=len(selected), missing_parent_selection=3*len(rows)-len(selected),
            delta_nominal_Phi=stats(p['delta_nominal_Phi'] for p in selected),
            delta_true_ref_m=stats(p['delta_nominal_true_ref_m'] for p in selected),
            delta_actual_halfM=stats(p['delta_actual_halfM'] for p in selected),
            metric_sacrifice_comparable=sum(p['delta_nominal_Phi'] is not None and p['delta_nominal_true_ref_m'] is not None for p in selected),
            true_ref_prediction_actual_comparable=sum(p['delta_nominal_true_ref_m'] is not None and p['delta_actual_halfM'] is not None for p in selected),
            mean_prediction_actual_comparable=sum(p['delta_nominal_Phi'] is not None and p['delta_actual_halfM'] is not None for p in selected),
            average_metric_sacrifices_true_reference=sum(p['average_metric_sacrifices_true_reference'] for p in selected),
            predicted_true_reference_gain_actual_loss=sum(p['predicted_true_reference_gain_actual_loss'] for p in selected),
            predicted_mean_gain_actual_loss=sum(p['predicted_mean_gain_actual_loss'] for p in selected),
            guard_violation=sum(p['guard_violation'] is True for p in selected),
            other_MD_parent_rows=sum(p['family'] == family and p['role'] == 'other_parent'
                for row in rows for p in row['parent_comparisons']))
    ranking = {}
    for metric in ('nominal_Phi', 'nominal_true_ref_m'):
        pairs = [p[metric] for row in rows for p in row['ranking']['rows']]
        strict = sum(p['strict'] for p in pairs); same = sum(p['concordant'] for p in pairs)
        ranking[metric] = dict(pair_denominator=len(pairs), complete=sum(p['complete'] for p in pairs),
            missing=sum(not p['complete'] for p in pairs), strict_pairs=strict, concordant=same,
            reversed=sum(p['reversed'] for p in pairs), concordance=None if not strict else same/strict,
            predicted_ties=sum(p['predicted_tie'] for p in pairs), actual_ties=sum(p['actual_tie'] for p in pairs))
    result['ranking'] = ranking
    return result


def aggregate_planning(histories):
    logs = [h['planning'] for h in histories]
    qps = [q for log in logs for q in log['QPs']]
    masters = [m for log in logs for m in log['masters']]
    iterations = [i for m in masters for i in m['iterations']]
    parameter_known = [q for q in qps if q['original_parameter_domain'].get('known')]
    family_costs = {}
    for family in FAMILIES:
        values = [c for log in logs for c in log['family_costs'] if c['family'] == family]
        family_costs[family] = dict(denominator=len(histories), present=sum(c['present'] for c in values),
            actual_seconds=stats(c['actual_seconds'] for c in values),
            square_dependency_seconds=stats(c['square_dependency_seconds'] for c in values),
            standalone_seconds=stats(c['standalone_seconds'] for c in values),
            soft_budget_exceeded=sum(c['standalone_seconds'] is not None and c['budget_seconds'] is not None
                and c['standalone_seconds'] > c['budget_seconds'] for c in values))
    return dict(independent_histories=len(histories), expected_plan_slots=len(histories)*12,
        family_costs=family_costs,
        present_plan_logs=sum(r['present'] for log in logs for r in log['rows']),
        top_planning_failures=[dict(spec=h['spec'], family=r['family'], domain=r['domain'], failure=r['planning_failure'])
            for h in histories for r in h['planning']['rows'] if r['planning_failure']],
        accepted_MILP_steps=sum(e['kind'] == 'MILP_step' for log in logs for e in log['accepted_events']),
        guard_rejected_events=sum(len(log['guard_rejected_events']) for log in logs),
        guarded_accepted_floor_checked=sum(log['guarded_accepted_floor_checked'] for log in logs),
        guarded_accepted_floor_violations=sum(log['guarded_accepted_floor_violations'] for log in logs),
        guarded_accepted_floor_unknown=sum(log['guarded_accepted_floor_unknown'] for log in logs),
        accepted_MILP_mean_violations=sum(log['accepted_MILP_mean_violations'] for log in logs),
        QP_attempts=len(qps), QP_completed=sum(q['completed'] is True for q in qps),
        QP_with_failure_records=sum(bool(q['failures']) for q in qps),
        QP_gap=stats(q['inner_gap'] for q in qps),
        QP_saved_domain_failures=sum((q.get('saved_domain_audit') or {}).get('feasible') is False for q in qps),
        QP_original_domain_known=len(parameter_known), QP_original_domain_unknown=len(qps)-len(parameter_known),
        QP_original_line_failures=sum(q['original_parameter_domain'].get('line_failure') is True for q in parameter_known),
        QP_original_source_failures=sum(q['original_parameter_domain'].get('source_failure') is True for q in parameter_known),
        master_attempts=len(masters), master_completed=sum(m['completed'] is True for m in masters),
        master_gap=stats(m['gap'] for m in masters), master_iterations=len(iterations),
        compression_used=sum(i['compression_used'] is True for i in iterations),
        segment_binary=stats(i['n_segment_binary'] for i in iterations),
        segment_binary_saving=stats(i['segment_binary_saving'] for i in iterations),
        bound_scope='frozen corrected bounded-residual proxy only; not complete-AC bounds')


def summarize_records(specs, records, *, candidate_count=4):
    if candidate_count != 4: raise ValueError('v4 declares four retained candidates')
    if len({s['index'] for s in specs}) != len(specs) or len({s['seed'] for s in specs}) != len(specs):
        raise ValueError('unique independent history index and seed required within phase')
    expected = {s['index']: s for s in specs}
    if set(records)-set(expected): raise ValueError('extra record outside frozen phase')
    histories = []; regime_rows = []
    for index, spec in sorted(expected.items()):
        raw = records.get(index)
        if raw is not None and raw.get('spec') != spec: raise ValueError('record identity differs from frozen history specification')
        record = raw or {}
        history = dict(spec=spec, status='missing_record' if raw is None else raw.get('status', 'unknown'),
            true_name=record.get('true_name'), failure=record.get('failure'),
            seconds=number(record.get('seconds')), physics_calls=record.get('physics_calls'),
            planning=planning_logs(record), regimes=[])
        for regime in spec['regimes']:
            methods = {name: project_method(record, regime, name, candidate_count=candidate_count) for name in METHODS}
            row = dict(spec=spec, regime=regime, record_status=history['status'], true_name=record.get('true_name'), methods=methods,
                parent_comparisons=parent_comparisons(methods), ranking=rank_pairs(methods))
            regime_rows.append(row); history['regimes'].append(regime)
        histories.append(history)
    by_group = {}
    for bank, regime in sorted({(row['spec']['bank'], row['regime']) for row in regime_rows}):
        by_group[bank+'/'+regime] = aggregate_regimes([r for r in regime_rows if r['spec']['bank'] == bank and r['regime'] == regime])
    monotonicity = []
    # Nominal score/Q audits occur once per historical plan despite two futures.
    for history in histories:
        row = next((r for r in regime_rows if r['spec']['index'] == history['spec']['index']), None)
        if row is None: continue
        raw = records.get(history['spec']['index'], {})
        for comp in row['parent_comparisons']:
            if comp['family'] == 'square':
                qlogs = raw.get('families', {}).get('square', {}).get('logs', {})
                delta = difference(qlogs.get(comp['child'], {}).get('Q'), qlogs.get(comp['parent'], {}).get('Q'))
                metric = 'precision_Q'
            else:
                delta = comp['delta_nominal_Phi']; metric = 'common_complete_AC_Phi'
            monotonicity.append(dict(spec=history['spec'], family=comp['family'], child=comp['child'], parent=comp['parent'],
                objective=metric, delta=delta, violation=sign(delta) == -1 if delta is not None else None))
    return dict(history_denominator=len(specs), records_present=len(records),
        record_status=dict(Counter(h['status'] for h in histories)),
        overall=aggregate_regimes(regime_rows), by_bank_regime=by_group,
        planning=aggregate_planning(histories),
        monotonicity=dict(expected_edges=12*len(specs), known=sum(r['delta'] is not None for r in monotonicity),
            violations=sum(r['violation'] is True for r in monotonicity), rows=monotonicity),
        histories=histories, rows=regime_rows)


def _read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data), dict(
        path=str(Path(path).resolve()), bytes=len(data), sha256=sha256(data).hexdigest())


def render(data, path):
    lines = ['# 四域嵌套规划与动作排序审计', '',
        '独立统计单位是一条已冻结历史；同历史的 regular/shift 未来为配对观测。规划只计一次，缺失历史保留12方法及每方法4候选槽。Phi 是共同完整AC预测的平均最近半SSE差；平方基线的主规划目标是 Q，不要求其 Phi 随域单调。', '']
    for phase, value in data['phases'].items():
        lines += [f'## {phase}', '', f"历史：{value['records_present']}/{value['history_denominator']} 个文件；域边单调已知 {value['monotonicity']['known']}/{value['monotonicity']['expected_edges']}，违反 {value['monotonicity']['violations']}。", '',
            '|场景|方法|历史/有效|M均值|M中位|规划问题|物理失败/缺失|拟合失败/缺失|raw正确/完整集|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        fmt = lambda x: '—' if x is None else f'{x:.3f}'
        for group, g in value['by_bank_regime'].items():
            for name, method in g['methods'].items():
                physical = method['physical_status']; fitting = method['fit_status']; decision = method['decisions']['raw']
                lines.append(f"|{group}|{name}|{method['denominator']}/{method['usable']}|{fmt(method['M']['mean'])}|{fmt(method['M']['median'])}|{method['planning_failure']}|{physical.get('failed',0)}/{physical.get('missing',0)}|{fitting.get('failed',0)}/{fitting.get('missing',0)}|{decision['unique_correct']}/{decision['complete_subset']}|")
        lines += ['', '|场景|家族最高Phi父比较|选中父/预期|平均指标牺牲真ref|真ref预测升而实际降|平均预测升而实际降|guard违反|',
                  '|---|---|---:|---:|---:|---:|---:|']
        for group, g in value['by_bank_regime'].items():
            for family, p in g['parent_comparisons'].items():
                lines.append(f"|{group}|{family}|{p['selected_parent_rows']}/{p['expected_children']}|{p['average_metric_sacrifices_true_reference']}|{p['predicted_true_reference_gain_actual_loss']}|{p['predicted_mean_gain_actual_loss']}|{p['guard_violation']}|")
        planning = value['planning']
        lines += ['', f"QP {planning['QP_completed']}/{planning['QP_attempts']} 达到间隙要求；原参数域可核查 {planning['QP_original_domain_known']}，线路越界 {planning['QP_original_line_failures']}、源越界 {planning['QP_original_source_failures']}；master {planning['master_completed']}/{planning['master_attempts']} 达到代理间隙。", '',
                  '动作对一致率、并列和缺失分母、逐域及同域平方对照、完整求解界、压缩证书、被拒提案与逐历史误差分解保存于 JSON。它们是描述性诊断，未进行新的误拒率校准。', '']
    Path(path).write_text('\n'.join(lines)+'\n', encoding='utf8')


def run(root, output, *, phases=PHASES, require_complete=False):
    root, output = Path(root), Path(output)
    if output.exists() or output.with_suffix('.md').exists(): raise FileExistsError('preserve previous analysis; choose a new output name')
    frozen, freeze_source = _read(root/'freeze.json')
    if not set(phases) <= set(PHASES): raise ValueError('unknown phase')
    specs_all = [s for p in PHASES for s in frozen['specifications'][p]]
    if len({s['seed'] for s in specs_all}) != len(specs_all): raise ValueError('development/evaluation seeds overlap')
    result = dict(schema='mobile_v4_saved_actions_analysis_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        policy_id=frozen['policy_id'], source_sha256=frozen['source_sha256'], freeze_source=freeze_source,
        independent_unit='one frozen historical seed; future regimes and methods are paired within it',
        gap_decomposition='delta_actual_halfM = delta_Phi + (delta_true_ref_m-delta_Phi) + (delta_actual_halfM-delta_true_ref_m)',
        decomposition_scope='algebraic accounting, not causal identification: last term includes reference/forecast mean error, future measurement/input noise, and numerical AC refit differences; saved solver gaps/failures are reported separately',
        scope='read-only descriptive diagnosis of saved plans; no fitting, AC, new optimization, threshold reuse, or truth-based action choice',
        missing_rule='keep all frozen history, regime, 12-method and 4-candidate slots; numeric summaries use complete physical candidate sets',
        ranking_rule='unique exact action keys within history/regime; duplicate method slots do not create pairs; ties remain separate',
        square_rule='Q is the square-family planning objective; its reporting Phi is not promised monotone',
        methods=list(METHODS), phases={}, input_files=[], tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    for phase in phases:
        specs = frozen['specifications'][phase]; expected = {str(s['index']).zfill(3)+'.json.gz': s for s in specs}; records = {}
        for path in sorted((root/phase).glob('*.json.gz')):
            if path.name not in expected: raise ValueError('extra history file outside frozen phase')
            record, source = _read(path); result['input_files'].append(source)
            if record.get('policy_id') != frozen['policy_id']: raise ValueError('record policy differs from frozen policy')
            records[expected[path.name]['index']] = record
        if require_complete and len(records) != len(specs): raise ValueError('requested phase has missing history files')
        result['phases'][phase] = summarize_records(specs, records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2), encoding='utf8')
    render(result, output.with_suffix('.md'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=(*PHASES, 'all'), default='all'); parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = run(args.root, args.output, phases=PHASES if args.phase == 'all' else (args.phase,), require_complete=args.require_complete)
    print(json.dumps({p:dict(histories=v['history_denominator'], present=v['records_present'],
        monotonicity_violations=v['monotonicity']['violations']) for p, v in result['phases'].items()}, ensure_ascii=False))


if __name__ == '__main__': main()
