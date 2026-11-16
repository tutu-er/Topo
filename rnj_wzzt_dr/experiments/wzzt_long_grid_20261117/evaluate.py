"""Independent read-only scoring for the long WZZT study.

This module imports no fitter, solver, network simulator, or production selector.
The file-system adapter is deliberately separate from the array-only checks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def close(actual, expected, message, *, rtol=1e-9, atol=1e-12):
    require(np.isfinite(actual) and np.isfinite(expected)
            and np.isclose(actual, expected, rtol=rtol, atol=atol), message)


def point_arrays(point, n):
    """Validate a stored laminar response model without cleaning its supports."""
    supports = [tuple(map(int, c)) for c in point['supports']]
    r, x = np.asarray(point['r'], float), np.asarray(point['x'], float)
    require(r.shape == x.shape == (len(supports),), 'Point coefficient dimensions differ')
    require(np.isfinite(r).all() and np.isfinite(x).all()
            and np.all(r >= 0) and np.all(x >= 0), 'Invalid nonnegative coefficients')
    require(len(supports) == len(set(supports)), 'Duplicate stored supports')
    for c in supports:
        require(bool(c) and tuple(sorted(set(c))) == c and 0 <= min(c)
                and max(c) < n, 'Invalid canonical support')
    sets = [set(c) for c in supports]
    for i, a in enumerate(sets):
        for b in sets[i+1:]:
            require(not a & b or a <= b or b <= a, 'Nonlaminar point')
    return supports, r, x


def matrices(point, n):
    supports, r, x = point_arrays(point, n)
    R, X = np.zeros((n, n)), np.zeros((n, n))
    for c, rr, xx in zip(supports, r, x, strict=True):
        R[np.ix_(c, c)] += rr
        X[np.ix_(c, c)] += xx
    return R, X


def observation_arrays(observation, labels=None):
    p, q = np.asarray(observation['P'], float), np.asarray(observation['Q'], float)
    require(p.ndim == 2 and p.shape == q.shape and p.size > 0, 'Invalid P/Q shape')
    if 'Y' in observation:
        y = np.asarray(observation['Y'], float)
    else:
        v, root = np.asarray(observation['V'], float), np.asarray(observation['root'], float)
        require(v.shape == p.shape and root.shape == (p.shape[0],), 'Invalid V/root shape')
        y = root[:, None]**2-v**2
    require(y.shape == p.shape and all(np.isfinite(a).all() for a in (p, q, y)),
            'Nonfinite or mismatched observations')
    if labels is not None:
        require(list(map(int, observation['labels'])) == list(map(int, labels)),
                'Observation terminal ordering changed')
    return p, q, y


def loss_totals(point, observations):
    """Sum actual cells; unequal split lengths are never averaged equally."""
    totals = dict(sse=0., sae=0., cells=0, rows=0)
    for observation in observations:
        p, q, y = observation_arrays(observation)
        R, X = matrices(point, p.shape[1])
        residual = y-p@R.T-q@X.T
        totals['sse'] += float(np.sum(residual**2))
        totals['sae'] += float(np.sum(np.abs(residual)))
        totals['cells'] += y.size
        totals['rows'] += y.shape[0]
    require(totals['cells'] > 0, 'Empty scoring observations')
    totals.update(mse=totals['sse']/totals['cells'], mae=totals['sae']/totals['cells'])
    return totals


def stored_internal(point, n):
    return sum(1 < len(c) < n for c in point['supports'])


def active_supports(point, n, zero_tol):
    require(np.isfinite(zero_tol) and zero_tol >= 0, 'Invalid zero tolerance')
    cs, r, x = point_arrays(point, n)
    return {c for c, rr, xx in zip(cs, r, x, strict=True) if max(rr, xx) > zero_tol}


def strict_select(rows, *, metric='mse', reference=1., tie=1e-12):
    """Reproduce normalized-loss tie, then STORED count/loss/source index."""
    require(rows and np.isfinite(reference) and reference > 0
            and np.isfinite(tie) and tie >= 0, 'Invalid selector configuration')
    require(len({r['index'] for r in rows}) == len(rows), 'Duplicate candidate IDs')
    values = {r['index']: float(r[metric])/reference for r in rows}
    require(all(np.isfinite(v) and v >= 0 for v in values.values()), 'Invalid candidate loss')
    best = min(values.values())
    eligible = [r for r in rows if values[r['index']] <= best+tie]
    selected = min(eligible, key=lambda r: (r['stored_internal'], values[r['index']], r['index']))
    return dict(index=selected['index'], metric=metric, reference=float(reference),
                normalized_tie=float(tie), physical_tie=float(reference*tie),
                value=float(selected[metric]), normalized_value=values[selected['index']])


def physical_geometry(branches, labels, root_bus, zbase_ohm):
    """Physical descendant supports and response coefficients, including weak lines.

    Parallel series edges with identical observed descendants are aggregated for
    identifiable-support metrics, while edge multiplicity/depth is preserved.
    """
    require(np.isfinite(zbase_ohm) and zbase_ohm > 0, 'Invalid impedance base')
    labels = list(map(int, labels)); root_bus = int(root_bus)
    require(len(labels) == len(set(labels)), 'Duplicate physical labels')
    positions = {v: i for i, v in enumerate(labels)}
    parent, children, edges = {}, defaultdict(list), {}
    for edge in branches:
        if str(edge.get('status', 'true')).lower() not in ('true', '1', '1.0'):
            continue
        a, b = int(edge['from_bus']), int(edge['to_bus'])
        require(b not in parent and b != root_bus and a != b, 'Not a rooted radial network')
        parent[b] = a; children[a].append(b); edges[b] = edge
    seen, stack, depths = {}, set(), {root_bus: 0}

    def descend(node):
        require(node not in stack, 'Cycle in physical topology')
        if node in seen:
            return seen[node]
        stack.add(node)
        result = {positions[node]} if node in positions else set()
        for child in children[node]:
            depths[child] = depths[node]+1
            result.update(descend(child))
        stack.remove(node); seen[node] = tuple(sorted(result))
        return seen[node]

    require(descend(root_bus) == tuple(range(len(labels))) and set(parent) <= set(seen),
            'Physical branches or measured terminals disconnected')
    groups = {}
    for child, edge in edges.items():
        support = seen[child]
        if not support:
            continue
        r, x = float(edge['r_ohm']), float(edge['x_ohm'])
        require(np.isfinite(r) and np.isfinite(x) and min(r, x) >= 0, 'Invalid physical impedance')
        row = groups.setdefault(support, dict(support=list(support), branch_depths=[],
                                             r_ohm=0., x_ohm=0., physical_branch_count=0))
        row['branch_depths'].append(depths[child]); row['r_ohm'] += r; row['x_ohm'] += x
        row['physical_branch_count'] += 1
    for support, row in groups.items():
        row.update(min_root_depth=min(row['branch_depths']), max_root_depth=max(row['branch_depths']),
                   downstream_fraction=len(support)/len(labels),
                   kind='common' if len(support) == len(labels) else 'terminal' if len(support) == 1 else 'internal')
    point = dict(supports=[list(c) for c in groups],
                 r=[2*v['r_ohm']/zbase_ohm for v in groups.values()],
                 x=[2*v['x_ohm']/zbase_ohm for v in groups.values()])
    R, X = matrices(point, len(labels))
    return dict(groups=groups, Hstar=np.column_stack((R, X)), labels=labels,
                zbase_ohm=float(zbase_ohm), depth_definition='root depth 0; edge depth is child bus depth')


def structural_metrics(predicted, truth):
    tp, fp, fn = len(predicted & truth), len(predicted-truth), len(truth-predicted)
    return dict(exact=predicted == truth, tp=tp, fp=fp, fn=fn, truth_count=len(truth),
                predicted_count=len(predicted), f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=tp/len(predicted) if predicted else None,
                recall=tp/len(truth) if truth else None,
                missing=[list(c) for c in sorted(truth-predicted)],
                extra=[list(c) for c in sorted(predicted-truth)])


def stratified_topology(point, geometry, zero_tol=1e-9):
    n = len(geometry['labels']); active = active_supports(point, n, zero_tol)
    truth = set(geometry['groups'])
    internal = lambda values: {c for c in values if 1 < len(c) < n}
    result = dict(internal=structural_metrics(internal(active), internal(truth)),
                  full=structural_metrics(active, truth), strata={})
    filters = {
        'common': lambda c, row: len(c) == n,
        'terminal': lambda c, row: len(c) == 1,
        'internal': lambda c, row: 1 < len(c) < n,
        'internal_root_depth_le2': lambda c, row: 1 < len(c) < n and row['min_root_depth'] <= 2,
        'internal_downstream_fraction_ge_half': lambda c, row: 1 < len(c) < n and len(c)/n >= .5,
        'internal_downstream_fraction_lt_half': lambda c, row: 1 < len(c) < n and len(c)/n < .5,
    }
    for name, predicate in filters.items():
        target = {c for c, row in geometry['groups'].items() if predicate(c, row)}
        result['strata'][name] = dict(true_supports=len(target), recovered_supports=len(target & active),
                                     recall=len(target & active)/len(target) if target else None,
                                     physical_branches=sum(geometry['groups'][c]['physical_branch_count'] for c in target))
    extra = active-truth
    # A false support has no true branch depth. Its inferred support-tree depth
    # is explicitly a different diagnostic, never used in true-depth recall.
    false_rows = []
    for c in sorted(extra):
        inferred_depth = 1+sum(set(c) < set(d) for d in active)
        false_rows.append(dict(support=list(c), kind='common' if len(c) == n else 'terminal' if len(c) == 1 else 'internal',
                               downstream_fraction=len(c)/n, true_root_depth=None,
                               predicted_support_tree_depth=inferred_depth))
    result['false_supports'] = false_rows
    result['false_support_counts'] = dict(Counter(row['kind'] for row in false_rows))
    result['internal_fp_root_near_in_predicted_tree'] = sum(
        row['kind'] == 'internal' and row['predicted_support_tree_depth'] <= 2 for row in false_rows)
    result['internal_fp_downstream_fraction_ge_half'] = sum(
        row['kind'] == 'internal' and row['downstream_fraction'] >= .5 for row in false_rows)
    return result


def response_errors(point, geometry, Hstar=None):
    n = len(geometry['labels']); R, X = matrices(point, n)
    h = geometry['Hstar'] if Hstar is None else np.asarray(Hstar, float)
    require(h.shape == (n, 2*n) and np.isfinite(h).all(), 'Invalid physical response matrix')
    matrix = {}
    for name, a, b in [('R', R, h[:, :n]), ('X', X, h[:, n:])]:
        denominator = float(np.linalg.norm(b)); absolute = float(np.linalg.norm(a-b))
        matrix[name] = dict(absolute_frobenius=absolute,
                            relative_frobenius=absolute/denominator if denominator else None,
                            reference_norm=denominator)
    weights = {c: (r, x) for c, r, x in zip(*point_arrays(point, n), strict=True)}
    lines = []
    for c in sorted(set(weights) | set(geometry['groups'])):
        t, f = geometry['groups'].get(c), weights.get(c)
        row = dict(support=list(c), status='matched' if t is not None and f is not None else 'missing' if t is not None else 'extra')
        for j, axis in enumerate(('R', 'X')):
            target = t[axis.lower()+'_ohm'] if t is not None else None
            fitted = f[j]*geometry['zbase_ohm']/2 if f is not None else None
            row[axis] = dict(true_ohm=target, fitted_ohm=fitted,
                             absolute_error_ohm=abs(fitted-target) if target is not None and fitted is not None else None,
                             relative_error=abs(fitted-target)/target if target is not None and target > 0 and fitted is not None else None)
        lines.append(row)
    return dict(matrix=matrix, lines=lines, line_scope='Same stored physical support; zero coefficients retained, missing/extra explicit')


def evaluate_bank(native, train, validation, new_day, geometry, *, reference_mae=None, Hstar=None):
    """Score every native parameter point. No deduplication or refitting."""
    bank = native['frozen_candidates']; require(bool(bank), 'Empty candidate bank')
    labels = list(geometry['labels']); n = len(labels)
    for observation in (train, validation, new_day):
        if observation is not None:
            observation_arrays(observation, labels)
    config = native['config']; reference = float(native['reference_mse'])
    tie, zero = float(config['validation_tie']), float(config['zero_tol'])
    rows = []
    for ordinal, item in enumerate(bank):
        point = item.get('point', item); index = int(item.get('index', ordinal))
        if item.get('labels') is not None:
            require(list(item['labels']) == labels, 'Candidate labels differ')
        scores = dict(train=loss_totals(point, [train]), validation=loss_totals(point, [validation]))
        for loss in ('mse','mae'):
            if loss in point:
                close(scores['train'][loss], point[loss], 'Saved training loss differs: '+loss)
        if new_day is not None:
            scores.update(new_day=loss_totals(point, [new_day]), cumulative=loss_totals(point, [validation, new_day]))
        if 'validation_mse' in item:
            close(scores['validation']['mse'], item['validation_mse'], 'Saved candidate MSE differs')
        if 'validation_scaled_mse' in item:
            close(scores['validation']['mse']/reference, item['validation_scaled_mse'], 'Saved normalized MSE differs')
        if 'validation_mae' in item:
            close(scores['validation']['mae'], item['validation_mae'], 'Saved candidate MAE differs')
        R, X = matrices(point, n)
        for key, actual in (('R', R), ('X', X)):
            if key in item:
                require(np.allclose(actual, np.asarray(item[key]), rtol=1e-12, atol=1e-14),
                        'Saved response matrix differs: '+key)
        rows.append(dict(index=index, stored_internal=stored_internal(point, n), scores=scores,
                         topology=stratified_topology(point, geometry, zero),
                         response=response_errors(point, geometry, Hstar)))
    require(len({r['index'] for r in rows}) == len(rows), 'Duplicate bank IDs')
    selections = {}
    for split in rows[0]['scores']:
        if split == 'train':
            continue
        ranking = [dict(index=r['index'], stored_internal=r['stored_internal'], **r['scores'][split]) for r in rows]
        views = dict(mse=strict_select(ranking, reference=reference, tie=tie),
                     legacy_physical_mae=strict_select(ranking, metric='mae', tie=tie))
        views['normalized_mae'] = (strict_select(ranking, metric='mae', reference=reference_mae, tie=tie)
                                   if reference_mae is not None else None)
        for selection in views.values():
            if selection is not None:
                selected = next(r for r in rows if r['index'] == selection['index'])
                selection['topology'] = selected['topology']
                selection['response'] = selected['response']
        selections[split] = views
    expected = native.get('selected', {}).get('index')
    if expected is not None:
        require(selections['validation']['mse']['index'] == expected, 'Actual native winner violates frozen strict-MSE rule')
        if native['selected'].get('point') is not None:
            winner = next(item.get('point', item) for i, item in enumerate(bank)
                          if int(item.get('index', i)) == expected)
            require(point_arrays(winner, n)[0] == point_arrays(native['selected']['point'], n)[0],
                    'Selected support identity differs')
            for a, b in zip(point_arrays(winner, n)[1:], point_arrays(native['selected']['point'], n)[1:], strict=True):
                require(np.array_equal(a, b), 'Selected coefficients differ')
    expected_mae = native.get('selected_mae_view', {}).get('index')
    if expected_mae is not None:
        require(selections['validation']['legacy_physical_mae']['index'] == expected_mae,
                'Saved physical-MAE diagnostic winner differs')
    return dict(models=rows, selections=selections,
                coverage={view: any(row['topology'][view]['exact'] for row in rows) for view in ('internal', 'full')},
                native_candidate_count=len(rows), reference_mse=reference, reference_mae=reference_mae,
                scoring_scope='validation only and validation+new day; train never included in final cumulative score',
                mae_scope='legacy physical MAE reproduces old diagnostic; normalized MAE requires shared training reference')


def training_score(point, train, reference, alpha, *, loss='mse'):
    return loss_totals(point, [train])[loss]/reference+alpha*stored_internal(point, observation_arrays(train)[0].shape[1])


def solver_evidence(native, train, *, training_loss='mse', training_reference=None):
    """Separate prescan, actual start, native solution, closure, and certificates."""
    cfg = native['config']; alpha = float(cfg['alpha'])
    reference = float(training_reference if training_reference is not None else native['reference_mse'])
    require(reference > 0 and training_loss in ('mse', 'mae'), 'Invalid training objective scale')
    score = lambda p: training_score(p, train, reference, alpha, loss=training_loss)
    result = []
    for ordinal, event in enumerate(native.get('events', [])):
        solver = event.get('solver') or {}
        initial, raw, closed = (event.get(k) for k in ('initial_point', 'raw_point', 'closed_point'))
        start_score = score(initial) if initial is not None else None
        raw_score = score(raw) if raw is not None else None
        closed_score = score(closed) if closed is not None else None
        for actual, key in ((start_score, 'initial_score'), (raw_score, 'native_returned_score'), (closed_score, 'closed_score')):
            if actual is not None and event.get(key) is not None:
                close(actual, event[key], 'Training telemetry differs: '+key, rtol=1e-7)
        if raw_score is not None and solver.get('objective') is not None:
            close(raw_score, solver['objective'], 'Raw native objective differs', rtol=1e-7)
        bound = finite(solver.get('bound')); incumbent = raw_score if raw_score is not None else finite(solver.get('objective'))
        known_same_domain = [v for v in (incumbent, start_score,
                             finite(solver.get('fixed_family_objective', solver.get('fixed_family_qp_objective')))) if v is not None]
        best_known = min(known_same_domain) if known_same_domain else None
        bound_conflict = (solver.get('bound_consistent_with_fixed_family_qp') is False or
                          (bound is not None and best_known is not None
                           and bound > best_known+1e-7*(1+abs(best_known))))
        valid_bound = bound is not None and not bound_conflict and not solver.get('bound_withdrawn', False)
        gap = finite(solver.get('gap')); status = solver.get('status')
        requested_gap = finite(solver.get('requested_gap')); absolute_gap = finite(solver.get('requested_absolute_gap'))
        certificate = bool(valid_bound and incumbent is not None and status == 2
                           and (incumbent-bound <= (absolute_gap or 0.)+1e-7*(1+abs(incumbent))
                                or (gap is not None and requested_gap is not None and gap <= requested_gap)))
        first_time = finite(solver.get('first_incumbent_seconds'))
        progress = solver.get('progress', solver.get('events', []))
        if first_time is None:
            times = [finite(row.get('seconds')) for row in progress
                     if finite(row.get('objective')) is not None and row.get('callback_event') != 'FINAL'
                     and (row.get('callback_event') == 'MIPSOL' or (row.get('incumbent_count') or 0) > 0)]
            times = [t for t in times if t is not None]
            first_time = min(times) if times else None
        improved_times = [finite(row.get('seconds')) for row in progress
                          if start_score is not None and finite(row.get('objective')) is not None
                          and row['objective'] < start_score-float(cfg['gain_tol'])]
        improved_times = [t for t in improved_times if t is not None]
        prior_bounds = [finite(row.get('bound')) for row in progress if row.get('callback_event') != 'FINAL']
        prior_bounds = [b for b in prior_bounds if b is not None]
        max_prior_bound = max(prior_bounds) if prior_bounds else None
        history_conflict = (max_prior_bound is not None and best_known is not None
                            and max_prior_bound > best_known+1e-7*(1+abs(best_known)))
        if history_conflict:
            valid_bound = certificate = False
        result.append(dict(event_index=ordinal, stage=event.get('stage'), accepted=event.get('accepted', False),
                           status=status, solve_started=bool(solver.get('solve_started', False)), timed_out=status == 9,
                           seconds=finite(solver.get('seconds')), first_incumbent_seconds=first_time,
                           first_incumbent_scope='unavailable' if first_time is None else 'saved native callback; may include submitted start',
                           first_improvement_over_actual_initial_seconds=min(improved_times) if improved_times else None,
                           initial_source=event.get('initial_source'), actual_initial_score=start_score,
                           initial_record_available=initial is not None, raw_score=raw_score, closed_score=closed_score,
                           native_gain=None if start_score is None or raw_score is None else start_score-raw_score,
                           native_improved_initial=None if start_score is None or raw_score is None else raw_score < start_score-float(cfg['gain_tol']),
                           closure_gain=None if raw_score is None or closed_score is None else raw_score-closed_score,
                           pretrial_seconds=finite(event.get('pretrial_seconds')), pretrial_QP_count=event.get('pretrial_QP_count'),
                           current_to_initial_gain=finite(event.get('current_to_initial_gain')),
                           raw_lower_bound=bound, valid_lower_bound=bound if valid_bound else None,
                           best_same_domain_objective=best_known,
                           maximum_prior_lower_bound=max_prior_bound, bound_history_conflict=history_conflict,
                           bound_conflict=bound_conflict, final_gap=gap, native_subproblem_certificate=certificate,
                           global_tree_optimality=False, returned_point_source=event.get('returned_point_source')))
    return dict(events=result, native_calls=sum(r['solve_started'] for r in result),
                timeouts=sum(r['timed_out'] for r in result), native_improvement_known=sum(r['native_improved_initial'] is not None for r in result),
                native_improvements=sum(r['native_improved_initial'] is True for r in result),
                bound_conflicts=sum(r['bound_conflict'] for r in result),
                bound_history_conflicts=sum(r['bound_history_conflict'] for r in result),
                native_seconds=sum(r['seconds'] or 0. for r in result),
                QP_pretrial_seconds=sum(r['pretrial_seconds'] or 0. for r in result),
                total_seconds=finite(native.get('total_seconds')), stop_reason=native.get('stop_reason'),
                global_tree_optimality=False)


def status_summary(rows):
    """Pending and failed outcomes retain their denominator, never become zero accuracy."""
    counts = Counter(row['status'] for row in rows)
    completed = [row for row in rows if row['status'] == 'complete']
    correct = sum(bool(row.get('internal_correct')) for row in completed)
    terminal = bool(rows) and all(row['status'] in {
        'complete', 'failed', 'interrupted', 'budget_unrun', 'audit_failed', 'missing_final_result'
    } for row in rows)
    return dict(planned=len(rows), status_counts=dict(counts), completed=len(completed),
                internal_correct=correct, accuracy_among_completed=correct/len(completed) if completed else None,
                accuracy_all_planned=correct/len(rows) if terminal else None,
                all_tasks_terminal=terminal, all_tasks_successful=len(completed) == len(rows),
                complete=terminal)


def paired_summary(left, right):
    """Pair by locked input ID; exclude neither failures nor absent counterparts from accounting."""
    require(len({r['id'] for r in left}) == len(left) and len({r['id'] for r in right}) == len(right), 'Duplicate paired input IDs')
    a, b = {r['id']: r for r in left}, {r['id']: r for r in right}
    keys = sorted(set(a) | set(b)); pairs = []
    for key in keys:
        x, y = a.get(key), b.get(key)
        if x is not None and y is not None and x['status'] == y['status'] == 'complete':
            pairs.append((key, x, y))
    gains = [key for key, x, y in pairs if not x['internal_correct'] and y['internal_correct']]
    losses = [key for key, x, y in pairs if x['internal_correct'] and not y['internal_correct']]
    full = [(key,x,y) for key,x,y in pairs if 'full_correct' in x and 'full_correct' in y]
    times = [y['seconds']/x['seconds'] for _,x,y in pairs
             if (x.get('seconds') or 0)>0 and (y.get('seconds') or 0)>0]
    return dict(planned_union=len(keys), completed_pairs=len(pairs), unpaired_or_incomplete=len(keys)-len(pairs),
                gains=len(gains), losses=len(losses), gain_ids=gains, loss_ids=losses,
                net_internal_correct=len(gains)-len(losses),
                full_support_pairs=len(full), full_support_gains=sum(not x['full_correct'] and y['full_correct'] for _,x,y in full),
                full_support_losses=sum(x['full_correct'] and not y['full_correct'] for _,x,y in full),
                paired_time_ratio_right_over_left_median=float(np.median(times)) if times else None,
                paired_time_ratio_count=len(times),
                scope='Descriptive paired inputs; repetitions/grid points share physical days, no iid confidence claim')


class Reader:
    def __init__(self):
        self.sources = {}

    def checked(self, ref):
        require(isinstance(ref, dict) and ref.get('path') and ref.get('sha256'), 'Expected hash-bound file reference')
        path = Path(ref['path']).resolve()
        require(path.is_file(), 'Missing source: '+str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == ref['sha256'], 'Changed source: '+str(path))
        self.sources[str(path)] = dict(path=str(path), sha256=digest)
        return path

    def json(self, ref):
        return json.loads(self.checked(ref).read_text(encoding='utf8'))

    def npz(self, ref):
        with np.load(self.checked(ref), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf8')


def _list_payload(value, key):
    result = value if isinstance(value, list) else value[key]
    require(isinstance(result, list), 'Expected list: '+key)
    return result


def _input_rows(reader, protocol):
    """A manifest may combine panels; repeated identical entries are permitted."""
    references = protocol.get('inputs_refs')
    if references is None:
        reference = protocol.get('inputs_ref', protocol.get('inputs'))
        require(reference is not None, 'Protocol has no hash-bound input manifest')
        references = [reference]
    elif isinstance(references, dict):
        references = list(references.values())
    entries = {}
    for reference in references:
        value = reader.json(reference)
        rows = _list_payload(value, 'inputs')
        for row in rows:
            require('id' in row, 'Input lacks ID')
            if row['id'] in entries:
                require(entries[row['id']] == row, 'Conflicting duplicate input entry')
            entries[row['id']] = row
    return entries


def _source_ref_checks(reader, value):
    """Verify references recursively, without opening future data during fitting."""
    if isinstance(value, dict):
        if value.get('path') and value.get('sha256'):
            reader.checked(value)
        else:
            for item in value.values():
                _source_ref_checks(reader, item)
    elif isinstance(value, list):
        for item in value:
            _source_ref_checks(reader, item)


def load_input(reader, entry):
    """Truth is opened here, in this post-fit evaluator only."""
    require(entry.get('status', 'complete') == 'complete', 'Input preparation did not complete')
    files = entry['files']
    observations = {}
    for key in ('train', 'validation', 'new_day'):
        reference = files.get(key+'.npz', files.get(key))
        require(reference is not None, 'Missing observation ref: '+key)
        observations[key] = reader.npz(reference)
    labels = list(map(int, entry['labels']))
    for value in observations.values():
        observation_arrays(value, labels)
    lineage = reader.json(entry['observations_sealed']) if entry.get('observations_sealed') else None
    if lineage:
        for key, observation in observations.items():
            sampling = lineage['sampling'][key]
            require(len(sampling['indices']) == len(observation['P']), 'Sealed observation row count differs')
        ix = [set(lineage['sampling'][key]['indices']) for key in ('train', 'validation', 'new_day')]
        require(all(not ix[i] & ix[j] for i in range(3) for j in range(i+1, 3)), 'Train/validation/new rows overlap')
    if 'N' in entry:
        require(len(observations['train']['P']) == entry['N'], 'Training N differs')
    evaluation = reader.json(entry['evaluation_ref'])
    refs = evaluation['evaluation_case_refs']
    metadata = reader.json(refs['metadata.json'])
    with reader.checked(refs['branches.csv']).open(encoding='utf8', newline='') as stream:
        branches = list(csv.DictReader(stream))
    zbase = float(entry['zbase_ohm'])
    close(zbase, float(metadata['base_kv'])**2/float(metadata['base_mva']), 'Impedance base differs')
    geometry = physical_geometry(branches, labels, int(metadata['root_bus']), zbase)
    hstar_ref = entry.get('Hstar_ref', evaluation.get('Hstar_ref'))
    Hstar = None
    if hstar_ref:
        saved = reader.npz(hstar_ref)
        require(list(map(int, saved['labels'])) == labels, 'Hstar terminal order differs')
        Hstar = np.asarray(saved['Hstar'], float)
        require(Hstar.shape == geometry['Hstar'].shape
                and np.allclose(Hstar, geometry['Hstar'], rtol=1e-10, atol=1e-12),
                'Sealed Hstar differs from physical branch reconstruction')
    return dict(**observations, geometry=geometry, Hstar=Hstar,
                row_counts={key: len(value['P']) for key, value in observations.items()})


def observation_fingerprint(observation):
    h = hashlib.sha256(repr(tuple(map(int, observation['labels']))).encode())
    for values in observation_arrays(observation):
        a = np.ascontiguousarray(values, dtype=np.float64)
        h.update(repr(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def _mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def group_summary(rows):
    result = status_summary(rows)
    complete = [row for row in rows if row['status'] == 'complete']
    result['metrics'] = {}
    for split in ('validation', 'new_day', 'cumulative'):
        result['metrics'][split] = {}
        for view in ('mse', 'normalized_mae', 'legacy_physical_mae'):
            values = [row['bank']['selections'].get(split, {}).get(view) for row in complete]
            values = [v for v in values if v is not None]
            metrics = dict(evaluated=len(values), internal_correct=sum(v['topology']['internal']['exact'] for v in values),
                           full_correct=sum(v['topology']['full']['exact'] for v in values),
                           internal_F1_mean=_mean(v['topology']['internal']['f1'] for v in values),
                           R_relative_error_mean=_mean(v['response']['matrix']['R']['relative_frobenius'] for v in values),
                           X_relative_error_mean=_mean(v['response']['matrix']['X']['relative_frobenius'] for v in values))
            for kind in ('internal', 'full'):
                correct = metrics[kind+'_correct']
                metrics[kind+'_accuracy_completed'] = correct/len(values) if values else None
                metrics[kind+'_accuracy_planned'] = (correct/len(rows) if rows and result['all_tasks_terminal']
                                                     and len(values) == len(complete) else None)
            metrics['strata'] = {}
            if values:
                for stratum in values[0]['topology']['strata']:
                    s = [v['topology']['strata'][stratum] for v in values]
                    tp = sum(v['recovered_supports'] for v in s)
                    fn = sum(v['true_supports']-v['recovered_supports'] for v in s)
                    metrics['strata'][stratum] = dict(tp=tp, fn=fn, recall=tp/(tp+fn) if tp+fn else None)
                metrics['false_supports'] = {
                    kind: sum(v['topology']['false_support_counts'].get(kind, 0) for v in values)
                    for kind in ('common', 'internal', 'terminal')}
                metrics['false_supports']['large_internal'] = sum(
                    v['topology']['internal_fp_downstream_fraction_ge_half'] for v in values)
                metrics['false_supports']['near_predicted_root_internal'] = sum(
                    v['topology']['internal_fp_root_near_in_predicted_tree'] for v in values)
            result['metrics'][split][view] = metrics
    result['coverage'] = {kind: sum(row['bank']['coverage'][kind] for row in complete) for kind in ('internal', 'full')}
    result['coverage']['completed_banks'] = len(complete)
    result['solver'] = dict(native_calls=sum(r['solver']['native_calls'] for r in complete),
        timeouts=sum(r['solver']['timeouts'] for r in complete),
        native_improvements=sum(r['solver']['native_improvements'] for r in complete),
        native_improvement_known=sum(r['solver']['native_improvement_known'] for r in complete),
        bound_conflicts=sum(r['solver']['bound_conflicts'] for r in complete),
        bound_history_conflicts=sum(r['solver']['bound_history_conflicts'] for r in complete),
        task_seconds_sum=sum(r.get('seconds') or 0. for r in rows),
        completed_task_seconds_mean=_mean(r.get('seconds') for r in complete),
        native_seconds_sum=sum(r['solver']['native_seconds'] for r in complete),
        QP_pretrial_seconds_sum=sum(r['solver']['QP_pretrial_seconds'] for r in complete),
        total_cap_exceeded=sum(bool(r.get('total_cap_exceeded')) for r in rows),
        global_tree_optimality=False)
    return result


def _paired_views(rows):
    partitions = defaultdict(list)
    for row in rows:
        partitions[row['panel']].append(row)
    result = []
    for panel, panel_rows in sorted(partitions.items()):
        methods = sorted({r['method'] for r in panel_rows})
        losses = sorted({r['loss'] for r in panel_rows})
        comparisons = []
        for loss in losses:
            for i, left in enumerate(methods):
                for right in methods[i+1:]:
                    comparisons.append(((left, loss), (right, loss)))
        for method in methods:
            if 'mae' in losses and 'mse' in losses:
                comparisons.append(((method, 'mae'), (method, 'mse')))
        for left, right in comparisons:
            for split in ('validation', 'new_day', 'cumulative'):
                sides = []
                for method, loss in (left, right):
                    side = []
                    for row in panel_rows:
                        if (row['method'], row['loss']) == (method, loss):
                            s = dict(id=row['input_id'], status=row['status'])
                            if row['status'] == 'complete':
                                s['internal_correct'] = row['bank']['selections'][split]['mse']['topology']['internal']['exact']
                                s['full_correct'] = row['bank']['selections'][split]['mse']['topology']['full']['exact']
                                s['seconds'] = row.get('seconds')
                            side.append(s)
                    sides.append(side)
                result.append(dict(panel=panel, split=split, selection='strict normalized MSE',
                                   left=dict(method=left[0], loss=left[1]), right=dict(method=right[0], loss=right[1]),
                                   **paired_summary(*sides)))
    return result


def evaluate_run(run, out):
    """Read-only study evaluation. Existing output directories are never overwritten."""
    began = perf_counter(); run, out = Path(run).resolve(), Path(out).resolve()
    require(not out.exists(), 'Output exists; choose a new directory')
    out.mkdir(parents=True)
    reader = Reader(); protocol_ref = file_ref(run/'protocol.json')
    protocol = reader.json(protocol_ref)
    tasks_ref = protocol.get('tasks_ref', file_ref(run/'tasks.json'))
    tasks = _list_payload(reader.json(tasks_ref), 'tasks')
    require(len({t['task_id'] for t in tasks}) == len(tasks), 'Duplicate task IDs')
    require(len({(t['panel'], t['input_id'], t['method'], t['loss']) for t in tasks}) == len(tasks), 'Duplicate task coordinates')
    entries = _input_rows(reader, protocol)
    _source_ref_checks(reader, protocol.get('sources', []))
    root_complete = (run/'complete.json').is_file()
    if root_complete:
        reader.json(file_ref(run/'complete.json'))
    _write(out/'protocol.json', dict(source_protocol=protocol_ref, tasks=tasks_ref,
        evaluator=file_ref(__file__), scope='Post-fit independent array scoring; no solver, fit, AC, or new measurements'))
    raw = {}; preliminary_errors = {}; reference_mae = {}; references_mse = {}
    for task in tasks:
        directory = (run/task['result_dir']).resolve()
        require(directory.is_relative_to(run), 'Task result escapes run directory')
        path = directory/'result.json'
        if not path.is_file():
            continue
        try:
            record = reader.json(file_ref(path)); require(record['task_id'] == task['task_id'], 'Task ID mismatch')
            if record.get('config') is not None:
                require(record['config'] == task['config'], 'Task configuration differs from locked plan')
            if record.get('source_protocol'):
                require(reader.checked(record['source_protocol']) == run/'protocol.json', 'Wrong task protocol')
            _source_ref_checks(reader, record.get('input_refs', []))
            if record.get('status') == 'complete':
                # Metadata-only first pass avoids retaining all native callback/fit records in RAM.
                native = (reader.json(record['native']) if record.get('reference_mse') is None
                          or record.get('training_reference_loss') is None else {})
                reference = float(record['reference_mse'] if record.get('reference_mse') is not None else native['reference_mse'])
                key = task['input_id']
                if key in references_mse:
                    close(reference, references_mse[key], 'Input MSE reference differs across tasks')
                references_mse[key] = reference
                if task['loss'] == 'mae':
                    ref = float(record['training_reference_loss'] if record.get('training_reference_loss') is not None else native['reference_loss'])
                    if key in reference_mae:
                        close(ref, reference_mae[key], 'Input MAE reference differs across tasks')
                    reference_mae[key] = ref
            raw[task['task_id']] = record
        except Exception as exc:
            preliminary_errors[task['task_id']] = repr(exc)
    cache = {}; rows = []; errors = []
    for task in tasks:
        entry = entries.get(task['input_id'], {})
        row = dict(task_id=task['task_id'], input_id=task['input_id'], panel=task['panel'],
                   method=task['method'], loss=task['loss'], case=entry.get('case'),
                   N=entry.get('N', entry.get('training_rows')), pq_sd=entry.get('pq_sd', entry.get('pq_relative_sd')),
                   root_sd=entry.get('root_relative_sd', entry.get('meter_relative_sd')),
                   repeat=entry.get('repeat', entry.get('history_repeat')), sample_mode=entry.get('sample_mode'),
                   status='missing_final_result' if root_complete else 'pending')
        try:
            if task['task_id'] in preliminary_errors:
                raise ValueError(preliminary_errors[task['task_id']])
            if task['task_id'] in raw:
                record = raw[task['task_id']]
                row.update(status=record['status'], seconds=finite(record.get('seconds')),
                           total_cap_exceeded=record.get('total_cap_exceeded'), result_ref=file_ref(run/task['result_dir']/'result.json'),
                           native_ref=record.get('native'), failure=record.get('error', record.get('reason')))
                require(row['status'] in ('complete','failed','interrupted','budget_unrun','partial'), 'Unknown task status')
                if row['status'] == 'complete':
                    require(bool(entry), 'Task input missing from manifest')
                    native = reader.json(record['native'])
                    if task.get('config') is not None:
                        require(native['config'] == task['config'], 'Native configuration differs from locked plan')
                    require(not native.get('truth_used', False) and not native.get('test_used', False)
                            and not record.get('truth_used', False) and not record.get('new_day_loaded', False),
                            'Fit receipt declares use of truth or evaluation day')
                    require(native['config'].get('loss', native.get('training_loss', 'mse')) == task['loss'], 'Training loss mismatch')
                    close(native['reference_mse'], references_mse[task['input_id']], 'Result/native MSE reference differs')
                    if record.get('training_reference_loss') is not None:
                        close(record['training_reference_loss'], native.get('reference_loss', native['reference_mse']),
                              'Result/native training reference differs')
                    if task['input_id'] not in cache:
                        cache[task['input_id']] = load_input(reader, entry)
                    data = cache[task['input_id']]
                    for key in ('train','validation'):
                        if record.get('input_refs'):
                            require(record['input_refs'][key+'.npz'] == entry['files'][key+'.npz'], 'Task used different '+key+' source')
                        fingerprint = observation_fingerprint(data[key])
                        if record.get('input_fingerprints'):
                            require(record['input_fingerprints'][key] == fingerprint, 'Task observed fingerprint differs: '+key)
                        if native.get('handoff', {}).get(key+'ing_sha256' if key == 'train' else 'validation_sha256'):
                            require(native['handoff'][key+'ing_sha256' if key == 'train' else 'validation_sha256'] == fingerprint,
                                    'Native handoff observed fingerprint differs: '+key)
                    bank = evaluate_bank(native, data['train'], data['validation'], data['new_day'], data['geometry'],
                                         reference_mae=reference_mae.get(task['input_id']), Hstar=data['Hstar'])
                    training_reference = record.get('training_reference_loss', native.get('reference_loss', native['reference_mse']))
                    row.update(bank=bank, solver=solver_evidence(native, data['train'], training_loss=task['loss'],
                                                               training_reference=training_reference), row_counts=data['row_counts'],
                               internal_correct=bank['selections']['validation']['mse']['topology']['internal']['exact'])
        except Exception as exc:
            row.update(original_status=row['status'], status='audit_failed', error=repr(exc))
            errors.append(dict(task_id=task['task_id'], error=repr(exc)))
        rows.append(row)
    with (out/'records.jsonl').open('w', encoding='utf8') as stream:
        for row in rows:
            stream.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False)+'\n')
    grouped = defaultdict(list); panels = defaultdict(list)
    for row in rows:
        grouped[(row['panel'], row['case'], row['N'], row['pq_sd'], row['root_sd'], row['method'], row['loss'])].append(row)
        panels[(row['panel'], row['method'], row['loss'])].append(row)
    groups = [dict(zip(('panel','case','N','pq_sd','root_sd','method','loss'), key), **group_summary(value))
              for key, value in sorted(grouped.items(), key=lambda x: str(x[0]))]
    panel_summaries = [dict(panel=key[0], method=key[1], loss=key[2], **group_summary(value))
                       for key, value in sorted(panels.items())]
    # Detect changes even if a result was rewritten after being read.
    for reference in list(reader.sources.values()):
        try:
            reader.checked(reference)
        except Exception as exc:
            errors.append(dict(scope='source_changed_during_evaluation', error=repr(exc)))
    summary = status_summary(rows)
    report = dict(status='failed' if errors else ('passed' if summary['all_tasks_terminal'] else 'incomplete'),
                  source_protocol=protocol_ref, source_tasks=tasks_ref, source_inputs=list(reader.sources.values()),
                  evaluator=file_ref(__file__), records=file_ref(out/'records.jsonl'), **summary,
                  errors=errors, groups=groups, panels=panel_summaries, paired=_paired_views(rows),
                  elapsed_seconds=perf_counter()-began,
                  metric_scope='Internal proper clades and full supports are separate; full includes singletons and any common stem.',
                  denominator_scope='Pending tasks have no accuracy. At terminal completion failures/unrun remain in planned accuracy; completed subset also shown.',
                  dependence_scope='Panels, training lengths, noise levels and repeated noise share four physical networks/days. Long and grid can repeat the same input under distinct budgets.',
                  truth_scope='Physical impedances/supports used only in post-fit evaluation; no true-parameter fit or new AC execution.')
    _write(out/'report.json', report)
    fields=['panel','case','N','pq_sd','root_sd','method','loss','planned','completed','split','selection','internal_correct',
            'full_correct','internal_accuracy_completed','internal_accuracy_planned','full_accuracy_completed',
            'full_accuracy_planned','internal_F1_mean','R_relative_error_mean','X_relative_error_mean']
    with (out/'summary.csv').open('w', encoding='utf8', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for group in groups:
            for split, views in group['metrics'].items():
                for view, metrics in views.items():
                    writer.writerow({**{k:group.get(k) for k in fields if k in group},
                        'split':split,'selection':view,**{k:metrics[k] for k in fields if k in metrics}})
    lines=['# Independent frozen-candidate evaluation','',
        f"Status: {report['status']}; planned {report['planned']}, completed {report['completed']}, audit errors {len(errors)}.",'',
        '| Panel | Method | Training loss | Complete / planned | Internal val | Internal val+new | Full val+new | Mean seconds |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for group in panel_summaries:
        val=group['metrics']['validation']['mse'];cum=group['metrics']['cumulative']['mse']
        sec=group['solver']['completed_task_seconds_mean']
        lines.append(f"| {group['panel']} | {group['method']} | {group['loss']} | {group['completed']}/{group['planned']} | "
                     f"{val['internal_correct']} | {cum['internal_correct']} | {cum['full_correct']} | {sec if sec is not None else 'NA'} |")
    lines += ['',report['denominator_scope'],'',report['dependence_scope'],'',
              'The table uses the original strict normalized MSE rule. MAE cross-selection, all negatives, stratified supports and per-task solver evidence are retained in summary.csv and records.jsonl.',
              'Native subproblem bounds apply only where consistent with saved feasible points; they do not certify the globally best topology.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    _write(out/'complete.json', dict(status=report['status'], report=file_ref(out/'report.json'),
                                   records=report['records'], errors=len(errors)))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();result=evaluate_run(args.run,args.out)
    if not args.no_plots:
        plot_results(args.out)
    print(json.dumps({k:result[k] for k in ('status','planned','completed','errors')},ensure_ascii=False))
    if result['status']=='failed':
        raise SystemExit(1)


def plot_results(out):
    """Static aggregate figures; only completed subsets are colored."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    accuracy_cmap=plt.get_cmap('viridis').copy();accuracy_cmap.set_bad('.90')
    time_cmap=plt.get_cmap('coolwarm').copy();time_cmap.set_bad('.90')
    out=Path(out);report=json.loads((out/'report.json').read_text(encoding='utf8'))
    records=[json.loads(line) for line in (out/'records.jsonl').read_text(encoding='utf8').splitlines() if line]
    figures=out/'figures';figures.mkdir(exist_ok=True)
    refs=[]
    for panel,case in sorted({(r['panel'],r['case']) for r in records if r['status']=='complete'}):
        groups=[g for g in report['groups'] if g['panel']==panel and g['case']==case]
        methods=sorted({g['method'] for g in groups})
        ns=sorted({g['N'] for g in groups});axis_key='root_sd' if panel=='root_noise' else 'pq_sd'
        noises=sorted({g[axis_key] for g in groups})
        labels=[format(x*100,'.3g') for x in noises]
        fig,axes=plt.subplots(2,len(methods),figsize=(max(6.,4.1*len(methods)),7.4),squeeze=False)
        for col,method in enumerate(methods):
            for i,loss in enumerate(('mae','mse')):
                ax=axes[i,col];matrix=np.full((len(ns),len(noises)),np.nan);notes={}
                for g in groups:
                    if (g['method'],g['loss'])!=(method,loss):continue
                    row,column=ns.index(g['N']),noises.index(g[axis_key]);m=g['metrics']['cumulative']['mse']
                    matrix[row,column]=m['internal_accuracy_completed'] if m['evaluated'] else np.nan
                    notes[(row,column)]=f"{m['internal_correct']}/{m['evaluated']}\n({g['completed']}/{g['planned']})"
                im=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap=accuracy_cmap,aspect='auto')
                for (row,column),note in notes.items():
                    ax.text(column,row,note,ha='center',va='center',fontsize=8,
                            color='white' if np.isfinite(matrix[row,column]) and matrix[row,column]<.5 else 'black')
                ax.set(xticks=range(len(noises)),xticklabels=labels,yticks=range(len(ns)),yticklabels=ns,
                       xlabel='Added root-voltage SD (%)' if axis_key=='root_sd' else 'Added P/Q SD (%)',
                       ylabel='Training rows N',title=method+' / train '+loss.upper())
        fig.suptitle(f'{case} — {panel}\nInternal recovery, validation + new day; strict MSE selection',fontsize=13)
        fig.subplots_adjust(top=.86,bottom=.15,left=.075,right=.88,hspace=.46,wspace=.4)
        cax=fig.add_axes((.91,.22,.015,.5));fig.colorbar(im,cax=cax,label='Correct / completed')
        fig.text(.05,.025,'Cell: correct / evaluated; (completed / planned). Gray = no completed result.\n'
                 'Noise/length cells share physical days; panels are not pooled as independent trials.',fontsize=9)
        name=f'{panel}__{case}__accuracy'
        for suffix in ('png','pdf'):
            path=figures/(name+'.'+suffix);fig.savefig(path,dpi=180);refs.append(file_ref(path))
        plt.close(fig)
        selected=[r for r in records if r['panel']==panel and r['case']==case]
        lookup={(r['input_id'],r['method'],r['loss']):r for r in selected}
        fig,axes=plt.subplots(1,len(methods),figsize=(max(6.,4.1*len(methods)),4.7),squeeze=False)
        for col,method in enumerate(methods):
            ax=axes[0,col];matrix=np.full((len(ns),len(noises)),np.nan);cells=defaultdict(list)
            for r in selected:
                if r['method']!=method or r['loss']!='mse' or r['status']!='complete':continue
                other=lookup.get((r['input_id'],method,'mae'))
                if other and other['status']=='complete' and (other.get('seconds') or 0)>0 and (r.get('seconds') or 0)>0:
                    cells[(ns.index(r['N']),noises.index(r[axis_key]))].append(r['seconds']/other['seconds'])
            for key,ratios in cells.items():matrix[key]=np.log2(np.median(ratios))
            im=ax.imshow(np.ma.masked_invalid(matrix),norm=TwoSlopeNorm(vmin=-3,vcenter=0,vmax=3),cmap=time_cmap,aspect='auto')
            for (row,column),ratios in cells.items():
                ax.text(column,row,f'{np.median(ratios):.2f}x\nn={len(ratios)}',ha='center',va='center',fontsize=9)
            ax.set(xticks=range(len(noises)),xticklabels=labels,yticks=range(len(ns)),yticklabels=ns,
                   xlabel='Added root-voltage SD (%)' if axis_key=='root_sd' else 'Added P/Q SD (%)',
                   ylabel='Training rows N',title=method)
        fig.suptitle(f'{case} — {panel}\nPaired task wall time: MSE / MAE (median ratio)',fontsize=13)
        fig.subplots_adjust(top=.79,bottom=.2,left=.075,right=.88,wspace=.4)
        cax=fig.add_axes((.91,.24,.015,.46));fig.colorbar(im,cax=cax,label='log2(time ratio), clipped color scale')
        fig.text(.05,.025,'Only both-completed pairs with positive recorded times. Failed/unrun tasks remain in report tables.\n'
                 'Task wall time includes setup and fitting; solver-only time is reported separately.',fontsize=9)
        name=f'{panel}__{case}__paired_time'
        for suffix in ('png','pdf'):
            path=figures/(name+'.'+suffix);fig.savefig(path,dpi=180);refs.append(file_ref(path))
        plt.close(fig)
    _write(figures/'manifest.json',dict(source_report=file_ref(out/'report.json'),
        source_records=file_ref(out/'records.jsonl'),generator=file_ref(__file__),figures=refs))
    return refs


def study(runroot, allow_partial=True):
    """Controller entry point; every call creates a new immutable report directory."""
    runroot=Path(runroot).resolve()
    parent=runroot/'evaluations';parent.mkdir(exist_ok=True)
    prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=parent/prefix
    report=evaluate_run(runroot,out)
    plot_results(out)
    receipt=dict(status=report['status'],report=file_ref(out/'report.json'),output=str(out),
                 planned=report['planned'],completed=report['completed'],all_tasks_terminal=report['all_tasks_terminal'])
    _write(out/'controller_receipt.json',receipt)
    if not allow_partial and not report['all_tasks_terminal']:
        raise RuntimeError('Final evaluation requested with pending tasks; partial report preserved at '+str(out))
    if report['status']=='failed':
        raise RuntimeError('Independent evaluation failed; complete errors preserved at '+str(out))
    return receipt


if __name__ == '__main__':
    main()
