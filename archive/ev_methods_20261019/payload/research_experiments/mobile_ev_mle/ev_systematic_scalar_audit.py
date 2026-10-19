"""Read-only source-QP and scalar chord arithmetic audit; no optimization.

This checks recorded feasible sources and explicit dual formulas.  MILP global
bound validity still relies on the stated solver bound and the audited model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def close(a, b, label, *, atol=2e-7, rtol=2e-9):
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.allclose(x, y, atol=atol, rtol=rtol):
        raise ValueError(label)


def qp_check(c, A, lower, upper, center, uB, row):
    """Full-SSE feasible primal and convex first-order source-box lower."""
    xi = np.asarray(row['physical_parameter'], float)
    if np.any(xi < lower - 1e-10) or np.any(xi > upper + 1e-10):
        raise ValueError('QP source box violation')
    residual = c + uB + A @ (xi - center)
    close(residual, row['residual'], 'QP residual')
    primal = float(residual @ residual)
    close(primal, row['primal_upper'], 'QP primal')
    g = 2 * A.T @ residual
    correction = float(np.minimum(g * (lower - xi), g * (upper - xi)).sum())
    guard = 256 * np.finfo(float).eps * (1 + abs(primal) + abs(correction)
        + float(abs(g) @ (abs(lower) + abs(upper) + 2 * abs(xi))))
    dual = max(0., min(primal, primal + correction - guard))
    close(guard, row['numeric_guard'], 'QP roundoff guard', atol=1e-12)
    close(dual, row['tangent_lower_numeric'], 'QP convex source-box lower')
    if dual > primal + 2e-7:
        raise ValueError('QP lower exceeds primal')
    return primal, dual, float(np.max(abs(residual - row['residual'])))


def node_check(c, b, G, lower, upper, center, row, power_range=None):
    p = float(row['p'])
    # 110 keeps the inner QP diagnostics compact; the complete feasible witness
    # and residual remain on the node.  Rebuild its primal from those arrays.
    source = np.asarray(row.get('physical_source', row['source_QP'].get('physical_parameter')), float)
    if source.shape != lower.shape or not np.isfinite(source).all():
        raise ValueError('finite matching node source required')
    if np.any(source < lower - 1e-10) or np.any(source > upper + 1e-10):
        raise ValueError('node source box violation')
    independent_residual = c + b * p + G @ (source - center)
    residual = np.asarray(row['residual'], float)
    close(independent_residual, residual, 'scalar feasible-source residual')
    error = float(np.max(abs(independent_residual - residual)))
    primal = float(independent_residual @ independent_residual)
    if 'physical_parameter' in row['source_QP']:
        qp_check(c, G, lower, upper, center, b * p, row['source_QP'])
    close(.5 * primal, row['primal_upper'], 'half-SSE scalar upper')
    coefficient = G.T @ residual
    intercept = float(residual @ (c - G @ center) - .5 * (residual @ residual)
        + coefficient @ np.where(coefficient >= 0, lower, upper))
    guard = float(row['dual_numeric_guard'])
    if not np.isfinite(guard) or guard < 0:
        raise ValueError('invalid scalar numerical guard')
    if power_range is not None:
        scale = max(abs(p), *(abs(float(v)) for v in power_range))
        d = c - G @ center
        expected_guard = 512 * np.finfo(float).eps * (1 + abs(residual @ d)
            + .5 * abs(residual @ residual)
            + float(abs(coefficient) @ np.maximum(abs(lower), abs(upper)))
            + abs(float(b @ residual)) * scale)
        close(expected_guard, guard, 'scalar numerical guard', atol=1e-20)
    close(intercept - guard, row['dual_intercept'], 'scalar Fenchel intercept')
    close(b @ residual, row['dual_slope'], 'scalar dual slope')
    dual = max(0., intercept + float(b @ residual) * p - guard)
    close(dual, row['dual_lower'], 'scalar point dual lower')
    close(row['primal_upper'] - row['dual_lower'], row['primal_dual_gap'], 'endpoint gap')
    return error


def _aggregate(names, pair_names, values, history):
    result = []
    for a in names:
        result.append(min(float(v) + history.get(b, 0.) - history.get(a, 0.)
            for (aa, b), v in zip(pair_names, values) if aa == a))
    return float(np.mean(result))


def audit_record(record):
    began = perf_counter()
    metrics = dict(tables=0, new_table_QP_calls=0, endpoint_node_references=0,
        action_QP_calls=0, failed_saved_QPs=0, raw_MILP_calls=0, physical_raw_actions=0,
        available_intervals=0, missing_intervals=0, raw_operation_failures=[], errors=[],
        max_QP_residual_error=0., max_endpoint_gap=0.)
    pairs = record['selection']['frozen_pairs']
    names = [row['name'] for row in record['candidates']]
    pair_names = [tuple(pair['name'].split('__to__')) for pair in pairs]
    n = len(record['training']['p'][0])
    for method in ('scalar_fixed', 'scalar_hist_fixed', 'scalar_mobile'):
        log = record['selection']['methods'][method]['raw']
        history = log.get('history_losses') or {}
        for grid in log.get('grids', []):
            label = method + '/' + str(grid['grid_points'])
            try:
                table = grid['table']
                if not table.get('success'):
                    metrics['errors'].append(dict(check=label, reason='incomplete table cannot be fully audited'))
                    continue
                T = len(table['background'][0])
                pair_errors = []
                metrics['tables'] += 1
                metrics['new_table_QP_calls'] += table['QP_calls']
                for k, pair in enumerate(pairs):
                    c, B, G = (np.asarray(pair[key], float) for key in ('c', 'B', 'A0'))
                    lo, hi, center = (np.asarray(pair[key], float) for key in ('lower', 'upper', 'base_center'))
                    error = 0.
                    for t in range(T):
                        rows = np.r_[np.arange(t * n, (t + 1) * n), T * n + t, T * n + T + t]
                        columns = np.arange(2 * t, 2 * t + 2)
                        ct, Gt = c[rows], G[np.ix_(rows, columns)]
                        low, high, base = lo[columns], hi[columns], center[columns]
                        errors = [table['background'][k][t]['primal_dual_gap']]
                        for j, site in enumerate(table['tables'][k][t]):
                            b = B[rows, t * n + j]
                            params = site['curve_parameters']
                            for labelkey, expected in (('c', ct), ('b', b), ('G', Gt),
                                    ('source_lower', low), ('source_upper', high), ('source_center', base)):
                                close(params[labelkey], expected, 'saved scalar parameters: ' + labelkey)
                            for node in site['nodes']:
                                err = node_check(ct, b, Gt, low, high, base, node, params['power_range'])
                                metrics['endpoint_node_references'] += 1
                                metrics['failed_saved_QPs'] += not node['success']
                                metrics['max_QP_residual_error'] = max(metrics['max_QP_residual_error'], err)
                                metrics['max_endpoint_gap'] = max(metrics['max_endpoint_gap'], node['primal_dual_gap'])
                            curvature = float(b @ b)
                            spacing = float(np.max(np.diff(site['grid']), initial=0.))
                            gap = max(node['primal_dual_gap'] for node in site['nodes'])
                            bound = curvature * spacing ** 2 / 8 + gap
                            close(curvature, site['curvature_upper'], label + ': curvature')
                            close(spacing, site['spacing'], label + ': spacing')
                            close(gap, site['endpoint_gap'], label + ': endpoint gap')
                            close(bound, site['error_upper'], label + ': chord error')
                            close([q['primal_upper'] for q in site['nodes']], site['primal_upper'], 'endpoint upper table')
                            errors.append(bound)
                        error += max(errors)
                    pair_errors.append(error)
                close(pair_errors, table['pair_error_upper'], 'pair chord error sum')
                nearest_error = np.mean([max(value for (a, _), value in zip(pair_names, pair_errors) if a == name)
                                         for name in names])
                close(nearest_error, table['nearest_error_upper'], 'nearest chord error upper')
                raw = grid.get('raw')
                if raw is None:
                    continue
                metrics['raw_MILP_calls'] += 1
                if raw['operation']['feasible']:
                    metrics['physical_raw_actions'] += 1
                else:
                    metrics['raw_operation_failures'].append(dict(method=method, grid=grid['grid_points'],
                        status=raw['status'], issues=raw['operation'].get('issues'),
                        max_violation=raw['operation'].get('max_violation')))
                action = raw.get('action_QP')
                if action is None:
                    metrics['missing_intervals'] += 1
                    if raw.get('frozen_objective_lower') is not None:
                        raise ValueError('nonphysical action has reported lower certificate')
                    continue
                lower, upper = [], []
                for pair, qp in zip(pairs, action['pairs']):
                    c, B, G = (np.asarray(pair[key], float) for key in ('c', 'B', 'A0'))
                    lo, hi, base = (np.asarray(pair[key], float) for key in ('lower', 'upper', 'base_center'))
                    primal, dual, err = qp_check(c, G, lo, hi, base, B @ np.asarray(raw['power_kw']).ravel(), qp)
                    metrics['action_QP_calls'] += 1
                    metrics['failed_saved_QPs'] += not qp['success']
                    metrics['max_QP_residual_error'] = max(metrics['max_QP_residual_error'], err)
                    lower.append(.5 * dual)
                    upper.append(.5 * primal)
                lb = _aggregate(names, pair_names, lower, history)
                primal = _aggregate(names, pair_names, upper, history)
                close(lb, action['dual_lower'], 'all-pair nearest dual lower')
                close(primal, action['primal_upper'], 'all-pair nearest primal')
                close(lb, raw['frozen_objective_lower'], 'frozen action lower')
                ub = raw['frozen_objective_upper']
                if ub is None:
                    metrics['missing_intervals'] += 1
                    continue
                tolerance = 2e-7 + 2e-8 * max(1., abs(ub), abs(lb))
                if not lb <= primal + tolerance or not primal <= ub + tolerance:
                    raise ValueError('source primal/dual and frozen MILP upper interval')
                close(ub - lb, raw['frozen_absolute_gap'], 'reported frozen absolute gap')
                if raw['solver_proxy_incumbent'] > ub + tolerance:
                    raise ValueError('MILP incumbent exceeds best bound')
                metrics['available_intervals'] += 1
            except (ValueError, TypeError, KeyError, IndexError, FloatingPointError) as exc:
                metrics['errors'].append(dict(check=label, reason=type(exc).__name__ + ': ' + str(exc)))
    metrics.update(case_id=record['case_id'], seed=record['seed'], success=not metrics['errors'],
                   seconds=perf_counter() - began)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, inputs = [], []
    for path in sorted(args.directory.glob('case_*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            record = json.load(stream)
        rows.append(audit_record(record))
        inputs.append(dict(path=path.name, sha256=sha256(path.read_bytes()).hexdigest()))
    fields = ('tables', 'new_table_QP_calls', 'endpoint_node_references', 'action_QP_calls', 'failed_saved_QPs',
              'raw_MILP_calls', 'physical_raw_actions', 'available_intervals', 'missing_intervals')
    result = dict(success=bool(rows and all(row['success'] for row in rows)), case_slots=len(rows),
        **{key: sum(row[key] for row in rows) for key in fields}, cases=rows, inputs=inputs,
        independent_tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Feasible source primal/explicit dual/chord arithmetic; raw invalid actions retain no lower certificate. '
              'MILP best-bound trusted as a numerical solver bound; not a nonlinear AC certificate.')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({key: value for key, value in result.items() if key not in ('cases', 'inputs')}))


if __name__ == '__main__':
    main()
