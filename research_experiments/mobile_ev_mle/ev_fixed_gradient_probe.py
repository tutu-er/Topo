"""Training-only fixed-line AC/I² response and gradient development audit.

This is a function-query experiment, not a selector or an independent policy
evaluation. Only four existing development histories are reused. Actual future
loads, truth, validation and future errors never enter the response problem.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from . import gradient_price_study as io
from .likelihood import FitResult, Observations, RadialCandidate
from .information_context import meter_from_record
from .ev_dynamic_loads import fit_forecast
from .ev_fixed_gradient import profile_pair, nearest_summary
from .ev_fixed_response import FixedResponse

METHODS = ('ac', 'linear', 'i2', 'anchored_i2')
INPUT_KEYS = ('training', 'training_meter', 'future_meter', 'candidates',
              'training_result', 'config')


def training_view(record):
    """Explicit boundary; unknown record fields cannot reach the experiment."""
    view = {k:record[k] for k in INPUT_KEYS if k != 'training_result'}
    view['training_result'] = {'fits':record['training_result']['fits']}
    # Keep only public horizon/gap/profile/source options from configuration.
    view['config'] = {k:record['config'][k] for k in
                      ('horizon', 'forecast_gap', 'profile', 'source')}
    return view


def restore(view):
    tr = Observations(**view['training'])
    tm, fm = (meter_from_record(view[k]) for k in ('training_meter', 'future_meter'))
    bank = []
    for row in view['candidates']:
        family = [tuple(s) for s in row['supports'] if 1 < len(s) < tr.n]
        candidate = RadialCandidate.from_supports(row['name'], tr.n, family)
        if candidate.supports != tuple(tuple(s) for s in row['supports']):
            raise ValueError('candidate edge order changed')
        bank.append(candidate)
    fits = {}
    for row in view['training_result']['fits']:
        data = {f.name:row[f.name] for f in fields(FitResult)}
        for k in ('r', 'x', 'p', 'q', 'source_p', 'source_q', 'bias_p', 'bias_q', 'bias_v'):
            data[k] = np.asarray(data[k], float)
        data['predicted'] = {k:np.asarray(v) for k, v in data['predicted'].items()}
        fits[row['candidate']] = FitResult(**data)
    cfg = view['config']
    forecast = fit_forecast(tr, tm, fits, horizon=cfg['horizon'], gap=cfg['forecast_gap'],
                            base_kw=20., kind='ar1')
    T = cfg['horizon']
    sigma = np.column_stack((np.broadcast_to(fm.v_sigma, (T, tr.n)),
        np.broadcast_to(fm.master_p_sigma, (T,)), np.broadcast_to(fm.master_q_sigma, (T,))))
    return bank, fits, forecast, sigma


def public_queries(T, n):
    if T != 6:
        raise ValueError('the declared development queries use six slots')
    constant = np.zeros((T, n)); constant[:, 0] = 2.
    highlow = np.zeros((T, n)); remain = 3.
    pmax, dt, eta = 6./1.05, .25, .95
    for t in (0, 2, 4, 1, 3, 5):
        highlow[t, 0] = min(pmax, remain/dt)
        remain -= dt*highlow[t, 0]
    signed = np.zeros((T, n)); signed[:3, 0] = pmax
    C = 3*pmax*dt
    signed[3, 0] = -eta*(eta*C-2.85)/dt
    return dict(none=np.zeros_like(constant), constant=constant, highlow=highlow,
                declared_v2g=signed)


def error_summary(estimate, reference):
    a, b = np.asarray(estimate), np.asarray(reference)
    error = a-b
    return dict(RMSE=float(np.sqrt(np.mean(error**2))),
        max_abs=float(np.max(abs(error), initial=0.)),
        relative_L2=float(np.linalg.norm(error)/max(np.linalg.norm(b), 1e-12)))


def audit_history(view, *, task_id):
    bank, fits, forecast, sigma = restore(view)
    p, q, root = (np.asarray(forecast[k], float) for k in ('p', 'q', 'root'))
    T, n = p.shape
    sources = {c.name:np.column_stack([forecast['source_by_candidate'][c.name][k]
               for k in ('source_p', 'source_q')]) for c in bank}
    queries = public_queries(T, n)
    operators, setup, line_bytes = {}, {}, {}
    for method in METHODS:
        operators[method] = {}
        began = perf_counter()
        for c in bank:
            fit, s = fits[c.name], sources[c.name]
            anchor = (p+queries['constant']/20., q, s[:, 0], s[:, 1], root)
            operators[method][c.name] = FixedResponse(c, fit.r, fit.x,
                method=method, anchor=anchor if method == 'anchored_i2' else None)
            line_bytes[c.name] = fit.r.tobytes()+fit.x.tobytes()
        setup[method] = perf_counter()-began
    response_rows, kernel_times = [], {m:[] for m in METHODS}
    for label, u in queries.items():
        for c in bank:
            s = sources[c.name]
            outputs = {}
            for method in METHODS:
                op = operators[method][c.name]
                try:
                    outputs[method] = op.evaluate(p+u/20., q, s[:, 0], s[:, 1], root)
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    outputs[method] = dict(success=False, failure=type(exc).__name__+':'+str(exc))
                    continue
                # Distinct nearby inputs avoid reporting exact-AC cache hits
                # as a response-kernel speedup. These queries do not select u.
                for repeat in range(3):
                    began = perf_counter()
                    op.evaluate(p+u/20.+(repeat+1)*1e-8, q, s[:, 0], s[:, 1], root)
                    kernel_times[method].append(perf_counter()-began)
            exact = outputs['ac']
            for method in METHODS[1:]:
                out = outputs[method]
                if 'failure' in out or 'failure' in exact:
                    response_rows.append(dict(query=label, candidate=c.name, method=method,
                        success=False, failure=dict(estimate=out.get('failure'), reference=exact.get('failure'))))
                    continue
                response_rows.append(dict(query=label, candidate=c.name, method=method,
                    success=True,
                    voltage=error_summary(out['values'][:, :n], exact['values'][:, :n]),
                    root_PQ=error_summary(out['values'][:, n:], exact['values'][:, n:]),
                    voltage_PQ_source_J=error_summary(out['jac'][:, :n], exact['jac'][:, :n])))
    rows, fits_cache = [], {}
    options = dict(view['config']['profile'])
    options['tol'] = 1e-11
    source_config = view['config']['source']
    bounds = np.stack((np.tile([source_config['source_p_bounds'][0], source_config['source_q_bounds'][0]], (T, 1)),
                       np.tile([source_config['source_p_bounds'][1], source_config['source_q_bounds'][1]], (T, 1))))

    def profile(method, a, b, power):
        return profile_pair(operators[method][a.name], operators[method][b.name],
            p, q, sources[a.name], sources[b.name], root, power, sigma,
            source_bounds=bounds, seed=task_id+500, **options)

    for label, u in queries.items():
        for a in bank:
            for b in bank:
                if a.name == b.name:
                    continue
                for method in METHODS:
                    result = profile(method, a, b, u)
                    result.update(method=method, a=a.name, b=b.name, query=label)
                    rows.append(result)
                    fits_cache[method, label, a.name, b.name] = result
    errors = []
    for label in queries:
        for a in bank:
            for b in bank:
                if a.name == b.name:
                    continue
                exact = fits_cache['ac', label, a.name, b.name]
                for method in METHODS[1:]:
                    out = fits_cache[method, label, a.name, b.name]
                    row = dict(query=label, a=a.name, b=b.name, method=method,
                               success=bool(out['success'] and exact['success']))
                    if row['success']:
                        row.update(distance_abs_error=abs(out['half_SSE']-exact['half_SSE']),
                            distance_relative_error=abs(out['half_SSE']-exact['half_SSE'])/max(exact['half_SSE'], 1e-12),
                            gradient=error_summary(out['gradient_per_kw'], exact['gradient_per_kw']))
                    errors.append(row)
    fd_rows = []
    a, b = bank[:2]
    direction = np.cos(np.arange(T*n)+.5).reshape(T, n)
    direction /= np.linalg.norm(direction)
    eps = .005
    for label in ('constant', 'declared_v2g'):
        for method in METHODS:
            out = fits_cache[method, label, a.name, b.name]
            high = profile(method, a, b, queries[label]+eps*direction)
            low = profile(method, a, b, queries[label]-eps*direction)
            row = dict(query=label, method=method, a=a.name, b=b.name,
                success=bool(out['success'] and high['success'] and low['success']))
            if row['success']:
                fd = (high['half_SSE']-low['half_SSE'])/(2*eps)
                analytic = float(np.sum(out['gradient_per_kw']*direction))
                row.update(analytic=analytic, finite_difference=fd, absolute_error=abs(analytic-fd),
                    relative_error=abs(analytic-fd)/max(abs(fd), 1e-9),
                    max_source_kkt_violation=max(r['source_kkt_violation'] for r in (out, high, low)),
                    active_masks_equal=bool(np.array_equal(out['active_mask'], high['active_mask'])
                        and np.array_equal(out['active_mask'], low['active_mask'])),
                    derivative_check_pass=bool(abs(analytic-fd) <= 1e-5+2e-4*abs(fd)))
            fd_rows.append(row)
    theta_unchanged = all(fits[c.name].r.tobytes()+fits[c.name].x.tobytes() == line_bytes[c.name]
                          for c in bank)
    scores = {method:{label:nearest_summary([r for r in rows if r['method'] == method and r['query'] == label])
              for label in queries} for method in METHODS}
    return dict(status='complete', task_id=task_id, forecast=forecast, setup_seconds=setup,
        kernel_seconds={m:dict(calls=len(v), median=float(np.median(v)) if v else None, total=float(np.sum(v)))
                        for m, v in kernel_times.items()},
        profile_seconds={m:sum(r['seconds'] for r in rows if r['method'] == m) for m in METHODS},
        pair_profiles=rows, pair_errors=errors, response_errors=response_rows,
        finite_differences=fd_rows, scores=scores,
        counters={m:{name:op.counters.copy() for name, op in ops.items()} for m, ops in operators.items()},
        theta_unchanged=theta_unchanged,
        input_sha256=hashlib.sha256(str(io.serial(view)).encode()).hexdigest(),
        uses_truth=False, uses_validation=False, uses_future_noise=False,
        scope='training-only response/gradient diagnostic; no action selection or topology evaluation')


def summarize(results):
    complete = [r for r in results if r.get('status') == 'complete']
    out = dict(histories=len(results), histories_complete=len(complete),
        driver_failures=[dict(task_id=r['task_id'], failure=r['failure']) for r in results if r.get('status') != 'complete'],
        theta_unchanged=all(r['theta_unchanged'] for r in complete) if len(complete) == len(results) else None, methods={})
    for method in METHODS:
        rows = [r for h in complete for r in h['pair_profiles'] if r['method'] == method]
        fd = [r for h in complete for r in h['finite_differences'] if r['method'] == method]
        kernels = [h['kernel_seconds'][method]['median'] for h in complete
                   if h['kernel_seconds'][method]['median'] is not None]
        item = dict(profile_expected=48*len(results), profile_total=len(rows), profile_success=sum(r['success'] for r in rows),
            profile_seconds=sum(h['profile_seconds'][method] for h in complete),
            setup_seconds=sum(h['setup_seconds'][method] for h in complete),
            kernel_median_seconds=float(np.median(kernels)) if kernels else None,
            finite_difference_success=sum(r['success'] for r in fd), finite_difference_total=len(fd),
            finite_difference_derivative_pass=sum(r.get('derivative_check_pass', False) for r in fd),
            finite_difference_same_active_mask=sum(r.get('active_masks_equal', False) for r in fd),
            finite_difference_max_relative_error=max((r['relative_error'] for r in fd if r['success']), default=None),
            max_source_kkt_violation=max((r['source_kkt_violation'] for r in rows if r['success']), default=None),
            max_scaled_source_kkt_violation=max((r['scaled_source_kkt_violation'] for r in rows if r['success']), default=None))
        if method != 'ac':
            err = [r for h in complete for r in h['pair_errors'] if r['method'] == method and r['success']]
            response = [r for h in complete for r in h['response_errors'] if r['method'] == method and r['success']]
            item.update(distance_relative_error_median=float(np.median([r['distance_relative_error'] for r in err])) if err else None,
                distance_relative_error_max=max((r['distance_relative_error'] for r in err), default=None),
                gradient_relative_L2_median=float(np.median([r['gradient']['relative_L2'] for r in err])) if err else None,
                gradient_relative_L2_max=max((r['gradient']['relative_L2'] for r in err), default=None),
                voltage_max_abs_error=max((r['voltage']['max_abs'] for r in response), default=None),
                root_PQ_max_abs_error=max((r['root_PQ']['max_abs'] for r in response), default=None))
        out['methods'][method] = item
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).parent
    from .ev_rx_study import source_hashes as dependency_hashes
    code = dependency_hashes()
    code.update({name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in
            ('ev_fixed_response.py', 'ev_fixed_gradient.py', 'ev_fixed_gradient_probe.py', 'ev_fixed_gradient_math.md')}
    )
    manifest = dict(version='fixed_response_gradient_development_v3_stationary_ties', seeds=list(range(992101, 992105)),
        methods=METHODS, input_keys=INPUT_KEYS, source_sha256=code,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        multistart=2, tolerance=1e-11, fd_epsilon_kw=.005,
        source_restart_tie='minimum projected gradient within 1e-11*(1+minimum_half_SSE)',
        scope='four reused development histories; response approximation only; frozen policy unchanged')
    io.write_json(args.output/'manifest.json', manifest)
    results = []
    for seed in manifest['seeds']:
        path = args.input/f'trial_{seed}.json.gz'
        record = io.read_json(path)
        try:
            result = audit_history(training_view(record), task_id=seed)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            result = dict(status='driver_failure', task_id=seed, failure=type(exc).__name__+':'+str(exc))
        result['provenance'] = dict(input_raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_policy_id=record['policy_id'], original_source_sha256=record['source_sha256'],
            verified_research_base='c3d3d16', new_source_sha256=code)
        io.write_json(args.output/f'trial_{seed}.json.gz', result, compressed=True)
        results.append(result)
        print(seed, result['status'], 'profiles', sum(r['success'] for r in result.get('pair_profiles', [])),
              '/', len(result.get('pair_profiles', [])), flush=True)
    report = summarize(results)
    io.write_json(args.output/'summary.json', report)
    print(io.serial(report), flush=True)


if __name__ == '__main__':
    main()
