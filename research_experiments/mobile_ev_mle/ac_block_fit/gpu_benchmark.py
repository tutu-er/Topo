"""Small CPU/GPU FP64 arithmetic comparison; no planning or scientific sample.

Run in an isolated process with single-thread BLAS. Synthetic fixtures only
check equations and arithmetic. Cold startup, transfers, synchronized kernels,
and combined calls are reported separately. Three warm repeats are the default.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
import traceback
import numpy as np

from .. import likelihood as lk
from ..reference_benchmark.derivatives import fit_response_derivatives
from .gpu import GPUWorkspace, configure_dependency_path


def fixture(n=32, t=192, *, zero_edges=True, seed=20401018):
    """Balanced laminar tree, ordinary varying P/Q, exact zero meter edges."""
    supports = []
    def split(ids):
        if len(ids) > 1:
            supports.append(tuple(ids))
            middle = len(ids)//2
            split(ids[:middle]); split(ids[middle:])
    split(list(range(n)))
    candidate = lk.RadialCandidate.from_supports('arithmetic_fixture', n, supports)
    rng = np.random.default_rng(seed+n+t)
    r = rng.uniform(.001, .006, candidate.m)
    x = r*rng.uniform(.15, .35, candidate.m)
    if zero_edges:
        r[list(candidate.terminal_edges)] = 0.
        x[list(candidate.terminal_edges)] = 0.
    phase = np.arange(t)[:, None]*2*np.pi/96
    p = rng.uniform(.008, .02, n)[None, :]*(1+.18*np.sin(phase+rng.uniform(0, 2, n)))
    q = p*(.23+.04*np.cos(phase))
    sp = -.02*(1+.3*np.sin(np.arange(t)*2*np.pi/96))
    sq = .2*sp
    return candidate, dict(r=r, x=x, p=p, q=q, source_p=sp, source_q=sq,
                           root_v=1+.002*np.sin(np.arange(t)*2*np.pi/96))


def compare(left, right, keys, *, atol=2e-10, rtol=2e-10):
    rows = {}
    for key in keys:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        rows[key] = dict(max_abs=float(np.max(abs(a-b))),
                         relative_norm=float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-300)),
                         passed=bool(np.allclose(a, b, atol=atol, rtol=rtol)))
    return dict(passed=all(row['passed'] for row in rows.values()), channels=rows,
                atol=atol, rtol=rtol)


def elapsed(call, *, synchronize=None, repeats=3):
    durations = []; result = None
    for _ in range(repeats):
        if synchronize: synchronize()
        start = perf_counter(); result = call()
        if synchronize: synchronize()
        durations.append(perf_counter()-start)
    return result, dict(seconds=durations, median_seconds=statistics.median(durations))


def paired_elapsed(calls, synchronize, repeats=3):
    """Alternate CPU/GPU order under the ongoing scientific CPU workload."""
    rows = {key: [] for key in calls}; order = []
    for repeat in range(repeats):
        keys = list(calls)
        if repeat % 2: keys.reverse()
        order.append(keys)
        for key in keys:
            if key.startswith('gpu'): synchronize()
            start = perf_counter(); calls[key]()
            if key.startswith('gpu'): synchronize()
            rows[key].append(perf_counter()-start)
    return {key: dict(seconds=values, median_seconds=statistics.median(values))
            for key, values in rows.items()}, order


def one_case(n, t, repeats=3, *, zero_edges=True):
    candidate, data = fixture(n, t, zero_edges=zero_edges)
    # CPU is the existing implementation, with no model changes.
    cpu = lk.ac_predict(candidate, **data)
    cpu_d = fit_response_derivatives(candidate, cpu, include_lines=True)
    start = perf_counter(); work = GPUWorkspace(candidate); dev = work.inputs(**data)
    work.synchronize(); upload_first = perf_counter()-start
    start = perf_counter(); physical = work.predict(dev); derivatives = work.derivatives(physical)
    work.synchronize(); first_kernel = perf_counter()-start
    out = work.host(physical); out_d = work.host(derivatives)
    checks = dict(physical=compare(cpu, out, ('voltage', 'v', 'master_p', 'master_q', 'branch_current', 'zbus')),
                  derivatives=compare(cpu_d, out_d, ('v', 'master_p', 'master_q')),
                  iterations_equal=cpu['iterations'] == out['iterations'])
    cpu_fixed = fit_response_derivatives(candidate, cpu, include_lines=False)
    gpu_fixed = work.host(work.derivatives(physical, include_lines=False))
    checks['source_only'] = compare(cpu_fixed, gpu_fixed, ('v', 'master_p', 'master_q'))
    def cpu_combined():
        result = lk.ac_predict(candidate, **data)
        return result, fit_response_derivatives(candidate, result, include_lines=True)
    _, h2d = elapsed(lambda: work.inputs(**data), synchronize=work.synchronize, repeats=repeats)
    _, d2h = elapsed(lambda: (work.host(physical), work.host(derivatives)), synchronize=work.synchronize, repeats=repeats)
    def gpu_resident():
        result = work.predict(dev)
        return result, work.derivatives(result)
    def gpu_end_to_end():
        inputs = work.inputs(**data); result = work.predict(inputs)
        return work.host(result), work.host(work.derivatives(result))
    paired, order = paired_elapsed(dict(
        cpu_forward=lambda: lk.ac_predict(candidate, **data),
        gpu_forward=lambda: work.predict(dev),
        cpu_derivative=lambda: fit_response_derivatives(candidate, cpu, include_lines=True),
        gpu_derivative=lambda: work.derivatives(physical),
        cpu_combined=cpu_combined, gpu_resident=gpu_resident, gpu_end_to_end=gpu_end_to_end),
        work.synchronize, repeats=repeats)
    cpu_total, gpu_total, gpu_e2e = (paired[k] for k in ('cpu_combined', 'gpu_resident', 'gpu_end_to_end'))
    passed = all(row.get('passed', True) for row in checks.values() if isinstance(row, dict))
    return dict(n=n, t=t, m=candidate.m, exact_zero_edges=int(np.count_nonzero(data['r'] == 0)),
        arithmetic_only=True, scientific_conditions_created=0, selector_called=False,
        dtype='float64/complex128', cpu_iterations=cpu['iterations'], gpu_iterations=out['iterations'],
        first_upload_seconds=upload_first, first_forward_and_derivatives_seconds=first_kernel,
        checks=checks, passed=passed, timing_order=order,
        cpu=dict(forward=paired['cpu_forward'], derivative=paired['cpu_derivative'], combined=cpu_total),
        gpu=dict(upload=h2d, forward=paired['gpu_forward'], derivative=paired['gpu_derivative'], download=d2h,
                 resident_combined=gpu_total, end_to_end=gpu_e2e),
        speedup_resident=cpu_total['median_seconds']/gpu_total['median_seconds'],
        speedup_end_to_end=cpu_total['median_seconds']/gpu_e2e['median_seconds'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--dependency-path', type=Path,
        help='Optional isolated CuPy directory, added after original NumPy/SciPy imports.')
    args = parser.parse_args(); begin = perf_counter()
    if args.dependency_path:
        # The isolated pip target also contains a NumPy wheel. Keep the same
        # already-imported original CPU NumPy/BLAS for a fair comparison.
        configure_dependency_path(args.dependency_path)
    record = dict(scope='FP64 AC and selected-RHS derivative arithmetic, not a full optimizer benchmark',
        source_policy_unchanged=True, original_frozen_batch_untouched=True, repetitions=args.repeats,
        CPU_contention='Original scientific workers remain running; these timings include contention.',
        full_fit_speedup_measured=False, device_resident_arrays=True)
    try:
        cold = perf_counter(); import cupy as cp
        record['import_seconds'] = perf_counter()-cold
        cold = perf_counter(); cp.cuda.Device(0).use(); cp.zeros(1, dtype=cp.float64)
        cp.cuda.get_current_stream().synchronize()
        record['context_first_allocation_seconds'] = perf_counter()-cold
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props['name']; name = name.decode() if isinstance(name, bytes) else str(name)
        record['environment'] = dict(python=sys.version, numpy=np.__version__, numpy_path=np.__file__, cupy=cp.__version__,
            scipy=importlib.metadata.version('scipy'), gpu=name,
            driver_version=cp.cuda.runtime.driverGetVersion(), runtime_version=cp.cuda.runtime.runtimeGetVersion(),
            total_gpu_memory=int(props['totalGlobalMem']), compute_capability=[props['major'], props['minor']])
        record['cases'] = [one_case(n, t, args.repeats) for n, t in ((3, 12), (32, 12), (32, 192))]
        record['all_passed'] = all(row['passed'] for row in record['cases'])
        record['status'] = 'complete' if record['all_passed'] else 'arithmetic_mismatch'
    except Exception as exc:
        record.update(status='failed', all_passed=False, error=type(exc).__name__+': '+str(exc),
                      traceback=traceback.format_exc())
    record['total_seconds'] = perf_counter()-begin
    record['source_sha256'] = {p.name: sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__), Path(__file__).with_name('gpu.py'))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=record['status'], output=str(args.output), total_seconds=record['total_seconds'])))
    return 0 if record.get('all_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
