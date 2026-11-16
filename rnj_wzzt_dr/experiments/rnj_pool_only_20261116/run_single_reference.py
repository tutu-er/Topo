"""Second bounded check: one fixed RNJ reference warm start; reuse sealed baseline."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from time import perf_counter

from ... import io
from .run import ROOT, check, fit_one, snapshot, utc


def prepare(out, baseline):
    out, baseline = Path(out).resolve(), Path(baseline).resolve()
    if out.exists():
        raise FileExistsError('Use a fresh output: '+str(out))
    completed = io.read(baseline/'complete.json')
    entries = io.read(baseline/'inputs.json')
    if completed['successful'] != 4 or len(entries) != 4:
        raise ValueError('The four baseline fits must be complete')
    previous = io.read(baseline/'protocol.json')
    check(previous['inputs_ref'])
    for entry in entries:
        for key in ('train.npz', 'validation.npz'):
            check(entry['files'][key])
        result = io.read(baseline/'cases'/entry['id']/'result.json')
        if result['status'] != 'complete':
            raise ValueError('Incomplete baseline: '+entry['id'])
        check(result['native'])
    out.mkdir(parents=True)
    run = out/'new'; run.mkdir()
    refs = snapshot(out/'new_source', 'new')
    io.write(run/'inputs.json', entries)
    protocol = dict(created_utc=utc(), version='new', iteration='B_single_reference',
        baseline=str(baseline), baseline_complete=io.ref(baseline/'complete.json'),
        source_root=str(out/'new_source'/'rnj_wzzt_dr'), sources=refs,
        inputs_ref=io.ref(run/'inputs.json'), inputs=4, planned_fits=4, method='101',
        method_seconds=120., per_mip_seconds=120., output_reserve_seconds=12.,
        qp_soft_seconds=300., workers=3, environment=io.environment(),
        no_new_AC=True, no_DR=True, no_E60=True,
        runner=io.ref(Path(__file__)), shared_runner=io.ref(Path(__file__).with_name('run.py')),
        source_document=io.ref(Path(__file__).with_name('PROTOCOL_B.md')),
        active_scope=io.ref(ROOT/'rnj_wzzt_dr/evaluation_scope.json'))
    for key in ('method', 'method_seconds', 'per_mip_seconds', 'output_reserve_seconds', 'qp_soft_seconds'):
        if protocol[key] != previous[key]:
            raise ValueError('Baseline budget differs: '+key)
    io.write(run/'protocol.json', protocol)
    io.write(out/'protocol.json', protocol)
    return protocol


def run(out):
    out = Path(out).resolve(); protocol = io.read(out/'protocol.json')
    if (out/'complete.json').exists():
        return io.read(out/'complete.json')
    lock = out/'running.lock'; fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    rows = []; started = perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            futures = {pool.submit(fit_one, str(out/'new'), entry):entry for entry in io.read(out/'new'/'inputs.json')}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = dict(input_id=entry['id'], version='new', status='worker_failed', error=repr(exc), seconds=None)
                rows.append(row)
                io.write(out/'progress.json', dict(completed=len(rows), planned=4, records=rows))
                print({k:row[k] for k in ('version', 'input_id', 'status')}, flush=True)
        result = dict(status='complete', planned=4, completed=len(rows),
            successful=sum(r['status']=='complete' for r in rows), failed=sum(r['status']!='complete' for r in rows),
            seconds=perf_counter()-started, records=rows)
        io.write(out/'complete.json', result)
        io.write(out/'new'/'complete.json', result)
        return result
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not a.resume:
        prepare(a.out, a.baseline)
    if not a.prepare_only:
        run(a.out)


if __name__ == '__main__':
    main()
