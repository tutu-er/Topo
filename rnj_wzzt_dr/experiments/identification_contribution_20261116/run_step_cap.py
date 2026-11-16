"""Eight paired 100 fits with a 10-second MIP cap and unchanged 120-second total."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback

from ... import io
from ..rnj_pool_only_20261116 import run as shared

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / 'outputs/identification_contribution_20261116_v1'


def verify(protocol):
    """Verify sealed runtime and driver bytes without loading evaluation arrays."""
    for reference in (protocol['source_protocol'], protocol['source_complete'],
                      protocol['source_inputs'], protocol['inputs_ref']):
        shared.check(reference)
    for reference in protocol['sources']:
        path = shared.check(reference).resolve()
        if not path.is_relative_to(Path(protocol['source_root']).resolve()):
            raise ValueError('Runtime reference lies outside the sealed source root')
    for item in protocol['driver_sources']:
        shared.check(item['original'])
        shared.check(item['snapshot'])
        if item['original']['sha256'] != item['snapshot']['sha256']:
            raise ValueError('Driver snapshot differs from its source')
    for entry in io.read(shared.check(protocol['inputs_ref'])):
        for key in ('train.npz', 'validation.npz'):
            shared.check(entry['files'][key])
    return True


def prepare(out, source_run=DEFAULT_SOURCE):
    out, source_run = Path(out).resolve(), Path(source_run).resolve()
    if out.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(out))
    source = io.read(source_run / 'protocol.json')
    complete = io.read(source_run / 'complete.json')
    if (complete.get('status'), complete.get('planned'), complete.get('completed'),
            complete.get('successful'), complete.get('failed')) != ('complete', 32, 32, 32, 0):
        raise ValueError('The original 32-fit comparison must be complete')
    expected = dict(total_seconds=120., per_mip_seconds=120.,
                    output_reserve_seconds=12., qp_seconds=300., workers=3)
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError('The original scheduling configuration changed')
    if source.get('methods') != ['RNJ1', 'RNJ15', '100', '101']:
        raise ValueError('Unexpected original methods')
    inputs_path = shared.check(source['inputs_ref'])
    entries = io.read(inputs_path)
    expected_ids = {f'history10__{case}__d08__h{repeat:02d}'
                    for case in ('soumalas11', 'pengwah18') for repeat in range(4)}
    if len(entries) != 8 or {entry['id'] for entry in entries} != expected_ids:
        raise ValueError('The eight predeclared input identities changed')
    original_tasks = {(r['input_id'], r['method']) for r in complete['records']}
    if original_tasks != {(name, method) for name in expected_ids for method in source['methods']}:
        raise ValueError('Original completed task ledger is incomplete or changed')
    for reference in source['sources']:
        shared.check(reference)
    if io.ref(Path(shared.__file__))['sha256'] != source['shared_runner']['sha256']:
        raise ValueError('Shared fit executor differs from the original comparison')
    out.mkdir(parents=True)
    shutil.copyfile(inputs_path, out / 'inputs.json')
    driver_sources = []
    for path in (Path(__file__), Path(__file__).with_name('PROTOCOL_CAP.md'),
                 Path(shared.__file__), Path(io.__file__),
                 ROOT / 'rnj_wzzt_dr/evaluation_scope.py'):
        target = out / 'driver_source_snapshot' / path.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        driver_sources.append(dict(original=io.ref(path), snapshot=io.ref(target)))
    protocol = dict(
        created_utc=shared.utc(), phase='development_scheduling_diagnostic', version='new',
        method='100', inputs=8, planned_fits=8, workers=3,
        method_seconds=120., per_mip_seconds=10., qp_soft_seconds=300.,
        output_reserve_seconds=12., only_algorithm_setting_change='per_mip_seconds: 120 -> 10',
        source_root=source['source_root'], sources=source['sources'],
        source_protocol=io.ref(source_run / 'protocol.json'),
        source_complete=io.ref(source_run / 'complete.json'), source_inputs=source['inputs_ref'],
        inputs_ref=io.ref(out / 'inputs.json'), driver_sources=driver_sources,
        baseline_records=complete['records'], environment=io.environment(),
        task_ids=[entry['id'] for entry in entries],
        common_preparation='once in each 100 fit; charged inside the same total budget',
        no_AC_generation=True, no_DR=True, no_E60=True, truth_in_fitting=False,
        evaluation_arrays_loaded=False, baseline_RNJ_refits=False,
        completed_or_failed_tasks_retried=False, partial_budget_reset_allowed=False)
    verify(protocol)
    io.write(out / 'protocol.json', protocol)
    io.write(out / 'prepared.json', dict(created_utc=shared.utc(),
        protocol=io.ref(out / 'protocol.json'), inputs=io.ref(out / 'inputs.json'),
        sources_unchanged=True, planned=8, fitting_started=False))
    return protocol


def _retained_records(out, entries, protocol_ref):
    records, pending = [], []
    for entry in entries:
        directory = out / 'cases' / entry['id']
        if not directory.exists():
            pending.append(entry)
            continue
        result_path = directory / 'result.json'
        if not result_path.exists():
            raise RuntimeError('Partial task retained; refusing a new budget: ' + str(directory))
        result = io.read(result_path)
        if result.get('input_id') != entry['id'] or result.get('status') not in ('complete', 'failed'):
            raise ValueError('Unrecognized retained result: ' + str(result_path))
        if result['status'] == 'complete' and result.get('source_protocol') != protocol_ref:
            raise ValueError('Retained result belongs to a different protocol')
        records.append(dict(input_id=entry['id'], version='new', status=result['status'],
                            result=io.ref(result_path), seconds=result['seconds'], reused=True))
    return records, pending


def run(out):
    out = Path(out).resolve()
    prepared = io.read(out / 'prepared.json')
    protocol = io.read(shared.check(prepared['protocol']))
    verify(protocol)
    if (out / 'complete.json').exists():
        return io.read(out / 'complete.json')
    entries = io.read(shared.check(protocol['inputs_ref']))
    records, pending = _retained_records(out, entries, prepared['protocol'])
    lock = out / 'running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            futures = {pool.submit(shared.fit_one, str(out), entry): entry for entry in pending}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = dict(input_id=entry['id'], version='new', status='worker_failed',
                               error=repr(exc), traceback=traceback.format_exc(), seconds=None)
                records.append(row)
                io.write(out / 'progress.json', dict(planned=8, completed=len(records), records=records))
                print({key: row[key] for key in ('input_id', 'status')}, flush=True)
        try:
            unchanged, source_error = verify(protocol), None
        except Exception as exc:
            unchanged, source_error = False, repr(exc)
        records.sort(key=lambda row: row['input_id'])
        result = dict(status='complete' if unchanged else 'invalid_source_changed', planned=8,
            completed=len(records), successful=sum(row['status'] == 'complete' for row in records),
            failed=sum(row['status'] != 'complete' for row in records), records=records,
            seconds_this_invocation=perf_counter()-started, sources_unchanged=unchanged,
            source_verification_error=source_error, protocol=prepared['protocol'],
            no_new_AC=True, no_DR=True, truth_in_fitting=False, independent_confirmation=False)
        io.write(out / 'complete.json', result)
        return result
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.resume:
        prepare(args.out, args.source_run)
    if not args.prepare_only:
        run(args.out)


if __name__ == '__main__':
    main()
