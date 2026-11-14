"""Run the prelocked grid through the existing eight-policy AC pipeline."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from ... import io
from ..distribution_panel_20261115 import run as panel
from ..distribution_panel_20261115.continuation import initialize
from . import data


def smoke_readiness(index, smoke_audit):
    """Require an audited integration run of this input and current sources."""
    if smoke_audit is None:
        raise ValueError('Formal grid requires --smoke-audit from the completed integration run')
    audit_path = Path(smoke_audit).resolve()
    report = io.read(audit_path)
    if (report.get('status') != 'passed' or report.get('expected_rows') != 8
            or report.get('audited_rows') != 8
            or report.get('audit_status_counts') != {'passed': 8}
            or report.get('task_errors') != []
            or report.get('original_failed_tasks') != []
            or report.get('original_execution_failed_rows') != 0
            or report.get('original_quote_exceptions') != 0):
        raise ValueError('Smoke audit must pass all eight rows without task, quote or execution errors')
    protocol_ref = report['source_protocol']
    protocol_path = panel.old.checked(protocol_ref)
    protocol = io.read(protocol_path)
    if (protocol.get('smoke') is not True or protocol.get('forecasts') != 1
            or protocol.get('repeats') != 1 or protocol.get('expected_rows') != 8
            or tuple(protocol.get('policies', ())) != panel.POLICIES):
        raise ValueError('Smoke source must be the one-forecast, one-repeat eight-policy integration run')
    if protocol.get('input_index') != io.ref(Path(index).resolve()):
        raise ValueError('Smoke audit used a different grid input index')
    if report.get('source_inputs') != protocol.get('inputs'):
        raise ValueError('Smoke audit input reference differs from the source protocol')
    panel.old.checked(protocol['inputs'])
    complete = io.read(protocol_path.parent/'complete.json')
    if complete.get('status') != 'complete' or complete.get('source_unchanged') is not True:
        raise ValueError('Smoke source is incomplete or reports changed sources')
    sources = protocol.get('sources', [])
    if not sources:
        raise ValueError('Smoke source manifest is empty')
    for reference in sources:
        panel.old.checked(reference)
    return dict(status='passed', audit=io.ref(audit_path), source_protocol=protocol_ref,
        source_completion=io.ref(protocol_path.parent/'complete.json'),
        grid_input=protocol['input_index'], verified_current_sources=len(sources),
        scope='Eight audited integration rows; same grid index and unchanged current runtime sources')


def prepare(index, out, *, smoke=False, smoke_audit=None):
    index, out = Path(index).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Keep previous grid output: '+str(out))
    readiness = None if smoke else smoke_readiness(index, smoke_audit)
    source = io.read(index)
    if source.get('source_unchanged') is not True:
        raise ValueError('Unchanged completed grid preparation is required')
    prepared = io.read(index.parent/'complete.json')
    if prepared.get('status') != 'complete' or prepared.get('input_index') != io.ref(index):
        raise ValueError('Grid preparation completion and index do not match')
    entries = deepcopy(source['inputs'])
    if not entries:
        raise ValueError('No prepared grid inputs')
    # Non-default history length AND noise must traverse the integration test.
    if smoke:
        selected = [e for e in entries if e['validation_rows'] == 24
                    and np.isclose(e['meter']['additional_V_relative_sd'], .001)]
        if not selected:
            raise ValueError('Smoke requires stride4, meter2 input')
        entries = selected[:1]
    p = dict(created_utc=panel.old.utc(), phase='grid_development', smoke=smoke,
        seed=2026111500 if smoke else 2026111504, forecasts=1, repeats=1 if smoke else 16,
        solver_seconds=2. if smoke else 600., anchor_seconds=2. if smoke else 150.,
        shared_seed_seconds=.5 if smoke else 2., workers=1 if smoke else 4,
        target_mass=.99, modes=list(panel.MODES), policies=list(panel.POLICIES),
        forecast_relative_sd=.05, reservation_prices=dict(P=.5,Q=.5),
        input_index=io.ref(index), source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        environment=io.environment(),
        driver_relative_path=Path(panel.__file__).resolve().relative_to(panel.old.ROOT).as_posix(),
        entry_driver_relative_path=Path(__file__).resolve().relative_to(panel.old.ROOT).as_posix(),
        validation_phase='Known physical days; new history and execution random streams',
        physical_and_resource_rule='Original physical cache, scaling and registered offers held fixed across grid',
        meter='Explicit per-entry P/Q and V/root standard deviations; same multiplier in history and execution',
        scoring='Per-entry validation rows plus96 new rows; unweighted residual cells; frozen H and sigma2',
        cost='Activation objective; requested capacity plus delivered activation settlement',
        independent_unit='Four fixed network/day worlds; repeated grid and execution rows are paired, not independent days',
        failure='Failed quotes execute independently validated anchor else zero; retain all statuses',
        action_checkpoint='Atomic per-return checkpoint before next method, shared original algorithms',
        smoke_audit=None if readiness is None else readiness['audit'], smoke_readiness=readiness)
    out.mkdir(parents=True)
    failures, ready = [], []
    for entry in entries:
        directory = out/'histories'/entry['id']
        directory.mkdir(parents=True)
        try:
            for reference in (entry['score'],entry['native_copy'],entry['resource_spec'],
                    entry['resource_context'],entry['files']['train.npz'],entry['files']['validation.npz'],
                    entry['physical_cache']['original'],entry['physical_cache']['cache']):
                panel.old.checked(reference)
            frozen_resource = io.read(entry['resource_spec']['path'])
            if frozen_resource['status'] != 'ready':
                raise ValueError('Frozen resource source is not ready')
            if (entry['p_resources'] != frozen_resource['p_resources']
                    or entry['q_resources'] != frozen_resource['q_resources']):
                raise ValueError('Grid resource list differs from original offer')
            entry.update(phase='grid_development', date_cluster=entry['start_date'],
                resource_variant='frozen_original_history_offers', physical_contract_fraction=.1,
                forecast_relative_sd=p['forecast_relative_sd'])
            with np.load(entry['physical_cache']['original']['path'],allow_pickle=False) as z:
                power = np.c_[z['P'][288:384],z['Q'][288:384]]
            prediction = power*(1+p['forecast_relative_sd']*panel.old.rng(
                p['seed'],entry['index'],0,991).normal(size=power.shape))
            np.savez_compressed(directory/'forecasts.npz',forecast00=prediction)
            entry['forecast_ref'] = io.ref(directory/'forecasts.npz')
            io.write(directory/'forecast_receipt.json',dict(seed=p['seed'],index=entry['index'],
                forecast_relative_sd=p['forecast_relative_sd'],forecast=entry['forecast_ref'],
                physical_PQ_source=entry['physical_cache']['original'],
                new_day_voltage_used_by_planner=False,truth_network_used_by_planner=False,
                scope='Evaluation-side P/Q-to-forecast adapter; optimizer receives forecast only'))
            ready.append(entry)
        except Exception as exc:
            failure=dict(id=entry['id'],case=entry['case'],start_date=entry['start_date'],
                stage='resource_or_forecast_preparation',status='failed',error=repr(exc),traceback=traceback.format_exc())
            failures.append(failure);io.write(directory/'preparation_failure.json',failure)
    ledger = source['attempts'] if not smoke else [dict(status='complete',task=entries[0])]
    io.write(out/'qualification_ledger.json',ledger)
    io.write(out/'inputs.json',ready)
    io.write(out/'preparation_failures.json',failures)
    p.update(inputs=io.ref(out/'inputs.json'),qualification_ledger=io.ref(out/'qualification_ledger.json'),
        preparation_failures=io.ref(out/'preparation_failures.json'),original_histories=len(entries),
        runnable_histories=len(ready),planned_histories=len(ledger),
        expected_rows=len(ready)*p['repeats']*len(panel.POLICIES),
        planned_rows=len(ledger)*p['repeats']*len(panel.POLICIES))
    p['sources']=panel.snapshot(out,ready)
    path=Path(__file__).with_name('PROTOCOL.md').resolve()
    target=out/'source_snapshot'/path.relative_to(panel.old.ROOT)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    p['sources'].append(io.ref(path))
    p.update(panel.old._git_receipt([Path(r['path']) for r in p['sources']]))
    io.write(out/'protocol.json',p)
    return p


def run(out):
    out=Path(out).resolve();p=io.read(out/'protocol.json')
    for ref in [*p['sources'],p['inputs'],p['qualification_ledger'],p['preparation_failures']]:
        panel.old.checked(ref)
    if (out/'complete.json').exists():
        return io.read(out/'complete_summary.json')
    if (out/'lock.json').exists():
        raise RuntimeError('Inspect process and unfinished budgets before resuming')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=panel.old.utc()))
    entries=io.read(out/'inputs.json');receipts=[];failures=[]
    try:
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=initialize,initargs=(str(out),)) as pool:
            jobs={pool.submit(panel.run_forecast,(e,0)):e for e in entries}
            for future in as_completed(jobs):
                e=jobs[future]
                try:receipts.append(future.result())
                except Exception as exc:
                    failures.append(dict(id=e['id'],case=e['case'],start_date=e['start_date'],
                        forecast_index=0,status='task_failed',error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(total=len(entries),completed=len(receipts),failed_tasks=failures,receipts=receipts))
                print(f"{len(receipts)+len(failures)}/{len(entries)} grid forecasts; {len(failures)} failed",flush=True)
        result=dict(total_forecasts=len(entries),completed_forecasts=len(receipts),failed_tasks=failures,
            preparation_failed_histories=len(io.read(out/'preparation_failures.json')),
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),
            execution_failed_rows=sum(r['failed_rows'] for r in receipts),
            feasible_quotes_by_mode={m:sum(r['new_gate_feasible'][m] for r in receipts) for m in panel.MODES},
            anchor_fallbacks=sum(r['anchor_fallbacks'] for r in receipts),
            zero_fallbacks=sum(r['zero_fallbacks'] for r in receipts),receipts=receipts)
        io.write(out/'complete_summary.json',result)
        io.write(out/'complete.json',dict(status='complete',main=io.ref(out/'complete_summary.json'),
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources'])))
        return result
    finally:
        (out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-audit',type=Path,
        help='Passed eight-row smoke audit report.json; required for formal preparation')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not args.resume:
        if args.index is None:parser.error('--index is required')
        prepare(args.index,args.out,smoke=args.smoke,smoke_audit=args.smoke_audit)
    elif bool(io.read(args.out/'protocol.json')['smoke'])!=args.smoke:
        raise ValueError('Resume smoke flag differs from sealed run')
    if not args.prepare_only:
        result=run(args.out);print({k:v for k,v in result.items() if k!='receipts'})


if __name__=='__main__':main()
