"""Refit the original identifier on a paired sampling/noise sensitivity grid.

Existing 384-row physical trajectories, normalization and resource contracts are
immutable. No AC is solved here. Each grid cell gets a new complete native bank;
truth and new-day scores are opened only after that bank has been sealed.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...observations import Observation, fingerprint
from ...history import fit_observed, freeze_history
from ...identification.algorithm import SearchConfig
from ..distribution_panel_20261115.data import additional_search, append_candidate
from ..flexibility_sweep_20261113.fitting import native_json_record, pack_score
from ..measured_pq_20261114 import physical, validate

ROOT = Path(__file__).resolve().parents[3]
STRIDES = (1, 2, 4)
METER_MULTIPLIERS = (.5, 1., 2.)
HISTORY_NOISE_SEED = 2026111503
CASES = ('pengwah18', 'flynn16', 'paper15', 'soumalas11')
REQUESTED = dict(pengwah18='2024-01-04', flynn16='2024-12-19',
                 paper15='2024-01-04', soumalas11='2024-01-04')


def utc():
    return datetime.now(timezone.utc).isoformat()


def checked(reference):
    path = Path(reference['path']).resolve()
    if io.ref(path)['sha256'] != reference['sha256']:
        raise RuntimeError('Changed sealed input: '+str(path))
    return path


def entry_refs(entry):
    refs = [entry['resource_context'], entry['resource_spec'],
            entry['physical_cache']['original'], entry['physical_cache']['cache']]
    refs += list(entry['files'].values())
    refs += [entry[k] for k in ('score', 'native_copy') if k in entry]
    return refs


def available(entry):
    """Availability only; never inspect candidate quality or numerical outcomes."""
    for reference in entry_refs(entry):
        checked(reference)
    if len(entry['p_resources']) != 7 or not entry['q_resources']:
        raise ValueError('Expected the original P and Q contract lists')
    return True


def choose_inputs(entries, availability=available):
    chosen, ledger = [], []
    for case in CASES:
        candidates = sorted((e for e in entries if e['case'] == case), key=lambda e: e['start_date'])
        wanted = case+'__'+REQUESTED[case]
        candidates.sort(key=lambda e: e['id'] != wanted)
        selected = None
        for entry in candidates:
            if case in ('pengwah18', 'flynn16') and entry['id'] != wanted:
                continue
            try:
                if not availability(entry):
                    raise ValueError('Input is not available')
                selected = deepcopy(entry)
                ledger.append(dict(case=case, id=entry['id'], status='selected',
                    preferred_id=wanted, fallback_used=entry['id'] != wanted))
                break
            except (FileNotFoundError, KeyError, ValueError) as exc:
                ledger.append(dict(case=case, id=entry['id'], status='unavailable', error=repr(exc)))
        if selected is None:
            raise ValueError('No permitted cached input for '+case+': '+repr(ledger))
        chosen.append(selected)
    return chosen, ledger


def sample_indices(stride):
    if stride not in STRIDES:
        raise ValueError('Only the locked strides 1, 2, 4 are allowed')
    return dict(train=np.arange(0, 192, stride), validation=np.arange(192, 288, stride),
                new_day=np.arange(288, 384))


def standard_noise(n, original_index, seed=HISTORY_NOISE_SEED):
    if n < 1 or original_index < 0:
        raise ValueError('Positive terminal count and nonnegative original index required')
    result = {}
    for name, component, shape in (('P', 2, (384, n)), ('Q', 3, (384, n)),
                                   ('V', 4, (384, n)), ('root', 5, (384,))):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2718, int(original_index), component]))
        result[name] = rng.normal(size=shape)
    return result


def observed_splits(clean, noise, stride, multiplier):
    if multiplier not in METER_MULTIPLIERS:
        raise ValueError('Only the locked meter multipliers .5, 1, 2 are allowed')
    labels = tuple(map(int, clean['labels'])); shape = (384, len(labels))
    for name in ('P', 'Q', 'V', 'root'):
        wanted = (384,) if name == 'root' else shape
        if np.shape(clean[name]) != wanted or np.shape(noise[name]) != wanted:
            raise ValueError('Wrong clean/noise shape for '+name)
        if not np.isfinite(clean[name]).all() or not np.isfinite(noise[name]).all():
            raise ValueError('Nonfinite clean/noise array')
    measured = {name: np.asarray(clean[name])*(1+multiplier*(.005 if name in ('P', 'Q') else .0005)*noise[name])
                for name in ('P', 'Q', 'V', 'root')}
    if np.any(measured['V'] <= 0) or np.any(measured['root'] <= 0):
        raise ValueError('Meter errors produced nonpositive voltage')
    y = measured['root'][:, None]**2-measured['V']**2
    return {name: Observation(measured['P'][idx], measured['Q'][idx], y[idx], labels,
                             measured['V'][idx], measured['root'][idx])
            for name, idx in sample_indices(stride).items()}


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def source_paths():
    paths = {Path(m.__file__).resolve() for m in list(sys.modules.values())
             if getattr(m, '__file__', None) and str(m.__file__).endswith('.py')
             and Path(m.__file__).resolve().is_relative_to(ROOT)}
    paths.add(Path(__file__).resolve())
    case_root = physical.ac_backend.ROOT/'cases'
    paths.update(case_root/case/name for case in CASES for name in ('metadata.json', 'buses.csv', 'branches.csv'))
    return sorted(paths)


def prepare(source_run, out, workers=4):
    source_run, out = Path(source_run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing grid preparation: '+str(out))
    if not 1 <= workers <= 4:
        raise ValueError('Use 1..4 single-thread workers')
    if shutil.disk_usage(out.parent).free < 5*2**30:
        raise RuntimeError('Less than 5 GiB free workspace space')
    document = Path(__file__).with_name('PROTOCOL.md')
    if not document.exists():
        raise FileNotFoundError('The grid PROTOCOL.md must be locked before preparation')
    old_protocol = io.read(source_run/'protocol.json')
    old_complete = io.read(source_run/'complete.json')
    if old_complete.get('status') != 'complete' or not old_complete.get('source_unchanged'):
        raise ValueError('A completed, historically unchanged source run is required')
    # Current research modules may acquire compatible parameter hooks. Validate
    # the executed old snapshot, not a claim that old live source is immutable.
    old_snapshot_refs = []
    for reference in old_protocol['sources']:
        path = source_run/'source_snapshot'/Path(reference['path']).resolve().relative_to(ROOT)
        if io.ref(path)['sha256'] != reference['sha256']:
            raise ValueError('Old executed source snapshot changed: '+str(path))
        old_snapshot_refs.append(io.ref(path))
    original_entries = io.read(checked(old_protocol['inputs']))
    if isinstance(original_entries, dict):
        original_entries = original_entries['inputs']
    chosen, selection = choose_inputs(original_entries)
    out.mkdir(parents=True)
    input_refs = [io.ref(source_run/'protocol.json'), io.ref(source_run/'complete.json'), old_protocol['inputs']]
    tasks = []
    for entry in chosen:
        clean = arrays(checked(entry['physical_cache']['original']))
        n = len(clean['labels'])
        if np.shape(clean['P']) != (384, n) or len(clean['timestamps']) != 384:
            raise ValueError('Expected a complete original 384-point cached physical window')
        original_index = int(entry.get('original_input_index', entry['index']))
        if original_index != int(entry['index']):
            raise ValueError('Original RNG index mismatch')
        directory = out/'base_inputs'/entry['id']; directory.mkdir(parents=True)
        noise = standard_noise(n, original_index)
        np.savez_compressed(directory/'standard_noise.npz', **noise)
        noise_ref = io.ref(directory/'standard_noise.npz')
        input_refs.extend(entry_refs(entry)); input_refs.append(noise_ref)
        for stride in STRIDES:
            for multiplier in METER_MULTIPLIERS:
                tag = str(multiplier).replace('.', 'p')
                identifier = entry['id']+f'__s{stride}__m{tag}'
                tasks.append(dict(id=identifier, grid_order=len(tasks), index=original_index,
                    original_input_index=original_index, original_id=entry['id'],
                    case=entry['case'], start_date=entry['start_date'], phase='known_world_grid',
                    stride=stride, meter_multiplier=multiplier,
                    training_points=192//stride, validation_points=96//stride, new_points=96,
                    training_rows=192//stride, validation_rows=96//stride,
                    training_hours=48, validation_hours=24, history_observation_spacing_minutes=15*stride,
                    forecast_relative_sd=.05, meter_PQ_relative_sd=.005*multiplier,
                    meter_V_relative_sd=.0005*multiplier, base_kw=float(entry['base_kw']),
                    meter=dict(additional_PQ_relative_sd=.005*multiplier,
                               additional_V_relative_sd=.0005*multiplier),
                    standard_noise=noise_ref, source_entry=deepcopy(entry)))
    paths = source_paths()
    sources = [io.ref(path) for path in paths]
    for path in paths:
        dest = out/'source_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    refs = list({(r['path'], r['sha256']): r for r in input_refs}.values())
    protocol = dict(created_utc=utc(), phase='known_world_sampling_and_noise_grid',
        source_document=io.ref(document), source_run=str(source_run), source_completed_run=input_refs[1],
        previous_source_snapshots=old_snapshot_refs, input_refs=refs, sources=sources,
        selected_inputs=[e['id'] for e in chosen], selection_ledger=selection,
        tasks=tasks, planned_inputs=36, workers=workers,
        strides=list(STRIDES), meter_multipliers=list(METER_MULTIPLIERS),
        historical_noise_seed=HISTORY_NOISE_SEED,
        noise_stream='SeedSequence([2026111503,2718,original_input_index,component]); components P2,Q3,V4,root5; same underlying 384-row Z across all cells',
        method='101', method_config=asdict(validate._config(30.)), method_budget_seconds=30.,
        additional_E_pool_search_seconds=60., new_day_identifier_inputs=False,
        baseline_scaling='Original full-history-derived physical normalization frozen, not recomputed from sampled observations',
        resources='Original source-run resource_spec/P/Q offers frozen across all cells; no reselection or Q95 recomputation',
        sampling='Take indices 0:192:stride and 192:288:stride; retain full 288:384 new day. Fewer samples, unchanged two-day/one-day observation windows.',
        no_AC_solve=True, root_physical_trajectory='Original cached root unchanged, no resampling',
        new_day_file='One fixed diagnostic observation at the new meter multiplier; execution runner draws separate paired meter/response streams',
        validation_scope='All frozen native candidates scored with this cell validation plus the full new day; every residual cell has equal weight',
        truth_rule='Original structure labels opened only after candidate seal; never injected into candidate pool or resource placement',
        failure_rule='All 36 cells retained, including missing candidates, no extra point, fit failures and time limits; no partial budget reset',
        interpretation='Sensitivity/development on already opened physical days, not new independent confirmation',
        environment=validate._environment(), **validate._git_receipt(paths))
    io.write(out/'protocol.json', protocol)
    return protocol


def verify_protocol(protocol):
    for reference in [protocol['source_document'], *protocol['sources'],
                      *protocol['previous_source_snapshots'], *protocol['input_refs']]:
        checked(reference)


def fit_cell(task, out):
    out = Path(out); protocol = io.read(out/'protocol.json')
    directory = out/'grid'/task['id']
    if (directory/'complete.json').exists():
        return io.read(directory/'complete.json')
    if directory.exists():
        raise RuntimeError('Partial grid cell requires explicit recovery; do not reset fit budgets')
    directory.mkdir(parents=True); began = perf_counter(); stage = 'observations'
    io.write(directory/'started.json', dict(task=task, pid=os.getpid(), created_utc=utc()))
    try:
        prior = task['source_entry']
        clean = arrays(checked(prior['physical_cache']['original']))
        noise = arrays(checked(task['standard_noise']))
        splits = observed_splits(clean, noise, task['stride'], task['meter_multiplier'])
        refs = {}
        for name, obs in splits.items():
            path = directory/(name+'.npz')
            np.savez_compressed(path, P=obs.p, Q=obs.q, Y=obs.y, V=obs.voltage,
                                root=obs.root, labels=obs.labels)
            refs[name] = dict(ref=io.ref(path), sha256=fingerprint(obs))
        sampling = {name: dict(indices=idx.tolist(), points=len(idx),
            timestamps=[str(clean['timestamps'][i]) for i in idx])
            for name, idx in sample_indices(task['stride']).items()}
        io.write(directory/'observations_sealed.json', dict(created_utc=utc(), observations=refs,
            physical_source=prior['physical_cache']['original'], standard_noise=task['standard_noise'],
            multiplier=task['meter_multiplier'], sampling=sampling, underlying_noise_shared_across_grid=True))
        stage = 'fit101'; tick = perf_counter()
        original = native_json_record(fit_observed(splits['train'], splits['validation'], method='101',
                                      config=SearchConfig(**protocol['method_config'])))
        original_seconds = perf_counter()-tick
        io.write(directory/'original_101_native.json', original)
        old_frozen = freeze_history(original, splits['validation'])
        stage = 'additional_E60'
        search = additional_search(original, splits['train'], protocol['additional_E_pool_search_seconds'])
        io.write(directory/'search_60s.json', native_json_record(search))
        stage = 'freeze_native_bank'
        enriched = append_candidate(original, search.get('closed_point'), splits['train'], splits['validation'])
        enriched['candidate_enrichment'] = dict(original_native=io.ref(directory/'original_101_native.json'),
            source_search=io.ref(directory/'search_60s.json'), original_points_preserved=len(original['frozen_candidates']),
            added_points=int(search.get('closed_point') is not None), original_fit_actual_seconds=original_seconds,
            extra_search_status=search['status'], extra_search_actual_seconds=search.get('total_seconds'),
            original_runtime_fields_scope='Original 30-second fit only; extra E search accounted separately',
            truth_used=False, new_day_values_used=False)
        io.write(directory/'101_native.json', native_json_record(enriched))
        score = pack_score(enriched, splits['train'], splits['validation'])
        io.write(directory/'score.json', score)
        frozen = freeze_history(enriched, splits['validation'])
        io.write(directory/'candidates_sealed.json', dict(created_utc=utc(), native=io.ref(directory/'101_native.json'),
            original_native=io.ref(directory/'original_101_native.json'), score=io.ref(directory/'score.json'),
            training=refs['train'], validation=refs['validation'], parameter_sha256=frozen.parameter_sha256,
            original_parameter_sha256=old_frozen.parameter_sha256, truth_used=False, new_day_values_used=False))
        stage = 'offline_qualification'
        truth = physical.offline_truth(task['case'], splits['train'].labels)
        evaluations = {}
        for name, native, bank, seconds in (
            ('original', original, old_frozen, original_seconds),
            ('enriched', enriched, frozen, original_seconds+(search.get('total_seconds') or 0.))):
            evaluations[name] = validate._evaluate_identifier(dict(status='fitted', result=native,
                frozen=bank, seconds=seconds), splits, truth)
        metadata = {k:v for k,v in task.items() if k != 'source_entry'}
        row = dict(**metadata, status='complete', seconds=perf_counter()-began, methods=evaluations,
            original_fit_actual_seconds=original_seconds, original_fit_budget_seconds=30.,
            extra_search=io.ref(directory/'search_60s.json'), extra_search_status=search['status'],
            candidate_enrichment=enriched['candidate_enrichment'],
            pq={name:validate.pq_diagnostics(obs) for name,obs in splits.items()},
            physical_cache=prior['physical_cache'], no_new_AC=True,
            offline_truth_after_candidate_seal=True, offline_truth_QP_run=False)
        io.write(directory/'result.json', row)
        files = {name:io.ref(directory/name) for name in ('train.npz','validation.npz','new_day.npz',
            '101_native.json','original_101_native.json','observations_sealed.json','candidates_sealed.json','result.json')}
        entry = dict(**metadata, files=files, score=io.ref(directory/'score.json'), native_copy=files['101_native.json'],
            resource_context=prior['resource_context'], physical_cache=prior['physical_cache'],
            resource_spec=prior['resource_spec'], p_resources=deepcopy(prior['p_resources']),
            q_resources=deepcopy(prior['q_resources']), fixed_resources_from_original_id=prior['id'],
            fixed_resource_source_score=prior['score'],
            historical_native_index=enriched['selected']['index'], historical_p_map=score['p_map'],
            map_matches_native_topology=score['map_matches_native_topology'])
        completion = dict(status='complete', task=metadata, entry=entry,
                          result=io.ref(directory/'result.json'), seconds=perf_counter()-began)
    except Exception as exc:
        row = dict(id=task['id'], status='failed', stage=stage, seconds=perf_counter()-began,
                   error=repr(exc), traceback=traceback.format_exc())
        io.write(directory/'result.json', row)
        completion = dict(status='failed', task={k:v for k,v in task.items() if k!='source_entry'},
                          result=io.ref(directory/'result.json'), stage=stage, error=row['error'])
    io.write(directory/'complete.json', completion)
    return completion


def run(out, workers=None):
    out = Path(out).resolve(); protocol = io.read(out/'protocol.json')
    if workers is not None and workers != protocol['workers']:
        raise ValueError('Worker count differs from prepared protocol')
    verify_protocol(protocol)
    if (out/'complete.json').exists():
        return io.read(out/'complete.json')
    lock = out/'lock.json'
    with lock.open('x', encoding='utf-8') as f:
        f.write('{"pid":'+str(os.getpid())+',"created_utc":"'+utc()+'"}')
    receipts = []; began = perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            futures = {pool.submit(fit_cell, task, str(out)): task for task in protocol['tasks']}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    receipt = future.result()
                except Exception as exc:
                    receipt = dict(status='worker_failed', task={k:v for k,v in task.items() if k!='source_entry'},
                                   error=repr(exc), traceback=traceback.format_exc())
                receipts.append(receipt)
                io.write(out/'progress.json', dict(planned_inputs=36, finished=len(receipts), receipts=receipts))
                print(f'{len(receipts)}/36 {task["id"]} {receipt["status"]}', flush=True)
        receipts.sort(key=lambda r:r['task']['grid_order'])
        entries = [r['entry'] for r in receipts if r['status']=='complete']
        failures = [r for r in receipts if r['status']!='complete']
        changed = []
        try:
            verify_protocol(protocol)
        except Exception as exc:
            changed.append(repr(exc))
        io.write(out/'grid_index.json', dict(phase='known_world_grid', inputs=entries, input_count=len(entries),
            planned_inputs=36, attempts=receipts, failures=failures, protocol=io.ref(out/'protocol.json'),
            fixed_original_inputs=protocol['selected_inputs'], stride_grid=list(STRIDES),
            meter_multiplier_grid=list(METER_MULTIPLIERS), physical_worlds=4,
            no_new_AC=True, resources_fixed=True, source_unchanged=not changed))
        result = dict(status='complete' if not changed else 'source_changed', planned_inputs=36,
            prepared_inputs=len(entries), failed_inputs=len(failures), failures=failures,
            input_index=io.ref(out/'grid_index.json'), seconds=perf_counter()-began,
            source_unchanged=not changed, changed_sources_or_inputs=changed, no_new_AC=True)
        io.write(out/'complete.json', result)
        return result
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument('--prepare', action='store_true')
    phase.add_argument('--run', action='store_true')
    parser.add_argument('--source-run', type=Path, default=ROOT/'outputs/distribution_panel_confirmation_20261115_v1')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/flexibility_grid_data_20261115_v1')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    result = prepare(args.source_run, args.out, args.workers) if args.prepare else run(args.out, args.workers)
    print({k:result[k] for k in ('planned_inputs','prepared_inputs','failed_inputs','source_unchanged') if k in result})


if __name__ == '__main__':
    main()
