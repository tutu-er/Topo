"""Replay frozen candidate banks after training-only impedance-floor pruning.

No structure search, new AC, or procurement is performed. Truth and existing
new-day observations are opened only after every candidate bank is sealed.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...dr.selection_mse import choose_mse
from ...identification.optimization import QPFitter, errors, internal_count
from ...observations import fingerprint


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / 'data/impedance_floor_20261116_v1'
RELATIVE_GRID = (0., .01, .02, .04, .08, .12, .16, .20, .24, .32)
OHM_GRID = (0., .0005, .001, .002, .003, .005, .008, .01, .015, .02)
SCOPES = ('all', 'non_singleton')
CURRENT_GROUPS = ('history288', 'history10')


def utc():
    return datetime.now(timezone.utc).isoformat()


def checked(reference):
    path = Path(reference['path'])
    if io.ref(path)['sha256'] != reference['sha256']:
        raise ValueError('Sealed file changed: ' + str(path))
    return path


def grid_for(entry):
    metrics = [('rx75', RELATIVE_GRID)]
    if entry['source_group'] in CURRENT_GROUPS:
        metrics.append(('ohm', OHM_GRID))
    return [dict(id=f'{metric}__{scope}__f{str(value).replace(".", "p")}',
                 metric=metric, scope=scope, threshold=value)
            for metric, values in metrics for scope in SCOPES for value in values]


def score_bank(candidates, validation, reference_mse, validation_tie, *, new_day=None):
    """Use the frozen scale/tie rule and every source candidate in original order."""
    if not candidates:
        raise ValueError('Cannot select from an empty candidate bank')
    ids = [r['source_candidate_id'] for r in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError('Source candidate IDs must remain unique')
    if new_day is not None and tuple(validation.labels) != tuple(new_day.labels):
        raise ValueError('Historical and new-day observation labels differ')
    cells = validation.y.size + (0 if new_day is None else new_day.y.size)
    loss_rows = []
    for row in candidates:
        point = row['point']
        historical = errors(validation, point)
        new = None if new_day is None else errors(new_day, point)
        sse = historical['mse'] * validation.y.size
        if new is not None:
            sse += new['mse'] * new_day.y.size
        loss_rows.append(dict(source_candidate_id=row['source_candidate_id'],
            validation_mse=historical['mse'], validation_mae=historical['mae'],
            new_day_mse=None if new is None else new['mse'],
            new_day_mae=None if new is None else new['mae'],
            scored_mse=sse / cells, internal_count=internal_count(point, validation.n)))
    selection = choose_mse([r['scored_mse'] for r in loss_rows],
        [r['internal_count'] for r in loss_rows], K=1. / reference_mse,
        candidate_ids=ids, relative_band=0., epsilon_scaled=validation_tie)
    selection.update(scoring_cells=cells, validation_rows=validation.points,
        new_day_rows=0 if new_day is None else new_day.points,
        training_rescored=False, parameter_refitted_during_evaluation=False)
    return dict(selection=selection, losses=loss_rows)


def topology(point, truth_supports, n, zero_tol):
    active = {tuple(sorted(C)) for C, r, x in
              zip(point['supports'], point['r'], point['x'], strict=True)
              if max(r, x) > zero_tol}
    truth = {tuple(sorted(C)) for C in truth_supports}
    internal = lambda family: {C for C in family if 1 < len(C) < n}
    a, t = internal(active), internal(truth)
    return dict(full_exact=active == truth, full_missing=len(truth-active),
        full_extra=len(active-truth), internal_exact=a == t,
        internal_missing=len(t-a), internal_extra=len(a-t),
        active_support_count=len(active), common_active=tuple(range(n)) in active,
        active_singletons=sum(len(C) == 1 for C in active))


def evaluate_bank(plan, validation, reference_mse, validation_tie, zero_tol, evaluation):
    if plan['status'] != 'complete':
        return dict(status='unavailable_failed_candidate_bank', reason=plan.get('error'))
    candidates = plan['candidates']
    truth = evaluation['truth_supports']
    by_id = {r['source_candidate_id']: r for r in candidates}
    supports = {i: topology(r['point'], truth, validation.n, zero_tol) for i, r in by_id.items()}
    historical = plan['historical_scoring']
    result = dict(status='complete', candidate_count=len(candidates),
        bank_has_internal_truth=any(v['internal_exact'] for v in supports.values()),
        bank_has_full_truth=any(v['full_exact'] for v in supports.values()),
        candidate_topologies=[dict(source_candidate_id=i, **v) for i, v in supports.items()],
        historical=historical,
        history_selected_topology=supports[historical['selection']['selected']],
        independent_new_day_available=evaluation['new_day'] is not None)
    if evaluation['new_day'] is not None:
        cumulative = score_bank(candidates, validation, reference_mse, validation_tie,
                                new_day=evaluation['new_day'])
        result.update(cumulative=cumulative,
            cumulative_selected_topology=supports[cumulative['selection']['selected']])
    else:
        result.update(cumulative=None, cumulative_selected_topology=None,
            cumulative_unavailable_reason='Source has historical training/validation only; no independent new-day file')
    return result


def prune_bank(candidates, fitter, setting, calibration, zbase_ohm, *, prune_function):
    """A failed candidate invalidates that grid cell; never silently remove rivals."""
    rows, failures = [], []
    started = perf_counter()
    for source in candidates:
        row = dict(source_candidate_id=source['source_candidate_id'])
        before = len(fitter.records)
        began = perf_counter()
        try:
            point, receipt = prune_function(source['point'], fitter, setting['threshold'],
                metric=setting['metric'], scope=setting['scope'], calibration=calibration,
                zbase_ohm=zbase_ohm)
            row.update(status='complete', point=point, pruning=receipt)
        except Exception as exc:
            row.update(status='failed', error=type(exc).__name__ + ': ' + str(exc),
                traceback=traceback.format_exc(), pruning=getattr(exc, 'receipt', None),
                last_point=getattr(exc, 'last_point', None))
            failures.append(row['source_candidate_id'])
        row.update(seconds=perf_counter()-began,
                   qp_record_indices=list(range(before, len(fitter.records))))
        rows.append(row)
    return dict(setting=setting, status='failed' if failures else 'complete',
        failed_candidate_ids=failures, candidates=rows, seconds=perf_counter()-started,
        error=None if not failures else 'One or more source candidates failed; selection withheld for the entire grid cell')


def grid_bank(candidates, fitter, setting, calibration, calibration_error, zbase_ohm, *, prune_function):
    if calibration_error is not None and setting['metric'] == 'rx75' and setting['threshold'] > 0:
        return dict(setting=setting, status='failed',
            failed_candidate_ids=[c['source_candidate_id'] for c in candidates],
            candidates=[dict(source_candidate_id=c['source_candidate_id'], status='not_run',
                             error='Training calibration unavailable') for c in candidates],
            error='Training calibration unavailable for positive relative floor',
            calibration_failure=calibration_error, seconds=0.)
    return prune_bank(candidates, fitter, setting, calibration, zbase_ohm,
                      prune_function=prune_function)


def source_paths():
    # Import actual runtime dependencies before taking the snapshot. Later
    # independent audit/report files deliberately do not belong to this closure.
    from . import pruning, sources  # noqa: F401
    paths = {Path(__file__).with_name('PROTOCOL.md'), Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.suffix == '.py' and p.is_relative_to(ROOT) and 'outputs' not in p.parts:
                if not p.name.startswith('test_'):
                    paths.add(p)
    return sorted(paths)


def prepare(out, *, workers=3):
    from . import sources
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('New output required: ' + str(out))
    if not 1 <= workers <= 3:
        raise ValueError('Use one to three single-thread workers')
    paths = source_paths()
    if any(not p.exists() for p in paths):
        raise FileNotFoundError('Runtime or locked PROTOCOL.md is not ready')
    entries = sources.inventory(root=ROOT)
    ids = [r['id'] for r in entries]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate input IDs')
    out.mkdir(parents=True)
    refs = []
    for path in paths:
        destination = out / 'source_snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        refs.append(io.ref(path))
    protocol = dict(created_utc=utc(), source_document=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        sources=refs, workers=workers, planned_inputs=len(entries),
        relative_grid=list(RELATIVE_GRID), physical_ohm_grid=list(OHM_GRID), scopes=list(SCOPES),
        physical_grid_groups=list(CURRENT_GROUPS),
        fixed_support_qp_soft_seconds=300.,
        planned_grid_cells=sum(len(grid_for(e)) for e in entries),
        threshold_definition='pruning.py; each grid starts from each original frozen candidate',
        fitting='Original training only; same reference MSE and coefficient bounds; shared support-family QP cache per input',
        selection='All original candidate IDs/order retained; strict validation MSE with original frozen scale and numerical ties',
        failed_candidate_policy='Retain failed trace and withhold the whole grid cell; do not drop failed competitors',
        evaluation='Truth and existing new-day arrays opened after all input candidate banks are sealed',
        interpretation='Already used histories; development replay, not a new independent experiment',
        no_new_structure_search=True, no_new_AC=True, no_procurement=True,
        environment=io.environment())
    io.write(out / 'inputs.json', entries)
    protocol['inputs'] = io.ref(out / 'inputs.json')
    io.write(out / 'protocol.json', protocol)
    io.write(out / 'prepared.json', dict(status='prepared', protocol=io.ref(out / 'protocol.json')))
    return protocol


def verify_sources(protocol):
    for reference in [*protocol['sources'], protocol['inputs']]:
        checked(reference)


def run_input(out, entry):
    from . import pruning, sources
    out = Path(out)
    directory = out / 'inputs' / entry['id']
    if (directory / 'complete.json').exists():
        complete = io.read(directory / 'complete.json')
        if complete['input_id'] != entry['id']:
            raise ValueError('Saved task identity differs')
        checked(complete['result'])
        return complete
    if directory.exists():
        raise RuntimeError('Partial input retained; automatic budget reset is disabled: ' + str(directory))
    directory.mkdir(parents=True)
    io.write(directory / 'started.json', dict(input_id=entry['id'], pid=os.getpid(), created_utc=utc()))
    began = perf_counter()
    stage = 'load_training'
    try:
        protocol = io.read(out / 'protocol.json'); verify_sources(protocol)
        data = sources.load_training(entry)
        for reference in data['source_refs']:
            checked(reference)
        native, train, validation = data['native'], data['train'], data['validation']
        if train.labels != validation.labels:
            raise ValueError('Training/validation labels changed')
        candidates = [dict(source_candidate_id=r['index'], point=r['point'])
                      for r in native['frozen_candidates']]
        reference_mse, tie = data['reference_mse'], data['validation_tie']
        baseline = dict(status='complete', setting=dict(id='original_frozen', metric=None,
            scope=None, threshold=None), candidates=candidates,
            historical_scoring=score_bank(candidates, validation, reference_mse, tie))
        selected = baseline['historical_scoring']['selection']['selected']
        baseline['saved_selected_index'] = data['expected_selected_index']
        baseline['saved_selection_reproduced'] = selected == data['expected_selected_index']
        if not baseline['saved_selection_reproduced']:
            raise ValueError(f'Original frozen baseline selection not reproduced: {selected} != {data["expected_selected_index"]}')
        io.write(directory / 'baseline.json', baseline)
        stage = 'training_calibration'
        cstart = perf_counter()
        calibration, calibration_error = None, None
        try:
            calibration = pruning.calibrate(train)
        except Exception as exc:
            calibration_error = dict(error=type(exc).__name__ + ': ' + str(exc),
                                     traceback=traceback.format_exc())
        calibration_seconds = perf_counter()-cstart
        io.write(directory / 'calibration.json', dict(calibration=calibration,
            status='complete' if calibration_error is None else 'failed',
            failure=calibration_error,
            seconds=calibration_seconds, training_sha256=fingerprint(train),
            validation_used=False, truth_used=False, new_day_used=False))
        fitter = QPFitter(train, reference_mse=reference_mse, bound=data['bound'],
            qp_seconds=protocol['fixed_support_qp_soft_seconds'])
        plans = []
        stage = 'floor_grid'
        for setting in grid_for(entry):
            plan = grid_bank(candidates, fitter, setting, calibration, calibration_error,
                             data['zbase_ohm'], prune_function=pruning.prune)
            if plan['status'] == 'complete':
                plan['historical_scoring'] = score_bank(plan['candidates'], validation, reference_mse, tie)
            plans.append(plan)
            io.write(directory / 'progress.json', dict(input_id=entry['id'],
                completed_grid_cells=len(plans), planned_grid_cells=len(grid_for(entry)),
                last_setting=setting, failed_grid_cells=sum(p['status'] != 'complete' for p in plans),
                elapsed_seconds=perf_counter()-began, unique_QP_fits=len(fitter.records)))
            io.write(directory / 'banks_partial.json', plans)
            io.write(directory / 'qp_records.json', fitter.records)
            io.write(directory / 'qp_cache_points.json',
                [dict(family=[list(C) for C in family], point=point)
                 for family, point in fitter.cache.items()])
        stage = 'seal'
        io.write(directory / 'original_native.json', native)
        io.write(directory / 'banks.json', dict(original_native=io.ref(directory / 'original_native.json'),
            final_bank_schema='candidates retain source_candidate_id and complete point; selected ID in historical_scoring.selection',
            original_structure_search_bounds_apply_to_original_search_only=True,
            new_global_structure_optimality_claim=False, baseline=baseline, grid=plans))
        seal = dict(input_id=entry['id'], created_utc=utc(), banks=io.ref(directory / 'banks.json'),
            calibration=io.ref(directory / 'calibration.json'), qp_records=io.ref(directory / 'qp_records.json'),
            qp_cache_points=io.ref(directory / 'qp_cache_points.json'),
            original_source_refs=data['source_refs'], source_protocol=io.ref(out / 'protocol.json'),
            training_sha256=fingerprint(train), validation_sha256=fingerprint(validation),
            reference_mse=reference_mse, validation_tie=tie, zero_tol=data['zero_tol'],
            coefficient_bound=data['bound'], zbase_ohm=data['zbase_ohm'],
            truth_used=False, new_day_used=False, calibration_seconds=calibration_seconds,
            preparation_and_grid_seconds=perf_counter()-began)
        io.write(directory / 'candidates_sealed.json', seal)
        # No evaluation loader or true supports appear above this boundary.
        stage = 'evaluation'
        evaluation = sources.load_evaluation(entry)
        for reference in evaluation['source_refs']:
            checked(reference)
        result = dict(status='complete', input_id=entry['id'], source_group=entry['source_group'],
            case=entry['case'], input=entry, seal=io.ref(directory / 'candidates_sealed.json'),
            evaluation_source_refs=evaluation['source_refs'],
            baseline=evaluate_bank(baseline, validation, reference_mse, tie, data['zero_tol'], evaluation),
            grid=[dict(setting=p['setting'], planning_status=p['status'],
                       **evaluate_bank(p, validation, reference_mse, tie, data['zero_tol'], evaluation)) for p in plans],
            planned_grid_cells=len(plans), successful_grid_cells=sum(p['status'] == 'complete' for p in plans),
            failed_grid_cells=sum(p['status'] != 'complete' for p in plans),
            unique_QP_fits=len(fitter.records), cache_families=len(fitter.cache),
            calibration_seconds=calibration_seconds, seconds=perf_counter()-began,
            truth_used_for_pruning=False, source_unchanged=True)
        verify_sources(protocol)
    except Exception as exc:
        result = dict(status='failed', input_id=entry['id'], source_group=entry['source_group'],
            case=entry['case'], stage=stage, error=type(exc).__name__ + ': ' + str(exc),
            traceback=traceback.format_exc(), seconds=perf_counter()-began,
            partial_artifacts_preserved=True, automatic_retry=False)
    io.write(directory / 'result.json', result)
    complete = dict(status=result['status'], input_id=entry['id'], result=io.ref(directory / 'result.json'),
        seconds=result['seconds'], created_utc=utc(), planned_grid_cells=len(grid_for(entry)),
        successful_grid_cells=result.get('successful_grid_cells', 0),
        failed_grid_cells=len(grid_for(entry))-result.get('successful_grid_cells', 0))
    io.write(directory / 'complete.json', complete)
    return complete


def run(out):
    out = Path(out).resolve()
    protocol = io.read(out / 'protocol.json'); verify_sources(protocol)
    if (out / 'complete.json').exists():
        return io.read(out / 'complete.json')
    entries = io.read(checked(protocol['inputs']))
    lock = out / 'running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps(dict(pid=os.getpid(), created_utc=utc())).encode()); os.close(fd)
    started = perf_counter(); rows = []
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            futures = {pool.submit(run_input, str(out), entry): entry for entry in entries}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = dict(status='worker_failed', input_id=entry['id'],
                        error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc(),
                        partial_input_preserved=True, planned_grid_cells=len(grid_for(entry)),
                        successful_grid_cells=0, failed_grid_cells=len(grid_for(entry)))
                rows.append(record)
                io.write(out / 'progress.json', dict(planned_inputs=len(entries), completed_inputs=len(rows),
                    failed_inputs=sum(r['status'] != 'complete' for r in rows), records=rows,
                    seconds=perf_counter()-started))
                print(json.dumps(dict(completed=len(rows), planned=len(entries), id=entry['id'],
                                      status=record['status'])), flush=True)
        verify_sources(protocol)
        complete = dict(status='complete', planned_inputs=len(entries), completed_inputs=len(rows),
            successful_inputs=sum(r['status'] == 'complete' for r in rows),
            failed_inputs=sum(r['status'] != 'complete' for r in rows), records=rows,
            planned_grid_cells=protocol['planned_grid_cells'],
            successful_grid_cells=sum(r['successful_grid_cells'] for r in rows),
            failed_grid_cells=sum(r['failed_grid_cells'] for r in rows),
            seconds=perf_counter()-started, source_unchanged=True,
            protocol=io.ref(out / 'protocol.json'), created_utc=utc())
        io.write(out / 'complete.json', complete)
        return complete
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare_only and args.resume:
        parser.error('--prepare-only and --resume are mutually exclusive')
    if not args.resume:
        prepare(args.out, workers=args.workers)
    if not args.prepare_only:
        run(args.out)


if __name__ == '__main__':
    main()
