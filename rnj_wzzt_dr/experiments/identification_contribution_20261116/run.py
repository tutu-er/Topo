"""Sealed, matched identification baselines; no truth or new-day reads in fitting."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from time import perf_counter
import traceback

from ... import io
from ...evaluation_scope import load_catalog
from ..rnj_pool_only_20261116.run import ROOT, check, package_from_snapshot, snapshot, utc

METHODS = ('RNJ1', 'RNJ15', '100', '101')


def prepare(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a new output: '+str(out))
    selected = [e for e in load_catalog() if e['source_group']=='history10' and e.get('train_days')==8]
    if len(selected)!=8 or {(e['case'], e['history_repeat']) for e in selected} != {
            (case, repeat) for case in ('soumalas11', 'pengwah18') for repeat in range(4)}:
        raise ValueError('The predeclared eight inputs are not available')
    entries = [dict(id=e['id'], case=e['case'], history_repeat=e['history_repeat'],
        labels=e['labels'], files={'train.npz':e['train_ref'], 'validation.npz':e['validation_ref'], 'new_day.npz':e['new_day_ref']},
        Hstar_ref=e['Hstar_ref'], evaluation_case_refs=e['evaluation_case_refs']) for e in selected]
    out.mkdir(parents=True)
    sources = snapshot(out/'source_snapshot', 'new')
    io.write(out/'inputs.json', entries)
    protocol = dict(created_utc=utc(), inputs=8, planned_fits=32, methods=list(METHODS),
        total_seconds=120., per_mip_seconds=120., output_reserve_seconds=12., qp_seconds=300., workers=3,
        source_root=str(out/'source_snapshot'/'rnj_wzzt_dr'), sources=sources,
        inputs_ref=io.ref(out/'inputs.json'), environment=io.environment(),
        protocol_document=io.ref(Path(__file__).with_name('PROTOCOL.md')), runner=io.ref(__file__),
        shared_runner=io.ref(ROOT/'rnj_wzzt_dr/experiments/rnj_pool_only_20261116/run.py'),
        no_AC_generation=True, no_DR=True, no_E60=True, truth_in_fitting=False,
        common_preparation='once physically per input, charged once to each method',
        matched_output_protection='basis and singleton-only in every method',
        selection='same validation MSE, ties and .08 training-only non-terminal contraction')
    io.write(out/'protocol.json', protocol)
    return protocol


def direct_baseline(algorithm, train, validation, common, config, method):
    """Use the actual fixed-support fitter, contraction and final selector."""
    started = perf_counter()
    deadline = started + config.total_seconds-common['seconds']
    fitter = algorithm.QPFitter(train, reference_mse=common['reference_mse'],
        bound=config.response_bound, qp_seconds=config.qp_seconds)
    reference_point = dict(common['reference_point'], objective=1., reference_mse=common['reference_mse'])
    fitter.cache[algorithm.trees.canonical(reference_point['supports'])] = reference_point
    families = ([common['guidance']['reference_family']] if method=='RNJ1' else
                common['guidance']['pool_families'])
    unique = list(dict.fromkeys(algorithm.trees.canonical(family) for family in families))
    path = [fitter.fit(algorithm.trees.basis(train.n))]
    origins = ['basis']
    for i, family in enumerate(unique):
        if perf_counter() >= deadline-config.output_reserve_seconds:
            raise TimeoutError('Direct RNJ fitting used the structural budget')
        point, _ = algorithm._clean(fitter, fitter.fit(family), config)
        path.append(point); origins.append('fixed_reference_RNJ' if method=='RNJ1' else 'RNJ_family_'+str(i))
    star = fitter.fit([(i,) for i in range(train.n)])
    raw_bank = [*path, star]; origins.append('singleton_protection')
    calibration = algorithm.output_floor.calibrate(common['guidance'], train)
    output, contraction = algorithm.output_floor.contract_bank(raw_bank, fitter,
        config.internal_floor, calibration, deadline=deadline)
    selected, mae, candidates = algorithm._output(output, train, validation, common['reference_mse'], config)
    result = dict(method=method, config=vars(config), selected=selected, selected_mse_view=selected,
        selected_mae_view=mae, selection_metric='validation_mse', frozen_candidates=candidates,
        path=path, events=[], QP=fitter.records, MIQP=[], output_contraction=contraction,
        uncontracted_output_path=raw_bank, candidate_origins=origins,
        reference_mse=common['reference_mse'], common_seconds=common['seconds'],
        fresh_seconds=perf_counter()-started, stop_reason='direct_RNJ_candidates_fitted',
        raw_RNJ_family_count=len(families), unique_RNJ_family_count=len(unique),
        native_bound_conflicts=0, complete_RNJ_output_fallback=False,
        truth_used=False, test_used=False,
        guarantee_scope='Fixed-support fits only; direct RNJ comparison baseline, no structural search')
    result['total_seconds'] = result['fresh_seconds']+common['seconds']
    result['total_cap_exceeded'] = result['total_seconds']>config.total_seconds
    result['competition'] = algorithm.competition(result, validation)
    return result


def fit_input(out, entry):
    out = Path(out); protocol = io.read(out/'protocol.json'); directory = out/'cases'/entry['id']
    if directory.exists():
        raise FileExistsError('Existing task retained: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json', dict(created_utc=utc(), pid=os.getpid()))
    rows = []; common = None
    try:
        for ref in protocol['sources']:
            check(ref)
        package = package_from_snapshot(protocol['source_root'], 'contribution')
        algorithm = import_module(package+'.identification.algorithm')
        observations = import_module(package+'.observations')
        history = import_module(package+'.history')
        train = observations.load(check(entry['files']['train.npz']))
        validation = observations.load(check(entry['files']['validation.npz']))
        if list(train.labels)!=entry['labels'] or list(validation.labels)!=entry['labels']:
            raise ValueError('Observation labels changed')
        config = algorithm.SearchConfig(**{k:protocol[k] for k in (
            'total_seconds', 'per_mip_seconds', 'output_reserve_seconds', 'qp_seconds')})
        common = algorithm.prepare_common(train, config)
    except Exception as exc:
        shared_error = dict(error=repr(exc), traceback=traceback.format_exc())
    for method in METHODS:
        target = directory/method; target.mkdir()
        started = perf_counter()
        try:
            if common is None:
                raise RuntimeError('Shared preparation failed: '+shared_error['error'])
            def heartbeat(row):
                io.write(target/'progress.json', dict(stage=row['stage'], elapsed_seconds=row['elapsed_seconds']))
            native = (direct_baseline(algorithm, train, validation, common, config, method)
                      if method.startswith('RNJ') else
                      algorithm.fit(train, validation, method=method, config=config, common=common, heartbeat=heartbeat))
            native['handoff'] = dict(schema='rnj_wzzt_dr_v2', training_sha256=observations.fingerprint(train),
                validation_sha256=observations.fingerprint(validation), parameter_rule='Same native contraction and strict selection')
            history.freeze_history(native, validation)
            io.write(target/'native.json', native)
            result = dict(status='complete', method=method, input_id=entry['id'], native=io.ref(target/'native.json'),
                seconds=perf_counter()-started, method_total_seconds=native['total_seconds'],
                total_cap_exceeded=native['total_cap_exceeded'], sources_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in protocol['sources']),
                truth_used=False, new_day_loaded=False, source_protocol=io.ref(out/'protocol.json'))
        except Exception as exc:
            result = dict(status='failed', method=method, input_id=entry['id'], seconds=perf_counter()-started,
                error=repr(exc), traceback=traceback.format_exc(),
                failure_details={k:getattr(exc,k) for k in ('receipt','qp_records','last_point') if hasattr(exc,k)})
        io.write(target/'result.json', result)
        rows.append(dict(input_id=entry['id'], method=method, status=result['status'], result=io.ref(target/'result.json')))
        print({k:rows[-1][k] for k in ('input_id','method','status')}, flush=True)
    return rows


def run(out):
    out = Path(out).resolve(); protocol = io.read(out/'protocol.json')
    if (out/'complete.json').exists():
        return io.read(out/'complete.json')
    lock = out/'running.lock'; fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    rows = []; started = perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            futures = {pool.submit(fit_input, str(out), entry):entry for entry in io.read(out/'inputs.json')}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    rows.extend(dict(input_id=entry['id'], method=m, status='worker_failed', error=repr(exc)) for m in METHODS)
                io.write(out/'progress.json', dict(completed=len(rows), planned=32, records=rows))
        result = dict(status='complete', planned=32, completed=len(rows), successful=sum(r['status']=='complete' for r in rows),
            failed=sum(r['status']!='complete' for r in rows), seconds=perf_counter()-started, records=rows)
        io.write(out/'complete.json', result)
        return result
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.resume:
        prepare(args.out)
    if not args.prepare_only:
        run(args.out)


if __name__ == '__main__':
    main()
