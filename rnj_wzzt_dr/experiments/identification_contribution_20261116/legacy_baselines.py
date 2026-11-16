"""Matched fixed-QP baselines and saved-bank replay for the active main 240.

There is no MIQP/AC execution here. Truth is opened only by evaluate(), after
all 960 planned method slots have terminal receipts and their banks are sealed.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
from importlib import import_module
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...evaluation_scope import load_catalog
from ..rnj_pool_only_20261116.run import ROOT, check, package_from_snapshot, snapshot, utc
from .run import direct_baseline

METHODS = ('RNJ1', 'RNJ15', '100', '101')
GROUP = 'large96_main'
PLANNED_INPUTS = 240


def select_main(entries):
    selected = [deepcopy(e) for e in entries if e['source_group'] == GROUP]
    if len(selected) != PLANNED_INPUTS or len({e['id'] for e in selected}) != PLANNED_INPUTS:
        raise ValueError('The sealed active main-240 denominator is incomplete or duplicated')
    for e in selected:
        if e.get('train_rows') != [0, 64] or e.get('validation_rows') != [64, 96] or e.get('new_day_ref'):
            raise ValueError('This replay requires exactly train[0:64], validation[64:96], no new day')
        if e['train_ref'] != e['validation_ref']:
            raise ValueError('Training and validation must slice the same sealed day')
    return selected


def load_rows(observations, entry):
    whole = observations.load(check(entry['train_ref']))
    if whole.points != 96 or list(whole.labels) != entry['labels']:
        raise ValueError('The original 96-row observed day or label order changed')
    result = []
    for name, bounds in (('train', (0, 64)), ('validation', (64, 96))):
        a, b = bounds
        part = observations.Observation(whole.p[a:b], whole.q[a:b], whole.y[a:b], whole.labels,
            None if whole.voltage is None else whole.voltage[a:b],
            None if whole.root is None else whole.root[a:b])
        if observations.fingerprint(part) != entry[name+'_fingerprint']:
            raise ValueError('Original '+name+' fingerprint does not reproduce')
        result.append(part)
    return tuple(result)


def candidate_points(native, method, n):
    if native.get('method') != method or native.get('selection_metric') != 'validation_mse':
        raise ValueError('Saved method or final validation metric differs')
    if native.get('truth_used') or native.get('test_used') or native.get('output_contraction') is not None:
        raise ValueError('Expected an uncontracted, observation-only original bank')
    rows = native['frozen_candidates']
    if not rows or [r['index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Saved candidate order must be contiguous and unchanged')
    points = [deepcopy(r['point']) for r in rows]
    family = lambda p: {tuple(c) for c in p['supports']}
    singletons = {(i,) for i in range(n)}
    if not any(family(p) == singletons for p in points):
        raise ValueError('Original singleton protection model is missing; do not add one')
    if not any(family(p) == singletons | {tuple(range(n))} for p in points):
        raise ValueError('Original basis model is missing; do not add one')
    return points


def check_common(common, original):
    """Numerical equivalence of independently rebuilt training-only geometry."""
    if common['training_sha256'] != original['training_sha256'] or common['labels'] != original['labels']:
        raise ValueError('Rebuilt common calibration used different training data')
    if not np.isclose(common['reference_mse'], original['reference_mse'], rtol=1e-10, atol=1e-15):
        raise ValueError('Rebuilt training reference MSE differs from the original search')
    a, b = common['guidance'], original['guidance']
    if a['reference_family'] != b['reference_family']:
        # JSON lists versus the current canonical tuple representation.
        if tuple(map(tuple, a['reference_family'])) != tuple(map(tuple, b['reference_family'])):
            raise ValueError('Reference RNJ family changed')
    if {tuple(c) for c in a['pool']} != {tuple(c) for c in b['pool']}:
        raise ValueError('Rebuilt raw-RNJ connection pool changed')
    deviations = {}
    for key in ('R', 'X', 'shared', 'depth'):
        x, y = np.asarray(a[key], float), np.asarray(b[key], float)
        if x.shape != y.shape or not np.allclose(x, y, rtol=1e-10, atol=1e-13):
            raise ValueError('Rebuilt training geometry changed: '+key)
        deviations[key] = float(np.max(np.abs(x-y)))
    return dict(passed=True, reference_rebuilt=float(common['reference_mse']),
        reference_original=float(original['reference_mse']), maximum_absolute_deviations=deviations,
        rtol=1e-10, geometry_atol=1e-13, reference_atol=1e-15)


def _read_old(entry, method):
    protocol = io.read(check(entry['protocol_ref']))
    wrapper = io.read(check(entry['legacy_methods'][method]))
    if (wrapper['status'] != 'fitted' or wrapper['method'] != method or
            wrapper['protocol_sha256'] != entry['protocol_ref']['sha256'] or
            wrapper['input_sha256'] != entry['train_ref']['sha256'] or
            wrapper['source_sha256_before'] != protocol['source_sha256'] or
            wrapper['source_sha256_after'] != protocol['source_sha256'] or wrapper.get('truth_used')):
        raise ValueError('Original method receipt does not match its sealed execution')
    if wrapper['common_sha256'] != entry['legacy_common_ref']['sha256']:
        raise ValueError('Original method used a different common calibration')
    candidate_points(wrapper['result'], method, len(entry['labels']))
    return wrapper


def prepare(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a new output: '+str(out))
    entries = select_main(load_catalog())
    for entry in entries:
        try:
            if entry.get('availability') != 'available':
                raise ValueError('Original catalog slot unavailable: '+str(entry.get('load_errors')))
            source_root = check(entry['protocol_ref']).parent
            seal = io.read(check(entry['candidate_seal_ref']))
            job = Path(entry['native_ref']['path']).parents[1].name
            entry['legacy_methods'] = {method: dict(path=str(source_root/'runs'/job/method/'result.json'),
                sha256=seal['result_sha256'][f'runs/{job}/{method}/result.json']) for method in ('100', '101')}
            common_key = f'common/{job}/common.json'
            entry['legacy_common_ref'] = dict(path=str(source_root/common_key), sha256=seal['common_sha256'][common_key])
            if entry['legacy_methods']['101']['sha256'] != entry['native_ref']['sha256']:
                raise ValueError('Catalog and original candidate seal disagree')
            for ref in [*entry['legacy_methods'].values(), entry['legacy_common_ref'], entry['train_ref'], *entry['executed_source_refs']]:
                check(ref)
            entry['preparation_status'] = 'ready'
        except Exception as exc:
            entry.update(preparation_status='failed', preparation_error=repr(exc))
    out.mkdir(parents=True)
    sources = snapshot(out/'source_snapshot', 'new')
    helper_paths = [Path(__file__), Path(__file__).with_name('run.py'),
        ROOT/'rnj_wzzt_dr/experiments/rnj_pool_only_20261116/run.py',
        ROOT/'rnj_wzzt_dr/experiments/rnj_pool_only_20261116/compare.py',
        ROOT/'rnj_wzzt_dr/experiments/impedance_floor_20261116/sources.py',
        ROOT/'rnj_wzzt_dr/evaluation_scope.py', ROOT/'rnj_wzzt_dr/evaluation_scope.json', ROOT/'rnj_wzzt_dr/io.py']
    io.write(out/'inputs.json', entries)
    protocol = dict(created_utc=utc(), methods=list(METHODS), source_group=GROUP, planned_inputs=240,
        planned_method_slots=960, new_direct_baseline_slots=480, saved_bank_replay_slots=480,
        source_root=str(out/'source_snapshot'/'rnj_wzzt_dr'), sources=sources,
        helper_sources=[io.ref(p) for p in helper_paths], inputs_ref=io.ref(out/'inputs.json'),
        protocol_document=io.ref(Path(__file__).with_name('PROTOCOL_LEGACY.md')), environment=io.environment(),
        total_seconds=120., per_mip_seconds=120., output_reserve_seconds=12., qp_seconds=300.,
        internal_floor=.08, workers=3, no_new_MIQP=True, no_new_AC=True, no_new_observed_data=True,
        training_rows=[0,64], validation_rows=[64,96], new_day=None,
        time_scope='RNJ includes one rebuilt common charge; replay separates original cost and new postprocessing',
        truth_boundary='Evaluation only after all 960 terminal method receipts and fit_complete.json')
    io.write(out/'protocol.json', protocol)
    return protocol


class _NoStructuralSearch:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Structural optimization is forbidden in fixed-bank replay')


def replay_bank(algorithm, old, train, validation, common, config, method):
    started = perf_counter(); deadline = started+config.total_seconds-common['seconds']
    points = candidate_points(old, method, train.n)
    if not np.isclose(old['reference_mse'], common['reference_mse'], rtol=1e-10, atol=1e-15):
        raise ValueError('Saved candidate scale differs from rebuilt training scale')
    if old['config']['validation_tie'] != config.validation_tie or old['config']['response_bound'] != config.response_bound:
        raise ValueError('Saved candidate numerical selection or coefficient bounds differ')
    fitter = algorithm.QPFitter(train, reference_mse=common['reference_mse'],
        bound=config.response_bound, qp_seconds=config.qp_seconds)
    calibration = algorithm.output_floor.calibrate(common['guidance'], train)
    output, contraction = algorithm.output_floor.contract_bank(points, fitter, config.internal_floor,
        calibration, deadline=deadline)
    selected, mae, candidates = algorithm._output(output, train, validation, common['reference_mse'], config)
    if len(candidates) != len(points):
        raise ValueError('Replay added or discarded a candidate slot')
    native = dict(method=method, config=vars(config), selected=selected, selected_mse_view=selected,
        selected_mae_view=mae, selection_metric='validation_mse', frozen_candidates=candidates,
        path=points, events=[], QP=fitter.records, MIQP=[], output_contraction=contraction,
        uncontracted_output_path=points, candidate_origins=['legacy_candidate_'+str(i) for i in range(len(points))],
        reference_mse=common['reference_mse'], common_seconds=common['seconds'],
        fresh_seconds=perf_counter()-started, stop_reason='saved_bank_contracted_without_structural_search',
        native_bound_conflicts=0, complete_RNJ_output_fallback=False, truth_used=False, test_used=False,
        guarantee_scope='Fixed saved bank with .08 postprocessing; no rerun of the historical search')
    native['total_seconds'] = native['fresh_seconds']+common['seconds']
    native['total_cap_exceeded'] = native['total_seconds']>config.total_seconds
    native['competition'] = algorithm.competition(native, validation)
    return native


def fit_input(out, entry):
    out = Path(out); protocol = io.read(out/'protocol.json'); directory = out/'cases'/entry['id']
    directory.mkdir(parents=True, exist_ok=True)
    rows = []; common = None; shared_error = None
    try:
        for ref in [*protocol['sources'], *protocol['helper_sources']]: check(ref)
        existing = {}
        for method in METHODS:
            result_file = directory/method/'result.json'
            if not result_file.exists(): continue
            saved = io.read(result_file)
            if saved.get('source_protocol', {}).get('sha256') != io.ref(out/'protocol.json')['sha256']:
                raise ValueError('Existing method receipt belongs to a different protocol')
            if saved.get('native'):
                check(saved['native'])
                old_seal = io.read(check(saved['candidate_seal']))
                check(old_seal['common'])
            existing[method] = dict(id=entry['id'], method=method, status=saved['status'], result=io.ref(result_file))
        if len(existing) == len(METHODS):
            return [existing[m] for m in METHODS]
        if entry['preparation_status'] != 'ready':
            raise ValueError(entry['preparation_error'])
        key = 'legacy_'+hashlib.sha256(protocol['source_root'].encode()).hexdigest()[:12]
        package = package_from_snapshot(protocol['source_root'], key)
        algorithm = import_module(package+'.identification.algorithm')
        observations = import_module(package+'.observations')
        history = import_module(package+'.history')
        algorithm.StructuralSolver = _NoStructuralSearch
        train, validation = load_rows(observations, entry)
        old_wrappers = {method:_read_old(entry, method) for method in ('100', '101')}
        old_common = io.read(check(entry['legacy_common_ref']))['common']
        config = algorithm.SearchConfig(**{k:protocol[k] for k in (
            'total_seconds','per_mip_seconds','output_reserve_seconds','qp_seconds','internal_floor')})
        config.validate()
        if (directory/'common.json').exists():
            saved_common = io.read(directory/'common.json')
            if saved_common['original'] != entry['legacy_common_ref'] or saved_common['source_protocol'] != io.ref(out/'protocol.json'):
                raise ValueError('Saved common calibration belongs to a different input/protocol')
            common = saved_common['common']; check_common(common, old_common)
        else:
            if (directory/'common_started.json').exists():
                raise RuntimeError('Partial common preparation retained; do not reset its budget')
            io.write(directory/'common_started.json', dict(created_utc=utc(), pid=os.getpid()))
            common = algorithm.prepare_common(train, config)
            alignment = check_common(common, old_common)
            # Use the exact historical Mref after checking numerical equivalence.
            common['reference_mse'] = float(old_common['reference_mse'])
            io.write(directory/'common.json', dict(common=common, alignment=alignment,
                original=entry['legacy_common_ref'], source_protocol=io.ref(out/'protocol.json'),
                truth_used=False, validation_used=False))
    except Exception as exc:
        common = None; shared_error = dict(error=repr(exc), traceback=traceback.format_exc())
    for method in METHODS:
        target = directory/method; target.mkdir(exist_ok=True)
        result_file = target/'result.json'
        if result_file.exists():
            saved = io.read(result_file)
            if saved.get('source_protocol', {}).get('sha256') != io.ref(out/'protocol.json')['sha256']:
                raise ValueError('Existing method receipt belongs to a different protocol')
            if saved.get('native'): check(saved['native'])
            rows.append(dict(id=entry['id'], method=method, status=saved['status'], result=io.ref(result_file)))
            continue
        started = perf_counter()
        try:
            if (target/'started.json').exists():
                raise RuntimeError('Previous partial attempt retained; do not reset its budget')
            io.write(target/'started.json', dict(created_utc=utc(), pid=os.getpid()))
            if common is None:
                raise RuntimeError('Shared preparation failed: '+shared_error['error'])
            native = (direct_baseline(algorithm, train, validation, common, config, method)
                if method.startswith('RNJ') else replay_bank(algorithm, old_wrappers[method]['result'],
                    train, validation, common, config, method))
            native['handoff'] = dict(schema='rnj_wzzt_dr_v2', training_sha256=observations.fingerprint(train),
                validation_sha256=observations.fingerprint(validation), parameter_rule='Matched .08 saved-bank/direct RNJ processing')
            history.freeze_history(native, validation)
            io.write(target/'native.json', native)
            io.write(target/'candidate_seal.json', dict(native=io.ref(target/'native.json'),
                common=io.ref(directory/'common.json'), training=entry['train_ref'], validation=entry['validation_ref'],
                train_rows=[0,64], validation_rows=[64,96], truth_used=False, created_utc=utc()))
            original = old_wrappers[method] if method in ('100','101') else None
            unchanged = all(io.ref(ref['path'])['sha256']==ref['sha256'] for ref in [*protocol['sources'], *protocol['helper_sources']])
            if not unchanged: raise ValueError('Executed source changed during fixed-QP processing')
            result = dict(status='complete', method=method, input_id=entry['id'], native=io.ref(target/'native.json'),
                candidate_seal=io.ref(target/'candidate_seal.json'), seconds=perf_counter()-started,
                common_seconds=common['seconds'], new_processing_seconds=native['total_seconds'],
                original_method_seconds=None if original is None else original['total_seconds'],
                original_method=None if original is None else entry['legacy_methods'][method],
                time_scope='new direct baseline' if original is None else 'historical search plus separately recorded postprocessing',
                total_cap_exceeded=native['total_cap_exceeded'], sources_unchanged=unchanged,
                structural_solver_calls=0, truth_used=False, source_protocol=io.ref(out/'protocol.json'))
        except Exception as exc:
            result = dict(status='failed', method=method, input_id=entry['id'], seconds=perf_counter()-started,
                error=repr(exc), traceback=traceback.format_exc(), shared_error=shared_error,
                failure_details={k:getattr(exc,k) for k in ('receipt','qp_records','last_point') if hasattr(exc,k)},
                source_protocol=io.ref(out/'protocol.json'))
        io.write(result_file, result)
        rows.append(dict(id=entry['id'], method=method, status=result['status'], result=io.ref(result_file)))
    return rows


def run(out):
    out = Path(out).resolve(); protocol = io.read(out/'protocol.json')
    if (out/'fit_complete.json').exists(): return io.read(out/'fit_complete.json')
    check(protocol['inputs_ref'])
    for ref in [*protocol['sources'], *protocol['helper_sources']]: check(ref)
    lock = out/'running.lock'; fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    rows = []; started = perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            futures = {pool.submit(fit_input, str(out), entry):entry for entry in io.read(protocol['inputs_ref']['path'])}
            for future in as_completed(futures):
                entry = futures[future]
                try: rows.extend(future.result())
                except Exception as exc:
                    rows.extend(dict(id=entry['id'], method=method, status='worker_failed', error=repr(exc)) for method in METHODS)
                io.write(out/'progress.json', dict(completed=len(rows), planned=960, records=rows))
                print(json.dumps(dict(completed=len(rows), planned=960), ensure_ascii=False), flush=True)
        result = dict(status='complete', planned=960, completed=len(rows),
            successful=sum(r['status']=='complete' for r in rows), failed=sum(r['status']!='complete' for r in rows),
            seconds=perf_counter()-started, records=rows, truth_opened=False,
            protocol=io.ref(out/'protocol.json'), sources_unchanged=all(io.ref(r['path'])['sha256']==r['sha256']
                for r in [*protocol['sources'], *protocol['helper_sources']]))
        if len(rows) != 960 or len({(r['id'],r['method']) for r in rows}) != 960:
            raise RuntimeError('Terminal method denominator is incomplete or duplicated')
        io.write(out/'fit_complete.json', result)
        return result
    finally:
        lock.unlink(missing_ok=True)


def evaluate(out):
    """Independent saved-array selection/topology checks; no fits are called."""
    from ..impedance_floor_20261116.sources import load_evaluation
    from ..rnj_pool_only_20261116.compare import audit_native, strict_selection, topology
    out = Path(out).resolve(); report_dir = out/'evaluation'
    if report_dir.exists(): raise FileExistsError('Keep previous evaluation: '+str(report_dir))
    completion = io.read(out/'fit_complete.json'); protocol = io.read(check(completion['protocol']))
    if completion['completed'] != 960 or completion['truth_opened'] or not completion['sources_unchanged']:
        raise ValueError('All source-consistent planned fits must terminate before evaluation')
    for ref in [*protocol['sources'], *protocol['helper_sources']]: check(ref)
    entries = io.read(check(protocol['inputs_ref'])); terminal = {(r['id'],r['method']):r for r in completion['records']}
    if len(entries)!=240 or len(terminal)!=960: raise ValueError('Changed evaluation denominator')
    key = 'legacy_eval_'+hashlib.sha256(protocol['source_root'].encode()).hexdigest()[:12]
    package = package_from_snapshot(protocol['source_root'], key)
    observations = import_module(package+'.observations')
    rows, errors, truth_refs = [], [], []
    for entry in entries:
        data = None; evaluation = None
        for method in METHODS:
            slot = terminal[(entry['id'],method)]
            row = dict(id=entry['id'], case=entry['case'], method=method, status=slot['status'])
            try:
                if slot['status'] != 'complete':
                    row['failure'] = io.read(check(slot['result'])) if slot.get('result') else slot
                    rows.append(row); continue
                result = io.read(check(slot['result'])); native = io.read(check(result['native']))
                seal = io.read(check(result['candidate_seal']))
                if seal['native'] != result['native'] or seal['truth_used']: raise ValueError('Candidate seal mismatch')
                check(seal['common'])
                if seal['train_rows'] != [0,64] or seal['validation_rows'] != [64,96]:
                    raise ValueError('Candidate seal changed the historical split')
                if data is None:
                    train, valid = load_rows(observations, entry)
                    convert = lambda a: dict(P=a.p,Q=a.q,Y=a.y,labels=list(a.labels))
                    data = dict(train=convert(train), validation=convert(valid))
                    evaluation = load_evaluation(entry)
                    truth_refs.extend(evaluation['source_refs'])
                truth = evaluation['truth_supports']; n = len(entry['labels'])
                checked_native = audit_native(native, **data, new_day=None, truth=truth)
                raw = strict_selection(native['uncontracted_output_path'], [data['validation']],
                    native['reference_mse'], native['config']['validation_tie'])
                raw_topology = topology(native['uncontracted_output_path'][raw['selected']],truth,n,native['config']['zero_tol'])
                final = checked_native['phases']['validation']
                row.update(status='complete', audit=checked_native, raw_selection=raw,
                    raw_topology=raw_topology, final=final,
                    selected_origin=native['candidate_origins'][final['index']],
                    new_processing_seconds=result['new_processing_seconds'], original_method_seconds=result['original_method_seconds'],
                    total_cap_exceeded=result['total_cap_exceeded'], new_QP_count=len(native['QP']),
                    new_MIQP_count=len(native['MIQP']), native_ref=result['native'], result_ref=slot['result'])
                if row['new_MIQP_count'] != 0: raise ValueError('Structural solve appeared in fixed-bank replay')
            except Exception as exc:
                row.update(status='audit_failed', error=repr(exc)); errors.append(dict(id=entry['id'],method=method,error=repr(exc)))
            rows.append(row)
    summaries = {}
    for method in METHODS:
        subset = [r for r in rows if r['method']==method]; ok = [r for r in subset if r['status']=='complete']
        summaries[method] = dict(planned=240, completed=len(ok), failed=240-len(ok),
            raw_internal=sum(r['raw_topology']['internal_exact'] for r in ok), raw_full=sum(r['raw_topology']['full_exact'] for r in ok),
            internal=sum(r['final']['topology']['internal_exact'] for r in ok), full=sum(r['final']['topology']['full_exact'] for r in ok),
            internal_coverage=sum(r['audit']['coverage']['internal'] for r in ok),
            full_coverage=sum(r['audit']['coverage']['full'] for r in ok),
            new_processing_seconds=sum(r['new_processing_seconds'] for r in ok),
            historical_method_seconds=sum(r['original_method_seconds'] or 0 for r in ok),
            total_cap_exceeded=sum(r['total_cap_exceeded'] for r in ok), new_QP_count=sum(r['new_QP_count'] for r in ok), new_MIQP_count=0)
    by_key = {(r['id'],r['method']):r for r in rows}; paired = []
    for baseline in ('RNJ1','RNJ15'):
        for search in ('100','101'):
            matches=[(by_key[(e['id'],baseline)],by_key[(e['id'],search)]) for e in entries]
            matches=[(a,b) for a,b in matches if a['status']==b['status']=='complete']
            item=dict(baseline=baseline, search=search, planned_pairs=240, completed_pairs=len(matches))
            for metric in ('internal_exact','full_exact'):
                item[metric]=dict(new=sum(not a['final']['topology'][metric] and b['final']['topology'][metric] for a,b in matches),
                    lost=sum(a['final']['topology'][metric] and not b['final']['topology'][metric] for a,b in matches))
            paired.append(item)
    for ref in [*protocol['sources'], *protocol['helper_sources'], *truth_refs]: check(ref)
    report = dict(status='passed' if not errors and all(r['status']=='complete' for r in rows) else 'incomplete_or_failed',
        planned=960, completed=sum(r['status']=='complete' for r in rows), errors=errors, rows=rows,
        summary=summaries, paired=paired, protocol=io.ref(out/'protocol.json'), fit_completion=io.ref(out/'fit_complete.json'),
        evaluation_sources=truth_refs, postprocessing_source=io.ref(__file__),
        scope='Known main 240; 64 train and 32 validation, no independent new day; no new MIQP/AC; pressure excluded')
    report_dir.mkdir(); io.write(report_dir/'report.json', report)
    lines=['# 主240：固定RNJ与旧联合库的同后处理对照','',
        f"状态：{report['status']}；完成 {report['completed']}/960；独立重算错误 {len(errors)}。",'',
        '|方法|完成/计划|收缩前内部/全支持|收缩后内部/全支持|本次处理秒|历史方法秒|新QP|新MIQP|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for method,s in summaries.items():
        lines.append(f"|{method}|{s['completed']}/240|{s['raw_internal']}/{s['raw_full']}|{s['internal']}/{s['full']}|{s['new_processing_seconds']:.3f}|{s['historical_method_seconds']:.3f}|{s['new_QP_count']}|0|")
    lines += ['', '恢复计数分母均为240，失败保留。RNJ1/RNJ15共享基础项与单节点保护模型。100/101只处理旧库，不重搜、不补候选；其历史方法秒和本次处理秒分别列出，不能把后处理时间当作联合算法运行时间。', '',
        '本批重复使用既有96点日，只有64训练＋32验证，没有新的独立测试日。内部连接恢复与全支持恢复分别统计。旧压力160不在本批。', '',
        '|基线→旧搜索库|有效/计划配对|内部新增/丢失|全支持新增/丢失|','|---|---|---|---|']
    for p in paired:
        a,b=p['internal_exact'],p['full_exact'];lines.append(f"|{p['baseline']}→{p['search']}|{p['completed_pairs']}/240|{a['new']}/{a['lost']}|{b['new']}/{b['lost']}|")
    (report_dir/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (report_dir/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method',*next(iter(summaries.values())).keys()]);writer.writeheader()
        writer.writerows(dict(method=m,**s) for m,s in summaries.items())
    io.write(out/'complete.json',dict(status=report['status'], report=io.ref(report_dir/'report.json'),
        planned=960, completed=report['completed'], fit_complete=io.ref(out/'fit_complete.json')))
    return report


def self_test():
    """Small pure tests: no fitting, structural solver or physical evaluation."""
    import tempfile
    from ... import observations
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary)/'day.npz'; z=np.arange(192,dtype=float).reshape(96,2)+1
        np.savez(p,P=z,Q=2*z,Y=3*z,labels=[101,102])
        whole=observations.load(p)
        entry=dict(labels=[101,102],train_ref=io.ref(p),train_fingerprint=observations.fingerprint(
            observations.Observation(z[:64],2*z[:64],3*z[:64],(101,102))),
            validation_fingerprint=observations.fingerprint(observations.Observation(z[64:],2*z[64:],3*z[64:],(101,102))))
        train,valid=load_rows(observations,entry)
        assert train.points==64 and valid.points==32 and train.p[-1,0]<valid.p[0,0]
        entries=[dict(id=str(i),source_group=GROUP,train_rows=[0,64],validation_rows=[64,96],
            train_ref=entry['train_ref'],validation_ref=entry['train_ref'],new_day_ref=None) for i in range(240)]
        assert len(select_main([*entries,dict(id='retired',source_group='large96_pressure')]))==240
        try: select_main(entries[:-1])
        except ValueError: pass
        else: raise AssertionError('A missing active input was accepted')
        points=[dict(supports=[[0],[1],[0,1]],r=[1,1,0],x=[0,0,0]),dict(supports=[[0],[1]],r=[1,1],x=[0,0])]
        native=dict(method='100',selection_metric='validation_mse',output_contraction=None,
            frozen_candidates=[dict(index=i,point=p) for i,p in enumerate(points)])
        assert candidate_points(native,'100',2)==points
        missing=deepcopy(native);missing['frozen_candidates']=missing['frozen_candidates'][:1]
        try: candidate_points(missing,'100',2)
        except ValueError: pass
        else: raise AssertionError('A missing original protection model was silently repaired')
        try: _NoStructuralSearch()
        except RuntimeError: pass
        else: raise AssertionError('Structural optimization was not blocked')
    return dict(status='passed', checks=6, fixed_QP_calls=0, MIQP_calls=0, AC_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--run-only',action='store_true')
    parser.add_argument('--evaluate-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()));return
    if args.out is None: parser.error('--out is required except for --self-test')
    if args.evaluate_only:
        report=evaluate(args.out)
    else:
        if not args.resume: prepare(args.out)
        if args.prepare_only: print('prepared; no fits executed');return
        result=run(args.out)
        if args.run_only: print(json.dumps({k:result[k] for k in ('status','planned','completed','failed')}));return
        report=evaluate(args.out)
    print(json.dumps({k:report[k] for k in ('status','planned','completed','summary')},ensure_ascii=False))
    if report['status']!='passed': raise SystemExit(1)


if __name__=='__main__':
    main()
