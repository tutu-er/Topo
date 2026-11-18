"""Forty history-only refits with independently controlled P/Q meter noise.

The original 384-point physical worlds, forecast and four-equipment catalogues
are immutable. Training uses rows 0:192; validation 192:288; evaluation 288:384.
No AC is solved and no true-support point is added to the native candidate bank.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
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
from ..flexibility_grid_20261115.data import checked, arrays, utc


ROOT = Path(__file__).resolve().parents[3]
CASES = ('soumalas11', 'pengwah18')
PQ_SDS = (0., .0025, .005, .01, .02)
VOLTAGE_SD = .0005
HISTORY_REPEATS = 4
EVALUATION_PATHS = 16
NOISE_SEED = 2026111521
SOURCE_RUN = ROOT/'data/equipment_flex_execution_20261115_v1'
SOURCE_AUDIT = ROOT/'outputs/equipment_flex_audit_20261115_v1/report.json'
DEFAULT_OUT = ROOT/'data/history288_pq_noise_data_20261115_v1'


def sample_indices():
    return dict(train=np.arange(192), validation=np.arange(192, 288), new_day=np.arange(288, 384))


def split_noise(n, original_index, history_repeat, *, split, path=0, seed=NOISE_SEED):
    """Independent history/evaluation streams; no noise-amplitude key."""
    if (n < 1 or original_index < 0 or history_repeat not in range(HISTORY_REPEATS)
            or split not in (0, 1) or path not in range(EVALUATION_PATHS)
            or (split == 0 and path != 0)):
        raise ValueError('Invalid locked noise-stream coordinates')
    count = 288 if split == 0 else 96
    result = {}
    for name, component in (('P', 2), ('Q', 3), ('V', 4), ('root', 5)):
        shape = (count,) if name == 'root' else (count, n)
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(seed), 2718, int(original_index), int(history_repeat), split, path, component]))
        result[name] = rng.normal(size=shape)
    return result


def observed_splits(clean, standard_noise, pq_relative_sd):
    if pq_relative_sd not in PQ_SDS:
        raise ValueError('P/Q standard deviation is outside the locked five levels')
    labels = tuple(map(int, clean['labels'])); measured = {}
    if len(set(labels)) != len(labels):
        raise ValueError('Duplicate terminal labels')
    for name in ('P', 'Q', 'V', 'root'):
        shape = (384,) if name == 'root' else (384, len(labels))
        value, noise = np.asarray(clean[name], float), np.asarray(standard_noise[name], float)
        if value.shape != shape or noise.shape != shape or not np.isfinite(value).all() or not np.isfinite(noise).all():
            raise ValueError('Invalid clean/noise array: '+name)
        sd = pq_relative_sd if name in ('P', 'Q') else VOLTAGE_SD
        measured[name] = value*(1+sd*noise)
    if np.any(measured['V'] <= 0) or np.any(measured['root'] <= 0):
        raise ValueError('Nonpositive measured voltage')
    y = measured['root'][:, None]**2-measured['V']**2
    return {name: Observation(measured['P'][idx], measured['Q'][idx], y[idx], labels,
            measured['V'][idx], measured['root'][idx]) for name, idx in sample_indices().items()}


def source_entries(entries):
    """Select named cached worlds, with identical duplicate catalogue references."""
    answer = []
    keys = ('physical_cache', 'resource_context', 'forecast_ref', 'equipment_spec', 'base_kw', 'index')
    for case in CASES:
        found = sorted((e for e in entries if e['case'] == case and e['start_date'] == '2024-01-04'), key=lambda e:e['id'])
        if not found:
            raise ValueError('Missing predeclared world: '+case+'__2024-01-04')
        first = found[0]
        for row in found[1:]:
            for key in keys:
                # Identical equipment copies may have distinct paths but must
                # retain exactly the same sealed content.
                a, b = first[key], row[key]
                equal = a['sha256'] == b['sha256'] if key in ('equipment_spec', 'forecast_ref') else a == b
                if not equal:
                    raise ValueError('Duplicate world changes frozen '+key)
        item = {key:deepcopy(first[key]) for key in keys}
        item.update(case=case, start_date='2024-01-04', original_id=case+'__2024-01-04',
            original_input_index=int(first['index']), source_equipment_input_id=first['id'],
            forecast_key='forecast00', forecast_relative_sd=.05)
        answer.append(item)
    return answer


def source_paths():
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if (path.suffix == '.py' and path.is_relative_to(ROOT) and 'outputs' not in path.parts
                    and not path.name.startswith('test_') and path.name not in ('audit.py','report.py','plot.py')):
                paths.add(path)
    paths.add(Path(__file__).with_name('PROTOCOL.md'))
    for case in CASES:
        paths.update(physical.ac_backend.ROOT/'cases'/case/name for name in ('metadata.json','buses.csv','branches.csv'))
    return sorted(paths)


def prepare(out, *, source_run=SOURCE_RUN, source_audit=SOURCE_AUDIT, workers=3):
    out, source_run = Path(out).resolve(), Path(source_run).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing preparation: '+str(out))
    if not 1 <= workers <= 3:
        raise ValueError('Use 1..3 single-thread workers')
    if shutil.disk_usage(out.parent).free < 5*2**30:
        raise RuntimeError('Less than 5 GiB free workspace space')
    document = Path(__file__).with_name('PROTOCOL.md')
    if not document.exists():
        raise FileNotFoundError('Lock PROTOCOL.md before preparation')
    previous, done = io.read(source_run/'protocol.json'), io.read(source_run/'complete.json')
    audited = io.read(source_audit)
    if (done.get('status') != 'complete' or not done.get('source_unchanged')
            or audited.get('status') != 'passed' or audited.get('audited_rows') != 144
            or audited.get('source_protocol') != io.ref(source_run/'protocol.json')):
        raise ValueError('Completed independently audited equipment source required')
    old_snapshots = []
    for ref in previous['sources']:
        saved = source_run/'source_snapshot'/Path(ref['path']).resolve().relative_to(ROOT)
        checked(dict(path=str(saved), sha256=ref['sha256'])); old_snapshots.append(io.ref(saved))
    chosen = source_entries(io.read(checked(previous['inputs'])))
    refs = [io.ref(source_run/'protocol.json'), io.ref(source_run/'complete.json'),
            io.ref(source_audit), previous['inputs']]
    tasks = []; out.mkdir(parents=True)
    for source in chosen:
        for ref in (source['physical_cache']['original'], source['physical_cache']['cache'],
                    source['resource_context'], source['equipment_spec'], source['forecast_ref']):
            checked(ref); refs.append(ref)
        clean = arrays(source['physical_cache']['original']['path']); n = len(clean['labels'])
        if clean['P'].shape != (384,n) or len(clean['timestamps']) != 384:
            raise ValueError('Expected the immutable full 384-point world')
        times = [datetime.fromisoformat(str(t)) for t in clean['timestamps']]
        if any((b-a).total_seconds() != 900 for a,b in zip(times,times[1:])):
            raise ValueError('Original physical data is not uniformly 15-minute sampled')
        forecast = arrays(source['forecast_ref']['path'])[source['forecast_key']]
        if forecast.shape != (96,2*n) or not np.isfinite(forecast).all():
            raise ValueError('Original sealed forecast shape differs')
        for h in range(HISTORY_REPEATS):
            folder = out/'noise'/source['original_id']/f'h{h:02d}'; folder.mkdir(parents=True)
            history = split_noise(n,source['index'],h,split=0)
            paths = [split_noise(n,source['index'],h,split=1,path=r) for r in range(EVALUATION_PATHS)]
            evaluation = {key:np.stack([r[key] for r in paths]) for key in history}
            standard = {key:np.concatenate((history[key],evaluation[key][0])) for key in history}
            for name, value in (('history_noise',history),('evaluation_noise',evaluation),('standard_noise',standard)):
                np.savez_compressed(folder/(name+'.npz'), **value)
            noise_refs = {name:io.ref(folder/(name+'.npz')) for name in ('history_noise','evaluation_noise','standard_noise')}
            refs.extend(noise_refs.values())
            for sd in PQ_SDS:
                tag = f'{sd:g}'.replace('.','p')
                tasks.append(dict(id=source['original_id']+f'__h{h:02d}__pq{tag}',grid_order=len(tasks),
                    **deepcopy(source),phase='history288_pq_noise_development',history_repeat=h,noise_repeat=h,
                    pq_relative_sd=sd,meter_PQ_relative_sd=sd,meter_V_relative_sd=VOLTAGE_SD,
                    meter=dict(additional_PQ_relative_sd=sd,additional_V_relative_sd=VOLTAGE_SD),
                    training_rows=192,validation_rows=96,new_rows=96,training_points=192,validation_points=96,new_points=96,
                    training_hours=48,validation_hours=24,history_observation_spacing_minutes=15,
                    labels=clean['labels'].tolist(),noise_seed=NOISE_SEED,evaluation_paths=EVALUATION_PATHS,
                    **noise_refs))
    paths = source_paths(); sources = [io.ref(path) for path in paths]
    for path in paths:
        destination = out/'source_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,destination)
    protocol = dict(created_utc=utc(),phase='history288_pq_noise_development',source_document=io.ref(document),
        source_equipment_run=str(source_run),source_equipment_audit=io.ref(source_audit),
        previous_source_snapshots=old_snapshots,input_refs=list({(r['path'],r['sha256']):r for r in refs}.values()),
        sources=sources,tasks=tasks,planned_inputs=40,workers=workers,cases=list(CASES),pq_relative_sds=list(PQ_SDS),
        voltage_relative_sd=VOLTAGE_SD,history_repeats=HISTORY_REPEATS,evaluation_paths=EVALUATION_PATHS,noise_seed=NOISE_SEED,
        noise_stream='SeedSequence([seed,2718,original_index,history_repeat,split,path,component]); split0 history288/path0, split1 evaluation96/path0..15; P2,Q3,V4,root5; no PQ-level key',
        split='Train0:192, validation192:288, evaluation288:384; all15min and no stride',
        method='101',method_config=asdict(validate._config(120.)),method_budget_seconds=120.,additional_E_pool_search_seconds=60.,
        history_scoring='Validation96 only; training192 fits parameters. Final validation96+new96 residual cells weighted equally.',
        new_day_file='Evaluation path0 only, used after candidate sealing for diagnostic scoring; all16 meter streams separately saved for execution',
        frozen_physics='Original cached384P/Q/V/root and scaling unchanged; no new AC',
        resources='One original audited four-equipment specification per physical world; no old independent-Q catalogue',
        forecast='Exact original sealed forecast00 (5 percent); fixed across PQ levels and history repetitions',
        new_day_identifier_inputs=False,truth_rule='True structure used only after candidate seal for evaluation; no oracle fit or candidate addition',
        failure_rule='Completed success/failure markers reused; existing partial cells rejected without resetting budgets; every one of40 tasks retained',
        interpretation='Known-world development sensitivity; two physical days with four independent historical noise repetitions, not forty independent network days',
        environment=validate._environment(),**validate._git_receipt(paths))
    io.write(out/'protocol.json',protocol)
    return protocol


def verify_protocol(protocol):
    for ref in [protocol['source_document'],*protocol['sources'],*protocol['previous_source_snapshots'],*protocol['input_refs']]:
        checked(ref)


def cached_completion(directory, task):
    directory = Path(directory)
    if (directory/'complete.json').exists():
        saved = io.read(directory/'complete.json')
        if saved['task'] != task:
            raise ValueError('Completed cell belongs to a different locked task')
        checked(saved['result'])
        if saved.get('entry'):
            entry=saved['entry']
            for ref in [entry['score'],entry['native_copy'],*entry['files'].values()]:checked(ref)
        return saved
    if directory.exists():
        raise RuntimeError('Partial cell needs explicit recovery; no automatic budget reset: '+str(directory))
    return None


def fit_cell(task, out):
    out=Path(out);p=io.read(out/'protocol.json');directory=out/'histories'/task['id']
    cached=cached_completion(directory,task)
    if cached is not None:return cached
    directory.mkdir(parents=True);began=perf_counter();stage='observations'
    io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=utc(),task=task))
    try:
        clean=arrays(checked(task['physical_cache']['original']))
        noise=arrays(checked(task['standard_noise']))
        splits=observed_splits(clean,noise,task['pq_relative_sd']);refs={}
        for name,obs in splits.items():
            path=directory/(name+'.npz')
            np.savez_compressed(path,P=obs.p,Q=obs.q,Y=obs.y,V=obs.voltage,root=obs.root,labels=obs.labels)
            refs[name]=dict(ref=io.ref(path),sha256=fingerprint(obs))
        sampling={name:dict(indices=idx.tolist(),points=len(idx),timestamps=[str(clean['timestamps'][i]) for i in idx])
                  for name,idx in sample_indices().items()}
        io.write(directory/'observations_sealed.json',dict(created_utc=utc(),observations=refs,sampling=sampling,
            physical_source=task['physical_cache']['original'],standard_noise=task['standard_noise'],
            history_noise=task['history_noise'],evaluation_noise=task['evaluation_noise'],
            noise_seed=NOISE_SEED,history_repeat=task['history_repeat'],pq_relative_sd=task['pq_relative_sd'],
            voltage_relative_sd=VOLTAGE_SD,evaluation_path=0,new_day_used_for_fitting=False))
        stage='fit101';tick=perf_counter()
        original=native_json_record(fit_observed(splits['train'],splits['validation'],method='101',config=SearchConfig(**p['method_config'])))
        fit_seconds=perf_counter()-tick;io.write(directory/'original_101_native.json',original)
        original_frozen=freeze_history(original,splits['validation'])
        stage='additional_E60';search=additional_search(original,splits['train'],p['additional_E_pool_search_seconds'])
        io.write(directory/'search_60s.json',native_json_record(search))
        stage='freeze_candidates';enriched=append_candidate(original,search.get('closed_point'),splits['train'],splits['validation'])
        enriched['candidate_enrichment']=dict(original_native=io.ref(directory/'original_101_native.json'),source_search=io.ref(directory/'search_60s.json'),
            original_points_preserved=len(original['frozen_candidates']),added_points=int(search.get('closed_point') is not None),
            original_fit_actual_seconds=fit_seconds,extra_search_status=search['status'],extra_search_actual_seconds=search.get('total_seconds'),
            original_runtime_fields_scope='Original120s fit only; extra E60 accounted separately',truth_used=False,new_day_values_used=False)
        io.write(directory/'101_native.json',native_json_record(enriched))
        score=pack_score(enriched,splits['train'],splits['validation']);io.write(directory/'score.json',score)
        frozen=freeze_history(enriched,splits['validation'])
        io.write(directory/'candidates_sealed.json',dict(created_utc=utc(),native=io.ref(directory/'101_native.json'),
            original_native=io.ref(directory/'original_101_native.json'),score=io.ref(directory/'score.json'),
            training=refs['train'],validation=refs['validation'],parameter_sha256=frozen.parameter_sha256,
            original_parameter_sha256=original_frozen.parameter_sha256,truth_used=False,new_day_values_used=False))
        stage='offline_evaluation';truth=physical.offline_truth(task['case'],splits['train'].labels)
        evaluations={}
        for name,native,bank,seconds in [('original',original,original_frozen,fit_seconds),
                ('enriched',enriched,frozen,fit_seconds+(search.get('total_seconds') or 0.))]:
            evaluations[name]=validate._evaluate_identifier(dict(status='fitted',result=native,frozen=bank,seconds=seconds),splits,truth)
        row=dict(id=task['id'],status='complete',methods=evaluations,seconds=perf_counter()-began,
            original_fit_actual_seconds=fit_seconds,original_fit_budget_seconds=120.,extra_search=io.ref(directory/'search_60s.json'),
            extra_search_status=search['status'],candidate_enrichment=enriched['candidate_enrichment'],
            pq={name:validate.pq_diagnostics(obs) for name,obs in splits.items()},
            offline_truth_after_candidate_seal=True,offline_truth_QP_run=False,no_new_AC=True)
        io.write(directory/'result.json',row)
        files={name:io.ref(directory/name) for name in ('train.npz','validation.npz','new_day.npz','101_native.json',
            'original_101_native.json','search_60s.json','observations_sealed.json','candidates_sealed.json','result.json')}
        entry=dict(task,files=files,score=io.ref(directory/'score.json'),native_copy=files['101_native.json'],
            historical_native_index=enriched['selected']['index'],historical_p_map=score['p_map'],
            planned_history_target_native=enriched['selected']['index'],map_matches_native_topology=score['map_matches_native_topology'])
        completion=dict(status='complete',task=task,entry=entry,result=io.ref(directory/'result.json'),seconds=perf_counter()-began)
    except Exception as exc:
        row=dict(id=task['id'],status='failed',stage=stage,seconds=perf_counter()-began,error=repr(exc),traceback=traceback.format_exc())
        details={key:getattr(exc,key) for key in ('append_receipt','output_contraction','qp_records','last_point') if hasattr(exc,key)}
        if details:row['failure_details']=native_json_record(details)
        io.write(directory/'result.json',row)
        completion=dict(status='failed',task=task,result=io.ref(directory/'result.json'),stage=stage,error=repr(exc),seconds=perf_counter()-began)
    io.write(directory/'complete.json',completion)
    return completion


def run(out):
    out=Path(out).resolve();p=io.read(out/'protocol.json');verify_protocol(p)
    if (out/'complete.json').exists():
        result=io.read(out/'complete.json');checked(result['input_index']);return result
    lock=out/'lock.json'
    with lock.open('x',encoding='utf-8') as f:f.write('{"pid":'+str(os.getpid())+',"created_utc":"'+utc()+'"}')
    began=perf_counter();receipts=[]
    try:
        with ProcessPoolExecutor(max_workers=p['workers']) as pool:
            futures={pool.submit(fit_cell,task,str(out)):task for task in p['tasks']}
            for future in as_completed(futures):
                task=futures[future]
                try:receipt=future.result()
                except Exception as exc:
                    receipt=dict(status='worker_failed',task=task,error=repr(exc),traceback=traceback.format_exc())
                receipts.append(receipt)
                io.write(out/'progress.json',dict(planned_inputs=40,finished=len(receipts),receipts=receipts))
                print(f'{len(receipts)}/40 {task["id"]} {receipt["status"]}',flush=True)
        receipts.sort(key=lambda r:r['task']['grid_order'])
        entries=[r['entry'] for r in receipts if r['status']=='complete'];failures=[r for r in receipts if r['status']!='complete']
        changes=[]
        try:verify_protocol(p)
        except Exception as exc:changes.append(repr(exc))
        index=dict(phase=p['phase'],inputs=entries,attempts=receipts,failures=failures,protocol=io.ref(out/'protocol.json'),
            planned_inputs=40,input_count=len(entries),physical_worlds=2,history_repeats=4,pq_relative_sds=list(PQ_SDS),
            source_unchanged=not changes,no_new_AC=True,device_specs_fixed=True,forecast_fixed=True)
        io.write(out/'data_index.json',index)
        result=dict(status='complete' if not changes else 'source_changed',planned_inputs=40,prepared_inputs=len(entries),
            failed_inputs=len(failures),failures=failures,input_index=io.ref(out/'data_index.json'),seconds=perf_counter()-began,
            source_unchanged=not changes,changed_sources_or_inputs=changes,no_new_AC=True)
        io.write(out/'complete.json',result);return result
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--source-run',type=Path,default=SOURCE_RUN)
    parser.add_argument('--source-audit',type=Path,default=SOURCE_AUDIT)
    parser.add_argument('--workers',type=int,default=3)
    stage=parser.add_mutually_exclusive_group();stage.add_argument('--prepare-only',action='store_true');stage.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not args.resume:prepare(args.out,source_run=args.source_run,source_audit=args.source_audit,workers=args.workers)
    if not args.prepare_only:print(run(args.out))


if __name__=='__main__':main()
