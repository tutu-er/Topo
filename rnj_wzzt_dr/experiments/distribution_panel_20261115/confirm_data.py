"""Readiness-gated confirmation preparation; development source remains frozen.

The case body is copied verbatim from data.generate_one except its phase/path
and readiness guard. Tests compare both ASTs, while all algorithms stay shared.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import os
import shutil
import sys
from time import perf_counter
import traceback
import numpy as np

from . import data
from .data import (io,checked,utc,profiles,physical,validate,full_ac,fit_observed,
    SearchConfig,native_json_record,freeze_history,append_candidate,additional_search,
    pack_score,fingerprint,ROOT,SPLIT)


def check_readiness(reference):
    path=checked(reference) if isinstance(reference,dict) else Path(reference).resolve()
    readiness=io.read(path)
    if readiness.get('confirmation_ready') is not True:
        raise PermissionError('Development readiness has not authorized confirmation')
    refs=readiness.get('checked_refs')
    if isinstance(refs,dict):refs=list(refs.values())
    if not isinstance(refs,list) or not refs:
        raise ValueError('Readiness must include nonempty checked_refs')
    for ref in refs:checked(ref)
    return dict(receipt=io.ref(path),checked_refs=refs)


def prepare(source,out,readiness,workers=4):
    gate=check_readiness(readiness)
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve previous confirmation preparation')
    if workers<1 or workers>4:raise ValueError('Use1..4 workers')
    if shutil.disk_usage(out.parent).free<35*2**30:raise RuntimeError('Less than35GiB free workspace space')
    prior=io.read(source/'protocol.json');pre=io.read(source/'preflight.json');done=io.read(source/'development_complete.json')
    if done['status']!='complete' or not done['source_unchanged']:
        raise ValueError('Unchanged completed development data required')
    for ref in [*prior['sources'],prior['archive'],prior['source_plan'],prior['registration_metadata']]:checked(ref)
    tasks=[t for t in prior['tasks'] if t['phase']=='confirmation']
    if len(tasks)!=96 or len({t['id'] for t in tasks})!=96:
        raise ValueError('Use all96 prelocked confirmation inputs')
    rows=[r for r in pre['rows'] if r['phase']=='confirmation']
    if {r['id'] for r in rows}!={t['id'] for t in tasks}:raise ValueError('Preflight/task mismatch')
    paths={Path(r['path']).resolve() for r in prior['sources']}
    paths.add(Path(__file__).resolve())
    out.mkdir(parents=True)
    for path in sorted(paths):
        destination=out/'source_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    p=deepcopy(prior)
    p.update(created_utc=utc(),phase='confirmation_data',tasks=tasks,workers=workers,
        readiness=gate['receipt'],readiness_checked_refs=gate['checked_refs'],
        source_development_protocol=io.ref(source/'protocol.json'),source_preflight=io.ref(source/'preflight.json'),
        source_development_complete=io.ref(source/'development_complete.json'),
        allowed_execution_phase='confirmation',confirmation_execution_authorized=True,
        planned_inputs=96,preflight_complete_inputs=sum(r['status']=='data_complete' for r in rows),
        missing_inputs_retained=sum(r['status']!='data_complete' for r in rows),
        original_calendar_unchanged=True,no_development_refits=True,
        case_implementation='Copied frozen data.generate_one body except phase, output path and explicit readiness gate; shared numerical algorithms',
        sources=[io.ref(path) for path in sorted(paths)],environment=validate._environment(),
        **validate._git_receipt(sorted(paths)))
    io.write(out/'protocol.json',p)
    return p


def prepare_one(task,out):
    out=Path(out);p=io.read(out/'protocol.json')
    if task['phase']!='confirmation':raise PermissionError('This entry is confirmation-only')
    check_readiness(p['readiness'])
    directory=out/'confirmation'/task['id']
    if (directory/'complete.json').exists():return io.read(directory/'complete.json')
    if directory.exists():raise RuntimeError('Partial input needs review; do not reset fit budget: '+str(directory))
    directory.mkdir(parents=True);began=perf_counter();stage='data_profile'
    io.write(directory/'started.json',dict(task=task,pid=os.getpid(),created_utc=utc()))
    try:
        profile=profiles.build_case(task['case'],p['archive']['path'],task['start_date'],expected_sha256=p['archive']['sha256'])
        stage='background_AC';generated=physical.generate(task['case'],profile,seed=task['seed'],voltage_sigma=.0005,pq_sigma=.005,root_sigma=.003)
        validate._save_observations(directory,generated)
        io.write(directory/'physical_receipt.json',generated['diagnostics'])
        splits=generated['splits'];clean=generated['private']
        observation_refs={name:dict(ref=io.ref(directory/(name+'.npz')),sha256=fingerprint(obs)) for name,obs in splits.items()}
        io.write(directory/'observations_sealed.json',dict(observations=observation_refs,profile_metadata=profile['metadata'],created_utc=utc()))
        state,checks=full_ac.solve(clean['network'],clean['labels'],clean['p'],clean['q'],clean['root'])
        checks['max_sealed_voltage_difference_pu']=float(np.max(abs(state['terminal_voltage_pu']-clean['voltage'])))
        if checks['max_sealed_voltage_difference_pu']>1e-12:raise ArithmeticError('Full cache differs from original AC')
        state['timestamps_utc']=np.asarray(clean['timestamps']);np.savez_compressed(directory/'full_ac.npz',**state)
        cache=dict(case=task['case'],start_date=task['start_date'],points=384,original=io.ref(directory/'private_physical.npz'),
            cache=io.ref(directory/'full_ac.npz'),network_hash=generated['diagnostics']['original_network_sha256'],checks=checks)
        io.write(directory/'full_ac_receipt.json',cache)
        meta=profile['metadata']
        rc={key:deepcopy(meta[key]) for key in ('node_mapping','scales','resource_metadata_at_history_start','source_resource_metadata',
             'resource_metadata_origin','resource_metadata_note','source_archive_sha256','local_timezone','local_start','scale_fit_split','scaling') if key in meta}
        rc.update(case=task['case'],start_date=task['start_date'],existing_physical=cache['original'],
            PQ_bitwise_equal_to_executed_inputs=True,resource_metadata_known_at='history_start',
            extra_device_baseline_added=False,units=dict(P='kW',Q='kvar'))
        io.write(directory/'resource_context.json',rc)
        stage='original_101';tick=perf_counter()
        original=native_json_record(fit_observed(splits['train'],splits['validation'],method='101',config=SearchConfig(**p['method_config'])))
        original_seconds=perf_counter()-tick;io.write(directory/'original_101_native.json',original)
        original_frozen=freeze_history(original,splits['validation'])
        stage='additional_E60';search=additional_search(original,splits['train'],p['additional_E_pool_search_seconds'])
        io.write(directory/'search_60s.json',native_json_record(search))
        stage='freeze_combined_bank';enriched=append_candidate(original,search.get('closed_point'),splits['train'],splits['validation'])
        enriched['candidate_enrichment']=dict(original_native=io.ref(directory/'original_101_native.json'),
            source_search=io.ref(directory/'search_60s.json'),original_points_preserved=len(original['frozen_candidates']),
            added_points=int(search.get('closed_point') is not None),original_fit_actual_seconds=original_seconds,
            extra_search_status=search['status'],extra_search_actual_seconds=search.get('total_seconds'),
            original_runtime_fields_scope='Original30s fit only; extra search accounted separately',truth_used=False,new_day_values_used=False)
        io.write(directory/'101_native.json',native_json_record(enriched));score=pack_score(enriched,splits['train'],splits['validation'])
        io.write(directory/'score.json',score)
        frozen=freeze_history(enriched,splits['validation'])
        io.write(directory/'candidates_sealed.json',dict(created_utc=utc(),native=io.ref(directory/'101_native.json'),
            original_native=io.ref(directory/'original_101_native.json'),score=io.ref(directory/'score.json'),
            training=observation_refs['train'],validation=observation_refs['validation'],parameter_sha256=frozen.parameter_sha256,
            original_parameter_sha256=original_frozen.parameter_sha256,truth_used=False,new_day_values_used=False))
        stage='offline_evaluation'
        truth=physical.offline_truth(task['case'],splits['train'].labels)
        evaluate=lambda native,fro,seconds:validate._evaluate_identifier(dict(status='fitted',result=native,frozen=fro,seconds=seconds),splits,truth)
        evaluations=dict(original=evaluate(original,original_frozen,original_seconds),enriched=evaluate(enriched,frozen,original_seconds+(search.get('total_seconds') or 0.)))
        row=dict(**task,status='complete',seconds=perf_counter()-began,original_fit_actual_seconds=original_seconds,
            original_fit_budget_seconds=30.,extra_search=io.ref(directory/'search_60s.json'),extra_search_status=search['status'],
            candidate_enrichment=enriched['candidate_enrichment'],methods=evaluations,
            pq={name:validate.pq_diagnostics(obs) for name,obs in splits.items()},physical=generated['diagnostics'],
            full_AC_checks=checks,offline_truth_after_candidate_seal=True,offline_truth_QP_run=False)
        io.write(directory/'result.json',row)
        files={name:io.ref(directory/name) for name in ('train.npz','validation.npz','new_day.npz','101_native.json',
            'original_101_native.json','observations_sealed.json','candidates_sealed.json','result.json')}
        entry=dict(**task,files=files,score=io.ref(directory/'score.json'),native_copy=files['101_native.json'],
            resource_context=io.ref(directory/'resource_context.json'),physical_cache=cache,base_kw=clean['network']['base_kw'],
            historical_native_index=enriched['selected']['index'],historical_p_map=score['p_map'],
            map_matches_native_topology=score['map_matches_native_topology'])
        completion=dict(status='complete',task=task,entry=entry,result=io.ref(directory/'result.json'),seconds=perf_counter()-began)
    except Exception as exc:
        row=dict(**task,status='failed',stage=stage,seconds=perf_counter()-began,error=repr(exc),traceback=traceback.format_exc())
        io.write(directory/'result.json',row);completion=dict(status='failed',task=task,result=io.ref(directory/'result.json'),error=row['error'],stage=stage)
    io.write(directory/'complete.json',completion);return completion


def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    check_readiness(p['readiness'])
    for ref in [*p['sources'],p['archive'],p['registration_metadata'],p['source_plan'],p['source_preflight'],p['source_development_complete']]:checked(ref)
    if (out/'confirmation_complete.json').exists():return io.read(out/'confirmation_complete.json')
    if (out/'lock.json').exists():raise RuntimeError('Inspect existing confirmation lock before resuming')
    pre=io.read(checked(p['source_preflight']));tasks={t['id']:t for t in p['tasks']}
    rows=[r for r in pre['rows'] if r['phase']=='confirmation']
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=utc(),phase='confirmation'))
    receipts=[dict(status='data_incomplete',task=tasks[r['id']],error='Prelocked window has missing/nonfinite source bins')
        for r in rows if r['status']!='data_complete']
    try:
        with ProcessPoolExecutor(max_workers=p['workers']) as pool:
            jobs={pool.submit(prepare_one,tasks[r['id']],str(out)):tasks[r['id']] for r in rows if r['status']=='data_complete'}
            for future in as_completed(jobs):
                task=jobs[future]
                try:receipt=future.result()
                except Exception as exc:receipt=dict(status='worker_failed',task=task,error=repr(exc),traceback=traceback.format_exc())
                receipts.append(receipt);io.write(out/'confirmation_progress.json',dict(planned_inputs=96,finished=len(receipts),receipts=receipts))
                print(str(len(receipts))+'/96 '+task['id']+' '+receipt['status'],flush=True)
        receipts.sort(key=lambda r:r['task']['index']);entries=[r['entry'] for r in receipts if r['status']=='complete']
        io.write(out/'confirmation_resource_context.json',dict(rows=[io.read(e['resource_context']['path']) for e in entries],
            inputs=[e['resource_context'] for e in entries],rule='History-only resource registration and train-only scales'))
        io.write(out/'confirmation_AC_index.json',dict(status='complete',rows=[e['physical_cache'] for e in entries],total_time_points=384*len(entries)))
        index=dict(phase='confirmation',inputs=entries,input_count=len(entries),planned_inputs=96,attempts=receipts,
            failures=[r for r in receipts if r['status']!='complete'],resource_context=io.ref(out/'confirmation_resource_context.json'),
            AC_index=io.ref(out/'confirmation_AC_index.json'),preflight=p['source_preflight'],protocol=io.ref(out/'protocol.json'),
            archive=p['archive'],split=SPLIT,readiness=p['readiness'],calendar_unchanged=True)
        io.write(out/'confirmation_index.json',index)
        done=dict(status='complete',planned_inputs=96,prepared_inputs=len(entries),failed_inputs=96-len(entries),
            input_index=io.ref(out/'confirmation_index.json'),readiness=p['readiness'],
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources']))
        io.write(out/'confirmation_complete.json',done);return done
    finally:(out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('outputs/distribution_panel_data_20261115_v1'))
    parser.add_argument('--out',type=Path,default=Path('outputs/distribution_panel_confirmation_data_20261115_v1'))
    parser.add_argument('--readiness',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve()
    if not args.resume:prepare(args.source,args.out,args.readiness,args.workers)
    else:
        p=io.read(args.out/'protocol.json')
        if io.ref(args.readiness)!=p['readiness']:raise ValueError('Resume requires the identical locked readiness receipt')
    if not args.prepare_only:print(run(args.out))


if __name__=='__main__':main()
