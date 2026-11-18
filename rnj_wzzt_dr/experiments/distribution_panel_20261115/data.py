"""Fixed-calendar P/Q preflight and development-only frozen input preparation."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone,timedelta
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from ... import io
from ...history import fit_observed,freeze_history
from ...observations import fingerprint
from ...identification.algorithm import SearchConfig,_output
from ...identification.competition import competition
from ...identification.optimization import errors
from ..measured_pq_20261114 import profiles,physical,validate
from ..measured_pq_flex_20261114 import full_ac,pool_diagnostic
from ..flexibility_sweep_20261113.fitting import native_json_record,pack_score

ROOT=Path(__file__).resolve().parents[3]
CASES=('paper15','flynn16','pengwah18','soumalas11')
COUNTS=dict(paper15=15,flynn16=16,pengwah18=18,soumalas11=11)
DEVELOPMENT=('2024-02-05','2024-05-20','2024-08-05','2024-11-04')
CONFIRMATION=dict(
    winter=('2024-01-04','2024-02-12','2024-02-22','2024-12-02','2024-12-12','2024-12-19'),
    spring=('2024-03-04','2024-03-14','2024-03-25','2024-04-04','2024-05-13','2024-05-27'),
    summer=('2024-06-03','2024-06-13','2024-06-24','2024-07-04','2024-08-12','2024-08-22'),
    autumn=('2024-09-02','2024-09-12','2024-09-23','2024-10-03','2024-11-11','2024-11-21'))
EXPOSED=(('2024-01-15','2024-01-25'),('2024-04-15','2024-05-13'),
         ('2024-07-15','2024-07-19'),('2024-10-14','2024-10-18'))
SPLIT=profiles.SPLIT
SEEDS=dict(development=2026111501,confirmation=2026111502)


def utc():return datetime.now(timezone.utc).isoformat()


def checked(ref):
    if io.ref(ref['path'])['sha256']!=ref['sha256']:raise ValueError('Changed input/source: '+ref['path'])
    return Path(ref['path'])


def calendar():
    dates=[('development',s,d) for s,d in zip(CONFIRMATION,DEVELOPMENT)]
    dates += [('confirmation',s,d) for s,ds in CONFIRMATION.items() for d in ds]
    windows=[];tasks=[];tz=ZoneInfo('Europe/Berlin')
    local=lambda d:datetime.fromisoformat(d+'T08:00:00').replace(tzinfo=tz)
    for phase,season,date in dates:
        start=local(date);stop=start+timedelta(days=4)
        if start.utcoffset()!=stop.utcoffset():raise ValueError('Calendar crosses DST')
        if any(start<b and a<stop for a,b in windows):raise ValueError('Calendar windows overlap')
        if any(start<local(b) and local(a)<stop for a,b in EXPOSED):raise ValueError('Previously exposed calendar interval')
        windows.append((start,stop))
        for case in CASES:
            index=len(tasks)
            tasks.append(dict(id=case+'__'+date,index=index,original_input_index=index,phase=phase,season=season,
                case=case,start_date=date,seed=SEEDS[phase]+index,terminal_count=COUNTS[case],
                local_start=start.isoformat(),local_stop=stop.isoformat(),
                execution_start=(start+timedelta(days=3)).isoformat(),timezone='Europe/Berlin'))
    return tasks


def preflight_rows(tasks,times,finite,valid_minutes,feeder_ids):
    """Only boolean finiteness, timestamps and count metadata cross this API."""
    times=pd.DatetimeIndex(pd.to_datetime(times,utc=True))
    finite=np.asarray(finite);counts=np.asarray(valid_minutes)
    if finite.dtype!=np.bool_:raise TypeError('Pass finite flags, never numerical P/Q')
    if times.has_duplicates or times.isna().any():raise ValueError('Invalid archive time axis')
    if finite.shape!=counts.shape or finite.shape!=(len(times),len(feeder_ids)):
        raise ValueError('Archive coverage dimensions differ')
    if list(map(int,feeder_ids[:18]))!=list(range(1,19)):raise ValueError('Expected predeclared feeders1..18')
    rows=[]
    for task in tasks:
        start=pd.Timestamp(task['local_start']);want=pd.date_range(start.tz_convert('UTC'),periods=384,freq='15min')
        ix=times.get_indexer(want);n=task['terminal_count'];bad=np.ones((384,n),bool)
        found=ix>=0;bad[found]=(counts[ix[found],:n]<12)|~finite[ix[found],:n]
        split_bad={key:int(bad[a:b].sum()) for key,(a,b) in SPLIT.items()}
        rows.append(dict(**task,status='data_complete' if not bad.any() else 'data_incomplete',
            missing_timestamp_rows=int((~found).sum()),incomplete_node_time_bins=int(bad.sum()),
            incomplete_bins_by_split=split_bad,future_day_complete=split_bad['new_day']==0,
            qualification_uses='timestamps, P/Q finite flags and original minute counts only',
            replacement_date=None))
    return rows


def prepare(archive,out,workers=4):
    archive,out=Path(archive).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve existing prepared data root')
    if workers<1 or workers>4:raise ValueError('Use1..4 single-thread workers')
    if shutil.disk_usage(out.parent).free<35*2**30:raise RuntimeError('Less than35GiB free workspace space')
    source_manifest=io.read(archive.with_name('sources.json'))
    checked(source_manifest['metadata'])
    archive_receipt=io.read(archive.with_name('archive.json'))
    if io.ref(archive)['sha256']!=archive_receipt['archive']['sha256']:raise ValueError('Archive differs from existing receipt')
    package=ROOT/'rnj_wzzt_dr'
    paths={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and str(m.__file__).endswith('.py') and Path(m.__file__).resolve().is_relative_to(package)}
    paths.update((Path(__file__).resolve(),Path(full_ac.backend.__file__).resolve()))
    paths.update(full_ac.backend.ROOT/'cases'/case/name for case in CASES for name in ('metadata.json','buses.csv','branches.csv'))
    plan=Path(__file__).parents[1]/'distribution_gate_20261114'/'PAPER_TEST_PLAN.md'
    sources=[io.ref(p) for p in sorted(paths)];out.mkdir(parents=True)
    for path in sorted(paths):
        destination=out/'source_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    config=validate._config(30.)
    protocol=dict(created_utc=utc(),source_plan=io.ref(plan),sources=sources,archive=io.ref(archive),
        archive_receipt=io.ref(archive.with_name('archive.json')),source_manifest=io.ref(archive.with_name('sources.json')),
        registration_metadata=source_manifest['metadata'],tasks=calendar(),workers=workers,
        environment=validate._environment(),**validate._git_receipt(sorted(paths)),
        method='101',method_budget_seconds=30.,method_config=asdict(config),additional_E_pool_search_seconds=60.,
        extra_E_rule='Last recorded original E pool/keep/incumbent only; train-only independent60s search then original fixed-support QP; all original frozen points retained',
        missing_E_rule='No substitute pool: retain original bank and record no recorded E input',
        append_rule='Append returned closed_point if any; no truth/validation/new-day driven search or filtering; original strict validation final selection',
        extra_E_timing='60s original search deadline, post-search fixed-support QP and constructor time separately included in actual total',
        split=SPLIT,seeds=SEEDS,noise=dict(PQ_relative_sd=.005,V_relative_sd=.0005,root_physical_sigma=.003),
        preflight_scope='All112 fixed windows, including future finite flags; no confirmation magnitudes/statistics/AC/fitting',
        allowed_execution_phase='development',confirmation_execution_authorized=False,
        no_truth_QP=True,no_new_day_identifier_inputs=True,device_baseline_added=False,
        failure_rule='Keep missing dates, fit failures, timeouts and no-added-point outcomes; never replace dates or reset a partial task budget',
        interpretation='Measured net source-feeder curves scaled from history only and applied to four original synthetic AC networks')
    io.write(out/'protocol.json',protocol)
    return protocol


def preflight(out):
    out=Path(out);p=io.read(out/'protocol.json');dest=out/'preflight.json'
    if dest.exists():return io.read(dest)
    for ref in [*p['sources'],p['archive'],p['source_manifest'],p['archive_receipt'],p['registration_metadata'],p['source_plan']]:checked(ref)
    began=perf_counter()
    with np.load(p['archive']['path'],allow_pickle=False) as z:
        # Numeric source arrays are immediately reduced to finite flags; no
        # confirmation values, voltage or numerical shape statistics are saved.
        finite=np.isfinite(z['p_kw'])&np.isfinite(z['q_kvar'])
        times=z['timestamps_utc'];counts=z['valid_minutes'];ids=z['feeder_ids']
    rows=preflight_rows(p['tasks'],times,finite,counts,ids)
    result=dict(created_utc=utc(),status='complete',protocol=io.ref(out/'protocol.json'),archive=p['archive'],
        rows=rows,planned_inputs=len(rows),complete_by_phase={phase:sum(r['phase']==phase and r['status']=='data_complete' for r in rows)
            for phase in SEEDS},seconds=perf_counter()-began,confirmation_voltage_generated=False,
        confirmation_fits=0,confirmation_future_magnitudes_analyzed=False,
        no_date_replacement=True,source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources']))
    io.write(dest,result);return result


def append_candidate(original,point,train,validation):
    """Finalize only the extra training point; retain the old bank verbatim.

    Missing internal_floor in a saved configuration means the historical zero
    floor, not whatever default a newer SearchConfig happens to use. Appending
    records its own fitting/verification cost, separate from the original fit.
    """
    enriched=deepcopy(original)
    if point is None:return enriched
    began=perf_counter();stage='check_original_bank'
    old=original['frozen_candidates'];saved_config=dict(original['config'])
    saved_config.setdefault('internal_floor',0.)
    config=SearchConfig(**saved_config)
    receipt=dict(status='running',source_stage='original_E_pool_independent_60s',
        appended_index=len(old),original_candidate_count=len(old),internal_floor=float(config.internal_floor),
        legacy_missing_floor_means_zero='internal_floor' not in original['config'],
        raw_point=deepcopy(point),QP=[],floor_seconds=0.,
        timing_scope='Additional append finalization and verification only; original fit/search timing is preserved')
    try:
        before=freeze_history(original,validation)
        stage='check_extra_training_point'
        if not np.isclose(errors(train,point)['mse'],point['mse'],rtol=1e-10,atol=1e-18):
            raise ArithmeticError('Extra point training MSE mismatch')
        final=point
        if config.internal_floor>0.:
            from ...identification import output_floor
            from ...identification.optimization import QPFitter
            stage='contract_extra_point'
            previous=original.get('output_contraction')
            if not isinstance(previous,dict) or previous.get('status')!='complete':
                raise ValueError('Positive-floor native bank requires its complete original contraction receipt')
            if previous.get('candidate_count')!=len(old) or len(previous.get('candidates',[]))!=len(old):
                raise ValueError('Original contraction candidate denominator is inconsistent')
            if float(previous['threshold'])!=config.internal_floor:
                raise ValueError('Saved contraction threshold differs from the native configuration')
            fitter=QPFitter(train,reference_mse=original['reference_mse'],bound=config.response_bound,
                            qp_seconds=config.qp_seconds)
            tick=perf_counter()
            try:
                points,single=output_floor.contract_bank([point],fitter,config.internal_floor,
                    previous['calibration'])
            finally:
                receipt['floor_seconds']=perf_counter()-tick
                receipt['QP']=deepcopy(fitter.records)
            if len(points)!=1 or single.get('candidate_count')!=1 or len(single.get('candidates',[]))!=1:
                raise ValueError('Single-candidate contraction returned an invalid denominator')
            final=points[0];contraction=deepcopy(single['candidates'][0]);contraction['index']=len(old)
            merged=deepcopy(previous)
            merged['candidates'].append(contraction);merged['candidate_count']=len(old)+1
            merged['completed_candidates']=len(old)+1
            # Keep the initial fit's elapsed time and all prefix receipts intact.
            merged['append_seconds']=float(previous.get('append_seconds',0.))+receipt['floor_seconds']
            merged['append_QP']=[*deepcopy(previous.get('append_QP',[])),*deepcopy(fitter.records)]
            merged['append_lineage']=[*deepcopy(previous.get('append_lineage',[])),
                dict(index=len(old),source_stage=receipt['source_stage'],seconds=receipt['floor_seconds'],
                     original_parameter_sha256=contraction['original_parameter_sha256'],
                     final_parameter_sha256=contraction['final_parameter_sha256'])]
            enriched['output_contraction']=merged
            receipt['contraction']=deepcopy(contraction)
        stage='strict_validation_selection'
        selected,mae,candidates=_output([r['point'] for r in old]+[final],train,validation,original['reference_mse'],config)
        added=candidates[-1];added['source_stage']='original_E_pool_independent_60s'
        enriched['frozen_candidates'].append(added)
        enriched.update(selected=selected,selected_mse_view=deepcopy(selected),selected_mae_view=mae)
        enriched['competition']=competition(enriched,validation)
        enriched.setdefault('handoff',{})['parameter_rule']=(
            'Original frozen candidate prefix retained verbatim; extra training-only E60 point finalized '
            'with the same frozen non-singleton output floor before strict validation; no new-day refit'
            if config.internal_floor>0. else
            'Original accepted path retained verbatim plus a training-only original-E-pool60s point; zero output floor; no new-day refit')
        if enriched['frozen_candidates'][:-1]!=old:raise ArithmeticError('Original candidate prefix changed')
        if config.internal_floor>0. and enriched['output_contraction']['candidates'][:-1]!=original['output_contraction']['candidates']:
            raise ArithmeticError('Original contraction receipt prefix changed')
        stage='freeze_enriched_bank'
        after=freeze_history(enriched,validation)
        if not np.array_equal(before.H,after.H[:len(old)]) or not np.array_equal(before.sse,after.sse[:len(old)]):
            raise ArithmeticError('Frozen original matrices/history SSE changed')
        receipt.update(status='complete',final_candidate_count=len(old)+1,seconds=perf_counter()-began,
                       prefix_parameters_unchanged=True,prefix_history_SSE_unchanged=True,
                       validation_used_for_refitting=False,new_day_used=False)
        enriched['candidate_append_records']=[*deepcopy(original.get('candidate_append_records',[])),receipt]
        return enriched
    except Exception as exc:
        receipt.update(status='failed',stage=stage,error_type=type(exc).__name__,error=str(exc),
                       seconds=perf_counter()-began,original_bank_preserved=True)
        if hasattr(exc,'receipt'):receipt['contraction_failure']=deepcopy(exc.receipt)
        error=RuntimeError('Extra candidate append failed at '+stage+': '+str(exc))
        error.append_receipt=receipt
        raise error from exc


def additional_search(original,train,seconds=60.):
    began=perf_counter()
    events=[r for r in original['events'] if r.get('stage')=='E' and all(k in r for k in ('pool','keep','point'))]
    if not events:return dict(status='not_run_no_recorded_E_pool',budget_seconds=seconds,closed_point=None,
        total_seconds=0.,truth_used=False,validation_used=False,new_day_used=False)
    event=events[-1];saved_config=dict(original['config'])
    saved_config.setdefault('internal_floor',0.)
    config=SearchConfig(**saved_config)
    try:
        result=pool_diagnostic.run_search(train,event['pool'],event['keep'],event['point'],config,original['reference_mse'],seconds)
        point=result['closed_point']
        if point is not None and not set(map(tuple,point['supports']))<=set(map(tuple,event['pool'])):
            raise ArithmeticError('Extra search left the original E pool')
        result.update(status='returned_point' if point is not None else 'no_returned_point',truth_used=False,validation_used=False,new_day_used=False,
            original_E_pool=event['pool'],original_E_keep=event['keep'],original_E_warm=event['point'])
        return result
    except Exception as exc:
        return dict(status='search_exception_original_bank_retained',budget_seconds=seconds,closed_point=None,
            total_seconds=perf_counter()-began,error=repr(exc),traceback=traceback.format_exc(),
            truth_used=False,validation_used=False,new_day_used=False)


def generate_one(task,out):
    out=Path(out);p=io.read(out/'protocol.json')
    if task['phase']!='development':raise PermissionError('Confirmation generation is not authorized by this data stage')
    directory=out/'development'/task['id']
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
        details={key:getattr(exc,key) for key in ('append_receipt','output_contraction','qp_records','last_point') if hasattr(exc,key)}
        if details:row['failure_details']=native_json_record(details)
        io.write(directory/'result.json',row);completion=dict(status='failed',task=task,result=io.ref(directory/'result.json'),error=row['error'],stage=stage)
    io.write(directory/'complete.json',completion);return completion


def run_development(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in [*p['sources'],p['archive'],p['registration_metadata'],p['source_plan']]:checked(ref)
    if (out/'development_complete.json').exists():return io.read(out/'development_complete.json')
    if (out/'lock.json').exists():raise RuntimeError('Inspect existing data preparation lock before resuming')
    pre=preflight(out);tasks=[r for r in pre['rows'] if r['phase']=='development']
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=utc(),phase='development'))
    receipts=[dict(status='data_incomplete',task=r,error='Fixed window has missing/nonfinite source bins') for r in tasks if r['status']!='data_complete']
    try:
        with ProcessPoolExecutor(max_workers=p['workers']) as pool:
            jobs={pool.submit(generate_one,{k:v for k,v in row.items() if k in p['tasks'][0]},str(out)):row for row in tasks if row['status']=='data_complete'}
            for future in as_completed(jobs):
                task=jobs[future]
                try:receipt=future.result()
                except Exception as exc:receipt=dict(status='worker_failed',task=task,error=repr(exc),traceback=traceback.format_exc())
                receipts.append(receipt);io.write(out/'development_progress.json',dict(planned_inputs=16,finished=len(receipts),receipts=receipts))
                print(str(len(receipts))+'/16 '+task['id']+' '+receipt['status'],flush=True)
        receipts.sort(key=lambda r:r['task']['index']);entries=[r['entry'] for r in receipts if r['status']=='complete']
        rc=dict(rows=[io.read(e['resource_context']['path']) for e in entries],inputs=[e['resource_context'] for e in entries],
            rule='Individual train-derived scales and registrations known at each history start; no new device baseline')
        io.write(out/'development_resource_context.json',rc)
        io.write(out/'development_AC_index.json',dict(status='complete',rows=[e['physical_cache'] for e in entries],total_time_points=384*len(entries)))
        index=dict(phase='development',inputs=entries,input_count=len(entries),planned_inputs=16,attempts=receipts,
            failures=[r for r in receipts if r['status']!='complete'],resource_context=io.ref(out/'development_resource_context.json'),
            AC_index=io.ref(out/'development_AC_index.json'),preflight=io.ref(out/'preflight.json'),protocol=io.ref(out/'protocol.json'),
            archive=p['archive'],split=SPLIT,no_confirmation_fits_or_AC=True)
        io.write(out/'development_index.json',index)
        done=dict(status='complete',planned_inputs=16,prepared_inputs=len(entries),failed_inputs=16-len(entries),
            input_index=io.ref(out/'development_index.json'),confirmation_preflight_only=True,
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources']))
        io.write(out/'development_complete.json',done);return done
    finally:(out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('data/measured_pq_sources_20261114/paired_pq_15min.npz'))
    parser.add_argument('--out',type=Path,default=Path('outputs/distribution_panel_data_20261115_v1'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve()
    if not args.resume:prepare(args.archive,args.out,args.workers)
    if args.prepare_only:return
    result=preflight(args.out);print(result['complete_by_phase'],flush=True)
    if not args.preflight_only:print(run_development(args.out),flush=True)


if __name__=='__main__':main()
