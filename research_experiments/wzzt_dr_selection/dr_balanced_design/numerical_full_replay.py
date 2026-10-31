"""Two prelocked planning-observation full fits with explicit LP Method=2.

This is a numerical configuration replay, never a new DR test or price choice.
prepare locks old observations and current source hashes; run makes exactly two
attempts with the existing fit_observed; score reads old evaluator truth only
after both attempts and their output hashes have been frozen. No physical data
producer, new future, AC execution or original experiment overwrite is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    path=Path(path);part=path.with_suffix(path.suffix+'.partial')
    part.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    part.replace(path)


def source_setup(repo):
    sys.path.insert(0,str(repo))
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    return source_hashes


def complete_original_barrier(root,cfg):
    # Presence only; no new test or truth contents are inspected.
    expected=[root/'aggregate.json',root/'oracle_selected.json',root/'oracle_selected.sha256.json']
    for case in cfg['cases']:
        for phase in ('planning','test'):
            folder=root/case/phase
            expected += [folder/'phase_ended.json',folder/'fits_frozen.json']
            if (folder/'phase.lock').exists():raise RuntimeError('original physical/fit phase still active')
    if any(not p.exists() for p in expected):raise RuntimeError('original 64-condition batch must finish first')


def training_and_validation(selected):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,input_hash
    with Path(selected['history_path']).open('rb') as stream:history=pickle.load(stream)['history']
    with Path(selected['observed_packet_path']).open('rb') as stream:packet=pickle.load(stream)
    if packet['metadata'].get('observed_only') is not True:raise RuntimeError('packet is not observed-only')
    train=history['training']+packet['training'];valid=history['validation']+packet['validation']
    hashes={}
    for split,scenes in (('training',train),('validation',valid)):
        obs,days,_=normalize(scenes,split)
        if len(obs[0])!=480 or list(obs[0].columns)!=selected['terminal_labels']:
            raise RuntimeError('replay does not preserve full 480/480 terminal observations')
        hashes[split]=input_hash(obs,days)
    if hashes!=selected['input_sha256']:raise RuntimeError('reconstructed observations differ from failed original fit')
    return train,valid


def prepare(root):
    root=Path(root).resolve();cfg=read(root/'protocol.json')
    complete_original_barrier(root,cfg)
    probe_path=root/'lp_method_probe_v1'/'protocol.json'
    probe=read(probe_path)
    if sha(probe_path)!=read(probe_path.parent/'protocol.sha256.json')['sha256']:
        raise RuntimeError('original two-observation probe lock changed')
    if len(probe['selected'])!=2 or [v['record_id'] for v in probe['selected']]!=['single__path1','opposed__path1']:
        raise RuntimeError('only the two prelocked planning observations are allowed')
    out=root/'numerical_full_replay_v1';out.mkdir(exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('numerical replay protocol already frozen')
    sources=source_setup(cfg['repository'])()
    inputs={str(root/'protocol.json'):sha(root/'protocol.json'),str(probe_path):sha(probe_path),str(Path(__file__).resolve()):sha(__file__)}
    selected=[]
    for old in probe['selected']:
        result_path=Path(old['failed_result_path'])
        original=read(result_path)
        if sha(result_path)!=probe['input_sha256'][str(result_path)]:raise RuntimeError('original failed fit changed')
        record_id=old['record_id'];case=old['case']
        location=root/case/'planning'/'records'/record_id
        receipt_path=location/'physical'/'events_receipt.json'
        receipt=read(receipt_path)
        truth_path=receipt_path.parent/'truth_evaluation.json'
        item=dict(record_id=record_id,case=case,phase='planning',policy=old['policy'],response_path=old['response_path'],
            history_path=old['history_path'],observed_packet_path=old['observed_packet_path'],
            terminal_labels=original['terminal_labels'],root_label=original['root_label'],
            input_sha256=original['input_sha256'],original_failed_result_path=str(result_path),
            original_failure=old['original_failure'],original_record_status=old['original_record_status'],
            actual_payment_reused=receipt.get('actual_payment'),committed_payment_reused=receipt.get('committed_payment'),
            operation_checks_pass=receipt.get('all_operation_checks_pass') is True and receipt.get('quote_qualification_pass') is True,
            old_truth_path_for_postfreeze_scoring=str(truth_path),old_truth_sha256_for_postfreeze_scoring=receipt['private_files']['truth_evaluation.json']['sha256'])
        for path in (Path(old['history_path']),Path(old['observed_packet_path']),result_path,receipt_path):
            expected=probe['input_sha256'].get(str(path))
            if expected and sha(path)!=expected:raise RuntimeError('old observation input changed')
            inputs[str(path)]=sha(path)
        training_and_validation(item)
        identity=dict(input_sha256=item['input_sha256'],source_sha256=sources,root_label=item['root_label'],search_seconds=25.,lp_method=2)
        item['cache_key']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        item['new_cache_directory']=str(out/'fit_cache'/item['cache_key'])
        selected.append(item)
    score_source=Path(cfg['repository'])/'research_experiments/wzzt_dr_selection/dr_balanced_design/budget_reference_diagnostic.py'
    inputs[str(score_source)]=sha(score_source)
    test_receipt=Path(__file__).parent/'lp_method_test_topo_v2_receipt.json'
    if read(test_receipt)['status']!='passed':raise RuntimeError('parameter unit checks have not passed')
    inputs[str(test_receipt)]=sha(test_receipt)
    p=dict(id='numerical_full_replay_v1',status='frozen',frozen_utc=now(),script_sha256=sha(__file__),
        repository=cfg['repository'],executable=sys.executable,main_protocol_sha256=sha(root/'protocol.json'),
        original_algorithm_source_sha256=cfg['algorithm_source_sha256'],new_algorithm_source_sha256=sources,
        source_change='explicit lp_method parameter; default1 unchanged; this isolated replay selects2',
        settings=dict(lp_method=2,actual_lp_method_required=2,per_path_seconds=25.,LP_time_limit_seconds=15.,lp_threads=1,
            feasibility_tolerance=1e-9,optimality_tolerance=1e-9,coefficient_bound=2.,
            max_extra_starts=2,relative_gain_threshold=.001,final_validation_fraction=.03,resolution_factor=.16),
        planned_full_identifier_attempts=2,selected=selected,input_sha256=inputs,
        new_future_or_physical_events=0,AC_calls=0,price_selection_calls=0,
        frozen_truth_not_read_during_preparation=True,
        truth_gate='score only after all two full fit attempts and result hashes are frozen',
        original_results_or_oracle_may_be_modified=False,
        same_observations_new_numerical_configuration=True,new_independent_samples=0,
        interpretation='numerical replay of two preselected Method1 failures; no DR price-effect estimate')
    write(out/'protocol.json',p)
    write(out/'protocol.sha256.json',dict(sha256=sha(out/'protocol.json')))
    return dict(status='prepared',protocol=str(out/'protocol.json'),sha256=sha(out/'protocol.json'),records=[r['record_id'] for r in selected],full_fits_run=0,truth_files_read=0)


def check(root):
    root=Path(root).resolve();out=root/'numerical_full_replay_v1';p=read(out/'protocol.json')
    if sha(out/'protocol.json')!=read(out/'protocol.sha256.json')['sha256'] or sha(__file__)!=p['script_sha256']:
        raise RuntimeError('numerical replay protocol/script changed')
    cfg=read(root/'protocol.json')
    complete_original_barrier(root,cfg)
    if sha(root/'protocol.json')!=p['main_protocol_sha256']:raise RuntimeError('original protocol changed')
    if len(p['selected'])!=2 or p['planned_full_identifier_attempts']!=2:raise RuntimeError('replay size changed')
    for path,expected in p['input_sha256'].items():
        if sha(path)!=expected:raise RuntimeError('frozen replay input changed: '+path)
    if source_setup(p['repository'])()!=p['new_algorithm_source_sha256']:
        raise RuntimeError('actual algorithm source closure changed')
    return root,out,p


def run(root):
    root,out,p=check(root)
    if (out/'fits_frozen.json').exists() or (out/'progress.json').exists():
        raise FileExistsError('existing numerical attempts retained; never rerun or extend implicitly')
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import fit_observed
    rows=[];started=time.perf_counter();lock=out/'replay.lock'
    with lock.open('x',encoding='utf-8') as stream:json.dump(dict(started_utc=now(),protocol_sha256=sha(out/'protocol.json')),stream)
    try:
        for selected in p['selected']:
            record=dict(record_id=selected['record_id'],status='started',actual_payment_reused=selected['actual_payment_reused'],
                        original_failure_retained=selected['original_failure'],started_utc=now())
            rows.append(record)
            write(out/'progress.json',dict(protocol_sha256=sha(out/'protocol.json'),planned=2,rows=rows))
            then=time.perf_counter();fit_dir=Path(selected['new_cache_directory'])
            try:
                check(root)
                train,valid=training_and_validation(selected)
                fit=fit_observed(train,valid,fit_dir,root_label=selected['root_label'],search_seconds=25.,lp_method=2)
                if fit['lp_method']!=2 or fit['settings']['lp_method']!=2 or fit['settings']['actual_lp_method']!=2:
                    raise RuntimeError('actual replay solver method differs from protocol')
                if any(v.get('lp_method')!=2 for v in fit['LP_solves']):raise RuntimeError('some LP used another method')
                record.update(status='complete',search_stops=fit['search_stop_statuses'],LP_calls=len(fit['LP_solves']),
                    selected_origins={k:v.get('selected_origin',v.get('origin')) for k,v in fit['models'].items()},
                    actual_lp_method=fit['settings']['actual_lp_method'])
            except Exception as exc:
                record.update(status='failed',exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            finally:
                record['wall_seconds']=time.perf_counter()-then
                for name in ('result.json','checkpoint.json'):
                    path=fit_dir/name
                    if path.exists():record[name.replace('.json','')+'_path']=str(path);record[name.replace('.json','')+'_sha256']=sha(path)
                record['ended_utc']=now()
                write(out/'progress.json',dict(protocol_sha256=sha(out/'protocol.json'),planned=2,rows=rows))
                print(json.dumps(dict(record_id=record['record_id'],status=record['status'],LP_calls=record.get('LP_calls'),wall_seconds=record['wall_seconds']),ensure_ascii=True),flush=True)
        frozen=dict(status='all_attempts_frozen',frozen_utc=now(),protocol_sha256=sha(out/'protocol.json'),rows=rows,
            planned=2,completed=sum(r['status']=='complete' for r in rows),wall_seconds=time.perf_counter()-started,
            truth_files_read=0,new_physical_events=0,original_records_modified=0,oracle_modified=0)
        write(out/'fits_frozen.json',frozen)
        write(out/'fits_frozen.sha256.json',dict(sha256=sha(out/'fits_frozen.json')))
        return dict(status='all_attempts_frozen',planned=2,completed=frozen['completed'],truth_files_read=0,output=str(out/'fits_frozen.json'))
    finally:lock.unlink(missing_ok=True)


def score(root):
    root,out,p=check(root)
    if (out/'score.json').exists():raise FileExistsError('scored supplement retained')
    frozen_path=out/'fits_frozen.json'
    if sha(frozen_path)!=read(out/'fits_frozen.sha256.json')['sha256']:raise RuntimeError('frozen fit output changed')
    frozen=read(frozen_path)
    if frozen['protocol_sha256']!=sha(out/'protocol.json') or frozen['planned']!=2 or len(frozen['rows'])!=2:
        raise RuntimeError('both numerical fit attempts must freeze before evaluator truth')
    from research_experiments.wzzt_dr_selection.dr_balanced_design.budget_reference_diagnostic import score_fit
    by={r['record_id']:r for r in frozen['rows']};rows=[]
    for selected in p['selected']:
        original=by[selected['record_id']]
        record=dict(record_id=selected['record_id'],status=original['status'],
                    actual_payment_reused=selected['actual_payment_reused'],original_joint_failure_retained=True)
        rows.append(record)
        if original['status']!='complete':continue
        result_path=Path(original['result_path'])
        if sha(result_path)!=original['result_sha256']:raise RuntimeError('fit result changed after freeze')
        truth_path=Path(selected['old_truth_path_for_postfreeze_scoring'])
        if sha(truth_path)!=selected['old_truth_sha256_for_postfreeze_scoring']:raise RuntimeError('old evaluator truth changed')
        truth=read(truth_path)['true_supports']
        record.update(models=score_fit(read(result_path),truth,selected['root_label']),
                      operation_checks_pass=selected['operation_checks_pass'],truth_sha256=sha(truth_path),
                      status='evaluated' if selected['operation_checks_pass'] else 'evaluated_operation_failed')
    result=dict(status='scored_numerical_replay',protocol_sha256=sha(out/'protocol.json'),fits_frozen_sha256=sha(frozen_path),
        rows=rows,planned=2,evaluated=sum(r['status']=='evaluated' for r in rows),
        new_independent_samples=0,price_selection_calls=0,new_physical_events=0,
        original_results_modified=0,oracle_modified=0,
        interpretation='same two planning observations under a declared numerical configuration; no new DR strategy result')
    write(out/'score.json',result)
    return dict(status='scored',report=str(out/'score.json'),evaluated=result['evaluated'],planned=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','score'));parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(globals()[args.command](args.root),ensure_ascii=True),flush=True)
