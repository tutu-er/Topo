"""Prepare/run four isolated fixed-family LP numerical probes, without truth.

Preparation selects the first two distinct flynn planning first-bank status-9
failures in original row order. Run waits for every original phase to finish.
The only changed solver setting is Method=0 or 2. Original artifacts, oracle,
prices, frontend fits, search and DR results are never rewritten or rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import traceback


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    path=Path(path);part=path.with_suffix(path.suffix+'.partial')
    part.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    part.replace(path)


def sources_unchanged(cfg):
    repo=Path(cfg['repository'])
    for path,expected in cfg['source_sha256'].items():
        p=Path(path)
        if not p.is_absolute():p=repo/p
        if sha(p)!=expected:raise RuntimeError('bound source changed: '+str(p))


def prepare(root):
    root=Path(root).resolve();cfg=read(root/'protocol.json')
    out=root/'lp_method_probe_v1';out.mkdir(exist_ok=True)
    target=out/'protocol.json'
    if target.exists():raise FileExistsError('LP probe protocol already frozen; never replace inputs')
    sources_unchanged(cfg)
    case='flynn16';case_cfg=cfg['cases'][case]
    ended_path=root/case/'planning'/'phase_ended.json'
    ended=read(ended_path)
    if ended['protocol_sha256']!=sha(root/'protocol.json'):raise RuntimeError('planning source protocol mismatch')
    inputs={str(p):sha(p) for p in (root/'protocol.json',ended_path,Path(__file__).resolve())}
    selected=[];seen=set()
    for order,row in enumerate(ended['rows']):
        location=root/case/'planning'/'records'/row['record_id']
        request_path=location/'fit'/'fit_request.json'
        if not request_path.exists():continue
        request=read(request_path);key=request['cache_key']
        cache=(root/case/'fit_cache'/key).resolve()
        if cache.parent!=(root/case/'fit_cache').resolve():raise RuntimeError('cache outside original case')
        result_path=cache/'result.json';meta_path=cache/'cache_record.json'
        if not result_path.exists() or not meta_path.exists():continue
        result=read(result_path)
        solves=result.get('LP_solves',[])
        first_bank=(result.get('status')=='failed' and result.get('stage')=='building_pool'
                    and len(solves)==1 and solves[0].get('status')==9
                    and solves[0].get('accepted') is False
                    and result.get('settings',{}).get('lp_time_limit_seconds')==15.
                    and result.get('pool_support_indices')
                    and result.get('exception')=='fixed-family LP status 9')
        if not first_bank or key in seen:continue
        meta=read(meta_path)
        if meta['identity']['algorithm_source_sha256']!=cfg['algorithm_source_sha256']:
            raise RuntimeError('failed-cache source mismatch')
        if meta.get('result_sha256') and sha(result_path)!=meta['result_sha256']:
            raise RuntimeError('original failed result hash mismatch')
        receipt_path=Path(row.get('physical_receipt_path',location/'physical'/'events_receipt.json'))
        receipt=read(receipt_path);packet_path=receipt_path.parent/'packet.pkl'
        if sha(packet_path)!=receipt['private_files']['packet.pkl']['sha256']:
            raise RuntimeError('observed-only packet hash mismatch')
        history_path=Path(case_cfg['history_packet']['path'])
        if sha(history_path)!=case_cfg['history_packet']['sha256']:raise RuntimeError('history hash mismatch')
        if row.get('physical_receipt_sha256') and sha(receipt_path)!=row['physical_receipt_sha256']:
            raise RuntimeError('physical receipt changed')
        if result['source_sha256']!=cfg['algorithm_source_sha256']:raise RuntimeError('original fit source mismatch')
        for path in (request_path,result_path,meta_path,receipt_path,packet_path,history_path):inputs[str(path)]=sha(path)
        selected.append(dict(case=case,phase='planning',original_row_order=order,record_id=row['record_id'],
            policy=row['policy'],response_path=row['path'],cache_key=key,
            failed_result_path=str(result_path),cache_record_path=str(meta_path),
            history_path=str(history_path),observed_packet_path=str(packet_path),
            training_input_sha256=result['input_sha256']['training'],
            expected_training_rows=480,expected_terminal_labels=result['terminal_labels'],
            original_failure=dict(status=9,location='first raw RNJ bank fixed-family LP (raw_R_0)',
                stage=result['stage'],family=solves[0]['family'],seconds=solves[0].get('seconds'),
                solution_count=solves[0].get('solution_count'),time_limit_seconds=15.,method=1),
            original_record_status=row['status']))
        seen.add(key)
        if len(selected)==2:break
    if len(selected)!=2:raise RuntimeError('fewer than two qualifying distinct first-bank planning failures; do not substitute observations')
    protocol=dict(id='fixed_family_lp_method_probe_v1',status='frozen',frozen_utc=now(),
        script_sha256=sha(__file__),main_protocol_sha256=sha(root/'protocol.json'),
        repository=cfg['repository'],algorithm_source_sha256=cfg['algorithm_source_sha256'],
        selection_rule='first two distinct flynn planning first-bank LP status9 failures in original row order',
        selected=selected,methods=[0,2],planned_LPs=4,LP_time_limit_seconds=15.,
        total_declared_optimizer_time_limit_seconds=60.,
        wall_time_includes_extra_loading_and_model_construction=True,
        original_method=1,changed_parameter_only='Method',
        unchanged=dict(Threads=1,FeasibilityTol=1e-9,OptimalityTol=1e-9,
                       coefficient_lower_bound=0.,coefficient_upper_bound=2.,
                       inactive_pool_coefficients=0.,objective='mean absolute voltage-squared-drop residual',
                       preprocess='raw',training_rows=480),
        no_truth_or_test_inputs=True,no_QP_or_build_bank=True,
        no_original_failure_or_oracle_changes=True,
        run_barrier='all original planning/test phase terminal files and aggregate must exist; no original phase lock',
        input_sha256=inputs)
    write(target,protocol)
    write(out/'protocol.sha256.json',dict(sha256=sha(target)))
    return dict(status='prepared',protocol=str(target),protocol_sha256=sha(target),selected=[r['record_id'] for r in selected],planned_LPs=4)


def attr(model,name):
    try:
        value=getattr(model,name)
        if isinstance(value,float) and not __import__('math').isfinite(value):return None
        return value
    except Exception:return None


def run(root):
    root=Path(root).resolve();out=root/'lp_method_probe_v1'
    protocol_path=out/'protocol.json';p=read(protocol_path);cfg=read(root/'protocol.json')
    if sha(protocol_path)!=read(out/'protocol.sha256.json')['sha256'] or sha(__file__)!=p['script_sha256']:
        raise RuntimeError('probe protocol/source lock mismatch')
    if sha(root/'protocol.json')!=p['main_protocol_sha256']:raise RuntimeError('main protocol changed')
    if p['methods']!=[0,2] or len(p['selected'])!=2 or p['planned_LPs']!=4:raise RuntimeError('probe size differs from frozen 2x2 plan')
    # Existence barriers only: never open test results or any topology truth.
    barrier=[root/'aggregate.json',root/'oracle_selected.json',root/'oracle_selected.sha256.json']
    for case in cfg['cases']:
        for phase in ('planning','test'):
            folder=root/case/phase
            barrier.extend([folder/'fits_frozen.json',folder/'phase_ended.json'])
            if (folder/'phase.lock').exists():raise RuntimeError('original phase still active; numerical probe refused')
    if any(not v.exists() for v in barrier):raise RuntimeError('all original 64 batch records must finish before numerical probing')
    if (out/'ended.json').exists() or (out/'progress.json').exists():raise FileExistsError('probe attempt retained; no automatic retries')
    sources_unchanged(cfg)
    for path,expected in p['input_sha256'].items():
        if sha(path)!=expected:raise RuntimeError('frozen probe input changed: '+path)
    sys.path.insert(0,p['repository'])
    import numpy as np
    from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,input_hash,arrays,matrices
    from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP
    records=[];started=time.perf_counter();lock=out/'probe.lock'
    with lock.open('x',encoding='utf-8') as stream:json.dump(dict(started_utc=now(),script_sha256=p['script_sha256']),stream)
    try:
        for chosen in p['selected']:
            original=read(chosen['failed_result_path'])
            with Path(chosen['history_path']).open('rb') as stream:history=pickle.load(stream)['history']
            with Path(chosen['observed_packet_path']).open('rb') as stream:packet=pickle.load(stream)
            if packet['metadata'].get('observed_only') is not True:raise RuntimeError('packet is not observed-only')
            # The serialized packet also contains validation observations; only
            # its training list is accessed, normalized or used by this probe.
            obs,days,_=normalize(history['training']+packet['training'],'training')
            if input_hash(obs,days)!=chosen['training_input_sha256']:raise RuntimeError('reconstructed training observations differ from original failed LP')
            if len(obs[0])!=480 or list(obs[0].columns)!=chosen['expected_terminal_labels']:
                raise RuntimeError('training shape/order differs from failed fit')
            data=arrays(obs,'raw');family=chosen['original_failure']['family'];pool=original['pool_support_indices']
            previous_features_sha=None
            for method in p['methods']:
                record=dict(record_id=chosen['record_id'],cache_key=chosen['cache_key'],method=method,
                    status='started',original_failure=chosen['original_failure'],original_joint_record_unchanged=True,
                    original_LP_retried=False,independent_numerical_method_probe=True)
                records.append(record);lp=None;then=time.perf_counter()
                try:
                    for path,expected in p['input_sha256'].items():
                        if sha(path)!=expected:raise RuntimeError('input changed before isolated LP probe: '+path)
                    lp=PoolLP(None,pool,'raw',bound=2.,data=data)
                    # All other parameters come from the frozen original class.
                    lp.model.setParam('Method',method)
                    parameters={key:float(getattr(lp.model.Params,key)) for key in ('Method','Threads','TimeLimit','FeasibilityTol','OptimalityTol')}
                    if parameters!=dict(Method=float(method),Threads=1.,TimeLimit=15.,FeasibilityTol=1e-9,OptimalityTol=1e-9):
                        raise RuntimeError('unexpected solver parameter drift')
                    feature_sha=hashlib.sha256(np.asarray(lp.features,dtype='<f8').tobytes()).hexdigest()
                    if previous_features_sha is not None and feature_sha!=previous_features_sha:raise RuntimeError('method probes do not share identical features')
                    previous_features_sha=feature_sha
                    record.update(parameters=parameters,feature_sha256=feature_sha,
                                  feature_shape=list(lp.features.shape),target_size=int(lp.y.size),
                                  target_sha256=hashlib.sha256(np.asarray(lp.y,dtype='<f8').tobytes()).hexdigest())
                    try:
                        fitted=lp.fit(family)
                        record['status']='optimal'
                    except Exception as exc:
                        record.update(status='solver_nonoptimal_or_failed',exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                    record['solver']={name:attr(lp.model,name) for name in
                        ('Status','SolCount','Runtime','Work','IterCount','BarIterCount','ObjVal','ObjBound',
                         'ConstrVio','BoundVio','DualVio','MaxVio','NumVars','NumConstrs')}
                    record['saved_LP_solves']=lp.solves
                    if record['solver']['Status']==2:
                        values=np.asarray(lp.weights.X,float)
                        active=[lp.lookup[tuple(s)] for s in family];k=len(lp.pool)
                        r=values[active];x=values[np.asarray(active)+k]
                        rm,xm=matrices(lp.n,family,r,x)
                        direct=float(np.mean(np.abs(lp.p@rm.T+lp.q@xm.T-lp.y)))
                        feature=float(np.mean(np.abs(lp.features@values-lp.y.ravel())))
                        objective=float(lp.model.ObjVal)
                        inactive=np.ones(2*k,bool);inactive[active]=False;inactive[np.asarray(active)+k]=False
                        residual_pass=bool(np.isclose(direct,objective,atol=1e-8,rtol=1e-8) and np.isclose(feature,objective,atol=1e-8,rtol=1e-8))
                        bounds_pass=bool(values.min()>=-1e-9 and values.max()<=2.+1e-9 and np.max(np.abs(values[inactive]),initial=0.)<=1e-9)
                        feasibility_pass=bool(float(lp.model.ConstrVio)<=1e-7 and float(lp.model.BoundVio)<=1e-7)
                        record.update(direct_family_residual_mae=direct,feature_residual_mae=feature,
                            solver_objective=objective,direct_objective_agreement_pass=residual_pass,
                            coefficient_bounds_pass=bounds_pass,primal_feasibility_pass=feasibility_pass,
                            active_family=family,active_r=r.tolist(),active_x=x.tolist(),
                            accepted_numerical_probe=record['status']=='optimal' and residual_pass and bounds_pass and feasibility_pass)
                    else:record['accepted_numerical_probe']=False
                except Exception as exc:
                    record.update(status='probe_failed',exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),accepted_numerical_probe=False)
                finally:
                    record['wall_seconds']=time.perf_counter()-then
                    if lp is not None:lp.close()
                    write(out/'progress.json',dict(protocol_sha256=sha(protocol_path),planned_LPs=4,records=records,
                                                  wall_seconds=time.perf_counter()-started))
                    print(json.dumps(dict(record_id=record['record_id'],method=method,status=record['status'],
                                          solver_status=record.get('solver',{}).get('Status'),wall_seconds=record['wall_seconds']),ensure_ascii=True),flush=True)
        for path,expected in p['input_sha256'].items():
            if sha(path)!=expected:raise RuntimeError('original artifact changed during numerical probe')
        summary=dict(status='complete',protocol_sha256=sha(protocol_path),script_sha256=p['script_sha256'],
            planned_LPs=4,recorded_LPs=len(records),accepted_numerical_probes=sum(r['accepted_numerical_probe'] for r in records),
            records=records,wall_seconds=time.perf_counter()-started,
            total_solver_runtime=sum(r.get('solver',{}).get('Runtime') or 0. for r in records),
            declared_total_solver_limit_seconds=60.,truth_files_read=0,test_files_read=0,
            QP_calls=0,AC_calls=0,build_bank_calls=0,main_results_modified=0,oracle_modified=0,
            interpretation='isolated numerical diagnostic of the same objective; no DR-effect or full-identifier recovery claim')
        write(out/'ended.json',summary)
        return dict(status='complete',report=str(out/'ended.json'),recorded_LPs=len(records),accepted=summary['accepted_numerical_probes'])
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run'));parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(prepare(args.root) if args.command=='prepare' else run(args.root),ensure_ascii=True),flush=True)
