"""Create compact, explicitly incomplete delivery traces from immutable raw trials.

The full raw records stay local. Slim traces preserve measurements, every
executed fit, all task scores/failures, optimizer bounds, and the best QPs.
They omit repeated LP certificates and nonbest parameter/residual vectors.
No scientific fitting, optimization, simulator or selector is imported.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from . import ev_compact_analysis as analysis

SCHEMA='ev_compact_slim_trace_v1'
FULL_QP_FIELDS=('physical_parameter','xi','z','residual','original_action_lower','original_action_upper')
QP_TRACE_FIELDS=('success','status','message','optimality','primal_upper','tangent_lower_numeric',
    'q_primal_upper','q_tangent_lower','numerical_gap','numeric_guard','physical_bound_violation',
    'active_lower','active_upper','fixed_indices','seconds','fallback','domain','scope')
FIT_SUMMARY_FIELDS=('name','candidate','supports','loss','half_loss','loglik','success','status',
    'message','optimality','nfev','global_MLE_certified')
DROP_AUDITS={'LP_audit','LP_audits','lp_audit','lp_audits'}
DROP_VALIDATION_VECTORS={'physical_enclosure_lower','physical_enclosure_upper'}


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as handle:
        return json.load(handle)


def _audit_summary(value):
    """Keep the audit conclusion and numerical bounds, omit repeated LP vectors."""
    if isinstance(value,list):return [_audit_summary(x) for x in value]
    if not isinstance(value,dict):return value
    answer={key:_audit_summary(item) for key,item in value.items()
            if key not in DROP_AUDITS|DROP_VALIDATION_VECTORS}
    removed=sorted(set(value)&(DROP_AUDITS|DROP_VALIDATION_VECTORS))
    if removed:answer['delivery_omitted_audit_fields']=removed
    return answer


def slim_prediction(prediction):
    if prediction is None:return None
    answer={key:deepcopy(value) for key,value in prediction.items() if key!='records'}
    records=[]
    for record in prediction.get('records') or []:
        # The action, forecast, task losses, failure and call counts remain.
        # Full predicted fit state/line arrays and repeated observations do not.
        item={key:deepcopy(value) for key,value in record.items()
              if key not in ('fits','predicted_observations')}
        item['fits']=[{key:deepcopy(value) for key,value in fit.items() if key in FIT_SUMMARY_FIELDS}
                      for fit in record.get('fits') or []]
        item['delivery_prediction_fits_are_summaries']=True
        records.append(item)
    answer['records']=records
    return answer


def slim_master(master):
    answer=_audit_summary(master)
    # Scalars, all solver bounds and dimensions, starts, selected u/x and
    # witness numerical guards remain. Large repeated LP vectors are gone.
    answer['delivery_LP_certificates_are_summaries']=True
    return answer


def slim_exchange(exchange):
    if exchange is None:return None
    answer={key:_audit_summary(value) for key,value in exchange.items()
            if key not in ('witnesses','evaluations','iterations')}
    # Witness arrays are omitted; counts/origins/domain conclusions remain.
    answer['witness_summary']=[[
        {key:_audit_summary(value) for key,value in witness.items()
         if key not in ('physical_intercept','physical_slope')}
        for witness in group] for group in exchange.get('witnesses') or []]
    evaluations=exchange.get('evaluations') or [];best=exchange.get('best_evaluation_index')
    if evaluations and (not isinstance(best,int) or not 0<=best<len(evaluations)):
        raise ValueError('exchange evaluations require a valid saved best index')
    answer['evaluations']=[]
    for index,evaluation in enumerate(evaluations):
        item={key:deepcopy(value) for key,value in evaluation.items()
              if key not in ('qp_results','operation_x')}
        if index==best:
            item['operation_x']=deepcopy(evaluation.get('operation_x'))
            item['qp_results']=deepcopy(evaluation.get('qp_results') or [])
            item['delivery_QP_parameters_and_residuals_complete']=True
        else:
            item['qp_results']=[{key:deepcopy(value) for key,value in qp.items() if key in QP_TRACE_FIELDS}
                                for qp in evaluation.get('qp_results') or []]
            item['delivery_QP_parameters_and_residuals_complete']=False
        answer['evaluations'].append(item)
    answer['iterations']=[]
    for iteration in exchange.get('iterations') or []:
        item={key:deepcopy(value) for key,value in iteration.items() if key!='master'}
        if iteration.get('master') is not None:item['master']=slim_master(iteration['master'])
        answer['iterations'].append(item)
    answer['delivery_full_witness_bank_omitted']=True
    return answer


def slim_method(record):
    answer={key:_audit_summary(value) for key,value in record.items()
            if key not in ('raw','before_recenter','raw_prediction','proxy_AC_checks')}
    for key in ('raw','before_recenter'):
        if key in record:answer[key]=slim_exchange(record[key])
    if 'raw_prediction' in record:answer['raw_prediction']=slim_prediction(record['raw_prediction'])
    if 'proxy_AC_checks' in record:
        answer['proxy_AC_checks']=[]
        for rows in record['proxy_AC_checks']:
            reduced=[]
            for row in rows:
                item={key:deepcopy(value) for key,value in row.items() if key!='check'}
                check=row.get('check') or {}
                item['check']={key:deepcopy(value) for key,value in check.items()
                    if key not in ('complete_AC_residual','proxy_residual','proxy_minus_AC')}
                item['check']['delivery_AC_gate_vectors_omitted']=True
                reduced.append(item)
            answer['proxy_AC_checks'].append(reduced)
    return answer


def slim_trial(trial,*,raw_path,raw_sha256,raw_bytes=None):
    if trial.get('artifact_schema')==SCHEMA:raise ValueError('input is already a slim trace, not original raw')
    selection=trial.get('selection') or {}
    answer={key:deepcopy(value) for key,value in trial.items()
            if key not in ('selection','policy_config')}
    answer['selection']={key:_audit_summary(value) for key,value in selection.items()
        if key not in ('methods','predicted_AC','all_prediction_records')}
    answer['selection']['methods']={name:slim_method(record) for name,record in (selection.get('methods') or {}).items()}
    answer['selection']['predicted_AC']={name:slim_prediction(prediction)
        for name,prediction in (selection.get('predicted_AC') or {}).items()}
    answer['selection']['all_prediction_records']=[slim_prediction(prediction)
        for prediction in selection.get('all_prediction_records') or []]
    methods=analysis.declared_methods(trial);present=[row['method'] for row in trial.get('rows') or []]
    answer.update(artifact_schema=SCHEMA,delivery=dict(
        source_raw_path=str(Path(raw_path).resolve()),source_raw_sha256=raw_sha256,source_raw_bytes=raw_bytes,
        source_raw_is_immutable=True,complete_raw_replay_replacement=False,
        declared_methods=list(methods),saved_method_rows=present,missing_planned_method_rows=sorted(set(methods)-set(present)),
        preservation='historical data/fits, truth/meters, all actual noisy+noiseless observations/fits, operation ledgers; all H-only task scores/failures; every exchange iteration and pair intervals; complete best QP parameters/residuals',
        omissions=['full H-only prediction fit parameters and residual arrays',
            'repeated H-only predicted observation arrays','repeated LP certificate vectors',
            'physical witness bank intercept/slope arrays','nonbest QP parameter/residual and operation vectors',
            'same-witness AC gate residual vectors','duplicate policy_config (see freeze.json)'],
        policy_config_ref='freeze.json',exporter_sha256=sha256_file(__file__),
        exporter_dependencies_sha256={Path(analysis.__file__).name:sha256_file(analysis.__file__),
            Path(analysis.core.__file__).name:sha256_file(analysis.core.__file__)}))
    return answer


def _write_identical_or_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('refusing to overwrite different delivery artifact '+str(path))
    else:path.write_bytes(data)


def export_directory(source,target,*,allow_partial=False):
    source=Path(source).resolve();target=Path(target).resolve()
    if source==target:raise ValueError('delivery directory must differ from original raw directory')
    freeze=read_json(source/'freeze.json');planned=set(freeze.get('seeds') or [])
    for name in ('freeze.json','source_snapshot.json.gz'):
        if not (source/name).is_file():raise ValueError('required frozen source evidence missing: '+name)
    paths=sorted(source.glob('trial_*.json.gz'))
    # Require explicit permission to export an unfinished batch.
    if not allow_partial and len(paths)!=len(planned):raise ValueError('batch incomplete; use --allow-partial for a labelled partial export')
    metadata=[];seen=set();raw_total=0;slim_total=0
    for path in paths:
        before=sha256_file(path);size=path.stat().st_size;trial=read_json(path);seed=trial.get('seed')
        if seed not in planned or seed in seen:raise ValueError('unplanned or duplicate historical sample')
        seen.add(seed)
        if (trial.get('source_sha256')!=freeze.get('source_sha256') or trial.get('config')!=freeze.get('config')
                or trial.get('phase')!=freeze.get('phase') or trial.get('policy_id')!=freeze.get('policy_id')):
            raise ValueError('raw trial differs from frozen source/config/phase/policy')
        reduced=slim_trial(trial,raw_path=path,raw_sha256=before,raw_bytes=size)
        raw=(json.dumps(reduced,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')
        compressed=gzip.compress(raw,compresslevel=9,mtime=0)
        output=target/(path.name[:-8]+'.slim.json.gz')
        if output.resolve()==path.resolve():raise ValueError('output would replace original raw')
        _write_identical_or_new(output,compressed)
        after=sha256_file(path)
        if before!=after:raise ValueError('original raw changed during delivery export')
        metadata.append(dict(seed=seed,raw_path=str(path),raw_sha256=before,raw_bytes=size,
            slim_file=output.name,slim_sha256=hashlib.sha256(compressed).hexdigest(),slim_bytes=len(compressed),
            raw_unchanged=True,method_rows=len(trial.get('rows') or []),declared_methods=list(analysis.declared_methods(trial))))
        raw_total+=size;slim_total+=len(compressed)
    copies=[]
    for name in ('freeze.json','source_snapshot.json.gz'):
        original=source/name
        if not original.exists():raise ValueError('required frozen source evidence missing: '+name)
        data=original.read_bytes();_write_identical_or_new(target/name,data)
        copies.append(dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),byte_identical=True))
    manifest=dict(artifact_schema=SCHEMA,source_directory=str(source),target_directory=str(target),phase=freeze.get('phase'),
        planned_seeds=sorted(planned),exported_seeds=sorted(seen),missing_seeds=sorted(planned-seen),partial=seen!=planned,
        history_unit='one saved independent historical sample',raw_bytes=raw_total,slim_bytes=slim_total,
        slim_over_raw_ratio=slim_total/raw_total if raw_total else None,complete_raw_replay_replacement=False,
        raw_retention='full raw files stay at the saved local paths; no raw file is modified or deleted',
        trial_files=metadata,source_files=copies,exporter_sha256=sha256_file(__file__),
        exporter_dependencies_sha256={Path(analysis.__file__).name:sha256_file(analysis.__file__),
            Path(analysis.core.__file__).name:sha256_file(analysis.core.__file__)})
    _write_identical_or_new(target/'delivery_manifest.json',(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode('utf-8'))
    scope=('# Compact EV evidence delivery\n\n'
        'These .slim.json.gz files are explicit slim traces, not byte-equivalent raw files. '
        'The complete original raw is retained locally; its exact compressed-byte SHA-256, size and path are in delivery_manifest.json.\n\n'
        'Preserved: historical measurements/fits, truth and measurement sigmas, every actual noisy/noiseless observation and fit, '
        'selected physical route/resource ledger, all H-only task scores/failures, all solver bounds/gaps/scales/timing, '
        'each exchange evaluation pair interval and status, and complete physical parameters/residuals of every round best QP.\n\n'
        'Omitted: repeated LP vectors, full H-only prediction fit parameter/residual arrays, nonbest QP parameter/residual vectors, '
        'full witness intercept/slope banks, repeated predicted observation arrays and AC gate residual vectors. '
        'A complete numerical replay of those omitted stages requires the original raw plus the byte-preserved frozen source snapshot.\n\n'
        'Method failures remain in the planned method denominator. Missing batch histories are listed separately. '
        'Development, evaluation and calibration exports retain their original phases.\n')
    _write_identical_or_new(target/'DELIVERY_SCOPE.md',scope.encode('utf-8'))
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();manifest=export_directory(args.source,args.output,allow_partial=args.allow_partial)
    print(json.dumps({key:manifest[key] for key in ('phase','exported_seeds','missing_seeds','raw_bytes','slim_bytes','slim_over_raw_ratio')},ensure_ascii=False))


if __name__=='__main__':main()
