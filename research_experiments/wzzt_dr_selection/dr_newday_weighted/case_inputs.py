"""Read-only adapters for five predeclared complete historical conditions.

No topology graph, test observation, forecast, fitter or physical producer is
used. Historical pickle headers and observed-scene row/column metadata are
checked; forecast entries are never accessed. Old input bindings remain the
authority rather than newly computed hashes being treated as old evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

SCRATCH=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
SOURCE_ROOT=Path(r'D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full/source')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path))


def checked(item):
    if not isinstance(item,dict) or 'path' not in item or 'sha256' not in item:
        raise ValueError('explicit old path/sha256 binding required')
    path=Path(item['path']).resolve()
    if sha(path)!=item['sha256']:raise ValueError('historical input hash drift: '+str(path))
    return path


def history_header(history_reference,case,root_label,base_kw):
    with checked(history_reference).open('rb') as stream:packet=pickle.load(stream)
    # Unpickling restores the archived object; no forecast field or test file
    # is accessed. All inspected scenes belong to historical fitting periods.
    history=packet['history'] if 'history' in packet else packet
    labels=list(history['labels'])
    if len(labels)!=len(set(labels)) or not labels:raise ValueError('invalid historical labels')
    if int(history['root_label'])!=root_label or float(history['base_kw'])!=base_kw:
        raise ValueError('historical root/power base differs from original metadata')
    rows=[]
    for role in ('training','validation'):
        scenes=history[role]
        if len(scenes)!=3:raise ValueError('three full archived days required per role')
        total=0
        for scene in scenes:
            p,q=scene['P_terminal'],scene['Q_terminal']
            if getattr(p,'shape',None)!=(96,len(labels)) or getattr(q,'shape',None)!=(96,len(labels)):
                raise ValueError('96-row complete P/Q day required')
            if hasattr(p,'columns') and list(p.columns)!=labels:raise ValueError('historical P order changed')
            if hasattr(q,'columns') and list(q.columns)!=labels:raise ValueError('historical Q order changed')
            total+=len(p)
        if total!=288:raise ValueError('complete288 historical rows required')
        rows.append(total)
    if 'source_metadata_path' in history and not Path(history['source_metadata_path']).name=='metadata.json':
        raise ValueError('unexpected historical metadata declaration')
    return dict(labels=labels,rows=rows,base_kv=float(history['base_kv']),base_mva=float(history['base_mva']),
        source_case_name=history.get('source_case_name'),history_case=case)


def _finish(name,case,job_id,voltage_sigma,history,fit,metadata,source_root,evidence):
    history=deepcopy(history);fit=deepcopy(fit);metadata=deepcopy(metadata)
    checked(history);checked(fit);meta=read(checked(metadata))
    root_label=int(meta['root_bus']);base_kw=1000*float(meta['base_mva'])
    source_root=Path(source_root).resolve()
    if not source_root.is_dir():raise ValueError('declared original physical source missing')
    header=history_header(history,case,root_label,base_kw)
    if abs(header['base_kv']-float(meta['base_kv']))>1e-12 or abs(header['base_mva']-float(meta['base_mva']))>1e-12:
        raise ValueError('historical electrical bases differ from bound metadata')
    if header['source_case_name'] and header['source_case_name']!=metadata['case_name']:
        raise ValueError('historical case name differs from bound original metadata')
    return dict(status='frozen',name=name,case=case,job_id=job_id,
        noise=dict(voltage_sigma=voltage_sigma,pq_sigma=.005),voltage_sigma=voltage_sigma,pq_sigma=.005,
        history_packet=history,historical_baseline_result=fit,source_root=str(source_root),source_metadata=metadata,
        labels=header['labels'],root_label=root_label,base_kw=base_kw,base_kv=header['base_kv'],base_mva=header['base_mva'],
        history_rows=header['rows'],input_authorities=evidence,
        history_read_scope='only archived history training/validation; forecast entry not accessed',
        no_new_fit=True,no_test_future_read=True,no_true_graph_read=True,
        prior_future_boundary='older research futures may already be known; use new prelocked independent replicates',
        parameter_scope='existing full historical result, no parameter conversion/refit; candidates selected downstream using history only')


def build_cases(scratch=SCRATCH):
    """Return all five conditions in a fixed order, without truth filtering."""
    scratch=Path(scratch).resolve();cases=[]
    for name,relative,expected_case,expected_job in [
        ('paper15_b2_v0005','wzzt_targeted_pricing_v1/run_v1','paper15','main__paper15__b2__reference__v0005__N288'),
        ('flynn16_b4_v0005','wzzt_targeted_pricing_v1/flynn_run_v1','flynn16','main__flynn16__b4__reference__v0005__N288')]:
        folder=scratch/relative;prepared=read(folder/'baseline_prepared.json');pfile=checked(prepared['protocol'])
        p=read(pfile);ended=read(folder/'baseline_ended.json')
        if p['case']!=expected_case or p['job_id']!=expected_job:raise ValueError('predeclared targeted condition mismatch')
        if ended['status']!='complete' or ended['protocol_sha256']!=sha(pfile):raise ValueError('old baseline receipt mismatch')
        metadata=p['source_metadata'];checked(p['inputs']['source_metadata'])
        if checked(metadata)!=checked(p['inputs']['source_metadata']):raise ValueError('metadata input binding differs')
        cases.append(_finish(name,expected_case,expected_job,float(p['voltage_sigma']),prepared['history_packet'],
            ended['result'],metadata,SOURCE_ROOT,[reference(folder/'baseline_prepared.json'),prepared['protocol'],reference(folder/'baseline_ended.json')]))
    folder=scratch/'wzzt_budget_reference_diagnostic_v1/run_v1';pfile=folder/'protocol.json';p=read(pfile)
    for case in ('flynn16','pengwah18'):
        old=p['cases'][case];expected_job=f'main__{case}__b5__reference__v0002__N288'
        if old['job_id']!=expected_job:raise ValueError('predeclared original zero-yield condition mismatch')
        cases.append(_finish(f'{case}_b5_v0002',case,expected_job,.0002,old['history_packet'],old['baseline_frozen_result'],
            old['source_metadata'],p['physical_source_root'],[reference(pfile)]))
    folder=scratch/'wzzt_flynn_local_qualification_v1/run_v1';pfile=folder/'baseline_protocol.json';p=read(pfile)
    qfile=folder/'qualification_protocol.json';q=read(qfile);ended=read(folder/'baseline_ended.json')
    expected_job='main__flynn16__b7__reference__v0002__N288'
    if p['job_id']!=q['job_id'] or p['job_id']!=expected_job:raise ValueError('predeclared b7 condition mismatch')
    if ended['status']!='complete' or ended['protocol_sha256']!=sha(pfile):raise ValueError('old b7 receipt mismatch')
    history=dict(path=str(folder/'history_and_forecast.pkl'),sha256=p['input_packet_sha256'])
    fit=dict(path=ended['result_path'],sha256=ended['result_sha256'])
    cases.append(_finish('flynn16_b7_v0002','flynn16',expected_job,.0002,history,fit,q['source_metadata'],SOURCE_ROOT,
        [reference(pfile),reference(qfile),reference(folder/'baseline_ended.json')]))
    return cases


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scratch',type=Path,default=SCRATCH)
    args=parser.parse_args();cases=build_cases(args.scratch)
    print(json.dumps(dict(status='complete',conditions=len(cases),networks=len({c['case'] for c in cases}),
        table=[dict(name=c['name'],rows=c['history_rows'],labels=c['labels'],root=c['root_label'],base_kw=c['base_kw'],
            noise=c['noise'],history_sha256=c['history_packet']['sha256'],fit_sha256=c['historical_baseline_result']['sha256']) for c in cases],
        new_fit_calls=0,new_future_reads=0,true_graph_reads=0),indent=2))
