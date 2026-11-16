"""One explicitly outcome-qualified, previously seen mechanism diagnostic.

Preparation slices the original observed day only. The unchanged shared worker
receives no truth arrays. Evaluation reuses the saved RNJ15 baseline and opens
physical support information after the single current 101 attempt terminates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from ... import io, observations
from ..rnj_pool_only_20261116 import run as shared

DEFAULT_LEGACY = shared.ROOT/'outputs/identification_contribution_legacy240_20261116_v1/evaluation/report.json'
DEFAULT_SOURCE = shared.ROOT/'outputs/identification_contribution_20261116_v1'


def _slice_observed(entry):
    whole = observations.load(shared.check(entry['train_ref']))
    if (whole.points != 96 or list(whole.labels) != entry['labels'] or
            entry['train_ref'] != entry['validation_ref'] or
            entry['train_rows'] != [0,64] or entry['validation_rows'] != [64,96]):
        raise ValueError('Expected the original common 96-row day and 64/32 split')
    parts = []
    for key, a, b in [('train',0,64), ('validation',64,96)]:
        part = observations.Observation(whole.p[a:b], whole.q[a:b], whole.y[a:b], whole.labels)
        if observations.fingerprint(part) != entry[key+'_fingerprint']:
            raise ValueError('Original '+key+' fingerprint does not reproduce')
        parts.append(part)
    return tuple(parts)


def prepare(out, legacy_report=DEFAULT_LEGACY, source_run=DEFAULT_SOURCE):
    out, legacy_report, source_run = map(lambda p:Path(p).resolve(), (out,legacy_report,source_run))
    if out.exists(): raise FileExistsError('Use a fresh diagnostic directory: '+str(out))
    report = io.read(legacy_report)
    if report['status'] != 'passed' or report['completed'] != 960 or report['planned'] != 960:
        raise ValueError('The complete independently replayed main-240 report is required')
    legacy_root = legacy_report.parent.parent
    legacy_complete = io.read(legacy_root/'complete.json')
    if legacy_complete['report'] != io.ref(legacy_report): raise ValueError('Legacy report is not the completed sealed report')
    lookup = {(r['id'],r['method']):r for r in report['rows']}
    ids = sorted({r['id'] for r in report['rows']})
    if len(lookup)!=960 or len(ids)!=240: raise ValueError('Legacy method/input denominator changed')
    qualified = [identifier for identifier in ids
        if lookup[(identifier,'RNJ15')]['status'] == lookup[(identifier,'101')]['status'] == 'complete'
        and not lookup[(identifier,'RNJ15')]['audit']['coverage']['internal']
        and lookup[(identifier,'101')]['final']['topology']['internal_exact']]
    if len(qualified)!=58: raise ValueError('The predeclared 58-case diagnostic qualification changed')
    identifier = qualified[0]
    old_protocol = io.read(shared.check(report['protocol']))
    entries = io.read(shared.check(old_protocol['inputs_ref']))
    entry = next(e for e in entries if e['id']==identifier)
    if entry['source_group']!='large96_main': raise ValueError('Retired or different input group selected')
    baseline = lookup[(identifier,'RNJ15')]
    shared.check(baseline['native_ref']); shared.check(baseline['result_ref'])
    source_protocol = io.read(source_run/'protocol.json')
    for ref in source_protocol['sources']: shared.check(ref)
    train, validation = _slice_observed(entry)
    out.mkdir(parents=True)
    files = {}
    for name, observed in [('train.npz',train),('validation.npz',validation)]:
        path = out/'observations'/name; path.parent.mkdir(exist_ok=True)
        np.savez(path,P=observed.p,Q=observed.q,Y=observed.y,labels=np.asarray(observed.labels,int))
        if observations.fingerprint(observations.load(path)) != observations.fingerprint(observed):
            raise ValueError('Saved P/Q/Y split changed the original observed arrays')
        files[name] = io.ref(path)
    # This is the only entry the fitting worker receives. Evaluation metadata
    # and the qualification report are kept in separately named sealed files.
    worker_entry = dict(id=identifier,case=entry['case'],history_repeat=entry.get('history_repeat'),
        source_group='large96_main_diagnostic',labels=entry['labels'],files=files)
    io.write(out/'inputs.json',[worker_entry])
    io.write(out/'evaluation_input.json',entry)
    io.write(out/'qualification.json',dict(legacy_report=io.ref(legacy_report),
        legacy_complete=io.ref(legacy_root/'complete.json'), qualified_count=58,qualified_ids=qualified,
        selection_rule='sort IDs ascending; first with no RNJ15 postprocessed internal truth but old101 selected internal truth',
        selected_id=identifier, selection_used_known_outcomes=True,
        baseline_RNJ15_native=baseline['native_ref'],baseline_RNJ15_result=baseline['result_ref'],
        old101_native=lookup[(identifier,'101')]['native_ref'],
        input_origin=entry['train_ref'],train_fingerprint=observations.fingerprint(train),
        validation_fingerprint=observations.fingerprint(validation),
        scope='Previously seen, outcome-qualified single-case mechanism check; not an unbiased performance estimate'))
    helpers = [Path(__file__), Path(shared.__file__),
        shared.ROOT/'rnj_wzzt_dr/experiments/rnj_pool_only_20261116/compare.py',
        shared.ROOT/'rnj_wzzt_dr/experiments/impedance_floor_20261116/sources.py',
        shared.ROOT/'rnj_wzzt_dr/observations.py',shared.ROOT/'rnj_wzzt_dr/io.py']
    protocol = dict(created_utc=shared.utc(),version='new',method='101',planned_fits=1,
        method_seconds=300.,per_mip_seconds=30.,output_reserve_seconds=30.,qp_soft_seconds=300.,workers=1,
        budget_scope='300 total; original snapshot applies per_mip_seconds to W; E uses remaining structural deadline',
        source_root=source_protocol['source_root'],sources=source_protocol['sources'],
        reused_source_protocol=io.ref(source_run/'protocol.json'),helper_sources=[io.ref(p) for p in helpers],
        inputs_ref=io.ref(out/'inputs.json'),evaluation_input=io.ref(out/'evaluation_input.json'),
        qualification=io.ref(out/'qualification.json'),protocol_document=io.ref(Path(__file__).with_name('PROTOCOL_MECHANISM.md')),
        environment=io.environment(),no_new_AC=True,no_DR=True,no_E60=True,
        baseline_RNJ15_refit=False,truth_used_for_case_qualification=True,truth_passed_to_worker=False,
        training_rows=64,validation_rows=32,independent_new_day=False)
    io.write(out/'protocol.json',protocol)
    return dict(prepared=True,selected_id=identifier,qualified_count=58,worker_inputs=1,
        protocol=io.ref(out/'protocol.json'))


def fit(out):
    out=Path(out).resolve(); protocol=io.read(out/'protocol.json')
    for ref in [*protocol['sources'],*protocol['helper_sources'],protocol['inputs_ref']]: shared.check(ref)
    if (out/'fit_complete.json').exists(): return io.read(out/'fit_complete.json')
    entry, = io.read(protocol['inputs_ref']['path'])
    target=out/'cases'/entry['id']; started=perf_counter()
    if (target/'result.json').exists():
        saved=io.read(target/'result.json')
        if saved['source_protocol'] != io.ref(out/'protocol.json'):
            raise ValueError('Saved result belongs to a different protocol')
        if saved.get('native'): shared.check(saved['native'])
        row=dict(input_id=entry['id'],version='new',status=saved['status'],result=io.ref(target/'result.json'),
            seconds=saved['seconds'],reused_completed_result=True)
    else:
        # fit_one refuses an existing partial case directory, preserving both
        # its unfinished evidence and the original budget instead of restarting.
        row=shared.fit_one(str(out),entry)
    unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in [*protocol['sources'],*protocol['helper_sources']])
    result=dict(status='complete',planned=1,completed=1,successful=int(row['status']=='complete'),
        failed=int(row['status']!='complete'),records=[row],seconds=perf_counter()-started,
        sources_unchanged=unchanged,protocol=io.ref(out/'protocol.json'))
    io.write(out/'fit_complete.json',result)
    return result


def evaluate(out):
    from ..impedance_floor_20261116.sources import load_evaluation
    from ..rnj_pool_only_20261116.compare import audit_native, topology
    out=Path(out).resolve(); destination=out/'evaluation'
    if destination.exists(): raise FileExistsError('Keep previous evaluation: '+str(destination))
    completion=io.read(out/'fit_complete.json'); protocol=io.read(shared.check(completion['protocol']))
    if completion['completed']!=1 or not completion['sources_unchanged']:
        raise ValueError('The one source-consistent fitting attempt must terminate first')
    for ref in [*protocol['sources'],*protocol['helper_sources']]: shared.check(ref)
    qualification=io.read(shared.check(protocol['qualification']))
    entry=io.read(shared.check(protocol['evaluation_input']))
    worker_entry,=io.read(shared.check(protocol['inputs_ref']))
    result=io.read(shared.check(completion['records'][0]['result']))
    report=dict(status=result['status'],planned=1,completed=int(result['status']=='complete'),
        selected_id=entry['id'],qualification=protocol['qualification'],protocol=io.ref(out/'protocol.json'),
        result=completion['records'][0]['result'],errors=[],
        scope='Known outcome-qualified case; original 64/32 observed day; no independent test day, no AC or DFS')
    if result['status']=='complete':
        try:
            train,valid=(observations.load(shared.check(worker_entry['files'][name])) for name in ('train.npz','validation.npz'))
            if (observations.fingerprint(train)!=qualification['train_fingerprint'] or
                    observations.fingerprint(valid)!=qualification['validation_fingerprint']):
                raise ValueError('Worker observations differ from the selected original split')
            truth_data=load_evaluation(entry);truth=truth_data['truth_supports'];n=len(entry['labels'])
            convert=lambda x:dict(P=x.p,Q=x.q,Y=x.y,labels=list(x.labels))
            data=dict(train=convert(train),validation=convert(valid),new_day=None,truth=truth)
            native=io.read(shared.check(result['native']))
            baseline=io.read(shared.check(qualification['baseline_RNJ15_native']))
            current_audit=audit_native(native,**data);baseline_audit=audit_native(baseline,**data)
            if baseline_audit['coverage']['internal']:
                raise ValueError('Independent replay contradicts the RNJ15 lack-of-truth qualification')
            stages=[]
            for event in native['events']:
                row={key:event.get(key) for key in ('stage','accepted','initial_source','returned_point_source',
                    'pretrial_QP_count','pretrial_seconds','current_to_initial_gain','native_improved_initial')}
                row['solver']={key:event.get('solver',{}).get(key) for key in ('status','solve_started','seconds','objective','bound','gap')}
                for key in ('initial_point','raw_point','closed_point','point'):
                    if event.get(key) is not None:
                        row[key+'_topology']=topology(event[key],truth,n,native['config']['zero_tol'])
                stages.append(row)
            found=current_audit['coverage']['internal'];final=current_audit['phases']['validation']['topology']
            report.update(status='passed',native=result['native'],baseline_native=qualification['baseline_RNJ15_native'],
                current=current_audit,baseline=baseline_audit,stages=stages,
                found_true_internal_missing_from_RNJ15=found,selected_true_internal=final['internal_exact'],
                selected_full_support=final['full_exact'],total_cap_exceeded=result['total_cap_exceeded'],
                native_raw_true_internal=any(r.get('raw_point_topology',{}).get('internal_exact',False) for r in stages),
                evaluation_sources=truth_data['source_refs'])
            for ref in truth_data['source_refs']: shared.check(ref)
        except Exception as exc:
            report.update(status='audit_failed',errors=[repr(exc)])
    else:
        report['failure']=result
    for ref in [*protocol['sources'],*protocol['helper_sources']]: shared.check(ref)
    destination.mkdir();io.write(destination/'report.json',report)
    lines=['# 当前代码的单例机制检查','',f"输入：{entry['id']}；状态：{report['status']}。",'',
        '本例从已完成结果中的58项资格集合按ID取首例，使用真拓扑作事后资格选择。它检验当前实现能否复现一个机制，不估计总体准确率。','']
    if report['status']=='passed':
        c,b=report['current'],report['baseline'];ct=c['phases']['validation']['topology'];bt=b['phases']['validation']['topology']
        lines += ['|方法|库含真内部/全支持|最终内部/全支持|原生限时/次数|', '|---|---|---|---|',
            f"|保存RNJ15|{b['coverage']['internal']}/{b['coverage']['full']}|{bt['internal_exact']}/{bt['full_exact']}|未重跑|",
            f"|当前101|{c['coverage']['internal']}/{c['coverage']['full']}|{ct['internal_exact']}/{ct['full_exact']}|{c['MIQP_timeouts']}/{c['MIQP_count']}|",'',
            f"库内找到RNJ15缺失的真内部结构：{report['found_true_internal_missing_from_RNJ15']}；原生返回中出现真内部结构：{report['native_raw_true_internal']}。",'',
            '初值、固定QP预试、原生返回、QP收尾、输出收缩与最终验证的作用分开记录在report.json。限时未找到不构成不存在该结构的证明。']
    else:lines += [str(report.get('errors') or result.get('error'))]
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    io.write(out/'complete.json',dict(status=report['status'],report=io.ref(destination/'report.json'),
        fit_completion=io.ref(out/'fit_complete.json')))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--legacy-report',type=Path,default=DEFAULT_LEGACY)
    parser.add_argument('--source-run',type=Path,default=DEFAULT_SOURCE)
    phase=parser.add_mutually_exclusive_group()
    phase.add_argument('--prepare-only',action='store_true');phase.add_argument('--fit-only',action='store_true')
    phase.add_argument('--evaluate-only',action='store_true');phase.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        print(prepare(args.out,args.legacy_report,args.source_run));return
    if args.fit_only:
        print(fit(args.out));return
    if not args.evaluate_only:
        if not args.resume: prepare(args.out,args.legacy_report,args.source_run)
        fit(args.out)
    report=evaluate(args.out)
    print({k:report.get(k) for k in ('status','selected_id','found_true_internal_missing_from_RNJ15','selected_true_internal','selected_full_support')})
    if report['status']!='passed': raise SystemExit(1)


if __name__=='__main__':
    main()
