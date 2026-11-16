"""Independent saved-array audit after all identification attempts have returned."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ... import io
from ..rnj_pool_only_20261116.compare import Reader, audit_native, response, strict_selection, topology
from .run import METHODS


def origins(native):
    if 'candidate_origins' in native:
        return native['candidate_origins']
    labels = ['basis']
    for event in native['events']:
        if event.get('accepted'):
            # Initial-source names describe the submitted start, not the origin
            # of an improved E/T return. Keep the two concepts separate.
            returned = event.get('returned_point_source', 'solver_return_with_QP_closure')
            labels.append(event['stage']+':'+returned)
    return [*labels, 'singleton_protection']


def summarize(rows, methods=METHODS):
    out = {}
    for method in methods:
        planned = [r for r in rows if r['method']==method]
        complete = [r for r in planned if r['status']=='complete']
        events = [e for r in complete for e in r['audit']['events']]
        item = dict(planned=len(planned), completed=len(complete), failed=len(planned)-len(complete),
            internal=sum(r['audit']['phases']['validation']['topology']['internal_exact'] for r in complete),
            full=sum(r['audit']['phases']['validation']['topology']['full_exact'] for r in complete),
            raw_internal=sum(r['raw']['topology']['internal_exact'] for r in complete),
            raw_full=sum(r['raw']['topology']['full_exact'] for r in complete),
            cumulative_internal=sum(r['audit']['phases']['cumulative']['topology']['internal_exact'] for r in complete),
            cumulative_full=sum(r['audit']['phases']['cumulative']['topology']['full_exact'] for r in complete),
            method_seconds=sum(r['audit']['seconds'] for r in complete),
            MIQP_started=sum(e['solve_started'] for e in events), MIQP_timeouts=sum(e['timed_out'] for e in events),
            native_improved_initial=sum(e['native_improved_initial'] is True for e in events),
            total_cap_exceeded=sum(r['total_cap_exceeded'] for r in complete))
        item['mean_method_seconds'] = item['method_seconds']/len(complete) if complete else None
        out[method] = item
    return out


def build(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Keep prior audit: '+str(out))
    reader = Reader(); completion = reader.json(run/'complete.json')
    if completion['completed']!=32:
        raise ValueError('All planned attempts must return before evaluation')
    protocol = reader.json(run/'protocol.json'); entries = reader.json(protocol['inputs_ref'])
    for key in ('runner', 'shared_runner', 'protocol_document'):
        reader.file(protocol[key])
    if protocol['methods']!=list(METHODS) or len(entries)!=8:
        raise ValueError('Changed planned input/method set')
    for ref in protocol['sources']:
        reader.file(ref)
    rows, errors = [], []
    for entry in entries:
        data = None
        saved_models = {}
        for method in METHODS:
            row = dict(id=entry['id'], case=entry['case'], method=method, status='missing')
            try:
                target = run/'cases'/entry['id']/method
                result = reader.json(target/'result.json')
                row.update(status=result['status'], error=result.get('error'))
                if result['status']!='complete':
                    rows.append(row)
                    continue
                if not result['sources_unchanged'] or result['truth_used'] or result['new_day_loaded']:
                    raise ValueError('Source or fitting information boundary changed')
                if data is None:
                    from ..rnj_pool_only_20261116.compare import observation
                    data = {role:observation(reader.arrays(entry['files'][role+'.npz']))
                            for role in ('train','validation','new_day')}
                    physical = reader.arrays(entry['Hstar_ref'])
                    if list(map(int,physical['labels']))!=data['train']['labels']:
                        raise ValueError('Physical node order changed')
                    truth = json.loads(str(physical['support_json'])); hstar=physical['Hstar']
                native = reader.json(result['native'])
                saved_models[method] = native
                if native['method']!=method:
                    raise ValueError('Wrong method result')
                row['audit'] = audit_native(native, **data, truth=truth, hstar=hstar)
                point_origins = origins(native)
                if len(point_origins)!=len(native['frozen_candidates']):
                    raise ValueError('Candidate origin length differs')
                row['selected_origin'] = point_origins[native['selected']['index']]
                row['total_cap_exceeded'] = result['total_cap_exceeded']
                points=native['uncontracted_output_path']
                raw = strict_selection(points, [data['validation']], native['reference_mse'], native['config']['validation_tie'])
                selected=points[raw['selected']]; n=len(entry['labels'])
                row['raw'] = dict(index=raw['selected'], origin=point_origins[raw['selected']],
                    mse=raw['rows'][raw['selected']]['mse'],
                    topology=topology(selected, truth, n, native['config']['zero_tol']))
                stage_rows=[]
                for event in native['events']:
                    detail={k:event.get(k) for k in ('stage','initial_source','returned_point_source','pretrial_QP_count',
                        'pretrial_seconds','pretrial_candidates_attempted','pretrial_stop_reason','current_to_initial_gain',
                        'native_improved_initial','accepted')}
                    for role,key in [('initial','initial_point'),('raw','raw_point'),('closed','closed_point')]:
                        if event.get(key) is not None:
                            detail[role+'_topology']=topology(event[key],truth,n,native['config']['zero_tol'])
                    stage_rows.append(detail)
                row['stages']=stage_rows
                if method.startswith('RNJ'):
                    row['RNJ_families']={k:native[k] for k in ('raw_RNJ_family_count','unique_RNJ_family_count')}
                row['status']='complete'
            except Exception as exc:
                row.update(status='audit_failed', error=str(exc))
                errors.append(dict(id=entry['id'], method=method, error=str(exc)))
            rows.append(row)
        # Source labels can prefer an earlier duplicate. Compare predictions
        # directly rather than equating the selected index with causation.
        if any(r['id']==entry['id'] and r['method']=='RNJ15' and r['status']=='complete' for r in rows):
            n = len(entry['labels'])
            direct = [response(c['point'], n) for c in saved_models['RNJ15']['frozen_candidates']]
            for row in (r for r in rows if r['id']==entry['id'] and r['status']=='complete'):
                fitted = response(saved_models[row['method']]['selected']['point'], n)
                differences = [float(np.max(np.abs(fitted-h))) for h in direct]
                row['RNJ15_equivalence'] = dict(
                    matching_candidate_indices=[i for i,h in enumerate(direct) if np.allclose(fitted,h,rtol=1e-10,atol=1e-12)],
                    min_max_abs_matrix_difference=min(differences),
                    scope='Final selected response versus all RNJ15 baseline candidates, including common protection models')
    reader.verify_unchanged()
    report=dict(status='passed' if not errors and all(r['status']=='complete' for r in rows) else 'incomplete_or_failed',
        planned=32, completed=sum(r['status']=='complete' for r in rows), errors=errors, rows=rows,
        summary=summarize(rows), sources=list(reader.sources.values()), audit_source=io.ref(__file__),
        protocol=protocol, execution_completion=completion,
        scope='Eight known history10 inputs, not independent confirmation; same .08 output processing and validation rule')
    out.mkdir(parents=True); io.write(out/'report.json',report)
    lines=['# 同后处理的拓扑辨识贡献核查','',f"状态：{report['status']}；完成{report['completed']}/32；核验错误{len(errors)}。",'',
        '|方法|完成/计划|收缩前内部/全部|收缩后内部/全部|累计内部/全部|平均方法秒|MIQP限时/次数|原生改善初值次数|',
        '|---|---|---|---|---|---:|---|---:|']
    for method,s in report['summary'].items():
        lines.append(f"|{method}|{s['completed']}/{s['planned']}|{s['raw_internal']}/{s['raw_full']}|{s['internal']}/{s['full']}|{s['cumulative_internal']}/{s['cumulative_full']}|{s['mean_method_seconds']}|{s['MIQP_timeouts']}/{s['MIQP_started']}|{s['native_improved_initial']}|")
    lines += ['', '各恢复计数分母为本方法的8个计划输入，失败未剔除。原生改善次数是阶段次数，不等于最终准确率新增次数。', '',
        '|输入|方法|状态|收缩前内部/全部|最终内部/全部|选择来源|', '|---|---|---|---|---|---|']
    for r in rows:
        if r['status']=='complete':
            a=r['raw']['topology'];b=r['audit']['phases']['validation']['topology']
            lines.append(f"|{r['id']}|{r['method']}|complete|{a['internal_exact']}/{a['full_exact']}|{b['internal_exact']}/{b['full_exact']}|{r['selected_origin']}|")
        else:lines.append(f"|{r['id']}|{r['method']}|{r['status']}|—|—|{r.get('error','')}|")
    lines += ['', '本批共用已知两网、同一时间窗口及四组噪声，未做新AC或灵活性采购。详细初值、预试、原生、收缩分解保存在report.json。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    io.write(out/'complete.json',dict(status=report['status'], report=io.ref(out/'report.json')))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();report=build(args.run,args.out)
    print(json.dumps({k:report[k] for k in ('status','planned','completed','summary')},ensure_ascii=False))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
