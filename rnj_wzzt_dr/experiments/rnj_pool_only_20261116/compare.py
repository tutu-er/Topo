"""Read-only, paired comparison of sealed identifier runs.

This module never imports the identifier, fits a model, or invokes an optimizer.
Historical source snapshots are provenance, not checked against today's code.
All fitted values and detailed support sets remain in the requested output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def response(point, n):
    """Independent sum of support outer products, in P-then-Q units."""
    h = np.zeros((n, 2*n))
    supports, r, x = (point[k] for k in ('supports', 'r', 'x'))
    require(len(supports) == len(r) == len(x), 'Coefficient lengths differ')
    seen = set()
    for support, rr, xx in zip(supports, r, x):
        require(all(isinstance(i, int) and not isinstance(i, bool) for i in support), 'Noninteger support')
        ix = np.asarray(support, dtype=int)
        require(len(ix) and len(set(ix)) == len(ix) and min(ix) >= 0 and max(ix) < n, 'Invalid support')
        key = tuple(sorted(support))
        require(key not in seen, 'Duplicate support')
        seen.add(key)
        require(np.isfinite([rr, xx]).all(), 'Nonfinite coefficient')
        h[np.ix_(ix, ix)] += rr
        h[np.ix_(ix, ix+n)] += xx
    return h


def observation(arrays):
    labels = list(map(int, arrays['labels']))
    p, q = (np.asarray(arrays[k], float) for k in ('P', 'Q'))
    y = (np.asarray(arrays['Y'], float) if 'Y' in arrays else
         np.asarray(arrays['root'], float).reshape(-1, 1)**2-np.asarray(arrays['V'], float)**2)
    require(p.shape == q.shape == y.shape and p.ndim == 2 and p.shape[1] == len(labels), 'Observation shape mismatch')
    require(p.size and len(set(labels)) == len(labels) and all(np.isfinite(a).all() for a in (p, q, y)), 'Invalid observations')
    return dict(P=p, Q=q, Y=y, labels=labels)


def loss(h, obs):
    residual = obs['Y']-np.c_[obs['P'], obs['Q']]@h.T
    sse = float(np.sum(residual.astype(np.longdouble)**2))
    return dict(sse=sse, mse=sse/residual.size, cells=int(residual.size))


def training_score(point, train, reference, alpha):
    n = len(train['labels'])
    internal = sum(1 < len(c) < n for c in point['supports'])
    return loss(response(point, n), train)['mse']/reference+alpha*internal


def strict_selection(points, observations, reference, tie):
    require(reference > 0 and tie >= 0 and len(points) > 0, 'Invalid selection inputs')
    n = len(observations[0]['labels'])
    cells = sum(o['Y'].size for o in observations)
    rows = []
    for i, point in enumerate(points):
        h = response(point, n)
        mse = sum(loss(h, o)['sse'] for o in observations)/cells
        rows.append(dict(index=i, internal=sum(1 < len(c) < n for c in point['supports']),
                         mse=mse, scaled_mse=mse/reference))
    best = min(row['scaled_mse'] for row in rows)
    eligible = [r for r in rows if r['scaled_mse'] <= best+tie]
    selected = min(eligible, key=lambda r: (r['internal'], r['scaled_mse'], r['index']))['index']
    return dict(selected=selected, cells=cells, rows=rows)


def topology(point, truth, n, zero):
    active = {tuple(sorted(c)) for c, r, x in zip(point['supports'], point['r'], point['x']) if max(r, x) > zero}
    target = {tuple(sorted(c)) for c in truth}
    ai, ti = ({c for c in group if 1 < len(c) < n} for group in (active, target))
    return dict(internal_exact=ai == ti, full_exact=active == target,
                internal_missing=len(ti-ai), internal_extra=len(ai-ti),
                full_missing=len(target-active), full_extra=len(active-target),
                active_supports=[list(c) for c in sorted(active, key=lambda c: (len(c), c))])


def matrix_error(h, truth, train):
    if truth is None:
        return None
    require(h.shape == truth.shape, 'Physical reference shape mismatch')
    n = h.shape[0]
    result = {}
    for name, ix in [('R', slice(0, n)), ('X', slice(n, 2*n))]:
        denominator = np.linalg.norm(truth[:, ix])
        result[name+'_relative_error'] = float(np.linalg.norm(h[:, ix]-truth[:, ix])/denominator) if denominator else None
    residual = np.c_[train['P'], train['Q']]@(h-truth).T
    result['historical_response_error_RMS'] = float(np.sqrt(np.mean(residual**2)))
    result['scope'] = 'Offline full response matrices; not individual line impedance or device-direction error'
    return result


def close(a, b, label, rtol=2e-8, atol=1e-12):
    require(np.isfinite([a, b]).all() and np.isclose(a, b, rtol=rtol, atol=atol), label)


def audit_native(native, train, validation, new_day, truth=None, hstar=None):
    n = len(train['labels'])
    require(validation['labels'] == train['labels'], 'Validation labels differ')
    if new_day is not None:
        require(new_day['labels'] == train['labels'], 'New-day labels differ')
    config = native['config']; reference = float(native['reference_mse'])
    points = [r['point'] for r in native['frozen_candidates']]
    selected = strict_selection(points, [validation], reference, config['validation_tie'])
    require(selected['selected'] == native['selected']['index'], 'Stored validation winner differs')
    for candidate, row in zip(native['frozen_candidates'], selected['rows']):
        require(list(candidate['labels']) == train['labels'], 'Candidate labels differ')
        close(row['mse'], candidate['validation_mse'], 'Stored validation MSE differs')
        if 'R' in candidate and 'X' in candidate:
            require(np.allclose(response(candidate['point'], n), np.c_[candidate['R'], candidate['X']], rtol=1e-12, atol=1e-13), 'Frozen H differs from supports')
    phases = {}
    for phase, obs in [('validation', [validation]), ('cumulative', [validation, new_day] if new_day is not None else None)]:
        if obs is None:
            phases[phase] = None
            continue
        decision = strict_selection(points, obs, reference, config['validation_tie'])
        i = decision['selected']; point = points[i]
        phases[phase] = dict(index=i, mse=decision['rows'][i]['mse'], cells=decision['cells'],
            topology=topology(point, truth, n, config['zero_tol']) if truth is not None else None,
            matrix=matrix_error(response(point, n), hstar, train))
    path = []
    for i, point in enumerate(native['path']):
        score = training_score(point, train, reference, config['alpha'])
        close(score, float(point['objective'])+config['alpha']*sum(1<len(c)<n for c in point['supports']), 'Path training score differs')
        path.append(dict(index=i, score=score, proper_supports=[[train['labels'][j] for j in c] for c in point['supports'] if 1<len(c)<n]))
    events = []
    current = native['path'][0]
    for e in native['events']:
        current_score = training_score(current, train, reference, config['alpha'])
        for key in ('before_score', 'current_score'):
            if e.get(key) is not None:
                close(current_score, e[key], 'Recorded pre-stage current score differs')
        row = dict(stage=e['stage'], accepted=e.get('accepted'), before_score=e.get('before_score'), after_score=e.get('after_score'), current_score=current_score)
        if e.get('point') is not None and e.get('after_score') is not None:
            actual = training_score(e['point'], train, reference, config['alpha'])
            close(actual, e['after_score'], 'Event closed training score differs')
        solver = e.get('solver', {})
        initial = e.get('initial_point', solver.get('initial_point'))
        initial_score = (training_score(initial, train, reference, config['alpha']) if initial is not None
                         else e.get('initial_score', solver.get('initial_objective', e.get('initial_objective'))))
        if initial is not None and e.get('initial_score') is not None:
            close(initial_score, e['initial_score'], 'Recorded initial score differs')
        if initial is not None and solver.get('initial_objective') is not None:
            close(initial_score, solver['initial_objective'], 'Recorded initial objective differs')
        current_to_initial = None if initial_score is None else current_score-initial_score
        if e.get('current_to_initial_gain') is not None:
            close(current_to_initial, e['current_to_initial_gain'], 'Recorded reference-initialization gain differs')
        native_objective = solver.get('objective')
        returned = native_objective
        if e.get('raw_point') is not None:
            returned = training_score(e['raw_point'], train, reference, config['alpha'])
        if e.get('raw_point') is not None and native_objective is not None:
            close(returned, native_objective,
                  'Native objective does not replay', rtol=1e-7)
        if e.get('raw_point') is not None and e.get('native_returned_score') is not None:
            close(training_score(e['raw_point'], train, reference, config['alpha']), e['native_returned_score'], 'Reported raw score differs')
        closed_score = solver.get('fixed_family_qp_objective')
        if e.get('closed_point') is not None:
            closed_score = training_score(e['closed_point'], train, reference, config['alpha'])
            close(closed_score, e['closed_score'], 'Reported closed score differs')
        # The native fixed-family field precedes zero cleanup. Keep it separate
        # from the explicitly stored point after cleanup.
        improved = None if returned is None or initial_score is None else returned < initial_score-config['gain_tol']
        if 'native_improved_initial' in e:
            require(bool(e['native_improved_initial']) == bool(improved), 'Native improvement flag differs')
        row.update(initial_score=initial_score,
                   current_to_initial_gain=current_to_initial,
                   initial_evidence='recomputed_point' if initial is not None else ('recorded_scalar' if initial_score is not None else 'unavailable'),
                   returned_score=returned, native_objective=native_objective, closed_score=closed_score,
                   pre_cleanup_qp_score=solver.get('fixed_family_qp_objective'),
                   native_gain=None if returned is None or initial_score is None else initial_score-returned,
                   closure_gain=None if returned is None or closed_score is None else returned-closed_score,
                   native_improved_initial=improved, initial_source=e.get('initial_source'),
                   pretrial_QP_count=e.get('pretrial_QP_count'),
                   status=solver.get('status'), timed_out=solver.get('status') == 9,
                   lower_bound=solver.get('bound'), gap=solver.get('gap'), seconds=solver.get('seconds'),
                   solve_started=solver.get('solve_started', False), scope=solver.get('scope'),
                   initial_record={k:v for k,v in solver.items() if k.startswith(('initial_', 'warm_', 'start_'))})
        events.append(row)
        if e.get('accepted'):
            require(e.get('point') is not None, 'Accepted event has no closed candidate')
            current = e['point']
    tops = [topology(p, truth, n, config['zero_tol']) for p in points] if truth is not None else None
    native_records = native.get('MIQP', [])
    return dict(status='complete', method=native['method'], config=config, reference_mse=reference,
                candidate_count=len(points), path=path, events=events, phases=phases,
                candidate_topologies=tops,
                coverage=None if tops is None else dict(internal=any(t['internal_exact'] for t in tops), full=any(t['full_exact'] for t in tops)),
                seconds=native.get('total_seconds'), common_seconds=native.get('common_seconds'),
                QP_count=len(native.get('QP', [])), MIQP_count=len(native_records),
                MIQP_timeouts=sum(r.get('status') == 9 for r in native_records),
                MIQP_without_incumbent=sum(r.get('solve_started', False) and r.get('objective') is None for r in native_records),
                native_bound_conflicts=native.get('native_bound_conflicts'),
                complete_RNJ_output_fallback=native.get('complete_RNJ_output_fallback'),
                output_floor=native.get('config', {}).get('internal_floor', 0.))


class Reader:
    def __init__(self):
        self.sources = {}

    def file(self, value, base=None):
        ref = value if isinstance(value, dict) else dict(path=str(value))
        path = Path(ref['path'])
        if not path.is_absolute() and base is not None and not path.exists():
            path = Path(base)/path
        path = path.resolve(); digest = sha(path)
        require(ref.get('sha256', digest) == digest, 'Source hash mismatch: '+str(path))
        self.sources[str(path)] = dict(path=str(path), sha256=digest)
        return path

    def json(self, value, base=None):
        return json.loads(self.file(value, base).read_text(encoding='utf-8-sig'))

    def arrays(self, value, base=None):
        with np.load(self.file(value, base), allow_pickle=False) as data:
            return {k:data[k].copy() for k in data.files}

    def verify_unchanged(self):
        for ref in self.sources.values():
            require(sha(ref['path']) == ref['sha256'], 'Source changed while comparing')


def manifest(reader, root):
    root = Path(root).resolve()
    protocol = reader.json(root/'protocol.json') if (root/'protocol.json').exists() else {}
    input_ref = protocol.get('inputs_ref', protocol.get('inputs'))
    if not isinstance(input_ref, (dict, str, Path)):
        input_ref = root/'inputs.json'
    value = reader.json(input_ref, root)
    # These are the immutable old/new snapshots actually imported by workers.
    # Do not substitute current repository files for historical paths.
    source_root = protocol.get('source_root')
    if source_root:
        for source in protocol.get('sources', []):
            path = reader.file(source, root)
            require(path.is_relative_to(Path(source_root).resolve()), 'Source is outside the declared runtime snapshot')
    entries = value.get('inputs', value.get('tasks')) if isinstance(value, dict) else value
    require(isinstance(entries, list), 'Inputs must be a list')
    by_id = {str(e['id']): e for e in entries}
    require(len(by_id) == len(entries), 'Duplicate matched input ID')
    return protocol, by_id


def read_case(reader, root, entry):
    root = Path(root)
    ref = entry.get('result')
    if ref is None:
        choices = [root/'cases'/entry['id']/'result.json', root/'inputs'/entry['id']/'result.json', root/'fits'/entry['id']/'result.json']
        ref = next((p for p in choices if p.exists()), choices[0])
    if not isinstance(ref, dict) and not Path(ref).exists():
        return dict(status='missing', reason='No completed result', result_path=str(ref)), None
    value = reader.json(ref, root)
    if 'frozen_candidates' in value:
        return dict(status='complete'), value
    if value.get('status') != 'complete':
        return value, None
    native_ref = value.get('native', value.get('native_ref', entry.get('native')))
    require(native_ref is not None, 'Completed result has no native reference')
    return value, reader.json(native_ref, root)


def entry_observations(reader, entry, base):
    files = entry.get('files', entry)
    values = {}
    for role in ('train', 'validation', 'new_day'):
        ref = files.get(role+'.npz', entry.get(role))
        values[role] = None if ref is None else observation(reader.arrays(ref, base))
    require(values['train'] is not None and values['validation'] is not None, 'Missing historical observations')
    return values


def paired_summary(rows):
    complete = [r for r in rows if r['status'] == 'complete']
    result = dict(planned_pairs=len(rows), completed_pairs=len(complete), phases={}, solver={})
    for phase in ('validation', 'cumulative'):
        usable = [r for r in complete if all(r[v]['phases'][phase] is not None and
                  r[v]['phases'][phase]['topology'] is not None for v in ('old', 'new'))]
        stats = dict(evaluated_pairs=len(usable), planned_pairs=len(rows))
        for kind in ('internal', 'full'):
            values = [(r['old']['phases'][phase]['topology'][kind+'_exact'],
                       r['new']['phases'][phase]['topology'][kind+'_exact']) for r in usable]
            stats[kind] = dict(old_correct=sum(a for a,b in values),new_correct=sum(b for a,b in values),
                               newly_recovered=sum(not a and b for a,b in values), lost=sum(a and not b for a,b in values))
        result['phases'][phase]=stats
    for version in ('old','new'):
        available=[r[version] for r in rows if (r.get(version) or {}).get('status')=='complete']
        events=[e for r in available for e in r['events'] if e['solve_started']]
        result['solver'][version]=dict(completed_fits=len(available),planned_fits=len(rows),
            MIQP_started=len(events),timeouts=sum(e['timed_out'] for e in events),
            known_initial_scores=sum(e['initial_score'] is not None for e in events),
            unknown_initial_scores=sum(e['initial_score'] is None for e in events),
            native_improved_initial=sum(e['native_improved_initial'] is True for e in events),
            native_no_incumbent=sum(e['returned_score'] is None for e in events))
    return result


def compare(old_run, new_run, out):
    out = Path(out)
    require(not out.exists(), 'Refusing to overwrite comparison output')
    out.mkdir(parents=True)
    reader = Reader(); old_protocol, old = manifest(reader, old_run); new_protocol, new = manifest(reader, new_run)
    ids = sorted(set(old)|set(new)); rows = []; errors = []
    for ident in ids:
        row = dict(id=ident, old=None, new=None)
        try:
            require(ident in old and ident in new, 'Unmatched planned input')
            observations = {}
            for version, root, entry in [('old', old_run, old[ident]), ('new', new_run, new[ident])]:
                observations[version] = entry_observations(reader, entry, root)
            for role in ('train', 'validation', 'new_day'):
                a,b = observations['old'][role],observations['new'][role]
                require((a is None) == (b is None), 'Held-out data availability differs')
                if a is not None:
                    require(a['labels'] == b['labels'] and all(np.array_equal(a[k], b[k]) for k in ('P','Q','Y')), 'Paired observations differ: '+role)
            for version, root, entry in [('old', old_run, old[ident]), ('new', new_run, new[ident])]:
                result, native = read_case(reader, root, entry)
                require(result.get('sources_unchanged', result.get('source_unchanged', True)), 'Fit source changed during execution')
                if native is None:
                    row[version] = dict(status=result.get('status','failed'), error=result.get('error',result.get('reason')))
                    continue
                truth = entry.get('truth_supports'); hstar = None
                physical_ref = entry.get('Hstar_ref', entry.get('Hstar', entry.get('files',{}).get('Hstar.npz')))
                if physical_ref is not None:
                    physical = reader.arrays(physical_ref, root);hstar = physical['Hstar']
                    require(list(map(int, physical['labels'])) == observations[version]['train']['labels'], 'Physical labels differ')
                    physical_truth = json.loads(str(physical['support_json']))
                    require(truth is None or {tuple(c) for c in truth} == {tuple(c) for c in physical_truth}, 'Two physical support references disagree')
                    truth = physical_truth
                row[version] = audit_native(native, **observations[version], truth=truth, hstar=hstar)
            row['status'] = 'complete' if all(row[v]['status']=='complete' for v in ('old','new')) else 'incomplete'
            if row['status'] == 'complete':
                require(row['old']['method'] == row['new']['method'], 'Paired methods differ')
                require(row['old']['config'] == row['new']['config'], 'Paired search settings differ')
                close(row['old']['reference_mse'],row['new']['reference_mse'],'Training reference changed',rtol=1e-12)
        except Exception as exc:
            row['status']='audit_failed';row['error']=str(exc);errors.append(dict(id=ident,error=str(exc)))
        rows.append(row)
    reader.verify_unchanged()
    summary = dict(status='passed' if not errors else 'failed', planned_pairs=len(ids),
        completed_pairs=sum(r['status']=='complete' for r in rows), errors=errors,
        scope='Known historical inputs; paired identifier rerun only. No new AC or DFS. No all-tree optimality claim.',
        old_run=str(Path(old_run).resolve()), new_run=str(Path(new_run).resolve()),
        old_protocol=old_protocol, new_protocol=new_protocol, rows=rows,
        paired_summary=paired_summary(rows),
        source_inputs=list(reader.sources.values()), comparison_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        created_utc=datetime.now(timezone.utc).isoformat())
    (out/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    columns=['id','version','status','paired_status','method','candidate_count','seconds','MIQP_count','MIQP_timeouts','validation_index','validation_mse','validation_internal','validation_full','cumulative_internal','cumulative_full','R_relative_error','X_relative_error']
    flat=[]
    for row in rows:
        for v in ('old','new'):
            data=row.get(v) or {}; item={k:data.get(k) for k in columns};item.update(id=row['id'],version=v,status=data.get('status',row['status']),paired_status=row['status'])
            if data.get('status')=='complete':
                val=data['phases']['validation'];cum=data['phases']['cumulative'];top=val['topology'] or {};m=val['matrix'] or {}
                item.update(validation_index=val['index'],validation_mse=val['mse'],validation_internal=top.get('internal_exact'),validation_full=top.get('full_exact'),
                    cumulative_internal=None if cum is None or cum['topology'] is None else cum['topology']['internal_exact'],
                    cumulative_full=None if cum is None or cum['topology'] is None else cum['topology']['full_exact'],**{k:m.get(k) for k in ('R_relative_error','X_relative_error')})
            flat.append(item)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(flat)
    lines=['# RNJ作用简化的配对比较','',f"独立重算：{summary['status']}；完整配对{summary['completed_pairs']}/{len(ids)}，核验错误{len(errors)}。",'',
        '本批使用已见历史输入，比较辨识器实现。恢复分别统计内部连接与全部活动支持；R/X为完整响应矩阵相对误差。没有运行新AC或灵活性调度。','',
        '|输入|版本|状态|验证内部/全部支持|验证MSE|R/X误差|用时秒|MIQP限时/次数|','|---|---|---|---|---:|---|---:|---|']
    for x in flat:
        f=lambda value:'—' if value is None else str(value)
        pct=lambda value:'—' if value is None else f'{100*value:.3f}%'
        lines.append(f"|{x['id']}|{x['version']}|{x['status']}|{f(x['validation_internal'])}/{f(x['validation_full'])}|{f(x['validation_mse'])}|{pct(x['R_relative_error'])}/{pct(x['X_relative_error'])}|{f(x['seconds'])}|{f(x['MIQP_timeouts'])}/{f(x['MIQP_count'])}|")
    lines += ['', '|评价数据|已评价配对/计划|内部正确：旧→新|内部新增/丢失|全部支持正确：旧→新|全部支持新增/丢失|',
              '|---|---|---|---|---|---|']
    for phase, stats in summary['paired_summary']['phases'].items():
        i, f=stats['internal'],stats['full']
        lines.append(f"|{phase}|{stats['evaluated_pairs']}/{stats['planned_pairs']}|{i['old_correct']}→{i['new_correct']}|{i['newly_recovered']}/{i['lost']}|{f['old_correct']}→{f['new_correct']}|{f['newly_recovered']}/{f['lost']}|")
    lines += ['', '初值收益=阶段开始的当前指标−实际提交初值指标；原生收益=提交初值指标−MIQP返回指标；收尾收益=MIQP返回指标−固定支持QP指标。单个RNJ参考初值带来的收益与MIQP进一步改善分开。旧版未记录初值时标为缺失，不从最终结果反推初值。限时记录保留，其下界只适用于对应的一次加边域或有限支持池。', '', '|输入|版本|阶段|初值来源|初值收益|初值|原生返回|原生收益|QP收尾收益|状态|差距|','|---|---|---|---|---:|---:|---:|---:|---:|---|---:|']
    for row in rows:
        for v in ('old','new'):
            for e in (row.get(v) or {}).get('events',[]):
                lines.append('|'+ '|'.join(str(q) if q is not None else '—' for q in (row['id'],v,e['stage'],e['initial_source'],e['current_to_initial_gain'],e['initial_score'],e['returned_score'],e['native_gain'],e['closure_gain'],e['status'],e['gap']))+'|')
    if errors: lines += ['', '核验失败：', *[f"- {e['id']}：{e['error']}" for e in errors]]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'complete.json').write_text(json.dumps(dict(status=summary['status'],report=dict(path=str((out/'report.json').resolve()),sha256=sha(out/'report.json'))),indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run',required=True);parser.add_argument('--new-run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();report=compare(args.old_run,args.new_run,args.out)
    print(json.dumps({k:report[k] for k in ('status','planned_pairs','completed_pairs')},ensure_ascii=False))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
