"""Small, saved-data-only effect and calibrated-decision summaries.

Inputs are ev_compact_analysis CSV/JSON and, optionally, ev_compact_collect
risk JSON. No fitting, action selection, calibration or optimization is run.
Each contrast contributes one paired measurement per independent history.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median

SCHEMA = 'ev_compact_effect_summary_v1'
PRIMARY = ('refined_v2g_mobile', 'capped_v2g_mobile')
COMPARATORS = ('constant', 'random_mobile')
METHODS = (*PRIMARY, *COMPARATORS)
OPTIMIZED_METHODS = ('charge_fixed', 'v2g_fixed', 'charge_mobile', 'v2g_mobile', *PRIMARY)
METRICS = ('actual_M', 'noiseless_M', 'R_relative', 'X_relative')
IDENTITY = ('phase', 'policy_id', 'family', 'domain', 'prediction',
            'history_sha256', 'truth_sha256', 'noise_sha256',
            'measurement_sha256', 'device_sha256', 'data_sha256')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(value):
    if value is None or isinstance(value, bool) or value == '':
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value):
    if isinstance(value, bool):
        return value
    if value in ('True', 'true', '1', 1):
        return True
    if value in ('False', 'false', '0', 0):
        return False
    if value in ('', None):
        return None
    raise ValueError('invalid saved boolean: '+str(value))


def _stats(values, tolerance=1e-9):
    values = [v for value in values if (v := _number(value)) is not None]
    return dict(n=len(values), mean=mean(values) if values else None,
                median=median(values) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                positive=sum(v > tolerance for v in values),
                negative=sum(v < -tolerance for v in values),
                tied=sum(abs(v) <= tolerance for v in values),
                sign_tolerance=tolerance)


def _complete(row):
    return row.get('status') == 'complete' and _number(row.get('actual_M')) is not None


def _metric(row, key):
    if key == 'actual_M' and not _complete(row):
        return None
    if key == 'noiseless_M' and _bool(row.get('noiseless_complete')) is not True:
        return None
    return _number(row.get(key))


def _method_summary(rows):
    return dict(history_denominator=len(rows), actual_complete=sum(_complete(r) for r in rows),
        failed_or_missing=sum(not _complete(r) for r in rows),
        status_counts=dict(sorted(Counter(r.get('status', 'missing_method') for r in rows).items())),
        candidate_miss_count=sum(_bool(r.get('candidate_miss')) is True for r in rows),
        operation_failure_count=sum(_bool(r.get('operation_feasible')) is False for r in rows),
        selection_failure_history_count=sum((_number(r.get('selection_failures')) or 0) > 0 for r in rows),
        direct_new_MILP_count=sum(_bool(r.get('selected_is_new_MILP_proposal')) is True for r in rows),
        direct_MILP_origin_count=sum(r.get('selected_origin') == 'MILP' for r in rows),
        from_MILP_count=sum(r.get('source_lineage_terminal_kind') == 'MILP' for r in rows),
        ancestral_MILP_count=sum(r.get('source_lineage_terminal_kind') == 'MILP'
            and r.get('source_lineage_terminal_method') != r['method'] for r in rows),
        ancestral_new_MILP_count=sum(r.get('source_lineage_terminal_new_MILP') is True
            and r.get('source_lineage_terminal_method') != r['method'] for r in rows),
        source_lineage_unknown_count=sum(r.get('source_lineage_status') == 'unknown' for r in rows),
        source_lineage_random_path_count=sum(r.get('source_lineage_status') == 'forbidden_random' for r in rows),
        selected_origin_counts=dict(sorted(Counter(r.get('selected_origin') or 'unrecorded' for r in rows).items())),
        proxy_completed_count=sum(_bool(r.get('proxy_completed')) is True for r in rows),
        metrics={key: _stats(_metric(r, key) for r in rows) for key in METRICS})


def source_lineage(method, methods):
    """Follow saved parent origins within the same H, never infer an unseen one.

    The CSV must contain each traversed parent's selected_origin. A quality or
    constant origin is a terminal public control rule; MILP records preserve
    the separate flag telling whether that domain actually added a new curve.
    """
    path, seen, current = [], set(), method
    while True:
        if current in seen:
            return dict(path=path+[current], status='unknown', reason='origin_cycle',
                        terminal_kind='unknown', terminal_method=None, terminal_new_MILP=None)
        seen.add(current)
        path.append(current)
        if current == 'random' or current.startswith('random_'):
            forbidden = method in OPTIMIZED_METHODS
            return dict(path=path, status='forbidden_random' if forbidden else 'resolved',
                        reason='random_in_primary_lineage' if forbidden else None,
                        terminal_kind='random', terminal_method=path[-2] if len(path)>1 else current,
                        terminal_new_MILP=None)
        row = methods.get(current)
        if row is None:
            return dict(path=path, status='unknown', reason='missing_parent_method',
                        terminal_kind='unknown', terminal_method=None, terminal_new_MILP=None)
        if current == 'none':
            return dict(path=path, status='resolved', reason=None,
                        terminal_kind='none', terminal_method=current, terminal_new_MILP=None)
        if current == 'constant' or current.startswith('quality'):
            return dict(path=path, status='resolved', reason=None,
                        terminal_kind='constant' if current == 'constant' else 'quality',
                        terminal_method=current, terminal_new_MILP=None)
        origin = row.get('selected_origin')
        if origin == 'MILP':
            return dict(path=path+['MILP'], status='resolved', reason=None,
                        terminal_kind='MILP', terminal_method=current,
                        terminal_new_MILP=_bool(row.get('selected_is_new_MILP_proposal')))
        if origin == 'constant' or (isinstance(origin, str) and origin.startswith('quality')):
            return dict(path=path+[origin], status='resolved', reason=None,
                        terminal_kind='constant' if origin == 'constant' else 'quality',
                        terminal_method=current, terminal_new_MILP=None)
        if not origin:
            if current == 'constant' or current.startswith('quality'):
                return dict(path=path, status='resolved', reason=None,
                            terminal_kind='constant' if current == 'constant' else 'quality',
                            terminal_method=current, terminal_new_MILP=None)
            return dict(path=path, status='unknown', reason='missing_selected_origin',
                        terminal_kind='unknown', terminal_method=None, terminal_new_MILP=None)
        current = str(origin)


def crosscheck_lineages(rows, reference_trace):
    """Compare two independently implemented recursions on saved CSV rows.

    ``reference_trace(method, same_history_rows)`` is supplied by the caller;
    the reporting module never imports the experiment or analysis programs.
    No unknown or random source is silently recoded as non-MILP evidence.
    """
    grouped = defaultdict(dict)
    for row in rows:
        seed, method = str(row['seed']), row['method']
        if method in grouped[seed]:
            raise ValueError('duplicate independent history/method')
        grouped[seed][method] = row
    records, differences = [], []
    reason_map = dict(origin_cycle='cycle', missing_parent_method='missing_parent',
                      missing_selected_origin='missing_origin', random_in_primary_lineage='random_in_primary_chain')
    for seed, bank in sorted(grouped.items()):
        for method, row in bank.items():
            local, other = source_lineage(method, bank), reference_trace(method, bank)
            from_milp = local['terminal_kind'] == 'MILP'
            ancestor = from_milp and local['terminal_method'] != method
            own = dict(path=local['path'], terminal=local['terminal_kind'],
                terminal_method=local['terminal_method'], from_MILP=from_milp,
                ancestral_MILP=ancestor, terminal_MILP_new_curve=local['terminal_new_MILP'],
                ancestral_new_MILP=(local['terminal_new_MILP'] if ancestor else False),
                error=reason_map.get(local['reason'], local['reason']))
            expected = dict(path=other['source_lineage'], terminal=other['source_lineage_terminal'],
                terminal_method=other['source_lineage_terminal_method'], from_MILP=other['source_lineage_from_MILP'],
                ancestral_MILP=other['source_lineage_ancestral_MILP'],
                terminal_MILP_new_curve=other['source_lineage_terminal_MILP_new_curve'],
                ancestral_new_MILP=other['source_lineage_ancestral_new_MILP'], error=other['source_lineage_error'])
            changed = [key for key in own if own[key] != expected[key]]
            direct_new = _bool(row.get('selected_is_new_MILP_proposal')) is True
            if direct_new and row.get('selected_origin') != 'MILP':
                changed.append('new_curve_flag_without_direct_MILP_origin')
            record = dict(seed=seed, method=method, direct_new_MILP=direct_new,
                          local=own, reference=expected, differences=changed)
            records.append(record)
            if changed:
                differences.append(dict(seed=seed, method=method, fields=changed))
    return dict(success=not differences, independent_histories=len(grouped),
                method_conditions=len(records), differences=differences, records=records,
                scope='saved source-lineage recursion; no fitting, selection or optimization')


def _pair_summary(rows):
    return dict(history_denominator=len(rows),
                complete_actual_pairs=sum(r['complete_actual_pair'] for r in rows),
                metrics={key: _stats(r['delta_'+key] for r in rows) for key in METRICS})


def summarize_rows(rows, *, metadata=None, risk=None):
    """Summarize one phase/policy; rows from different methods stay paired.

    Missing method rows are explicit failures for the histories that exist.
    Histories never executed are reported separately using analysis metadata.
    R/X errors are those saved under the true-candidate model; their available
    count can differ from the complete all-candidate fit count.
    """
    rows, metadata = list(rows), metadata or {}
    if not rows:
        raise ValueError('no saved method rows')
    by_history = defaultdict(dict)
    for original in rows:
        row = dict(original)
        seed, method = str(row.get('seed', '')), row.get('method')
        if not seed or not method or method in by_history[seed]:
            raise ValueError('missing or duplicate independent history/method')
        by_history[seed][method] = row
    phases = {r.get('phase') for r in rows}
    policies = {r.get('policy_id') for r in rows}
    if len(phases) != 1 or not next(iter(phases)) or len(policies) != 1 or not next(iter(policies)):
        raise ValueError('one phase and frozen policy required')
    phase, policy_id = next(iter(phases)), next(iter(policies))
    seeds = sorted(by_history, key=lambda s: (len(s), s))
    if metadata.get('phase', phase) != phase or metadata.get('policy_id', policy_id) != policy_id:
        raise ValueError('analysis metadata phase/policy mismatch')
    if 'seeds' in metadata and set(map(str, metadata['seeds'])) != set(seeds):
        raise ValueError('analysis metadata history coverage mismatch')
    for seed, methods in by_history.items():
        first = next(iter(methods.values()))
        for row in methods.values():
            if any(row.get(key) != first.get(key) for key in IDENTITY):
                raise ValueError('paired history inputs differ: '+seed)
        for method in METHODS:
            if method not in methods:
                methods[method] = dict({key: first.get(key) for key in IDENTITY},
                    seed=seed, method=method, status='missing_method')
        for method in METHODS:
            lineage = source_lineage(method, methods)
            methods[method].update({'source_lineage_'+key:value for key,value in lineage.items()})
    method_rows = [dict(by_history[s][m]) for s in seeds for m in METHODS]
    pairs = []
    for seed in seeds:
        for primary in PRIMARY:
            for comparator in COMPARATORS:
                left, right = by_history[seed][primary], by_history[seed][comparator]
                record = dict(seed=seed, family=left.get('family'), primary=primary,
                    comparator=comparator, primary_status=left['status'], comparator_status=right['status'],
                    complete_actual_pair=_complete(left) and _complete(right),
                    primary_origin=left.get('selected_origin') or 'unrecorded',
                    primary_is_new_MILP=_bool(left.get('selected_is_new_MILP_proposal')),
                    primary_lineage=left['source_lineage_path'],
                    primary_lineage_status=left['source_lineage_status'],
                    primary_lineage_terminal_kind=left['source_lineage_terminal_kind'],
                    primary_candidate_miss=_bool(left.get('candidate_miss')),
                    comparator_candidate_miss=_bool(right.get('candidate_miss')))
                for key in METRICS:
                    lv, rv = _metric(left, key), _metric(right, key)
                    record['primary_'+key], record['comparator_'+key] = lv, rv
                    record['delta_'+key] = lv-rv if lv is not None and rv is not None else None
                pairs.append(record)
    contrasts = {}
    for primary in PRIMARY:
        for comparator in COMPARATORS:
            selected = [r for r in pairs if r['primary'] == primary and r['comparator'] == comparator]
            contrasts[primary+'_vs_'+comparator] = _pair_summary(selected)
    families = {}
    for family in sorted({str(next(iter(v.values())).get('family', 'unrecorded')) for v in by_history.values()}):
        subset = [r for r in pairs if str(r['family']) == family]
        families[family] = {p+'_vs_'+c: _pair_summary([r for r in subset if r['primary'] == p and r['comparator'] == c])
                            for p in PRIMARY for c in COMPARATORS}
    planned = int(metadata.get('planned_histories', len(seeds)))
    if planned < len(seeds):
        raise ValueError('planned history count below observed count')
    result = dict(schema=SCHEMA, phase=phase, policy_id=policy_id, history_ids=seeds,
        independent_histories=len(seeds), planned_histories=planned,
        pending_histories=planned-len(seeds), method_measurements=len(method_rows),
        paired_measurements=len(pairs),
        independent_unit='one history; all methods and contrasts remain paired within it',
        descriptive_only=True, strategy_updated=False, fits_or_optimizers_called=0,
        scope='saved local full-AC attained losses; declared simulation distribution',
        metric_definitions=dict(actual_M='minimum wrong-candidate SSE minus true-candidate SSE',
            noiseless_M='same margin with future noise removed and history held fixed',
            R_relative='saved relative R error under the true-candidate model',
            X_relative='saved relative X error under the true-candidate model',
            delta='primary minus comparator; positive M and negative R/X changes favor primary'),
        methods={m: _method_summary([by_history[s][m] for s in seeds]) for m in METHODS},
        contrasts=contrasts, family_contrasts=families, paired_rows=pairs,
        method_rows=method_rows, risk=dict(available=False, reason='no saved collector risk report supplied'))
    if risk is not None:
        result['risk'] = _risk_summary(risk, result, metadata)
    return result


def _risk_summary(risk, effects, metadata):
    bundle = risk.get('standard_records') or {}
    if (bundle.get('schema') != 'ev_compact_standard_records_v1'
            or _hash({k:v for k,v in bundle.items() if k != 'bundle_id'}) != bundle.get('bundle_id')):
        raise ValueError('collector standard-record hash/schema mismatch')
    if bundle.get('spec', {}).get('policy_id') != effects['policy_id']:
        raise ValueError('risk and effect policy differ')
    phase = effects['phase']
    if phase not in ('evaluation', 'calibration'):
        raise ValueError('risk cannot be attached to development results')
    records = bundle.get(phase, [])
    ids = [str(r['history_id']) for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(effects['history_ids']):
        raise ValueError('risk and effect history coverage differ')
    manifest = {r['name']:r['sha256'] for r in bundle.get('raw_manifest', {}).get(phase, [])}
    analysis_hashes = {r['name']:r['sha256'] for r in metadata.get('trial_hashes', [])}
    if analysis_hashes and manifest != analysis_hashes:
        raise ValueError('risk and analysis raw-trial hashes differ')
    out = dict(available=True, phase=phase, alpha=risk.get('alpha'),
        bundle_id=bundle['bundle_id'], raw_trial_hashes_matched=bool(analysis_hashes),
        planned_calibration=risk.get('planned_calibration'),
        actual_calibration=risk.get('actual_calibration'),
        planned_evaluation=risk.get('planned_evaluation'),
        actual_evaluation=risk.get('actual_evaluation'),
        scope='marginal over calibration and one new H in the declared reference distribution',
        fixed_threshold_risk_confidence_claim=False, topology_probability=False,
        thresholds={}, decisions=[], methods={}, primary_family=None)
    for label in ('individual_policies', 'primary_family'):
        group = risk.get(label) or {}
        calibration = group.get('calibration') or {}
        spec = calibration.get('spec') or {}
        if spec.get('policy_id') != effects['policy_id']:
            raise ValueError('risk subgroup policy differs')
        if _hash({k:v for k,v in calibration.items() if k != 'calibration_id'}) != calibration.get('calibration_id'):
            raise ValueError('saved calibration content/hash mismatch')
        if label == 'primary_family' and (set(spec.get('methods', [])) != set(PRIMARY) or not spec.get('familywise')):
            raise ValueError('primary-family methods differ')
        thresholds = []
        for stratum, candidates in calibration.get('thresholds', {}).items():
            for candidate, methods in candidates.items():
                for method, q in methods.items():
                    if label == 'primary_family' or method in METHODS:
                        thresholds.append(dict(stratum=stratum, candidate=candidate, method=method,
                            **{k:q.get(k) for k in ('n', 'rank', 'threshold', 'infinite', 'reason', 'rank_marginal_error_bound')}))
        out['thresholds'][label] = thresholds
        if phase != 'evaluation':
            continue
        evaluation = group.get('evaluation') or {}
        if evaluation.get('calibration_id') != calibration['calibration_id']:
            raise ValueError('evaluation calibration identity differs')
        evaluated = evaluation.get('records', [])
        if len(evaluated) != len(ids) or {str(r['history_id']) for r in evaluated} != set(ids):
            raise ValueError('risk decision history coverage differs')
        selected_methods = METHODS if label == 'individual_policies' else PRIMARY
        summaries = {m: Counter(history_denominator=0, known_covered_histories=0, false_rejections=0, unique_correct=0,
            unresolved=0, computational_failures=0, candidate_misses=0, wrong_candidates_excluded=0)
            for m in selected_methods}
        family_false, family_unresolved, family_covered = 0, 0, 0
        for record in evaluated:
            one_false, one_unresolved = False, False
            miss, true = bool(record.get('candidate_miss')), record.get('true_name')
            for method in selected_methods:
                if method not in record.get('methods', {}):
                    raise ValueError('missing saved risk method decision')
                decision = record['methods'][method]
                if miss and decision.get('false_rejection') is not None:
                    raise ValueError('candidate-miss decision must have undefined false rejection')
                false = decision.get('false_rejection') is True
                unresolved = decision.get('unresolved') is True
                excluded = sum(a != true for a in decision.get('rejected', [])) if not miss else 0
                values = dict(history_denominator=1, known_covered_histories=int(not miss), false_rejections=int(false),
                    unique_correct=int(decision.get('unique_correct') is True), unresolved=int(unresolved),
                    computational_failures=int(decision.get('compute_failure') is True),
                    candidate_misses=int(miss), wrong_candidates_excluded=excluded)
                summaries[method].update(values)
                out['decisions'].append(dict(group=label, seed=str(record['history_id']), method=method,
                    candidate_miss=miss, false_rejection=None if miss else false,
                    unique_correct=decision.get('unique_correct'), unresolved=unresolved,
                    compute_failure=decision.get('compute_failure'), wrong_candidates_excluded=excluded,
                    retained=decision.get('retained'), rejected=decision.get('rejected')))
                one_false |= false
                one_unresolved |= unresolved
            family_false += one_false
            family_unresolved += one_unresolved
            family_covered += not miss
        if label == 'individual_policies':
            out['methods'] = {m:dict(v) for m,v in summaries.items()}
        else:
            out['primary_family'] = dict(history_denominator=len(ids),
                known_covered_histories=family_covered, candidate_misses=len(ids)-family_covered,
                any_primary_false_rejection=family_false,
                any_primary_unresolved=family_unresolved,
                methods={m:dict(v) for m,v in summaries.items()})
    return out


def _read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def summarize_directory(directory, *, risk_path=None):
    directory = Path(directory)
    csv_path = directory/'method_rows.csv'
    meta_path = directory/'summary.json'
    raw = csv_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    metadata = _read_json(meta_path) if meta_path.exists() else {}
    result = summarize_rows(rows, metadata=metadata, risk=_read_json(risk_path) if risk_path else None)
    result['input_sha256'] = {'method_rows.csv': hashlib.sha256(raw).hexdigest()}
    if meta_path.exists():
        result['input_sha256']['summary.json'] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    if risk_path:
        result['input_sha256'][str(Path(risk_path).resolve())] = hashlib.sha256(Path(risk_path).read_bytes()).hexdigest()
    result['assistant_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def markdown(result):
    def fmt(value):
        return '—' if value is None else f'{value:.4g}'
    lines = ['# EV独立历史效果摘要', '',
        f"阶段：{result['phase']}；完成独立历史 {result['independent_histories']}/{result['planned_histories']}，待完成 {result['pending_histories']}。",
        '同一历史内的方法和四项对照为配对测量；汇总分母均为历史数。', '',
        '| 方法 | 完整/历史 | 实际M均值 | 无噪M均值 | R误差均值 | X误差均值 | 本层新MILP | 继承MILP | 失败 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, rec in result['methods'].items():
        m = rec['metrics']
        lines.append(f"| {method} | {rec['actual_complete']}/{rec['history_denominator']} | {fmt(m['actual_M']['mean'])} | {fmt(m['noiseless_M']['mean'])} | {fmt(m['R_relative']['mean'])} | {fmt(m['X_relative']['mean'])} | {rec['direct_new_MILP_count']} | {rec['ancestral_MILP_count']} | {rec['failed_or_missing']} |")
    lines += ['', '本层新MILP要求保存的新增曲线标志；继承MILP沿同历史父方法追到MILP终点。from_MILP_count另含直接与继承两者。缺失父来源或循环记unknown，随机进入主方法来源链单列。',
        'M是最小错误候选SSE减真候选SSE；R/X是在真候选模型下的相对误差。各指标有效数及原始失败保存在JSON。', '',
        '| 主方法−对照 | 实际M差均值 | 中位数 | 正/平/负 | 有效配对/历史 | 无噪M差均值 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, rec in result['contrasts'].items():
        m, n = rec['metrics']['actual_M'], rec['metrics']['noiseless_M']
        lines.append(f"| {name} | {fmt(m['mean'])} | {fmt(m['median'])} | {m['positive']}/{m['tied']}/{m['negative']} | {m['n']}/{rec['history_denominator']} | {fmt(n['mean'])} |")
    risk = result['risk']
    if not risk['available']:
        lines += ['', '未附校准报告：不从M正负或预测分数推算误拒、排错、未决。']
    elif risk['phase'] == 'calibration':
        lines += ['', '当前为校准数据：仅保存各候选阈值与有效校准数，评价决策另列。']
    else:
        lines += ['', '以下采用collector保存的逐方法校准决策。排错量为被排除的错误候选个数；唯一正确为仅保留真候选。', '',
            '| 方法 | 误拒/含真历史 | 全部历史 | 候选漏真 | 唯一正确 | 未决 | 计算失败 | 排除错误候选数 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method, rec in risk['methods'].items():
            lines.append(f"| {method} | {rec['false_rejections']}/{rec['known_covered_histories']} | {rec['history_denominator']} | {rec['candidate_misses']} | {rec['unique_correct']} | {rec['unresolved']} | {rec['computational_failures']} | {rec['wrong_candidates_excluded']} |")
        family = risk['primary_family']
        if family:
            lines += ['', f"两主方法族使用共同校准阈值：至少一次误拒 {family['any_primary_false_rejection']}/{family['known_covered_histories']} 条含真历史；全部历史 {family['history_denominator']}，漏真 {family['candidate_misses']}。该分母仍为独立历史。"]
        lines += ['', '校准范围是冻结算法与声明模拟分布；风险保证平均校准数据和下一历史，不是固定一次阈值的高置信风险结论。']
    lines += ['', '这里只读取保存文件，不运行拟合、选择或优化，不用校准与评价结果调整策略。', '']
    return '\n'.join(lines)


def _write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in row.items()})


def write_summary(directory, output, *, risk_path=None):
    result = summarize_directory(directory, risk_path=risk_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (output/'summary.md').write_text(markdown(result), encoding='utf-8')
    _write_csv(output/'paired_primary.csv', result['paired_rows'])
    _write_csv(output/'risk_decisions.csv', result['risk'].get('decisions', []))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--risk', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = write_summary(args.analysis, args.output, risk_path=args.risk)
    print(json.dumps(dict(phase=result['phase'], independent_histories=result['independent_histories'],
        paired_measurements=result['paired_measurements'], risk_available=result['risk']['available'])))


if __name__ == '__main__':
    main()
