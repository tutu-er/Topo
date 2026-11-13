"""Read-only aggregate reporting; no selection or numerical solver calls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np
from scipy.stats import t

from ... import io
from .study import collect, qualify, paired_summary


def mean(values):
    a = [v for v in values if v is not None]
    return None if not a else float(np.mean(a))


def interval(values):
    if len(values) < 2:
        return None
    a = np.asarray(values)
    half = float(t.ppf(.975, len(a)-1)*np.std(a, ddof=1)/np.sqrt(len(a)))
    return [float(a.mean()-half), float(a.mean()+half)]


def summary(rows):
    valid = [r for r in rows if r['status'] == 'complete']
    covered = [r for r in valid if r['p_true'] is not None]
    return dict(attempted_rows=len(rows), completed_rows=len(valid), failed_rows=len(rows)-len(valid),
        truth_covered_rows=len(covered),
        mean_p_true=mean([r['p_true'] for r in covered]),
        p_true_80_count=sum(r['p_true'] >= .8 for r in covered),
        p_true_90_count=sum(r['p_true'] >= .9 for r in covered),
        p_true_95_count=sum(r['p_true'] >= .95 for r in covered),
        posterior_correct_count=sum(r['map_correct'] for r in valid),
        native_correct_count=sum(r['native_map_correct'] for r in valid),
        mean_minimum_logodds=mean([r['target_minimum_logodds'] for r in valid]),
        mean_planned_fee_yuan=mean([r['planned_fee_yuan'] for r in valid]),
        mean_settled_fee_yuan=mean([r['settled_fee_yuan'] for r in valid]),
        max_planned_fee_yuan=max((r['planned_fee_yuan'] for r in valid), default=None),
        mean_delivered_abs_kwh=mean([r['delivered_abs_kwh'] for r in valid]),
        voltage_cells_outside_095_105=sum(r['voltage_cells_outside_screen'] for r in valid),
        device_audit_failures=sum(not r['device']['passed'] for r in valid if r['device'] is not None))


def paired(rows, strategy):
    by = {(r['id'], r['day'], r['strategy']): r for r in rows if r['status'] == 'complete'}
    pairs = [(by[(r['id'], r['day'], 'no_dfs')], r) for r in rows if r['status'] == 'complete' and r['strategy'] == strategy
             and (r['id'], r['day'], 'no_dfs') in by and r['p_true'] is not None]
    posterior = [b['p_true']-a['p_true'] for a, b in pairs]
    logs = [b['target_minimum_logodds']-a['target_minimum_logodds'] for a, b in pairs if a['target_minimum_logodds'] is not None]
    runner = [b['target_runner_logodds']-a['target_runner_logodds'] for a, b in pairs if a['target_runner_logodds'] is not None]
    return dict(paired_days=len(pairs), mean_p_true_gain=mean(posterior), descriptive_paired_mean_95_interval=interval(posterior),
        mean_min_logodds_gain=mean(logs), mean_runner_logodds_gain=mean(runner),
        improved_days=sum(x > 1e-12 for x in posterior), decreased_days=sum(x < -1e-12 for x in posterior),
        decrease_over_1pp_days=sum(x < -.01 for x in posterior),
        mean_gain_pp=None if not posterior else float(100*np.mean(posterior)),
        minimum_gain_pp=None if not posterior else float(100*min(posterior)),
        maximum_gain_pp=None if not posterior else float(100*max(posterior)),
        mean_planned_fee_yuan=mean([b['planned_fee_yuan'] for a, b in pairs]),
        mean_delivered_abs_kwh=mean([b['delivered_abs_kwh'] for a, b in pairs]))


def solver_summary(out, phase):
    records = [p for f in sorted((out/phase).glob('*/day*/plans.json')) for p in io.read(f).values()]
    feasible = [p for p in records if p['feasible']]
    return dict(quotes=len(records), feasible_quotes=len(feasible), failed_or_unresolved=len(records)-len(feasible),
        status_counts=dict(Counter(p['status'] for p in records)),
        source_counts=dict(Counter(p.get('source', 'no_feasible_point') for p in records)),
        globally_closed_count=sum(p.get('global_optimality_verified', False) for p in feasible),
        verified_initial_count=sum(p.get('initial', {}).get('evidence_feasible', False) and p.get('initial', {}).get('domain_feasible', False) for p in records),
        retained_initial_count=sum(p.get('source') == 'supporting_LP_initial_point_retained' for p in records),
        native_time_limit_count=sum(p.get('native_status') == 9 for p in records),
        mean_solver_seconds=mean([p['seconds'] for p in records]),
        mean_relative_fee_gap=mean([p.get('relative_gap') for p in feasible]),
        max_relative_fee_gap=max((p.get('relative_gap', 0) for p in feasible), default=None),
        mean_verified_initial_fee_improvement_yuan=mean([p.get('improvement_from_verified_initial_yuan') for p in feasible]),
        same_planned_fee_failures=sum(not p.get('rotation_same_planned_fee', True) for p in records))


def make(out, destination):
    destination.mkdir(parents=True, exist_ok=True)
    history = qualify(out)
    fitted = [r for r in history if r['status'] == 'fitted']
    hsummary = dict(total_conditions=len(history), fitted=len(fitted),
        status_counts=dict(Counter(r['status'] for r in history)),
        qualification_counts=dict(Counter(r.get('qualification', r['status']) for r in history)),
        native_stop_counts=dict(Counter(r['native_stop_reason'] for r in fitted)),
        search_budget_reached=sum(r['search_budget_reached'] for r in fitted),
        method_cap_exceeded=sum(r['method_cap_exceeded'] for r in fitted),
        fit_total_seconds=sum(r['fit_seconds'] for r in history if 'fit_seconds' in r))
    phases = {}
    per_case = []
    for phase in ('development', 'confirmation'):
        rows = collect(out, phase)
        strategies = sorted({r['strategy'] for r in rows if r['strategy'] != 'all'})
        phases[phase] = dict(independent_new_days=len({(r['id'], r['day']) for r in rows}),
            historical_conditions=len({r['id'] for r in rows}),
            strategy={s: summary([r for r in rows if r['strategy'] == s]) for s in strategies},
            paired_with_no_dfs={s: paired(rows, s) for s in strategies if s != 'no_dfs'},
            solvers=solver_summary(out, phase),
            phase_failures=[io.read(p) for p in (out/phase).glob('*/complete.json') if io.read(p).get('failed_rows', 0)],
            development_primary_pairs=paired_summary(rows))
        for identity in sorted({r['id'] for r in rows}):
            selected = [r for r in rows if r['id'] == identity]
            first = selected[0]
            for strategy in strategies:
                values = summary([r for r in selected if r['strategy'] == strategy])
                compare = {} if strategy == 'no_dfs' else paired(selected, strategy)
                per_case.append(dict(phase=phase, id=identity, strategy=strategy,
                    qualification=first['qualification'], adoption=first['adoption'], history_points=first['history_points'],
                    voltage_sigma=first['voltage_sigma'], profile=first['profile'], historical_p_map=first['historical_p_map'],
                    **values, paired=compare))
    worker_failures = [io.read(f) for f in out.glob('*checkpoint.json') if any(r.get('status') == 'worker_failed' for r in io.read(f).get('receipts', []))]
    result = dict(source_protocol=io.ref(out/'protocol.json'), report_source=io.ref(__file__), history=hsummary,
        phases=phases, worker_failure_checkpoints=worker_failures,
        scope='Development-selected conditional examples; independent confirmation is separate. All rates use actual AC plus forecast, response and meter errors. Frozen candidate posterior is a plug-in model score.')
    io.write(destination/'summary.json', result)
    io.write(destination/'per_case.json', per_case)
    safe_history = [{k: r.get(k) for k in ('id', 'adoption', 'history_points', 'voltage_sigma', 'profile', 'history_seed',
        'status', 'fit_seconds', 'truth_present', 'qualification', 'p_map', 'p_true', 'original_winner_correct',
        'posterior_MAP_correct', 'distinct_topologies', 'native_stop_reason', 'search_budget_reached', 'method_cap_exceeded')} for r in history]
    io.write(destination/'history_summary.json', safe_history)
    fields = list(safe_history[0]) if safe_history else []
    with (destination/'history_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(safe_history)
    lines = ['# 灵活性弱证据扫描结果', '',
        f"历史条件 {len(history)}；完成辨识 {len(fitted)}。资格统计：{hsummary['qualification_counts']}。", '',
        '各条件共用部分历史随机流；条件数不等于独立历史样本数。32日确认针对开发阶段选出的条件。', '']
    for phase, label in [('development', '开发'), ('confirmation', '独立确认')]:
        p = phases[phase]
        lines.extend([f"{label}：{p['historical_conditions']}个历史条件，{p['independent_new_days']}条独立新日。", '',
            '|策略|完成/尝试|平均真拓扑后验|后验≥90%|完整恢复|平均计划费/元|', '|---|---:|---:|---:|---:|---:|'])
        for name, s in p['strategy'].items():
            prob = '—' if s['mean_p_true'] is None else f"{s['mean_p_true']:.6f}"
            fee = '—' if s['mean_planned_fee_yuan'] is None else f"{s['mean_planned_fee_yuan']:.6f}"
            lines.append(f"|{name}|{s['completed_rows']}/{s['attempted_rows']}|{prob}|{s['p_true_90_count']}/{s['truth_covered_rows']}|{s['native_correct_count']}/{s['completed_rows']}|{fee}|")
        lines.append('')
        lines.append(f"求解统计：{p['solvers']}")
        lines.append('')
    lines.extend(['逐例数据见per_case.json，缺真候选与错误winner见history_summary.csv。', '',
        '费用为预设资源报价下的激活补偿，不是整网日运营成本；没有可用线路额定容量，未核查热限。源码、配置、预算和私有原始轨迹的哈希随protocol与各日封存记录保存。', ''])
    (destination/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = make(args.out.resolve(), args.destination.resolve())
    print(result['history'])
