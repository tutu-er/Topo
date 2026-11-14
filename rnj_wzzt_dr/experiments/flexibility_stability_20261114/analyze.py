"""Aggregate completed independent audits; never import experiment runtime.

Only sealed JSON/NPZ files are read. No optimization, AC solve, fitting or
production score is called. Separate phases and resource variants retain their
own denominators; repeated errors on one known load day are not new networks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


POLICIES = ('no_dfs', 'anchor', 'activation', 'stable_activation', 'capacity', 'capacity_rotation')
VARIANTS = ('original_p10_q36', 'expanded_p10_q36')
P_ENERGY_TOLERANCE = 1e-6
NAMES = dict(no_dfs='无采购', anchor='共同初始方案', activation='优化调用费',
    stable_activation='小额节约保留初始方案', capacity='优化容量与调用总费', capacity_rotation='同总费轮换')
RESOURCE_NAMES = dict(original_p10_q36='原7项P＋15项Q', expanded_p10_q36='扩展14项P＋15项Q')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def stat(values):
    values = np.asarray([float(value) for value in values if value is not None], float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite summary input')
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
        min=float(values.min()) if len(values) else None, max=float(values.max()) if len(values) else None)


def mean(values):
    return stat(values)['mean']


def quantity(row, scope, unit):
    return sum(device['absolute_energy'] for device in row[scope] if device['unit'] == unit)


class SealedFiles:
    def __init__(self, audit_root):
        self.inputs = {}
        self.cache = {}
        self.audit_root = Path(audit_root).resolve()
        self.complete = self.verified(self.audit_root/'complete.json')
        if self.complete['status'] not in ('passed', 'failed') or not self.complete['source_unchanged']:
            raise ValueError('Completed audit with stable source required: '+str(audit_root))
        self.report = self.verified_reference(self.complete['report'])
        self.manifest = self.verified_reference(self.complete['inputs'])
        self.allowed = {str(Path(item['path']).resolve()): item['sha256'] for item in self.manifest}
        self.audit_protocol = self.verified_reference(self.report['protocol'])
        self.run_protocol = self.verified_reference(self.audit_protocol['source_run'])
        self.source_root = Path(self.audit_protocol['source_run']['path']).parent
        self.entries = self.audited(self.run_protocol['inputs']['path'])

    def checked(self, path, digest=None):
        item = ref(path)
        if digest is not None and item['sha256'] != digest:
            raise ValueError('Changed audited file: '+str(path))
        self.inputs[item['path']] = item
        return Path(path)

    def verified(self, path):
        return read(self.checked(path))

    def verified_reference(self, item):
        return read(self.checked(item['path'], item['sha256']))

    def audited(self, path):
        path = str(Path(path).resolve())
        if path not in self.allowed:
            raise ValueError('File not in the completed audit manifest: '+path)
        if path not in self.cache:
            checked = self.checked(path, self.allowed[path])
            if checked.suffix == '.npz':
                with np.load(checked, allow_pickle=False) as data:
                    self.cache[path] = {key: data[key].copy() for key in data.files}
            else:
                self.cache[path] = read(checked)
        return self.cache[path]

    def forecast_folder(self, entry, draw):
        return self.source_root/'fraction0.10'/entry['id']/f'forecast{draw:02d}'

    def forecast_J(self, row, policy):
        pair = row['evaluation'].get('pair')
        if pair is None:
            return None
        folder = Path(row['source']).parents[1]/'exact_requested'/policy
        path = folder/'forecast_pair_evidence.npz'
        if str(path.resolve()) not in self.allowed:
            return None
        arrays = self.audited(path)
        matches = [i for i, item in enumerate(arrays['pairs']) if set(map(int, item)) == set(pair)]
        if len(matches) != 1:
            raise ValueError('Missing or duplicate forecast evidence pair')
        return float(arrays['J'][matches[0]])


def paired(a, b):
    key = lambda row: (row['forecast'], row['repeat'])
    aa, bb = {key(row): row for row in a}, {key(row): row for row in b}
    added = lost = new_wrong = removed_wrong = 0
    weight, clean, observed, costs, P, Q = [], [], [], [], [], []
    for identity in sorted(set(aa) & set(bb)):
        x, y = aa[identity], bb[identity]
        if x['audit_status'] != 'passed' or y['audit_status'] != 'passed':
            continue
        ex, ey = x['evaluation'], y['evaluation']
        added += bool(ex['native_correct'] and not ey['native_correct'])
        lost += bool(ey['native_correct'] and not ex['native_correct'])
        new_wrong += bool(ex['wrong_high'] and not ey['wrong_high'])
        removed_wrong += bool(ey['wrong_high'] and not ex['wrong_high'])
        weight.append(ex['p_true']-ey['p_true'])
        if 'evidence' in ex and 'evidence' in ey:
            clean.append(ex['evidence']['clean_logodds']-ey['evidence']['clean_logodds'])
            observed.append(ex['evidence']['observed_logodds']-ey['evidence']['observed_logodds'])
        costs.append(x['settled_cost']['total_yuan']-y['settled_cost']['total_yuan'])
        P.append(quantity(x, 'delivered', 'kW')-quantity(y, 'delivered', 'kW'))
        Q.append(quantity(x, 'delivered', 'kvar')-quantity(y, 'delivered', 'kvar'))
    return dict(complete_pairs=len(weight), added_recoveries=added, lost_recoveries=lost,
        new_wrong_high=new_wrong, removed_wrong_high=removed_wrong,
        mean_true_weight_difference=mean(weight), minimum_true_weight_difference=stat(weight)['min'],
        true_weight_decreased_paths=sum(value < -1e-12 for value in weight),
        mean_clean_logodds_difference=mean(clean), mean_observed_logodds_difference=mean(observed),
        mean_settled_total_fee_difference_yuan=mean(costs), mean_delivered_P_difference_kwh=mean(P),
        mean_delivered_Q_difference_kvarh=mean(Q))


def group(files, phase, variant, policy, rows, baseline, expected):
    good = [row for row in rows if row['audit_status'] == 'passed']
    correct = sum(row['evaluation']['native_correct'] for row in good)
    increments = [row['evaluation']['evidence_increment'] for row in good if 'evidence_increment' in row['evaluation']]
    predicted = []
    for row in good:
        a, b = files.forecast_J(row, policy), files.forecast_J(row, 'no_dfs')
        if a is not None and b is not None:
            predicted.append(a-b)
    compare = paired(rows, baseline)
    result = dict(phase=phase, resource_variant=variant, policy=policy, planned_paths=expected,
        audited_paths=len(rows), complete_paths=len(good), failed_or_unavailable_paths=expected-len(good),
        internal_correct=correct, internal_accuracy=correct/expected if expected else None,
        full_response_support_exact=sum(row['evaluation']['native_full_support_exact'] for row in good),
        correct_weight_at_least_090=sum(row['evaluation']['true_high'] for row in good),
        wrong_weight_at_least_090=sum(row['evaluation']['wrong_high'] for row in good),
        mean_true_weight=mean(row['evaluation']['p_true'] for row in good),
        forecast_J_gain_mean=mean(predicted), forecast_J_gain_available_paths=len(predicted),
        clean_logodds_gain_mean=mean(row['clean_logodds'] for row in increments),
        observed_logodds_gain_mean=mean(row['observed_logodds'] for row in increments),
        actual_J_gain_mean=mean(row['J'] for row in increments),
        bias_change_mean=mean(row['bias'] for row in increments), meter_projection_change_mean=mean(row['meter'] for row in increments),
        negative_observed_gain_paths=sum(row['observed_logodds'] < -1e-8 for row in increments),
        delivered_P_nonzero_paths=sum(quantity(row, 'delivered', 'kW') > P_ENERGY_TOLERANCE for row in good),
        delivered_Q_nonzero_paths=sum(quantity(row, 'delivered', 'kvar') > P_ENERGY_TOLERANCE for row in good),
        mean_requested_P_kwh=mean(quantity(row, 'requested', 'kW') for row in good),
        mean_requested_Q_kvarh=mean(quantity(row, 'requested', 'kvar') for row in good),
        mean_delivered_P_kwh=mean(quantity(row, 'delivered', 'kW') for row in good),
        mean_delivered_Q_kvarh=mean(quantity(row, 'delivered', 'kvar') for row in good),
        mean_reserved_P_kw=mean(row['planned_cost']['reserved_P_kw'] for row in good),
        mean_reserved_Q_kvar=mean(row['planned_cost']['reserved_Q_kvar'] for row in good),
        mean_planned_activation_yuan=mean(row['planned_cost']['activation_yuan'] for row in good),
        mean_planned_total_yuan=mean(row['planned_cost']['total_yuan'] for row in good),
        mean_settled_activation_yuan=mean(row['settled_cost']['activation_yuan'] for row in good),
        mean_reserved_capacity_yuan=mean(row['settled_cost']['availability_yuan'] for row in good),
        mean_settled_total_yuan=mean(row['settled_cost']['total_yuan'] for row in good),
        mean_settled_P_activation_yuan=mean(row['settled_cost']['active_activation_yuan'] for row in good),
        mean_settled_Q_activation_yuan=mean(row['settled_cost']['reactive_activation_yuan'] for row in good),
        voltage_min_pu=min((row['electrical']['all_bus_voltage']['minimum'] for row in good), default=None),
        voltage_max_pu=max((row['electrical']['all_bus_voltage']['maximum'] for row in good), default=None),
        voltage_excursion_paths=sum(row['electrical']['cells_below_095']+row['electrical']['cells_above_105'] > 0 for row in good),
        maximum_branch_current_A=max((row['electrical']['maximum_branch_current_A'] for row in good), default=None),
        maximum_root_S_kVA=max((row['electrical']['root_S_kVA']['maximum'] for row in good), default=None),
        mean_active_loss_kwh=mean(row['electrical']['active_loss_kwh'] for row in good))
    result.update({'vs_zero_'+key: value for key, value in compare.items()})
    reference = {(row['forecast'], row['repeat']): row for row in baseline if row['audit_status'] == 'passed'}
    result['mean_active_loss_change_kwh'] = mean(row['electrical']['active_loss_kwh']-reference[(row['forecast'], row['repeat'])]['electrical']['active_loss_kwh']
        for row in good if (row['forecast'], row['repeat']) in reference)
    return result


def solver_summary(phase, variant, policy, quotes, plans):
    available = [q for q in quotes if q['resource_variant'] == variant and q['policy'] == policy]
    feasible = [q for q in available if q['feasible']]
    initials = [q for q in feasible if (q.get('initial') or {}).get('feasible')]
    savings = [q['initial']['cost_yuan']-q['objective_cost_yuan'] for q in initials]
    result = dict(phase=phase, resource_variant=variant, policy=policy, quotes=len(available),
        feasible=len(feasible), zero_fallbacks=sum(q['zero_fallback'] for q in available),
        statuses=dict(Counter(q['status'] for q in available)), native_statuses=dict(Counter(str(q['native_status']) for q in available)),
        objective='activation_plus_daily_reserved_capacity' if policy == 'capacity' else 'activation_compensation',
        feasible_initial_quotes=len(initials), found_without_feasible_initial=sum(not (q.get('initial') or {}).get('feasible') for q in feasible),
        final_materially_better_than_initial=sum(value > 1e-6 for value in savings),
        initial_retained_to_1e_6_yuan=sum(abs(value) <= 1e-6 for value in savings),
        initial_to_final_objective_saving=stat(savings),
        objective_cost=stat(q['objective_cost_yuan'] for q in feasible),
        valid_lower_bound=stat(q.get('lower_bound_yuan') for q in available),
        relative_gap=stat(q.get('relative_gap') for q in feasible),
        quote_wall_seconds=stat(q.get('quote_wall_seconds') for q in available),
        charged_total_seconds=stat(q.get('charged_total_seconds') for q in available),
        shared_preparation_seconds=stat(q.get('shared_preparation_seconds') for q in available),
        native_runtime_seconds=stat((q.get('raw_native_attributes') or {}).get('Runtime') for q in available),
        quote_budget_exceeded=sum(bool(q.get('quote_budget_exceeded')) for q in available),
        globally_proved_optimal=sum(bool(q.get('global_optimality_verified')) for q in available),
        strict_buffer_negative_quotes=sum(not q['direct_original']['strict_buffered_nonnegative'] for q in feasible),
        minimum_original_buffered_margin=min((q['direct_original']['minimum_original_buffered_margin'] for q in feasible), default=None),
        minimum_guard=min((q['direct_original']['minimum_guard'] for q in feasible), default=None))
    if policy == 'anchor':
        result['charged_total_seconds'] = stat(value['anchor_shared_seconds'] for key, value in plans.items() if key[0] == variant)
        result['time_scope'] = 'Common domain/portfolio plus anchor stage; not just native optimizer runtime'
    else:
        result['time_scope'] = 'Shared preparation/anchor stage plus this objective call; same shared stage is charged to each comparison, not executed twice'
    return result


def phase_summary(files):
    report, protocol = files.report, files.run_protocol
    phase = protocol['phase']; records = report['records']
    expected_per_group = protocol['forecasts']*protocol['repeats']
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['resource_variant'], row['strategy'])].append(row)
    plans, actions = {}, {}
    for entry in files.entries:
        variant = entry['resource_variant']
        for draw in range(protocol['forecasts']):
            folder = files.forecast_folder(entry, draw)
            # A failed task remains in all planned denominators, without trying
            # to consume unfinished artifacts that were not independently read.
            if str((folder/'plans.json').resolve()) not in files.allowed:
                continue
            saved = files.audited(folder/'plans.json')
            shared = saved['activation'].get('shared_preparation_seconds')
            plans[(variant, draw)] = dict(stable=saved['stable_activation'], anchor_shared_seconds=shared)
            actions[(variant, draw)] = dict(planner=files.audited(folder/'planner.npz'), config=files.audited(folder/'resource_case.json'))
    group_rows = [group(files, phase, variant, policy, grouped[(variant, policy)], grouped[(variant, 'no_dfs')], expected_per_group)
        for variant in VARIANTS for policy in POLICIES]
    solver_rows = [solver_summary(phase, variant, policy, report['quotes'], plans)
        for variant in VARIANTS for policy in ('anchor', 'activation', 'capacity')]
    stable = []
    for variant in VARIANTS:
        rows = [value['stable'] for (v, _), value in plans.items() if v == variant]
        stable.append(dict(phase=phase, resource_variant=variant, planned_forecasts=protocol['forecasts'], recorded_forecasts=len(rows),
            anchor_available=sum(row['anchor_available'] for row in rows), anchor_retained=sum(row['retained_anchor'] for row in rows),
            refinement_selected=sum(row['feasible'] and not row['retained_anchor'] for row in rows),
            zero_fallback=sum(row['zero_fallback'] for row in rows), activation_saving_over_anchor_yuan=stat(row['saving_over_anchor_yuan'] for row in rows)))
    cross = []
    for policy in POLICIES:
        row = dict(phase=phase, policy=policy, **paired(grouped[(VARIANTS[1], policy)], grouped[(VARIANTS[0], policy)]))
        P_diff, Q_diff, Q_max, changed = [], [], [], 0
        for draw in range(protocol['forecasts']):
            if any((variant, draw) not in actions for variant in VARIANTS):
                continue
            a, b = (actions[(variant, draw)] for variant in VARIANTS)
            amap = {res['id']: (unit, a['planner'][policy][:, col]) for col, (res, unit) in enumerate(zip(a['config']['resources'], a['config']['action_units'], strict=True))}
            bmap = {res['id']: (unit, b['planner'][policy][:, col]) for col, (res, unit) in enumerate(zip(b['config']['resources'], b['config']['action_units'], strict=True))}
            pd = qd = qm = 0.
            for identity in set(amap) | set(bmap):
                unit = (amap.get(identity) or bmap[identity])[0]
                x = amap.get(identity, (unit, np.zeros(24)))[1]
                y = bmap.get(identity, (unit, np.zeros(24)))[1]
                difference = np.abs(y-x)
                if unit == 'kW': pd += float(difference.sum())
                else: qd += float(difference.sum()); qm = max(qm, float(difference.max()))
            P_diff.append(pd); Q_diff.append(qd); Q_max.append(qm); changed += qd > 1e-6
        complete_rows = [item for variant in VARIANTS for item in grouped[(variant, policy)] if item['audit_status'] == 'passed']
        zero_P = bool(complete_rows) and all(quantity(item, 'delivered', 'kW') <= P_ENERGY_TOLERANCE for item in complete_rows)
        row.update(forecasts_compared=len(P_diff), requested_P_action_L1_difference_kwh=stat(P_diff),
            requested_Q_action_L1_difference_kvarh=stat(Q_diff), maximum_Q_action_difference_kvar=max(Q_max, default=None),
            Q_actions_different_forecasts=changed, both_resources_delivered_P_zero_to_1e_6_kwh=zero_P,
            interpretation=('Both variants delivered no material P. Any outcome difference is not an executed P effect; changed Q actions arise from comparing separate nonconvex searches with different offered variables/initial pools.'
                if zero_P and changed else 'Compare actual P/Q purchases and action changes before attributing resource-variant outcome differences.'))
        before = next(item for item in group_rows if item['resource_variant'] == VARIANTS[0] and item['policy'] == policy)
        after = next(item for item in group_rows if item['resource_variant'] == VARIANTS[1] and item['policy'] == policy)
        row['mean_forecast_J_gain_difference'] = (after['forecast_J_gain_mean']-before['forecast_J_gain_mean']
            if after['forecast_J_gain_mean'] is not None and before['forecast_J_gain_mean'] is not None else None)
        cross.append(row)
    comparisons = []
    for variant in VARIANTS:
        for policy, reference in [('activation','anchor'), ('stable_activation','activation'),
                ('stable_activation','anchor'), ('capacity','activation'), ('capacity','capacity_rotation')]:
            comparisons.append(dict(phase=phase, resource_variant=variant, policy=policy, reference=reference,
                **paired(grouped[(variant, policy)], grouped[(variant, reference)])))
    return dict(phase=phase, audit_status=report['status'], source_audit=ref(files.audit_root/'report.json'), source_run=ref(files.source_root/'protocol.json'),
        expected_execution_rows=report['expected_rows'], audited_execution_rows=report['audited_rows'],
        original_quote_exceptions=report['original_quote_exceptions'], original_failed_tasks=len(report['original_failed_tasks']),
        original_failed_execution_rows=report['original_execution_failed_rows'], audit_task_errors=len(report['task_errors']),
        audit_row_errors=sum(row['audit_status'] != 'passed' for row in records),
        exact_requested_AC_failures=len(report['exact_diagnostic_failures']), unmatched_rotation_fees=len(report['rotation_unmatched_fees']),
        historical_inputs=protocol.get('original_histories', 1), forecasts=protocol['forecasts'], repeats=protocol['repeats'], seed=protocol['seed'],
        anchor_budget_seconds=protocol['anchor_seconds'], refinement_budget_seconds=protocol['solver_seconds'],
        groups=group_rows, solvers=solver_rows, stable=stable, comparisons=comparisons, cross_resources=cross)


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def markdown(phases):
    lines = ['# 容量采购与动作稳定性：独立审计汇总', '',
        '每个阶段、每种资源分别报告。所有采购策略的实际总费均为“请求峰值容量费＋实际调用费”；P 和 Q 的数量分别以 kWh 和 kvarh 计。', '',
        '这些路径来自同一已知负荷日和固定候选库。不同预测、响应及表计噪声不等于新增独立网络或负荷日；资源配置共享相同随机路径。恢复指内部拓扑，完整响应支持另列。', '']
    for phase in phases:
        lines += ['## '+phase['phase'], '',
            f"审计状态 {phase['audit_status']}，已核查 {phase['audited_execution_rows']}/{phase['expected_execution_rows']} 条；原报价异常 {phase['original_quote_exceptions']}，原任务失败 {phase['original_failed_tasks']}，原执行失败 {phase['original_failed_execution_rows']}；审计任务/行错误 {phase['audit_task_errors']}/{phase['audit_row_errors']}；理想执行诊断失败 {phase['exact_requested_AC_failures']}，轮换未匹配费用 {phase['unmatched_rotation_fees']}。", '',
            f"共同准备及初始方案预算 {phase['anchor_budget_seconds']:g} 秒，两个优化目标各追加 {phase['refinement_budget_seconds']:g} 秒。每资源配置 {phase['forecasts']} 份预测，每份 {phase['repeats']} 条配对执行路径。", '']
        for variant in VARIANTS:
            rows = [r for r in phase['groups'] if r['resource_variant'] == variant]
            lines += ['### '+RESOURCE_NAMES[variant], '',
                '|策略|内部正确/计划数|相对无采购新增/丢失|错误权重≥.9|正确权重≥.9|平均真权重|平均实际总费/元|',
                '|---|---:|---:|---:|---:|---:|---:|']
            for row in rows:
                lines.append(f"|{NAMES[row['policy']]}|{row['internal_correct']}/{row['planned_paths']}|{row['vs_zero_added_recoveries']}/{row['vs_zero_lost_recoveries']}|{row['wrong_weight_at_least_090']}|{row['correct_weight_at_least_090']}|{fmt(row['mean_true_weight'])}|{fmt(row['mean_settled_total_yuan'])}|")
            lines += ['', '|策略|预测ΔJ|无量测误差的AC ΔL|实测ΔL|负实测ΔL路径|P交付/kWh|Q交付/kvarh|购P峰值/kW|购Q峰值/kvar|容量费/元|调用费/元|',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for row in rows:
                lines.append('|'+NAMES[row['policy']]+'|'+ '|'.join(fmt(row[key]) for key in ('forecast_J_gain_mean', 'clean_logodds_gain_mean',
                    'observed_logodds_gain_mean'))+f"|{row['negative_observed_gain_paths']}|"+'|'.join(fmt(row[key]) for key in (
                    'mean_delivered_P_kwh','mean_delivered_Q_kvarh','mean_reserved_P_kw','mean_reserved_Q_kvar',
                    'mean_reserved_capacity_yuan','mean_settled_activation_yuan'))+'|')
            lines += ['', 'ΔJ 表示模型预测的区分距离增量；ΔL 表示真候选相对历史主要错误候选的对数赔率增量，均相对同一路径的无采购。前者使用预测功率；后两列分别使用无额外量测误差的 AC 值和最终实测值。', '',
                '|策略|有P交付路径|完整响应支持正确|全节点电压范围/pu|越限路径(.95–1.05)|平均网损/kWh|相对无采购网损变化/kWh|',
                '|---|---:|---:|---:|---:|---:|---:|']
            for row in rows:
                lines.append(f"|{NAMES[row['policy']]}|{row['delivered_P_nonzero_paths']}|{row['full_response_support_exact']}|{fmt(row['voltage_min_pu'],6)}–{fmt(row['voltage_max_pu'],6)}|{row['voltage_excursion_paths']}|{fmt(row['mean_active_loss_kwh'],6)}|{fmt(row['mean_active_loss_change_kwh'],6)}|")
            stable = next(row for row in phase['stable'] if row['resource_variant'] == variant)
            lines += ['', f"稳定规则保留初始方案 {stable['anchor_retained']}/{stable['planned_forecasts']} 次；接受细化方案 {stable['refinement_selected']} 次；失败回零 {stable['zero_fallback']} 次。", '',
                '|求解目标|可行/报价数|原生状态及次数|费用均值/元|有效下界均值/元|相对间隙均值|计入共享准备后平均秒数|初值实质改善/保持|',
                '|---|---:|---|---:|---:|---:|---:|---:|']
            for row in phase['solvers']:
                if row['resource_variant'] != variant: continue
                lines.append(f"|{NAMES[row['policy']]}|{row['feasible']}/{row['quotes']}|{row['native_statuses']}|{fmt(row['objective_cost']['mean'])}|{fmt(row['valid_lower_bound']['mean'])}|{fmt(row['relative_gap']['mean'])}|{fmt(row['charged_total_seconds']['mean'],2)}|{row['final_materially_better_than_initial']}/{row['initial_retained_to_1e_6_yuan']}|")
            lines += ['', '初始方案和调用费细化的上下界只针对调用费；容量方案针对容量与调用总费。原生状态9表示达到时间限制。初值实质改善以费用下降超过 1e−6 元计；间隙未闭合的报价保留有效上下界，不称最优。', '']
        lines += ['### 两种资源配置的配对差别', '',
            '|策略|扩展后新增/丢失正确|新增/减少错误高权重|平均真权重变化|平均实测ΔL变化|平均费用变化/元|Q动作不同的预测数|',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in phase['cross_resources']:
            lines.append(f"|{NAMES[row['policy']]}|{row['added_recoveries']}/{row['lost_recoveries']}|{row['new_wrong_high']}/{row['removed_wrong_high']}|{fmt(row['mean_true_weight_difference'])}|{fmt(row['mean_observed_logodds_difference'])}|{fmt(row['mean_settled_total_fee_difference_yuan'])}|{row['Q_actions_different_forecasts']}/{row['forecasts_compared']}|")
        flagged = [NAMES[row['policy']] for row in phase['cross_resources'] if row['both_resources_delivered_P_zero_to_1e_6_kwh'] and row['Q_actions_different_forecasts']]
        if flagged:
            lines += ['', '以下策略在两种资源配置均无实质 P 交付（每路径 ≤1e−6 kWh），但 Q 方案发生变化：'+ '、'.join(flagged)+'。其效果差别应归于不同搜索得到的 Q 方案；新增有功变量、初值池和非凸搜索过程同时变化，不能归为实际有功调用的收益。']
        lines += ['', '设备容量费和调用费为预先锁定的示例报价。200 kVA 仅为标幺基准；原网络缺支路热额定值及变压器铭牌，电流和根视在功率可核算，热负载率尚无依据。', '']
    return '\n'.join(lines)+'\n'


def save_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def run(audits, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve completed analysis: '+str(out))
    files = [SealedFiles(path) for path in audits]
    phases = [phase_summary(item) for item in files]
    if len({phase['phase'] for phase in phases}) != len(phases):
        raise ValueError('Each supplied audit must have a distinct phase')
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/'analyze.py')
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=ref(__file__), phases=phases,
        quantities='Energy quantities and actual fees are means over completed audited paths; correctness denominators include every planned path.',
        P_materiality_threshold_kwh=P_ENERGY_TOLERANCE,
        interpretation='Separate stages and resources; one known physical load day, frozen historical candidates. No pooled population recovery claim or new simulation.')
    write(out/'SUMMARY.json', result)
    for filename, key in [('outcomes.csv','groups'),('solvers.csv','solvers'),('stable.csv','stable'),
            ('policy_comparisons.csv','comparisons'),('resource_comparisons.csv','cross_resources')]:
        save_csv(out/filename, [row for phase in phases for row in phase[key]])
    (out/'REPORT.md').write_text(markdown(phases), encoding='utf-8')
    write(out/'inputs_manifest.json', list({path: value for item in files for path, value in item.inputs.items()}.values()))
    write(out/'complete.json', dict(status='complete', summary=ref(out/'SUMMARY.json'),
        report=ref(out/'REPORT.md'), inputs=ref(out/'inputs_manifest.json'),
        source_unchanged=ref(__file__)['sha256'] == result['source']['sha256']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, action='append', required=True, help='Completed independent audit root; repeat for separate phases')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    answer = run(args.audit, args.out)
    print(json.dumps(dict(phases=[item['phase'] for item in answer['phases']], groups=sum(len(item['groups']) for item in answer['phases']), out=str(args.out))))
