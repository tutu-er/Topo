"""Audited, explicitly oracle summaries of the two frozen-matrix QCQPs.

No network solution, optimization or fit is performed. Source quote and
linear-delivery evidence remain separate from deployment/AC performance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from ... import io


MODES = ('distance_only', 'protected')
NAMES = dict(distance_only='只增距离', protected='距离增加＋总误差降低')


class Reader:
    def __init__(self):
        self.refs = {}

    def path(self, reference):
        reference = reference if isinstance(reference, dict) else io.ref(reference)
        path = Path(reference['path']).resolve()
        current = io.ref(path)
        if current['sha256'] != reference['sha256']:
            raise ValueError('Source hash mismatch: '+str(path))
        if str(path) in self.refs and self.refs[str(path)] != current:
            raise ValueError('Source changed during report: '+str(path))
        self.refs[str(path)] = current
        return path

    def json(self, reference):
        return io.read(self.path(reference))

    def verify(self):
        return all(io.ref(path) == ref for path, ref in self.refs.items())


def statistics(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
        minimum=min(values) if values else None, maximum=max(values) if values else None)


def improvement(direct, zero):
    J, J0 = np.asarray(direct['J'], float), np.asarray(zero['J'], float)
    e, e0 = float(direct['e']), float(zero['e'])
    if J.shape != J0.shape or J.size == 0 or min(e, e0) < 0 or np.any(J < 0) or np.any(J0 < 0):
        raise ValueError('Invalid oracle norm evidence')
    gains = (J-J0)/np.maximum(1., J0)
    ratios = np.divide(J, J0, out=np.zeros_like(J), where=J0 > 0)
    return dict(e=e, e0=e0, error_ratio=None if e0 == 0 else e/e0,
        error_reduction_fraction=None if e0 == 0 else 1-e/e0,
        sum_J=float(J.sum()), sum_J0=float(J0.sum()),
        sum_distance_ratio=None if J0.sum() == 0 else float(J.sum()/J0.sum()),
        minimum_normalized_distance_gain=float(gains.min()),
        normalized_distance_gains=gains.tolist(),
        minimum_distance_ratio_over_nonzero_baselines=float(ratios[J0>0].min()) if np.any(J0>0) else None,
        zero_baseline_distance_count=int(np.count_nonzero(J0 == 0)),
        double_improvement_raw=bool(e < e0 and np.all(J > J0)),
        double_improvement_above_1e_minus6=bool(e0 > 0 and 1-e/e0 > 1e-6 and gains.min() > 1e-6))


def linear_summary(linear, zero):
    ratios = list(linear['response_error_to_distance_ratio'])
    # A zero distance is separately visible, never silently discarded from a max.
    undefined = sum(value is None for value in ratios)
    maximum = None if undefined else max(map(float, ratios))
    now, before = bool(linear['internal_correct']), bool(zero['internal_correct'])
    return dict(minimum_logodds=float(linear['minimum_logodds']),
        baseline_minimum_logodds=float(zero['minimum_logodds']),
        minimum_logodds_change=float(linear['minimum_logodds']-zero['minimum_logodds']),
        strongest_foreign_native=int(linear['strongest_foreign_native']),
        native_selected_index=int(linear['native_selected_index']),
        baseline_native_selected_index=int(zero['native_selected_index']),
        internal_correct=now, baseline_internal_correct=before,
        newly_correct=bool(now and not before), lost_correct=bool(before and not now),
        maximum_error_to_distance_ratio=maximum, undefined_zero_distance_ratios=undefined,
        all_norm_ratios_below_half=bool(undefined == 0 and maximum < .5),
        per_competitor_logodds=list(linear['logodds']),
        per_competitor_error_to_distance_ratio=ratios)


def plan_summary(mode, plan, zero, audited_quote):
    direct, linear = plan['direct'], plan['linear']
    z = float(direct['direct_z'])
    initial = plan.get('initial') or {}
    candidates = initial.get('candidates') or []
    source_bounds = plan.get('bounds') or {}
    bounds = dict(source_bounds)
    audited_history = audited_quote.get('native_bound_history') or {}
    additional_withdrawal = bool(audited_history.get('additional_native_withdrawal'))
    if bounds and 'audit_effective_upper_bound' in audited_history:
        effective = float(audited_history['audit_effective_upper_bound'])
        lower = float(bounds['lower_bound'])
        if effective < lower-1e-7*max(1., abs(lower)):
            raise ValueError('Audited effective upper bound contradicts the feasible objective')
        bounds.update(upper_bound=effective, relative_gap=audited_history['audit_relative_gap'],
            absolute_gap=effective-lower,
            bounds_closed_numerically=bool(effective-lower <= 1e-7*max(1., abs(lower), abs(effective))),
            upper_bound_source='audit_resource_box_after_native_conflict' if additional_withdrawal
                else source_bounds.get('upper_bound_source'),
            native_upper_bound_withdrawn=bool(source_bounds.get('native_upper_bound_withdrawn') or additional_withdrawal))
    return dict(mode=mode, status=plan.get('status'), direct_z=z,
        returned_objective_z=plan.get('z'),
        z_classification=('above_1e-6' if z > 1e-6 else 'positive_at_or_below_1e-6' if z > 0 else 'zero_or_negative'),
        positive_direction_found=bool(plan.get('positive_direction_found', z > 1e-6)),
        norms=improvement(direct, zero), linear=linear_summary(linear, zero),
        cost=plan['cost'], quantities=plan['quantities'],
        solver=dict(native_status=plan.get('native_status'), source=plan.get('source'),
            optimizer_invoked=plan.get('optimizer_invoked'), seconds=plan.get('seconds'),
            requested_seconds=plan.get('requested_seconds'), solver_exception=plan.get('solver_exception'),
            objective_sense='maximize', objective='common normalized improvement z; not cost',
            bounds=bounds, original_solver_bounds=source_bounds,
            independent_native_bound_audit=audited_history,
            additional_native_withdrawal=additional_withdrawal,
            native_attributes=plan.get('native_attributes'),
            initial_name=initial.get('name'), initial_z=initial.get('z'),
            initial_cost_yuan=initial.get('cost_yuan'), seed_count=len(candidates),
            feasible_seed_count=sum(bool(c.get('feasible')) for c in candidates),
            improvement_from_verified_initial_z=plan.get('improvement_from_verified_initial_z'),
            initial_candidates=candidates,
            native_infeasibility_withdrawn=plan.get('native_infeasibility_withdrawn'),
            native_upper_bound_withdrawn=bounds.get('native_upper_bound_withdrawn'),
            upper_bound_conflicts=bounds.get('historical_bound_conflicts', []),
            global_optimality_verified=plan.get('global_optimality_verified', False)))


def delivery_summary(rows, mode, zero, expected):
    chosen = [r for r in rows if r['mode'] == mode]
    if len(chosen) != expected or sorted(int(r['repeat']) for r in chosen) != list(range(expected)):
        raise ValueError('Missing, duplicate or unexpected paired delivery repeats')
    good = [r for r in chosen if r['status'] == 'complete']
    failed = [r for r in chosen if r['status'] != 'complete']
    margins = [float(r['linear']['minimum_logodds']) for r in good]
    correct = sum(bool(r['linear']['internal_correct']) for r in good)
    changes = [improvement(r['direct'], zero) for r in good]
    return dict(expected_rows=expected, complete_rows=len(good), failed_rows=len(failed),
        failures=failed, internal_correct=correct, zero_reference_correct=expected*int(zero['internal_correct']),
        new_correct=sum(bool(r['linear']['internal_correct']) and not zero['internal_correct'] for r in good),
        lost_correct=sum(not r['linear']['internal_correct'] and bool(zero['internal_correct']) for r in good),
        native_selected_counts=dict(Counter(str(r['linear']['native_selected_index']) for r in good)),
        double_improvement_above_1e_minus6=sum(c['double_improvement_above_1e_minus6'] for c in changes),
        double_improvement_raw=sum(c['double_improvement_raw'] for c in changes),
        positive_direct_z_above_1e_minus6=sum(r['direct']['direct_z'] > 1e-6 for r in good),
        negative_direct_z_count=sum(r['direct']['direct_z'] < -1e-7 for r in good),
        minimum_logodds=statistics(margins),
        direct_z=statistics([r['direct']['direct_z'] for r in good]),
        error_ratio=statistics([r['direct']['e']/zero['e'] for r in good]) if zero['e'] > 0 else statistics([]),
        settled_total_yuan=statistics([r['cost']['total_yuan'] for r in good]),
        active_abs_kwh=statistics([r['quantities']['active_abs_kwh'] for r in good]),
        reactive_abs_kvarh=statistics([r['quantities']['reactive_abs_kvarh'] for r in good]),
        scope='Original response multipliers applied to forecast background; Y=Hstar X; no AC or measurement noise')


def number(value, precision=4):
    return '—' if value is None else f'{value:.{precision}g}'


def write_markdown(report, path):
    lines = ['# 已知真实响应矩阵下的方向诊断', '',
        '本报告使用真实一阶响应矩阵和正确内部拓扑指定目标，属于离线oracle诊断。候选R/X与原资源域保持冻结。'
        '新日背景是已封存预测功率，输出为线性模型Y=H*X；没有运行AC或添加量测噪声。'
        '以下正确率仅描述这一线性诊断，不能作为可部署方法的恢复率。', '',
        f"独立审计通过；{report['quote_count']}张报价、{report['delivery_rows']}条交付记录。"
        f"报价异常{report['quote_exceptions']}，交付失败{report['failed_delivery_rows']}。"
        f"{report['case_count']}个已有开发设置来自{report['network_date_backgrounds']}组网络/日期背景；"
        f"每方案{report['repeats']}条eta重复不增加独立物理日期数量。", '',
        f"内部真族覆盖{report['internal_truth_covered_cases']}/{report['case_count']}，"
        f"完整支持真候选覆盖{report['full_support_truth_covered_cases']}/{report['case_count']}。"
        f"完整支持选择正确：计划{report['planned_full_support_correct']}/{report['quote_count']}，"
        f"交付{report['delivered_full_support_correct']}/{report['delivery_rows']}。"
        '下表的内部正确与完整线路支持恢复分别统计。', '',
        '## 方向变化与线性全库选择', '',
        'e/e0是总响应误差能量比；ΣJ/ΣJ0是全部异族候选距离之和的比值。'
        '最小标准化增量按min[(Jj−Jj0)/max(1,Jj0)]计算，在Jj0<1时不是百分比。'
        '直接z为各原式约束允许的共同增量。小于等于1e−6的正数照实保留。', '',
        '|输入|方案|e/e0|ΣJ/ΣJ0|最小标准化增量|直接z|双目标改善|最弱L：零→动作|赢家：零→动作|最大误差/距离|',
        '|---|---|---:|---:|---:|---:|---|---|---|---:|']
    for case in report['cases']:
        for mode in MODES:
            p = case['plans'][mode]; n, l = p['norms'], p['linear']
            lines.append(f"|{case['id']}|{NAMES[mode]}|{number(n['error_ratio'])}|{number(n['sum_distance_ratio'])}|"
                f"{number(n['minimum_normalized_distance_gain'])}|{number(p['direct_z'])}|"
                f"{'是' if n['double_improvement_above_1e_minus6'] else '微小正改善' if n['double_improvement_raw'] else '否'}|"
                f"{number(l['baseline_minimum_logodds'])}→{number(l['minimum_logodds'])}|"
                f"{l['baseline_native_selected_index']}→{l['native_selected_index']}|{number(l['maximum_error_to_distance_ratio'])}|")
    lines += ['', '双目标改善与正确选择分开统计：降低误差能量、增加候选距离，仍可能因误差投影方向变化而损害排序。'
        '最弱L使用指定正确参数点对全部异族原生点的累计线性对数赔率；同一正确族的其他参数点可能影响最终族选择。', '',
        '## 交付变化、费用和调用量', '',
        '|输入|方案|线性内部正确/重复数|交付双目标改善/重复数|零参考|新增正确/丢失正确|计划费用/元|交付平均费用/元|计划P/kWh|计划Q/kvarh|交付最弱L均值|',
        '|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for case in report['cases']:
        for mode in MODES:
            p, d = case['plans'][mode], case['deliveries'][mode]
            lines.append(f"|{case['id']}|{NAMES[mode]}|{d['internal_correct']}/{d['expected_rows']}|"
                f"{d['double_improvement_above_1e_minus6']}/{d['expected_rows']}|"
                f"{d['zero_reference_correct']}/{d['expected_rows']}|{d['new_correct']}/{d['lost_correct']}|"
                f"{number(p['cost']['total_yuan'])}|{number(d['settled_total_yuan']['mean'])}|"
                f"{number(p['quantities']['active_abs_kwh'])}|{number(p['quantities']['reactive_abs_kvarh'])}|"
                f"{number(d['minimum_logodds']['mean'])}|")
    lines += ['', '交付双目标改善逐条检查总误差降低与全部异族标准化距离增加均超过1e−6。'
        '设备随机响应可能破坏计划时的方向约束，故该次数与线性内部正确次数分别报告。']
    lines += ['', '## 求解状态与界', '',
        '本轮最大化z，可行点给下界，原生分支定界或资源盒松弛给上界。下表不是费用界；费用也不是优化目标。'
        '冲突的原生界按原收据撤回，零动作始终可行；未找到正z不证明不存在改善方向。', '',
        '|输入|方案|状态/原生状态|z下界|z上界|相对gap|总秒数|初值及z|返回来源|原生界撤回|',
        '|---|---|---|---:|---:|---:|---:|---|---|---|']
    for case in report['cases']:
        for mode in MODES:
            p = case['plans'][mode]; s, b = p['solver'], p['solver']['bounds']
            lines.append(f"|{case['id']}|{NAMES[mode]}|{p['status']}/{s['native_status']}|"
                f"{number(b.get('lower_bound'))}|{number(b.get('upper_bound'))}|{number(b.get('relative_gap'))}|"
                f"{number(s['seconds'])}|{s['initial_name']} / {number(s['initial_z'])}|{s['source']}|"
                f"{'是（审计新增）' if s['additional_native_withdrawal'] else str(bool(s['native_upper_bound_withdrawn']))}|")
    lines += ['', '## 旧动作的同口径线性参考', '',
        '这些动作只被代入相同的预测背景与H*，没有重新求解或重跑AC。它们不计为本轮新报价。', '',
        '|输入|旧动作|线性内部正确|最弱L|赢家|总误差e|激活及预留费用/元|',
        '|---|---|---|---:|---:|---:|---:|']
    for case in report['cases']:
        for name, r in case['old_action_linear_references'].items():
            lines.append(f"|{case['id']}|{name}|{r['linear']['internal_correct']}|"
                f"{number(r['linear']['minimum_logodds'])}|{r['linear']['native_selected_index']}|"
                f"{number(r['linear']['e'])}|{number(r['cost']['total_yuan'])}|")
    lines += ['', '![三个设置的配对方向变化](paired_direction.png)', '',
        '图中灰点是零动作参考，彩点是计划点；每行表示同一输入与模型，三个横轴分别显示误差比、'
        '最小标准化距离增量及全库最弱线性赔率。它们不是实际AC恢复率。', '',
        '完整原生界、全部初值、冲突记录、原动作参考及来源哈希保存在report.json中。'
        '表中优先采用独立审计后的有效上界；原求解器收据原样保留。'
        '底层记录中的true_native_weight以全部原生参数点归一化，不能与旧代表库p_true直接比较；本报告不据权重判断改善。']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    fig, axes = plt.subplots(1, 3, figsize=(14, max(5, 1.35*len(report['cases'])+1.7)), sharey=True)
    labels, values = [], []
    for case in report['cases']:
        short = case['id'].replace('__2024-01-04__', ' ').replace('__', ' ')
        for mode in MODES:
            p = case['plans'][mode]
            labels.append(short+'\n'+NAMES[mode])
            values.append((mode, [p['norms']['error_ratio'], p['norms']['minimum_normalized_distance_gain'],
                                 p['linear']['minimum_logodds']],
                           [1., 0., p['linear']['baseline_minimum_logodds']]))
    colors = {'distance_only':'#2878B5', 'protected':'#D97620'}
    for y, (mode, now, before) in enumerate(values):
        for column, ax in enumerate(axes):
            if now[column] is None:
                continue
            ax.plot([before[column], now[column]], [y, y], color=colors[mode], lw=2, alpha=.65)
            ax.scatter(before[column], y, color='#9A9A9A', s=30, zorder=3)
            ax.scatter(now[column], y, color=colors[mode], s=56, zorder=4)
            ax.annotate(number(now[column], 3), (now[column], y), xytext=(5, 6),
                textcoords='offset points', color=colors[mode], fontsize=8)
    for ax, title in zip(axes, ['总响应误差比 e/e0（越小越好）', '最小标准化距离增量（越大越好）', '全库最弱线性赔率 L（越大越好）']):
        ax.set_title(title, fontsize=10); ax.grid(axis='x', alpha=.22)
        ax.margins(x=.13)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].axvline(1., color='#888888', ls=':', lw=1)
    axes[1].axvline(0., color='#888888', ls=':', lw=1)
    axes[2].axvline(0., color='#888888', ls=':', lw=1)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('已知真实响应矩阵的离线诊断：双目标变化与线性选择分开观察', fontsize=13)
    fig.text(.5, .015, '固定预测背景；无AC、无新量测噪声。灰点：零动作；蓝色：只增距离；橙色：误差保护。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(out/'paired_direction.png', dpi=180)
    fig.savefig(out/'paired_direction.pdf')
    plt.close(fig)


def report(audit_path, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    reader = Reader()
    audit_path = reader.path(audit_path)
    audit = reader.json(audit_path)
    if audit.get('status') != 'passed':
        raise ValueError('A passed independent oracle audit is required')
    audited_sources = audit.get('source_inputs')
    if not isinstance(audited_sources, list) or not audited_sources:
        raise ValueError('Independent audit source hash inventory is required')
    for reference in audited_sources:
        reader.path(reference)
    protocol_ref = audit['source_protocol']
    protocol = reader.json(protocol_ref)
    root = reader.path(protocol_ref).parent
    complete = reader.json(root/'complete.json')
    if complete['status'] != 'complete' or not complete['source_unchanged']:
        raise ValueError('Completed unchanged oracle source required')
    entries = reader.json(protocol['inputs'])
    if tuple(protocol['modes']) != MODES or not protocol['oracle']:
        raise ValueError('Expected the two explicitly oracle modes')
    if audit.get('errors'):
        raise ValueError('Audit receipt contains unresolved errors')
    for expected, actual in (('expected_quotes', 'audited_quotes'),
                             ('expected_delivery_rows', 'audited_delivery_rows')):
        if audit.get(expected) != audit.get(actual) or audit.get(expected) is None:
            raise ValueError('Independent audit did not cover its complete denominator')
    if (audit['expected_quotes'] != protocol['expected_quotes']
            or audit['expected_delivery_rows'] != protocol['expected_delivery_rows']):
        raise ValueError('Audit and completed run expected denominators differ')
    quote_path = reader.path(audit['quotes'])
    audited_rows = [json.loads(line) for line in quote_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    audited_quotes = {(row['id'], row['mode']):row for row in audited_rows}
    if len(audited_quotes) != len(audited_rows) or len(audited_quotes) != protocol['expected_quotes']:
        raise ValueError('Duplicate or missing independently audited quotes')
    record_path = reader.path(audit['records'])
    audited_deliveries = [json.loads(line) for line in record_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(audited_deliveries) != protocol['expected_delivery_rows']:
        raise ValueError('Independent delivery record denominator differs')
    coverage = {c['id']:c for c in audit['cases']}
    cases = []
    quote_count = delivery_rows = failed_rows = quote_exceptions = 0
    for entry in entries:
        directory = root/'cases'/entry['id']
        done = reader.json(directory/'complete.json')
        if done['status'] != 'complete':
            raise ValueError('Incomplete input retained; a full report cannot be issued')
        seal = reader.json(directory/'seal.json')
        if seal['protocol'] != protocol_ref or seal['score'] != entry['score']:
            raise ValueError('Case seal protocol or score mismatch')
        plans = reader.json(done['plans'])
        if done['plans'] != seal['plans'] or set(plans) != set(MODES):
            raise ValueError('Plan references or mode count differ')
        reader.path(seal['planner'])
        zero = reader.json(directory/'zero.json')
        deliveries = reader.json(done['deliveries'])
        old = reader.json(directory/'old_action_linear_references.json')
        for mode in MODES:
            audited_quote = audited_quotes[(entry['id'], mode)]
            if reader.json(audited_quote['source_quote']) != plans[mode]:
                raise ValueError('Audited individual quote differs from sealed combined plans')
        packed = {mode:plan_summary(mode, plans[mode], zero, audited_quotes[(entry['id'], mode)]) for mode in MODES}
        delivered = {mode:delivery_summary(deliveries, mode, zero, protocol['repeats']) for mode in MODES}
        for mode in MODES:
            independent = audited_quotes[(entry['id'], mode)]['linear']
            if bool(independent['native_map_correct']) != packed[mode]['linear']['internal_correct']:
                raise ValueError('Audited and recorded plan internal selection differ')
            packed[mode]['linear']['native_full_support_exact'] = bool(independent['native_full_support_exact'])
            paths = [r for r in audited_deliveries if r['id'] == entry['id'] and r['mode'] == mode]
            if (len(paths) != protocol['repeats']
                    or sum(bool(r['linear']['native_map_correct']) for r in paths) != delivered[mode]['internal_correct']):
                raise ValueError('Audited and recorded delivery internal selections differ')
            delivered[mode]['native_full_support_exact'] = sum(bool(r['linear']['native_full_support_exact']) for r in paths)
        cases.append(dict(id=entry['id'], case=entry['case'], start_date=entry['start_date'],
            target_native=int(entry['oracle_target']), competitors=entry['oracle_competitors'],
            true_native_indices=coverage[entry['id']]['true_native_indices'],
            full_support_exact_native_indices=coverage[entry['id']]['full_support_exact_native_indices'],
            zero=zero, plans=packed, deliveries=delivered, old_action_linear_references=old,
            source_completion=io.ref(directory/'complete.json'), source_seal=io.ref(directory/'seal.json')))
        quote_count += len(plans); delivery_rows += len(deliveries)
        failed_rows += sum(d['failed_rows'] for d in delivered.values())
        quote_exceptions += sum(p['status'] == 'exception' for p in plans.values())
    if quote_count != protocol['expected_quotes'] or delivery_rows != protocol['expected_delivery_rows']:
        raise ValueError('Full-run planned denominators are not met')
    if failed_rows != complete['failed_delivery_rows'] or quote_exceptions != complete['quote_exceptions']:
        raise ValueError('Failure counts differ from run completion')
    result = dict(status='complete', oracle=True, source_audit=io.ref(audit_path),
        source_protocol=protocol_ref, source_inputs=protocol['inputs'], report_source=io.ref(__file__),
        case_count=len(cases), quote_count=quote_count, delivery_rows=delivery_rows,
        failed_delivery_rows=failed_rows, quote_exceptions=quote_exceptions, repeats=protocol['repeats'],
        network_date_backgrounds=len({(e['case'], e['start_date']) for e in entries}),
        distinct_calendar_dates=len({e['start_date'] for e in entries}),
        native_bound_conflicts=audit.get('native_bound_conflicts'),
        internal_truth_covered_cases=sum(bool(c['true_native_indices']) for c in cases),
        full_support_truth_covered_cases=sum(bool(c['full_support_exact_native_indices']) for c in cases),
        planned_full_support_correct=sum(c['plans'][m]['linear']['native_full_support_exact'] for c in cases for m in MODES),
        delivered_full_support_correct=sum(c['deliveries'][m]['native_full_support_exact'] for c in cases for m in MODES),
        cases=cases, summary_by_mode={mode:dict(
            quotes=len(cases), positive_z_above_1e_minus6=sum(c['plans'][mode]['direct_z'] > 1e-6 for c in cases),
            small_positive_z=sum(0 < c['plans'][mode]['direct_z'] <= 1e-6 for c in cases),
            double_improvement_above_1e_minus6=sum(c['plans'][mode]['norms']['double_improvement_above_1e_minus6'] for c in cases),
            planned_linear_correct=sum(c['plans'][mode]['linear']['internal_correct'] for c in cases),
            planned_new_correct=sum(c['plans'][mode]['linear']['newly_correct'] for c in cases),
            planned_lost_correct=sum(c['plans'][mode]['linear']['lost_correct'] for c in cases),
            delivered_linear_correct=sum(c['deliveries'][mode]['internal_correct'] for c in cases),
            delivered_expected=sum(c['deliveries'][mode]['expected_rows'] for c in cases),
            delivered_new_correct=sum(c['deliveries'][mode]['new_correct'] for c in cases),
            delivered_lost_correct=sum(c['deliveries'][mode]['lost_correct'] for c in cases)) for mode in MODES},
        interpretation='Oracle linear-model direction diagnostic, not deployable topology recovery or AC/noisy performance',
        weight_scope='Any retained true_native_weight uses the complete native bank; not comparable with old representative-bank p_true',
        bound_scope='Maximum z: feasible action LB; native/box UB; costs are reported but not optimized')
    if not reader.verify():
        raise ValueError('Input source changed during report')
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/'report_source.py')
    plot(result, out)
    write_markdown(result, out/'REPORT.md')
    result.update(sources_unchanged=reader.verify(), source_refs=list(reader.refs.values()),
        artifacts={name:io.ref(out/name) for name in ('REPORT.md', 'paired_direction.png', 'paired_direction.pdf', 'report_source.py')})
    if not result['sources_unchanged']:
        raise ValueError('Input source changed before artifact completion')
    io.write(out/'report.json', result)
    io.write(out/'complete.json', dict(status='complete', source_audit=io.ref(audit_path),
        sources_unchanged=True, quote_count=quote_count, delivery_rows=delivery_rows, report=io.ref(out/'report.json')))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    answer = report(args.audit, args.out)
    print({key:answer[key] for key in ('status', 'quote_count', 'delivery_rows', 'quote_exceptions', 'failed_delivery_rows')})
