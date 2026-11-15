"""Paired single-target versus original multi-target AC development readout.

Reuses the established full-native score and K/input/AC/meter decompositions.
Only geometry-budget diagnostics are disabled. No planning, fitting or AC runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ... import io
from ..frozen_rx_guard_20261115 import report as shared


MODES = ('normal_fixed', 'mgf_fixed', 'mgf_forecast')
NAMES = dict(normal_fixed='固定功率正态门槛', mgf_fixed='固定功率指数门槛', mgf_forecast='随机功率指数门槛')
BASELINES = ('no_dfs', 'anchor', *MODES, *(m+'_rotation' for m in MODES))


def summarize(rows):
    result = shared.summarize(rows)
    good = [r for r in rows if r['audit_status'] == 'passed']
    for stage in ('planned', 'settled'):
        for cost in ('activation_yuan', 'availability_yuan', 'total_yuan'):
            result[stage+'_'+cost] = shared.stats([r[stage+'_cost'][cost] for r in good])
    result['zero_requested_rows'] = sum(r['quantities']['requested_P_kwh'] == 0.
        and r['quantities']['requested_Q_kvarh'] == 0. for r in good)
    return result


def quote_rows(reader, quotes, entries, root):
    answer, timing = [], []
    for tag, entry in entries.items():
        directory = root/'cases'/tag
        seal = reader.json(directory/'seal.json')
        plans = reader.json(seal['plans'])
        complete = reader.json(directory/'complete.json')
        old = Path(entry['prior_directory'])
        shared_receipt = reader.json(old/'shared_initial_receipt.json')
        anchor = reader.json(old/'anchor_plan.json')
        walls = []
        for audited in (q for q in quotes if q['id'] == tag):
            name = audited.get('policy', audited.get('strategy'))
            if name not in plans:
                raise ValueError('Audited quote absent from sealed plans: '+str(name))
            source = plans[name]
            certificate = audited.get('certificate') or {}
            native = certificate.get('raw_native_attributes') or source.get('native_attributes') or {}
            initial = source.get('initial') or source.get('initial_feasible')
            wall = audited.get('quote_wall_seconds', source.get('quote_wall_seconds', source.get('seconds')))
            row = dict(audited, policy=name, data_origin='new_single_target_quote',
                original_solver_receipt=source, native_attributes=native, initial=initial,
                quote_wall_seconds=wall, initial_points_source=source.get('initial_points'),
                objective_scope='Original activation fee only; reservation and total charges are reported separately',
                original_bound_scope='Single historical target, all foreign native competitors and original forward direction guards',
                zero_scope='A zero fallback is an executed baseline, not a feasible upper bound for a failed procurement quote')
            answer.append(row)
            if wall is not None:
                walls.append(float(wall))
        timing.append(dict(id=tag, new_quote_wall_seconds_sum=sum(walls),
            case_wall_seconds_including_execution=complete.get('seconds'),
            shared_prediction_models_seconds=complete.get('shared_prediction_models_seconds'),
            old_shared_pool_generation_seconds=shared_receipt.get('seconds'),
            old_shared_pool_and_anchor_seconds=shared_receipt.get('shared_preparation_seconds'),
            old_anchor_quote_seconds=anchor.get('quote_wall_seconds', anchor.get('seconds')),
            old_shared_work_reexecuted=False,
            meaning='Old shared pool and anchor are reused. New quote wall times and once-per-case preparation are separate; do not count reused work as newly executed.'))
    if len(answer) != len(quotes):
        raise ValueError('Not all independent quotes were matched')
    return answer, timing


def baseline_quote_rows(reader, protocol, entries):
    """Reuse the original audit's cost certificates on their original domains."""
    audit = reader.json(protocol['baseline_audit'])
    if audit.get('status') != 'passed':
        raise ValueError('Passed original grid audit required for baseline quotes')
    original_protocol = reader.json(audit['source_protocol'])
    original_root = Path(audit['source_protocol']['path']).parent
    all_quotes = reader.lines(audit['quotes'])
    rows = [q for q in all_quotes if q['id'] in entries and q['policy'] in MODES]
    if len(rows) != len(entries)*len(MODES) or len({(q['id'],q['policy']) for q in rows}) != len(rows):
        raise ValueError('Original multi-target quote denominator differs')
    repository = Path(__file__).resolve().parents[3]
    source_index = {str(Path(r['path']).resolve()):r for r in original_protocol['sources']}
    result = []
    for q in rows:
        entry = entries[q['id']]
        seal = reader.json(entry['prior_seal'])
        plans = reader.json(seal['plans'])
        source = plans[q['policy']]
        if q['status'] != source['status'] or q['source'] != source['source']:
            raise ValueError('Original audit quote and sealed plan differ')
        snapshots = []
        for key in ('solver_source','evidence_source'):
            ref = source.get(key)
            if not ref:
                continue
            if source_index.get(str(Path(ref['path']).resolve()), {}).get('sha256') != ref['sha256']:
                raise ValueError('Original quote source absent from original protocol')
            snapshot = original_root/'source_snapshot'/Path(ref['path']).resolve().relative_to(repository)
            snapshot_ref = dict(path=str(snapshot), sha256=ref['sha256'])
            reader.checked(snapshot_ref)
            snapshots.append(dict(original_source=ref, executed_snapshot=snapshot_ref))
        if source.get('diagnostics_receipt'):
            reader.json(source['diagnostics_receipt'])
        for ref in (source.get('diagnostic_files') or {}).values():
            reader.checked(ref)
        result.append(dict(q, data_origin='reused_original_multitarget_quote',
            original_solver_receipt=source, executed_source_snapshots=snapshots,
            original_bound_scope=source.get('raw_bound_scope', q.get('bound_scope')),
            bound_reuse_scope='Original multi-target QCQP only; its lower bound is not a single-target lower bound',
            original_audit=protocol['baseline_audit'],
            trajectory_certificate_available=bool((q.get('certificate') or {}).get('native_bound_history'))))
    return result


def quotation_summary(quotes):
    result = shared.quote_summary(quotes)
    nonzero = [q for q in quotes if q.get('new_gate_feasible') and
        (q.get('certificate') or {}).get('executed_activation_yuan', q.get('cost_yuan',0.)) > 1e-12]
    result.update(zero_sufficient_quotes=sum(q.get('status') == 'zero_sufficient' for q in quotes),
        native_time_limit_quotes=sum(q.get('native_status') == 9 for q in quotes),
        original_wall_budget_overrun_quotes=sum(bool(q.get('quote_budget_exceeded',
            (q.get('original_solver_receipt') or {}).get('quote_budget_exceeded'))) for q in quotes),
        feasible_initial_quotes=sum(bool((q.get('initial') or {}).get('feasible')) for q in quotes),
        verified_initial_retained_quotes=sum(q.get('source') == 'verified_initial_retained' for q in quotes),
        nonzero_accepted_quotes=len(nonzero),
        nonzero_accepted_gap=shared.stats([(q.get('certificate') or {}).get('relative_gap',q.get('relative_gap')) for q in nonzero]),
        gap_scope='Zero sufficient quotes have a nonnegative-cost zero optimum; nonzero accepted gaps are reported separately')
    return result


def number(value, precision=4):
    return '—' if value is None else f'{value:.{precision}g}'


def avg(summary, key):
    return summary[key]['mean']


def markdown(result):
    lines = ['# 单目标与原多目标：配对AC开发重放', '',
        '单目标由历史评分确定，采购不使用真实拓扑。候选H、历史校正K、预测输入、资源合同及误差法则保持冻结；'
        '每种概率门槛仍比较全部异族原生候选，仅保留历史目标出发的方向保护。'
        '本报告复用原零采购、多目标及轮换记录，新的单目标动作使用原AC网络；与旧动作完全一致时复用已核验AC结果。'
        '物理日期、预测和随机流均来自已有开发数据，因此是配对开发重放，不是新的独立确认。', '',
        f"独立审计通过。复用基线{result['baseline_rows']}行，新增策略记录{result['new_rows']}行，"
        f"其中新算AC {result['new_ac_rows']}行、复用AC {result['reused_ac_rows']}行；"
        f"新报价{result['quotes']['n']}张。完整支持与内部结构正确分别统计。"
        f"{result['case_count']}个设置来自{result['network_date_backgrounds']}组网络/日期背景，"
        f"每策略{result['repeats']}条配对路径不增加独立日期数量。", '',
        f"内部真族覆盖{result['internal_truth_covered_cases']}/{result['case_count']}，"
        f"完整支持真候选覆盖{result['full_support_truth_covered_cases']}/{result['case_count']}。", '',
        '采购目标为激活费最小；预留费和总费用单列。旧轮换只对应旧多目标报价，未构造新单目标同费轮换。']
    if result['smoke']:
        lines[2:2] = ['本文件仅核验接线、数据口径和报告生成。求解预算及重复数是接线设置，表中数值不作为正式效果结论。', '']
    for mode in MODES:
        lines += ['', '## '+NAMES[mode], '',
            '|输入|策略|内部正确|完整支持正确|错误高权重|零动作/回退|相对零新增/丢失|相对多目标新增/丢失|激活费/元|预留费/元|总费/元|P/kWh|Q/kvarh|',
            '|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|']
        for case in result['cases']:
            for policy, label in (('no_dfs', '零采购'), (mode, '原多目标'), ('single_'+mode, '单目标')):
                s = case['strategies'][policy]
                paired = case['paired_vs_no_dfs'][policy]
                comparison = case['paired_vs_multitarget'].get(policy)
                comp = '—' if comparison is None else f"{comparison['new_recoveries']}/{comparison['lost_recoveries']}"
                lines.append(f"|{case['id']}|{label}|{s['recovered']}/{s['n']}|{s['full_support_recovered']}/{s['n']}|"
                    f"{s['wrong_high']}|{s['zero_requested_rows']}/{s['zero_fallback_rows']}|"
                    f"{paired['new_recoveries']}/{paired['lost_recoveries']}|{comp}|"
                    f"{number(avg(s,'settled_activation_yuan'))}|{number(avg(s,'settled_availability_yuan'))}|"
                    f"{number(avg(s,'settled_total_yuan'))}|{number(s['quantities']['delivered_P_kwh']['mean'])}|"
                    f"{number(s['quantities']['delivered_Q_kvarh']['mean'])}|")
    lines += ['', '费用为各配对执行路径的均值；失败路径不虚构费用或归入成功分母。'
        '“零动作/回退”分别给出实际零动作条数与报价回退条数，零动作可能是合法最小费解。', '',
        '## 历史固定竞争对：预测证据如何变成实际证据', '',
        '下表全为相对同一路径零采购的对数赔率增量L。顺序是预测功率与K均值→实际交付功率与K均值→干净AC→含量测评分。'
        'J是候选预测距离，另列在JSON中，不能代替这些赔率增量。', '',
        '|输入|策略|固定目标/竞争者|预测ΔL|实际输入K ΔL|干净AC ΔL|量测后ΔL|输入/交付差额|响应模型差额|量测差额|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in result['cases']:
        for mode in MODES:
            for policy in (mode, 'single_'+mode):
                pair = case['strategies'][policy]['historical_fixed_pair']
                inc = pair['increments']; gap = pair['forecast_actual_gap']['components']
                lines.append('|'+case['id']+'|'+policy+'|'+str(pair['pairs'])+'|'+
                    '|'.join(number(inc[k]['mean']) for k in ('forecast_K_logodds','actual_input_K_logodds','clean_logodds','observed_logodds'))+'|'+
                    '|'.join(number(gap[k]['mean']) for k in ('input_and_delivery','response_model','meter'))+'|')
    lines += ['', '## 全库最危险竞争者', '',
        '固定近对改善可能被另一个竞争者反超。这里逐路径重新在完整异族库中取最弱赔率，真实族仅用于离线评价。'
        '不同阶段的最危险竞争者可能不同，因此这一张表的列差不解释成同一对的误差分解。', '',
        '|输入|策略|预测K最弱L|实际输入K最弱L|干净AC最弱L|含量测最弱L|实际最危险竞争者计数|最终赢家计数|',
        '|---|---|---:|---:|---:|---:|---|---|']
    for case in result['cases']:
        for policy in ('no_dfs', *[name for mode in MODES for name in (mode, 'single_'+mode)]):
            s = case['strategies'][policy]; whole = s['whole_bank_fixed_true_target']
            fields = []
            for law in ('forecast_K','actual_input_K','clean','observed'):
                item = whole.get(law)
                fields.append(number(None if item is None else item['minimum_logodds']['mean']))
            lines.append('|'+case['id']+'|'+policy+'|'+'|'.join(fields)+'|'+
                str(whole['observed']['strongest_competitor_counts'])+'|'+str(s['audited_winner_counts'])+'|')
    lines += ['', '## 新报价状态、初值与激活费界', '',
        '下界和上界只针对原激活费目标及本轮单目标约束；总费用不是这个目标。'
        '限时结果、保留初值、零回退及审计撤回的原生界分别记录。', '',
        '|输入|策略|状态/原生状态|原式可行|零回退|激活费下界|激活费上界|相对gap|秒数|初值|返回来源|界冲突/撤回|',
        '|---|---|---|---|---|---:|---:|---:|---:|---|---|---|']
    for q in result['quotes']['rows']:
        cert = q.get('certificate') or {}; source = q['original_solver_receipt']; initial = q.get('initial') or {}
        hist = cert.get('native_bound_history') or {}
        lines.append(f"|{q['id']}|{q['policy']}|{q.get('status')}/{q.get('native_status')}|"
            f"{q.get('new_gate_feasible',q.get('feasible'))}|{q.get('zero_fallback')}|"
            f"{number(cert.get('lower_bound_yuan',q.get('lower_bound_yuan')))}|"
            f"{number(cert.get('upper_bound_yuan',cert.get('executed_activation_yuan') if cert.get('accepted') else None))}|"
            f"{number(cert.get('relative_gap',q.get('relative_gap')))}|{number(q['quote_wall_seconds'])}|"
            f"{initial.get('name','—')}|{q.get('source',source.get('source'))}|"
            f"{bool(hist.get('recorded_bound_conflict_with_verified_buffered_point'))}/"
            f"{bool(cert.get('lower_bound_withdrawn',q.get('lower_bound_withdrawn')))}|")
    lines += ['', '## 原多目标报价：独立保留的求解记录', '',
        '以下9张报价来自原网格审计及当时封存的代码、计划和日志。费用界只适用于原多目标约束；'
        '单目标删除部分约束后，原下界不作迁移。旧审计未提供完整界轨迹证书时标为未另核，不填成“无冲突”。', '',
        '|输入|模式|状态/原生状态|原式可行|原回退|激活费下界|激活费上界|相对gap|报价秒数|初值|返回来源|原界撤回|界轨迹复核|',
        '|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|']
    for q in result['baseline_quotes']['rows']:
        cert = q.get('certificate') or {}; initial = q.get('initial') or {}
        upper = cert.get('upper_bound_yuan', cert.get('executed_activation_yuan',q.get('cost_yuan'))
            if q.get('new_gate_feasible') else None)
        lines.append(f"|{q['id']}|{q['policy']}|{q.get('status')}/{q.get('native_status')}|"
            f"{q.get('new_gate_feasible')}|{bool(q.get('anchor_fallback') or q.get('zero_fallback'))}|"
            f"{number(cert.get('lower_bound_yuan',q.get('lower_bound_yuan')))}|{number(upper)}|"
            f"{number(cert.get('relative_gap',q.get('relative_gap')))}|{number(q.get('quote_wall_seconds'))}|"
            f"{initial.get('name','—')}|{q.get('source')}|"
            f"{bool(cert.get('lower_bound_withdrawn',q.get('lower_bound_withdrawn')))}|"
            f"{'已核' if q['trajectory_certificate_available'] else '未另核'}|")
    withdrawn = [q for q in result['baseline_quotes']['rows'] if q.get('lower_bound_withdrawn')]
    for q in withdrawn:
        source = q['original_solver_receipt']; native = q.get('native_attributes') or {}
        lines += ['', f"原{q['id']} / {q['policy']}的原生状态为{q.get('native_status')}，"
            f"原生下界{number(native.get('ObjBound'))}元已因{source.get('lower_bound_withdrawal_reason')}撤回；"
            f"保留独立可行点上界{number(q.get('cost_yuan'))}元与有效下界{number(q.get('lower_bound_yuan'))}元。"
            '该原生状态不作为已证最优。']
    lines += ['', '|求解批次|报价数|原生到时限|合法零解|非零可行|有可行初值|保留初值|非零gap均值|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, q in (('原多目标',result['baseline_quotes']),('新单目标',result['quotes'])):
        lines.append(f"|{name}|{q['n']}|{q['native_time_limit_quotes']}|{q['zero_sufficient_quotes']}|"
            f"{q['nonzero_accepted_quotes']}|{q['feasible_initial_quotes']}|{q['verified_initial_retained_quotes']}|"
            f"{number(q['nonzero_accepted_gap']['mean'])}|")
    lines += ['', '## 复用的anchor和旧轮换', '',
        '这些仅是旧多目标报价对应的对照，不是新单目标同费对照。', '',
        '|输入|策略|内部正确|完整支持正确|错误高权重|激活费/元|预留费/元|总费/元|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for case in result['cases']:
        for policy in ('anchor', *(m+'_rotation' for m in MODES)):
            s = case['strategies'][policy]
            lines.append(f"|{case['id']}|{policy}|{s['recovered']}/{s['n']}|{s['full_support_recovered']}/{s['n']}|{s['wrong_high']}|"
                f"{number(avg(s,'settled_activation_yuan'))}|{number(avg(s,'settled_availability_yuan'))}|{number(avg(s,'settled_total_yuan'))}|")
    lines += ['', '## 完整性与边界', '',
        f"源哈希复核：{result['sources_unchanged']}；完整原生SSE直接复算最大误差：{result['maximum_native_SSE_replay_error']:.3g}。",
        f"执行记录失败{len(result['recorded_failures'])}条；原任务异常：{result['original_failed_tasks']}；审计任务异常：{result['task_errors']}。",
        '概率门槛是声明分布及冻结规划均值下的条件结果；此处另报实际AC表现，不据有限重复路径声称概率覆盖。'
        '只选一个历史目标可能减少无必要采购，也可能强化错误历史目标，真族缺失与历史误选资格保留在JSON。'
        '完整支持、全库最危险竞争者、全部报价与初值、逐路径证据分解和时间来源都保存在report.json。']
    return '\n'.join(lines)+'\n'


def report(audit_path, out):
    reader = shared.Reader()
    audit = reader.json(audit_path)
    if audit.get('status') != 'passed':
        raise ValueError('Passed independent single-target audit required')
    protocol = reader.json(audit['source_protocol'])
    root = Path(audit['source_protocol']['path']).parent
    complete = reader.json(root/'complete.json')
    if complete['status'] != 'complete' or not complete.get('source_unchanged', False):
        raise ValueError('Completed unchanged single-target run required')
    entries = reader.json(audit['source_inputs'])
    if isinstance(entries, dict):
        entries = entries['inputs']
    entrymap = {e['id']:e for e in entries}
    coverage = {c['id']:c for c in audit['input_coverage']}
    if set(coverage) != set(entrymap):
        raise ValueError('Audited input coverage and manifest differ')
    modes = tuple(protocol['modes'])
    if modes != MODES:
        raise ValueError('Expected the three unchanged probability modes')
    new = reader.lines(audit['records'])
    baseline = reader.lines(audit['baseline_records'])
    quotes = reader.lines(audit['quotes'])
    raw = baseline+new
    if len({shared.key(r) for r in raw}) != len(raw):
        raise ValueError('Duplicate new/reused execution row')
    if any(r['audit_status'] not in ('passed','recorded_failure') for r in raw):
        raise ValueError('Unverified execution included')
    # baseline_only affects solely context.geometry; every new action is still
    # read and receives the same score / model-gap / whole-bank recomputation.
    post = shared.Postprocess(reader, entrymap, coverage, root, baseline_only=True)
    enriched = [post.enrich(r, 'reused_grid_baseline') for r in baseline]
    enriched += [post.enrich(r, 'new_single_target_execution') for r in new]
    cases = []
    newnames = tuple('single_'+m for m in modes)
    expected_new = len(entries)*len(newnames)*protocol['repeats']
    expected_old = len(entries)*len(BASELINES)*protocol['repeats']
    if len(new) != expected_new or len(baseline) != expected_old or len(quotes) != len(entries)*len(modes):
        raise ValueError('Locked execution or quote denominator differs')
    for tag, entry in entrymap.items():
        rows = [r for r in enriched if r['id'] == tag]
        names = (*BASELINES, *newnames)
        if set(r['strategy'] for r in rows) != set(names):
            raise ValueError('Unexpected or missing strategy')
        if any(sum(r['strategy'] == name for r in rows) != protocol['repeats'] for name in names):
            raise ValueError('Policy repeats differ')
        strategies = {name:summarize([r for r in rows if r['strategy'] == name]) for name in names}
        cases.append(dict(id=tag, dimensions={key:entry.get(key) for key in
            ('case','original_id','start_date','stride','meter_multiplier','training_rows','validation_rows','base_kw')},
            coverage=coverage[tag], historical_target=post.contexts[tag]['historical'],
            offline_fixed_true_target=post.contexts[tag]['true_target'], strategies=strategies,
            paired_vs_no_dfs={name:shared.paired(rows, name, 'no_dfs') for name in names},
            paired_vs_multitarget={'single_'+mode:shared.paired(rows, 'single_'+mode, mode) for mode in modes}))
    quotes, timing = quote_rows(reader, quotes, entrymap, root)
    quote_summary = quotation_summary(quotes)
    baseline_quotes = quotation_summary(baseline_quote_rows(reader, protocol, entrymap))
    aggregate = {}
    for mode in modes:
        aggregate[mode] = dict(multitarget=summarize([r for r in enriched if r['strategy'] == mode]),
            single_target=summarize([r for r in enriched if r['strategy'] == 'single_'+mode]),
            paired_vs_multitarget=shared.paired(enriched, 'single_'+mode, mode),
            paired_vs_no_dfs=shared.paired(enriched, 'single_'+mode, 'no_dfs'))
    reader.checked(__file__)
    if not reader.unchanged():
        raise ValueError('Sources changed during report')
    result = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        source_audit=io.ref(audit_path), source_protocol=audit['source_protocol'], source_inputs=audit['source_inputs'],
        report_source=io.ref(__file__), phase=protocol.get('phase'), smoke=bool(protocol.get('smoke')),
        case_count=len(entries), repeats=protocol['repeats'], modes=list(modes),
        network_date_backgrounds=len({(e['case'],e['start_date']) for e in entries}),
        baseline_rows=len(baseline), new_rows=len(new), cases=cases, aggregate_by_mode=aggregate,
        new_ac_rows=complete.get('new_ac_rows'), reused_ac_rows=complete.get('reused_ac_rows'),
        zero_sufficient_quotes=complete.get('zero_sufficient_quotes'),
        internal_truth_covered_cases=sum(bool(c['truth_in_native_bank']) for c in coverage.values()),
        full_support_truth_covered_cases=sum(bool(c['full_support_truth_in_native_bank']) for c in coverage.values()),
        quotes=quote_summary, baseline_quotes=baseline_quotes, case_timing=timing, path_records=enriched,
        recorded_failures=[r for r in enriched if r['audit_status'] != 'passed'],
        original_failed_tasks=audit.get('original_failed_tasks', []), task_errors=audit.get('task_errors', []),
        original_execution_failed_rows=audit.get('original_execution_failed_rows', []),
        maximum_native_SSE_replay_error=post.maximum_error, sources_unchanged=True,
        statistical_scope='Known physical days, frozen forecasts and original paired eta/meter streams; new single-target AC action executions, not independent confirmation',
        planning_scope='History-only single target; all foreign native competitors; unchanged error law and forward historical direction guards',
        objective_scope='Minimize activation fee; availability and total settlement separately reported',
        rotation_scope='Original rotations are matched only to old multi-target plans; there is no new single-target same-fee rotation',
        input_refs=[dict(path=p, sha256=h) for p,h in reader.refs.items()])
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    io.write(out/'report.json', result)
    (out/'REPORT.md').write_text(markdown(result), encoding='utf-8')
    io.write(out/'complete.json', dict(status='complete', sources_unchanged=True,
        baseline_rows=len(baseline), new_rows=len(new), report=io.ref(out/'report.json'), markdown=io.ref(out/'REPORT.md')))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = report(args.audit, args.out)
    print({key:result[key] for key in ('status','baseline_rows','new_rows','maximum_native_SSE_replay_error','sources_unchanged')})
