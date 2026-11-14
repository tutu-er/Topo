"""Describe every predeclared sampling/noise cell after independent audit.

Read saved JSON only. No identifier, optimizer, AC simulation, interval,
significance test, or post-hoc threshold selection is run here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

from ... import io
from ..distribution_panel_20261115 import report as panel


POLICIES = panel.POLICIES
MODES = panel.MODES
STRIDES = (1, 2, 4)
MULTIPLIERS = (.5, 1., 2.)
HIGH_WEIGHT = .9
DIMENSIONS = ('id', 'original_id', 'case', 'start_date', 'grid_order', 'stride',
              'meter_multiplier', 'training_rows', 'validation_rows',
              'new_points', 'forecast_relative_sd')


def dimensions(ledger, entries, *, smoke=False):
    """Join ready entries to the complete ledger, including failed fits."""
    ready = {e['id']: e for e in entries}
    if len(ready) != len(entries):
        raise ValueError('Duplicate prepared grid identity')
    result = {}
    for attempt in ledger:
        task = attempt.get('task', attempt)
        tag = task['id']
        if tag in result:
            raise ValueError('Duplicate qualification identity: '+tag)
        entry = ready.get(tag, {})
        for key in DIMENSIONS:
            if key in task and key in entry and task[key] != entry[key]:
                raise ValueError('Grid dimension changed: '+tag+'/'+key)
        merged = dict(task, **entry)
        d = {key: merged[key] for key in DIMENSIONS if key in merged}
        d['training_rows'] = int(merged.get('training_rows', merged.get('training_points', 0)))
        d['validation_rows'] = int(merged.get('validation_rows', merged.get('validation_points', 0)))
        d['new_points'] = int(merged.get('new_points', 96))
        d['forecast_relative_sd'] = float(merged.get('forecast_relative_sd', .05))
        if d.get('stride') not in STRIDES or d.get('meter_multiplier') not in MULTIPLIERS:
            raise ValueError('Unexpected sampling/noise cell: '+tag)
        if (d['training_rows'], d['validation_rows'], d['new_points']) != (192//d['stride'], 96//d['stride'], 96):
            raise ValueError('Sampling count differs from fixed grid: '+tag)
        if not d.get('original_id') or d['forecast_relative_sd'] != .05:
            raise ValueError('Missing original world or changed prediction error: '+tag)
        d.update(meter_PQ_relative_sd=.005*d['meter_multiplier'],
                 meter_V_relative_sd=.0005*d['meter_multiplier'],
                 history_spacing_minutes=15*d['stride'],
                 new_day_scoring_fraction=96/(d['validation_rows']+96))
        result[tag] = d
    if set(ready)-set(result):
        raise ValueError('Ready input outside qualification ledger')
    worlds = {d['original_id'] for d in result.values()}
    actual = {(d['original_id'], d['stride'], d['meter_multiplier']) for d in result.values()}
    if len(actual) != len(result):
        raise ValueError('Repeated world/stride/noise cell')
    if not smoke:
        expected = {(w, s, m) for w in worlds for s in STRIDES for m in MULTIPLIERS}
        if len(worlds) != 4 or len(result) != 36 or actual != expected:
            raise ValueError('Formal report requires the complete four-world, 36-cell ledger')
    return result


def summarize(rows):
    answer = panel.summarize(rows)
    done = [r for r in rows if r['audit_status'] == 'passed']
    values = [float(r['evaluation']['p_true']) for r in done]
    if any(not math.isfinite(p) or p < 0 or p > 1+1e-12 for p in values):
        raise ValueError('Invalid audited true-family weight')
    answer.update(mean_p_true_completed=panel.mean_or_none(values),
        p_true_denominator=len(values),
        native_correct_and_true_high=sum(panel.recovered(r) and r['evaluation']['p_true'] >= HIGH_WEIGHT for r in done),
        weight_scope='Sum of fixed representative-bank weights of correct internal topology families; absent true family has weight zero. Failed executions have unknown weight, not zero.',
        wrong_high_scope='A wrong representative winner has normalized weight at least 0.9; native-bank correctness is reported separately.')
    return answer


def paired(rows, policy, baseline):
    # Reuse paired counts, but discard date/bootstrap fields: grid cells are
    # repeated conditions on four worlds, not new networks or independent days.
    old = panel.paired_comparison(rows, policy, baseline)
    answer = {key: old[key] for key in ('policy', 'baseline', 'n', 'gained', 'lost', 'pooled_path_difference')}
    a = {panel.item_key(r)[:3]: r for r in rows if r['strategy'] == policy}
    b = {panel.item_key(r)[:3]: r for r in rows if r['strategy'] == baseline}
    both = [key for key in a if a[key]['audit_status'] == b[key]['audit_status'] == 'passed']
    differences = [a[k]['evaluation']['p_true']-b[k]['evaluation']['p_true'] for k in both]
    answer.update(completed_pairs=len(both), mean_p_true_change_completed=panel.mean_or_none(differences),
        new_wrong_high=sum(a[k]['evaluation']['wrong_high'] and not b[k]['evaluation']['wrong_high'] for k in both),
        removed_wrong_high=sum(b[k]['evaluation']['wrong_high'] and not a[k]['evaluation']['wrong_high'] for k in both),
        pair_scope='Same cell, prediction and execution random stream; failures count as unrecovered and their unavailable weights are omitted from weight means.')
    return answer


def solver_summary(quotes, mode, expected):
    answer = panel.solver_summary(quotes, mode)
    selected = [q for q in quotes if q['policy'] == mode]
    if len(selected) > expected:
        raise ValueError('More quotes than predeclared forecasts')
    feasible_initial = [q for q in selected if (q.get('initial') or {}).get('feasible')]
    comparable = [q for q in feasible_initial if q.get('new_gate_feasible')
                  and q.get('cost_yuan') is not None and q['initial'].get('cost_yuan') is not None]
    answer.update(expected_quotes=expected, missing_quotes=expected-len(selected),
        infeasible_or_unresolved_quotes=sum(not q['new_gate_feasible'] for q in selected),
        native_time_limit_quotes=sum(q.get('native_status') == 9 for q in selected),
        quote_exceptions=sum(q.get('status') == 'exception' for q in selected),
        infeasibility_proved=sum(bool(q.get('infeasibility_proved')) for q in selected),
        source_counts=dict(Counter(str(q.get('source')) for q in selected)),
        feasible_initial_count=len(feasible_initial),
        initial_name_counts=dict(Counter(str((q.get('initial') or {}).get('name')) for q in selected)),
        final_strictly_cheaper_than_initial=sum(q['cost_yuan'] < q['initial']['cost_yuan']-1e-8 for q in comparable),
        final_equal_to_initial_cost=sum(abs(q['cost_yuan']-q['initial']['cost_yuan']) <= 1e-8 for q in comparable),
        initial_comparison_count=len(comparable),
        quote_wall_seconds=panel.distribution([q.get('quote_wall_seconds') for q in selected]),
        native_attributes_status='Preserved in audited quotes; time limit is not proof of infeasibility or global optimality.',
        bound_scope='Activation-cost bound for the stated sufficient QCQP. Reported procurement bills also include reserved capacity; fallback bills are not automatically upper bounds for a failed gate.')
    return answer


def evidence_changes(rows, policy):
    """Keep the audit's fixed true/wrong pair and the paired zero baseline."""
    selected = {panel.item_key(r)[:3]:r for r in rows if r['strategy'] == policy}
    baseline = {panel.item_key(r)[:3]:r for r in rows if r['strategy'] == 'no_dfs'}
    if set(selected) != set(baseline):
        raise ValueError('Unequal paired evidence denominators')
    predicted, clean, observed = [], [], []
    missing_pair = failed = missing_mean = 0
    for key, row in selected.items():
        zero = baseline[key]
        if row['audit_status'] != 'passed' or zero['audit_status'] != 'passed':
            failed += 1
            continue
        a, b = row['evaluation'], zero['evaluation']
        if 'pair' not in a or 'pair' not in b:
            missing_pair += 1
            continue
        if a['pair'] != b['pair']:
            raise ValueError('Audit pair changed between procurement and zero action')
        if 'evidence' not in a or 'evidence' not in b:
            raise ValueError('Audited pair lacks its score decomposition')
        clean.append(a['evidence']['clean_logodds']-b['evidence']['clean_logodds'])
        observed.append(a['evidence']['observed_logodds']-b['evidence']['observed_logodds'])
        if 'historical_mean_diagnostic' in a and 'historical_mean_diagnostic' in b:
            predicted.append(a['historical_mean_diagnostic']['K_nominal_forecast_mu']
                             -b['historical_mean_diagnostic']['K_nominal_forecast_mu'])
        else:
            missing_mean += 1
    return dict(planned_pairs=len(selected), actual_evidence_pairs=len(observed),
        forecast_mean_pairs=len(predicted), failed_execution_pairs=failed,
        unavailable_fixed_pair=missing_pair, unavailable_historical_mean=missing_mean,
        mean_forecast_K_logodds_increment=panel.mean_or_none(predicted),
        mean_actual_clean_logodds_increment=panel.mean_or_none(clean),
        mean_actual_observed_logodds_increment=panel.mean_or_none(observed),
        negative_observed_increment_count=sum(v < -1e-8 for v in observed),
        negative_threshold=-1e-8,
        scope='Same audit-fixed true/wrong parameter pair and paired no-action path. Missing true families or unavailable comparisons remain unavailable. No gain ratio or outcome-selected rival.')


def aggregate(rows, quotes, ids, qualification, coverage, protocol):
    ids = set(ids)
    selected = [r for r in rows if r['id'] in ids]
    offered = [q for q in quotes if q['id'] in ids]
    qualifications = [r for r in qualification if r['id'] in ids]
    assessed = [c for c in coverage if c['id'] in ids]
    forecasts = sum(q['data_valid'] for q in qualifications)*protocol['forecasts']
    return dict(cells=len(ids), physical_worlds=len({r['original_id'] for r in qualifications}),
        qualification_counts=dict(planned=len(qualifications), data_valid=sum(r['data_valid'] for r in qualifications),
            history_prepared=sum(r['history_prepared'] for r in qualifications),
            resources_prepared=sum(r['resources_and_forecast_prepared'] for r in qualifications)),
        coverage_counts=dict(assessed=len(assessed), truth_in_native_bank=sum(r['truth_in_native_bank'] for r in assessed),
            truth_in_target_set=sum(r['truth_in_target_set'] for r in assessed), history_correct=sum(r['history_correct'] for r in assessed),
            full_support_truth_in_native_bank=sum(r['full_support_truth_in_native_bank'] for r in assessed)),
        groups={p: summarize([r for r in selected if r['strategy'] == p]) for p in POLICIES},
        paired_vs_no_dfs={p: paired(selected, p, 'no_dfs') for p in POLICIES},
        paired_vs_anchor={p: paired(selected, p, 'anchor') for p in MODES},
        paired_vs_rotation={p: paired(selected, p, p+'_rotation') for p in MODES},
        predicted_vs_actual_evidence={p:evidence_changes(selected,p) for p in POLICIES},
        solver={mode: solver_summary(offered, mode, forecasts) for mode in MODES})


def csv_rows(group, metadata):
    for policy in POLICIES:
        m = group['groups'][policy]; comparison = group['paired_vs_no_dfs'][policy]
        quote = group['solver'].get(policy, {})
        evidence = group['predicted_vs_actual_evidence'][policy]
        settled, planned = m['settled_cost_mean_completed'], m['planned_cost_mean_completed']
        voltage = m['voltage_range_all_buses'] or (None, None)
        yield dict(**metadata, strategy=policy, planned_paths=m['n'], completed=m['completed'],
            recorded_failures=m['recorded_failures'], internal_recovered=m['recovered'],
            full_support_recovered=m['full_support_recovered'], mean_p_true=m['mean_p_true_completed'],
            p_true_denominator=m['p_true_denominator'], true_weight_at_least_09=m['true_high'],
            native_correct_and_true_weight_at_least_09=m['native_correct_and_true_high'],
            wrong_winner_weight_at_least_09=m['wrong_high'], gained_vs_no_dfs=comparison['gained'],
            lost_vs_no_dfs=comparison['lost'], mean_p_true_change_vs_no_dfs=comparison['mean_p_true_change_completed'],
            planned_total_yuan=planned['total_yuan'], settled_total_yuan=settled['total_yuan'],
            settled_activation_yuan=settled['activation_yuan'], settled_availability_yuan=settled['availability_yuan'],
            cost_denominator=m['cost_denominator'], requested_P_kwh=m['requested_P_kwh_mean_completed'],
            delivered_P_kwh=m['delivered_P_kwh_mean_completed'], requested_Q_kvarh=m['requested_Q_kvarh_mean_completed'],
            delivered_Q_kvarh=m['delivered_Q_kvarh_mean_completed'],
            voltage_minimum=voltage[0], voltage_maximum=voltage[1], voltage_screen_failed=m['voltage_screen_failed'],
            forecast_K_logodds_increment=evidence['mean_forecast_K_logodds_increment'],
            clean_logodds_increment=evidence['mean_actual_clean_logodds_increment'],
            observed_logodds_increment=evidence['mean_actual_observed_logodds_increment'],
            observed_logodds_decreased=evidence['negative_observed_increment_count'],
            actual_evidence_denominator=evidence['actual_evidence_pairs'],
            forecast_evidence_denominator=evidence['forecast_mean_pairs'],
            unavailable_fixed_pair=evidence['unavailable_fixed_pair'],
            expected_quotes=quote.get('expected_quotes'), audited_quotes=quote.get('n'),
            missing_quotes=quote.get('missing_quotes'), zero_action_sufficient=quote.get('zero_action_sufficient'),
            gate_feasible=quote.get('new_gate_feasible'), gate_failed_or_unresolved=quote.get('infeasible_or_unresolved_quotes'),
            native_time_limit=quote.get('native_time_limit_quotes'), quote_exceptions=quote.get('quote_exceptions'),
            anchor_fallbacks=quote.get('anchor_fallbacks'), zero_fallbacks=quote.get('zero_fallbacks'),
            nonzero_feasible_gap_mean=quote.get('nonzero_feasible_quotes', {}).get('gap', {}).get('mean'),
            charged_seconds_mean=quote.get('charged_seconds_mean'),
            returned_point_sources=str(quote.get('source_counts', {})), initial_sources=str(quote.get('initial_name_counts', {})))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Empty result table')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def markdown(result):
    overall = result['overall']; counts = overall['qualification_counts']
    text = ['# 历史采样与量测误差网格结果', '',
        '短时接线测试，不计入36格正式结果。' if result['smoke'] else '阶段：已知物理日的完整开发网格。', '',
        f"预定 {counts['planned']} 格，历史准备完成 {counts['history_prepared']} 格，资源及预测准备完成 {counts['resources_prepared']} 格。独立逐行审计通过。", '',
        f"共 {result['physical_worlds']} 个固定网络日，每格 {result['forecasts']} 份预测、{result['repeats']} 条配对执行路径。网格改变历史采样密度和附加量测误差，仍覆盖原两日训练、一日验证；重复路径与九个条件共用物理背景。以下是开发敏感性结果，不将其当作独立日期确认。", '',
        '|策略|内部恢复/全流程路径|新增/丢失|完整支持恢复|平均真族权重|真族权重≥0.9/错误高权重|结算均费（元）|',
        '|---|---:|---:|---:|---:|---:|---:|']
    def number(v): return '缺失' if v is None else f'{v:.6g}'
    for policy in POLICIES:
        g = overall['groups'][policy]; c = overall['paired_vs_no_dfs'][policy]
        text.append(f"|{policy}|{g['recovered']}/{g['n']}|{c['gained']}/{c['lost']}|{g['full_support_recovered']}|{number(g['mean_p_true_completed'])}|{g['true_high']}/{g['wrong_high']}|{number(g['settled_cost_mean_completed']['total_yuan'])}|")
    text += ['', '逐格完整表如下。采样列为训练点/验证点；倍率作用于标准差。恢复结果按全部预定执行路径计数，失败路径留在分母。', '',
        '|输入|训练/验证|量测倍率|候选库含真内部族|历史选对|无采购|旧参考|normal_fixed|mgf_fixed|mgf_forecast|',
        '|---|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for cell in result['cells']:
        d, coverage = cell['dimensions'], cell['coverage']
        def flag(key): return '未核得' if coverage is None else ('是' if coverage[key] else '否')
        values = [f"{cell['groups'][p]['recovered']}/{cell['groups'][p]['n']}" for p in ('no_dfs','anchor',*MODES)]
        text.append('|'+ '|'.join([d['original_id'],f"{d['training_rows']}/{d['validation_rows']}",str(d['meter_multiplier']),
            flag('truth_in_native_bank'),flag('history_correct'),*values])+'|')
    text += ['', '|方法|预定/取得报价|零动作过线|门槛可行|超时|旧参考/零回退|非零可行报价平均gap|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        q = overall['solver'][mode]
        text.append(f"|{mode}|{q['expected_quotes']}/{q['n']}|{q['zero_action_sufficient']}|{q['new_gate_feasible']}|{q['native_time_limit_quotes']}|{q['anchor_fallbacks']}/{q['zero_fallbacks']}|{number(q['nonzero_feasible_quotes']['gap']['mean'])}|")
    text += ['', 'grid_cells.csv 包含全部格、全部八策略的恢复、权重、费用、P/Q交付、电压及求解状态；stride_meter.csv 按九个采样×误差条件汇总。report.json 另列相对旧参考和费用轮换的配对新增/丢失、初值来源、上下界及全部失败。',
        '权重来自冻结代表库，内部正确性来自完整原生候选库，两者分别报告。完整支持恢复同时检查共同根响应项及末端项。失败路径未知权重和费用保持缺失，不填零。',
        '最终评分行数以各格封存的 validation_rows 与新日96点为准：验证96/48/24点时，新日分别占1/2、2/3、4/5。旧辨识收据中的固定“96+96”说明不用于本表。',
        '预测收益采用历史均值模型的对数赔率增量，与同路径无采购比较；实际分别列干净AC及含表计误差的增量，并统计后者下降超过1e−8的路径。竞争对由独立审计固定，缺真族时不编造收益，也不使用增益比率。',
        '调用费是优化目标；结算费包括请求峰值容量费与实际交付调用费。非零报价的gap单独报告，零采购证书不稀释该数值。轮换费用未匹配数量见 report.json。',
        '电压为执行后的全母线检查；规划中未加入电压约束，热额定值缺失。声明模型下的概率门槛与实际AC恢复分开。',
        '原先两例较好的结果使用旧历史噪声与旧参数库，见 CORE_MODEL 的单独说明；未与本轮重新辨识结果合并。', '']
    return '\n'.join(text)


def run(audit_path, out):
    audit_path, out = Path(audit_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    audit = io.read(audit_path)
    if audit.get('status') != 'passed':
        raise ValueError('A passed independent audit is required')
    refs = [io.ref(audit_path), audit['source_protocol'], audit['source_inputs'], audit['records'], audit['quotes']]
    protocol = io.read(panel.checked(audit['source_protocol']))
    if tuple(protocol['policies']) != POLICIES or tuple(protocol['modes']) != MODES:
        raise ValueError('Grid policy list differs from audited implementation')
    if audit['source_inputs'] != protocol['inputs']:
        raise ValueError('Audit and execution input references disagree')
    if protocol.get('phase') != 'grid_development':
        raise ValueError('This report requires a grid development run')
    entries = panel.entries_from(io.read(panel.checked(audit['source_inputs'])))
    ledger = io.read(panel.checked(protocol['qualification_ledger']))
    preparation = io.read(panel.checked(protocol['preparation_failures']))
    refs += [protocol['qualification_ledger'], protocol['preparation_failures']]
    dims = dimensions(ledger, entries, smoke=protocol.get('smoke', False))
    rows, qualification = panel.complete_panel(panel.lines(audit['records']), ledger, preparation, protocol)
    rows = [dict(r, **{k:v for k,v in dims[r['id']].items() if k not in r}) for r in rows]
    qualification = [dict(r, **{k:v for k,v in dims[r['id']].items() if k not in r}) for r in qualification]
    quotes = panel.lines(audit['quotes'])
    identities = {(q['id'], int(q['forecast']), q['policy']) for q in quotes}
    if len(identities) != len(quotes) or any(q['id'] not in dims or q['policy'] not in MODES
            or not 0 <= int(q['forecast']) < protocol['forecasts'] for q in quotes):
        raise ValueError('Duplicate or undeclared quotation')
    coverage = audit['input_coverage']; coverage_by_id = {c['id']:c for c in coverage}
    if len(coverage_by_id) != len(coverage) or set(coverage_by_id)-set(dims):
        raise ValueError('Duplicate or undeclared coverage record')
    def group(ids): return aggregate(rows, quotes, ids, qualification, coverage, protocol)
    ordered = sorted(dims.values(), key=lambda d:(d.get('grid_order', 0), d['id']))
    cells = [dict(dimensions=d, qualification=next(q for q in qualification if q['id']==d['id']),
                  coverage=coverage_by_id.get(d['id']), **group([d['id']])) for d in ordered]
    by_grid = [dict(stride=s, meter_multiplier=m, **group([d['id'] for d in ordered
               if d['stride']==s and d['meter_multiplier']==m])) for s in STRIDES for m in MULTIPLIERS
               if any(d['stride']==s and d['meter_multiplier']==m for d in ordered)]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=io.ref(__file__),
        helper_sources=[io.ref(panel.__file__)], audit=io.ref(audit_path), source_protocol=audit['source_protocol'],
        source_inputs=audit['source_inputs'], phase=protocol['phase'], smoke=protocol.get('smoke',False),
        forecasts=protocol['forecasts'], repeats=protocol['repeats'], high_weight_threshold=HIGH_WEIGHT,
        physical_worlds=len({d['original_id'] for d in ordered}), full_pipeline_strategy_rows=len(rows),
        qualification=qualification, cells=cells, by_stride_meter=by_grid, overall=group(dims),
        by_world={world:group([d['id'] for d in ordered if d['original_id']==world])
                  for world in sorted({d['original_id'] for d in ordered})},
        failed_qualification=[q for q in qualification if q['failure_category']],
        failed_execution_rows=[{k:r.get(k) for k in ('id','forecast','repeat','strategy','original_status','audit_status')}
                              for r in rows if r['audit_status'] != 'passed'],
        failed_or_unresolved_quotes=[q for q in quotes if not q['new_gate_feasible'] or q['native_status']==9 or q['status']=='exception'],
        rotation_unmatched_fees=audit['rotation_unmatched_fees'],
        original_failed_tasks=audit.get('original_failed_tasks',[]),
        original_quote_exceptions=audit.get('original_quote_exceptions',[]),
        ideal_requested_diagnostic_failures=audit.get('ideal_requested_diagnostic_failures',[]),
        statistical_scope='Descriptive paired sensitivity study on four fixed network/day worlds. Grid cells, forecast draws and execution repetitions are not independent physical days. No confidence interval, significance test or outcome-selected threshold is used.',
        weighting_scope='Each predeclared grid cell and its paths have equal descriptive weight; failed data-valid cells remain unrecovered. Lower validation count changes both fitting data and final history/new-day score balance.',
        probability_scope='Mean p_true is conditional on the frozen representative candidate library and shared score scale, not a calibrated truth probability or frequency guarantee.',
        old_core_results_scope='Old successful cases are separate historical references, not pooled with newly perturbed/refitted grid cells.')
    for reference in refs:
        panel.checked(reference)
    result['inputs_unchanged'] = True
    result['input_refs'] = refs
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/'report.py')
    io.write(out/'report.json', result)
    write_csv(out/'grid_cells.csv', (r for c in cells for r in csv_rows(c, dict(c['dimensions'],
        source_status=c['qualification']['source_status'], failure_category=c['qualification']['failure_category'],
        truth_in_native_bank=None if c['coverage'] is None else c['coverage']['truth_in_native_bank'],
        truth_in_target_set=None if c['coverage'] is None else c['coverage']['truth_in_target_set'],
        history_correct=None if c['coverage'] is None else c['coverage']['history_correct']))))
    write_csv(out/'stride_meter.csv', (r for g in by_grid for r in csv_rows(g,
        dict(stride=g['stride'],meter_multiplier=g['meter_multiplier'],cells=g['cells'],physical_worlds=g['physical_worlds']))))
    (out/'REPORT.md').write_text(markdown(result), encoding='utf8')
    io.write(out/'complete.json', dict(status='complete',inputs_unchanged=True,
        report=io.ref(out/'report.json'),text=io.ref(out/'REPORT.md'),
        cell_csv=io.ref(out/'grid_cells.csv'),grid_csv=io.ref(out/'stride_meter.csv')))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.audit, args.out)
    print(dict(cells=len(result['cells']), rows=result['full_pipeline_strategy_rows'],
               physical_worlds=result['physical_worlds'], inputs_unchanged=result['inputs_unchanged']))
