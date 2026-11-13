"""Join paired procurement intensities without double counting new days."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ... import io
from .report import make, summary, paired, bias_and_energy
from .study import collect


def run(base, supplement, destination):
    destination.mkdir(parents=True, exist_ok=True)
    reports = [make(base, destination/'main'), make(supplement, destination/'intensity')]
    rows, plans, diagnostics, adverse = [], [], [], []
    names = ['no_dfs', 'optimized_small', 'rotation_small', 'optimized_minimal', 'rotation_minimal',
             'optimized_two', 'rotation_two', 'optimized_four', 'rotation_four']
    mapping = [dict(no_dfs='no_dfs', optimized_g0='optimized_small', rotation_g0='rotation_small',
                    optimized_g1='optimized_minimal', rotation_g1='rotation_minimal'),
               dict(no_dfs='no_dfs', optimized_g0='optimized_two', rotation_g0='rotation_two',
                    optimized_g1='optimized_four', rotation_g1='rotation_four')]
    baseline = {}
    for index, directory in enumerate((base, supplement)):
        own = collect(directory, 'confirmation')
        diag = bias_and_energy(directory, 'confirmation', own)
        for d in diag:
            if index and d['strategy'] == 'no_dfs':
                continue
            diagnostics.append(dict(d, strategy=mapping[index][d['strategy']]))
        for r in own:
            if r['strategy'] == 'all':
                rows.append(dict(r, source_study=index))
                continue
            if r['strategy'] == 'no_dfs':
                key = (r['id'], r['day'])
                if index:
                    original = baseline[key]
                    require = all(original[k] == r[k] for k in ('replicate', 'status', 'p_true', 'native_map_correct'))
                    if not require:
                        raise AssertionError('The two intensity studies did not use the same baseline path')
                    for file in ('physical_background.npz', 'no_dfs.npz'):
                        a = np.load(base/'confirmation'/r['id']/f"day{r['day']:03d}"/file)
                        b = np.load(supplement/'confirmation'/r['id']/f"day{r['day']:03d}"/file)
                        if any(not np.array_equal(a[k], b[k]) for k in a.files):
                            raise AssertionError('Paired physical or metering arrays differ')
                    continue
                baseline[key] = r
            rows.append(dict(r, strategy=mapping[index][r['strategy']], source_study=index))
        for path in sorted((directory/'confirmation').glob('*/day*/plans.json')):
            for k, p in io.read(path).items():
                plans.append(dict(p, strategy=mapping[index]['optimized_'+k]))
        own_baseline = {(r['id'], r['day']): r for r in own if r['strategy'] == 'no_dfs' and r['status'] == 'complete'}
        for strategy in ('optimized_g0', 'optimized_g1'):
            candidates = [r for r in own if r['strategy'] == strategy and r['status'] == 'complete' and r['p_true'] is not None]
            if not candidates:
                continue
            worst = min(candidates, key=lambda r: r['p_true']-own_baseline[(r['id'], r['day'])]['p_true'])
            before = own_baseline[(worst['id'], worst['day'])]
            selection = {e['id']: e for e in io.read(directory/'confirmation_selection.json')['entries']}
            s = io.read(selection[worst['id']]['score']['path'])
            Ht, Hj = [np.asarray(s['parameters'][i]['H']) for i in (worst['target'], worst['runner'])]
            daydir = directory/'confirmation'/worst['id']/f"day{worst['day']:03d}"
            with np.load(daydir/'physical_background.npz') as z:
                root = np.array(z['root'])
            decomposition = []
            for name, r in [('no_dfs', before), (strategy, worst)]:
                with np.load(daydir/(name+'.npz')) as z:
                    power = np.c_[z['true_P'], z['true_Q']]
                    residual = root[:, None]**2-z['true_V']**2-power@Ht.T
                difference = power@(Ht-Hj).T
                distance = float(np.square(difference).sum()/(2*s['sigma2']))
                bias = float(np.sum(residual*difference)/s['sigma2'])
                historical = s['models'][worst['target']]['log_probability']-s['models'][worst['runner']]['log_probability']
                measured = r['target_runner_logodds']-historical
                decomposition.append(dict(strategy=name, model_distance=distance, frozen_parameter_bias_term=bias,
                    noiseless_actual_new_logodds=distance+bias, measured_new_logodds=measured,
                    meter_error_contribution=measured-distance-bias))
            adverse.append(dict(strategy=mapping[index][strategy], id=worst['id'], day=worst['day'],
                p_true_before=before['p_true'], p_true_after=worst['p_true'],
                change_pp=100*(worst['p_true']-before['p_true']), planned_fee_yuan=worst['planned_fee_yuan'],
                native_correct_before=before['native_map_correct'], native_correct_after=worst['native_map_correct'],
                decomposition=decomposition))
    table = {s: dict(summary([r for r in rows if r['strategy'] == s]),
                    paired={} if s == 'no_dfs' else paired(rows, s)) for s in names}
    solver = {}
    for strategy in [n for n in names if n.startswith('optimized')]:
        selected = [p for p in plans if p['strategy'] == strategy]
        solver[strategy] = dict(attempted_quotes=len(selected), feasible_quotes=sum(p['feasible'] for p in selected),
            unresolved=sum(p.get('status') == 'unresolved' for p in selected),
            proved_infeasible=sum(p.get('infeasibility_proved', False) for p in selected),
            global_fee_bound_closed=sum(p.get('global_optimality_verified', False) for p in selected),
            native_time_limits=sum(p.get('native_status') == 9 for p in selected),
            retained_initial_points=sum(p.get('source') == 'supporting_LP_initial_point_retained' for p in selected),
            native_improved_verified_initial_points=sum(p.get('source') == 'native_QCQP_improved_initial_point' for p in selected),
            native_found_without_verified_initial=sum(p.get('source') == 'native_QCQP_found_feasible' for p in selected),
            verified_initial_points=sum(p.get('initial', {}).get('domain_feasible', False) and p.get('initial', {}).get('evidence_feasible', False) for p in selected),
            same_planned_fee_failures=sum(not p.get('rotation_same_planned_fee', True) for p in selected),
            mean_seconds=float(np.mean([p['seconds'] for p in selected])) if selected else None,
            mean_relative_fee_gap=float(np.mean([p.get('relative_gap', 0) for p in selected if p['feasible']])) if any(p['feasible'] for p in selected) else None)
    outcome = dict(history=reports[0]['history'], independent_confirmation_days=len(baseline),
        confirmation_conditions=len({k[0] for k in baseline}), table=table, solver=solver,
        procurement_evidence_targets=dict(minimal=float(np.log(1.1)), small=float(np.log(1.5)), two=2., four=4.),
        energy_and_bias=diagnostics, worst_posterior_drops=adverse,
        source_protocols=[io.ref(d/'protocol.json') for d in (base, supplement)],
        confirmation_selections=[io.ref(d/'confirmation_selection.json') for d in (base, supplement)],
        report_source=io.ref(__file__),
        interpretation='Targeted development search; one confirmation sample shared across all policies. Failed planning quotes execute zero and remain in all-day results. Activation fees use example offers, energy value excludes losses.')
    io.write(destination/'COMPARISON.json', outcome)
    labels = dict(no_dfs='无灵活性服务', optimized_minimal='优化，增量log(1.1)', rotation_minimal='同费轮换，log(1.1)',
        optimized_small='优化，增量log(1.5)', rotation_small='同费轮换，log(1.5)', optimized_two='优化，增量2',
        rotation_two='同费轮换，增量2', optimized_four='优化，增量4', rotation_four='同费轮换，增量4')
    lines = ['# 配网灵活性：独立AC新日比较', '',
        f"{outcome['confirmation_conditions']}个开发选定历史条件，共{len(baseline)}条独立新日；所有强度共享这些路径。", '',
        '|方案|平均真拓扑后验|相对无服务/百分点|提高/下降天数|后验≥90%|完整恢复|平均激活费/元|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for strategy in ['no_dfs', 'optimized_minimal', 'rotation_minimal', 'optimized_small', 'rotation_small', 'optimized_two', 'rotation_two', 'optimized_four', 'rotation_four']:
        row = table[strategy]
        compare = row['paired']
        if row['mean_p_true'] is None:
            continue
        change = '—' if not compare else f"{compare['mean_gain_pp']:.3f}"
        direction = '—' if not compare else f"{compare['improved_days']}/{compare['decreased_days']}"
        lines.append(f"|{labels[strategy]}|{100*row['mean_p_true']:.3f}%|{change}|{direction}|{row['p_true_90_count']}/{row['truth_covered_rows']}|{row['native_correct_count']}/{row['completed_rows']}|{row['mean_planned_fee_yuan']:.4f}|")
    lines.extend(['', '提高/下降按逐日后验差超过1e-12计数；求解未决的零动作回退保留在分母。完整恢复使用原完整参数库的strict MSE选择；后验使用训练集预选参数的不同拓扑库。', '',
        '|目标|可行报价|证明最优|达到时限|仅保留初值|平均求解时间/秒|', '|---|---:|---:|---:|---:|---:|'])
    for name, r in solver.items():
        lines.append(f"|{labels[name]}|{r['feasible_quotes']}/{r['attempted_quotes']}|{r['global_fee_bound_closed']}|{r['native_time_limits']}|{r['retained_initial_points']}|{r['mean_seconds']:.3f}|")
    lines.extend(['', '求解时间包含LP初值、模型及QCQP。未闭合报价保留原问题有效下界与可行上界，逐日报价仍在私有输出中；无法在时限内找到可行点记为未决，不记作无解。', '',
        '历史覆盖、所有开发对照、参数偏差分解、电能价值变化与逐强度完整统计见COMPARISON.json及main/intensity。强度2和4在初始开发结果后、独立确认前补充锁定；两天开发路径被复用，没有增加独立样本数。', ''])
    (destination/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('base', 'supplement', 'destination'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base.resolve(), args.supplement.resolve(), args.destination.resolve())
    print('independent_confirmation_days', result['independent_confirmation_days'])
