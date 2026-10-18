"""Read-only arithmetic summary of saved three/seven-day AC fits.

No simulator, AC solver or estimator is imported. Evaluator outputs are clearly
marked oracle diagnostics; this module does not feed truth back into inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from math import isfinite
from pathlib import Path

import numpy as np


CANDIDATES = ('star', 'pair01', 'pair02', 'pair12')
TIE_SSE = 1e-6
ERROR_FIELDS = ('r_leaf_relative', 'x_leaf_relative', 'r_total_relative',
                'x_total_relative', 'beta_leaf_relative',
                'local_true_PQ_true_junction_RMSE_sigma_holdout',
                'contrast_RMSE_sigma_holdout',
                'true_PQ_voltage_RMSE_sigma_holdout')


def _key(spec):
    return f"{spec['network']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def _finite(value):
    try:
        value = float(value)
        return value if isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _read(path, manifest):
    path = Path(path)
    data = path.read_bytes()
    manifest[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def _stats(values, denominator=None):
    raw = list(values)
    good = [x for x in (_finite(v) for v in raw) if x is not None]
    return dict(available=len(good), denominator=len(raw) if denominator is None else int(denominator),
                mean=float(np.mean(good)) if good else None,
                minimum=min(good) if good else None, maximum=max(good) if good else None,
                sample_standard_deviation=float(np.std(good, ddof=1)) if len(good) > 1 else None)


def _ranking(fits):
    if any(not fits.get(k, {}).get('qualified') or _finite(fits[k].get('loss')) is None
           for k in CANDIDATES):
        return dict(available=False, reason='at_least_one_candidate_unqualified_or_missing',
                    absolute_SSE_tolerance=TIE_SSE)
    ordered = sorted(CANDIDATES, key=lambda k:(float(fits[k]['loss']), k))
    groups = []
    for candidate in ordered:
        if not groups or float(fits[candidate]['loss'])-float(fits[groups[-1][0]]['loss']) > TIE_SSE:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    star_rank = next(i+1 for i, group in enumerate(groups) if 'star' in group)
    return dict(available=True, tied_groups=groups, star_tie_group_rank=star_rank,
                star_in_first_group='star' in groups[0], unique_first=len(groups[0]) == 1,
                absolute_SSE_tolerance=TIE_SSE,
                interpretation='local attained-loss ordering; not a certified global likelihood rank')


def _attempt_summary(fitted):
    attempts = fitted.get('attempts', [])
    losses = [_finite(a.get('loss')) for a in attempts]
    valid = [x for x in losses if x is not None]
    gradients = [_finite(a.get('projected_gradient')) for a in attempts]
    return dict(count=len(attempts), qualified=sum(bool(a.get('qualified')) for a in attempts),
                physical=sum(bool(a.get('physical')) for a in attempts),
                loss_min=min(valid) if valid else None, loss_max=max(valid) if valid else None,
                loss_range=max(valid)-min(valid) if valid else None,
                projected_gradient=_stats(gradients),
                stop_reasons=dict(Counter(a.get('reason', 'exception') for a in attempts)),
                parameter_spread_available=False,
                parameter_spread_reason='per-start R/X or theta is not recorded; cannot infer it from SSE or PG')


def summarize(directory):
    directory = Path(directory)
    manifest = {}
    config = _read(directory/'configuration.json', manifest)
    audit_path = directory/'audit.json'
    audit = _read(audit_path, manifest) if audit_path.exists() else None
    audit_ok = bool(audit is not None and (audit.get('audit_pass') is True or
                    audit.get('passed') is True or audit.get('status') == 'passed'))
    completion = None
    for name in ('completed.json', 'progress.json'):
        if (directory/name).exists():
            completion = _read(directory/name, manifest)
            break
    rows, task_rows, failures, missing, diagnostic_rows = [], [], [], [], []
    for spec in config['tasks']:
        key = _key(spec)
        path = directory/'results'/f'{key}.json.gz'
        if not path.exists():
            missing.append(key)
            task_rows.append(dict(spec=spec, key=key, status='missing', qualified=0,
                                  ranking=dict(available=False, reason='missing_task')))
            for name in CANDIDATES:
                rows.append(dict(**spec, key=key, candidate=name, task_status='missing',
                                 qualified=False, status='missing'))
            continue
        record = _read(path, manifest)
        inference_path = directory/'inference'/f'{key}.json.gz'
        inference = _read(inference_path, manifest) if inference_path.exists() else {}
        actual_sha = manifest.get(str(inference_path.resolve()))
        sealed_sha = record.get('inference_sha256')
        if sealed_sha and actual_sha != sealed_sha:
            raise ValueError('saved inference hash mismatch: '+key)
        fits = inference.get('fits', {})
        evaluation = record.get('evaluation', {}).get('candidates', {})
        if record.get('status') != 'completed':
            failures.append(dict(key=key, status=record.get('status'), error=record.get('error'),
                                 traceback=record.get('traceback')))
        task = dict(spec=spec, key=key, status=record.get('status'),
                    seconds=_finite(record.get('seconds')), qualified=int(record.get('qualified', 0)),
                    ranking=_ranking(fits),
                    inference_seal_matches=bool(sealed_sha and actual_sha == sealed_sha),
                    truth_read_after_inference_seal=record.get('evaluation', {}).get('truth_read_after_inference_seal'))
        task_rows.append(task)
        for name in CANDIDATES:
            fitted = fits.get(name, {})
            diag = inference.get('diagnostics', {}).get(name, {})
            result = evaluation.get(name, {})
            information = fitted.get('information', {})
            attempts = _attempt_summary(fitted)
            row = dict(**spec, key=key, candidate=name, task_status=record.get('status'),
                       status=fitted.get('status', 'missing'), qualified=bool(fitted.get('qualified')),
                       loss=_finite(fitted.get('loss')), J=_finite(fitted.get('J')),
                       projected_gradient=_finite(fitted.get('projected_gradient')),
                       profile_rank=information.get('rank'), profile_width=information.get('width'),
                       active_line_coordinate_count=information.get('active_line_coordinate_count'),
                       active_line_coordinates=information.get('active_line_coordinates', []),
                       interior_CRLB_interpretation_applicable=information.get('interior_CRLB_interpretation_applicable'),
                       seconds=_finite(fitted.get('seconds')), ac_calls=fitted.get('ac_calls', 0),
                       attempts=attempts, attempts_count=attempts['count'],
                       attempts_qualified=attempts['qualified'], attempts_loss_range=attempts['loss_range'],
                       attempts_parameter_spread='unavailable_per_start_parameters_not_saved',
                       evaluation_only=True,
                       oracle_true_root_is_additional_information_for_U=spec['boundary'] == 'U',
                       local_response_uses_true_PQ_and_true_junction=True,
                       loss_recomputed_difference=diag.get('loss_absolute_difference'),
                       derivative_checks=diag.get('finite_direction_checks', []),
                       # Full derivative trajectories stay in the sealed raw
                       # records; the summary needs only identity check values.
                       local_identity={k:v for k,v in diag.get('local_star_identity', {}).items()
                                       if k not in ('dL_dr','dL_dx')},
                       diagnostic_information=diag.get('profile_information', {}),
                       observed_pq_excitation=diag.get('observed_pq_excitation', []))
            raw_channels=int(spec['N'])*(9+(spec['boundary']=='R'))
            row['raw_random_meter_count'] = raw_channels
            row['SSE_per_raw_random_meter'] = row['loss']/raw_channels if row['loss'] is not None else None
            for field in ERROR_FIELDS:
                row[field] = _finite(result.get(field))
            row['r_leaf_each'] = result.get('r_leaf_each')
            row['x_leaf_each'] = result.get('x_leaf_each')
            rows.append(row)
            if diag:
                diagnostic_rows.append(dict(key=key, candidate=name, diagnostics=diag))
            if not row['qualified']:
                failures.append(dict(key=key, candidate=name, status=row['status'],
                                     projected_gradient=row['projected_gradient'], attempts=attempts))
    groups = []
    for network in sorted({s['network'] for s in config['tasks']}):
        for length in sorted({int(s['N']) for s in config['tasks']}):
            for boundary in ('U', 'R', 'K'):
                expected = [s for s in config['tasks'] if (s['network'], s['N'], s['boundary']) == (network, length, boundary)]
                if not expected:
                    continue
                group_rows = [r for r in rows if (r['network'], r['N'], r['boundary']) == (network, length, boundary)]
                stars = [r for r in group_rows if r['candidate'] == 'star']
                qualified = [r for r in stars if r['qualified']]
                tasks = [r for r in task_rows if (r['spec']['network'], r['spec']['N'], r['spec']['boundary']) == (network, length, boundary)]
                rankings = [t['ranking'] for t in tasks if t['ranking'].get('available')]
                group = dict(network=network, N=length, boundary=boundary,
                    expected_histories=len(expected), completed_tasks=sum(t['status'] == 'completed' for t in tasks),
                    qualified_true_star=len(qualified), qualified_candidates=sum(r['qualified'] for r in group_rows),
                    candidate_denominator=4*len(expected), metrics={},
                    root_interpretation='common trunk R/X nonunique where the original-domain gauge is feasible' if boundary == 'U' else
                    'extra noisy cut-voltage sensor' if boundary == 'R' else 'extra ideal known cut voltage',
                    rank_counts=dict(Counter(str(r.get('profile_rank')) for r in stars)),
                    downstream_rank_counts=dict(Counter(str(r.get('diagnostic_information', {}).get('downstream_after_trunk_profile', {}).get('rank')) for r in stars)),
                    active_bound_true_star=sum(bool(r.get('active_line_coordinate_count')) for r in stars),
                    ranking_available=len(rankings), ranking_denominator=len(expected),
                    true_star_in_best_tie_group=sum(r['star_in_first_group'] for r in rankings),
                    unique_true_star_best=sum(r['star_in_first_group'] and r['unique_first'] for r in rankings),
                    per_start_parameter_spread='not available; starts record SSE, PG and stop reason only')
                for field in ERROR_FIELDS:
                    group['metrics'][field] = _stats((r.get(field) for r in qualified), len(expected))
                group['metrics']['SSE_per_raw_random_meter'] = _stats((r.get('SSE_per_raw_random_meter') for r in qualified), len(expected))
                for field in ('r_leaf_each', 'x_leaf_each'):
                    group[field] = [_stats((r[field][j] for r in qualified if r.get(field) is not None), len(expected)) for j in range(3)]
                group['active_line_coordinates'] = {r['key']:r.get('active_line_coordinates', []) for r in stars}
                if boundary == 'R':
                    group['without_added_meter_rank_counts'] = dict(Counter(str(r.get('diagnostic_information', {}).get('without_added_root_meter_at_same_point', {}).get('rank')) for r in stars))
                groups.append(group)
    paired, paired_groups = [], []
    star_map = {(r['network'], r['seed'], r['N'], r['boundary']):r for r in rows if r['candidate'] == 'star'}
    families = sorted({(s['network'], int(s['seed'])) for s in config['tasks']})
    for network, seed in families:
        for boundary in ('U', 'R', 'K'):
            short = star_map.get((network, seed, 288, boundary))
            long = star_map.get((network, seed, 672, boundary))
            if short is None and long is None:
                continue
            valid = bool(short and long and short['qualified'] and long['qualified'])
            pair = dict(network=network, seed=seed, boundary=boundary, available=valid,
                        interpretation='same-family 7-day minus 3-day; negative errors mean improvement', differences={})
            for field in ERROR_FIELDS:
                a, b = (short or {}).get(field), (long or {}).get(field)
                pair['differences'][field] = b-a if valid and a is not None and b is not None else None
            paired.append(pair)
    for network in sorted({s['network'] for s in config['tasks']}):
        for boundary in ('U', 'R', 'K'):
            parts = [p for p in paired if p['network'] == network and p['boundary'] == boundary]
            if parts:
                paired_groups.append(dict(network=network, boundary=boundary,
                    available=sum(p['available'] for p in parts), denominator=len(parts),
                    differences={field:_stats((p['differences'][field] for p in parts), len(parts)) for field in ERROR_FIELDS}))
    finite_checks = [d for r in rows for d in r.get('derivative_checks', [])]
    identities = [r.get('local_identity', {}) for r in rows if r.get('local_identity', {}).get('supported')]
    def maximum(values):
        good = [v for v in (_finite(v) for v in values) if v is not None]
        return max(good) if good else None
    summary = dict(schema='local_rx_three_day_summary_v1',
        pilot=bool(config.get('pilot')), scientific_interpretation='pilot schema check only' if config.get('pilot') else 'frozen ordinary-load main batch',
        independent_random_families=len(families), expected_independent_random_families=config.get('independent_random_families'),
        expected_conditions=len(config['tasks']), completed_conditions=sum(t['status'] == 'completed' for t in task_rows),
        missing_conditions=missing, candidate_denominator=4*len(config['tasks']),
        qualified_candidates=sum(r['qualified'] for r in rows),
        qualified_true_star=sum(r['qualified'] for r in rows if r['candidate'] == 'star'),
        true_star_denominator=len(config['tasks']), failures=failures,
        grouping='network x history length x boundary; seeds are paired across length/boundary',
        candidate_ordering_rule=dict(all_four_must_qualify=True, absolute_SSE_tie_tolerance=TIE_SSE,
                                    global_optimality_certified=False),
        audit=dict(status='passed' if audit_ok else 'pending_or_failed', evidence=audit,
                   file=str(audit_path.resolve()) if audit is not None else None),
        computational_cost=dict(wall_seconds=(completion or {}).get('elapsed_seconds'),
            worker_count=config.get('workers'), fit_reported_seconds_sum=sum(r.get('seconds') or 0 for r in rows),
            task_seconds_sum=sum(t.get('seconds') or 0 for t in task_rows),
            fit_reported_ac_calls=sum(int(r.get('ac_calls') or 0) for r in rows),
            diagnostic_finite_direction_replay_calls_not_in_fit_counter=11*len(diagnostic_rows),
            completed_evaluation_only_replay_calls=sum(1+2*sum(
                r['key'] == t['key'] and r.get('loss') is not None for r in rows)
                for t in task_rows if t['status'] == 'completed'),
            note='fit seconds sum overlaps across workers; wall time is controller elapsed; diagnostics add AC replay cost'),
        diagnostic_maxima=dict(raw_SSE_recomputation_absolute=maximum(r.get('loss_recomputed_difference') for r in rows),
            finite_direction_residual_relative=maximum(d.get('residual_relative_error') for d in finite_checks),
            finite_direction_residual_absolute=maximum(d.get('residual_max_abs_error') for d in finite_checks),
            exact_star_branch_identity=maximum(d.get('branch_max_abs') for d in identities),
            exact_star_contrast_identity=maximum(d.get('contrast_max_abs') for d in identities)),
        information_scope=dict(local_metrics_use_only_measured_data=True, added_derived_likelihood_terms=0,
            R_adds_independent_meter=True, R_is_not_same_meter_budget_as_U=True,
            U_domain_unchanged=True, pairwise_voltage_differences_are_correlated=True,
            root_meter_removal_is_same_point_information_check=True,
            holdout_response_uses_oracle_PQ_and_junction=True,
            holdout_response_is_operational_forecast=False,
            U_true_root_response_is_additional_boundary_counterfactual=True,
            whole_feeder_reference_meter_information_not_used=True),
        groups=groups, paired_length_groups=paired_groups, paired_length_histories=paired,
        tasks=task_rows, candidate_rows=rows,
        source_hashes=config.get('source_manifest'), analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes=manifest)
    return summary


def _num(value, percentage=False):
    value = _finite(value)
    return '—' if value is None else f'{100*value:.2f}%' if percentage else f'{value:.5g}'


def markdown(summary):
    lines = ['# 三天/七天普通量测：保存结果汇总', '',
        '**本文件只读取封存结果，不重新拟合。**', '',
        f"{'Pilot，仅核查数据格式。' if summary['pilot'] else '主批：'} "
        f"{summary['completed_conditions']}/{summary['expected_conditions']}条件完成；"
        f"{summary['qualified_candidates']}/{summary['candidate_denominator']}候选拟合合格；"
        f"{summary['qualified_true_star']}/{summary['true_star_denominator']}真star拟合合格。"
        f"独立网络/种子族={summary['independent_random_families']}；长度与边界配对。", '',
        f"独立审计状态：{summary['audit']['status']}。输入文件哈希见summary.json。", '',
        '## 1. 分叉以下参数与全块参数分开报告', '',
        '每格为合格真star历史的均值；分母始终为该组预定历史数。U的共同主干存在不可识别等价性，U全块误差只作任意等价代表的诊断。', '',
        '|网络|点数|入口|合格/总数|末端R|末端X|末端β|全块R|全块X|活动边界历史|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in summary['groups']:
        metric = group['metrics']
        values = [_num(metric[f]['mean'], True) for f in ERROR_FIELDS[:5]]
        # ERROR_FIELDS order: leaf R/X, total R/X, beta; table puts beta first.
        lines.append(f"|{group['network']}|{group['N']}|{group['boundary']}|{group['qualified_true_star']}/{group['expected_histories']}|"
                     f"{values[0]}|{values[1]}|{values[4]}|{values[2]}|{values[3]}|{group['active_bound_true_star']}|")
    lines += ['', '## 2. 局部响应、额外入口与反事实', '',
        '下面全部响应使用封存后的真实P/Q与真实分叉电压，是评价诊断，不是规划时可用的预测精度。'
        '原入口回代在U中还提供了未计量的真实入口，必须单列为额外信息反事实。', '',
        '|网络|点数|入口|局部真分叉留出RMSE/σ|平方电压差RMSE/σ|原入口反事实RMSE/σ|共享秩计数|下游秩计数|',
        '|---|---:|---|---:|---:|---:|---|---|']
    for group in summary['groups']:
        m = group['metrics']
        lines.append(f"|{group['network']}|{group['N']}|{group['boundary']}|"
            f"{_num(m['local_true_PQ_true_junction_RMSE_sigma_holdout']['mean'])}|"
            f"{_num(m['contrast_RMSE_sigma_holdout']['mean'])}|"
            f"{_num(m['true_PQ_voltage_RMSE_sigma_holdout']['mean'])}|"
            f"{group['rank_counts']}|{group['downstream_rank_counts']}|")
    lines += ['', 'R为新增σ=.0005pu的入口电压表；K为理想已知入口，均不与U拥有相同计量预算。'
        'R在同一点去掉新增表计行后的秩计数：']
    for group in summary['groups']:
        if group['boundary'] == 'R':
            lines.append(f"- {group['network']} / {group['N']}：{group.get('without_added_meter_rank_counts', {})}。")
    lines += ['', '## 3. 七天减三天的配对差', '',
        '相对误差以百分点表示，负值表示改善；只纳入同族两次均合格的历史，分母保留。', '',
        '|网络|入口|可配对/总数|末端R差/百分点|末端X差/百分点|β差/百分点|局部响应RMSE差/σ|',
        '|---|---|---:|---:|---:|---:|---:|']
    for group in summary['paired_length_groups']:
        dif = group['differences']
        values = [_num(None if dif[f]['mean'] is None else 100*dif[f]['mean']) for f in
                  ('r_leaf_relative', 'x_leaf_relative', 'beta_leaf_relative')]
        lines.append(f"|{group['network']}|{group['boundary']}|{group['available']}/{group['denominator']}|"
                     f"{values[0]}|{values[1]}|{values[2]}|{_num(dif['local_true_PQ_true_junction_RMSE_sigma_holdout']['mean'])}|")
    lines += ['', '## 4. 候选排序、优化与费用', '',
        '只有四个候选均合格才给局部取得损失排序；SSE绝对差≤1e-6作为并列。排序没有全局最优保证，不作为拓扑正确概率。', '',
        '|网络|点数|入口|排序可用/总数|真star在最小并列组|真star唯一最小|',
        '|---|---:|---|---:|---:|---:|']
    for group in summary['groups']:
        lines.append(f"|{group['network']}|{group['N']}|{group['boundary']}|{group['ranking_available']}/{group['ranking_denominator']}|"
                     f"{group['true_star_in_best_tie_group']}|{group['unique_true_star_best']}|")
    cost = summary['computational_cost']
    lines += ['', '四初值按各自SSE、投影梯度、物理状态和停止原因保存。原记录没有每个初值的R/X，无法由这些记录推断参数离散程度。', '',
        f"主控制器墙钟={_num(cost['wall_seconds'])}秒；拟合报告AC调用={cost['fit_reported_ac_calls']}；"
        f"诊断另加{cost['diagnostic_finite_direction_replay_calls_not_in_fit_counter']}次AC回放。"
        '各拟合seconds相加包含并行重叠，不当作总墙钟。', '',
        '最大诊断差：`'+json.dumps(summary['diagnostic_maxima'], ensure_ascii=False)+'`。', '',
        '完整失败、缺失、活动边界、逐历史结果和四初值记录在summary.json及candidate_details.csv。'
        '平方电压差、信息矩阵和P/Q激励指标都是派生诊断；没有重复计入似然。', '']
    return '\n'.join(lines)


def write_outputs(directory, summary):
    directory = Path(directory)
    (directory/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    fields = ['network', 'seed', 'N', 'boundary', 'candidate', 'task_status', 'status', 'qualified',
        'loss', 'J', 'SSE_per_raw_random_meter', 'projected_gradient', 'profile_rank', 'profile_width',
        'active_line_coordinate_count', 'active_line_coordinates', 'interior_CRLB_interpretation_applicable',
        'seconds', 'ac_calls', 'attempts_count', 'attempts_qualified', 'attempts_loss_range',
        'attempts_parameter_spread', *ERROR_FIELDS, 'r_leaf_each', 'x_leaf_each',
        'evaluation_only', 'local_response_uses_true_PQ_and_true_junction',
        'oracle_true_root_is_additional_information_for_U', 'loss_recomputed_difference', 'attempts']
    with (directory/'candidate_details.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in summary['candidate_rows']:
            writer.writerow({key:json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list))
                             else row.get(key) for key in fields})
    (directory/'results_generated.md').write_text(markdown(summary), encoding='utf-8')


def write_gradient_scales(directory):
    """Receiving-end constraint partials at fitted leaf states, without truth."""
    rows=[];groups=defaultdict(list);manifest={}
    for path in sorted((Path(directory)/'inference').glob('*.json.gz')):
        value=_read(path,manifest);fitted=value['fits'].get('star',{})
        if any(name not in fitted for name in ('r','x','p','q','v')):continue
        r,x=(np.asarray(fitted[name])[1:] for name in ('r','x'))
        p,q,v=(np.asarray(fitted[name]) for name in ('p','q','v'))
        ell=(p*p+q*q)/(v*v)
        gp,gq=2*(p+r*ell),2*(q+x*ell)
        row=dict(spec=value['spec'],
            R_I2_derivative_fraction=(np.linalg.norm(r*ell,axis=0)/np.linalg.norm(p,axis=0)).tolist(),
            X_I2_derivative_fraction=(np.linalg.norm(x*ell,axis=0)/np.linalg.norm(q,axis=0)).tolist(),
            R_X_local_constraint_column_cosine=(np.sum(gp*gq,axis=0)/
                (np.linalg.norm(gp,axis=0)*np.linalg.norm(gq,axis=0))).tolist())
        rows.append(row)
        spec=value['spec'];groups[(spec['network'],spec['N'],spec['boundary'])].append(row)
    summaries=[]
    for key,values in groups.items():
        row=dict(network=key[0],N=key[1],boundary=key[2],histories=len(values))
        for field in ('R_I2_derivative_fraction','X_I2_derivative_fraction','R_X_local_constraint_column_cosine'):
            data=np.asarray([v[field] for v in values]).ravel()
            row[field]=dict(mean=float(data.mean()),minimum=float(data.min()),maximum=float(data.max()))
        summaries.append(row)
    result=dict(truth_used=False,additional_fitting=False,
        formula='ell=(p^2+q^2)/v^2; norm(r*ell)/norm(p); norm(x*ell)/norm(q); partial constraint derivative cosine, not total dv/dtheta',
        input_sha256=manifest,rows=rows,groups=summaries)
    (Path(directory)/'local_gradient_scale.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    summary = summarize(args.directory)
    write_outputs(args.directory, summary)
    write_gradient_scales(args.directory)
    status={key:summary[key] for key in ('completed_conditions','expected_conditions',
        'qualified_candidates','candidate_denominator','independent_random_families')}
    status['audit']=summary['audit']['status']
    print(json.dumps(status,ensure_ascii=False))


if __name__ == '__main__':
    main()
