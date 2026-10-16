"""Offline summaries of frozen Bayesian EV records and their independent audit.

Reads saved arrays only. No AC solving, fitting, planning or data generation.
Run with --self-test for arithmetic and exact-history-coverage checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np


METHODS = ('constant', 'square__fixed_charge', 'square__mobile_charge',
           'square__mobile_bidir', 'bayes__mobile_charge', 'bayes__mobile_bidir')
LABELS = dict(zip(METHODS, ('恒充', '平方 F+', '平方 M+', '平方 M±', 'Bayes M+', 'Bayes M±')))
ATTR_METHODS = ('constant', 'square__mobile_bidir', 'bayes__mobile_bidir')
REGIMES = ('regular', 'shift')
FACTORS = ('lines', 'PQ', 'source')
VOLTAGE_BASE_V = 400.  # Declared balanced LV nominal line-to-line voltage.
POWER_BASE_KW = 20.


def _read(path):
    path = Path(path); payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)


def _serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    return value


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def _identity(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def exact_coverage(expected, present):
    target = [_identity(v) for v in expected]; actual = [_identity(v) for v in present]
    counts = Counter(actual)
    return dict(exact=len(actual) == len(set(actual)) and set(actual) == set(target),
        expected_count=len(target), present_count=len(actual),
        missing=[v for v in expected if _identity(v) not in counts],
        unexpected=[v for v in present if _identity(v) not in target],
        duplicates=[json.loads(k) for k, count in counts.items() if count > 1])


def _average(values, expected_count):
    values = [float(v) for v in values if v is not None]
    _require(np.isfinite(values).all(), 'finite aggregation values required')
    return dict(mean=float(np.mean(values)) if values else None,
        available_count=len(values), expected_count=expected_count)


def _rmse(prediction, truth, scale, units):
    p, t = np.asarray(prediction, float)*scale, np.asarray(truth, float)*scale
    _require(p.shape == t.shape and np.isfinite(p).all() and np.isfinite(t).all(), 'forecast/truth shape')
    error = float(np.sqrt(np.mean((p-t)**2)))
    rms = float(np.sqrt(np.mean(t*t)))
    return dict(RMSE=error, units=units, relative_RMSE=error/rms if rms > 0 else None,
                truth_RMS=rms)


def operation_summary(method, config):
    decoded, saved = method['decoded'], method['operation']
    charge, discharge = np.asarray(decoded['charge_kw'], float), np.asarray(decoded['discharge_kw'], float)
    dt = float(config['dt_hours']); charge_energy = float(charge.sum()*dt); discharge_energy = float(discharge.sum()*dt)
    arcs = decoded['selected_arcs']; moves = sum(a['kind'] == 'travel' for a in arcs)
    driving = float(sum(a['driving_energy_kwh'] for a in arcs))
    start = float(config['initial_soc']*config['battery_capacity_kwh'])
    terminal = start+float(config['eta_charge'])*charge_energy-discharge_energy/float(config['eta_discharge'])-driving
    observed_terminal = float(saved['energy_kwh'][-1])
    _require(moves == int(saved['moves']), 'saved movement count differs from selected travel arcs')
    _require(abs(terminal-observed_terminal) < 2e-6, 'charging/discharging/drive energy identity')
    _require(abs(charge_energy+discharge_energy-float(saved['throughput_kwh'])) < 2e-6, 'grid throughput identity')
    _require(abs(charge_energy-discharge_energy-float(saved['grid_energy_kwh'])) < 2e-6, 'net grid energy identity')
    _require(abs(terminal-float(config['terminal_energy_kwh'])) < 2e-6, 'common terminal battery task')
    return dict(moves=int(moves), moved=moves > 0, movement_permitted=method['domain'].startswith('mobile'),
        charge_energy_kwh=charge_energy, discharge_energy_kwh=discharge_energy,
        net_grid_energy_kwh=charge_energy-discharge_energy, grid_throughput_kwh=charge_energy+discharge_energy,
        driving_energy_kwh=driving, terminal_energy_kwh=observed_terminal,
        requested_terminal_energy_kwh=float(config['terminal_energy_kwh']),
        terminal_energy_identity_error_kwh=abs(terminal-observed_terminal),
        discharged=discharge_energy > 1e-6, driver_cost=float(saved['driver_cost']),
        cost_units=saved.get('cost_units'), feasible=bool(saved['feasible']))


def _line_errors(trained, truth):
    result = {}
    for key in ('r', 'x'):
        fitted, target = np.asarray(trained[key], float), np.asarray(truth[key], float)
        _require(fitted.shape == target.shape, 'true-candidate edge ordering and dimensions')
        result[key.upper()] = dict(relative_L2=float(np.linalg.norm(fitted-target)/np.linalg.norm(target)),
            fitted_pu=fitted, true_pu=target, per_edge_relative_error=(fitted-target)/target)
    return result


def _aggregate_attribution(rows, histories):
    result = []
    for method in ATTR_METHODS:
        for regime in REGIMES:
            for channel in ('v', 'master_p', 'master_q'):
                selected = [r for r in rows if (r['method'], r['regime'], r['channel']) == (method, regime, channel)]
                mean_mse = _average([r['nominal_MSE'] for r in selected], histories)
                contributions = {f: _average([r['contributions'][f] for r in selected], histories)['mean'] for f in FACTORS}
                scale = VOLTAGE_BASE_V**2 if channel == 'v' else POWER_BASE_KW**2
                unit = 'V_LL^2' if channel == 'v' else 'kW^2' if channel == 'master_p' else 'kvar^2'
                denominator = mean_mse['mean']
                result.append(dict(method=method, regime=regime, channel=channel,
                    histories_available=len(selected), histories_expected=histories,
                    average_nominal_MSE_pu_squared=denominator,
                    average_nominal_MSE_physical=None if denominator is None else denominator*scale,
                    physical_units=unit,
                    average_contributions_pu_squared=contributions,
                    average_contributions_physical={f: None if v is None else v*scale for f, v in contributions.items()},
                    fraction_of_average_nominal_MSE={f: None if denominator is None or denominator <= 0 else v/denominator for f, v in contributions.items()},
                    identity_error=None if denominator is None else abs(sum(contributions.values())-denominator)))
    return result


def analyze(directory, audit_path):
    directory, audit_path = Path(directory), Path(audit_path)
    freeze = _read(directory/'freeze.json'); audit = _read(audit_path)
    _require(audit['policy_id'] == freeze['policy_id'], 'audit/freeze policy identity')
    expected = freeze['specifications']['evaluation']; total = len(expected)
    _require(total == 8, 'this frozen report expects exactly eight predeclared histories')
    audited = {int(r['spec']['index']): r for r in audit['records']}
    present, histories, methods, operations, allocations, legacy, issues = [], [], [], [], [], [], []
    inputs = []
    for path in sorted((directory/'evaluation').glob('history_*.json.gz')):
        record = _read(path); spec = record['spec']; index = int(spec['index']); present.append(spec)
        raw_hash = sha256(path.read_bytes()).hexdigest()
        inputs.append(dict(file=str(path.resolve()), sha256=raw_hash))
        _require(record['policy_id'] == freeze['policy_id'], 'raw/freeze policy identity')
        review = audited.get(index)
        _require(review is not None and review['input_sha256'] == raw_hash, 'audit/raw identity '+str(index))
        if record.get('status') != 'complete':
            histories.append(dict(index=index, spec=spec, status=record.get('status'), failure=record.get('failure')))
            issues.append(dict(index=index, reason='recorded incomplete history retained'))
            continue
        true_name = record['true_name']; true_index = record['candidate_names'].index(true_name)
        trained = next(r for r in record['training_result']['fits'] if r['candidate'] == true_name)
        prediction = record['prediction']; latent = np.asarray(prediction['candidate_updates'][true_name]['future_latent_mean'], float)
        base = float(prediction['base_kw']); n = len(record['observed']['p'][0])
        _require(base == POWER_BASE_KW, 'declared physical power base')
        forecast_errors = {}
        for regime in REGIMES:
            true = record['true_future_inputs_evaluation_only'][regime]
            forecast_errors[regime] = dict(
                p=_rmse(latent[:, :n], true['p'], base, 'kW'),
                q=_rmse(latent[:, n:2*n], true['q'], base, 'kvar'),
                source_p=_rmse(latent[:, -2], true['source_p'], base, 'kW'),
                source_q=_rmse(latent[:, -1], true['source_q'], base, 'kvar'))
        histories.append(dict(index=index, spec=spec, status=record['status'], true_name=true_name,
            trained_true_candidate_line_errors=_line_errors(trained, record['true_lines']),
            prior_truth_mass=float(prediction['prior'][true_index]), prior_truth_log_mass=float(prediction['log_prior'][true_index]),
            true_candidate_forecast_errors=forecast_errors, independent_audit_success=bool(review['success'])))
        for method in METHODS:
            if method not in record['methods']:
                issues.append(dict(index=index, method=method, reason='predeclared method missing')); continue
            saved = record['methods'][method]; key = saved['action_key']
            action = record['actions'][key]
            calibration = review['actions'][key]['independent']['declared_model_internal_calibration']
            operation = operation_summary(saved, record['operation_config'])
            operations.append(dict(index=index, true_name=true_name, method=method, action_key=key, **operation))
            for regime in REGIMES:
                actual = action['actual'][regime]
                _require(int(actual['n_independent_histories']) == 1 and int(actual['n_noise']) == 32, 'actual conditional-noise denominator')
                selected = [int(np.argmax(r['posterior'])) for r in actual['samples']]
                counts = Counter(record['candidate_names'][j] for j in selected)
                methods.append(dict(index=index, true_name=true_name, method=method, regime=regime, action_key=key,
                    accuracy=float(actual['correct']), confidence=float(actual['confidence']),
                    confidence_minus_accuracy=float(actual['confidence']-actual['correct']),
                    true_probability=float(actual['true_probability']),
                    credible_90_true_excluded=float(actual['true_rejected']),
                    wrong_unique=float(actual['wrong_unique']), unresolved=float(actual['unresolved']),
                    mean_credible_size=float(actual['credible_size']), brier=float(actual['brier']),
                    n_conditional_meter_draws=32, n_independent_histories=1,
                    selected_label_counts=dict(counts), planning_risk=float(saved['planning_score']['risk']),
                    independent_model_risk=float(saved['independent_score']['risk']),
                    model_internal_accuracy=1-float(calibration['MAP_error']),
                    model_internal_confidence=float(calibration['confidence']),
                    model_internal_credible_90_excluded=float(calibration['true_rejected']),
                    model_internal_confidence_minus_accuracy=float(calibration['confidence_minus_correct']),
                    model_internal_confidence_bins=calibration['confidence_bins'],
                    model_internal_samples_per_candidate=int(calibration['sample_count_per_candidate']),
                    operation_moves=operation['moves'], grid_discharge_kwh=operation['discharge_energy_kwh']))
        for method in ATTR_METHODS:
            if method not in record['methods']:
                continue
            key = record['methods'][method]['action_key']
            for regime in REGIMES:
                identifier = regime+'__'+key
                if identifier not in review.get('mean_error_attribution', {}):
                    issues.append(dict(index=index, method=method, regime=regime, reason='audited 2^3 attribution missing')); continue
                attribution = review['mean_error_attribution'][identifier]
                _require(method in attribution['methods'], 'attribution method/action identity')
                bank = record['actions'][key]['forecast']['bank']
                covariance = np.asarray(bank['covariances'], float)[0, true_index]
                sigma_v = np.broadcast_to(np.asarray(record['future_meter']['v_sigma'], float), (len(latent), n)).ravel()
                voltage_variance = np.diag(covariance)[2*len(latent)*n:3*len(latent)*n]*sigma_v**2
                for channel in ('v', 'master_p', 'master_q'):
                    nominal = float(attribution['mean_MSE_by_mask'][channel]['000'])
                    values = {f: float(attribution['allocation'][channel]['contributions'][f]) for f in FACTORS}
                    allocations.append(dict(index=index, true_name=true_name, method=method, regime=regime, channel=channel,
                        action_key=key, nominal_MSE=nominal,
                        MSE_by_mask=attribution['mean_MSE_by_mask'][channel], contributions=values,
                        fraction_of_nominal_MSE={f: v/nominal if nominal > 0 else None for f, v in values.items()},
                        voltage_RMSE_over_rms_meter_sigma=float(np.sqrt(nominal/np.mean(sigma_v**2))) if channel == 'v' else None,
                        voltage_RMSE_over_rms_predictive_sd=float(np.sqrt(nominal/np.mean(voltage_variance))) if channel == 'v' else None))
                identifier = regime+'__'+key; old = record.get('legacy_joint_audit', {}).get(identifier)
                if old is None:
                    issues.append(dict(index=index, method=method, regime=regime, reason='legacy comparison missing'))
                    continue
                row = dict(index=index, true_name=true_name, method=method, regime=regime, action_key=key,
                    complete=bool(old.get('complete')), global_optimality_certified=False,
                    scope='local complete AC joint line/source refit; measured PQ inputs; same observed+future data')
                if old.get('complete'):
                    loss = old['losses']; selected = min(loss, key=loss.get)
                    nearest_wrong = min((c for c in loss if c != true_name), key=loss.get)
                    M = float(loss[nearest_wrong]-loss[true_name])
                    _require(abs(M-float(old['M'])) < 1e-8 and selected == old['selected'], 'legacy M/selected identity')
                    row.update(M=M, selected=selected, selected_true=selected == true_name,
                        nearest_wrong=nearest_wrong, true_joint_SSE=float(loss[true_name]), nearest_wrong_joint_SSE=float(loss[nearest_wrong]))
                else:
                    row['failures'] = old.get('failures', {})
                legacy.append(row)
    coverage = exact_coverage(expected, present)
    averages, operation_averages = {}, {}
    for method in METHODS:
        selected = [r for r in methods if r['method'] == method]
        internal = [r for r in selected if r['regime'] == 'regular']  # One planned distribution per history.
        averages[method] = dict(
            declared_Gaussian={field.removeprefix('model_internal_'): _average([r[field] for r in internal], total)
                for field in ('model_internal_accuracy', 'model_internal_confidence', 'model_internal_credible_90_excluded', 'model_internal_confidence_minus_accuracy')},
            actual_simulation={regime: {field: _average([r[field] for r in selected if r['regime'] == regime], total)
                for field in ('accuracy', 'confidence', 'confidence_minus_accuracy', 'credible_90_true_excluded', 'wrong_unique', 'unresolved', 'brier')}
                for regime in REGIMES})
        ops = [r for r in operations if r['method'] == method]
        operation_averages[method] = dict(history_count=len(ops), expected_histories=total,
            moving_plans=sum(r['moved'] for r in ops), no_move_plans=sum(not r['moved'] for r in ops),
            plans_with_discharge=sum(r['discharged'] for r in ops),
            averages={field: _average([r[field] for r in ops], total)['mean'] for field in (
                'moves', 'charge_energy_kwh', 'discharge_energy_kwh', 'net_grid_energy_kwh',
                'grid_throughput_kwh', 'driving_energy_kwh', 'terminal_energy_kwh', 'driver_cost')})
    focus = []
    for index in (0, 4):
        history = next((r for r in histories if r['index'] == index), None)
        focus.append(dict(index=index, history=history,
            regular_voltage_allocations=[r for r in allocations if r['index'] == index and r['regime'] == 'regular' and r['channel'] == 'v'],
            actual_results=[r for r in methods if r['index'] == index], legacy=[r for r in legacy if r['index'] == index]))
    return dict(schema='ev_bayes_small_offline_diagnostics_v1', success=coverage['exact'] and bool(audit['success']) and not issues,
        policy_id=freeze['policy_id'], coverage=coverage, issues=issues, histories=histories, method_results=methods,
        averages=averages, operations=operations, operation_averages=operation_averages,
        mean_MSE_attribution=allocations, mean_MSE_attribution_averages=_aggregate_attribution(allocations, total),
        legacy_comparisons=legacy, focus_star_histories=focus,
        counts=dict(independent_histories=total, completed_histories=sum(r['status'] == 'complete' for r in histories),
            predefined_methods=len(METHODS), method_regime_rows=len(methods), operation_rows=len(operations),
            legacy_method_regime_rows=len(legacy), attribution_channel_rows=len(allocations)),
        distribution_definitions=dict(
            declared_Gaussian='per-history candidate mixture weighted by the saved evidence posterior; random latent process and new meters; 2048 integration samples per candidate',
            actual_simulation='eight predeclared independent histories, two of each truth structure; one fixed latent trajectory per regime and 32 paired new meter-noise draws',
            averaging='equal history weight; regular/shift share history and plan; 32 draws are conditional replicates, not new histories'),
        attribution_definition=dict(
            target='clean complete AC mean MSE at the same saved action',
            factors={'lines': 'replace trained R/X by true R/X in the true candidate',
                'PQ': 'replace predicted ordinary true load P/Q by the saved true ordinary load',
                'source': 'replace predicted same-interface source P/Q by the saved true source'},
            rule='average marginal MSE reductions over the six factor orders; signed contributions sum to nominal MSE minus all-truth MSE',
            interpretation='mean prediction-error allocation; excludes covariance and topology-decision-risk attribution',
            units=dict(voltage_base_V_LL=VOLTAGE_BASE_V, power_base_kW=POWER_BASE_KW,
                voltage='pu^2 and V_LL^2', master_p='pu^2 and kW^2', master_q='pu^2 and kvar^2')),
        input_evidence=dict(root=str(directory.resolve()), raw_files=inputs, audit_file=str(audit_path.resolve()),
            audit_sha256=sha256(audit_path.read_bytes()).hexdigest(), audit_tool_sha256=audit['tool_sha256'],
            freeze_sha256=sha256((directory/'freeze.json').read_bytes()).hexdigest()),
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), new_fits=0, new_plans=0,
        new_AC_calls=0, new_simulations=0, synthetic_truth_used_posthoc_only=True)


def _fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}g}'


def _percent(value):
    return '—' if value is None else f'{100*value:.1f}%'


def render_markdown(result):
    lines = ['# 冻结贝叶斯 EV 算例：离线诊断', '',
        f"本报告读取冻结的 {result['coverage']['present_count']}/{result['coverage']['expected_count']} 条评价历史和独立审计；新增拟合、规划、潮流计算与仿真均为0。",
        '每条历史包含32次配对计量噪声；regular与shift使用同一历史和计划。平均值按8条历史等权计算。', '',
        '## 固定训练参数与逐历史表现', '',
        'R、X误差为真候选各边参数误差的二范数除以真参数二范数。π真为20点证据后的真候选权重。', '',
        '|历史|真结构|R误差|X误差|π真|', '|---:|---|---:|---:|---:|']
    for row in result['histories']:
        e = row.get('trained_true_candidate_line_errors', {})
        lines.append(f"|{row['index']}|{row.get('true_name', row['status'])}|{_percent(e.get('R',{}).get('relative_L2'))}|{_percent(e.get('X',{}).get('relative_L2'))}|{_fmt(row.get('prior_truth_mass'))}|")
    lines += ['', '下表各格为“MAP正确率 / 平均最大后验 / 90%最小后验集合排真率”，均按本历史的32次计量噪声计算。', '',
        '|历史/结构/情景|'+'|'.join(LABELS[m] for m in METHODS)+'|', '|---|'+'|'.join(['---:']*len(METHODS))+'|']
    for history in result['histories']:
        for regime in REGIMES:
            cells = []
            for method in METHODS:
                row = next((r for r in result['method_results'] if r['index'] == history['index'] and r['regime'] == regime and r['method'] == method), None)
                cells.append('缺失' if row is None else ' / '.join(_percent(row[k]) for k in ('accuracy', 'confidence', 'credible_90_true_excluded')))
            lines.append(f"|{history['index']} / {history.get('true_name','失败')} / {regime}|"+'|'.join(cells)+'|')
    lines += ['', '## 声明模型内部检查与物理仿真', '',
        '内部检查从各历史保存的联合Gaussian候选混合分布抽样，并按证据后候选权重积分。物理仿真固定每条历史的真实结构与未来轨迹，再改变计量噪声。两列对应不同分布；内部归一化和校准通过后，仍需观察物理数据中的预测误差。', '',
        '|方法|Gaussian内部：正确/置信/排真|regular：正确/置信/排真|shift：正确/置信/排真|', '|---|---:|---:|---:|']
    for method, row in result['averages'].items():
        inside = row['declared_Gaussian']; inside_cell = ' / '.join(_percent(inside[k]['mean']) for k in ('accuracy', 'confidence', 'credible_90_excluded'))
        outside = [' / '.join(_percent(row['actual_simulation'][regime][k]['mean']) for k in ('accuracy', 'confidence', 'credible_90_true_excluded')) for regime in REGIMES]
        lines.append('|'+LABELS[method]+'|'+inside_cell+'|'+'|'.join(outside)+'|')
    lines += ['', '逐动作的confidence分箱与积分误差保存在独立审计JSON；本诊断JSON保留六方法逐历史的分箱。', '',
        '## 普通负荷情景下，两条star历史为何失配', '',
        '下表使用真候选在证据条件化后的负荷和源预测，RMSE单位依次为kW、kvar、kW、kvar。', '',
        '|历史|P RMSE|Q RMSE|源P RMSE|源Q RMSE|', '|---:|---:|---:|---:|---:|']
    for focus in result['focus_star_histories']:
        history = focus['history']; errors = history['true_candidate_forecast_errors']['regular']
        lines.append(f"|{focus['index']}|"+'|'.join(_fmt(errors[k]['RMSE']) for k in ('p', 'q', 'source_p', 'source_q'))+'|')
    lines += ['', '|历史/计划|电压均值RMSE（V，线电压基准）|线路分摊|负荷分摊|源分摊|RMSE/预测标准差均方根|', '|---|---:|---:|---:|---:|---:|']
    for focus in result['focus_star_histories']:
        for row in focus['regular_voltage_allocations']:
            fractions = row['fraction_of_nominal_MSE']
            lines.append(f"|{focus['index']} / {LABELS[row['method']]}|{_fmt(np.sqrt(row['nominal_MSE'])*VOLTAGE_BASE_V)}|"+
                '|'.join(_percent(fractions[f]) for f in FACTORS)+'|'+_fmt(row['voltage_RMSE_over_rms_predictive_sd'])+'|')
    fractions = [r['fraction_of_nominal_MSE']['lines'] for f in result['focus_star_histories'] for r in f['regular_voltage_allocations']]
    lines += ['', f"两条star历史的预定三计划中，线路占regular电压均值MSE的{_percent(min(fractions))}—{_percent(max(fractions))}。普通负荷与源的逐时预测误差较小，固定训练线路是本组电压均值失配的主要来源。",
        '该分摊逐次替换线路、普通P/Q和源的真值，并对六种替换顺序取平均。它解释同一动作下的均值MSE；实际排错还取决于竞争候选的响应和预测协方差。', '',
        '## 三因素均值误差分摊', '',
        '下表给出各历史等权平均后的物理单位MSE与分摊。线路、普通P/Q、源的有符号分摊之和等于原MSE减去全真值MSE；负值表示替换后暴露了原先互相抵消的误差。电压采用400 V线电压基准，总有功/无功采用20 kW/kvar基准。', '',
        '|方法/情景/通道|均值MSE（物理单位²）|线路|普通P/Q|源|单位|', '|---|---:|---:|---:|---:|---|']
    for row in result['mean_MSE_attribution_averages']:
        lines.append(f"|{LABELS[row['method']]} / {row['regime']} / {row['channel']}|{_fmt(row['average_nominal_MSE_physical'])}|"+
            '|'.join(_fmt(row['average_contributions_physical'][f]) for f in FACTORS)+'|'+row['physical_units']+'|')
    lines += ['', '## EV实际使用的自由度', '',
        'M表示允许移动；优化器可以选择停留。移动方案计数按8条历史计算一次，regular/shift共用计划。充放电、驾驶能量和电池终值由保存的完整操作重算。', '',
        '|方法|移动/8|未移动/8|放电计划/8|充电kWh|放电kWh|净取电kWh|吞吐kWh|驾驶kWh|终值kWh|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, row in result['operation_averages'].items():
        avg = row['averages']
        lines.append(f"|{LABELS[method]}|{row['moving_plans']}|{row['no_move_plans']}|{row['plans_with_discharge']}|"+
            '|'.join(_fmt(avg[f]) for f in ('charge_energy_kwh', 'discharge_energy_kwh', 'net_grid_energy_kwh', 'grid_throughput_kwh', 'driving_energy_kwh', 'terminal_energy_kwh'))+'|')
    lines += ['', '## 旧完整AC联合拟合对照', '',
        'M=最小错误候选SSE−真候选SSE；M>0表示真候选在本次达到的局部拟合中损失更小。此处使用直接测得的P/Q，并允许线路与源同时解释历史和新增数据。', '',
        '|历史/真结构/情景|恒充 M/选中|平方 M± M/选中|Bayes M± M/选中|', '|---|---:|---:|---:|']
    for history in result['histories']:
        for regime in REGIMES:
            cells = []
            for method in ATTR_METHODS:
                row = next((r for r in result['legacy_comparisons'] if (r['index'], r['regime'], r['method']) == (history['index'], regime, method)), None)
                cells.append('缺失' if row is None else '拟合未完成' if not row['complete'] else _fmt(row['M'])+' / '+row['selected'])
            lines.append(f"|{history['index']} / {history.get('true_name','失败')} / {regime}|"+'|'.join(cells)+'|')
    lines += ['', '概率可靠度的主要改进方向是处理固定训练R/X造成的预测偏差，再检验规划指标。旧联合拟合在历史4恢复star、历史0仍偏向错误候选，说明两条失败历史的信息条件也有差异。', '',
        f"证据：独立审计成功={result['success']}；完整性={result['coverage']['present_count']}/{result['coverage']['expected_count']}；缺失/身份问题={len(result['issues'])}。所有逐历史结果、操作能量、来源哈希及分摊公式见 diagnostics.json。", '']
    return '\n'.join(lines)


def self_test():
    expected = [dict(index=0), dict(index=1)]
    assert exact_coverage(expected, list(reversed(expected)))['exact']
    assert not exact_coverage(expected, [expected[0]])['exact']
    assert not exact_coverage(expected, [expected[0], expected[0]])['exact']
    assert not exact_coverage(expected, expected+[dict(index=2)])['exact']
    assert _average([0., 1.], 2) == dict(mean=.5, available_count=2, expected_count=2)
    assert _average([None], 2) == dict(mean=None, available_count=0, expected_count=2)
    rows = [dict(method='constant', regime='regular', channel='v', nominal_MSE=m,
        contributions=dict(lines=line, PQ=pq, source=source)) for m, line, pq, source in ((10., 7., 2., 1.), (30., 9., 10., 11.))]
    value = _aggregate_attribution(rows, 2)[0]
    assert value['average_nominal_MSE_pu_squared'] == 20.
    assert value['average_contributions_pu_squared'] == dict(lines=8., PQ=6., source=6.)
    assert value['fraction_of_average_nominal_MSE']['lines'] == .4
    assert value['identity_error'] == 0.
    config = dict(dt_hours=.25, initial_soc=.4, battery_capacity_kwh=60., eta_charge=.95,
                  eta_discharge=.95, terminal_energy_kwh=24+.95*.5-.125/.95-.2)
    method = dict(domain='mobile_bidir', decoded=dict(charge_kw=[[2.]], discharge_kw=[[.5]],
        selected_arcs=[dict(kind='travel', driving_energy_kwh=.2)]), operation=dict(
        moves=1, energy_kwh=[24., config['terminal_energy_kwh']], throughput_kwh=.625,
        grid_energy_kwh=.375, driver_cost=30., cost_units='worker_minutes', feasible=True))
    value = operation_summary(method, config)
    assert value['moved'] and value['discharged'] and value['charge_energy_kwh'] == .5
    assert value['discharge_energy_kwh'] == .125 and value['terminal_energy_identity_error_kwh'] == 0.
    print('SELF_TEST_PASS exact coverage, missing/duplicate histories, equal-history means, MSE allocation/units, optional movement and energy identity', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input'); parser.add_argument('--audit'); parser.add_argument('--output')
    parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.input is None:
            return
    if args.input is None:
        parser.error('--input or --self-test required')
    began = perf_counter(); root = Path(args.input)
    audit = Path(args.audit) if args.audit else root/'independent_audit_evaluation.json'
    output = Path(args.output) if args.output else root/'report'
    json_path, md_path = output/'diagnostics.json', output/'diagnostics.md'
    _require(not json_path.exists() and not md_path.exists(), 'preserve existing diagnostic evidence; use a fresh output directory')
    result = analyze(root, audit); result['seconds'] = perf_counter()-began
    output.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_serial(result), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    md_path.write_text(render_markdown(result), encoding='utf-8')
    print('DIAGNOSTICS_SAVED', result['success'], result['counts'], flush=True)


if __name__ == '__main__':
    main()
