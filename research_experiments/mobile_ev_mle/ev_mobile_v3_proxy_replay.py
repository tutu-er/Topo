"""Offline shared-line tangent scores for 11 prespecified saved EV actions.

No training, source fit, EV optimization, simulation or future-data-dependent
expansion is performed. The expansion action is always saved scalar_fixed.
The extra 52-record route audit only decodes saved operation vectors.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import traceback
from types import SimpleNamespace

import numpy as np

from . import ev_fixed_train_design as fixed
from . import ev_mobile_v2_study as frozen
from . import ev_mobile_v3_ac as ac
from . import ev_mobile_v3_projection as projected
from . import ev_mobile_cases_v2 as newcases
from . import ev_systematic_cases as oldcases
from . import gradient_price_study as io
from . import likelihood as lk
from .information_context import meter_from_record

INDICES = (0, 12, 24, 36)
EXTRA = (('n3', 9), ('n3', 20), ('n4', 14))
METHODS = ('scalar_fixed', 'scalar_mobile')


def fingerprint(value):
    return sha256(json.dumps(io.serial(value), sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def signs(predicted_delta, actual_delta, *, tolerance=1e-8):
    """Keep ties and both directions, rather than only a favourable reversal."""
    sign = lambda v: 0 if abs(v) <= tolerance else (1 if v > 0 else -1)
    p, a = sign(float(predicted_delta)), sign(float(actual_delta))
    category = ('both_tied' if p == a == 0 else 'predicted_tie' if p == 0
                else 'actual_tie' if a == 0 else 'same_order' if p == a else 'reversed')
    return dict(predicted_sign=p, actual_sign=a, category=category,
                predicted_rise_actual_fall=p > 0 and a < 0,
                predicted_fall_actual_rise=p < 0 and a > 0, tolerance=tolerance)


def saved_source_score(raw, method):
    """Read a complete saved score for exactly the actual selected power."""
    actual = next(m for m in raw['methods'] if m['name'] == method)
    u = np.asarray(actual['power_kw'], float)
    row = raw['selection']['methods'][method]['raw']
    choices = [('selected_profile', row.get('selected_profile'), row.get('selected_score'),
                row.get('selected_aggregation'))]
    choices.extend((f'initial_candidates[{i}].profile', r.get('profile'), r.get('score'),
                    r.get('aggregation')) for i, r in enumerate(row.get('initial_candidates', [])))
    choices.extend((f'grids[{i}].raw_AC_profile', r.get('raw_AC_profile'), r.get('raw_AC_score'),
                    r.get('raw_AC_aggregation')) for i, r in enumerate(row.get('grids', [])))
    names = [c['name'] for c in raw['candidates']]
    expected = {(a, b) for a in names for b in names if a != b}
    for path, profile, score, aggregation in choices:
        if not profile or not profile.get('success') or score is None:
            continue
        power = np.asarray(profile.get('power_kw'), float)
        if power.shape != u.shape or not np.array_equal(power, u):
            continue
        pairs = profile.get('pairs', [])
        if len(pairs) != len(expected) or {(p.get('a'), p.get('b')) for p in pairs} != expected:
            continue
        if not all(p.get('success') for p in pairs) or not np.isfinite(score):
            continue
        nearest = [] if aggregation is None else aggregation.get('nearest', [])
        return dict(score=float(score), path=path, complete_pair_count=len(pairs),
                    nearest=nearest, same_power_exact=True, units='half standardized SSE')
    raise ValueError('No complete saved fixed-line source score for exact selected action: ' + method)


def restore_inputs(raw):
    """Restore training/planning primitives, accepting no future measurements."""
    training = lk.Observations(**raw['training'])
    bank = tuple(lk.RadialCandidate.from_supports(c['name'], training.n, c['supports'])
                 for c in raw['candidates'])
    if any(tuple(map(tuple, c['supports'])) != b.supports for c, b in zip(raw['candidates'], bank)):
        raise ValueError('Canonical candidate ordering differs from saved R/X ordering')
    fits = {}
    array_fields = ('r', 'x', 'source_p', 'source_q', 'p', 'q', 'bias_p', 'bias_q', 'bias_v')
    for row in raw['training_result']['fits']:
        value = dict(row)
        for key in array_fields:
            if value.get(key) is not None:
                value[key] = np.asarray(value[key], float)
        fits[row.get('candidate', row.get('name'))] = SimpleNamespace(**value)
    if set(fits) != {b.name for b in bank} or not all(f.success for f in fits.values()):
        raise ValueError('Every candidate needs its complete saved training fit')
    return dict(training=training, bank=bank, fits=fits,
                tm=meter_from_record(raw['training_meter']),
                fm=meter_from_record(raw['future_meter']), forecast=raw['forecast'])


def replay_record(raw, *, expected_sources=None):
    began = perf_counter()
    row = {k: raw[k] for k in ('physical_bank', 'index', 'seed', 'case_id')}
    row.update(success=False, parameter_fit_calls=0, source_fit_calls=0, MILP_calls=0,
               case_generation_calls=0)
    try:
        if expected_sources is not None and raw['source_sha256'] != expected_sources:
            raise ValueError('Frozen scientific source mismatch')
        inputs = restore_inputs(raw)
        methods = {m['name']: m for m in raw['methods']}
        uF, uM = (np.asarray(methods[name]['power_kw'], float) for name in METHODS)
        if uF.shape != uM.shape:
            raise ValueError('Fixed/mobile control dimensions differ')
        row['action_displacement'] = dict(max_absolute_kw=float(np.max(abs(uM-uF))),
            l2_kw=float(np.linalg.norm(uM-uF)), same_power_exact=bool(np.array_equal(uF,uM)))
        before = {name: fixed._line_id(f.r, f.x) for name, f in inputs['fits'].items()}
        row['input_hashes'] = dict(training=fingerprint(raw['training']),
            training_meter=fingerprint(raw['training_meter']), future_meter=fingerprint(raw['future_meter']),
            forecast=fingerprint(raw['forecast']), trained_lines=before,
            saved_training_fits=fingerprint(raw['training_result']['fits']),
            candidates=fingerprint(raw['candidates']),
            uF=sha256(uF.astype('<f8').tobytes()).hexdigest(),
            uM=sha256(uM.astype('<f8').tobytes()).hexdigest())
        context_start = perf_counter()
        context = fixed.build_fixed_contexts(inputs['training'], inputs['bank'], inputs['fits'],
            inputs['tm'], inputs['fm'], -fixed.CONFIG['command_power_kw'],
            fixed.CONFIG['command_power_kw'], uF, base_kw=raw['config']['base_kw'],
            forecast=inputs['forecast'], **raw['config']['source'])
        row['fixed_context_seconds'] = perf_counter() - context_start
        if not context.get('success'):
            raise ValueError('Training-only fixed context failed: ' + str(context['diagnostics']))
        projection_start = perf_counter()
        pairs, diagnostics = ac.projected_context(context, uF)
        row['projection_seconds'] = perf_counter() - projection_start
        qF, qM = projected.aggregate_pairs(pairs, uF), projected.aggregate_pairs(pairs, uM)
        row['new_proxy'] = dict(fixed=qF, mobile=qM, delta=qM['score'] - qF['score'],
            expansion_method='scalar_fixed', expansion_power_kw=uF,
            own_profile=diagnostics['own_profile'], training_diagnostics=diagnostics['diagnostics'],
            pair_diagnostics=[dict(a=p.a_label, b=p.b_label, projected_control_rank=p.rank,
                nuisance=p.projector.diagnostics(),
                nuisance_fingerprint=p.projector.matrix_fingerprint,
                projected_residual_hash=fingerprint(p.projected_residual),
                projected_control_hash=fingerprint(p.projected_control)) for p in pairs],
            scope=diagnostics['scope'])
        after = {name: fixed._line_id(f.r, f.x) for name, f in inputs['fits'].items()}
        if before != after:
            raise RuntimeError('Saved trained R/X mutated during derivative assembly')
        row['trained_lines_unchanged'] = True
        row['derivative_assembly_calls'] = dict(future_context=len(inputs['bank']),
            history=len(inputs['bank']), future_projection=len(inputs['bank']))
        # Observed truth label and final inference enter only after both proxy
        # scores have been computed at the prechosen fixed-parent expansion.
        oldF, oldM = (saved_source_score(raw, name) for name in METHODS)
        postF, postM = (methods[name]['inference']['joint_total'] for name in METHODS)
        if not all(p.get('complete') and p.get('M') is not None and np.isfinite(p['M'])
                   for p in (postF, postM)):
            raise ValueError('Complete saved joint-total inference required for both actions')
        row['old_proxy'] = dict(fixed=oldF, mobile=oldM, delta=oldM['score'] - oldF['score'])
        row['actual'] = dict(fixed_M=float(postF['M']), mobile_M=float(postM['M']),
            delta_M=float(postM['M'] - postF['M']), units='full standardized SSE gap',
            true_name_used_only_for_offline_explanation=raw['true_name'])
        row['old_ranking'] = signs(row['old_proxy']['delta'], row['actual']['delta_M'])
        row['new_ranking'] = signs(row['new_proxy']['delta'], row['actual']['delta_M'])
        truth = raw['true_name']
        true_gap = lambda result: next(r['half_gap'] for r in result['nearest'] if r['a'] == truth)
        row['true_reference_new_proxy'] = dict(fixed=true_gap(qF), mobile=true_gap(qM),
                                               delta=true_gap(qM)-true_gap(qF))
        row['success'] = True
    except Exception as exc:
        row.update(failure=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    row['seconds'] = perf_counter() - began
    return row


def public_domain(raw):
    u = np.asarray(next(m for m in raw['methods'] if m['name'] == 'scalar_mobile')['power_kw'])
    T, n = u.shape
    case = raw['case_id']
    if case.startswith('remote8_H'):
        return newcases.service_domain('remote8', T, n, mobile=True)
    if case == 'twinblock12':
        return newcases.service_domain(case, T, n, mobile=True)
    return oldcases.service_domain(case, T, n, mobile=True)


def route_record(raw):
    """Decode the saved selected x, never solve to infer a route from u."""
    began = perf_counter()
    row = {k: raw[k] for k in ('physical_bank', 'index', 'seed', 'case_id')}
    row['success'] = False
    try:
        domain = public_domain(raw)
        selection = raw['selection']['methods']['scalar_mobile']['raw']
        x = selection.get('selected_operation_x')
        if x is None:
            raise ValueError('Saved selected complete operation vector missing')
        audit = domain.check(x)
        if not audit['feasible']:
            raise ValueError('Saved selected x fails original public operation domain: ' + str(audit))
        decoded = domain.decode(x)
        method = next(m for m in raw['methods'] if m['name'] == 'scalar_mobile')
        u = np.asarray(method['power_kw'], float)
        error = float(np.max(abs(decoded['u_kw'] - u)))
        if error > 1e-7:
            raise ValueError('Saved selected x differs from the actually evaluated power')
        travel = [a for a in decoded['selected_arcs'] if a.kind == 'travel']
        if len(travel) != 1 or travel[0].origin != 0:
            raise ValueError('Expected exactly one saved relocation from site0')
        trip = travel[0]; departure = trip.departure
        local = u[:departure, 0]
        charged = bool(np.any(local > 1e-8)); discharged = bool(np.any(local < -1e-8))
        category = ('immediate_departure' if departure == 0 else
                    'local_power_then_move' if charged or discharged else 'wait_then_move')
        actual_route = method.get('operation', {}).get('audit', {}).get('route', [])
        actual_travel = [a for a in actual_route if a['kind'] == 'travel']
        row.update(success=True, category=category, departure_slot=departure,
            destination=trip.destination, arrival_slot=trip.arrival,
            local_charge_before_move=charged, local_discharge_before_move=discharged,
            local_charge_kwh=float(domain.config.dt_hours * np.maximum(local, 0).sum()),
            local_discharge_kwh=float(domain.config.dt_hours * np.maximum(-local, 0).sum()),
            destination_nonzero_slots=np.flatnonzero(abs(u[:, trip.destination]) > 1e-8),
            worker_minutes=audit['driver_cost'], driving_energy_kwh=audit['travel_energy_kwh'],
            selected_x_actual_power_max_difference_kw=error,
            driver_replay_departure_slot=actual_travel[0]['departure'] if len(actual_travel) == 1 else None,
            driver_replay_destination=actual_travel[0]['destination'] if len(actual_travel) == 1 else None,
            selected_and_replay_trip_equal=bool(len(actual_travel) == 1 and
                actual_travel[0]['departure'] == departure and actual_travel[0]['destination'] == trip.destination),
            selected_operation_sha256=sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest())
    except Exception as exc:
        row.update(failure=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    row['seconds'] = perf_counter() - began
    return row


def summarize(rows):
    def group(chosen):
        complete = [r for r in chosen if r['success']]
        return dict(denominator=len(chosen), complete=len(complete), failed=len(chosen)-len(complete),
            old_counts=dict(Counter(r['old_ranking']['category'] for r in complete)),
            new_counts=dict(Counter(r['new_ranking']['category'] for r in complete)),
            old_rise_actual_fall=sum(r['old_ranking']['predicted_rise_actual_fall'] for r in complete),
            new_rise_actual_fall=sum(r['new_ranking']['predicted_rise_actual_fall'] for r in complete),
            corrected_old_reversals=sum(r['old_ranking']['category'] == 'reversed' and
                r['new_ranking']['category'] == 'same_order' for r in complete),
            new_reversals=sum(r['old_ranking']['category'] != 'reversed' and
                r['new_ranking']['category'] == 'reversed' for r in complete))
    result = group(rows)
    result['representatives'] = group([r for r in rows if r['role'] == 'predeclared_representative'])
    result['events'] = group([r for r in rows if r['role'] == 'observed_decision_change_explanation'])
    return result


def render(payload, path):
    s = payload['summary']
    lines = ['# 已保存EV动作的共享线路投影指标回放', '',
        '使用两库index 0/12/24/36共8条预声明代表，另加N3 index 9、20与N4 index 14三条既有变化事件。动作取原scalar固定/移动计划；展开点始终为固定父计划。没有重新训练、源拟合、EV规划、候选筛选或轨迹生成。', '',
        '令$r_{ab}$为竞争模型$b$对历史与参考$a$预测未来的标准化残差，$J_b$为共享线路及逐时源的导数，$B_{ab}$为两模型对EV功率的响应差。投影矩阵$P_b$去掉能由这些参数一阶调整解释的部分。', '',
        r'$$q_{ab}(u)=\tfrac12\|P_b[r_{ab}+B_{ab}(u-u_F)]\|^2-q_{aa}(u_F),\qquad Q(u)=\frac1K\sum_a\min_{b\ne a}q_{ab}(u).$$', '',
        '$u_F$为固定父功率，$K$为候选数。每个参考仍使用原来一组训练线路；竞争线路的线性调整跨历史与未来共享。$J_b$允许无界的一阶补偿，数值秩及历史支撑保存在JSON。该式是冻结切线模型，未给非线性AC距离下界或最终排错概率。', '',
        r'原$\phi$为固定训练线路、只重估源的完整AC预测半SSE距离；实际$M$为竞争候选最小总SSE减真候选总SSE，使用保存的历史+验证+EV联合线路拟合。排序比较只比较增减方向，不直接比较三者绝对大小。', '',
        f"完成{s['complete']}/{s['denominator']}历史；原指标预测升而实际降{s['old_rise_actual_fall']}条，新投影为{s['new_rise_actual_fall']}条；原反转转为同序{s['corrected_old_reversals']}条，新增反转{s['new_reversals']}条。", '',
        '|库/index/用途|原Δφ|新ΔQ|实际ΔM|原排序→新排序|', '|---|---:|---:|---:|---|']
    for r in payload['histories']:
        label=f"{r['physical_bank']}/{r['index']}/"+('代表' if r['role']=='predeclared_representative' else '事件')
        if not r['success']:
            lines.append(f"|{label}|失败：{r.get('failure')}||||");continue
        lines.append(f"|{label}|{r['old_proxy']['delta']:.6g}|{r['new_proxy']['delta']:.6g}|{r['actual']['delta_M']:.6g}|{r['old_ranking']['category']} → {r['new_ranking']['category']}|")
    lines.extend(['',f"8条代表：{s['representatives']}；3条解释事件：{s['events']}。", '',
        '三条变化事件用于解释已观察问题，整个11条回放不构成新策略的独立验证。自由切线补偿可能过强；远离固定父动作时，线路与功率交叉变化及源边界仍可影响排序。', ''])
    routes = payload.get('routes')
    if routes:
        lines.extend(['## 原移动动作的出发与本地操作', '',
            '直接读取原selected_operation_x，用同case_id的公开动作域解码并检查约束。没有从净功率重新求路线。', '',
            '|库|实际移动分母|完整解码|立即出发|本地先充/放后移动|仅等待后移动|选中与执行出发相同|',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for bank, a in routes['summary'].items():
            c=a['categories'];lines.append(f"|{bank}|{a['denominator']}|{a['complete']}|{c.get('immediate_departure',0)}|{c.get('local_power_then_move',0)}|{c.get('wait_then_move',0)}|{a['same_driver_replay_trip']}|")
        lines.extend(['', '本地有功率后移点说明实际使用了两个地点的操作窗口；它本身不证明两地均贡献了辨识收益。两地点信息贡献仍应结合相同曲线的空间反事实。立即离开或只在起点等待的动作，主要体现到达另一地点及其可用时段。', ''])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def run(root, output, *, parent_audit=None):
    root, output = Path(root), Path(output)
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('Preserve prior diagnostic output; choose a new path')
    sources = frozen.source_hashes()
    slots = [(bank, i, 'predeclared_representative') for bank in ('n3', 'n4') for i in INDICES]
    slots += [(bank, i, 'observed_decision_change_explanation') for bank, i in EXTRA]
    rows = []
    for bank, index, role in slots:
        try:
            choices = list((root / ('mobile_v2_formal_eval_' + bank)).glob(f'case_{index:03d}_*.json.gz'))
            if len(choices) != 1:
                raise ValueError('Exactly one frozen raw record required')
            path = choices[0]; content = path.read_bytes(); raw = io.read_json(path)
            row = replay_record(raw, expected_sources=sources)
            row['raw_source'] = dict(path=str(path.resolve()), bytes=len(content), sha256=sha256(content).hexdigest())
        except Exception as exc:
            row = dict(physical_bank=bank, index=index, success=False,
                       failure=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        row['role'] = role; rows.append(row)
        print(json.dumps(dict(bank=bank, index=index, role=role, success=row['success'],
                              seconds=row.get('seconds')), ensure_ascii=False), flush=True)
        io.write_json(output.with_suffix('.checkpoint.json'), dict(histories=rows, summary=summarize(rows)))
    payload = dict(schema='mobile_v3_saved_proxy_replay_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), summary=summarize(rows), histories=rows,
        uses_future_truth_for_expansion=False, reference_action='saved scalar_fixed',
        source_sha256=sources, tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        new_module_sha256={name:sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('ev_mobile_v3_ac.py','ev_mobile_v3_projection.py')},
        no_new_training=True, no_parameter_fit=True, no_EV_planning=True, no_case_generation=True,
        statistical_scope='offline explanation of fixed saved actions; not independent strategy validation')
    if parent_audit is not None:
        parent_path = Path(parent_audit); parent = io.read_json(parent_path); route_rows = []
        for saved in parent['rows']:
            try:
                path = Path(saved['raw_path']); content = path.read_bytes()
                if sha256(content).hexdigest() != saved['raw_sha256']:
                    raise ValueError('Parent audit raw hash differs')
                raw = io.read_json(path)
                if raw['source_sha256'] != sources:
                    raise ValueError('Frozen source mismatch in route audit')
                row = route_record(raw)
                row['raw_source'] = dict(path=str(path.resolve()), sha256=saved['raw_sha256'])
            except Exception as exc:
                row = dict(physical_bank=saved['bank'], index=saved['index'], success=False,
                           failure=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
            route_rows.append(row)
        summary = {}
        for bank in ('n3','n4'):
            chosen=[r for r in route_rows if r['physical_bank']==bank]; good=[r for r in chosen if r['success']]
            summary[bank]=dict(denominator=len(chosen),complete=len(good),failed=len(chosen)-len(good),
                categories=dict(Counter(r['category'] for r in good)),
                local_charge_before_move=sum(r['local_charge_before_move'] for r in good),
                local_discharge_before_move=sum(r['local_discharge_before_move'] for r in good),
                local_both_before_move=sum(r['local_charge_before_move'] and r['local_discharge_before_move'] for r in good),
                same_driver_replay_trip=sum(r['selected_and_replay_trip_equal'] for r in good))
        payload['routes']=dict(parent_audit_path=str(parent_path.resolve()),
            parent_audit_sha256=sha256(parent_path.read_bytes()).hexdigest(),summary=summary,rows=route_rows,
            MILP_calls=0,AC_calls=0)
    io.write_json(output,payload);render(payload,output.with_suffix('.md'))
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs'))
    parser.add_argument('--output', type=Path, default=Path('outputs/mobile_v3_proxy_replay.json'))
    parser.add_argument('--parent-audit', type=Path, default=Path('outputs/mobile_v3_parent_initial_audit.json'))
    args=parser.parse_args(); result=run(args.root,args.output,parent_audit=args.parent_audit)
    print(json.dumps(dict(proxy=result['summary'],routes=result.get('routes',{}).get('summary')),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
