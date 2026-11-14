"""Post-hoc, full-domain activation-cost certificates; no optimizer imports.

Reads only frozen source300 forecasts, candidate scores, resource configuration
and quoted actions. It neither reads AC observations nor modifies a plan or its
native solver bound. The certificate is for nominal quoted activation cost.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
from statistics import NormalDist
import sys

import numpy as np


# Historical code alias: this is the common S_base=200 kVA. Numerically P[kW]
# and Q[kvar] are both divided by 200; it is not a device's active-power rating.
BASE_KW = 200.
HOURS, REPEAT, DT = 24, 4, 1.
BUFFER = 1e-8
ALLOWED_ENTRY_IDS = {
    'pengwah18__2024-10-14__registered_plus1',
    'pengwah18__2024-10-14__registered_q95_36',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path, expected=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Frozen input changed: '+str(path))
    return dict(path=str(path), sha256=digest)


def checked(ref):
    return reference(ref['path'], ref['sha256'])


def window(clock, start, end):
    return ((clock >= start) & (clock < end) if start < end
            else (clock >= start) | (clock < end))


def declared_arrays(config, labels, forecast):
    """Independent transcription of the frozen mixed-unit resource contract.

    Only power boxes are needed: relaxing energy/guard constraints enlarges the
    domain and preserves the lower bound. Exact source-action fees are checked.
    """
    labels = list(map(int, labels))
    if labels != config['expected_terminal_labels']:
        raise ValueError('Resource labels/order changed')
    resources = config['resources']; n, r = len(labels), len(resources)
    if forecast.shape != (HOURS*REPEAT, 2*n):
        raise ValueError('Expected the frozen 96-sample forecast')
    B = np.zeros((2*n, r)); lo = np.zeros((HOURS, r)); hi = lo.copy()
    plus, minus = lo.copy(), lo.copy()
    clock = (8+np.arange(HOURS)+.5) % 24
    pricing = config['pricing']; multipliers = pricing['clock_hour_multiplier']
    mult = np.full(HOURS, multipliers['default'])
    for start, end in pricing['peak_clock_windows']:
        mult[window(clock, start, end)] = multipliers['morning_evening_peak']
    for start, end in pricing['night_clock_windows']:
        mult[window(clock, start, end)] = multipliers['night']
    if pricing['reservation_fee_yuan_per_kw_day'] != 0.:
        raise ValueError('This certificate handles the declared activation-only fee')
    export = np.maximum(0., -forecast[:, :n]*BASE_KW).reshape(HOURS, REPEAT, n).min(1)
    for j, row in enumerate(resources):
        node = labels.index(int(row['node_label'])); kind = row['kind']
        if kind == 'reactive':
            B[n+node, j] = 1.
            hours = row.get('available_clock_hours')
            active = np.ones(HOURS, bool) if hours is None or hours == list(range(24)) else window(clock, *hours)
            lo[active, j], hi[active, j] = -float(row['capacity_kvar']), float(row['capacity_kvar'])
            factor = np.asarray(row.get('hourly_fee_multiplier', np.ones(HOURS)), float)
            plus[:, j] = DT*factor*float(row['fee_positive_yuan_per_kvarh'])
            minus[:, j] = DT*factor*float(row['fee_negative_yuan_per_kvarh'])
            if row.get('reservation_fee_yuan_per_kvar_day', 0.) != 0.:
                raise ValueError('Nonzero capacity fee requires a separate cost statement')
        else:
            if kind not in ('pv', 'ev', 'thermal') or row['q_over_p'] != 0.:
                raise ValueError('Unexpected P response specification')
            B[node, j] = 1.; cap = float(row['contract_kw'])
            if kind == 'pv':
                hi[:, j] = np.where(window(clock, 10, 14), np.minimum(cap, export[:, node]), 0.)
            else:
                active = window(clock, 18, 7) if kind == 'ev' else np.ones(HOURS, bool)
                lo[active, j], hi[active, j] = -cap, cap
            plus[:, j] = DT*mult*float(row['fee_positive_yuan_per_kwh'])
            minus[:, j] = DT*mult*float(row['fee_negative_yuan_per_kwh'])
    if (not np.isfinite(B).all() or not np.isfinite(plus).all()
            or not np.isfinite(minus).all() or np.any(plus < 0) or np.any(minus < 0)):
        raise ValueError('Negative or nonfinite prices are outside the certificate assumptions')
    return B, lo, hi, plus, minus


def fee(u, plus, minus):
    return float((plus*np.maximum(u, 0)+minus*np.maximum(-u, 0)).sum())


def constants(d0, effect, plus, minus, positive_allowed, negative_allowed):
    """A column occupies REPEAT successive sample rows in a single hour."""
    h, repeat, n = d0.shape
    norms = np.broadcast_to(math.sqrt(repeat)*np.linalg.norm(effect, axis=0), plus.shape)
    linear = 2*np.einsum('htn,nr->hr', d0, effect)
    directions = []
    for sign, prices, allowed in ((1, plus, positive_allowed), (-1, minus, negative_allowed)):
        for hour, resource in np.argwhere(allowed):
            c, a_norm, b = float(prices[hour, resource]), float(norms[hour, resource]), float(sign*linear[hour, resource])
            if c == 0 and a_norm > 0:
                return dict(valid_finite_M=False, reason='available_zero_price_direction_has_nonzero_response',
                            M=None, L=None, directions=[])
            if c == 0:  # A truly zero response contributes neither norm nor linear term.
                continue
            directions.append(dict(hour=int(hour), resource=int(resource), sign=sign,
                fee_per_native_unit=c, column_norm=a_norm, linear_coefficient=b,
                norm_per_yuan=a_norm/c, linear_per_yuan=b/c))
    M = max((x['norm_per_yuan'] for x in directions), default=0.)
    L = max(0., max((x['linear_per_yuan'] for x in directions), default=0.))
    return dict(valid_finite_M=True, reason=None, M=M, L=L, directions=directions)


def cost_bounds(J0, required, M, L):
    if required <= J0:
        return dict(triangle_yuan=0., quadratic_yuan=0., unreachable_no_response=False)
    if M is None:
        return dict(triangle_yuan=0., quadratic_yuan=0., unreachable_no_response=False)
    if M == 0:
        return dict(triangle_yuan=None, quadratic_yuan=None, unreachable_no_response=True)
    delta = required-J0
    triangle = (math.sqrt(required)-math.sqrt(J0))/M
    # Stable positive root of M^2*C^2+L*C-delta=0.
    quadratic = 2*delta/(L+math.sqrt(L*L+4*M*M*delta))
    return dict(triangle_yuan=triangle, quadratic_yuan=quadratic, unreachable_no_response=False)


def numerical_checks(d0, effect, plus, minus, lo, hi, const, seed, samples=256):
    """Test sparse fee-budget actions and box actions, not only the saved plan.

    Energy constraints are deliberately not imposed: the claimed inequalities
    hold on this larger domain. Random checks supplement the analytic proof.
    """
    rng = np.random.default_rng(seed); directions = const['directions']
    M, L = const['M'], const['L']; J0 = float(np.square(d0).sum())
    maximum = dict(norm=0., linear=0., triangle_J=0., quadratic_J=0.)
    minimum_slack = {key: math.inf for key in maximum}
    minimum_slack_nonzero = {key: math.inf for key in maximum}
    max_abs_identity = 0.; count = 0
    for index in range(samples+1):
        u = np.zeros_like(plus)
        if index and index % 2:
            u = rng.uniform(lo, hi)
        elif index:
            budget = 10**rng.uniform(-4., 2.)
            chosen = rng.choice(len(directions), size=int(rng.integers(1, 9)), replace=True)
            weights = rng.dirichlet(np.ones(len(chosen)))
            for k, weight in zip(chosen, weights, strict=True):
                item = directions[int(k)]
                u[item['hour'], item['resource']] += item['sign']*budget*weight/item['fee_per_native_unit']
        cost = fee(u, plus, minus)
        increment = np.repeat((u@effect.T)[:, None, :], d0.shape[1], axis=1)
        norm = float(np.linalg.norm(increment)); linear = float(2*np.sum(d0*increment))
        J = float(np.square(d0+increment).sum())
        max_abs_identity = max(max_abs_identity, abs(J-J0-linear-norm*norm))
        excess = dict(norm=norm-M*cost, linear=linear-L*cost,
            triangle_J=J-(math.sqrt(J0)+M*cost)**2,
            quadratic_J=J-J0-L*cost-M*M*cost*cost)
        for key, value in excess.items():
            maximum[key] = max(maximum[key], value)
            minimum_slack[key] = min(minimum_slack[key], -value)
            if cost > 0:
                minimum_slack_nonzero[key] = min(minimum_slack_nonzero[key], -value)
            if value > 1e-8*max(1., J, M*M*cost*cost):
                raise ArithmeticError('Numerical inequality failed: '+key)
        count += 1
    return dict(passed=True, seed=seed, samples=count,
                distribution='zero plus alternating independent resource-box draws and sparse signed cost budgets 1e-4..1e2 yuan; no AC or future stream',
                maximum_positive_violation=maximum, minimum_slack=minimum_slack,
                minimum_nonzero_action_slack=minimum_slack_nonzero,
                maximum_J_expansion_error=max_abs_identity)


def small_checks():
    # Direction-dependent prices: cheaper negative action determines M, whereas
    # the more expensive positive direction alone can increase the linear term.
    d = np.array([[[1.]]]); F = np.array([[2.]])
    plus, minus = np.array([[3.]]), np.array([[1.]])
    enabled = np.ones((1, 1), bool)
    c = constants(d, F, plus, minus, enabled, enabled)
    if c['M'] != 2. or abs(c['L']-4/3) > 1e-14:
        raise ArithmeticError('Directional prices were not respected')
    zero = constants(d, F, np.zeros((1, 1)), minus, enabled, enabled)
    if zero['valid_finite_M'] or cost_bounds(1., 4., zero['M'], zero['L'])['quadratic_yuan'] != 0.:
        raise ArithmeticError('A free nonzero response must withdraw the positive certificate')
    inactive = constants(d, F, np.zeros((1, 1)), minus, ~enabled, enabled)
    if not inactive['valid_finite_M'] or inactive['M'] != 2.:
        raise ArithmeticError('Unavailable free direction must be omitted')
    four = constants(np.repeat(d, 4, axis=1), F, plus, minus, enabled, enabled)
    if four['M'] != 4. or abs(four['L']-16/3) > 1e-14:
        raise ArithmeticError('Four observations require sqrt(4) column norm and 4x linear term')
    return dict(passed=True, tests=['asymmetric_sign_prices', 'available_zero_fee_withdrawal',
        'unavailable_zero_fee', 'four_sample_repetition'])


PROOF = r'''# 运行期间补充的事后费用下界

本证书使用已经冻结的300秒批次预测、候选参数、资源合同与报价。未读取实际新日电压、真实网络参数或执行结果；未优化、改动作或替换原生下界。费用指规划时的激活补偿，不含设备安装、容量预留和损耗，也不保证交付后的随机费用。

脚本仅适用于当前Pengwah18、2024-10-14历史窗口、23/36 kvar资源配置及12张延时报价协议。它显式检查案例标识、18个末端301—318、4次预测、24小时和每小时4条观测。共同归一化基准是S_base=200 kVA：有功kW与无功kvar数值均除以200；这是本案固定的标幺基准，不是任一设备容量。代码变量BASE_KW沿用历史名称，报告统一标为power_base_kva。

将近对预测差除以 sqrt(2 sigma^2)，按时间与节点排成向量，记为 d0+A u，故 J(u)=||d0+A u||^2，J0=||d0||^2。每个小时动作会重复进入4条15分钟观测，A的对应列范数为 sqrt(4)||F列||；费用只乘1小时，不能再乘4。P动作单位kW，Q动作单位kvar；各自的元/kWh、元/kvarh单价乘1小时后都是每单位动作的费用，因而可相加为元。

对每个允许的正/负动作方向，设 a_l 为带符号的A列，c_l>0为相应方向费用，v_l>=0为动作幅度。C=sum c_l v_l，Au=sum a_l v_l。定义

M=max_l ||a_l||/c_l，L=max(0,max_l 2 d0^T a_l/c_l)。

由三角不等式，||Au||<=M C；逐项相加有2 d0^T Au<=L C。因此

J(u)<= (sqrt(J0)+M C)^2，
J(u)<= J0+L C+M^2 C^2。

若原约束要求J(u)>=R>J0，则原问题的任何可行点均满足

C >= (sqrt(R)-sqrt(J0))/M，
C >= 2(R-J0)/(L+sqrt(L^2+4 M^2(R-J0)))。

因为L<=2 sqrt(J0) M，第二个界不弱于第一个界。这些不等式对更大的功率盒域就成立，加入原能量耦合、方向保护和其余竞争者要求仍然成立；没有施加新的内域限制。若可用的零费用方向具有非零响应，有限M公式不适用，本实现撤回为平凡0下界。若全部动作均无响应而R>J0，则标记该门槛不可达。负费用未被该推导覆盖，会显式拒绝。

本例历史两目标均需保护。h为两方向历史对数赔率中较小值，z=Phi^{-1}(1-.1/(K-1))，tau为冻结并列容差加1e-8原生缓冲。h<tau时，原式h+J-z sqrt(2J)>=tau与

R=.5 [z+sqrt(z^2+2(tau-h))]^2，J>=R

等价。本脚本独立核算该R，另报不含缓冲的R。该证书不承诺实际AC恢复，也不覆盖未纳入规划的原生参数点；它是既有代表候选QCQP的费用必要条件。

输出同时保留原300秒的native下界与上界。解析下界是本轮运行期间新增的事后证书，不冒充求解器输出；900秒原生结果继续按主批原记录报告。随机不等式检查用于查实现错误，证明来自上述解析推导。
'''


def run(extended_runs):
    rows = []; protocols = []; seed = 14001134
    for extended in extended_runs:
        protocol_ref = reference(Path(extended)/'protocol.json')
        protocol = read(protocol_ref['path']); protocols.append(protocol_ref)
        if protocol['forecasts'] != 4 or protocol['modes'] != ['symmetric_guard']:
            raise ValueError('This case-specific audit requires the frozen four-forecast symmetric-guard protocol')
        for record in protocol['source_quote_inputs']:
            if (record['entry_id'] not in ALLOWED_ENTRY_IDS
                    or record['forecast_index'] not in range(4)):
                raise ValueError('This audit is limited to Pengwah18 2024-10-14 and the current 23/36 kvar contracts')
            refs = {key: checked(record[key]) for key in ('source_score', 'resource_config', 'planner', 'plans', 'seal')}
            seal = read(refs['seal']['path'])
            for key in ('planner', 'plans'):
                if checked(seal[key]) != refs[key]:
                    raise ValueError('Continuation record and source seal disagree')
            if checked(seal['score']) != refs['source_score']:
                raise ValueError('Score reference is not the source-sealed score')
            config = read(refs['resource_config']['path']); score = read(refs['source_score']['path'])
            plans = read(refs['plans']['path'])['symmetric_guard']
            with np.load(refs['planner']['path'], allow_pickle=False) as data:
                forecast, prior = data['forecast'].copy(), data['symmetric_guard'].copy()
            models = np.asarray([item['H'] for item in score['parameters']], float)
            sse = np.asarray([item['history_sse'] for item in score['parameters']], float)
            sigma2 = float(score['sigma2']); K, n = len(models), len(score['labels'])
            if list(score['labels']) != list(range(301, 319)) or K != 4:
                raise ValueError('Expected the frozen Pengwah18 terminal order and four representative candidates')
            weights = np.exp(-(sse-sse.min())/(2*sigma2)); weights /= weights.sum()
            order = np.argsort(-weights, kind='stable')
            cut = min(int(np.searchsorted(np.cumsum(weights[order]), .99)), K-1)
            targets = [int(i) for i in order if weights[i] >= weights[order[cut]]-1e-14]
            if targets != plans['targets'] or len(targets) < 2:
                raise ValueError('Expected the unchanged two-sided target requirement')
            pair = min(((min(i,j),max(i,j)) for i in targets for j in targets if i<j),
                       key=lambda ij: (abs(sse[ij[0]]-sse[ij[1]]), ij))
            i,j = pair; D = models[i]-models[j]
            h = -abs(float((sse[j]-sse[i])/(2*sigma2)))
            z = NormalDist().inv_cdf(1-.1/(K-1))
            tie = max(1e-7, (score['cells']+len(forecast)*n)*score['reference_mse']*score['validation_tie']/(2*sigma2))
            if h >= tie:
                raise ValueError('This certificate requires the single high-branch case')
            required = .5*(z+math.sqrt(z*z+2*(tie+BUFFER-h)))**2
            unbuffered = .5*(z+math.sqrt(z*z+2*(tie-h)))**2
            B, lo, hi, plus, minus = declared_arrays(config, score['labels'], forecast)
            d0 = (forecast@D.T/math.sqrt(2*sigma2)).reshape(HOURS, REPEAT, n)
            effect = D@B/(BASE_KW*math.sqrt(2*sigma2))
            J0 = float(np.square(d0).sum())
            const = constants(d0, effect, plus, minus, hi>0, lo<0)
            if not const['valid_finite_M']:
                raise ValueError('This actual batch unexpectedly has a free nonzero-response direction')
            bounds = cost_bounds(J0, required, const['M'], const['L'])
            if bounds['quadratic_yuan'] < bounds['triangle_yuan']-1e-12:
                raise ArithmeticError('Quadratic bound is weaker than the triangle bound')
            all_pairs = [(a,b) for a in range(K) for b in range(a+1,K) if a in targets or b in targets]
            position = all_pairs.index(pair)
            prior_J = float(np.square(d0+np.repeat((prior@effect.T)[:,None,:], REPEAT, axis=1)).sum())
            prior_fee = fee(prior, plus, minus)
            errors = dict(source_zero_J=abs(J0-plans['baseline']['J'][position]),
                          source_action_J=abs(prior_J-plans['final']['J'][position]),
                          source_action_fee=abs(prior_fee-plans['cost_yuan']))
            if max(errors.values()) > 1e-7 or bounds['quadratic_yuan'] > prior_fee+1e-7:
                raise ArithmeticError('Independent score/fee or lower-bound check failed')
            if const['L'] > 2*math.sqrt(J0)*const['M']+1e-10:
                raise ArithmeticError('Cauchy consistency failed')
            check = numerical_checks(d0, effect, plus, minus, lo, hi, const, seed+len(rows))
            directions = const.pop('directions')
            max_M = max(directions, key=lambda item:item['norm_per_yuan'])
            max_L = max(directions, key=lambda item:item['linear_per_yuan'])
            for item in (max_M, max_L):
                item['resource_id'] = config['resources'][item['resource']]['id']
                item['node_label'] = config['resources'][item['resource']]['node_label']
            rows.append(dict(extended_run=str(Path(extended).resolve()), entry_id=record['entry_id'],
                forecast_index=record['forecast_index'], source_inputs=refs,
                target_representative_indices=targets, pair_representative_indices=list(pair),
                pair_native_indices=[score['parameters'][k]['index'] for k in pair],
                risk_per_competitor=.1/(K-1), z=z, historical_offset=h, tie=tie,
                native_buffer=BUFFER, required_J=required, unbuffered_required_J=unbuffered,
                J_at_zero=J0, constants=const, maximum_norm_direction=max_M,
                maximum_linear_direction=max_L, bounds=bounds,
                original_300_native_lower_bound_yuan=plans.get('lower_bound_yuan'),
                original_300_action_fee_yuan=prior_fee,
                analytic_bound_over_source300_incumbent=bounds['quadratic_yuan']/prior_fee,
                source_action_J=prior_J, source_action_original_buffered_margin=h+prior_J-z*math.sqrt(2*prior_J)-tie-BUFFER,
                source_consistency_errors=errors, numerical_checks=check,
                units=dict(P='kW', Q='kvar', cost='yuan', dt_hours=DT, repeated_samples=REPEAT,
                    power_base_kva=BASE_KW,
                    power_base_scope='Fixed common S_base=200 kVA: P[kW]/200 and Q[kvar]/200; not a device capacity'),
                resource_ids=[r['id'] for r in config['resources']],
                directional_fees_yuan_per_native_unit=dict(positive=plus.tolist(), negative=minus.tolist())))
    if len(rows) != 12:
        raise ValueError('Expected precisely the frozen twelve-quote supplementary audit')
    if any(name == 'gurobipy' or name.startswith('gurobipy.') for name in sys.modules):
        raise AssertionError('This calculation must not import an optimizer')
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-hoc supplementary original-domain nominal activation-cost certificate, developed during the extended run; native bounds and primary protocol untouched',
        case_scope=dict(case='pengwah18', historical_window_start='2024-10-14',
            allowed_entry_ids=sorted(ALLOWED_ENTRY_IDS), terminal_labels=list(range(301,319)),
            forecasts_per_entry=4, decision_hours=HOURS, samples_per_hour=REPEAT,
            expected_total_quotes=12, common_power_base_kva=BASE_KW,
            general_purpose_case_loader=False, explicit_case_assertions_passed=True),
        source=reference(__file__), extended_protocols=protocols,
        dependencies=dict(python=platform.python_version(), numpy=np.__version__, optimizer_imported=False),
        read_boundaries=dict(source300_forecasts=True, source300_quoted_actions=True, frozen_candidate_scores=True,
            resource_configuration=True, AC_observations=False, true_network=False, execution_results=False,
            current900_results=False, optimization_performed=False, actions_changed=False),
        small_checks=small_checks(), rows=rows,
        summary=dict(quotes=len(rows), numerical_inequality_samples=sum(r['numerical_checks']['samples'] for r in rows),
            triangle_lower_bound_yuan_range=[min(r['bounds']['triangle_yuan'] for r in rows), max(r['bounds']['triangle_yuan'] for r in rows)],
            quadratic_lower_bound_yuan_range=[min(r['bounds']['quadratic_yuan'] for r in rows), max(r['bounds']['quadratic_yuan'] for r in rows)],
            maximum_source_consistency_error=max(max(r['source_consistency_errors'].values()) for r in rows),
            all_random_checks_passed=all(r['numerical_checks']['passed'] for r in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended-run', action='append')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    roots = args.extended_run or ['outputs/flexibility_extended_paired900_20261114_v1',
                                 'outputs/flexibility_extended_fresh900_20261114_v1']
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = run(roots)
    snapshot = out/'lower_bound.py'; shutil.copy2(__file__, snapshot)
    result['source_snapshot'] = reference(snapshot)
    (out/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (out/'PROOF.md').write_text(PROOF, encoding='utf-8')
    print(json.dumps(dict(report=reference(out/'report.json'), summary=result['summary']), ensure_ascii=False))


if __name__ == '__main__':
    main()
