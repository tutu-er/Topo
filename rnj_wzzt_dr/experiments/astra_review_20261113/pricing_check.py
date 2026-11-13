"""Fixed old-forecast solve-time check; no new observations or AC execution."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from .solver import Recovery, plan


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'outputs/customer_flexibility_sweep_20261113_v1'
DAYS = (0, 1)
BUDGETS = (3., 12., 60.)


def verify_inputs(protocol):
    for ref in protocol['inputs']:
        if io.ref(ref['path'])['sha256'] != ref['sha256']:
            raise ValueError('Input drift: '+ref['path'])


def run(out):
    import gurobipy as gp
    if out.exists():
        raise FileExistsError('Preserve previous solve-time checks')
    out.mkdir(parents=True)
    history = BASE/'histories/s0_c013'
    paths = [history/'score.json', history/'train.npz', history/'validation.npz', BASE/'resource_configs/higher.json']
    predictions = []
    for day in DAYS:
        d = BASE/'confirmation/s0_c013'/f'day{day:03d}'
        seal = io.read(d/'policy_seal.json')
        if io.ref(d/'planner.npz')['sha256'] != seal['planner']['sha256']:
            raise ValueError('Old sealed forecast changed')
        paths.extend([d/'planner.npz', d/'policy_seal.json'])
        with np.load(d/'planner.npz') as z:
            predictions.append(np.array(z['forecast']))
    sources = [Path(__file__), Path(__file__).with_name('solver.py'),
        Path(__file__).parents[1]/'customer_flexibility_20261113/model.py',
        Path(__file__).parents[1]/'customer_flexibility_20261113/placement.py']
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), environment=io.environment(),
        days=list(DAYS), budgets_seconds=list(BUDGETS), total_quote_budget_seconds=150., threads=1,
        mode='calibrated', identity='s0_c013', inputs=[io.ref(p) for p in paths],
        sources=[io.ref(p) for p in sources], method='same direct native-bank Gaussian recovery QCQP at3/12/60 seconds',
        budget='plan timer includes mean-model construction, scalar rejection threshold, LP direction/line search and native solve; persistent solver environment startup outside per-quote budget',
        selection='Exactly old sealed day000 and day001 fixed before any long solve; no new-day or best-case selection',
        independent_quotes='Each duration starts from the same deterministic LP recipe; longer-budget quotes are not warm-started from shorter-budget incumbents',
        no_new_AC_days=True, no_actual_future_observations_opened=True, no_new_identifier_fits=True,
        base_kw=200., voltage_sigma=.0005, confidence_scope='declared frozen point forecast, nominal response and historically adjusted Gaussian mean; bounds refer to fixed sufficient-condition QCQP')
    io.write(out/'protocol.json', protocol)
    io.write(out/'lock.json', dict(pid=os.getpid(), protocol=io.ref(out/'protocol.json')))
    verify_inputs(protocol)
    score = io.read(history/'score.json')
    training, validation = load(history/'train.npz'), load(history/'validation.npz')
    case = CustomerCase(io.read(BASE/'resource_configs/higher.json'))
    rows = []
    began = perf_counter()
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        for day, prediction in zip(DAYS, predictions, strict=True):
            for seconds in BUDGETS:
                tick = perf_counter()
                try:
                    receipt, point = plan(case, score, validation, prediction, 200., mode='calibrated',
                        seconds=seconds, env=env, training=training, voltage_sigma=.0005)
                    check = dict(point_present=point is not None)
                    if point is not None:
                        model = Recovery(case, score, validation, prediction, 200., 'calibrated', .0005)
                        direct = model.verify_expansion(point)
                        check.update(original_resource_feasible=case.audit(point)['passed'],
                            original_probability_constraints_feasible=direct['feasible'],
                            recomputed_fee=case.fee(point), fee_error=abs(case.fee(point)-receipt['cost_yuan']),
                            direct_recovery_lower=direct['model_recovery_lower'], expanded_mu_J_bias_checked=True)
                        if not check['original_resource_feasible'] or not direct['feasible'] or check['fee_error']>1e-8:
                            raise ArithmeticError('Returned point fails independent original-domain checks')
                        np.savez_compressed(out/f'private_day{day:03d}_seconds{seconds:g}.npz', u=point,
                            direct_mu=direct['mu'], direct_J=direct['J'])
                    row = dict(day=day, seconds_requested=seconds, status='complete', receipt=receipt,
                        verification=check, wall_seconds_with_verification=perf_counter()-tick)
                except Exception as exc:
                    row = dict(day=day, seconds_requested=seconds, status='failed', error=repr(exc),
                        traceback=traceback.format_exc(), wall_seconds_with_verification=perf_counter()-tick)
                rows.append(row)
                io.write(out/'private_results.json', rows)
                print('pricing', day, seconds, row['status'], row.get('receipt', {}).get('status'),
                      row.get('receipt', {}).get('cost_yuan'), flush=True)
    for ref in protocol['sources']:
        if io.ref(ref['path'])['sha256'] != ref['sha256']:
            raise ValueError('Solver source changed during pricing check')
    summary = dict(protocol=io.ref(out/'protocol.json'), completed_quotes=len(rows),
        failed_quotes=sum(r['status'] != 'complete' for r in rows), total_wall_seconds=perf_counter()-began, by_budget={})
    for seconds in BUDGETS:
        selected = [r['receipt'] for r in rows if r['seconds_requested'] == seconds and r['status'] == 'complete']
        feasible = [r for r in selected if r['feasible']]
        bounds = [r['lower_bound_yuan'] for r in selected if r.get('lower_bound_yuan') is not None]
        summary['by_budget'][str(seconds)] = dict(planned_quotes=len(DAYS), completed_quotes=len(selected), feasible=len(feasible),
            unresolved=sum(not r['feasible'] and not r.get('infeasibility_proved', False) for r in selected),
            infeasible_proved=sum(r.get('infeasibility_proved', False) for r in selected),
            mean_feasible_cost=None if not feasible else float(np.mean([r['cost_yuan'] for r in feasible])),
            mean_available_lower_bound=None if not bounds else float(np.mean(bounds)),
            mean_feasible_relative_gap=None if not feasible else float(np.mean([r['relative_gap'] for r in feasible])),
            global_certificates=sum(r.get('global_optimality_verified', False) for r in feasible),
            initial_retained=sum(r.get('source') == 'verified_initial_retained' for r in feasible),
            native_improved_initial=sum(r.get('source') == 'native_improved_initial' for r in feasible),
            native_found=sum(r.get('source') == 'native_found' for r in feasible),
            mean_initial_seconds=None if not selected else float(np.mean([r.get('initial', {}).get('seconds', 0.) for r in selected])))
    io.write(out/'summary.json', summary)
    io.write(out/'complete.json', dict(status='complete' if not summary['failed_quotes'] else 'complete_with_failures',
        summary=io.ref(out/'summary.json'), results=io.ref(out/'private_results.json')))
    report(out, summary, rows)
    return summary


def report(out, summary, rows):
    def number(value):
        return '无' if value is None else f'{value:.4f}'
    lines = ['# 固定旧预测的报价时间核查', '',
        '输入固定为原弱证据历史的旧day000、day001规划预测；没有读取实际新日P/Q/V，没有新AC测试。每个预算独立从同一LP方向及线搜索初值流程开始，未沿用较短求解的结果。', '',
        '总计2份预测×3种预算=6次报价；单线程。每次预算包含模型构造、拒绝半径计算、初值生成和原QCQP求解。持久求解器环境启动不计入单次报价。', '',
        '|单次预算（秒）|可行／总数|已证不可行|未决|可行费用均值（元）|可用下界均值（元）|可行相对差距均值|费用界闭合|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seconds, r in summary['by_budget'].items():
        lines.append(f"|{float(seconds):g}|{r['feasible']}/{r['planned_quotes']}|{r['infeasible_proved']}|{r['unresolved']}|{number(r['mean_feasible_cost'])}|{number(r['mean_available_lower_bound'])}|{number(r['mean_feasible_relative_gap'])}|{r['global_certificates']}|")
    lines += ['', '初值与实际改进：', '', '|预算（秒）|保留已可行初值|原求解器改进初值|原求解器首次找到可行点|平均初值秒数|',
        '|---:|---:|---:|---:|---:|']
    for seconds, r in summary['by_budget'].items():
        lines.append(f"|{float(seconds):g}|{r['initial_retained']}|{r['native_improved_initial']}|{r['native_found']}|{number(r['mean_initial_seconds'])}|")
    lines += ['', f"失败报价：{summary['failed_quotes']}/6。所有返回点均重新核查原资源域、直接预测矩阵计算的概率约束、二次展开和费用。",
        '费用为原示例激活补偿。可行点给费用上界；报告的下界属于本次固定高斯模型及固定风险分配的QCQP，不能用于精确联合概率问题或实际AC恢复率的最优性声明。不同预算的可行均值仅在可行集合相同时可直接比较。', '',
        '本轮原始报价、控制数组和逐点约束留在忽略目录`'+out.relative_to(ROOT).as_posix()+'`，公开表格仅列两份固定预测的汇总。', '']
    Path(__file__).with_name('PRICING.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())
