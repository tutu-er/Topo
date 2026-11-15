"""Report audited node/hour P-Q bands, keeping reused observations separate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import csv

from ... import io
from ..frozen_rx_guard_20261115 import report as shared
from ..single_target_20261115 import report as single

MODES = ('normal_fixed', 'mgf_fixed', 'mgf_forecast')


def rename_policy(policy, origin):
    if origin == 'grid':
        return policy if policy in ('no_dfs', 'anchor') else 'oldmulti__'+policy
    if not policy.startswith('single_'):
        raise ValueError('Expected an explicitly single-target policy')
    return {'unbanded':'unbanded__', 'band':'band__', 'band_sdp':'band_sdp__'}[origin]+policy[7:]


def markdown(result):
    fmt = single.number
    lines = ['# 逐节点逐时段 P/Q 带：已保存配对路径', '',
        '带宽只用训练量测和相同当地小时的残差范围。冻结候选、R/X、误差法则、资源与费用保持原值；'
        '概率式仍以历史赢家为唯一条件目标、比较全部异族原生参数点。', '',
        f"本批为{'SDP初值对照' if result['variant']=='band_sdp' else '无SDP带约束对照'}。"
        f"{result['new_rows']}条政策记录；新计算AC {result['new_ac_rows']}条，动作完全相同时复用AC {result['reused_ac_rows']}条。"
        f"另外引用旧网格{result['grid_baseline_rows']}条、无带单目标{result['single_baseline_rows']}条、"
        f"已带约束{result['band_baseline_rows']}条，均不算新增独立样本。", '',
        '固定已研究的物理日、预测和设备/量测随机流；本表是开发配对重放。完整支持恢复与内部结构恢复分列。', '',
        '|输入|策略|内部恢复|完整支持|错误权重≥.9|零回退|平均真族权重|总费/元|实际P/kWh|实际Q/kvarh|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in result['cases']:
        for name, s in case['strategies'].items():
            if name.startswith('oldmulti__') or name == 'anchor':
                continue
            lines.append(f"|{case['id']}|{name}|{s['recovered']}/{s['n']}|{s['full_support_recovered']}/{s['n']}|"
                f"{s['wrong_high']}|{s['zero_fallback_rows']}|{fmt(s['p_true']['mean'])}|{fmt(s['settled_total_yuan']['mean'])}|"
                f"{fmt(s['quantities']['delivered_P_kwh']['mean'])}|{fmt(s['quantities']['delivered_Q_kvarh']['mean'])}|")
    lines += ['', '|输入|新策略|比较对象|新增正确|丢失正确|', '|---|---|---|---:|---:|']
    for case in result['cases']:
        for reference, comparisons in case['comparisons'].items():
            for name, pair in comparisons.items():
                lines.append(f"|{case['id']}|{name}|{reference}|{pair['new_recoveries']}|{pair['lost_recoveries']}|")
    lines += ['', '## 概率门槛、带约束与求解', '',
        '仅通过原概率式、原设备域和新带约束的报价计作可行。报价失败后的零动作保留在执行分母；'
        '实际交付的P/Q带超出情况另列，不与设备违规混淆。SDP只提供初值；其近似目标值不作QCQP有效费用下界。', '',
        '|输入|模式|状态/原生状态|可行|零回退|调用费下界|调用费上界|gap|报价秒|返回来源|',
        '|---|---|---|---|---|---:|---:|---:|---:|---|']
    for q in result['quotes']['rows']:
        cert = q['certificate']
        lines.append(f"|{q['id']}|{q['policy']}|{q.get('status')}/{q.get('native_status')}|"
            f"{q.get('new_gate_feasible')}|{q.get('zero_fallback')}|{fmt(cert.get('lower_bound_yuan'))}|"
            f"{fmt(cert.get('upper_bound_yuan',cert.get('executed_activation_yuan') if cert.get('accepted') else None))}|"
            f"{fmt(cert.get('relative_gap'))}|{fmt(q.get('quote_wall_seconds'))}|{q.get('source')}|")
    lines += ['', f"报价{result['quotes']['n']}份；原生限时{result['quotes']['native_time_limit_quotes']}份；"
        f"零费充分{result['quotes']['zero_sufficient_quotes']}份。有效界对象是调用费，不是预约加调用总费。",
        '费用下界仅对应带有数值buffer的编码QCQP（正带缩紧1e−7），不外推为未缩紧域的下界。'
        '历史下界与后来同编码域可行报价冲突时保留原始轨迹并撤回相关证书；最终状态或末端gap不替代该检查。', '',
        '## SDP初值阶段', '',
        '两种指数模式先分别生成候选，去重后与原池合并，再供两报价共用。SDP额外准备时间单列，'
        '不将相同600秒QCQP预算称作相同总耗时。没有新增候选时复用已核验的带约束方案，明确不计新QCQP求解。', '',
        '|输入|模式|SDP状态|额外秒数|新增种子|其中本模式概率可行|仅资源/带可行|近似松弛目标|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for stage in result['sdp_stages']:
        source=stage['source_receipt']
        lines.append(f"|{stage['id']}|{stage['mode']}|{stage['status']}|{fmt(stage.get('wall_seconds'))}|"
            f"{len(stage.get('added_seed_names',[]))}|{stage['added_probability_feasible_count']}|"
            f"{stage['added_probability_infeasible_count']}|{fmt(source.get('raw_sdp_objective_diagnostic'))}|")
    if not result['sdp_stages']:
        lines.append('|本批未使用SDP|—|—|—|—|—|—|—|')
    lines += ['', '|输入|共用SDP准备秒数（只计一次）|本批报价墙钟秒数之和|',
              '|---|---:|---:|']
    for timing in result['case_timing']:
        lines.append(f"|{timing['id']}|{fmt(timing['new_SDP_shared_preparation_seconds'])}|"
                     f"{fmt(timing['new_quote_wall_seconds_sum'])}|")
    lines += ['',
        '## 预测与实际证据', '',
        '下表固定比较历史赢家与历史最强异族，分别列预测功率下K均值、实际功率下K均值、干净AC和含量测评分的增量。'
        '全库最弱优势及对手改变另存机器表，避免只看一个旧对手。', '',
        '|输入|策略|预测ΔL|实际输入K ΔL|干净AC ΔL|量测ΔL|', '|---|---|---:|---:|---:|---:|']
    for case in result['cases']:
        for name, s in case['strategies'].items():
            if name.startswith('oldmulti__') or name in ('no_dfs','anchor'):
                continue
            pair = s.get('historical_fixed_pair')
            if not pair:
                continue
            fields = ('forecast_K_logodds','actual_input_K_logodds','clean_logodds','observed_logodds')
            values = [fmt(pair['increments'][k]['mean']) for k in fields]
            lines.append('|'+case['id']+'|'+name+'|'+'|'.join(values)+'|')
    lines += ['', '## 全母线及实际交付检查', '',
        '|输入|新策略|电压最小/最大 pu|越.95–1.05路径|最大支路A|最大根kVA|平均网损变化kWh|实际带超出路径/点|最大带超出kvar|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in result['operational_summary']:
        lines.append(f"|{row['id']}|{row['policy']}|{fmt(row['voltage_minimum'])}/{fmt(row['voltage_maximum'])}|"
            f"{row['voltage_failed_rows']}/{row['n']}|{fmt(row['maximum_branch_current_A'])}|{fmt(row['maximum_root_S_kVA'])}|"
            f"{fmt(row['active_loss_change_kwh']['mean'])}|{row['delivered_band_failed_rows']}/{row['delivered_band_violated_cells']}|"
            f"{fmt(row['maximum_delivered_band_violation_kvar'])}|")
    lines += ['', '实际交付带超出按原带加浮点舍入误差核算，属于规划范围诊断；物理设备违规单列。'
        '电压为采购后的AC检查，未加进采购规划；200kVA是标幺基准，设备热额定容量缺失。', '',
        '## 跨运行费用界检查', '',
        '仅在冻结历史、全部竞争者、概率式、资源、带宽及数值编码一致时比较。'
        '新可行点若低于旧有效下界，登记旧证书撤回；反向也检查。旧报告原样保留，不称新点击败已证最优。', '',
        '|输入|模式|旧带下界与SDP点冲突|新下界与旧带点冲突|', '|---|---|---|---|']
    for row in result['cross_run_certificates']:
        lines.append(f"|{row['id']}|{row['mode']}|{row['prior_band_bound_against_SDP_point']['conflict']}|"
            f"{row['new_SDP_bound_against_prior_band_point']['conflict']}|")
    if not result['cross_run_certificates']:
        lines.append('|本批无同域SDP对照|—|—|—|')
    lines += ['', '## 已保留的异常与范围', '',
        f"记录的任务失败：{len(result['original_failed_tasks'])}；执行失败：{result['original_execution_failed_rows']}；"
        f"审计错误：{len(result['task_errors'])}；原始评分逐值复算最大误差：{result['maximum_native_SSE_replay_error']:.3g}。",
        '电压和功率守恒来自已保存完整AC状态的独立核验；设备热额定资料仍缺失。'
        '当前P/Q带是训练数据支持的动作范围，不是参数误差概率界，也不保证辨识提高。', '']
    return '\n'.join(lines)


def report(audit_path, out):
    reader = shared.Reader()
    audited = reader.json(audit_path)
    if audited.get('status') != 'passed':
        raise ValueError('A passed independent audit is required')
    protocol = reader.json(audited['source_protocol'])
    root = Path(audited['source_protocol']['path']).parent
    complete = reader.json(root/'complete.json')
    if complete.get('status') != 'complete' or not complete.get('source_unchanged'):
        raise ValueError('Completed source snapshot required')
    entries = reader.json(audited['source_inputs'])
    entrymap = {e['id']:e for e in entries}
    coverage = {c['id']:c for c in audited['input_coverage']}
    if set(entrymap) != set(coverage):
        raise ValueError('Input and coverage denominators differ')
    variant = protocol['variant']
    if variant not in ('band','band_sdp'):
        raise ValueError('Unrecognized band experiment')
    post = shared.Postprocess(reader, entrymap, coverage, root, baseline_only=True)
    groups = [('grid','baseline_records'),('unbanded','single_baseline_records'),
              ('band','band_baseline_records'),(variant,'records')]
    enriched, counts = [], {}
    for origin, field in groups:
        records = reader.lines(audited[field]) if audited.get(field) else []
        counts[field] = len(records)
        for row in records:
            if row['audit_status'] not in ('passed','recorded_failure'):
                raise ValueError('An unaudited execution entered the report')
            value = post.enrich(row, 'new_execution' if field == 'records' else 'reused_'+origin)
            value['original_policy'] = value['strategy']
            value['strategy'] = rename_policy(value['strategy'], origin)
            value['planning_band'] = row.get('planning_band')
            value['delivered_band'] = row.get('delivered_band')
            enriched.append(value)
    if len({shared.key(r) for r in enriched}) != len(enriched):
        raise ValueError('Duplicate policy/paired execution')
    modes = tuple(protocol['modes'])
    expected = len(entries)*len(modes)*protocol['repeats']
    if counts['records'] != expected or audited['audited_rows'] != expected:
        raise ValueError('New execution denominator differs')
    cases = []
    for tag, entry in entrymap.items():
        rows = [r for r in enriched if r['id'] == tag]
        names = sorted({r['strategy'] for r in rows})
        summaries = {name:single.summarize([r for r in rows if r['strategy'] == name]) for name in names}
        comparisons = {'no_dfs':{}, 'single_without_band':{}}
        if variant == 'band_sdp':
            comparisons['band_without_sdp'] = {}
        for mode in modes:
            name = variant+'__'+mode
            comparisons['no_dfs'][name] = shared.paired(rows, name, 'no_dfs')
            comparisons['single_without_band'][name] = shared.paired(rows, name, 'unbanded__'+mode)
            if variant == 'band_sdp':
                comparisons['band_without_sdp'][name] = shared.paired(rows, name, 'band__'+mode)
        cases.append(dict(id=tag, dimensions={k:entry.get(k) for k in
            ('case','start_date','stride','meter_multiplier','training_rows','validation_rows')},
            coverage=coverage[tag], strategies=summaries, comparisons=comparisons))
    quotes, timing = single.quote_rows(reader, reader.lines(audited['quotes']), entrymap, root)
    stages=[]
    for item in audited.get('sdp_checks',[]):
        if 'source' in item:
            source=reader.json(item['source'])
            seeds=[r for r in audited.get('sdp_checks',[]) if r.get('id')==item['id']
                and r.get('mode')==item['mode'] and 'seed' in r]
            if {r['seed'] for r in seeds}!=set(item.get('added_seed_names',[])):
                raise ValueError('Unique added SDP seed denominator differs from independent seed audits')
            stages.append(dict(item,source_receipt=source,
                added_probability_feasible_count=sum(r['probability_feasible'] for r in seeds),
                added_probability_infeasible_count=sum(not r['probability_feasible'] for r in seeds)))
    for item in timing:
        item['new_SDP_shared_preparation_seconds']=sum(float(s.get('wall_seconds') or 0.) for s in stages if s['id']==item['id'])
        item['SDP_time_counting']='Once per input across both exponential seed generators; additional to all QCQP quote times'
    raw_new=reader.lines(audited['records'])
    zero={(r['id'],r['repeat']):r for r in reader.lines(audited['baseline_records']) if r['strategy']=='no_dfs'}
    operation=[]
    for tag,entry in entrymap.items():
        for mode in modes:
            policy='single_'+mode
            all_rows=[r for r in raw_new if r['id']==tag and r['strategy']==policy]
            good=[r for r in all_rows if r['audit_status']=='passed']
            operation.append(dict(id=tag,policy=policy,n=len(all_rows),completed=len(good),
                voltage_minimum=min((r['electrical']['all_bus_voltage']['minimum'] for r in good),default=None),
                voltage_maximum=max((r['electrical']['all_bus_voltage']['maximum'] for r in good),default=None),
                voltage_failed_rows=sum(not r['voltage_screen_passed'] for r in good),
                maximum_branch_current_A=max((r['electrical']['maximum_branch_current_A'] for r in good),default=None),
                maximum_root_S_kVA=max((r['electrical']['root_S_kVA']['maximum'] for r in good),default=None),
                active_loss_change_kwh=shared.stats([r['electrical']['active_loss_kwh']-
                    zero[(tag,r['repeat'])]['electrical']['active_loss_kwh'] for r in good]),
                delivered_band_failed_rows=sum(not r['delivered_band']['passed'] for r in good),
                delivered_band_violated_cells=sum(r['delivered_band']['violated_cells'] for r in good),
                maximum_delivered_band_violation_kvar=max((r['delivered_band']['maximum_violation_pu']*entry['base_kw'] for r in good),default=None),
                physical_requested_failed_rows=sum(any(v>1e-7 for v in r['requested_violations'].values()) for r in good),
                physical_delivered_failed_rows=sum(any(v>1e-7 for v in r['delivered_violations'].values()) for r in good)))
    reader.checked(__file__)
    if not reader.unchanged():
        raise ValueError('Inputs changed during reporting')
    result = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        source_audit=io.ref(audit_path), source_protocol=audited['source_protocol'], source_inputs=audited['source_inputs'],
        variant=variant, smoke=protocol.get('smoke',False), repeats=protocol['repeats'], modes=list(modes),
        cases=cases, path_records=enriched, new_rows=counts['records'],
        grid_baseline_rows=counts['baseline_records'], single_baseline_rows=counts['single_baseline_records'],
        band_baseline_rows=counts['band_baseline_records'], new_ac_rows=complete.get('new_ac_rows'),
        reused_ac_rows=complete.get('reused_ac_rows'), quotes=single.quotation_summary(quotes), case_timing=timing,
        original_failed_tasks=audited.get('original_failed_tasks',[]), task_errors=audited.get('task_errors',[]),
        original_execution_failed_rows=audited.get('original_execution_failed_rows',0),
        band_checks=audited.get('band_checks'), sdp_checks=audited.get('sdp_checks'),sdp_stages=stages,
        operational_summary=operation,cross_run_certificates=audited.get('cross_run_certificates',[]),
        maximum_native_SSE_replay_error=post.maximum_error, sources_unchanged=True,
        statistical_scope='Three developer-selected inputs from two known physical backgrounds; reused paired paths, not independent confirmation',
        bound_scope='Original single-target plus node-hour band activation objective; numerical SDP objective not a certified QCQP lower bound',
        input_refs=[dict(path=p,sha256=h) for p,h in reader.refs.items()])
    out = Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    io.write(out/'report.json',result)
    (out/'REPORT.md').write_text(markdown(result),encoding='utf8')
    rows = []
    for case in cases:
        for policy, s in case['strategies'].items():
            rows.append(dict(id=case['id'],policy=policy,n=s['n'],internal_recovery=s['recovered'],
                full_support_recovery=s['full_support_recovered'],wrong_high=s['wrong_high'],
                mean_p_true=s['p_true']['mean'],activation_yuan=s['settled_activation_yuan']['mean'],
                availability_yuan=s['settled_availability_yuan']['mean'],total_yuan=s['settled_total_yuan']['mean'],
                delivered_P_kwh=s['quantities']['delivered_P_kwh']['mean'],delivered_Q_kvarh=s['quantities']['delivered_Q_kvarh']['mean']))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    io.write(out/'complete.json',dict(status='complete',report=io.ref(out/'report.json'),markdown=io.ref(out/'REPORT.md')))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();r=report(args.audit,args.out)
    print({k:r[k] for k in ('status','variant','new_rows','maximum_native_SSE_replay_error')})
