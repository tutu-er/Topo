"""Aggregate results only; private observations, fitted scales and actions stay local."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,sha,utc
from ..posterior_mse_20261112.deliver import write_csv
from ..linear_ac_population_20261113.analyze import load_phase,summary,case_summaries
from ..bidirectional_price_dr_20261113.report import solver_rows
from .effects import prior_effects
from .study import SCENARIOS,OLD


def pricing_summary(records):
    groups=defaultdict(list)
    for r in records:
        role='qualified_quote' if r['original_quote_qualified'] else 'zero_fallback'
        groups['all',r['allow_shed'],role].append(r);groups[r['scenario'],r['allow_shed'],role].append(r)
    output=[]
    for (scenario,shed,role),rows in sorted(groups.items()):
        good=[r for r in rows if r['feasible'] and r['baseline']['feasible']]
        row=dict(scenario=scenario,allow_shed=shed,role=role,checks=len(rows),implemented=len(good),
            proven_infeasible=sum(r.get('infeasibility_proved',False) for r in rows),
            unresolved=sum(not r['feasible'] and not r.get('infeasibility_proved',False) for r in rows),
            mean_incremental_yuan=None if not good else float(np.mean([r['incremental_yuan'] for r in good])),
            minimum_incremental_yuan=None if not good else min(r['incremental_yuan'] for r in good),
            maximum_incremental_yuan=None if not good else max(r['incremental_yuan'] for r in good),
            negative_increment_count=sum(r['incremental_yuan']<0 for r in good),
            shedding_used=sum(r['components']['shed_MWh']>1e-9 for r in good))
        for component in ('energy_bill_yuan','generation_bid_yuan','transfer_compensation_yuan','shedding_loss_yuan'):
            row['mean_change_'+component]=None if not good else float(np.mean([r['components'][component]-r['baseline']['components'][component] for r in good]))
        for factor in (.5,1.,1.5):
            row[f'mean_increment_wholesale_x{factor:g}']=None if not good else float(np.mean([r['wholesale_price_sensitivity_incremental_yuan'][str(factor)] for r in good]))
        output.append(row)
    return output


def publish(run,longrun,polish,prices,ed_polish,destination):
    require(not destination.exists(),'Preserve prior public results');cfg=io.read(run/'protocol.json')
    audits={p:io.read(run/f'{p}_guard_audit.json') for p in ('pilot','population','confirmation')}
    for p,a in audits.items():require(a['status']=='passed','Require complete independent '+p+' audit')
    optimization=io.read(run/'optimization_audit.json');require(optimization['status']=='passed','Optimization replay incomplete')
    for root in (longrun,polish,prices,ed_polish):require(io.read(root/'complete.json')['status']=='passed','Incomplete auxiliary stage '+str(root))
    qual=io.read(run/'qualification.json');aggregates=[];cases=[];solvers=[]
    for scenario in SCENARIOS:
        for phase in ('population','confirmation'):
            rows=load_phase(run/scenario,phase);groups={'all':rows}
            if phase=='population':
                for name,ids in qual['groups'].items():groups[name]=[r for r in rows if r['id'] in ids]
                for correct in (True,False):groups['target_correct' if correct else 'target_wrong']=[r for r in rows if r.get('truth_covered') and r['historical_target_correct']==correct]
            for name,data in groups.items():
                for r in summary(data,name,phase):r.update(scenario=scenario);r.pop('mean_DR_kwh',None);aggregates.append(r)
            for r in case_summaries(rows,phase):
                data=[v for v in rows if v['id']==r['id'] and v['strategy']=='optimized_'+r['gamma_label']]
                r.update(scenario=scenario,planning_failures=sum(v['planning_failed'] for v in data),
                    global_gap_closed=sum(v.get('global_optimality_verified') is True for v in data));r.pop('mean_DR_kwh',None);cases.append(r)
        for phase in ('pilot','population','confirmation'):solvers.extend(solver_rows(run/scenario,phase))
    effects=prior_effects(OLD);native=io.read(longrun/'summary.json');polished=io.read(polish/'summary.json');polmap={r['key']:r for r in polished}
    cost_groups=defaultdict(list)
    for r in native:cost_groups[r['id'],r['alpha'],r['gamma']].append((r,polmap[r['key']]))
    cost_rows=[]
    for (identity,alpha,gamma),pairs in sorted(cost_groups.items()):
        cost_rows.append(dict(id=identity,alpha=alpha,gamma=gamma,problems=len(pairs),
            old_mean_fee=float(np.mean([r['old_cost'] for r,p in pairs])),native_mean_fee=float(np.mean([r['new_cost'] for r,p in pairs])),
            polished_mean_fee=float(np.mean([p['cost'] for r,p in pairs])),
            native_mean_relative_improvement=float(np.mean([r['relative_cost_improvement'] for r,p in pairs])),
            polished_min_relative_improvement=min(p['original_to_polished_improvement'] for r,p in pairs),
            polished_max_relative_improvement=max(p['original_to_polished_improvement'] for r,p in pairs),
            final_min_relative_gap=min(p['relative_gap'] for r,p in pairs),final_max_relative_gap=max(p['relative_gap'] for r,p in pairs),
            global_gap_closed=sum(p['global_gap_closed'] for r,p in pairs),
            mean_additional_native_seconds=float(np.mean([r['new_call_seconds'] for r,p in pairs])),
            mean_additional_polish_seconds=float(np.mean([p['seconds'] for r,p in pairs]))))
    fixed=io.read(prices/'fixed_plan_results.json');price_rows=pricing_summary(fixed)
    improved_ED={(r['id'],r['scenario']):r for r in io.read(ed_polish/'summary.json')};joint=[]
    for r in io.read(prices/'joint_results.json'):
        row=dict(id=r['id'],scenario=r['scenario'],status=r['status'],feasible=r['feasible'],native_status=r['native_status'])
        if r['feasible']:
            p=improved_ED[r['id'],r['scenario']]
            row.update(native_premium_upper_yuan=r['information_premium_yuan'],feasible_premium_upper_yuan=p['premium_yuan'],
                premium_lower_yuan=p['premium_lower_bound_yuan'],premium_relative_gap=p['premium_relative_gap'],
                extra_convex_seconds=p['seconds'],cost_optimality_proved=p['premium_relative_gap']<=1e-5,
                retained_initial_ED=r.get('source')=='verified_ED_initial_point_retained')
        joint.append(row)
    destination.mkdir(parents=True)
    write_csv(destination/'overall_and_groups.csv',aggregates);write_csv(destination/'cases.csv',cases)
    write_csv(destination/'where_it_helps.csv',effects);write_csv(destination/'longer_solve_and_QP.csv',cost_rows)
    write_csv(destination/'price_accounting.csv',price_rows);write_csv(destination/'joint_ED_quotes.csv',joint)
    io.write(destination/'solver_summary.json',solvers)
    lookup={(r['phase'],r['scenario'],r['group'],r['strategy']):r for r in aggregates}
    lines=['# 配网灵活性：偏差、求解与价格核算','',
        '本轮重点是有限候选库中真候选已有、但历史证据仍不足的情形。拓扑、线路拟合参数、历史SSE、共同sigma²和最终完整库评分保持冻结；新增历史残差检查仅影响规划。', '',
        '## 哪些情形更有效','',
        '先对上一轮400输入、1600新日、30%容量、gamma=2的完整记录分组。真值只用于事后分组。下表历史权重是旧验证数据给真候选的权重；两列均值则包含完整历史和无调节/调节新日观测。', '',
        '|历史真候选权重|输入/新日|无调节→灵活性调节平均真后验|完整恢复|90%达标|',
        '|---|---:|---:|---:|---:|']
    for label in ('<50%','50%-90%','90%-99%','>=99%'):
        r=next(v for v in effects if v['group']=='historical_true_weight' and v['stratum']==label)
        lines.append(f"|{label}|{r['inputs']}/{r['days']}|{r['baseline_mean']:.2%}→{r['optimized_mean']:.2%}|{r['baseline_recovered']}→{r['optimized_recovered']}|{r['baseline_pass90']}→{r['optimized_pass90']}|")
    lines+=['', '收益主要集中在原有证据不足组。历史真权重至少99%的122个输入、488日，完整恢复均488次，均值只增加约0.024个百分点。历史真权重不足50%的10个输入、40日，平均真后验增加17.86个百分点，恢复21→29，90%达标10→23；其中9次挽回原先错误选择、1次使原先正确选择变错，均保留。', '',
        '## 简单历史残差修正','',
        '在原32点历史验证数据上，为每个候选、每个电压输出估计一个比例c：c=1+sum(mu*(Y-mu))/(sum(mu²)+32*sigma²)。mu为原冻结候选的预测。原评分不使用c，只在规划中用c*mu估计历史均值偏差。', '',
        '对每一对规划候选，保留原名义平均对数赔率增量至少gamma，并分别检查两端修正均值下各自的证据增量不为负。所有新增条件都是二次约束，整体仍为QCQP。这里使用少量比例估计，没有加入协方差模型。该近似是否能代表调节方向上的真实偏差，由原真实AC模拟检验。', '',
        '新批选择全部28个含真候选且历史真权重不足99%的输入，加4个容易输入和4个缺真候选对照；每输入16个独立新日，共576个背景。四组共享背景、5%预测误差和表计流，独立背景数不乘4。每计划0.8秒。下表主组28输入×16=448日，原历史最高权重候选仍为操作目标，没有自动换成真拓扑。', '',
        '|容量/方法|平均真后验|完整恢复/448|90%达标|错误唯一|未报价|平均归一化费用|',
        '|---|---:|---:|---:|---:|---:|---:|']
    b=lookup['population','nominal30','evidence_gap','no_dr']
    lines.append(f"|无调节|{b['mean_true_posterior']:.2%}|{b['full_recoveries']}|{b['true_pass90']}|{b['wrong_unique']}|0|0|")
    for scenario in SCENARIOS:
        r=lookup['population',scenario,'evidence_gap','optimized_g2']
        lines.append(f"|{scenario}|{r['mean_true_posterior']:.2%}|{r['full_recoveries']}|{r['true_pass90']}|{r['wrong_unique']}|{r['planning_failures']}|{r['mean_fee']:.6g}|")
    lines+=['', '30%容量下，增加残差检查后错误唯一17→12；50%容量下20→13。平均后验分别增加约0.31、0.43个百分点。该比较包含全部未报价时的零动作；30%检查组多6次未报价。新增约束缩小可行域，不会让相同问题的全局最小费用下降；表中的可行费用还受未报价和未闭合的优化间隙影响。', '',
        '## 预锁四例的独立32日复验','',
        'gamma=4。各格为平均真候选后验；全部32日作为恢复和失败分母。容量30%下039均无报价，等于无调节，不解释成修正了参数偏差。', '',
        '|输入|无调节|原30%|检查30%|原50%|检查50%|',
        '|---|---:|---:|---:|---:|---:|']
    for identity in cfg['confirmation']['ids']:
        values=[next(r for r in cases if r['phase']=='confirmation' and r['scenario']==s and r['id']==identity) for s in SCENARIOS]
        lines.append('|'+identity+'|'+f"{values[0]['baseline_mean']:.2%}"+'|'+'|'.join(f"{r['optimized_mean']:.2%}" for r in values)+'|')
    lines+=['', '下降例039在无调节下平均真后验45.36%、完整恢复14/32；50%原方案29.87%、10/32，增加残差检查后35.76%、12/32。下降减轻但仍存在。每个输出的单比例不能完整表示节点之间的方向性响应误差。三正例的辨识收益基本保留，全部案例、目标正确/错误组与同费轮换在CSV中。', '',
        '## 延长时间后，费用还有多大空间','',
        '同一旧预测、同一名义QCQP、同一费用与容量，保留原已核可行点及有效下界。12个问题依次给2、10秒；三个剩余大间隙问题再给30秒。27次原生调用均到时间上限，未闭合费用界。两例实际改进（75.78%、2.80%），另10例保留旧点。', '',
        '再从长时间求解后的可行点出发，最多20次、总预算1秒，迭代解凸QP：用每个凸二次增量在当前点的切线作为充分条件。每一步保持原约束可行且费用不增加。这是局部可行改进；QP下界未作为原问题下界。', '',
        '|输入/容量|问题数|长时间原生平均降费|加凸QP后降费范围|最终费用间隙范围|全局费用界闭合|',
        '|---|---:|---:|---:|---:|---:|']
    for r in cost_rows:lines.append(f"|{r['id']}/{r['alpha']:.0%}|{r['problems']}|{r['native_mean_relative_improvement']:.2%}|{r['polished_min_relative_improvement']:.2%}～{r['polished_max_relative_improvement']:.2%}|{r['final_min_relative_gap']:.3%}～{r['final_max_relative_gap']:.3%}|{r['global_gap_closed']}|")
    improved=sum(r['original_to_polished_improvement']>1e-6 for r in polished);close=sum(r['relative_gap']<.02 for r in polished)
    lines+=['', f'加入可行凸QP后{improved}/12个问题降费，最大78.10%；{close}/12个最终费用间隙小于2%，0/12达到1e-5相对闭合标准。仍有一例约48.58%的间隙，费用空间尚未证明。建议求解流程为“解析点/可行初值 → 可行凸QP改善 → 原非凸QCQP补界”，并分别报告三者的耗时与贡献。此改进仅核旧费用问题，未把重规划叫作新的独立AC测试。', '',
        '## 名称、ED与示例价格','',
        '建议名称为“面向拓扑辨识的配网灵活性调度”；讨论合同与报价时称“配网灵活性服务采购”。发电再调度、负荷时移和储能均可供给灵活性。[National Grid 2026/27采购说明](https://dso.nationalgrid.co.uk/downloads/15141/nged-dso-distribution-flexibility-services-procurement-statement-2026-27.pdf)已有Distribution Flexibility Services及日前采购安排；[SSEN](https://www.ssen.co.uk/about-ssen/dso/flexibility/flexibility-services/)区分可用容量和实际调用支付。', '',
        '能源购售电与灵活性服务支付分别核算。研究运营主体可以是拥有采购/调度授权的园区或配网管理者；受监管DSO、发售电主体和交易机构的职责按相应制度确定。[2024电力市场运行基本规则](https://zfxxgk.ndrc.gov.cn/wap/iteminfo.jsp?id=20379)分别列明这些成员与交易、安全校核职责。[山东2026年规则](https://sdb.nea.gov.cn/dtyw/tzgg/202605/P020260508675725961073.pdf)包含SCUC/SCED安排。本轮ED是固定启停的连续资源核算，不将实际地区规则全部搬入。', '',
        '示例外部电价300/500/800元/MWh，售电按对应买电价90%；本地机组报价350/450/550元/MWh，另加凸二次调节项；负荷时移补偿100元/MWh，切负荷损失10000元/MWh。全部是用户允许的示例，未按某地区实际市场校准。资源也为声明场景：每节点固定开机的聚合电源0.06～0.60倍历史峰值、基线0.33、每15分钟爬坡0.15；负荷时移±0.15倍历史峰值且日能量守恒；可选切负荷最多毛负荷5%。', '',
        '固定启停、能量平衡、出力/爬坡、购售电和负荷时移构成凸QP；再加入辨识的二次约束，仍为QCQP。价格程序只读预测、历史功率尺度和名义base MVA，不读真实线路阻抗或实际未来P/Q/V。这里核有功资源可行性，原AC测试另核电压；未加入热稳定和单设备无功能力。', '',
        '|允许切负荷|合格规划报价|ED可实现|已证不可实现|仍未决|实际切负荷方案|平均增量费用（可实现报价）|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in price_rows:
        if r['scenario']=='all' and r['role']=='qualified_quote':lines.append(f"|{r['allow_shed']}|{r['checks']}|{r['implemented']}|{r['proven_infeasible']}|{r['unresolved']}|{r['shedding_used']}|{r['mean_incremental_yuan']:.3f}元|")
    lines+=['', '没有报价的26个零动作对照另列，未用它们抬高资源可实现率。固定计划价格敏感性只改变购售电账单，资源报价不变，未冒称在新电价下重新优化。能量节省可能使相对原净计划的增量费用为负。', '',
        '另锁四诊断例的第一个新日、两种容量，共8个联合ED+辨识问题。不切负荷，先解相同资源域、没有辨识约束的经济最优基线；联合问题相对它的费用为“信息增量费用”。它的基线是原域下界，局部切线QP的界不用于此处。6例得到可行报价，039两容量在声明ED资源下已证不可行；6个可行点再做最多2秒的可行凸ED-QP改善。', '',
        '|输入/容量|原生可行信息费|可行凸步骤后上界|原问题下界|间隙|',
        '|---|---:|---:|---:|---:|']
    for r in joint:
        if r['feasible']:lines.append(f"|{r['id']}/{r['scenario']}|{r['native_premium_upper_yuan']:.3f}元|{r['feasible_premium_upper_yuan']:.3f}元|{r['premium_lower_yuan']:.3f}元|{r['premium_relative_gap']:.1%}|")
        else:lines.append(f"|{r['id']}/{r['scenario']}|已证不可行|—|—|—|")
    lines+=['', '这些是示例报价和声明资源条件下的可行费用上界，六例最小信息费用均未证明；不能把1～6元的可行报价写成最小报价。联合经济调度计划只核名义辨识约束和资源可行性，没有冒充已完成新的实际AC恢复率测试。', '',
        '## 核验和保留的开发失败','',
        '主批与复验全部观测独立复算预测、交付、表计、完整支持后验和选择；全部规划点核原增量/容量，新增方向约束用独立SSE公式核对。对预定已有路径重放AC，四策略组逐日核背景及噪声一致。费用检查另重算39个原QCQP点、20个ED点和544条固定价格记录；凸子问题界没有充当原问题界。', '',
        '开发阶段另保留：凸QP改进v1的12条参数设置失败（1e-10最优性容差低于求解器允许值，尚未进入求解），改为允许的1e-9后用同一12问题核验；价格v1的3条联合费用复算检查失败，原因是超时点的绝对值辅助变量尚未收紧，检查误将辅助变量松弛当作物理费用不一致。v2验证费用差正好来自该松弛，再收紧辅助变量，原物理/价格/辨识约束均不变。544条已完成固定计划核算复用并保留来源，8个联合示例重新核验通过。没有重抽新日或覆盖失败目录。', '',
        '所有原始P/Q/V、校正系数、拟合矩阵、真实网络和动作/报价轨迹留在忽略的outputs。新物理测试704个独立背景，另4个接口预跑；优化重算和验证重放均不增加独立样本。']
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    io.write(destination/'VALIDATION.json',dict(status='passed',created_utc=utc(),unit_checks=8,
        physical_audits={p:{k:v for k,v in a.items() if k not in ('independent_physical','source')} for p,a in audits.items()},
        optimization={k:v for k,v in optimization.items() if k!='source'},candidate_R_X_refits=0,historical_output_calibration_estimated=True,
        new_independent_AC_backgrounds=704,raw_private_data_published=False))
    io.write(destination/'RECEIPT.json',dict(created_utc=utc(),protocol_sha256=sha(run/'protocol.json'),
        additional_protocols={name:sha(root/'protocol.json') for name,root in (('long',longrun),('polish',polish),('prices',prices),('ED_polish',ed_polish))},
        files={p.name:sha(p) for p in destination.iterdir() if p.is_file()},source=io.ref(__file__)))
    io.write(run/'complete.json',dict(status='complete',created_utc=utc(),public_report=str(destination/'REPORT.md'),source=io.ref(__file__)))
    print(dict(public=str(destination),independent_AC_backgrounds=704,guard_main_plans=audits['population']['plans'],long_problems=12),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','longrun','polish','prices','ed-polish','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();publish(a.run.resolve(),a.longrun.resolve(),a.polish.resolve(),a.prices.resolve(),a.ed_polish.resolve(),a.public.resolve())
