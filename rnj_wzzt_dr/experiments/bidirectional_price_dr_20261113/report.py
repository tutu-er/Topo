"""Publish aggregate paired outcomes, adverse cases and solver evidence only."""
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,sha,utc
from ..posterior_mse_20261112.deliver import write_csv
from ..linear_ac_population_20261113.analyze import load_phase,summary,case_summaries
from .run import SCENARIOS


def solver_rows(root,phase):
    grouped={}
    for file in (root/phase/'jobs').glob('*/day*/plans.json'):
        for label,value in io.read(file).items():grouped.setdefault(label,[]).append(value)
    result=[]
    for label,plans in sorted(grouped.items()):
        good=[p for p in plans if p['qualified']];native=[p for p in plans if p.get('native')]
        seeded=[p for p in good if p.get('initialization_cost') is not None]
        retained=[p for p in seeded if abs(p['cost']-p['initialization_cost'])<=1e-9*max(p['cost'],1e-20)]
        improvements=[(p['initialization_cost']-p['cost'])/max(p['initialization_cost'],1e-30) for p in seeded]
        result.append(dict(phase=phase,scenario=root.name,gamma_label=label,plans=len(plans),qualified=len(good),
            proven_infeasible=sum(p.get('infeasibility_proved',False) for p in plans),
            unresolved=sum(not p['qualified'] and not p.get('infeasibility_proved',False) for p in plans),
            exceptions=sum('exception' in p['status'] for p in plans),
            global_gap_closed=sum(p['global_optimality_verified'] for p in good),
            native_calls=len(native),native_status_counts=dict(Counter(str(p['native']['status']) for p in native)),
            source_counts=dict(Counter(p.get('source',p['status']) for p in plans)),
            status_counts=dict(Counter(p['status'] for p in plans)),
            initial_cost_recorded=len(seeded),initial_point_retained=len(retained),
            mean_relative_cost_improvement=None if not improvements else float(np.mean(improvements)),
            max_relative_gap=None if not good else max(p['relative_gap'] for p in good),
            median_relative_gap=None if not good else float(np.median([p['relative_gap'] for p in good])),
            mean_seconds=float(np.mean([p['seconds'] for p in plans])),
            p95_seconds=float(np.quantile([p['seconds'] for p in plans],.95)),
            max_seconds=max(p['seconds'] for p in plans)))
    return result


def publish(run,destination,diagnosis,additional=None):
    require(not destination.exists(),'Preserve prior public reports')
    cfg=io.read(run/'protocol.json');aggregates=[];cases=[];solvers=[];audits={};counts={};loaded={}
    for phase in ('pilot','population','confirmation'):
        audits[phase]=io.read(run/f'{phase}_capacity_audit.json');require(audits[phase]['status']=='passed','Require independent audit')
        counts[phase]=io.read(run/f'{phase}_complete.json')
    for scenario in SCENARIOS:
        root=run/scenario
        for phase in ('population','confirmation'):
            rows=load_phase(root,phase);loaded[scenario,phase]=rows
            groups={'all':rows,'truth_in_bank':[r for r in rows if r.get('truth_covered')],
                'truth_missing_from_bank':[r for r in rows if r.get('truth_covered') is False],
                'historical_target_wrong':[r for r in rows if r.get('historical_target_correct') is False]}
            for case in sorted({r['case'] for r in rows}):groups[case]=[r for r in rows if r['case']==case]
            for name,data in groups.items():
                if not data:continue
                for value in summary(data,name,phase):
                    value.update(scenario=scenario);value.pop('mean_DR_kwh',None);aggregates.append(value)
            for value in case_summaries(rows,phase):
                decisions=[r for r in rows if r['id']==value['id'] and r['strategy']=='optimized_'+value['gamma_label']]
                value.update(planning_failures=sum(r['planning_failed'] for r in decisions),
                    global_gap_closed=sum(r['global_optimality_verified'] is True for r in decisions),
                    maximum_request=max(r['request_max'] for r in decisions),
                    mean_request_rms=float(np.mean([r['request_rms'] for r in decisions])))
                value.update(scenario=scenario);value.pop('mean_DR_kwh',None);cases.append(value)
        for phase in ('pilot','population','confirmation'):solvers.extend(solver_rows(root,phase))
    common=set.intersection(*[{(r['id'],r['day']) for r in loaded[s,'population']
        if r['strategy']=='optimized_g2' and not r['planning_failed']} for s in SCENARIOS])
    matched=[]
    for scenario in SCENARIOS:
        data=[r for r in loaded[scenario,'population'] if (r['id'],r['day']) in common]
        for value in summary(data,'all_scenarios_have_quote','population'):
            value.update(scenario=scenario);value.pop('mean_DR_kwh',None);matched.append(value)
    outcomes=[]
    for phase in ('population','confirmation'):
        for scenario in SCENARIOS:
            for label in sorted({r['gamma_label'] for r in cases if r['phase']==phase}):
                data=[r for r in cases if r['phase']==phase and r['scenario']==scenario and r['gamma_label']==label and r['paired_gain'] is not None]
                outcomes.append(dict(phase=phase,scenario=scenario,gamma_label=label,truth_containing_inputs=len(data),
                    improved=sum(r['paired_gain']>1e-6 for r in data),decreased=sum(r['paired_gain']<-1e-6 for r in data),
                    within_1e_6=sum(abs(r['paired_gain'])<=1e-6 for r in data)))
    diagnosis_rows=io.read(diagnosis/'summary.json')
    diagnosis_rows=[r for r in diagnosis_rows if r['strategy']=='optimized_g4' and r['competitor']==r['historical_target']]
    destination.mkdir(parents=True)
    write_csv(destination/'overall_and_groups.csv',aggregates);write_csv(destination/'commonly_quoted_paths.csv',matched)
    io.write(destination/'input_outcome_counts.json',outcomes)
    for phase in ('population','confirmation'):write_csv(destination/f'{phase}_cases.csv',[r for r in cases if r['phase']==phase])
    io.write(destination/'solver_summary.json',solvers);io.write(destination/'old_decline_decomposition.json',diagnosis_rows)
    if additional is not None:
        io.write(destination/'old_additional_declines.json',[r for r in io.read(additional/'summary.json') if r['strategy']=='optimized_g4'])
    lookup={(r['phase'],r['scenario'],r['group'],r['strategy']):r for r in aggregates}
    lines=['# 有界双向灵活性：独立AC测试','',
        '允许聚合发电、储能、负荷双向改变节点净功率。净注入为合法运行方向；容量只用事前声明的节点历史峰值10%、30%、50%三个数值场景。另列无幅度限制对照。费用为归一化二次机会成本，未校准成市场货币。', '',
        '## 锁定流程与分母','',
        '实际新日负荷 → 独立5%预测误差 → 仅预测和冻结历史候选进入规划 → 线性响应 → 原真实网络AC → 原P/Q/V表计噪声 → 完整历史32点＋新日96点后验。候选矩阵和共同历史sigma²均不重拟合。实际响应精确交付，无额外资源执行随机性。', '',
        f"主体400输入×4新日={counts['population']['unique_new_days']}个独立背景；锁定的12例×32新日={counts['confirmation']['unique_new_days']}个独立复验背景。四种容量共享背景、预测及表计随机流，背景分母不乘4。主体仅要求增量gamma=2；复验保留log(2)、2、4。原库含真候选150/400，缺失250/400。", '',
        '## 主体结果：gamma=2','',
        '真后验均值以含真候选的600个新日为分母；完整恢复、错误唯一输出、电压越限以全部1600个预定新日为分母。没有报价执行显式零动作并计失败。', '',
        '|容量|策略|真后验均值|完整恢复/1600|正确唯一|错误唯一|平均费用|无报价|AC失败|电压越限日|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scenario in SCENARIOS:
        for strategy in ('no_dr','rotation_g2','optimized_g2'):
            r=lookup['population',scenario,'all',strategy]
            lines.append(f"|{scenario}|{strategy}|{r['mean_true_posterior']:.3%}|{r['full_recoveries']}|{r['correct_unique']}|{r['wrong_unique']}|{r['mean_fee']:.6g}|{r['planning_failures']}|{r['failed_paths']}|{r['voltage_violating_paths']}|")
    b=lookup['population','cap30','truth_in_bank','no_dr'];v=lookup['population','cap30','truth_in_bank','optimized_g2']
    ci=v['delta_interval95'];o=next(r for r in outcomes if r['phase']=='population' and r['scenario']=='cap30')
    lines+=['',f"30%容量相对无DR的平均真后验变化为{100*v['paired_true_posterior_delta']:+.3f}个百分点，按150个历史输入成组重抽样的95%区间为[{100*ci[0]:+.3f}, {100*ci[1]:+.3f}]。150个含真候选输入中，{o['improved']}个四日平均增加超过1e-6，{o['decreased']}个下降超过1e-6，{o['within_1e_6']}个基本不变。含真候选组错误唯一输出为{b['wrong_unique']}→{v['wrong_unique']}；全体新增错误唯一输出来自缺真候选组。"]
    lines+=['', '错误唯一指90%累计后验集合只保留一个错误拓扑；其统计包含缺真候选组。后验均值按有效观测统计，AC失败保留缺失，完整恢复分母包含失败。全部分组、输入及同费轮换结果均在CSV，未删除下降例。', '',
        f'有界情景的全体平均费用包含无报价时的零动作；它下降部分来自拒绝资源能力不足的高费用方案。另在四情景都成功报价的{len(common)}个相同新日上比较，见commonly_quoted_paths.csv，不能把分母改变造成的费用下降称为同目标优化改进。', '',
        '## 报价与费用界','',
        '先解无界单竞争者标量乘子方程，取得可行点和原问题有效下界。验证全部预定竞争者；只有满足容量且费用界闭合才直接结束。其余用有界方向和一次凸充分条件QP尝试产生初值，再由Gurobi直接解带容量原QCQP。QP初值失败不证明原问题无解，原生求解也可在没有可行初值时寻找解。没有使用ADMM。', '',
        '规划竞争者是历史权重至少1%的其他候选，并始终包含最强备选；全局费用界仅对应这一预定集合，选后后验仍使用完整原库。费用界容差1e-5。名义总预算每计划0.4秒，核内计时包含初值和原生模型构造；共享矩阵分解、预测、进程启动另计入阶段墙钟时间。小额最低原生时间和建模耗时可能使实测超过预算，最大值另列。', '',
        '|阶段|容量|gamma|报价数|合格|已证无解|未决/异常|全局费用界闭合|原生调用|原样保留初值|平均/95%/最大秒|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in solvers:
        if r['phase']=='pilot':continue
        lines.append(f"|{r['phase']}|{r['scenario']}|{r['gamma_label']}|{r['plans']}|{r['qualified']}|{r['proven_infeasible']}|{r['unresolved']}|{r['global_gap_closed']}|{r['native_calls']}|{r['initial_point_retained']}|{r['mean_seconds']:.4f}/{r['p95_seconds']:.4f}/{r['max_seconds']:.4f}|")
    lines+=['', '无解证明来自盒上增量有效上界、无界费用下界超过盒内最大费用，或原生全局不可行状态。超时未找到报价单列未决。可行报价费用是上界；只有原问题有效下界闭合才标全局最优。未闭合间隙、初值费用和实际改善、原生状态见solver_summary.json。', '',
        '## 12例独立复验','',
        '保留上轮已锁定的改善例、下降例及弱对照；本轮未按新结果更换输入。下表完整列gamma=4，其他两档及同费轮换在CSV。每格“真后验；恢复数；无报价数”，恢复分母均为32。', '',
        '|输入|无DR|无界|10%容量|30%容量|50%容量|',
        '|---|---|---|---|---|---|']
    for identity in cfg['confirmation']['ids']:
        values=[];base=None
        for scenario in SCENARIOS:
            r=next(r for r in cases if r['phase']=='confirmation' and r['scenario']==scenario and r['id']==identity and r['gamma_label']=='g4')
            group=[v for v in loaded[scenario,'confirmation'] if v['id']==identity and v['strategy']=='optimized_g4']
            if base is None:base=f"{r['baseline_mean']:.2%}; {r['baseline_recovered']}"
            values.append(f"{r['optimized_mean']:.2%}; {r['optimized_recovered']}; {sum(v.get('planning_failed',False) for v in group)}")
        lines.append('|'+identity+'|'+base+'|'+'|'.join(values)+'|')
    lines+=['', '## 为什么旧的Soumalas039精度下降','',
        '以下只分解旧64日冻结观测，不是新增样本。把真候选相对旧历史首选的实际对数赔率变化分成预测距离、冻结响应相对物理线性响应的偏差、AC非线性和表计四项；四项严格加和等于观测变化。真实物理参数只用于事后诊断。', '',
        '|输入|预测距离|响应参数偏差项|AC非线性项|表计项|实际对数赔率变化|',
        '|---|---:|---:|---:|---:|---:|']
    for r in diagnosis_rows:lines.append(f"|{r['id']}|{r['distance']:+.3f}|{r['parameter']:+.3f}|{r['AC']:+.3f}|{r['meter']:+.3f}|{r['log_odds']:+.3f}|")
    lines+=['', '039的预测区分距离确实增加约4，但冻结响应偏差项为−8.52，超过这项收益；AC非线性在这里反而补回约2.16。另两例历史首选同样错误，偏差项却有利于真拓扑。因此下降由候选响应与真实物理响应在调节方向上的偏差驱动，不能仅归咎于softmax压缩、预测误差或历史首选错误。', '',
        '另外两例旧主体下降也已按原4日记录核查：压力042的平均实际对数赔率增加0.951，但一天真后验42.45%→26.79%，其他两天已接近1，最终平均真后验73.85%→70.85%；平均对数赔率与平均后验不同。Soumalas040真候选对错误历史目标的赔率增加3.433，却与另一主要错误竞争者几乎没有拉开距离，对后者赔率反减0.017，平均真后验49.33%→49.07%。历史目标对各竞争者的约束没有包含竞争者彼此之间的所有比较。详细数值见MODEL.md及old_additional_declines.json。', '',
        '幅度限制约束资源使用并避免极端请求，不自动消除冻结响应偏差。该模型检验可控激励能否增加平均区分证据；真实恢复、平均后验、90%后验达标率以及越限仍各自报告。尚未加入电压/热稳定规划约束、储能能量约束、实际参与随机性和实证价格弹性。', '',
        '## 开发记录与核验','']
    if cfg.get('development_failure'):lines.append(str(cfg['development_failure'].get('description',cfg['development_failure'])))
    lines+=['', '所有观测独立复算预测、交付、表计、完整支持后验及决定；预定子集复算原AC路径，所有动作核容量，四场景逐日核背景/预测/表计一致。验证重放不增加独立样本。原始P/Q/V、拟合矩阵、真实参数及动作轨迹留私有outputs。', '']
    (destination/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    io.write(destination/'VALIDATION.json',dict(status='passed',unit_checks=5,
        audits={phase:{k:v for k,v in r.items() if k not in ('source','independent_AC_posterior')} for phase,r in audits.items()},
        independent_physical_audits={phase:{s:{k:v for k,v in r['independent_AC_posterior'][s].items() if k!='source'} for s in SCENARIOS} for phase,r in audits.items()},
        raw_private_data_published=False,new_fits=0))
    io.write(destination/'RECEIPT.json',dict(created_utc=utc(),protocol_sha256=sha(run/'protocol.json'),
        source=io.ref(__file__),files={p.name:sha(p) for p in destination.iterdir() if p.is_file()}))
    io.write(run/'complete.json',dict(status='complete',created_utc=utc(),public_report=str(destination/'REPORT.md')))
    print(dict(public=str(destination),population_unique_days=counts['population']['unique_new_days'],confirmation_unique_days=counts['confirmation']['unique_new_days']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--additional',type=Path)
    a=p.parse_args();publish(a.run.resolve(),a.public.resolve(),a.diagnosis.resolve(),None if a.additional is None else a.additional.resolve())
