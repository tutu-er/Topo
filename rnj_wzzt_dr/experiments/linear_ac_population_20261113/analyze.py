"""Population and independently confirmed examples; all failures remain visible."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,sha,utc
from ..posterior_mse_20261112.deliver import write_csv


def interval(values,seed=2026111307):
    values=np.asarray(values,float)
    if not len(values):return None
    if len(values)==1:return [float(values[0]),float(values[0])]
    rng=np.random.default_rng(seed);index=rng.integers(0,len(values),(2000,len(values)))
    return np.quantile(values[index].mean(axis=1),[.025,.975]).tolist()


def load_phase(run,phase):
    completion=io.read(run/phase/'complete.json');rows=[]
    for receipt in completion['receipts']:
        require(sha(receipt['result']['path'])==receipt['result']['sha256'],'Completed result changed')
        rows.extend(io.read(receipt['result']['path']))
    return rows


def summary(rows,group,phase):
    baseline={(r['id'],r['day']):r for r in rows if r['strategy']=='no_dr'};result=[]
    for strategy in sorted({r['strategy'] for r in rows}):
        data=[r for r in rows if r['strategy']==strategy];valid=[r for r in data if r['status']=='complete']
        pairs=[(r,baseline[r['id'],r['day']]) for r in valid if baseline[r['id'],r['day']]['status']=='complete']
        probability=[(r,b) for r,b in pairs if r['p_true'] is not None and b['p_true'] is not None]
        cluster=defaultdict(list)
        for r,b in probability:cluster[r['id']].append(r['p_true']-b['p_true'])
        means=[np.mean(v) for v in cluster.values()]
        covered=[r for r in valid if r['p_true'] is not None]
        quoted=[r for r in data if 'cost' in r]
        row=dict(phase=phase,group=group,strategy=strategy,inputs=len({r['id'] for r in data}),planned_paths=len(data),valid_paths=len(valid),
            failed_paths=len(data)-len(valid),planning_failures=sum(r.get('planning_failed',False) for r in data),
            posterior_paths_with_truth=len(covered),mean_true_posterior=None if not covered else float(np.mean([r['p_true'] for r in covered])),
            mean_target_posterior=None if not valid else float(np.mean([r['p_target'] for r in valid])),
            paired_true_posterior_delta=None if not means else float(np.mean(means)),delta_interval95=interval(means),
            full_recoveries=sum(r['map_correct'] for r in valid),full_recovery_rate_all=sum(r['map_correct'] for r in valid)/len(data),
            correct_unique=sum(r['correct_unique'] for r in valid),wrong_unique=sum(r['wrong_unique'] for r in valid),
            true_pass90=sum(r['true_pass90'] for r in valid),mean_retained=None if not valid else float(np.mean([r['retained_count'] for r in valid])),
            mean_missing=None if not valid else float(np.mean([r['missing'] for r in valid])),mean_extra=None if not valid else float(np.mean([r['extra'] for r in valid])),
            mean_fee=None if not quoted else float(np.mean([r['cost'] for r in quoted])),
            median_fee=None if not quoted else float(np.median([r['cost'] for r in quoted])),
            p95_fee=None if not quoted else float(np.quantile([r['cost'] for r in quoted],.95)),
            max_fee=None if not quoted else float(max(r['cost'] for r in quoted)),
            mean_request_rms=None if not quoted else float(np.mean([r['request_rms'] for r in quoted])),
            max_request=None if not quoted else float(max(r['request_max'] for r in quoted)),
            mean_DR_kwh=None if not quoted else float(np.mean([r['absolute_DR_kwh'] for r in quoted])),
            voltage_violating_paths=sum(r['voltage_violation_cells']>0 for r in valid),
            mean_negative_P_fraction=None if not quoted else float(np.mean([r['negative_P_fraction'] for r in quoted])))
        result.append(row)
    return result


def case_summaries(rows,phase):
    output=[]
    for identity in sorted({r['id'] for r in rows}):
        data=[r for r in rows if r['id']==identity];base={r['day']:r for r in data if r['strategy']=='no_dr'}
        for strategy in sorted({r['strategy'] for r in data if r['strategy'].startswith('optimized_')}):
            pairs=[(r,base[r['day']]) for r in data if r['strategy']==strategy and r['status']=='complete' and base[r['day']]['status']=='complete']
            pairs=[(r,b) for r,b in pairs if r['p_true'] is not None and b['p_true'] is not None]
            allpaths=[r for r in data if r['strategy']==strategy]
            quoted=[r for r in allpaths if 'cost' in r]
            label=strategy.split('_',1)[1];rotation={r['day']:r for r in data if r['strategy']=='rotation_'+label and r['status']=='complete'}
            example=dict(phase=phase,id=identity,case=data[0]['case'],gamma_label=label,paths=len(allpaths),paired_truth_paths=len(pairs),
                history_target_correct=None if not pairs else pairs[0][0]['historical_target_correct'],
                baseline_mean=None if not pairs else float(np.mean([b['p_true'] for r,b in pairs])),
                optimized_mean=None if not pairs else float(np.mean([r['p_true'] for r,b in pairs])),
                paired_gain=None if not pairs else float(np.mean([r['p_true']-b['p_true'] for r,b in pairs])),
                mean_gain_interval95=interval([r['p_true']-b['p_true'] for r,b in pairs]),
                baseline_recovered=sum(b['status']=='complete' and b['map_correct'] for b in base.values()),
                optimized_recovered=sum(r['status']=='complete' and r['map_correct'] for r in allpaths),
                baseline_unique=sum(b['status']=='complete' and b['correct_unique'] for b in base.values()),
                optimized_unique=sum(r['status']=='complete' and r['correct_unique'] for r in allpaths),
                baseline_pass90=sum(b['status']=='complete' and b['true_pass90'] for b in base.values()),
                optimized_pass90=sum(r['status']=='complete' and r['true_pass90'] for r in allpaths),
                mean_fee=None if not quoted else float(np.mean([r['cost'] for r in quoted])),
                mean_DR_kwh=None if not quoted else float(np.mean([r['absolute_DR_kwh'] for r in quoted])),
                rotation_mean=None if not pairs else float(np.mean([rotation[r['day']]['p_true'] for r,b in pairs if r['day'] in rotation])),
                failures=sum(r['status']!='complete' for r in allpaths),voltage_violating_paths=sum(r['status']=='complete' and r['voltage_violation_cells']>0 for r in allpaths))
            output.append(example)
    return output


def solver_summary(run,phase):
    plans=[]
    for file in (run/phase/'jobs').glob('*/day*/plans.json'):plans.extend(io.read(file).values())
    qualified=[p for p in plans if p['qualified']]
    return dict(phase=phase,plans=len(plans),qualified=len(qualified),failed=len(plans)-len(qualified),
        global_gap_closed=sum(p['global_optimality_verified'] for p in qualified),sources=dict(Counter(p.get('source',p['status']) for p in plans)),
        native_statuses=dict(Counter(str(p['native']['status']) for p in qualified if p.get('native'))),
        mean_seconds=float(np.mean([p['seconds'] for p in plans])),p95_seconds=float(np.quantile([p['seconds'] for p in plans],.95)),
        max_seconds=float(max(p['seconds'] for p in plans)),
        median_verified_gap=float(np.median([p['relative_gap'] for p in qualified])),
        max_verified_gap=float(max(p['relative_gap'] for p in qualified)),
        feasible_seed_retained=sum(p.get('source')=='feasible_direction_seed' for p in qualified),
        native_bounds_unavailable=sum(p.get('native') is not None and p['native'].get('bound') is None for p in qualified))


def publish(run,destination):
    require(not destination.exists(),'Preserve every previous public report')
    for phase in ('pilot','population','confirmation'):
        require(io.read(run/phase/'independent_audit.json')['status']=='passed','Require independent phase audit')
    population=load_phase(run,'population');confirmation=load_phase(run,'confirmation')
    inputs=io.read(run/'inputs.json');historical={r['request']['id']:io.read(r['scores']['path']) for r in inputs}
    groups={'all':population}
    for case in sorted({r['case'] for r in population}):groups[case]=[r for r in population if r['case']==case]
    groups['historical_pmap_below_90']=[r for r in population if historical[r['id']]['p_map']<.9]
    groups['historical_pmap_at_least_99']=[r for r in population if historical[r['id']]['p_map']>=.99]
    for covered in (True,False):groups['truth_in_bank' if covered else 'truth_missing_from_bank']=[r for r in population if r.get('truth_covered')==covered]
    for correct in (True,False):groups['historical_target_correct' if correct else 'historical_target_wrong']=[r for r in population if r.get('historical_target_correct')==correct]
    aggregate=[]
    for name,rows in groups.items():
        if rows:aggregate.extend(summary(rows,name,'population'))
    selection=io.read(run/'confirmation_selection.json');best_ids={r['id'] for r in selection['best']}
    aggregate.extend(summary([r for r in confirmation if r['id'] in best_ids],'selected_best','confirmation'))
    aggregate.extend(summary([r for r in confirmation if r['id'] not in best_ids],'selected_weak_controls','confirmation'))
    pop_cases=case_summaries(population,'population');confirm_cases=case_summaries(confirmation,'confirmation')
    for row in confirm_cases:row['selection_role']='best' if row['id'] in best_ids else 'weak_control'
    solvers=[solver_summary(run,p) for p in ('pilot','population','confirmation')]
    destination.mkdir(parents=True)
    write_csv(destination/'overall_and_groups.csv',aggregate);write_csv(destination/'population_cases.csv',pop_cases)
    write_csv(destination/'confirmation_cases.csv',confirm_cases);io.write(destination/'solver_summary.json',solvers)
    io.write(destination/'selection.json',{k:v for k,v in selection.items() if k!='ids'})
    protocol=io.read(run/'protocol.json');lookup={(r['phase'],r['group'],r['strategy']):r for r in aggregate}
    base=lookup['population','all','no_dr'];total=base['planned_paths'];covered=base['posterior_paths_with_truth']
    baselines=[r for r in population if r['strategy']=='no_dr' and r['status']=='complete']
    coverage=[]
    for case in ('all','paper15','soumalas11','flynn16','pengwah18'):
        subset=baselines if case=='all' else [r for r in baselines if r['case']==case]
        distinct={r['id']:r for r in subset}
        coverage.append(dict(case=case,inputs=len(distinct),truth_in_bank=sum(r['truth_covered'] for r in distinct.values()),
            historical_target_correct=sum(r['historical_target_correct'] for r in distinct.values())))
    write_csv(destination/'coverage.csv',coverage)
    coverage_all=coverage[0]
    lines=['# 线性DR真实AC总体结果','',
        '本轮采用用户确认的流程：先生成独立真实新日，再加5%预测噪声形成规划输入；规划只读预测与历史冻结候选；线性DR直接进入原AC网络；最后加入原P/Q/V表计噪声计算后验。没有使用HtP+e生成评价电压。',
        '',f"正式主体为400历史输入、{total}个独立新日背景；每个背景7策略，共{len(population)}条策略记录。选例复验另有{len(selection['ids'])}个输入、每输入64个新日，合计{len(confirmation)}条策略记录。案例、策略、日三个分母分开。",'',
        '## 放松范围与求解','',
        '设备功率、SOC、能量、时窗、充放电互斥及截断约束全部取消。响应为deltaP=历史节点峰值×小时请求，deltaQ=.35deltaP，确定性执行。费用为归一化请求平方的均值，是合成费用。物理电压仍由原网络AC计算；越限仅统计，不通过截断DR消除。',
        '', '目标为历史后验最高候选；规划竞争者为历史权重至少1%的其他候选，并始终包含最强备选。最优性范围对应这个事前确定的QCQP；评价后验和结构错漏仍使用完整原库。候选R/X及共同sigma²均冻结，64点训练只用于旧拟合，全部32点旧验证和96点新日共同评分。',
        '', '直接解原QCQP，不使用ADMM：单竞争者用标量乘子方程获得可行点和对偶下界；满足全部竞争者且上下界闭合时结束，否则用可行初值启动原生联合QCQP。计划核内时间包含初值生成和原生QCQP建模；按输入共享的响应矩阵分解、预测准备和进程启动不计入单次核内时间，整阶段耗时另存完成标记。实例初值原样保留时单列，不称全局最优。', '',
        '| 阶段 | 计划数 | 合格 | 闭合全局费用界 | 平均／95%规划秒 | 保留定向初值 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in solvers:lines.append(f"| {s['phase']} | {s['plans']} | {s['qualified']} | {s['global_gap_closed']} | {s['mean_seconds']:.4f}／{s['p95_seconds']:.4f} | {s['feasible_seed_retained']} |")
    lines+=['','## 全部400输入的总体效果','',
        f"原库含完整真拓扑的输入为{coverage_all['truth_in_bank']}/{coverage_all['inputs']}；其余{coverage_all['inputs']-coverage_all['truth_in_bank']}个输入属于候选库缺失。平均真拓扑后验只在原库含真拓扑的{covered}个有效新日上统计；完整恢复和正确唯一输出以全部{total}个预定新日为分母。缺真候选另列，DR不会生成新拓扑。95%均值区间按历史输入成组重抽样，保留同输入四日配对。",'',
        '| 策略 | 真后验均值 | 相对无DR变化（百分点）及95%区间 | 完整恢复 | 正确唯一 | 错误唯一 | 平均费用 | 失败 | 电压越限日 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    strategies=['no_dr']+[f'{s}_g{g:.6g}' for g in protocol['gamma'] for s in ('rotation','optimized')]
    for strategy in strategies:
        r=lookup['population','all',strategy];change=r['paired_true_posterior_delta'];ci=r['delta_interval95']
        diff='—' if change is None else f"{100*change:+.3f} [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}]"
        posterior='—' if r['mean_true_posterior'] is None else f"{r['mean_true_posterior']:.3%}"
        lines.append(f"| {strategy} | {posterior} | {diff} | {r['full_recoveries']}/{total} | {r['correct_unique']}/{total} | {r['wrong_unique']}/{total} | {r['mean_fee']:.6g} | {r['failed_paths']} | {r['voltage_violating_paths']} |")
    strongest=lookup['population','all','optimized_g4'];medium=lookup['population','all','optimized_g2']
    covered_base=lookup['population','truth_in_bank','no_dr'];covered_strong=lookup['population','truth_in_bank','optimized_g4']
    lines+=['','费用与请求幅度按全部已报价新日统计，包括AC失败日；失败不退费。同费轮换逐日精确匹配。后验均值只使用存在真实候选且有效的配对观测，失败数单列，完整恢复分母包含失败。计划DR能量在AC失败日仍是计划量，不解释成成功物理运行。',
        '', f"含真候选组的完整恢复为{covered_base['full_recoveries']}/{covered_base['planned_paths']}→{covered_strong['full_recoveries']}/{covered_strong['planned_paths']}。全部输入的错误唯一输出却从{base['wrong_unique']}升到{strongest['wrong_unique']}，其中库含真候选组从{covered_base['wrong_unique']}变为{covered_strong['wrong_unique']}；新增错误唯一输出主要来自库缺真候选。这表明更强DR也会让不完整候选库中的错误模型更自信，归一化后验不能自行发现库外真拓扑。",'',
        f"从增强量2升到4，含真候选组平均后验仅再增加{100*(strongest['mean_true_posterior']-medium['mean_true_posterior']):.3f}个百分点，全体完整恢复多{strongest['full_recoveries']-medium['full_recoveries']}日，平均费用变为{strongest['mean_fee']/medium['mean_fee']:.2f}倍。强度4存在{strongest['failed_paths']}次AC失败、{strongest['voltage_violating_paths']}个成功求解但电压越限的新日；最大单节点小时请求达到历史节点功率峰值的{strongest['max_request']:.2f}倍。无设备限制使少数极端方案主导费用，平均费用{strongest['mean_fee']:.6g}，中位数仅{strongest['median_fee']:.6g}。",'',
        '', '各网络、历史后验不足90%、历史目标正确／错误、库覆盖／缺失的全部分组见overall_and_groups.csv。所有400个输入、三档强度均在population_cases.csv保留，未只统计改善案例。后验是冻结候选和共同历史尺度下的归一化权重，真实恢复率另外统计。','',
        '## 已锁定案例的独立复验','',
        '按主体四日开发结果，每网络选两个改善排名最高输入和一个最弱对照，随后才打开每输入64条独立新日。主结果仍是上面的全体输入；下面是选择后独立复验，不将这些案例的平均改善冒充总体改善。表列gamma=4，其他两档全部保留在confirmation_cases.csv。','',
        '| 输入 | 角色 | 无DR→优化DR真后验 | 平均变化（百分点）及95%区间 | 完整恢复／64 | 90%达标／64 | 费用 | 电压越限日 |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for r in confirm_cases:
        if r['gamma_label']!='g4':continue
        ci=r['mean_gain_interval95'];delta=r['paired_gain']
        label='改善排名靠前' if r['selection_role']=='best' else '最弱对照'
        lines.append(f"| {r['id']} | {label} | {r['baseline_mean']:.2%}→{r['optimized_mean']:.2%} | {100*delta:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | {r['baseline_recovered']}→{r['optimized_recovered']} | {r['baseline_pass90']}→{r['optimized_pass90']} | {r['mean_fee']:.6g} | {r['voltage_violating_paths']} |")
    positive=[r for r in confirm_cases if r['gamma_label']=='g4' and r['mean_gain_interval95'] is not None and r['mean_gain_interval95'][0]>0]
    negative=[r for r in confirm_cases if r['gamma_label']=='g4' and r['paired_gain'] is not None and r['paired_gain']<0]
    lines+=['',f"gamma=4复验中，{len(positive)}/{len(selection['ids'])}个输入的平均真后验改善区间下端大于0，{len(negative)}/{len(selection['ids'])}个输入的样本平均后验下降。全部下降、失败和对照均保留。",'',
        '## 历史开发失败与验证范围','',
        '先前v1主体已运行400输入，但3个新日因读取不存在的求解器ObjBound属性中断了评价，共21条记录。该批结果、源码与失败原样保留。随后只修复缺失界值处理和逐计划异常记录，不改变物理网络、候选、噪声、证据门槛或策略；正式v2采用另一独立随机流。v1不混入正式总体效果。',
        '', '各阶段对全部保存观测独立复算预测生成、线性交付、P/Q/V表计公式、后验及选模决定，并对预定子集重新运行同一AC路径核对电压。重放不是新的独立样本。原始观测、拟合参数、真实网络数值及报价轨迹留私有。',
        '', '## 数值条件与边界','',
        '增大gamma会要求更强的平均区分证据，并提高费用；它没有规定90%后验达标频率。所有幅度、负净功率和电压越限均在分组表报告。当前结果属于无设备能量约束、确定性线性响应及给定5%预测精度的AC模拟；热稳定约束未加入。现场设备可执行性不包含在这个模型中。本次预测输入和设备模型同时改变，不把新旧批次差异归因于单一求解算法。']
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audits={phase:io.read(run/phase/'independent_audit.json') for phase in ('pilot','population','confirmation')}
    io.write(destination/'VALIDATION.json',dict(status='passed',created_utc=utc(),audits={k:{a:b for a,b in v.items() if a!='source'} for k,v in audits.items()},
        original_solver_unit_tests=6,aggregation_unit_tests=2,private_raw_observations_published=False,new_identifier_fits=0,new_parameter_fits=0))
    plot(destination,aggregate,confirm_cases,protocol)
    io.write(destination/'RECEIPT.json',dict(created_utc=utc(),protocol_sha256=sha(run/'protocol.json'),selection_sha256=sha(run/'confirmation_selection.json'),
        files={p.name:sha(p) for p in destination.iterdir() if p.is_file()},source=io.ref(__file__)))
    io.write(run/'complete.json',dict(status='complete',created_utc=utc(),public_report=str(destination/'REPORT.md'),source=io.ref(__file__)))
    print({'population_contexts':total,'population_rows':len(population),'confirmation_rows':len(confirmation),'positive_confirmed_inputs':len(positive),'negative_inputs':len(negative)},flush=True)


def plot(destination,aggregate,cases,protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    pop={r['strategy']:r for r in aggregate if r['phase']=='population' and r['group']=='all'}
    levels=protocol['gamma']
    for strategy,label in [('optimized','Optimized linear DR'),('rotation','Same-fee rotation')]:
        rows=[pop[f'{strategy}_g{g:.6g}'] for g in levels]
        axes[0].plot(levels,[100*r['paired_true_posterior_delta'] for r in rows],'o-',label=label)
        axes[1].plot(levels,[r['full_recovery_rate_all'] for r in rows],'o-',label=label)
    axes[0].axhline(0,color='gray',lw=1);axes[0].set(xlabel='Required expected log-odds gain',ylabel='Mean true-posterior gain (pp)',title='Truth in library (150/400): actual AC')
    axes[0].legend(frameon=False,fontsize=8);axes[1].axhline(pop['no_dr']['full_recovery_rate_all'],ls='--',color='gray',label='No DR')
    axes[1].axhline(150/400,ls=':',color='black',label='Library coverage ceiling')
    axes[1].set(xlabel='Required expected log-odds gain',ylabel='Complete recovery fraction',ylim=(0,1),title='All 400 inputs and all scheduled paths');axes[1].legend(frameon=False,fontsize=8)
    examples=[r for r in cases if r['gamma_label']=='g4'];x=np.arange(len(examples));means=np.array([100*r['paired_gain'] for r in examples])
    bounds=np.array([[100*v for v in r['mean_gain_interval95']] for r in examples])
    axes[2].bar(x,means,color=['#2878b5' if r['selection_role']=='best' else '#999999' for r in examples])
    axes[2].errorbar(x,means,yerr=np.maximum(0,np.stack((means-bounds[:,0],bounds[:,1]-means))),fmt='none',ecolor='black',capsize=2)
    labels=[r['id'].replace('main__input','M').replace('pressure__input','P').replace('__101','') for r in examples]
    axes[2].axhline(0,color='gray',lw=1);axes[2].set(xticks=x,xticklabels=labels,xlabel='Input (M=main, P=pressure)',ylabel='Mean true-posterior gain (pp)',title='Selected examples + controls: 64 new days')
    axes[2].tick_params(axis='x',labelrotation=60)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for suffix in ('png','svg'):
        p=destination/('outcomes.'+suffix);fig.savefig(p,dpi=180,bbox_inches='tight')
        if suffix=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();publish(a.run.resolve(),a.public.resolve())
