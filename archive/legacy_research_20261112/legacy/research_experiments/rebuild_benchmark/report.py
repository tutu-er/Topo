"""Write tables and scientific plots from the frozen benchmark results."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from .run import METHODS,read,write,sha

LABELS={'original_leaf':'Original MILP / leaf','original_RNJ':'Original MILP / RNJ',
        'pool_RNJ':'Pool repair / RNJ','rebuild_RNJ':'Local rebuild / RNJ'}


def markdown(frame,columns,digits=4):
    lines=['|'+'|'.join(columns)+'|','|'+'|'.join('---' for _ in columns)+'|']
    for _,row in frame.iterrows():
        values=[]
        for col in columns:
            value=row[col]
            if isinstance(value,(float,np.floating)):value=f'{value:.{digits}f}' if np.isfinite(value) else 'unavailable'
            values.append(str(value))
        lines.append('|'+'|'.join(values)+'|')
    return '\n'.join(lines)


def report(output,analysis,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output).resolve();analysis=Path(analysis).resolve();destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(analysis/'per_model.csv');summary=pd.read_csv(analysis/'summary.csv');pairs=pd.read_csv(analysis/'paired.csv')
    verification=read(analysis/'verification.json');params=read(output/'frozen_method_parameters.json');tuned=read(output/'frozen_parameters.json')
    main=summary[(summary.panel=='main')&(summary.view=='tuned_bank')].copy()
    path=summary[(summary.panel=='main')&(summary.view=='tuned_path')].copy()
    contrast=pairs[(pairs.panel=='main')&(pairs.view=='tuned_bank')].copy()
    time=[]
    for method,g in f[(f.panel=='main')&(f.view=='tuned_bank')].groupby('method'):
        time.append(dict(method=method,inputs=len(g),search_median=float(g.search_seconds.median()),
            search_p90=float(g.search_seconds.quantile(.9)),selection_mean=float((g.elapsed_seconds-g.search_seconds).mean()),
            total_estimate_median=float(g.standalone_estimate_seconds.median()),
            MIP_calls=int(g.MIP_count.sum()),MIP_timeouts=int(g.MIP_timeouts.sum()),
            search_budget_overrun_p90=float(g.search_budget_overrun.quantile(.9))))
    timing=pd.DataFrame(time);timing.to_csv(analysis/'timing.csv',index=False)
    events=pd.read_csv(analysis/'MIP_events.csv');event_summary=events.groupby('method').agg(calls=('job','count'),
        timeouts=('status',lambda s:int((s==9).sum())),incumbents_valid=('incumbent_valid','sum'),
        binary_median=('binaries','median'),constraint_median=('constraints','median'),
        max_violation=('max_violation','max'),median_gap=('gap','median')).reset_index()
    event_summary.to_csv(analysis/'MIP_summary.csv',index=False)
    # Output policy changes are paired on exactly the same search trajectory.
    policy=[]
    for method in METHODS:
        g=f[(f.panel=='main')&(f.method==method)]
        wide=g.pivot(index='job',columns='view',values='clade_f1')
        for after,before in (('default_bank','legacy_bank'),('tuned_bank','default_bank')):
            d=wide[after]-wide[before]
            policy.append(dict(method=method,contrast=after+' minus '+before,mean_delta=float(d.mean()),
                               wins=int((d>1e-10).sum()),ties=int((abs(d)<=1e-10).sum()),losses=int((d< -1e-10).sum())))
    pd.DataFrame(policy).to_csv(analysis/'output_policy.csv',index=False)
    # Both-sided adverse cases survive: prediction and topology may disagree.
    adverses=[]
    for reference in METHODS[:-1]:
        g=f[(f.panel=='main')&(f.view=='tuned_bank')]
        a=g[g.method=='rebuild_RNJ'].set_index('job');b=g[g.method==reference].set_index('job')
        for job in a.index:
            d=a.loc[job,'clade_f1']-b.loc[job,'clade_f1']
            if abs(d)>1e-10:
                adverses.append(dict(job=job,reference=reference,F1_delta=d,
                    test_relative_change=a.loc[job,'test_mae']/b.loc[job,'test_mae']-1,
                    validation_relative_change=a.loc[job,'validation_mae']/b.loc[job,'validation_mae']-1,
                    new_TP=a.loc[job,'matched_clades'],old_TP=b.loc[job,'matched_clades'],
                    new_FP=a.loc[job,'predicted_clades']-a.loc[job,'matched_clades'],
                    old_FP=b.loc[job,'predicted_clades']-b.loc[job,'matched_clades'],
                    novel_true=a.loc[job,'novel_true'],novel_false=a.loc[job,'novel_false']))
    pd.DataFrame(adverses).to_csv(analysis/'non_ties.csv',index=False)
    extra=pd.read_csv(analysis/'additional_per_model.csv')
    assert read(analysis/'additional_verification.json')['passed'] and read(analysis/'crosssolver.json')['passed']
    full=extra[extra.phase=='fullwarm'];recovered=extra[(extra.phase=='recovery')&(extra.view=='tuned_bank')]
    g=full[(full.panel=='main')&(full.view=='tuned_bank')]
    full_timing=dict(method='original_leaf_fullwarm',inputs=len(g),search_median=float(g.search_seconds.median()),
        search_p90=float(g.search_seconds.quantile(.9)),selection_mean=float((g.elapsed_seconds-g.search_seconds).mean()),
        total_estimate_median=float(g.standalone_estimate_seconds.median()),MIP_calls=int(g.MIP_count.sum()),
        MIP_timeouts=int(g.MIP_timeouts.sum()),search_budget_overrun_p90=float((g.search_seconds-20).clip(lower=0).quantile(.9)))
    timing=pd.concat([timing,pd.DataFrame([full_timing])],ignore_index=True);timing.to_csv(analysis/'timing.csv',index=False)
    full_events=pd.read_csv(analysis/'additional_MIP_events.csv').query("phase=='fullwarm'").copy()
    full_events['method']='original_leaf_fullwarm'
    all_events=pd.concat([events,full_events],ignore_index=True)
    event_summary=all_events.groupby('method').agg(calls=('job','count'),
        timeouts=('status',lambda s:int((s==9).sum())),incumbents_valid=('incumbent_valid','sum'),
        binary_median=('binaries','median'),constraint_median=('constraints','median'),
        max_violation=('max_violation','max'),median_gap=('gap','median')).reset_index()
    event_summary.to_csv(analysis/'MIP_summary.csv',index=False)
    colors=['#718096','#2b6cb0','#2f855a','#c05621','#805ad5'];panels=('main','stress','topology','case33')
    case33=summary[(summary.panel=='case33')&(summary.view=='tuned_bank')&(summary.method=='original_leaf')].iloc[0]
    names=('Reference\n(n=108)','Stress\n(n=12)','Random trees\n(n=16)',
        f"Case33 variant\n({int(case33.available)}/{int(case33.inputs)} returned)")
    fig,axes=plt.subplots(1,2,figsize=(13,4.9),sharey=True)
    for ax,view,title in zip(axes,('tuned_bank','tuned_path'),('Shared RNJ bank + search path','Search path only')):
        for j,method in enumerate(METHODS):
            values=[summary[(summary.panel==panel)&(summary.view==view)&(summary.method==method)].F1.iloc[0] for panel in panels]
            ax.bar(np.arange(4)+(j-2)*.17,values,width=.16,color=colors[j],label=LABELS[method])
        values=[full[(full.panel==panel)&(full.view==view)].clade_f1.mean() for panel in panels]
        ax.bar(np.arange(4)+2*.17,values,width=.16,color=colors[4],label='Original MILP / full warm pool')
        ax.set_xticks(np.arange(4),names);ax.set_ylim(0,1.03);ax.set_title(title);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean terminal-support F1')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(bottom=.27,top=.9,wspace=.1);fig.savefig(destination/'comparison.png',dpi=170);plt.close(fig)
    cells=pd.read_csv(analysis/'main_cells.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    full_cells=full[(full.panel=='main')&(full.view=='tuned_bank')].groupby(['voltage_sigma','N']).agg(F1=('clade_f1','mean')).reset_index()
    full_cells.to_csv(analysis/'fullwarm_cells.csv',index=False)
    for ax,reference in zip(axes,('original_leaf_fullwarm','pool_RNJ')):
        pivot=cells.pivot(index=['voltage_sigma','N'],columns='method',values='F1')
        pivot['original_leaf_fullwarm']=full_cells.set_index(['voltage_sigma','N']).F1
        mat=(pivot.rebuild_RNJ-pivot[reference]).unstack('N');limit=max(.01,float(abs(mat.to_numpy()).max()))
        im=ax.imshow(mat,origin='lower',cmap='RdYlGn',vmin=-limit,vmax=limit,aspect='auto')
        for i in range(len(mat)):
            for j in range(len(mat.columns)):ax.text(j,i,f'{mat.iloc[i,j]:+.3f}',ha='center',va='center')
        ax.set_xticks(range(len(mat.columns)),mat.columns);ax.set_yticks(range(len(mat)),[f'{v*100:g}%' for v in mat.index])
        label=LABELS[reference] if reference in LABELS else 'Original MILP / full warm pool'
        ax.set_xlabel('Training snapshots');ax.set_ylabel('Voltage noise');ax.set_title('Rebuild minus '+label);fig.colorbar(im,ax=ax,label='F1 difference')
    fig.tight_layout();fig.savefig(destination/'main_cells.png',dpi=170);plt.close(fig)
    main_table=main[['method','inputs','available','F1','exact','test_mae']]
    full_summary=full.groupby(['panel','view']).agg(inputs=('job','count'),available=('model_available','sum'),
        F1=('clade_f1','mean'),exact=('exact','sum'),test_mae=('test_mae','mean')).reset_index()
    full_main=full_summary[(full_summary.panel=='main')&(full_summary.view=='tuned_bank')].assign(method='original_leaf_fullwarm')
    main_table=pd.concat([main_table,full_main[main_table.columns]],ignore_index=True)
    full_contrasts=[]
    for view in ('tuned_bank','tuned_path'):
        for panel in panels:
            a=f[(f.panel==panel)&(f.view==view)&(f.method=='rebuild_RNJ')].set_index('job')
            b=full[(full.panel==panel)&(full.view==view)].set_index('job');d=a.clade_f1-b.clade_f1
            full_contrasts.append(dict(panel=panel,view=view,inputs=len(d),mean_delta=d.mean(),
                wins=int((d>1e-10).sum()),ties=int((abs(d)<=1e-10).sum()),losses=int((d< -1e-10).sum())))
    full_contrasts=pd.DataFrame(full_contrasts);full_contrasts.to_csv(analysis/'fullwarm_pairs.csv',index=False)
    recovered_summary=recovered.groupby('method').agg(inputs=('job','count'),available=('model_available','sum'),
        F1=('clade_f1','mean'),exact=('exact','sum'),test_mae=('test_mae','mean')).reset_index()
    raw=pd.read_csv(analysis/'raw_changes.csv');bank_gain=pd.read_csv(analysis/'bank_contribution.csv')
    diagnosis=[]
    for method,g in raw.groupby('method'):
        h=bank_gain[bank_gain.method==method]
        diagnosis.append(dict(method=method,inputs=len(g),training_improved=int((g.training_relative_change< -1e-9).sum()),
            raw_F1_wins=int((g.F1_delta>1e-9).sum()),raw_F1_losses=int((g.F1_delta< -1e-9).sum()),
            net_TP=int(g.TP_delta.sum()),net_FP=int(g.FP_delta.sum()),
            final_F1_gain_over_bank=float(h.F1_delta.mean()),final_wins_over_bank=int((h.F1_delta>1e-9).sum()),
            final_losses_over_bank=int((h.F1_delta< -1e-9).sum())))
    diagnosis=pd.DataFrame(diagnosis);diagnosis.to_csv(analysis/'diagnosis_summary.csv',index=False)
    recovery_inputs=len(read(output/'recovery_protocol.json')['jobs'])
    panel_table=summary[summary.view=='tuned_bank'][['panel','method','inputs','F1','exact']]
    block=pd.read_csv(analysis/'paired_bundle.csv');block=block[(block.view=='tuned_bank')&(block.reference=='original_leaf')]
    matched=pairs[(pairs.panel=='main')&(pairs.view=='shared_bank')]
    supplement_text='Supplementary runs have not been summarized.'
    if (analysis/'supplement_per_model.csv').exists():
        s=pd.read_csv(analysis/'supplement_per_model.csv');rows=[]
        primary=f[(f.view=='tuned_bank')&f.job.isin(read(output/'analysis_addendum.json')['jobs'])]
        for method in METHODS:
            g=primary[primary.method==method];rows.append(dict(phase='primary20',method=method,inputs=len(g),F1=g.clade_f1.mean(),test_mae=g.test_mae.mean()))
        for (phase,method),g in s[s.view=='tuned_bank'].groupby(['phase','method']):
            rows.append(dict(phase=phase,method=method,inputs=len(g),F1=g.clade_f1.mean(),test_mae=g.test_mae.mean()))
        tab=pd.DataFrame(rows);tab.to_csv(analysis/'supplement_summary.csv',index=False)
        supplement_text=markdown(tab,['phase','method','inputs','F1','test_mae'],6)
    historical_text='Historical reference not summarized.'
    if (analysis/'historical_reference.csv').exists():
        h=pd.read_csv(analysis/'historical_reference.csv');tab=h.groupby('view').agg(inputs=('job','count'),F1=('clade_f1','mean'),exact=('exact','sum'),test_mae=('test_mae','mean')).reset_index()
        historical_text=markdown(tab,['view','inputs','F1','exact','test_mae'],6)
    params_table=pd.DataFrame([dict(method=method,view=view,**q['parameters']) for method,v in params['methods'].items() for view,q in v.items()])
    params_table.to_csv(analysis/'chosen_parameters.csv',index=False)
    text=f'''# 调参后的局部重构与原始 MILP：系统配对结果

本报告的算法和数据口径见 [冻结协议](PROTOCOL.md)，参数及完整开发网格见 [冻结记录](tuning_frozen.json)。开发 16 输入、留出 140 输入，四方法共 560 个工作进程。主面板有 108 输入、3 个 bundle；另外的压力、新随机树和 case33 改造例单列。所有数据是既有合成 AC 观测，测试重复未进入本轮调参。

## 参数与公平性

局部搜索参数：`b={tuned['parameters']['b']}`，相对改善门槛 `{tuned['parameters']['relative_gain']}`。输出收缩和验证带宽对每个方法、每个视图分别用开发验证数据校准；均先收缩、训练重拟合，再计算实际验证损失。调参不使用 F1 和测试损失。

{markdown(params_table,['method','view','b','relative_gain','resolution','validation_fraction'])}

四者共享同一个训练前端和暖启动信息。原始方法的二元支撑域仍完全自由，但每步固定已有结构；RNJ 同初值对照区分初始化收益，候选池修正对照区分新方法相对当前方法的收益。原始方法是有限时间实现，不是全树全局最优 oracle。

表中的 `original_leaf_fullwarm` 是在读取主结果后追加的更强基线，扫描全部相容池内暖启动候选，仍受同一 20 秒搜索软预算和 2 秒单次 MILP 上限约束，输出参数沿用冻结值。其结果单独保留，不替换原始四方法批次。

## 主面板：带共享 RNJ 库的实际输出

{markdown(main_table,['method','inputs','available','F1','exact','test_mae'],6)}

与局部方法的逐输入配对差值：

{markdown(contrast,['reference','inputs','mean_delta','wins','ties','losses'],6)}

使用完全相同输出参数的搜索对照：

{markdown(matched,['reference','inputs','mean_delta','wins','ties','losses'],6)}

三个独立 bundle 相对原始叶起点的平均差：

{markdown(block,['bundle','inputs','mean_delta'],6)}

108 个相关条件不当作 108 个独立重复。只有 3 个 bundle，不能靠行数给出很强的总体统计结论。

## 原始 MILP 的完整池暖启动对照

{markdown(full_summary[full_summary.view.isin(('tuned_bank','tuned_path'))],['panel','view','inputs','available','F1','exact','test_mae'],6)}

局部重构减去完整池暖启动原始 MILP：

{markdown(full_contrasts,['panel','view','inputs','mean_delta','wins','ties','losses'],6)}

此对照在主分析之后增加，未再用测试结果调参。它检查暖启动限制，不能把结果归因于联合层级搜索本身。每步候选扫描、建模及求解均消耗搜索预算，达到时限时可能尚未扫描完池。

## 训练改善与拓扑恢复的分离

{markdown(diagnosis,['method','inputs','training_improved','raw_F1_wins','raw_F1_losses','net_TP','net_FP','final_F1_gain_over_bank'],6)}

raw 指未做最终收缩选模的搜索终点，增量相对同方法初始家族；原始叶起点没有内部集合，F1 的变化不能与 RNJ 起点变化混作同一起点实验。净 TP/FP 为各输入增减的合计。最后一列则将最终输出与同一输出规则选择的公共 RNJ 库比较，分开查看 RNJ 初始化和新增搜索的贡献。

拟合改善同时可能来自恢复真边、加入弱假边或用错误结构近似噪声。F1 对每条额外连接计数，不随权重缩小而减小；验证 MAE 衡量预测，不直接识别真连接。逐步和逐例两侧变化都保存在 `accepted_steps.csv`、`raw_changes.csv` 和 `non_ties.csv`。

## 仅路径输出：分开查看初始化和搜索

{markdown(path[['method','inputs','F1','exact','test_mae']],['method','inputs','F1','exact','test_mae'],6)}

![F1 comparisons](comparison.png)

## 场景覆盖

{markdown(panel_table,['panel','method','inputs','F1','exact'],6)}

![Noise and sample-size effects](main_cells.png)

网络、噪声、样本数的全部单元值保存在表格中；压力/随机树/case33 的均值不能用参考网络主面板代替。原始叶起点含 RNJ 库的输出使用了 RNJ 信息，因而不能与仅路径输出混作一个算法。

## 计算与求解状态

{markdown(timing,['method','inputs','search_median','search_p90','selection_mean','total_estimate_median','MIP_calls','MIP_timeouts'],4)}

{markdown(event_summary,['method','calls','timeouts','incumbents_valid','binary_median','constraint_median','median_gap'],6)}

时间来自并发的单线程工作进程；前端和输出选模额外计入成本估算，没有称作独立顺序计时。20 秒是搜索软预算，构模或单个 LP 导致的超出保留实际值。相对 gap、incumbent 和 bound 是单个子问题的记录，不是全树最优性证明。触时可行 incumbent 可继续使用，未找到改善且界不充分时保留未决停止原因。

## 100 秒预算与扩池控制：固定八输入

{supplement_text}

100 秒两方法均不重新调参。扩池控制补入局部方法按训练数据考虑过的候选，需额外支付局部候选发现成本，不把它解释成免费的部署改进。

## 实际历史 100 秒原始实现：同八输入

{historical_text}

历史路径由当时完整池内暖启动实现产生，与本次“最多四个池内候选暖启动”的限时版本区分。legacy 为历史默认输出，refit 为同路径按本轮已冻结参数重拟合选模；没有重新按真拓扑挑路径。历史参考只在匹配的八个输入上比较，不把其 80 输入总体均值套到本轮 140 输入。

## 共享前端失败的统一恢复

主批次共有 {recovery_inputs} 个输入的共享前置步骤没有返回，本轮记录为 case33、N=288 固定家族 LP 触及 5 秒上限，四个方法均未启动搜索。原始 140 输入分母保留，F1 缺失按 0 计，预测 MAE 仅在可用结果上平均。因此主表 case33 的全分母 F1 与可用均值应分开理解；具体原因和数量以 `failures.json` 为准。

对所有这类输入统一把每个固定家族 LP 上限提高到 60 秒、工作进程硬上限提高到 900 秒；搜索目标预算仍为 20 秒，参数未变。补充结果如下，未覆盖主记录：

{markdown(recovered_summary,['method','inputs','available','F1','exact','test_mae'],6)}

## 核验与记录

独立重算通过 {verification['checks']} 项检查；工作进程失败 {verification['worker_failures']}，原分母完整保留。共享前端、参数文件和冻结代码均校验哈希。16 个固定家族的 HiGHS 独立重解另存 `crosssolver.json`。完整逐模型结果、求解状态和输出策略对照见 `records.zip`，各表的未失败预测指标与全分母 F1 分开解释。

源代码入口为 `research_experiments/rebuild_benchmark/`。复现顺序为 `freeze → prepare development → development → select → controls → select_methods → prepare evaluation → evaluation`；随后执行 `budget`、`augmented_pool`、`recovery`、`fullwarm` 及分析。各阶段使用新输出目录，旧原始数据和历史结果保持不变。
'''
    (destination/'RESULTS.md').write_text(text,encoding='utf-8')
    copied=('summary.csv','paired.csv','main_cells.csv','fullwarm_cells.csv','main_cases.csv','MIP_summary.csv','timing.csv','chosen_parameters.csv','output_policy.csv','supplement_summary.csv')
    for name in copied:
        if (analysis/name).exists():shutil.copy2(analysis/name,destination/name)
    with zipfile.ZipFile(destination/'records.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for file in sorted(analysis.iterdir()):
            if file.suffix in ('.csv','.json'):z.write(file,file.name)
        for name in ('protocol.json','frozen_parameters.json','frozen_method_parameters.json','analysis_addendum.json',
                     'fullwarm_protocol.json','recovery_protocol.json','common_preparation_reuse.json',
                     'evaluation_completion.json','fullwarm_completion.json','recovery_completion.json',
                     'budget_completion.json','augmented_pool_completion.json'):
            if (output/name).exists():z.write(output/name,'protocol/'+name)
        for source in ('source','source_supplement','source_recovery','source_fullwarm'):
            for name in ('engine.py','run.py'):
                file=output/source/'research_experiments/rebuild_benchmark'/name
                z.write(file,'frozen_code/'+source+'/'+name)
    # Keep evidence bytes stable under Windows checkout and whitespace checks.
    for file in destination.iterdir():
        if file.is_file() and file.suffix in ('.md','.csv','.json') and file.name!='delivery_manifest.json':
            file.write_bytes(file.read_bytes().replace(b'\r\n',b'\n'))
    manifest=destination/'delivery_manifest.json'
    write(manifest,{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='delivery_manifest.json'})
    manifest.write_bytes(manifest.read_bytes().replace(b'\r\n',b'\n'))
    print('REPORT',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('analysis');p.add_argument('destination');a=p.parse_args();report(a.output,a.analysis,a.destination)
