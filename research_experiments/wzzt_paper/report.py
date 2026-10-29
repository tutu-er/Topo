from pathlib import Path
import argparse
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha
from .common import MAIN,AC_METHODS
from .statistics import interval

ROOT=Path(__file__).resolve().parents[2]
LARGE=('per_job.csv','catalog.sqlite','selected_models.jsonl','datasets.json','unidentified.csv','changes_vs_multistart.csv','audit_checks.json','LP_checks.csv','greedy_events.csv','postfilter_guard_models.jsonl')


def physical_regimes(output,destination):
    """Describe actual noiseless AC data once per physical bundle, before N sampling."""
    seen=set();rows=[]
    for job in read(output/'protocol.json')['jobs']:
        p=read(output/'jobs'/job/'payload.json');key=(p['case'],p['condition'],p['bundle'])
        if key in seen or p['data_error']:continue
        seen.add(key);voltage=[];drop=[];snapshots=0
        for split in ('train','validation','test'):
            with np.load(p[split+'_truth']) as d:
                v=d['V'];root=d['root'];voltage.append(v.ravel());drop.append((root[...,None]-v).ravel());snapshots+=root.size
        v=np.concatenate(voltage);delta=100*np.concatenate(drop)
        rows.append(dict(case=p['case'],condition=p['condition'],bundle=p['bundle'],physical_snapshots=snapshots,
            minimum_V_pu=float(v.min()),maximum_V_pu=float(v.max()),
            drop_p05_pct=float(np.quantile(delta,.05)),drop_median_pct=float(np.median(delta)),drop_p95_pct=float(np.quantile(delta,.95)),
            scope='noiseless balanced AC; all nine days; root-minus-terminal voltage in percentage points of nominal pu'))
    frame=pd.DataFrame(rows);frame.to_csv(destination/'physical_regimes.csv',index=False)
    return frame[frame.condition=='reference'].groupby('case').agg(physical_bundles=('bundle','count'),
        minimum_V_pu=('minimum_V_pu','min'),maximum_V_pu=('maximum_V_pu','max'),
        mean_drop_median_pct=('drop_median_pct','mean'),mean_drop_p95_pct=('drop_p95_pct','mean')).reset_index()


def table(frame,columns):
    def fmt(x):return '—' if pd.isna(x) else f'{x:.5g}' if isinstance(x,float) else str(x)
    return '\n'.join(['|'+'|'.join(map(str,columns))+'|','|'+'|'.join('---' for _ in columns)+'|']+
        ['|'+'|'.join(fmt(r[c]) for c in columns)+'|' for r in frame.to_dict('records')])


def report(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();f=pd.read_csv(destination/'per_job.csv')
    regimes=physical_regimes(output,destination)
    overview=read(destination/'overview.json');audit=read(destination/'audit_summary.json');primary=pd.read_csv(destination/'primary_statistics.csv')
    summary=pd.read_csv(destination/'panel_summary.csv');main=summary[summary.panel=='main'].set_index('variant').loc[list(MAIN)].reset_index()
    cells=pd.read_csv(destination/'main_cell_intervals.csv');effects=pd.read_csv(destination/'panel_effect_intervals.csv')
    ridge=f[f.variant=='ridge_union'];mainridge=ridge[ridge.panel=='main']
    matrix=mainridge.groupby(['voltage_sigma','N']).clade_f1.mean().unstack().reset_index();matrix['voltage_sigma_pct']=matrix.voltage_sigma*100
    matrix.to_csv(destination/'main_ridge_matrix.csv',index=False,encoding='utf-8-sig')
    ablations=[]
    for panel,group in f.groupby('panel'):
        for method,base in (('ridge_1e3','ridge_union'),('ridge_1e2','ridge_union'),('restart100','restart_control'),('ridge100','ridge_union'),
            ('ridge_guard0','ridge_union'),('multistart_guard0','multistart'),('ridge_two_step','ridge_union'),
            ('ridge_two_step_guard0','ridge_union'),('WZZT_leaf_bank','WZZT_leaf_path')):
            d=group[group.variant==method]
            if not len(d):continue
            ref=group[group.variant==base].set_index('job_id').clade_f1;delta=d.clade_f1-d.job_id.map(ref)
            blocks=delta.groupby(d.case if panel=='topology' else d.bundle).mean()
            ablations.append(dict(panel=panel,variant=method,reference=base,inputs=len(d),F1=d.clade_f1.mean(),
                **interval(blocks),wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta < -1e-10).sum())))
    ablation=pd.DataFrame(ablations);ablation.to_csv(destination/'ablations.csv',index=False)
    ac=[read(p) for p in (output/'ac').glob('*.json')]
    acinfo=dict(planned_rows=int(f.ac_planned.sum()),evaluated_rows=int(f.ac_evaluated.sum()),valid_rows=int(f.ac_valid.eq(True).sum()),
        unique_records=len(ac),infrastructure_errors=sum('error' in a for a in ac),
        physically_valid_unique=sum(a.get('testing',{}).get('valid',False) for a in ac),
        locally_converged_unique=sum(a.get('fitting',{}).get('refit',{}).get('success') is True for a in ac),methods=AC_METHODS)
    write(destination/'ac_summary.json',acinfo)
    timing=read(output/'timing/summary.json');t=pd.DataFrame([{k:v for k,v in r.items() if k!='point'} for r in timing]);t.to_csv(destination/'standalone_timing.csv',index=False)
    timing_summary=t.groupby('variant').agg(replays=('job_id','count'),median_seconds=('estimator_seconds','median'),
        p90_seconds=('estimator_seconds',lambda a:a.quantile(.9)),median_process_seconds=('process_wall_seconds','median'),
        equal_structures=('structure_equal_to_shared','sum')).reset_index();timing_summary.to_csv(destination/'timing_summary.csv',index=False)
    physical=[]
    for path in (output/'data').rglob('diagnostics.json'):
        d=read(path);case=path.relative_to(output/'data').parts[0];condition=path.parent.parent.name
        for row in d['records']:physical.append(dict(case=case,condition=condition,voltage_sigma=d['config']['v_sigma'],**{k:row[k] for k in ('voltage_min','voltage_max','mean_drop','max_drop','design_rank','design_columns','condition_number','dynamic_signal_rms','dynamic_noise_rms')}))
    physical_frame=pd.DataFrame(physical)
    physical_frame['condition_number']=pd.to_numeric(physical_frame.condition_number,errors='coerce')
    physical_frame.groupby(['case','condition','voltage_sigma']).agg(voltage_min=('voltage_min','min'),voltage_max=('voltage_max','max'),
        mean_drop=('mean_drop','mean'),max_drop=('max_drop','max'),mean_design_condition=('condition_number','mean'),
        min_design_rank=('design_rank','min'),design_columns=('design_columns','max'),dynamic_signal_rms=('dynamic_signal_rms','mean'),dynamic_noise_rms=('dynamic_noise_rms','mean')).to_csv(destination/'physical_summary.csv')
    origins=f.groupby(['panel','variant','selected_origin'],dropna=False).size().reset_index(name='rows');origins.to_csv(destination/'selected_origins.csv',index=False)
    claims=[]
    for r in primary.to_dict('records'):
        claims.append(dict(claim=r['contrast'],status='conditional_mean_gain_supported' if r['lower']>0 and r['holm_sign_p']<.05 else 'mixed_or_insufficient_positive_evidence',
            evidence=f"paired mean {r['mean']:.6g}; 95% CI [{r['lower']:.6g},{r['upper']:.6g}]; Holm sign p={r['holm_sign_p']:.5g}",
            scope='four fixed balanced radial terminal-only networks; ten independent bundles; frozen parameters'))
    claims+= [dict(claim='universal topology recovery / original full-domain MILP optimality',status='not_established',evidence='finite candidate local paths; all counterexamples and wall limits retained',scope='no global certificate'),
              dict(claim='field data / unbalanced three-phase deployment',status='not_tested',evidence='synthetic grids and modified terminalized case33; balanced AC',scope='simulation only')]
    pd.DataFrame(claims).to_csv(destination/'claim_evidence.csv',index=False)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    methods={'NJ':'#8b9099','WZZT_leaf_path':'#bd6b48','multistart':'#235cb0','ridge_union':'#228c72'}
    fig,axes=plt.subplots(1,4,figsize=(16,4.7),sharey=True,layout='constrained')
    for ax,sigma in zip(axes,(0.,.0002,.0005,.001)):
        for method,color in methods.items():
            d=cells[(cells.voltage_sigma==sigma)&(cells.variant==method)].sort_values('N')
            ax.errorbar(d.N,d['mean'],yerr=np.array([d['mean']-d.lower,d.upper-d['mean']]),marker='o',lw=1.4,capsize=2,color=color,label=method)
        ax.set_title(f'V/root noise SD {100*sigma:g}%');ax.set_xscale('log',base=2);ax.set_xticks([24,48,96,288],labels=['24','48','96','288']);ax.grid(alpha=.2);ax.set_ylim(-.02,1.04);ax.set_xlabel('Training snapshots (3 days)')
    axes[0].set_ylabel('Nontrivial clade F1');axes[-1].legend(fontsize=8,loc='lower right')
    fig.suptitle('10 paired independent bundles; pointwise 95% block-bootstrap intervals; PQ noise SD 0.5%')
    for ext in ('png','pdf'):fig.savefig(destination/f'main_noise_size.{ext}',dpi=175)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained')
    y=np.arange(len(primary));ax.errorbar(primary['mean'],y,xerr=np.array([primary['mean']-primary.lower,primary.upper-primary['mean']]),fmt='o',capsize=4,color='#228c72')
    ax.set_yticks(y,labels=primary.contrast);ax.axvline(0,color='gray',lw=1);ax.set_xlabel('Paired F1 gain; 95% interval over 10 bundle means');ax.grid(axis='x',alpha=.2)
    for ext in ('png','pdf'):fig.savefig(destination/f'primary_effects.{ext}',dpi=175)
    plt.close(fig)
    stress=pd.read_csv(destination/'by_condition.csv');stress=stress[stress.panel=='stress']
    names=sorted(stress.condition.unique());fig,ax=plt.subplots(figsize=(10,6.5),layout='constrained');ys=np.arange(len(names))
    for offset,(method,color) in zip((-.22,0,.22),[(m,methods[m]) for m in ('NJ','multistart','ridge_union')]):
        d=stress[stress.variant==method].set_index('condition').loc[names];ax.plot(d.F1,ys+offset,'o',label=method,color=color)
    ax.set_yticks(ys,labels=names);ax.set_xlim(-.02,1.04);ax.set_xlabel('Mean clade F1; 4 networks x 5 bundles x N48/288');ax.grid(axis='x',alpha=.2);ax.legend()
    for ext in ('png','pdf'):fig.savefig(destination/f'stress_conditions.{ext}',dpi=175)
    plt.close(fig)
    topo=f[(f.panel=='topology')&f.variant.isin(methods)].copy();topo['terminals']=topo['case'].str.extract(r'random(\d+)').astype(int)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for method,color in methods.items():
        d=topo[topo.variant==method].groupby('terminals').clade_f1.mean();ax.plot(d.index,d.values,'o-',color=color,label=method)
    ax.set_xticks([8,16,24,32]);ax.set_ylim(-.02,1.04);ax.set_xlabel('Observed terminal count');ax.set_ylabel('Mean clade F1');ax.grid(alpha=.2);ax.legend();ax.set_title('12 new topologies; 3 bundles; two noise levels; N48/288')
    for ext in ('png','pdf'):fig.savefig(destination/f'unseen_topologies.{ext}',dpi=175)
    plt.close(fig)
    main_effects=effects[(effects.panel=='main')&effects.variant.isin(['ridge_union','ridge_guard0','multistart_guard0'])]
    postfilter=pd.read_csv(destination/'postfilter_summary.csv')
    postguard=pd.read_csv(destination/'postfilter_guard_summary.csv')
    guard_holdout=pd.read_csv(destination/'postfilter_guard_holdout_summary.csv')
    free=pd.read_csv(destination/'free_domain_comparisons.csv');free_check=read(destination/'free_domain_verification.json')
    retry=pd.read_csv(destination/'lp60_retry.csv');retry_check=read(destination/'lp60_retry_verification.json')
    joint=pd.read_csv(destination/'joint_extension_comparisons.csv');joint_check=read(destination/'joint_extension_verification.json')
    events=pd.read_csv(destination/'greedy_events.csv');mechanisms=[]
    for panel,d in ridge.groupby('panel'):
        available=d[d.model_available];e=events[events.panel==panel]
        eligible=e[e.full_sweep & (e.true_candidates_tested>0)]
        mechanisms.append(dict(panel=panel,inputs=len(d),available=len(available),
            pool_missing=int((available.candidate_pool_true_recall<1-1e-12).sum()),
            final_crossing=int(available.diagnostic_json.map(lambda s:bool(json.loads(s)['crossing_pairs'])).sum()),
            false_accept_events=len(e),full_sweep_with_true=len(eligible),
            strict_false_preference=int(eligible.strictly_better_than_tested_true.sum()),
            strict_preference_inputs=eligible[eligible.strictly_better_than_tested_true].job_id.nunique()))
    mechanisms=pd.DataFrame(mechanisms);mechanisms.to_csv(destination/'mechanism_summary.csv',index=False)
    origins=f[(f.panel=='main')&f.variant.isin(['multistart','ridge_union'])].groupby(['variant','selected_origin']).size().reset_index(name='inputs')
    text=['# RNJ+WZZT 论文证据：完整独立测试','',
        f"计划 {overview['planned_inputs']} 输入，已结束 {overview['completed_inputs']}，正常返回 {overview['returned_inputs']}；{overview['error_inputs']} 错误、{overview['timeout_inputs']} 工作进程触时。模型记录 {overview['model_rows']}/{overview['planned_model_rows']}，可用模型 {overview['available_models']}。所有失败进入原分母。",
        '', '## 测试方案与当前流程','',
        '结果前冻结的设计见 [PROTOCOL.md](PROTOCOL.md)。流程沿用上一轮：训练平方压降 → 前置有序 R/X、NJ/RNJ 候选池 → 完整 RNJ 联合 L1 库 → 三初值局部修正 → 两档正则化扩池追加修正 → 原 3% 验证简洁规则 → 0.16 最终收缩。AC 为独立训练后处理，未参与拓扑选择。严格保护删簇、两步及100秒分支作为预定对照，不按本轮真拓扑选择部署阈值。',
        '', '四旧网络主面板为640输入、10个独立bundle；400场景输入、12棵新随机树144输入、端点化case33的20输入分别报告。训练N=24/48/96/288对应相同三个物理日的不同采样密度；验证和测试各288点。主面板V=0仍有P/Q噪声0.5%，ideal_ac场景才去除全部计量噪声。',
        '', '指标为非平凡根向末端簇集合的F1，去掉叶簇和全末端簇；exact为该集合完全一致。R/X误差是平方压降响应矩阵的相对Frobenius误差，不用作逐条线路阻抗准确率。四旧网络和新随机树均为合成网络，负荷/DER曲线也由代码生成，没有导入实测曲线。case33使用本地参数表的hybrid-leaf改造，内部额定负荷移到服务端点；33行额定P/Q和32条骨干欧姆参数已与[MATPOWER官方表](https://matpower.org/docs/ref/matpower6.0/case33bw.html)逐项核对，结果见case33_provenance.json。添加28条服务边后得到32个端点；动态曲线仍由共同合成模型生成，不是原始IEEE33恒负荷工况。本地MVA基准10与该官方版本100的差异只改变标幺换算。',
        '', '物理电压与压降范围如下。每个物理bundle合并训练/验证/测试九日共864时刻，在加入计量噪声与N采样前计算；drop为根电压减端点电压，以额定标幺电压的百分点表示。表内中位数/P95再对bundle取均值，完整场景记录见physical_regimes.csv。压降尺度与拓扑可辨识性分别评价。','',
        table(regimes,['case','physical_bundles','minimum_V_pu','maximum_V_pu','mean_drop_median_pct','mean_drop_p95_pct']),
        '', '## 独立主面板','',table(main,['variant','rows','available','F1','exact','wins','ties','losses','R_error','X_error']),
        '', 'wins/ties/losses以多初值为参照；叶初值仅路径和加完整RNJ库的两行分别报告，完整候选保留与实际搜索修正不混为一个机制。selected_origins.csv 与逐条搜索终止状态保留最终模型来源。',
        '',table(origins,['variant','selected_origin','inputs']),
        '', '![噪声与数据量](main_noise_size.png)','',table(matrix,['voltage_sigma_pct',24,48,96,288]),
        '', '每格40输入，由四固定网络×10独立数据组构成。误差棒是10个bundle均值上的逐点95%区间，噪声/采样条件不当成独立重复。三个主比较另报精确符号检验及Holm校正；bootstrap区间和检验分别解释。',
        '', '![主比较差值区间](primary_effects.png)','',table(primary,['contrast','mean','lower','upper','independent_blocks','wins','ties','losses','holm_sign_p']),
        '', '## 场景、新拓扑与case33','', '![场景比较](stress_conditions.png)','',
        table(summary[(summary.panel!='main')&summary.variant.isin(['NJ','WZZT_leaf_path','multistart','ridge_union','ridge_guard0'])],['panel','variant','rows','F1','exact','wins','ties','losses']),
        '', '![新拓扑规模](unseen_topologies.png)','',
        '新树为固定随机生成器的12棵未调参拓扑，端点8/16/24/32各三棵；拓扑区间按树分块，并以共享的三个数据bundle为条件。case33仅一棵改造网络，五数据重复不能替代五棵网络。按条件/网络/数据量/噪声的完整表见 by_condition.csv、by_case.csv、by_noise_N.csv。',
        '', '## 组件与预算消融','',table(ablation[ablation.panel=='main'],['variant','reference','inputs','F1','mean','lower','upper','independent_blocks','wins','ties','losses']),
        '', '每行与其自身输入子集上的reference配对，不能用不同子集总体F1相减。单lambda子面板有3个独立bundle；100秒子面板有5个；两步子面板只有1个bundle，保留结果但不据此给出稳定性结论。联合删除是实际从扩池选中模型继续执行的一轮保护删除。',
        '', table(main_effects,['variant','mean','lower','upper','independent_blocks','wins','losses']),
        '', '单bundle两步消融之后，按JOINT_EXTENSION_PROTOCOL.md追加十bundle的组件测试。固定已有扩池收缩前模型库与初值，分别用新LP执行25秒单步追加修正和25秒宽度5两步提案；另报两步后追加15秒删除。它复用原主面板数据，属于观察初轮结果后的次要扩展，区间用于描述，不增加预定三个主检验。',
        '',table(joint,['variant','reference','inputs','F1','mean','lower','upper','independent_blocks','wins','ties','losses']),
        '', f"追加组件测试返回{joint_check['returned_inputs']}/{joint_check['planned_inputs']}输入，模型{joint_check['available_models']}/{joint_check['rows']}可用；数值/结构核验{joint_check['available_evidence_passed']}。单步与两步LP不共享缓存或基。全部失败仍进入该补充分母。",
        '', '## 自由支持MILP补充对照','',
        'FREE_DOMAIN_PROTOCOL.md说明额外80个匹配输入上的自由支持单步MILP：使用有限池可行暖启动，100秒路径预算、每次MILP10秒、MIPGap=1%，叶初始化；统一原验证/收缩规则。路径单独与保留完整RNJ库分别报告。它补齐自由域和有限池的实际比较，主检验保持原样。',
        '',table(free,['variant','reference','inputs','F1','mean','lower','upper','independent_blocks','wins','ties','losses']),
        '', f"补充模型 {free_check['available_models']}/{free_check['rows']} 可用，实际MILP调用 {free_check['MIP_attempts']} 次，零次MILP调用的输入 {free_check['inputs_without_MIP_attempt']} 项；数值核验 {free_check['all_passed']}。free_domain_MIP_events.csv保留native状态、incumbent、bound、gap及可行性。复用训练前置估计的成本未计入，表中路径预算不能解释成独立部署总耗时。该实现是项目的自由支持逐步策略及本轮统一选模，不称为原作者软件逐设置复现。",
        '', '## 失败、目标偏好与终止机制','',
        'unidentified.csv 保留所有未完整恢复项；failure_modes.csv 区分候选池漏真簇、最终交叉错误簇。greedy_events.csv 记录错误正权簇被接受时的训练损失、最佳已测试真簇提案损失及是否全扫：只有完整有限池扫描且错误端点严格更优的事件，才支持该一步目标偏好诊断；最终交叉关系本身不作为早期阻挡的因果证明。',
        '',table(mechanisms,['panel','inputs','available','pool_missing','final_crossing','false_accept_events','full_sweep_with_true','strict_false_preference','strict_preference_inputs']),
        '', 'pool_missing与final_crossing只统计可用的ridge_union模型；其余事件列汇总已正常返回输入的多条预定搜索，事件之间不独立。strict_false_preference要求完整扫描、至少一个真簇提案且训练损失优势超过1e-8，strict_preference_inputs按输入去重。该表证明观测到的一步训练目标偏好，长期阻挡仍需具体路径判断。',
        '', 'termination_counts.csv 保留每方法每面板的局部停止与触时。固定家族LP最优状态不扩展成全树最优性；900秒工作进程上限与25/100秒搜索路径预算分别记录。',
        '', f"原批次固定LP触时输入{overview['fixed_LP_time_limit_inputs']}组，另有LP_RETRY_PROTOCOL.md规定的60秒补充重试：{retry_check['returned_inputs']}/{retry_check['planned_inputs']}返回。原共享阶段未生成的模型保持不可用标记及零分分母，不能将它们解释成所有算法各自失败。扩展预算结果单列如下，保留原始15秒数据。",'',
        table(retry[retry.variant.isin(['NJ','multistart','ridge_union','ridge_postfilter_guard'])],['job_id','variant','returned','clade_f1','longest_LP_seconds','worker_seconds']),
        '', 'resolution_diagnostic.csv 另记录物理线路响应系数经过本轮数据所得校准与0.16滤波时损失的真簇，用于检查分辨率设置是否会主动删掉弱边。该诊断固定物理系数，不作为允许重新拟合参数后的结构恢复上界，也不参与估计器选模。',
        '', '补充探索性后处理敏感性见EXPLORATORY_DIAGNOSTICS.md：固定原先选中的收缩前扩池模型及其参数，只改变最终过滤阈值，不重新搜索或拟合。五个阈值全部保留，未用真簇选取部署阈值。','',table(postfilter,['panel','floor','rows','F1','exact','validation_mae','test_mae']),
        '', '探索性验证保护只在0/0.04/0.08/0.16间按验证误差和原3%简洁规则选输出；保证相同预选模型的验证误差不增，独立测试和F1仍单独评价。该规则在观察预检与主面板诊断后提出，以下为补充证据，不替换原固定0.16主结果。','',table(postguard,['panel','rows','F1','exact','delta_vs_ridge_union','wins','ties','losses','test_mae']),
        '', 'guard_freeze_receipt.json确认规则在正式新拓扑/case33结果产生前确定。独立补充检查排除动机预检所用random32_t0整棵树和case33的bundle0；剩余132个新树输入按11棵树分块，16个case33输入按四数据组分块。主面板不重复当作新规则的确认数据。','',table(guard_holdout,['panel','inputs','F1','mean','lower','upper','independent_blocks','wins','ties','losses']),
        '', '## AC与独立预测','',table(summary[summary.variant.isin(AC_METHODS)],['panel','variant','AC_planned','AC_evaluated','AC_valid','test_ac_rmse']),
        '', f"预定AC {acinfo['planned_rows']}行，实际评价{acinfo['evaluated_rows']}行、可行{acinfo['valid_rows']}行；去重{acinfo['unique_records']}个输入/参数组合，其中局部重估收敛{acinfo['locally_converged_unique']}个。局部least_squares(max_nfev=12)只用训练日，参数冻结后评价独立测试日；未收敛、物理不可行与基础设施异常分开保留。RMSE均值以可返回的物理预测为条件，并同时列出有效分母。",
        '', 'test_mae是平方压降MAE，单位p.u.^2；test_ac_rmse是电压幅值的相对RMSE，两者分别解释。探索性收缩保护本轮记录线性预测，未追加AC重估，不能将原三方法的AC结果移作它的证据。',
        '', '## 独立运行成本','',table(timing_summary,['variant','replays','median_seconds','p90_seconds','median_process_seconds','equal_structures']),
        '', '四旧网络、bundle0、中低/高噪声、N48/288的16输入各重放四方法，共64次，全批估计结束后顺序运行。各方法使用新进程且不共享前置矩阵、LP或搜索缓存；estimator_seconds不含输入读取/进程导入，process_wall_seconds包含进程启动，结构一致数另列。多初值/扩池计时为当前研究入口，其基本库仍生成固定基线。主批shared_worker_seconds包含额外对照，不能作为单方法部署耗时；大网络仅报告该批成本及路径预算，不从小网络计时外推。',
        '', '独立计时为拓扑估计，不含AC后处理。AC运行耗时另存于原始AC记录；其训练重估和冻结测试是独立评价阶段。',
        '', '硬件与求解环境见hardware.json及protocol.json：Intel Core i9-14900HX、32逻辑处理器，操作系统可见物理内存31.64GiB；主批使用四工作进程，每个Gurobi模型一线程。',
        '', '## 论文主张与证据边界','',table(pd.DataFrame(claims),['claim','status','evidence','scope']),
        '', '这批数据覆盖已声明的平衡径向、隐藏零注入、同步计量模拟域。对比结论使用预定配对差值及区间；记录数、局部求解状态、AC拟合误差和bootstrap区间各自承担相应证据，不互换成拓扑正确概率或全域保证。',
        '', '## 核验、复现与调用','',
        f"可用记录的数值核验：{audit['available_evidence_passed']}；检查{audit['LP_solves']}次保存LP、{audit['physical_models']}个不同模型，HiGHS独立重解{audit['independent_LP_checks']}个选中收缩前家族。输入返回覆盖与原始失败另列。八项单元检查通过。数据、实际计算代码、逐条原始记录与分析代码均保留哈希。",
        '', '```python','from research_experiments.wzzt_paper.records import find, load, load_trace',
        'catalog = r"D:\\0-github_workspace\\Topo\\docs\\wzzt_paper_20261029\\catalog.sqlite"',
        'm = load(catalog, panel="main", case="flynn16", condition="reference",',
        '         voltage_sigma=0.001, bundle=3, N=288, variant="ridge_union", with_observations=True)',
        'R, X = m["R"], m["X"]','```','',
        '大体积逐条表与SQLite在本机保留原文件，同时压缩到records.zip提交；从仓库重新取用可调用records.restore恢复。原始观测、checkpoint、完整搜索与LP原生记录位于outputs/wzzt_paper_20261029_full，输入元数据和哈希供定位。论文复用时保留失败分母及本节的网络/观测适用域。',
        '', '```powershell',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.prepare outputs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.run outputs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.free_domain outputs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.evaluate outputs/<new> docs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.free_domain outputs/<new> --evaluate docs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.statistics docs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.audit outputs/<new> docs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.resolution outputs/<new> docs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.timing outputs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.report outputs/<new> docs/<new>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_paper.verify outputs/<new> docs/<new>', '```','']
    (destination/'README.md').write_text('\n'.join(text),encoding='utf-8')
    for name in ('protocol.json','manifest.json','completion.json','data_errors.json'):shutil.copy2(output/name,destination/name)
    for filename,files in (('producer_source.zip',[p for p in (output/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts]),
                           ('analysis_source.zip',list(Path(__file__).parent.glob('*.py')))):
        with zipfile.ZipFile(destination/filename,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files):z.write(p,p.relative_to(output/'source').as_posix() if filename.startswith('producer') else p.name)
    write(destination/'analysis_source_manifest.json',{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    helper=ROOT/'research_experiments/wzzt_budget/audit.py';write(destination/'analysis_helper_manifest.json',{helper.relative_to(ROOT).as_posix():sha(helper)})
    raw=[p for directory in ('jobs','ac','timing','free_domain','lp60_retry','joint_extension') for p in (output/directory).rglob('*') if p.is_file() and p.suffix in ('.json','.log','.npz','.py')]
    write(destination/'result_manifest.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(raw)})
    with zipfile.ZipFile(destination/'records.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in LARGE:z.write(destination/name,name)
    (destination/'.gitignore').write_text('\n'.join('/'+name for name in LARGE)+'\n',encoding='utf-8')
    (destination/'.gitattributes').write_text('* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n',encoding='utf-8')
    write(destination/'delivery_manifest.json',{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='delivery_manifest.json'})
    print('paper report saved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');a=p.parse_args();report(a.output,a.destination)
