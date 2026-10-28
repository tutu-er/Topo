from pathlib import Path
import argparse
import json
import shutil
import zipfile
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha
from .common import MAIN,TARGET,AC_METHODS

ROOT=Path(__file__).resolve().parents[2]


def table(frame,columns):
    def fmt(v):
        if pd.isna(v):return '—'
        return f'{v:.5g}' if isinstance(v,float) else str(v)
    return '\n'.join(['|'+'|'.join(map(str,columns))+'|','|'+'|'.join('---' for _ in columns)+'|']+
                     ['|'+'|'.join(fmt(r[c]) for c in columns)+'|' for r in frame.to_dict('records')])


def report(output,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output).resolve();destination=Path(destination).resolve();protocol=read(output/'protocol.json')
    overview=read(destination/'overview.json');audit=read(destination/'audit_summary.json')
    frame=pd.read_csv(destination/'per_job.csv');main=frame[frame.panel=='main'];target=frame[frame.panel=='targeted']
    summary=pd.read_csv(destination/'main_summary.csv').set_index('variant').loc[list(MAIN)].reset_index()
    targets=pd.read_csv(destination/'target_summary.csv').set_index('variant').loc[list(TARGET)].reset_index()
    grid=pd.read_csv(destination/'by_noise_N.csv');bundles=pd.read_csv(destination/'by_bundle.csv')
    ridge=main[main.variant=='ridge_union'];multi=main[main.variant=='multistart']
    diagnosis=ridge.copy()
    diagnosis['unidentified']=diagnosis.clade_f1<1-1e-12
    diagnosis['unidentified_pool_missing']=diagnosis.unidentified&(diagnosis.candidate_pool_true_recall<1-1e-12)
    diagnosis['unidentified_crossing_pair']=diagnosis.unidentified&diagnosis.diagnostic_json.map(lambda s:bool(json.loads(s)['blocking_pairs']))
    failure=diagnosis.groupby(['voltage_sigma','N']).agg(inputs=('job_id','count'),unidentified=('unidentified','sum'),
        pool_missing=('unidentified_pool_missing','sum'),crossing_pairs=('unidentified_crossing_pair','sum')).reset_index()
    failure.to_csv(destination/'failure_modes.csv',index=False,encoding='utf-8-sig')
    stops=ridge.groupby('search_stop').size().reset_index(name='inputs');stops.to_csv(destination/'ridge_termination.csv',index=False,encoding='utf-8-sig')
    ridge[ridge.search_stop=='pool_wall_limit'].to_csv(destination/'ridge_wall_limit.csv',index=False,encoding='utf-8-sig')
    matrix=ridge.groupby(['voltage_sigma','N']).clade_f1.mean().unstack().reset_index();matrix['voltage_sigma_pct']=matrix.voltage_sigma*100
    matrix.to_csv(destination/'ridge_noise_size_matrix.csv',index=False,encoding='utf-8-sig')
    highcase=ridge[ridge.voltage_sigma==.001].groupby(['case','N']).clade_f1.mean().unstack().reset_index()
    highcase.to_csv(destination/'high_noise_case_matrix.csv',index=False,encoding='utf-8-sig')
    changes=ridge[abs(ridge.delta_vs_multistart)>1e-10]
    changes.to_csv(destination/'ridge_changes.csv',index=False,encoding='utf-8-sig')
    high48=multi[(multi.voltage_sigma==.001)&(multi.N==48)]
    high48.to_csv(destination/'high_noise_N48_vs_original.csv',index=False,encoding='utf-8-sig')
    paired=main.pivot(index='job_id',columns='variant',values='clade_f1')
    sensitivity=dict(union_vs_1e2_equal_rows=int((abs(paired.ridge_union-paired.ridge_1e2)<=1e-12).sum()),
        inputs=len(paired),ridge_vs_multistart_wins=int((ridge.delta_vs_multistart>1e-10).sum()),
        ridge_vs_multistart_losses=int((ridge.delta_vs_multistart < -1e-10).sum()),
        high48_vs_baseline25_wins=int((high48.delta_vs_baseline25>1e-10).sum()),
        high48_vs_baseline25_losses=int((high48.delta_vs_baseline25 < -1e-10).sum()))
    write(destination/'sensitivity.json',sensitivity)
    ac=[read(p) for p in (output/'ac').glob('*.json')]
    ac_summary=dict(planned_rows=int(frame.ac_planned.sum()),evaluated_rows=int(frame.ac_evaluated.sum()),valid_rows=int(frame.ac_valid.eq(True).sum()),
        unique_records=len(ac),infrastructure_errors=sum('error' in r for r in ac),
        valid_unique=sum(r.get('testing',{}).get('valid',False) for r in ac),
        local_converged_unique=sum(r.get('fitting',{}).get('refit',{}).get('success') is True for r in ac),
        methods=AC_METHODS,other_methods_AC_evaluated=False)
    write(destination/'ac_summary.json',ac_summary)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True,layout='constrained')
    colors={'NJ':'#8b9099','baseline25':'#c38a2d','multistart':'#235cb0','ridge_union':'#228c72'}
    for ax,sigma in zip(axes,protocol['voltage_sigmas']):
        for method in colors:
            d=grid[(grid.voltage_sigma==sigma)&(grid.variant==method)].sort_values('N')
            ax.plot(d.N,d.F1,'o-',color=colors[method],label=method)
        ax.set_title(f'Root/terminal V noise SD = {100*sigma:g}%');ax.set_xticks(protocol['sizes'])
        ax.set_ylim(-.02,1.04);ax.grid(alpha=.25);ax.set_xlabel('Total training snapshots across 3 days')
    axes[0].set_ylabel('Mean clade F1: 4 networks x 3 independent bundles')
    axes[2].legend(fontsize=8,loc='lower right');fig.suptitle('Fixed-parameter fine-grid tests on fresh independent data')
    fig.savefig(destination/'noise_size.png',dpi=175);fig.savefig(destination/'noise_size.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    b=bundles[bundles.variant=='ridge_union'].sort_values('bundle')
    ax.bar(b.bundle,b.delta_vs_multistart,color='#228c72');ax.axhline(0,color='gray',lw=1)
    ax.set_xticks([0,1,2],labels=['Bundle 0','Bundle 1','Bundle 2']);ax.set_ylabel('Mean paired F1 gain over multistart')
    ax.set_title('Three independent data bundles; all other conditions paired')
    for i,v in zip(b.bundle,b.delta_vs_multistart):ax.text(i,v+.00015,f'{v:.5f}',ha='center')
    fig.savefig(destination/'bundle_effect.png',dpi=175);plt.close(fig)
    target_ref=multi[(multi.voltage_sigma==.001)&multi.N.isin([24,48])].clade_f1.mean()
    lines=['# 当前流程与 RNJ+WZZT 细化重复测试','',
        f"正式批次 {overview['returned_inputs']}/180 组新输入返回；主面板 {overview['main_rows']}/1620、定向子面板 {overview['target_rows']}/144，共 {overview['selected_records']}/1764 条模型记录。算法、验证规则和阈值冻结，未重新按真簇调参。",
        '', '## 当前实际流程','',
        '完整步骤、公式、参数位置和新观测接口见 [FLOW.md](FLOW.md)。主线为平方压降 → 原始/去均值前置 R/X 与 NJ/RNJ 池 → 完整 RNJ 联合 L1 库和多初值修正 → 正则化扩池追加修正 → 3% 验证简洁规则 → 0.16 最终收缩 → 独立测试评价。后向删除、两步修正保留为实验组件。',
        '', '## 本轮实验设计','',
        '四网络 paper15/soumalas11/flynn16/pengwah18，电压相对高斯噪声标准差 0.02%/0.05%/0.1%，末端和根同档；P/Q 标准差固定 0.5%。N=24/36/48/72/96 为三个物理日总快照数。每个网络用三组独立训练/验证/测试数据：replicate 为 10/11/12、20/21/22、30/31/32。验证和测试各 288 点。',
        '', '同一 bundle 内各噪声档共用物理轨迹及标准正态误差抽样，N 采用嵌套采样；24/48/96 仍与旧均匀网格一致，36/72 为中间补点。各条件配对，三个 bundle 才是独立随机重复单位；结果报告均值、逐重复差值和计数，不把 180 个相关条件当成独立同分布的重复。',
        '', '本轮集中在更小样本和中高噪声，不含 288 点大样本；其总平均与旧 80 组 0.9319 的面板不同。算法效果通过本轮相同输入上的配对差值判断。四网络仍为既有文献启发合成网络，本轮增加了随机实现，未增加未知拓扑。',
        '', '## 主面板结果','', '![噪声与数据量细化](noise_size.png)','',
        table(summary,['variant','rows','F1','exact','delta_vs_multistart','wins','ties','losses','delta_vs_restart','pool_recall']),
        '', '结构指标比较去掉叶簇与全末端簇后的根向末端簇集合，F1=2×共同簇数/(预测簇数+真簇数)；exact 为该集合完全一致的条数，所有方法均保留失败/未恢复分母。',
        '', 'NJ 使用固定 raw/RX75/0.16 几何，普通 RNJ 使用固定 raw/RX75/0.24；RNJ_L1_bank 单独考察联合 L1 与原选模规则。参数/预测比较统一使用固定树联合 L1 重估，普通 NJ/RNJ 的结构保留原生家族；这与旧表中 NJ 的稠密矩阵参数误差口径分开。pool_recall 记录对应算法环境的可用候选池真簇覆盖率；固定 NJ/RNJ 本身不进行池内搜索。',
        '', 'restart_control 与扩池版本从同一个多初值选中模型出发，均最多追加 25 秒，各分支使用独立的新 LP/缓存。LP 共享相同并集列库，但控制分支只允许原池候选，其他列上界为零，确保比较候选域而非单纯追加计算时间。前置正则化矩阵为各 lambda 对照共享，搜索计时不能直接解释为独立部署总耗时。',
        '', '本轮 restart_control 在 180 项的最终 F1 都与 multistart 相同；扩池并集提升 6 项、持平 174 项、零退化。lambda=0.001 单独提升 3 项，lambda=0.01 单独与并集的逐项结构 F1 一致。单 lambda=0.01 可作为减少前置候选成本的精简候选；本轮没有独立测量其单方法端到端耗时。',
        '', '## 按噪声和样本量展开','',
        table(matrix,['voltage_sigma_pct',24,36,48,72,96]),
        '', '上表每格为四网络×三重复，共 12 项平均。0.02% 档当前扩池版全部完整恢复；0.1% 档到 96 点仍明显受限。个别条件随样本量存在回落，例如 0.05% 的 48→72 点，保留原始值而不作单调平滑。',
        '', '高电压噪声 0.1% 下各网络三重复的平均 F1：','',table(highcase,['case',24,36,48,72,96]),
        '', '扩池版仍有 85/180 项未完整恢复：0.02% 档 0 项、0.05% 档 27 项、0.1% 档 58 项。后两档分别有 3/23 项的候选池缺真簇，6/25 项所选错误簇与漏掉真簇交叉；两类可重叠，交叉关系只作最终模型诊断，未据此断言搜索历史中的因果。完整按数据量明细见 failure_modes.csv，所漏真簇和错误簇见 unidentified.csv。',
        '', '扩池追加路径的终止记录如下。pool_local_optimum 表示有限候选及训练改善阈值下停止；pool_wall_limit 表示在候选之间检查时已触及 25 秒路径预算。这一项保留在所有分母中，具体输入见 ridge_wall_limit.csv；180/180 工作进程返回与路径是否触时分别记录。','',table(stops,['search_stop','inputs']),
        '', '## 独立重复与具体改善','', '![逐重复配对增益](bundle_effect.png)','',
        table(bundles[bundles.variant.isin(['multistart','ridge_union'])],['bundle','variant','rows','F1','delta_vs_multistart','wins','losses']),
        '', '三个 bundle 的平均增益均为正，但独立重复数为 3；当前零退化为这批条件的观测结果。以下列出全部六项改善；selected_new_true_clades 为所选模型中原池外真簇数量，仅作事后机制诊断。','',
        table(changes,['case','bundle','voltage_sigma','N','clade_f1','delta_vs_multistart','selected_new_true_clades']),
        '', '## 48 点退化是否重复出现','',
        f"0.1% 电压噪声、48 点的新 12 项中，多初值相对 baseline25 提升 {sensitivity['high48_vs_baseline25_wins']} 项、退化 {sensitivity['high48_vs_baseline25_losses']} 项，其余持平。此前三项退化并非所有 48 点实例都会发生，仍有实例相关的选模风险。",
        '',table(high48,['case','bundle','clade_f1','delta_vs_baseline25']),
        '', '## 后向删除与两步宽度的定向子面板','',
        f"子面板预先固定为四网络×三 bundle×0.1% 电压噪声×24/48 点，共 24 组。其 multistart 基准平均 F1={target_ref:.5f}，与全部 180 组平均分开比较。后向删除只做一轮，delete_all 保留全部提案；guard0/1pct/3pct 仅把收缩前验证误差不超过当前模型 0%/1%/3% 的提案送入原最终选模规则。",
        '',table(targets,['variant','rows','F1','delta_vs_multistart','wins','ties','losses']),
        '', '严格 guard0 在这 24 项提升 7 项、17 项持平、零退化；1% 容差虽平均较高，仍有 4 项退化，3%/不保护有 5 项退化。它是单轮删除、限定高噪声小样本的组件结果，未与扩池输出组合验证，不把两者增益相加。两步宽度从 3 提到 5 均没有改变这 24 项最终 F1。',
        '', '## 参数与独立预测','',
        table(summary,['variant','R_error','X_error','linear_test_mae','AC_planned','AC_evaluated','AC_valid','test_ac_rmse']),
        '', f"AC 仅评价预先固定的 NJ、multistart、ridge_union：计划 {ac_summary['planned_rows']} 项，实际 {ac_summary['evaluated_rows']} 项，可行 {ac_summary['valid_rows']} 项；对应 {ac_summary['unique_records']} 个不同输入/参数组合，基础设施错误 {ac_summary['infrastructure_errors']}，其中 {ac_summary['local_converged_unique']} 个局部重估收敛。其余对照 AC 未评价，0/0 不是失败。结构 F1、线性预测误差和 AC 误差分别记录。",
        '', '## 调用已有结果与当前算法','',
        '记录库增加 bundle 维度，使用本轮专用读取接口：','', '```python',
        'from research_experiments.wzzt_fine.records import load, load_trace',
        'catalog = r"D:\\0-github_workspace\\Topo\\docs\\wzzt_fine_20261028\\catalog.sqlite"',
        'm = load(catalog, case="flynn16", voltage_sigma=0.0005, bundle=0, N=24,',
        '         variant="ridge_union", with_observations=True)',
        'R, X = m["R"], m["X"]',
        'P, Q, V, root = m["train"]',
        'trace = load_trace(catalog, case="flynn16", voltage_sigma=0.0005, bundle=0, N=24)', '```','',
        'per_job.csv 保存全部 1764 条；by_noise_N.csv、by_case_noise_N.csv、by_bundle.csv 保留不同汇总层次；changes_vs_multistart.csv 和 unidentified.csv 保留所有变化和未完整恢复项。完整搜索、模型库、正则化前置矩阵、LP 原生记录位于运行目录，并绑定字节哈希。',
        '', '## 独立核验与复现','',
        f"核验通过：{audit['all_passed']}。检查 {audit['data_checks']} 项数据/域约定、{audit['LP_solves']} 条 LP 记录、{audit['physical_model_checks']} 个不同物理模型、{audit['ridge_fronts']} 个正则化前置矩阵，并用 HiGHS 独立重解 {audit['independent_LP_checks']} 个选中收缩前家族 LP。四项新单元检查通过；新入口另在两例历史数据上复现 baseline25/multistart 四项已保存 F1。",
        '', '```powershell',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_fine.prepare outputs/<新目录>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_fine.run outputs/<新目录>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_fine.evaluate outputs/<新目录> docs/<新目录>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_fine.audit outputs/<新目录> docs/<新目录>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_fine.report outputs/<新目录> docs/<新目录>',
        '& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_fine.verify outputs/<新目录> docs/<新目录>', '```','']
    (destination/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    for name in ('protocol.json','manifest.json','completion.json'):shutil.copy2(output/name,destination/name)
    pilot=ROOT/'outputs/wzzt_fine_20261028_pilot/legacy_parity.json'
    if pilot.exists():shutil.copy2(pilot,destination/'legacy_parity.json')
    with zipfile.ZipFile(destination/'producer_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(read(output/'manifest.json')):
            if name.startswith('source/'):z.write(output/name,name.removeprefix('source/'))
    with zipfile.ZipFile(destination/'analysis_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(__file__).parent.glob('*.py')):z.write(p,p.name)
    write(destination/'analysis_source_manifest.json',{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    helper=ROOT/'research_experiments/wzzt_budget/audit.py';write(destination/'analysis_helper_manifest.json',{helper.relative_to(ROOT).as_posix():sha(helper)})
    files=[p for folder in ('jobs','ac') for p in (output/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.log')]
    write(destination/'result_manifest.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(files)})
    (destination/'.gitattributes').write_bytes(b'* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n')
    write(destination/'delivery_manifest.json',{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='delivery_manifest.json'})
    print('report saved',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');a=p.parse_args();report(a.output,a.destination)
