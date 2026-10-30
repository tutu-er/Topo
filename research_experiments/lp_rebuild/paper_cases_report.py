"""Report the matched four-network replay without pooling stress and main rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd

from .run import read,write,sha
from research_experiments.wzzt_paper.common import indices
from research_experiments.rebuild_benchmark.report import markdown

ORDER=['pool','original_fullwarm','native_joint','native_corrected','enum_control','best_floor','best_hybrid']
LABELS=['Pool repair','Forward MILP','Old local MILP','Local MILP + root','LP enumeration','LP + floor .04','Pool + joint regularized']


def audit(output):
    output=Path(output).resolve();catalog=read(output/'catalog.json');file=output/'input_audit.json'
    if not file.exists():
        models=[];inputs={}
        for case in ('paper15','soumalas11','flynn16','pengwah18'):
            row=next(v for v in catalog.values() if v['case']==case);net=Path(row['truth_network']);b=pd.read_csv(net/'buses.csv')
            models.append(dict(case=case,bus_types=b.bus_type.value_counts().to_dict(),metadata=read(net/'metadata.json'),
                buses_sha256=sha(net/'buses.csv'),branches_sha256=sha(net/'branches.csv')))
        for p in catalog.values():
            for name in ('train','validation','test'):
                if p[name] not in inputs:inputs[p[name]]=sha(p[name])
        write(file,dict(networks=models,input_hashes=inputs,job_count=len(catalog),
            N_indices={str(N):indices(N).tolist() for N in (24,48,96)},scope='observation hashes; no truth/test data used for tuning'))
    data=read(file);checks=0
    for name,digest in data['input_hashes'].items():assert sha(name)==digest;checks+=1
    for model in data['networks']:
        net=Path(next(v['truth_network'] for v in catalog.values() if v['case']==model['case']))
        assert sha(net/'buses.csv')==model['buses_sha256'] and sha(net/'branches.csv')==model['branches_sha256']
        assert read(net/'metadata.json')==model['metadata'];checks+=3
    write(output/'input_audit_verification.json',dict(passed=True,checks=checks,archives=len(data['input_hashes']),audit_sha256=sha(file)))
    print('INPUT AUDIT',checks,'checks',len(data['input_hashes']),'archives',flush=True)


def report(output,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    analysis=output/'analysis';audit(output)
    verification=read(analysis/'verification.json');assert verification['passed'] and read(analysis/'paper_verification.json')['passed']
    cross=read(analysis/'crosssolver.json');assert cross['passed']
    frozen=read(output/'frozen.json');original=read(output/'original_frozen.json')
    assert frozen['methods']==original['methods'] and frozen['evaluation_settings']==original['evaluation_settings']
    summary=pd.read_csv(analysis/'panel_summary.csv');pairs=pd.read_csv(analysis/'panel_paired.csv');f=pd.read_csv(analysis/'per_model.csv')
    main=summary[(summary.panel=='main')&(summary.view=='bank')].set_index('method').loc[ORDER].reset_index()
    path=summary[(summary.panel=='main')&(summary.view=='path')].set_index('method').loc[ORDER].reset_index()
    raw=summary[(summary.panel=='main')&(summary.view=='raw')].set_index('method').loc[ORDER].reset_index()
    pressure=pd.read_csv(analysis/'stress_conditions.csv');pressure=pressure[pressure.method.isin(ORDER)]
    contrast=pairs[(pairs.panel=='main')&(pairs.view=='bank')&pairs.method.isin(['enum_control','best_floor','best_hybrid'])&pairs.reference.isin(['pool','original_fullwarm','native_joint'])]
    blocks=pd.read_csv(analysis/'paired_bundle.csv');blocks=blocks[(blocks.view=='bank')&(blocks.method=='enum_control')&blocks.reference.isin(['pool','original_fullwarm'])]
    cases=pd.read_csv(analysis/'main_cases.csv');cases=cases[cases.method.isin(ORDER)]
    network=pd.DataFrame([dict(case=q['case'],terminals=q['bus_types']['observed_terminal'],hidden=q['bus_types']['hidden_internal'],
        root=q['metadata']['root_bus'],base_kV=q['metadata']['base_kv'],base_MVA=q['metadata']['base_mva']) for q in read(output/'input_audit.json')['networks']])
    colors=['#2f855a','#805ad5','#718096','#3182ce','#c05621','#d69e2e','#319795']
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.8));x=np.arange(len(ORDER))
    axes[0].bar(x,main.F1,color=colors);axes[0].set_ylim(0,1.02);axes[0].set_ylabel('Mean terminal-support F1')
    axes[1].bar(x,main.search_median,color=colors);axes[1].set_ylabel('Median search seconds')
    for ax in axes:ax.set_xticks(x,LABELS,rotation=40,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_title('Same 108 reference inputs');axes[1].set_title('20s common soft search cap')
    fig.tight_layout();fig.savefig(destination/'comparison.png',dpi=165);plt.close(fig)
    cells=pd.read_csv(analysis/'main_cells.csv');pivot=cells.pivot(index=['voltage_sigma','N'],columns='method',values='F1')
    for method,prefix,filename in [('enum_control','LP enumeration','noise_samples.png'),('best_hybrid','Hybrid','hybrid_noise_samples.png')]:
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
        for ax,reference,suffix in zip(axes,['original_fullwarm','pool'],[' minus forward MILP',' minus pool repair']):
            mat=(pivot[method]-pivot[reference]).unstack('N');limit=max(.01,float(np.abs(mat.to_numpy()).max()))
            im=ax.imshow(mat,origin='lower',cmap='RdYlGn',aspect='auto',vmin=-limit,vmax=limit)
            for i in range(len(mat)):
                for j in range(len(mat.columns)):ax.text(j,i,f'{mat.iloc[i,j]:+.3f}',ha='center',va='center')
            ax.set_xticks(range(len(mat.columns)),mat.columns);ax.set_yticks(range(len(mat)),[f'{v*100:g}%' for v in mat.index])
            ax.set_xlabel('Training snapshots');ax.set_ylabel('Voltage noise');ax.set_title(prefix+suffix,fontsize=11);fig.colorbar(im,ax=ax,label='F1 difference')
        fig.tight_layout();fig.savefig(destination/filename,dpi=165);plt.close(fig)
    selected=pd.DataFrame([dict(method=m,config_id=cid,**frozen['evaluation_settings'][cid]) for m,cid in frozen['methods'].items() if m in ORDER])
    selected.to_csv(analysis/'selected_parameters.csv',index=False)
    events=pd.DataFrame(read(analysis/'events.json'));native=events[events.status.notna()].copy()
    native['incumbent_valid']=native.audit.map(lambda a:bool(a['valid']))
    solver=native.groupby('config_id').agg(calls=('status','count'),status2=('status',lambda s:int((s==2).sum())),
        timeouts=('status',lambda s:int((s==9).sum())),incumbents_valid=('incumbent_valid','sum'),
        median_gap=('relative_gap','median')).reset_index()
    solver.to_csv(analysis/'MIP_status_summary.csv',index=False)
    missing=read(analysis/'verification.json')['failures']
    text=f'''# 四个文献风格算例的冻结参数复测

这是本次对话此前全面对照的同一输入矩阵：主面板 108 输入、压力面板 12 输入，合计 120。每网络 27 个主输入加 3 个压力输入。采用上轮 LP 枚举研究已冻结的参数，未重新调参；已看过的历史输入作为复现与回归测试，不称为新的盲测数据。

## 算例、量测与划分

{markdown(network,['case','terminals','hidden','root','base_kV','base_MVA'],4)}

四者均为文献风格的合成低压径向网络，不是实测馈线或原论文软件复现。根电压已观测，内部节点隐藏且零注入，终端具有同步 P/Q/V；动态负荷和 DER 曲线由代码生成，电压由径向 AC 潮流生成。拟合使用平方电压降 `Vroot²−Vterminal²`，响应系数吸收了线性近似式中的系数 2。

- 主面板：4 网络 × bundle 2/3/4 × 电压噪声 0.02%/0.05%/0.1% × 总训练样本 N=24/48/96，共 108 输入；P/Q 相对噪声为 0.5%。
- 同一物理数据组有三个训练物理日，各日原始 96 点，N 表示跨三日选取后的总样本数，不是每日样本数。验证、测试各为另外三个物理日，共各 288 点。
- 压力：4 网络 × 弱激励/根电压测量噪声/异常点，bundle 2、N=48、电压噪声 0.05%，共 12 输入。实际条件参数保存在 catalog 与原始数据协议；它们不与主面板混合求总体均值。
- 重用前述同一训练前端、候选库和量测文件，144 个不同观测归档及网络文件均校验字节哈希；没有重新抽样生成更容易的数据。

## 固定的方法与预算

{markdown(selected,['method','config_id','strategy','mode','floor','sparsity'],4)}

`best_floor` 与 `best_joint` 在原冻结记录中指向同一配置，本表仅展示一行；相同配置不会被算作额外独立运行。实际 7 个不同方法配置 ×120 输入，共 840 个工作进程。每个搜索软预算 20 秒；单次 MILP 2 秒、固定家族 LP 5 秒。原始逐步 MILP 扫描相容池内暖启动但保留自由二元支撑域；它仍是逐步贪心模型，不是全树全局最优模型。

所有方法均从公共 RNJ 库、自身路径和同一个单点家族备用候选中选最终模型，输出规则固定为 0.16 收缩、训练重拟合、0% 验证带宽。另报告仅路径（含相同单点备用项）和未收缩终点。另报包含前端、初始化与最终选模的总成本估算；并发单线程计时不称为独立顺序计时。

## 108 个主输入

{markdown(main,['method','inputs','available','F1','exact','train_MAE','test_MAE','search_median','total_estimate_median'],6)}

![Comparison](comparison.png)

配对 F1 差值：

{markdown(contrast,['method','reference','inputs','mean_delta','wins','ties','losses'],6)}

按三个 bundle 的平均差：

{markdown(blocks,['method','reference','bundle','inputs','mean_delta'],6)}

主面板只有三个独立 bundle，网络、噪声及样本数条件成对相关，不按 108 次独立重复推断成功概率。

## 分网络、噪声与样本量

{markdown(cases,['case','method','inputs','F1','exact','test_MAE'],6)}

![Noise and sample size](noise_samples.png)

![Hybrid noise and sample size](hybrid_noise_samples.png)

完整单元格值另存 main_cells.csv，每格为四网络×三个bundle的12个输入。差值图比较相同量测条件，不把某一档噪声结果推广到所有设置。

## 12 个压力输入

{markdown(pressure,['condition','method','inputs','F1','exact','test_MAE'],6)}

## 初始化与选模贡献

仅搜索路径输出：

{markdown(path,['method','F1','exact','test_MAE'],6)}

未收缩终点：

{markdown(raw,['method','F1','train_MAE','test_MAE','internal'],6)}

最终输出响应矩阵误差：

{markdown(main,['method','R_error','X_error'],6)}

## 完成与核验

原生MILP子问题的状态汇总：

{markdown(solver,['config_id','calls','status2','timeouts','incumbents_valid','median_gap'],6)}

status2按本轮0.1%相对gap容差终止，不等于所有子问题均达到精确零gap。触时但可行的incumbent可以继续用于路径；“工作进程返回”与“每步精确最优”分开解释。

正常返回 {verification['returned']}/{verification['planned']}，失败 {len(missing)}；失败保留原输入分母，缺失 F1 按 0 计，误差仅在可用结果上平均。逐次 LP/MILP 状态、触时、未决停止原因、界、gap 与原始目标重算保留在记录中。小邻域完整枚举及单步 MILP 的证书不代表整棵树的全局最优性。

独立目标、下界保持、层状性、参数界和代码/输入哈希等 {verification['checks']} 项检查通过；对两个有下界方案的最终固定家族另进行 {len(cross['records'])} 次 HiGHS 独立 LP 重解。计数是程序核查项，不是独立统计证据数。完整表格与协议见 records.zip，本地完整工作记录为 `outputs/paper_cases_lp_rebuild_20261031/`。

复现时使用新输出目录，依赖前述两轮冻结目录与原始合成数据：

```powershell
$py = 'D:/apps/miniconda3/envs/Topo/python.exe'
$replayOutput = 'outputs/paper_cases_lp_replay'
& $py -m research_experiments.lp_rebuild.paper_cases freeze $replayOutput
& $py -m research_experiments.lp_rebuild.paper_cases_report audit $replayOutput
& $py -m research_experiments.lp_rebuild.paper_cases run $replayOutput
& $py -m research_experiments.lp_rebuild.paper_cases analyze $replayOutput
& $py -m research_experiments.lp_rebuild.analyze crosssolver $replayOutput
& $py -m research_experiments.lp_rebuild.paper_cases_report report $replayOutput docs/paper_cases_lp_replay
```
'''
    (destination/'RESULTS.md').write_bytes(text.encode('utf-8'))
    for name in ('panel_summary.csv','panel_paired.csv','paired_bundle.csv','main_cases.csv','main_cells.csv','stress_conditions.csv','selected_parameters.csv','MIP_status_summary.csv'):
        shutil.copy2(analysis/name,destination/name)
    with zipfile.ZipFile(destination/'records.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in analysis.iterdir():
            # Generic scorer diagnostics include all 120 rows; label these
            # separately from the primary 108-row and stress-only summaries.
            aliases={name:'all_inputs_'+name.removesuffix('.csv')+'_diagnostic.csv'
                     for name in ('summary.csv','paired.csv','by_case.csv','by_bundle.csv')}
            if p.suffix in ('.csv','.json'):z.write(p,aliases.get(p.name,p.name))
        for name in ('protocol.json','frozen.json','original_frozen.json','replay_receipt.json','input_audit.json','input_audit_verification.json','evaluation_completion.json'):z.write(output/name,'protocol/'+name)
        for name in ('paper_cases.py','paper_cases_report.py','analyze.py'):z.write(Path(__file__).with_name(name),'controller_source/'+name)
        for name in ('engine.py','run.py'):z.write(output/'source/research_experiments/lp_rebuild'/name,'inference_source/'+name)
    for p in destination.iterdir():
        if p.suffix in ('.csv','.md','.json') and p.name!='delivery_manifest.json':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    receipt={p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}
    (destination/'delivery_manifest.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    print('REPORT',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('audit','report'));p.add_argument('output');p.add_argument('destination',nargs='?');a=p.parse_args()
    if a.command=='audit':audit(a.output)
    else:report(a.output,a.destination)
