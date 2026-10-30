"""Deliver the frozen LP-enumeration experiment, including negative results."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd

from .run import read,write,sha
from research_experiments.rebuild_benchmark.report import markdown

LABELS=dict(pool='Single pool repair',native_joint='Original local MILP',native_corrected='Local MILP + root',
    original_fullwarm='Forward MILP / full warm',enum_control='LP enumeration + root',best_floor='Selected floor',
    best_joint='Selected joint',best_hybrid='Selected hybrid')


def report(output,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    analysis=output/'analysis';protocol=read(output/'protocol.json');frozen=read(output/'frozen.json');verification=read(analysis/'verification.json')
    assert verification['passed']
    assert read(analysis/'crosssolver.json')['passed']
    f=pd.read_csv(analysis/'per_model.csv');s=pd.read_csv(analysis/'summary.csv');pairs=pd.read_csv(analysis/'paired.csv')
    bank=s[s.view=='bank'].copy();path=s[s.view=='path'].copy();raw=s[s.view=='raw'].copy()
    selected=pd.DataFrame([dict(method=k,config_id=v['config_id'],**v['config'],development_log_ratio=v['mean_log_ratio'],development_internal=v['mean_internal']) for k,v in frozen['selected'].items()])
    dev=pd.DataFrame([{k:v for k,v in r.items() if k not in ('config','records')}|r['config'] for r in frozen['grid']])
    selected.to_csv(analysis/'selected.csv',index=False);dev.to_csv(analysis/'development_grid.csv',index=False)
    covered=pd.read_csv(analysis/'old_domain_coverage.csv');edges=pd.read_csv(analysis/'old_edge_strength.csv')
    strength=edges.groupby(['correct','added']).agg(supports=('job','count'),median_depth=('normalized_depth','median'),
        below016=('normalized_depth',lambda q:int((q<.16).sum()))).reset_index()
    order=['pool','native_joint','native_corrected','original_fullwarm','enum_control','best_floor','best_joint','best_hybrid']
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.7));colors=['#2f855a','#718096','#3182ce','#805ad5','#c05621','#d69e2e','#c53030','#319795']
    x=np.arange(len(order));v=bank.set_index('method').loc[order]
    axes[0].bar(x,v.F1,color=colors);axes[0].set_ylim(0,1.03);axes[0].set_ylabel('Mean terminal-support F1')
    block=pd.read_csv(analysis/'by_bundle.csv')
    for i,m in enumerate(order):axes[0].scatter([i-.06,i+.06],block[block.method==m].F1,marker='.',color='#222222',zorder=3)
    axes[1].bar(x,v.search_median,color=colors);axes[1].set_ylabel('Median search seconds')
    for ax in axes:
        ax.set_xticks(x,[LABELS[m] for m in order],rotation=45,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_title('16 held-out inputs; dots are two bundle means');axes[1].set_title('20s common soft search cap')
    fig.tight_layout();fig.savefig(destination/'comparison.png',dpi=160);plt.close(fig)
    comparisons=pairs[(pairs.view=='bank')&pairs.method.isin(['enum_control','best_floor','best_joint','best_hybrid'])&pairs.reference.isin(['pool','native_corrected','original_fullwarm'])]
    text=f'''# 联合 LP 枚举：下界、稀疏性与先逐步后联合

本轮目标是解释联合重构为何退化，并在开发集选择改动后评价未用于此前局部法调参的 bundle 5/6。它是既有合成 AC 数据上的小规模扩展，不是新增现场数据。开发 16 输入，每个比较 11 个设置；留出 16 输入，四网络、两 bundle、两档噪声、N=48。两 bundle 内条件相关，不当作 16 个独立重复。

## 先修正搜索域，再比较算法

旧终点的 16 次完整池扫描中，{int(covered.exceeds_gate.sum())} 次仍有超过共同接收门槛的单步改善，其中 {int((covered.exceeds_gate&~covered.covered_by_one_local_region).sum())} 次不在任何旧四块邻域内。两个漏掉的改善是全集响应项，另一个涉及四块不能形成的集合。全集不属于局部真子集候选，零权重清理后又可能从保留家族中消失，导致旧联合实现无法重新添加。

本轮让全集响应系数在每个局部问题中始终可用，其系数可以为零，不计入非平凡内部连接数。另加入 `native_corrected`，在完全相同的修正邻域中解二元 MILP，区分修正搜索域与更换求解方式的效果。

四块有 26 个合法层级；新虚拟区域的整体连接再枚举有/无，因此至多 52 个固定家族 LP。每个 LP 仍联合重拟合外围和局部所有权重。列矩阵按区域一次扩充、固定家族拟合缓存复用，保持相同候选域和目标。

## 为什么考虑下界，又为什么不能只抬高 R/X

旧联合终点中新增正确/错误连接的强度分布：

{markdown(strength,['correct','added','supports','median_depth','below016'],6)}

强度为 `(a*r+b*x)/d`，`a,b,d` 由训练前置 RX75 标定。弱假连接多，为分辨率下界提供了动机，但真连接也可能很弱。该量不是欧姆值；响应系数吸收了近似式中的系数 2。没有线路长度、线型等硬先验时，下界是模型选择假设。

还核验了一个反例：Q=0 时，`a*r+b*x >= delta*d` 可以通过增加完全不影响预测的 X 系数满足。因此常数 R/X 下界不等于量测中有足够可识别信号。

本轮加入固定方向的量测响应投影。将某连接的 P/Q 响应列按时间中心化，选取训练数据确定的单位方向 u，计算 `gP=<u,AP>/sqrt(m)`、`gQ=<u,AQ>/sqrt(m)`，约束 `gP*r+gQ*x >= eta*f0`。由 Cauchy–Schwarz，实际中心化响应 RMS 至少为这个投影。Q 无激励时 gQ=0，不能用任意 X 逃过下界。f0 是初始训练 MAE，并非已知噪声标准差；该约束也没有排除其他连接对同一信号的解释。

## 开发集参数选择

深度下界 delta 为 0/0.04/0.08/0.16；投影下界 eta 为 0.5/1；连接数惩罚系数为初始 MAE 的 0.1%/0.5%。稀疏设置另有很小的 0.001 激活深度下界，确保固定家族的连接数等于非零内部响应项数。另比较无正则的先逐步后联合，以及 delta=0.08、惩罚0.1%的混合设置。所有搜索仍用 0.1% 改善门槛，总搜索软预算 20 秒。

连接数惩罚明确改变目标为 `原始训练MAE + lambda*f0*连接数`，所以同时报告原始 MAE。普通深度/投影下界仅限制参数域，固定家族仍最小化原始 MAE。下界贯穿搜索与最终重拟合，最终没有使用无下界 LP 抹掉它。最终收缩阈值固定为 0.16、验证带宽固定为 0%，不再按测试 F1 调这些参数。

{markdown(selected,['method','config_id','strategy','mode','floor','sparsity','development_log_ratio'],6)}

{markdown(dev,['config_id','available','mean_log_ratio','mean_internal','mean_seconds'],6)}

只有开发验证 MAE 参与参数选择；距最佳值 0.1% 内偏向内部集合更少的设置。最优联合、最优非零下界、最优混合三种参数分别冻结。相同配置只执行一次，表中同配置方法是别名，不是额外独立试验。

## 留出结果：公共 RNJ 库加搜索路径

{markdown(bank,['method','config_id','inputs','available','F1','exact','test_MAE','search_median'],6)}

{markdown(comparisons,['method','reference','inputs','mean_delta','wins','ties','losses'],6)}

![Held-out comparison](comparison.png)

## 仅路径、未收缩终点与响应矩阵误差

仅路径输出：

{markdown(path,['method','F1','exact','test_MAE'],6)}

未收缩终点：

{markdown(raw,['method','F1','train_MAE','test_MAE','internal'],6)}

带公共库最终输出的矩阵误差：

{markdown(bank,['method','R_error','X_error'],6)}

下界可以减少小假连接，也可以排除弱真连接、迫使错误连接系数卡在下界。验证预测与拓扑 F1 仍是不同目标；投影下界保证单个项自身有幅值，不保证该项相对其他项不可替代。

## 证据与复现

旧域开发试验完整保留在 `outputs/lp_rebuild_regularization_20261031/`，176/176 返回，没有读取本轮留出真拓扑/测试评分。发现根项遗漏后，新目录重新冻结修正模型并重新执行开发校准；未覆盖旧代码或结果。正式结果位于 `outputs/lp_rebuild_regularization_20261031_corrected/`。

最终工作进程 {verification['returned']}/{verification['planned']} 返回；独立重算和域/下界检查 {verification['checks']} 项通过。32 个已冻结输出家族另用 HiGHS LP 独立重解，保留相同响应下界。原生 MILP 的状态、界与 gap 及 LP 的状态和不可行分支保留在 records.zip。完整局部枚举仅在全部家族完成且 LP 求解通过时有该邻域的最优值，不是整棵树的全局最优证明。单个家族因下界不可行时跳过，LP 超时不当作不可行。

在 Topo 环境执行：`python -m research_experiments.lp_rebuild.run freeze <new>`，依次运行 `prepare`、`development`、`select`、`evaluation`；然后运行 `python -m research_experiments.lp_rebuild.analyze old <new>`、`evaluation <new>`、`crosssolver <new>`，最后 `python -m research_experiments.lp_rebuild.report <new> <docs>`。输入依赖既有合成 AC 数据；代码和输入哈希、参数冻结文件及逐输入结果清单均保留。
'''
    (destination/'RESULTS.md').write_bytes(text.encode('utf-8'))
    for name in ('summary.csv','paired.csv','selected.csv','development_grid.csv','by_case.csv','by_bundle.csv','old_domain_coverage.csv'):
        shutil.copy2(analysis/name,destination/name)
    with zipfile.ZipFile(destination/'records.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in analysis.iterdir():
            if p.suffix in ('.csv','.json'):z.write(p,p.name)
        for name in ('protocol.json','frozen.json','development_completion.json','evaluation_completion.json','prepare_completion.json'):z.write(output/name,'protocol/'+name)
        for p in (output/'source/research_experiments/lp_rebuild').glob('*.py'):z.write(p,'inference_source/'+p.name)
        for name in ('analyze.py','report.py'):z.write(Path(__file__).with_name(name),'analysis_source/'+name)
    for p in destination.iterdir():
        if p.suffix in ('.csv','.md','.json') and p.name!='delivery_manifest.json':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    manifest=destination/'delivery_manifest.json';manifest.write_bytes((__import__('json').dumps({p.name:sha(p) for p in destination.iterdir() if p.is_file() and p!=manifest},indent=2)+'\n').encode())
    print('REPORT',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');a=p.parse_args();report(a.output,a.destination)
