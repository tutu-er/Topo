"""Rebuild the Chinese research report from retained real-feeder checkpoints."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'theft_wzzt/outputs/theft_real65037'
REPORT=ROOT/'docs/theft_real_feeder_known_reference_20260920.md'


def load_rows(version):
    rows=[]
    for path in sorted((RESULTS/version).glob('seed*.json')):
        data=json.loads(path.read_text())
        if 'fits' in data:rows.append(dict(file=path.name,**data))
    return rows


def main():
    variants={name:load_rows(name) for name in ('v1_scalar','v2_phase','v3_tree_selection')}
    records=[]
    for variant,rows in variants.items():
        for row in rows:
            for name,fit in row['fits'].items():
                records.append(dict(variant=variant,file=row['file'],seed=row['seed'],balanced=row['balanced'],condition=row['condition'],method=name,
                                    status=fit['status'],mip_gap=fit.get('mip_gap'),source_support=fit.get('source_support'),
                                    source_region_correct=fit.get('source_region_correct'),r_relative_error=fit.get('r_relative_error'),
                                    x_relative_error=fit.get('x_relative_error'),r_excitation_weighted_error=fit.get('r_excitation_weighted_error'),
                                    x_excitation_weighted_error=fit.get('x_excitation_weighted_error'),
                                    all_tree_subproblems_optimal=fit.get('all_tree_subproblems_optimal'),
                                    heldout_mean_voltage_mae_v=fit.get('heldout_clean_mae_v_approx'),
                                    heldout_phase_voltage_mae_v=fit.get('heldout_phase_mae_v_approx'),
                                    clade_precision=fit.get('clade_precision'),clade_recall=fit.get('clade_recall'),
                                    amplitude_mae_kw=row['amplitude_mae_kw']))
    unique_solves=[]
    for version in variants:
        for path in sorted((RESULTS/version).glob('seed*.json')):
            data=json.loads(path.read_text())
            if version=='v3_tree_selection':
                fits=[data[k] for k in ('topology_only','theft') if k in data] if 'tree_index' in data else ([data] if 'status' in data else [])
            else:fits=list(data['fits'].values()) if 'fits' in data else ([data] if 'status' in data else [])
            for fit in fits:unique_solves.append(dict(variant=version,file=path.name,**{k:fit.get(k) for k in ('status','seconds','mip_gap','audit')}))
    summary=dict(completed_scenarios={k:len(v) for k,v in variants.items()},records=records,unique_solves=unique_solves)
    (RESULTS/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text=[r'''# 已知参考电压下的真实 65037 馈线：联合 wzzᵀ 与窃电选择分析

日期：2026-09-20。所有 Python 运行均使用 `D:\apps\miniconda3\envs\Topo\python.exe`。

## 1. 结论应如何理解

本轮已经从四终端线性原型推进到**公开真实网络结构上的三相四线 AC 仿真**。参考点电压逐时已知，直接使用绝对压降，未减均值，也没有逐电表自由截距。

结果支持“把窃电位置选择加入 wzzᵀ 拟合，可以在部分工况纠正忽略窃电造成的偏差”；尚不支持“联合方法普遍优于充分利用正常历史的两阶段方法”，更不支持已恢复全部物理线路或已达到现场部署精度。下面同时保留正面案例、失败案例和未闭合的整数最优性缺口。

## 2. 数据来源、假设与真实程度

- 来源：[作者公开网络数据](https://github.com/rubencarmona/lvnetworkdataset)，固定提交 `a710c7212324f5085e1c6536714dc838f13ed81b`；原始文件及 SHA256 见工作区 `data_external/lvnetworkdataset_20260920/manifest.json`。
- 65037 网络含 53 母线、52 支路、5 条根节点出线。本地可匹配 14 个节点的功率记录，总净负荷 260.925 kW。节点 40、48 有负荷标记但缺少功率记录，本轮明确设为零，未伪造实测值。
- 采用公开电缆 4×4 复阻抗矩阵，保留相间耦合和中性线。阻抗单位为 Ω/km；支路长度按 m 解释，长度和为 841.74 m。README 未直接注明长度单位，仍保留这一假设。
- 有功负号按作者 `loads.py` 转为用电正号。保留原始各相净注入，包括部分相别反送电；另生成总负荷不变的三相平衡对照。
- 时序是公开单一功率快照乘随机因子并增加无功激励形成；窃电也是合成。**没有实测窃电标签，也没有现场时序验证。**
- AC 模型采用恒功率负荷、参考点中性线接地、无下游附加接地和无并联支路；未给出的接地/相别资产信息没有补造。全部节点按四导体模型模拟，不能称为原论文 208 导体节点案例的逐项复现。
- [PSCC 2026 原论文](https://pscc.epfl.ch/modules/request.php?module=oc_program&action=view.php&id=34&file=1/34.pdf)说明其采用北西班牙改编网络及合成窃电；本轮是依据公开数据自行建模，未复现该文优化器，也未做与该文数值成绩的公平性能对比。

## 3. 保留电压参考，怎样使用 wzzᵀ

对每个电表、相别直接构造

\[
y_{it}^{\phi}=\frac{|V_{0t}^{\phi}|^2-|V_{it}^{\phi}|^2}{2000},
\]

其中电压单位 V、功率单位 kW。参考幅值约 230 V，并随时间变化且被完整观测；参考三相角间隔 120°是模型假设。不存在可吸收持续窃电的自由截距。

每条候选约化支路仍由下游电表指示向量 \(z_e\) 表示。三相不另选三棵树，而共用同一个 \(z_e\)。对本数据的相对称电缆，消去中性线电流后的相—中性线阻抗为

\[
W_e=Z_{pp,e}-Z_{pn,e}\mathbf1^\top-\mathbf1Z_{np,e}
       +Z_{nn,e}\mathbf1\mathbf1^\top
   =z_{1e}I+z_{ne}\mathbf1\mathbf1^\top.
\]

这里 \(z_{1e}=Z_{aa,e}-Z_{ab,e}\)，\(z_{ne}=Z_{nn,e}+Z_{ab,e}-Z_{an,e}-Z_{na,e}\)。必须保留相—中性线互阻抗与相间互阻抗的差异。复电压—电流传递矩阵具有

\[
\mathcal Z=\sum_e(z_ez_e^\top)\otimes W_e
\]

的块结构；实际拟合平方电压时进一步使用 120°相位旋转和一阶线性化。因此这是 **wzzᵀ 的按相扩展**，并非把原标量公式直接当作完整不平衡模型。

窗口内一个固定位置的窃电用 \(s_h\in\{0,1\},\sum_hs_h=1\) 选择已有候选位置；\(A_{eh}=1\{z_h\subseteq z_e\}\)，\(b_e=\sum_hA_{eh}s_h\)。窃电增量沿所选位置的整条祖先路径注入。对 \(r_eb_e,x_eb_e\) 使用有界精确 big-M 线性化，同时重估正常网络参数。当前窃电假设为三相均分、已知 \(Q/P=0.4\)。

总幅值来自观测总表—分表差减去估计线损，没有将仿真真实窃电量或真实线损交给逆模型。按相改进版用正常历史学到的参数估计相线及中性线 I²R 损耗，并只用正常历史总分表差校准一个比例系数。

## 4. 三轮实施与简化后的执行逻辑

1. **V1 标量对照**：14 电表平均平方电压，有限 clade 池上的联合拓扑/阻抗/位置 MILP；64 个当前样本，192 个正常历史样本仅用于候选和线损；独立 96 点验证。平衡及原始不平衡各做 null、15 kW 持续、30 kW 半窗口间歇三个场景。
2. **V2 按相全池搜索**：同一组 z 同时控制各相与中性线参数，把 128 点正常历史与 64 点当前窗口共同拟合。首次全池联合搜索在 45 s 时最优性缺口约 73%，因此中止这一实现路线，保存已完成场景和中止记录。
3. **V3 完整候选树选择**：用前 96 点正常数据生成 RNJ 树，以后 32 点正常数据的相电压预测排序，仅保留 3 棵完整候选树。逐树求解位置整数 MILP，比较目标值；不再在庞大的任意 clade 组合中搜索。当前观测和独立验证集都不参与候选树的预筛选。
4. 在每个 V3 场景中，两阶段方法先忽略窃电选择一棵树，再固定树联合定位与重估参数；联合方法同时比较各树的窃电模型。两者使用相同观测、幅值、候选树、参数界与惩罚。
5. 所有结果按树保存检查点。`optimal` 仅指该有限模型在设置容差内求解完成；`incomplete` 只代表已校验的可行解。只有所有相关候选子问题求解完成，才使用“已完成有限池搜索”的说法；这仍不是全部可能物理树的全局恢复。

V1→V3 同时改变了按相建模、历史数据使用、线损估计、正则与候选搜索，属于组合改进，不能把跨版本差异全部归因于其中一个因素。V3 内部的联合与两阶段比较才具有匹配的信息条件。

## 5. 已保存结果
''']
    text.append('\n完成场景数：'+ '；'.join(f'{k}={len(v)}' for k,v in variants.items())+'。V3 预定 2 个种子 × 2 类物理工况 × 3 个条件，共 12 个场景。\n')
    for version in ('v1_scalar','v3_tree_selection'):
        text.append(f'\n### {version}\n\n')
        text.append('|种子/工况/条件|幅值 MAE (kW)|定位：两阶段/联合|验证电压 MAE：忽略窃电/两阶段/联合 (V)|联合求解状态|\n|---|---:|---|---|---|\n')
        for row in variants[version]:
            fits=row['fits'];names=('topology_only','sequential','joint')
            metric='heldout_clean_mae_v_approx' if version=='v1_scalar' else 'heldout_phase_mae_v_approx'
            vals='/'.join(f"{fits[k].get(metric,float('nan')):.4f}" for k in names)
            if row['condition']=='clean':location='条件式位置输出，不计检测成功率'
            else:location='/'.join('正确' if fits[k].get('source_region_correct') else '错误' for k in names[1:])
            status=fits['joint']['status']
            if version=='v3_tree_selection':status+='；全池完成='+str(fits['joint']['all_tree_subproblems_optimal'])
            text.append(f"|{row['seed']}/{'平衡' if row['balanced'] else '不平衡'}/{row['condition']}|{row['amplitude_mae_kw']:.3f}|{location}|{vals}|{status}|\n")
    heldout_path=RESULTS/'selected_z_heldout/summary.json'
    if heldout_path.exists():
        validation=json.loads(heldout_path.read_text())
        text.append('\n### 固定已有 z 选择：预先固定协议后的独立测试窗口\n\n')
        text.append('两个正常历史模型 × 平衡/不平衡 × 母线5/10两个单点位置 × 持续/间歇 × 20个新测试种子，共320次评估。每个窗口64个AC样本；160个工况窗口分别由两个历史模型评估，评估间存在配对相关，不能视为320个独立同分布试验。没有同时发生多点窃电。\n\n')
        text.append('|物理工况|唯一选对约化区域|并列最优集合覆盖真值|前三候选覆盖真值|\n|---|---:|---:|---:|\n')
        for balanced in (True,False):
            groups=[g for g in validation['groups'] if g['balanced']==balanced]
            total=sum(g['evaluations'] for g in groups)
            counts=[sum(g[k] for g in groups) for k in ('unique_correct','minset_coverage','top3_coverage')]
            text.append(f"|{'平衡' if balanced else '不平衡'}|{counts[0]}/{total}|{counts[1]}/{total}|{counts[2]}/{total}|\n")
        text.append('\n各组幅值MAE为0.392–0.724 kW。所有试验位置的真实clade在候选树中；结果不能外推至候选遗漏的位置。前三候选是固定长度的排名列表，不是95%置信域，也不是检测成功率。该批结果支持区域筛查的可行性，尚不支持现场窃电事件准确率。\n')
        text.append('\n现阶段更简洁且已有验证的执行逻辑：**正常历史构造按相wzzᵀ → 校准相线/中性线损耗 → 以总分表差估计幅值 → 在已有z上作整数选择 → 输出并列区域及前三检查候选**。联合重估保留为增强步骤，需使用这个可行点作初始解并收紧数据支持的参数界，不能直接接受超时求解器返回的较差标签。\n')
    diagnostic=RESULTS/'frozen_z_diagnostic.json'
    if diagnostic.exists():
        discrete=json.loads(diagnostic.read_text())
        text.append('\n### 固定历史参数、只对已有 z 做精确整数选择\n\n')
        text.append('这是独立诊断基线：正常历史识别出的参数固定，位置严格 one-hot；没有连续位置混合。逐候选直接计算目标并与整数最优值核对。\n\n')
        text.append('|种子/工况/条件|全部并列最优 z（电表索引）|包含真实 z|唯一正确|\n|---|---|---|---|\n')
        for row in discrete['rows']:
            text.append(f"|{row.get('seed',0)}/{'平衡' if row['balanced'] else '不平衡'}/{row['condition']}|{row['tied_minimum_supports']}|{row['truth_in_minimum_set']}|{row['uniquely_correct']}|\n")
        improvements=sum(r.get('historical_point_improves_joint_incumbent',False) for r in discrete['rows'])
        text.append(f'\n可行点审计：{improvements} 个原窗口中，固定历史参数的可构造可行点已优于联合MILP的超时当前解。这证明部分错误来自求解尚未完成；它不证明完整联合模型的最优解更差。\n')
        text.append('\n并列容差为原始目标 1e-9，仅为数值等价检查，不是统计置信区域。种子0不平衡中，通往39号电表的相序 r/x 被估成0，因此 `{2}` 与 `{2,3,4}` 对当前三相均分窃电响应相同。这是**估计模型下的等价**，不是证明真实物理支路不存在。\n')
        text.append('\n额外的真值物理诊断显示，在原始不平衡快照的85%负荷、15 kW窃电下，母线5与39的最大相电压差约0.0486 V，而单次相电压噪声标准差约0.046 V。多时刻平均仍可能提高可分辨性，所以这些数值不能用来宣称理论上不可定位；它们说明需要控制模型偏差、参数边界及定位分辨率。该诊断仅用于解释结果，未送入候选生成或拟合。\n')
    text.append(r'''
V1 验证指标为平均平方相电压换算的近似 V 误差；V3 为各相独立误差的平均，**两列表不能直接做跨版本数值优劣比较**。所有表中 V 误差均以 230 V 一阶换算，原始目标是平方电压误差。

Null 仍调用条件式“窗口内一个位置”模型作压力测试。输出位置不是告警，不构成误报率统计。旧 M1 的 q95≈5.455 不适用于新目标和新观测维度，本轮没有修改旧校准或宣称新的 5% 误报控制。

## 6. 发现的边界及下一步

- **已有一个直接正例**：V1 平衡间歇窃电中，联合选择定位到正确 clade `{2,3,4}`，两阶段停留在邻近 `{1,2,3,4}`；对应电表编号分别为 `{39,41,42}` 与 `{38,39,41,42}`。这组相关 MILP 求解完成，说明允许拓扑选择与窃电解释相互调整有实际价值；一个种子不足以证明统计优势。
- **不平衡线损是关键混淆项**：V1 在无窃电的不平衡工况得到约 14.49 kW 的正幅值残差。幅值来自总分表差也不等于窃电真值。按相与中性线损耗必须进入模型，且还需计入窃电自身增加的线损和表计误差。
- **弱激励使“全部阻抗恢复准确”不成立**：V3 种子0平衡正常工况，51号电表平均功率约54 W、标准差约11 W；其一列贡献约99.7%的电阻矩阵平方误差。原始 R 矩阵相对误差约265%，而按独立负荷激励加权的响应误差约1.6%。两者都保留在 summary.json；不能只报告较好的一项。需要额外激励、线路先验或合并该不可可靠估计参数。
- **强不平衡仍有模型误差**：原始快照在本接地假设下出现较大中性线偏移，电压范围超出平衡工况。按相模型仍是一阶近似，没有估计全部相角，也没有闭环 AC 修正；不能把线性模型的有限池最优当成 AC 真实状态最优。
- **拓扑仅在约化电表空间辨识**：隐藏二度链、未观测支路、微小公共阻抗和中间电表的零长度等价末端都会限制物理定位；所选 z 是区域标识，不是所有物理母线的一一编号。候选池缺失真 clade 时，优化器不能凭空补回。
- **尚未验证**：多点同时窃电、单相窃电、未知功率因数、未知参考电压、异步量测、真实历史时序及现场标签；没有与现有文献做同信息条件下的算法性能比赛。

优先改进顺序：先对历史数据给出的 R/X 可辨识范围收紧有效 big-M，并报告区间而不是全部点参数；再用当前数据的测量噪声建立位置候选置信区域和独立零分布；然后加入按当前位置估计的新增线损及 AC 校正，最后扩展多位置离散份额模型。增加种子不能替代修正这些结构性问题。

论文层面，这一轮提供了“真实馈线结构、已知参考电压、窃电与阻抗混淆、三相损耗导致错误归因”的完整案例链，但尚不足以单独支撑广泛准确性或现场有效性的主张。更合适的贡献表述是：**具有共同拓扑支撑的按相 wzzᵀ 建模及离散窃电区域选择，连同可辨识性与计算证书的边界分析。**

## 7. 验证和复现

新增独立物理/优化检查共7项：二母线解析 AC 解、全馈线功率平衡、平均平方电压一阶 wzzᵀ、持续窃电整数乘积、三相块一阶一致性、两版输入配对一致性、按相整数定位。均通过。中性线等效阻抗测试曾发现测试基准遗漏互阻抗差异，已修正并重新通过；没有降低误差阈值掩盖问题。

每次有解的拟合均直接校验原始预测目标、big-M 乘积、线性约束/界/整数可行性。AC 检查包括支路欧姆定律和全网复功率平衡；未与第三方潮流软件逐节点交叉核验。原 theft MILP、零分布链、缓存和原单测没有修改，原测试本轮未重跑。

```powershell
Set-Location 'D:\0-github_workspace\Topo'
& 'D:\apps\miniconda3\envs\Topo\python.exe' -m pytest scripts/tests/test_real_feeder_joint_theft.py -x -q
& 'D:\apps\miniconda3\envs\Topo\python.exe' scripts/real_feeder_tree_selection.py --output theft_wzzt/outputs/theft_real65037/v3_tree_selection --seeds 2 --samples 64 --seconds 30
& 'D:\apps\miniconda3\envs\Topo\python.exe' scripts/report_real_feeder_theft.py
```

已存在的同协议检查点会续用；协议或脚本哈希变化时应使用新输出目录。V2 因求解缺口被中止，其 `execution_status.json` 保留原因；勿把该目录当作完成的矩阵。
''')
    counts={v:dict(total=sum(s['variant']==v for s in unique_solves),optimal=sum(s['variant']==v and s['status']=='optimal' for s in unique_solves)) for v in variants}
    text.append('\n唯一已保存求解计数（不重复计算被选中的副本）：`'+json.dumps(counts,ensure_ascii=False)+'`。\n')
    text.append('\n## 8. 图与完整输出\n\n![真实馈线与线损诊断](D:/0-github_workspace/Topo/theft_wzzt/outputs/theft_real65037/feeder_and_loss_diagnostic.png)\n\n结果汇总：`theft_wzzt/outputs/theft_real65037/summary.json`；各版本目录保留协议、逐树检查点、最优性缺口与直接校验误差。图另附可编辑 SVG。\n')
    REPORT.write_text(''.join(text),encoding='utf-8')
    print(json.dumps(dict(report=str(REPORT),counts=counts,completed=summary['completed_scenarios']),ensure_ascii=False))


if __name__=='__main__':main()
