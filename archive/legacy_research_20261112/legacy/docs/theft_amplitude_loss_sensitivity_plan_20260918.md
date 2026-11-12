# 幅值估计误差量化与线损估计误差敏感性 — 调研与实施方案（M7）

日期：2026-09-18
状态：**待人工审阅**（审阅通过后再动代码）
关联文档：`wzzt_credibility_joint_theft_model_20260917.md`（两模块设计）、`theft_wzzt_independent_folder_plan_20260917.md`（独立包方案）

---

## 0. 摘要（结论先行）

当前偷电辨识的幅值输入 `â_t = clip(P0_measured − ΣP_terminal − l̂_t, 0)` 使用**理想反事实线损** `loss_p_notheft`——这假设现场能精确知道"无偷电时的技术线损"，是本方法对现场不确定性最敏感的一处假设。文献调研确认：能量平衡法（总表−分表和−技术线损）是工业标准 NTL 路线，而其**公认瓶颈正是技术线损的精确计算**。

本方案做三件事，全部落在已有接口上、不改 MILP 本体：

1. **线损估计器变体 L0–L4**：把固定的理想线损替换为可插拔估计器，其中 L1（用主线辨识树拟合权重自洽计算二次损耗）是"无真值"的现实版本；
2. **幅值区间输出**：对声明的线损误差带给出 â_t 的包络区间而非点值，进入可信度报告；
3. **敏感性实验矩阵 M7**：系统扫线损偏置 β ∈ [−50%, +50%]，量化误报/漏报/定位退化曲线，回答"线损估错多少以内方法仍可用"。

预算：约 40 次 MILP 拟合，25–35 分钟机时，分 stage 存 JSON 可续跑。代码改动集中在一个新文件 `theft/loss_models.py`（约 120 行）+ `run_theft.py` 增加 3 个 stage，预计净增约 300 行。

---

## 1. 问题陈述

### 1.1 代码现状（已核对）

`theft_wzzt/theft/theft_model.py::data_from_scenario`（L299–334）：

- 幅值固定为 `amplitude = clip(P0_measured − ΣP_terminal − loss_p, 0)`；
- 默认 `loss_p = scenario["loss_p_notheft"]`——由 `theft_simulation.py` L193–199 对**无偷电反事实网络**再跑一次 AC 潮流得到的精确技术线损。这是理想化的外部估计，现场不可得；
- **已预留钩子**：`loss_p_estimate` / `loss_q_estimate` 参数可直接注入任意外部线损序列，无需改动函数本体。

### 1.2 为什么这是最大的现场不确定性

辨识链的其余环节都有误差预算：量测噪声进了 `voltage_scale`/`balance_scale`，拓扑与参数误差进了逐边权界（BOUNDS_FACTOR_IDENTIFIED = (0.5, 2.0)），定位歧义进了区域标签。**只有线损误差目前没有任何量化**——它以系统偏置形式直接进入 â_t，而 â_t 同时是 H1 的解释资源和 H0/H1 比较的分母。线损估错多少会误报（无偷电时 â>0）？估错多少会漏报或错位？目前无答案。

---

## 2. 文献调研结论

1. **能量平衡法是工业标准 NTL 路线，技术线损精确计算是其公认瓶颈。** 系统综述明确记载：负荷潮流法用台区总表与分表之和的差推断 NTL，"the difficult task in these methods is the precise calculation of technical losses in the network"。这直接支持我们把线损误差作为一等公民来量化。[Energies 2020 系统综述, §5.3.17](https://www.mdpi.com/1996-1073/13/18/4727)
2. **用智能电表+配变数据估计技术线损再相减，已有工业先例。** MERL TR2015-005（Lo/Ansari 系）提出恒电阻技术线损模型，用智能电表与配变表数据估计 TL 后从平衡差中扣除以检测偷电——是我们 L1 估计器（用拟合 R/X 自洽算损耗）的直接先例。[MERL TR2015-005](https://www.scribd.com/document/1004734671/TR2015-005)
3. **概率化技术线损建模已有先例**：对总损耗与技术损耗各自建立概率分布再处理，并推导技术线损对负荷变化的灵敏度——与我们"线损误差带 → 幅值区间"的思路同构。[ASTESJ 综述引 [13]](https://www.astesj.com/publications/ASTESJ_040420.pdf)
4. **用电压灵敏度反推实际用电量的 model-less 方法**（WLS 估计电压灵敏度系数 → 估计各用户真实用电 → 与表计量比对）与我们电压通道定位同族，可作为幅值估计精度的对照文献。[Pengwah, Razzaghi & Andrew, IEEE Trans. Power Delivery 2023](https://findanexpert.unimelb.edu.au/scholarlywork/1790565-model-less-non-technical-loss-detection-using-smart-meter-data)
5. **定位启示**：我们的差异点仍在 wzzᵀ 树原子 + 竞争解释（H0/H1 嵌套比较）+ 零分布校准；文献空白不在"检测"而在"**线损不确定下的可信度量化**"——这正是 M7 要填的。

---

## 3. 误差传播分析

记真值：总表 `m_t = a_t + ℓ_t(a) + ε_t`（a=偷电，ℓ=实际技术线损，ε=总表噪声），则

```
â_t = clip(m_t − Σp_t − l̂_t, 0) = clip(a_t + δℓ_t + ε_t, 0),   δℓ_t = ℓ_t(a) − l̂_t
```

（Σp_t 自身的噪声归入 ε。）三条影响路径：

1. **平衡通道**：H1 的平衡预测是 `â_t·Σs_th`，δℓ 的系统分量直接抬高/压低 â——**β<0（低估线损）在零场景产生虚假正幅值，是误报主通道**；
2. **电压通道**：â 作为固定输入乘 v=r·b 进入电压降方程，δℓ 迫使 R/X 在权界内吸收（吸收率待实证）或把选择推向错误位置；
3. **截断偏置**：clip(·,0) 使 E[â] > a（零场景下 â = max(δℓ+ε, 0) 恒非负），零分布校准已吸收 L0 下的这部分，但 δℓ 系统偏置会**平移整个零分布**——因此 M7 必须同时报告"固定阈值"（现场：阈值是历史校准的）与"重校准阈值"（oracle）两种误报率。

## 4. 线损估计器变体（`theft/loss_models.py`，新建）

统一接口：`estimate_loss(scenario, weights=None, tree=None, **kw) -> (loss_p, loss_q)`，输出与 `scenario["P_terminal"]` 同索引的 Series。

| 变体 | 定义 | 角色 |
|---|---|---|
| **L0** 理想 | `loss_p_notheft`（现状） | 上界基线 |
| **L1** 自洽二次损耗 | 用辨识树拟合权重 w（=2r_pu 约定）与量测负荷：边流 `F = p@z.T`（z=辨识树关联矩阵），`ℓ̂_t = Σ_e (w_r,e/2)·(F_p²+F_q²)/V̄_t²`，V̄ 取 `root_voltage` 量测；Q 损耗同理用 w_x | **现实版本主候选** |
| **L2** 系统偏置 | `l̂ = (1+β)·loss_p_notheft`，β ∈ {−0.5, −0.25, 0, +0.25, +0.5} | 敏感性扫参（核心曲线） |
| **L3** 随机无偏 | `l̂ = loss_p_notheft·(1+η_t)`，η~N(0, σ=0.25)，种子固定 | 区分系统偏置 vs 随机误差 |
| **L4** 恒定比例 | `l̂ = c·Σp_terminal`，c 取 L0 下均值比 | 工程常用近似对照 |

L1 说明：标准 DistFlow/LinDistFlow 二次损耗近似（Baran–Wu 形式），边流只用量测分表（无偷电假设，与"外部估计不知偷电"的现实一致）；不做外层迭代（见决策点 D2）。L1 与 L0 的差就是"主线参数辨识误差"对线损的影响，可分离评估。

## 5. 幅值区间输出

对声明的线损相对误差带 β ∈ [β_lo, β_hi]，逐时刻取包络：

```
â_lo,t = clip(m_t − Σp_t − (1+β_hi)·l̂_t, 0)   # 线损高估 → 幅值下界
â_hi,t = clip(m_t − Σp_t − (1+β_lo)·l̂_t, 0)   # 线损低估 → 幅值上界
```

报告字段（进 credibility/summary JSON）：`amplitude_envelope = [â_lo, â_hat, â_hi]`、`amplitude_rel_halfwidth`。另报**吸收率**：同一 â 扰动下，`H0_loss(fixed_rx) − H0_loss(refit_rx)` 占扰动前 H0_loss 的比例，量化 R/X 重估吞掉了几成线损误差（fixed_rx 走现有 `fixed_rx` 参数，零新增建模）。

## 6. 实验矩阵 M7（identified 模式）

| stage | 内容 | 拟合次数（估） |
|---|---|---|
| `m7_l1` | L1 vs L0 × {null(rep 10–14), exp2, exp3}：幅值相对误差 \|â−a\|/a、gain、区域覆盖 | 3 实验 × 2 变体 ≈ 6 主拟合 + 5 null |
| `m7_bias` | L2 扫 β × {null×5 rep, exp2, exp3}：固定阈值误报率 + 每 β 重校准阈值 + 告警率/覆盖/gain 余量曲线 | 5 β × (5 null + 2 exp) = 35 |
| `m7_report` | 汇总 JSON + 包络字段并入总 summary | — |

判定指标：① 零场景固定阈值误报率 vs β（关键鲁棒性曲线）；② 有偷电场景告警保持率与区域覆盖 vs β；③ 幅值相对误差 vs β；④ 吸收率。预算合计约 40 次拟合 × 30–50 s ≈ **25–35 min**，沿用现有分 stage JSON 续跑模式。L3/L4 各 3 次拟合作为补充点（+5 min）。

## 7. 代码改动点

1. **新建 `theft_wzzt/theft/loss_models.py`**（约 120 行）：L0–L4 估计器 + `amplitude_envelope()`；纯函数，不触碰 MILP。
2. `run_theft.py`：`run_detection` 加 `loss_model`/`loss_bias` 参数（在 `_window` 前注入 `data_from_scenario(scenario, loss_p_estimate=…)`）；新增 stage `m7_l1`/`m7_bias`/`m7_report`；结果文件名带 `m7_` 前缀，不覆盖现有 M5/M6 输出。
3. `credibility.py`：`compare_h0_h1` 不变；报告装配处加包络与吸收率字段。
4. `tests/` 新增 3–4 个单测：L1 在清洁数据上接近 L0（相对误差中位数 < 容差）、L2 β=0 退化为 L0、包络单调性（β_lo≤β_hi ⇒ â_lo≤â_hi）、L4 符号非负。

**不改**：MILP 本体、零分布校准链（identified q95≈5.5 继续作为固定阈值基准）、现有 12 个单测、任何 M5/M6 结果文件。

## 8. 决策点（请拍板）

- **D1 误差带默认 ±25% 还是 ±50%？** 建议：扫参网格覆盖 ±50%，报告默认 ±25% 带。
- **D2 L1 是否做外层迭代**（估出偷电→修正边流→重估线损）？建议：**不做**。迭代破坏"MILP 全局最优 + 固定输入"的干净叙述，且单次前向已能分离参数误差影响。
- **D3 幅值区间用包络法还是蒙特卡洛？** 建议：包络法（确定性、可复现、零额外拟合）；蒙特卡洛只在未来需要分布形态时再加。
- **D4 M7 是否纳入 exp5（不可定位歧义）/exp6（阻抗漂移）？** 建议：首批只做 null+exp2+exp3；exp6 与线损误差的交互（两种模型失配叠加）留作后续扩展。

## 9. 风险与备注

- L1 用辨识树权重算损耗，权重约定为 2·r_pu（`nominal_edge_weights` 注释已确认），实现时 r_pu = w/2，需单测锁定该约定防回归。
- exp6 的阻抗漂移会使 L1（拟合自未漂移网络）产生真实模型失配——这本身是卖点而非 bug，但首批不混入口径。
- HiGHS 在部分 null 拟合刷 stderr 警告为已知现象，结果经三重直接校验仍有效。
- 零场景 β<0 理论上必抬高 â；若实测误报率在 β=−0.25 已不可接受，结论应写为"方法要求线损估计相对误差 < X%"，这正是 M7 要给出的数字。
