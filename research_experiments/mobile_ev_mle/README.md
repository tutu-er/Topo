# 当前方向：AC 最大似然估计与 EV 模型判别设计

此目录目前只提供交接入口，**尚未实现或验证新的 MLE / EV 算法**。
新窗口可直接使用 [NEXT_PROMPT.txt](NEXT_PROMPT.txt)。

## 目标

被动 WZZT/RNJ 给出可修正的候选拓扑。固定各候选，用原始 P/Q/V 计量和 AC 方程联合估计阻抗及必要的未知状态，比较最大对数似然。沿 Dette 的思路选择 EV 地点、时段和功率曲线：即使竞争拓扑重新拟合，它仍难以同时解释历史与新增响应。

历史与未来共享线路参数和固定仪表偏差；共同源可逐时变化，但通过同一个物理注入点作用。EV 已包含在末端总 P/Q 中，不重复计入。逐时 15 分钟 AC 状态是否能代表平均计量，需要作为假设说明。

不以固定误差盒、外 LP 的正下界或所有方向最小奇异值替代这一目标。硬件、接入与电网运行约束仍然需要保留。

## 最少需要阅读的代码

1. `rnj_wzzt_core/rnj_wzzt/estimation/ac_measurement_fit.py::fit_ac_pqv`：已有固定拓扑 AC 加权 P/Q/V 联合拟合，线路参数跨时段共享。当前源位置和幅值固定，优化是局部的；还不是本任务完整统计方案。
2. `research_experiments/mobile_ev_probing/passive_bridge_v9.py`：可修正的被动候选生成。
3. `research_experiments/mobile_ev_probing/ac_general_v9.py`：径向树和 AC 潮流，可复用物理计算；不要直接沿用外 LP 判定。
4. `research_experiments/mobile_ev_probing/integrated_study_v9.py`：同时间、同电量对照的组织示例；旧选址是代理评分。

生产主线是 `rnj_wzzt_core`。新实验使用本目录，按需复用旧模块，避免为重命名复制整套算法。工作区已有 `pyproject.toml` 修改不属于本次清理，不应顺手提交。

## 理论起点

Dette–Titoff (2009), *Optimal discrimination designs*，正文第 2 节及第 4.2 节：
https://arxiv.org/pdf/0908.1912

原文提供模型间充分重拟合后的 T/KL 判别设计基础，不直接给出本项目 AC、逐时源项及自适应选址的统计保证。

采用明确的计量概率模型后，令 `ell_h,H(beta)` 为固定共享参数、充分拟合逐时状态后的历史对数似然。可从以下迁移开始，先验证再扩展：

```text
R_h(beta) = 2 * [max_beta ell_h,H(beta) - ell_h,H(beta)]
T_rh(a) = min_{beta, future states} [R_h(beta) + future weighted AC response difference]
```

历史项衡量竞争结构改变参数的代价。两候选具有相同已知高斯协方差时，未来加权平方差等于两倍 KL 距离。若协方差不同或依赖参数，须使用完整似然/KL 表达式。设计时可计入历史已有似然差；共享参数必须在历史与未来共同重拟合。以上是参考场景下的设计准则，不是有限样本拒绝概率。

## 先完成什么

先用小型局部拓扑，检查阻抗估计、共同源、AC 非线性和优化稳定性；通过后才扩大网络。比较同长度无 EV 观测、同电量恒充、固定/随机选址与设计选址。跨种子评估误拒绝、排错能力、候选覆盖与运行时间，并保留弱支路和选址失败。

不预先承诺 Gaussian 误差、已知 X/R、固定源 Q/P 或现场置信保证。先说明哪些来自已有数据、哪些是算例假设。似然比不是拓扑概率，参数 bootstrap 也不会自动成为严格有限样本保证。

旧结果位置见 [旧实验索引](../mobile_ev_probing/README.md)。当前报告 `docs/mobile_ev_probing_report_v2.tex` 与 PDF 保留为旧路线记录；新的模型经验证后在原文中明确修订，避免无区分地继续堆叠版本。
