# Pengwah18 无源：短边分辨率、路径过细化与固定支持重拟合

实验标识：20261018；报告核查更新：2026-10-19。对象为 `pengwah18__no_source__RNJ+WZZT` 的一个已结束算例：三个 reference 背景场景各96点，共288点全部拟合，无验证集。原始算法结果保持保存；本文另列运行结束后的 RX75 收缩与重拟合结果。

**已核实的结果是：原始路径先恢复全部7个真实非平凡簇，随后为降低训练MAE增加8个短伪簇；按同输入NJ矩阵校准的既有RX75尺度收缩短边，再固定支持重拟合，恢复F1=1，R/X误差也下降。** 本次实测范围为无源单组后处理。在线截断策略及有源优势由各自实验单独验证。

## 1. 原始路径为何从 F1=1 变为 0.6364

原始模型使用18个singleton叶原子，加上筛选得到的 `[306,307,308]` 作为初始支持。原始RNJ完整树已有7个正确非平凡簇，但当前hybrid入口只传入严格筛选的末端块。WZZT在此初始家族上逐步增加支持，并按无验证协议选择最后一个已接受路径点。

下表根据保存的 `model.json.path` 逐点事后评价。`path index` 从0开始；`iteration` 为前向循环计数，零权重剪枝会使它与实际原子数不同。family和active两种F1在本路径各点相同。

| path index | iteration | 实际原子数 | 训练MAE | family/active F1 |
|---:|---:|---:|---:|---:|
| 0 | 19 | 19 | 0.00190771417539 | 0.2500 |
| 1 | 20 | 20 | 0.000984721908619 | 0.2222 |
| 2 | 21 | 21 | 0.000822202077072 | 0.4000 |
| 3 | 22 | 22 | 0.000755899167028 | 0.3636 |
| 4 | 23 | 23 | 0.000692919205566 | 0.5000 |
| 5 | 24 | 23 | 0.000624192822217 | 0.6667 |
| 6 | 25 | 24 | 0.000516969860013 | 0.7692 |
| 7 | 26 | 24 | 0.000481375060457 | 0.9231 |
| **8** | **27** | **25** | **0.000453670910221** | **1.0000** |
| 9 | 28 | 26 | 0.000453280286075 | 0.9333 |
| 10 | 29 | 27 | 0.000453051034114 | 0.8750 |
| 11 | 30 | 28 | 0.000452888798177 | 0.8235 |
| 12 | 31 | 29 | 0.000452742149816 | 0.7778 |
| 13 | 32 | 30 | 0.000452656063861 | 0.7368 |
| 14 | 33 | 31 | 0.000452633665193 | 0.7000 |
| 15 | 34 | 32 | 0.000452624592765 | 0.6667 |
| **16，原始选中点** | **35** | **33** | **0.000452620318920** | **0.6364** |

首次达到F1=1后，训练MAE继续下降约 `1.05059e-6`，相对降幅约0.232%；最终保留15个非平凡簇，其中7个正确、8个多余，precision为7/15、recall为1。原始停止原因为 `max_atoms`。

原实现的改善阈值为 `1e-10 + 1e-8*abs(current_loss)`。末次改善仍有 `4.27385e-9`，超过约 `1.04526e-10` 的数值阈值，因而继续接受符合该实现。这个阈值控制优化上的微小改善；结构复杂度和测量分辨率需要独立规则。

证据：[原始模型及全部路径](/D:/0-github_workspace/Topo/outputs/source_ac288_benchmark_20261018/jobs/pengwah18__no_source__RNJ+WZZT/model.json)、[原始结果](/D:/0-github_workspace/Topo/outputs/source_ac288_benchmark_20261018/jobs/pengwah18__no_source__RNJ+WZZT/result.json)、[已保存路径汇总](/D:/0-github_workspace/Topo/outputs/source_ac288_benchmark_20261018/reports_checkpoint_9/accepted_paths.csv)。路径F1只用于本文诊断，未用于选中重拟合支持。

## 2. 求解时限与系数上界的证据

该原始job的19次保存求解均为 `status=0` 且 `optimality_certified=true`。最长单次耗时913.69秒，低于6000秒；原始计算总时2055.45秒，低于30000秒。原结果没有超时或未认证求解，末段过细化发生在已认证的改善扩展中。

上界的历史需要分开说明：

- 保存的R、X系数上界均为2，`bound_expansions=0`。
- 初始固定支持拟合 `path[0]` 的 `max(X weight)=2.0`，确实命中X上界；`max(R weight)=1.08622`。
- 当前core先初始化固定支持模型；边界检查放在后续“扩展已认证、存在候选且改善超过阈值”的分支中。初始触界本身不会触发扩界。扩界条件为候选系数达到上界的0.995倍；允许最多4次、每次乘2，并在扩界后重启路径。
- 从首次F1=1的 `path[8]` 到最后 `path[16]`，所有R权重最大值不超过0.093921，X权重最大值不超过0.117565，均远离上界2；最终最大R/X权重为0.092981/0.110616。

这些记录排除了末段伪短边来自当时的上界卡住或求解超时。早期初始化触界可能影响前向路径的历史选择，其影响范围需要另行对照；`bound_expansions=0`本身不足以证明整条路径从未触界。

代码：[冻结core初始化与扩界流程](/D:/0-github_workspace/Topo/outputs/source_ac288_runtime_20261018/rnj_wzzt/estimation/laminar_l1_milp.py:1629)、[改善后边界检查](/D:/0-github_workspace/Topo/outputs/source_ac288_runtime_20261018/rnj_wzzt/estimation/laminar_l1_milp.py:1700)。逐次求解记录保存在原始模型的 `solver_events`，原始预算保存在原始结果及模型的 `budget` 中。

## 3. 固定观测尺度，8条短伪边与7条真实边

校准读取同variant NJ在完全相同288行P/Q/Y和源输入上的前置ordered回归矩阵。它与RNJ共用该前置R/X回归；这一步没有使用最终WZZT矩阵或物理真值。

对每个通道的dense矩阵 \(M\in\{R_0,X_0\}\)，先构造两端距离

\[
d^M_{ij}=M_{ii}+M_{jj}-2M_{ij}.
\]

令 \(\mu_R,\mu_X\) 分别为各自全部严格正距离的均值，按既有RX75规则取

\[
a=\frac{0.75}{\mu_R},\qquad b=\frac{0.25}{\mu_X},\qquad
\tau=0.16\,\operatorname{median}\!\left[\operatorname{diag}(aR_0+bX_0)\right].
\]

实现保留既有极小量保护：通道距离均值下限为 `1e-12`，无正距离时回退1；根深中位数下限为 `1e-12`。本算例各量正常，未触发这些保护。

| 校准量 | 数值 |
|---|---:|
| \(\mu_R\) | 0.2459220811640305 |
| \(\mu_X\) | 0.2383088790837734 |
| \(a\) | 3.0497464743710774 |
| \(b\) | 1.0490586878725436 |
| 根深中位数 | 0.6213656285024757 |
| \(\tau\) | **0.09941850056039611** |

原子 \(k\) 的几何长度为 \(w_k=a r_k+b x_k\)。singleton叶总保留；其余原子在 `w + 1e-10 >= tau` 时保留，低于此边界则收缩。全体终端的root-stem原子也采用同一非singleton规则。`1e-10`为固定绝对几何数值容差；RX75长度为归一化几何量。

下表在完成保留/删除决策后，才读取物理真网标记“真/伪”。

| 最终原始支持的终端标签 | 原始RX75长度 | 事后物理评价 | 决策 |
|---|---:|---|---|
| 306,307,308 | 0.374831830 | 真簇 | 保留 |
| 313,314,315,316,317,318 | 0.353220979 | 真簇 | 保留 |
| 315,316,317,318 | 0.207811990 | 真簇 | 保留 |
| 310,311,312 | 0.254682395 | 真簇 | 保留 |
| 301,302,303,304,305 | 0.277426857 | 真簇 | 保留 |
| 309,310,311,312 | 0.310885593 | 真簇 | 保留 |
| 303,304,305 | 0.157565872 | 真簇 | 保留 |
| 303,305 | 0.041302050 | 伪簇 | 收缩 |
| 313,315,316,317,318 | 0.015631309 | 伪簇 | 收缩 |
| 316,318 | 0.055360063 | 伪簇 | 收缩 |
| 306,307,308,313,314,315,316,317,318 | 0.005023959 | 伪簇 | 收缩 |
| 315,317 | 0.025497787 | 伪簇 | 收缩 |
| 302,303,304,305 | 0.004686436 | 伪簇 | 收缩 |
| 310,311 | 0.003082678 | 伪簇 | 收缩 |
| 301,302,303,304,305,309,310,311,312 | 0.001455089 | 伪簇 | 收缩 |

8条伪边长度范围为 `[0.001455089,0.055360063]`；7条真边范围为 `[0.157565872,0.374831830]`。观察校准的阈值0.099418501位于两者之间，所有边都远离数值容差边界。

实现：[RX75校准与边界分类](/D:/0-github_workspace/Topo/research_experiments/rnj/rx75_edge_floor.py:51)。证据：[本次校准配置](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/config.json)、[原始分类及删边列表](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/steps/0000/input_family.json)、[校准NJ矩阵](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/calibration_baseline/model.npz)。

## 4. 实际执行：每次收缩后重新拟合

后处理先按固定尺度删短内部支持，再在冻结runtime中以原288行P/Q/Y和原源时序执行固定支持L1重拟合。R/X上界保持2，单次求解限时6000秒、总预算30000秒。固定支持解允许权重为零，没有额外正floor。若新拟合后又出现低于分辨率的非singleton支持，继续删除并重新拟合，直到稳定；后续每轮至少少一个支持，因而有限终止。

Source非零时，每次固定支持重拟合都会重新联合求解完整未知路径h及全部阻抗系数，随后重算g与预测。删除支持后保留旧h的做法不在本实现中。本次无源数据的h为null，实际固定支持问题为LP；有源h重新求解另由合成rank-2 Source测试覆盖。

| 指标 | 原始最后点 | RX75收缩并固定支持重拟合 |
|---|---:|---:|
| 总原子数 | 33 | 25 |
| 叶原子/非平凡簇数 | 18 / 15 | 18 / 7 |
| family F1 | 0.6363636364 | **1.0000000000** |
| active F1，阈值1e-8 | 0.6363636364 | **1.0000000000** |
| R矩阵相对Frobenius误差 | 5.735591% | **4.248165%** |
| X矩阵相对Frobenius误差 | 18.227510% | **13.650114%** |
| 训练MAE | 0.000452620318920 | 0.000453670910220 |

收缩了8个内部支持，重拟合一次即稳定；重拟合后最小非singleton长度为0.177658115，大于固定阈值。训练MAE回到较粗真实结构的水平，与原始 `path[8]` 的MAE差约 `1.4e-17`。这是拟合误差与结构分辨率的实际取舍。

重拟合证据：Gurobi `OPTIMAL`，报告的primal与dual bound均为 `0.0004536709102204997`；最大COO线性约束违反量 `1.39e-17`，界违反量0。此次为5234变量、10368约束、0整数变量的固定支持LP。归档包含原始primal与约束COO，dual bound来自求解器诊断，未保存对偶乘子。求解器时间7.93秒，worker计算时间8.97秒，父进程wall time15.51秒。求解认证覆盖该固定支持、固定界域的L1问题；原始全路径仍是贪心结构搜索。

证据：[父进程独立物理评分](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/evaluation.json)、[步骤及耗时](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/steps.json)、[求解诊断](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/steps/0000/solver_events.json)、[COO与primal](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/steps/0000/solver/000001.npz)、[预测及残差](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/predictions.npz)。

原始job与校准NJ的模型、结果、数组和已复制输入文件的SHA均未改变，见 [input_integrity.json](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/input_integrity.json)。[独立只读审计](/D:/0-github_workspace/Topo/tmp/source_ac288_rx75_refit_no_source_independent_audit_20261018.json)为193项通过，覆盖1个job、1次固定支持LP。原始0.6364成绩继续保留在原目录，新成绩使用独立的 `no_source_hybrid` 后处理目录。

## 5. 有源NJ/RNJ：时序已知、位置未知的FWL处理

将三个场景依次堆叠，记 \(P,Q,Y\in\mathbb R^{288\times n}\)，其中Y为同时观测的根电压平方减终端电压平方；已知源时序为 \(U=[u_P,u_Q]\in\mathbb R^{288\times2}\)。令 \(D=[P,Q]\)，\(B=[R^T;X^T]\)，则适配后的线性模型为

\[
Y=DB+UG+E,\qquad G\in\mathbb R^{2\times n}.
\]

G表示源P/Q对各终端的自由响应。源接入母线和真实根路径均未提供给NJ/RNJ。代码通过SVD识别U的列空间，令 \(M_U=I-UU^+\)，执行

\[
\widehat B=\arg\min_{B\in\mathcal C}
\|M_UY-M_UDB\|_F^2,\qquad
\widehat G=U^+(Y-D\widehat B).
\]

\(\mathcal C\) 为原有ordered R/X回归约束。上述消去在G自由、无截距且无额外G惩罚时成立：对任意固定B，最小二乘消去G恰好留下正交投影残差，因此保留B上的约束不会破坏等价性。

- rank(U)=0时回到原无源回归，G返回零；rank=1时只识别源P/Q的组合响应，保存的G为最小范数解；rank=2时两个源响应可分开表达，其统计可识别性还取决于联合设计。
- NJ/RNJ主R/X误差取同一个前置ordered回归矩阵；两者随后使用不同的树推断，所以R/X误差相同而F1可以不同。
- WZZT使用同一已知U，但其响应受共同物理路径约束。对原子指示向量 \(z_k\) 和未知路径位 \(h_k\)，\(R=\sum_k r_kz_kz_k^T\)、\(g_R=\sum_k r_kh_kz_k\)，X/gX同理；h满足祖先完整、互斥分支约束并与R/X共同拟合。该L1结构模型与FWL的自由G可行域不同。

代码：[源SVD与同步投影](/D:/0-github_workspace/Topo/research_experiments/rnj/source_ac96_trees.py:68)、[ordered回归及G回算](/D:/0-github_workspace/Topo/research_experiments/rnj/source_ac96_trees.py:124)、[Source固定支持联合求解](/D:/0-github_workspace/Topo/outputs/source_ac288_runtime_20261018/rnj_wzzt/estimation/laminar_l1_milp.py:1258)。源秩、残差和求解认证分别描述设计、拟合和优化证据，拓扑正确性仍按独立物理真值评价。

## 6. 严格筛选初值与完整RNJ树初值

本次原始hybrid采用严格筛选：完整直接兄弟终端组、组大小至少3、100次block-length=4 bootstrap、peer频率至少0.75、内部子簇频率至多0.1、全样本细容差无内部子簇，最多取两个互斥组。只有选中组同时进入 `initial_supports` 和不可拆分的 `terminal_groups`。

实际选中 `[306,307,308]`，peer频率1.00、内部子簇频率0.09，对应零基位置 `[5,6,7]`。其他终端为singleton，分区从18个叶缩为16组。label→position→initial支持→partition的传递与保存记录一致。严格筛选保守地只提供这一个非平凡初始支持，完整RNJ树中其余6个正确簇留给后续搜索。

新比较driver另设 `RNJ+WZZT_tree_init` 初始化消融：从同一次筛选调用的 `full_clades` 取得全部RNJ非平凡簇，作为普通初始支持；严格选中组仍单独作为atomic groups。普通初始支持的系数可以变为零，但原有支持还参与层状相容性约束；它们不会自动取得末端组的“不可拆分”语义，也不等同于允许任意替换错误初始拓扑的软warm start。若普通支持与严格组的并集不层状，driver保留明确失败结果，不按真值修补。

这项初始化消融的定义独立于当前无源后处理成绩。实现见 [当前严格筛选记录](/D:/0-github_workspace/Topo/outputs/source_ac288_benchmark_20261018/jobs/pengwah18__no_source__RNJ+WZZT/rnj_screen.json)、[原hybrid初值传入](/D:/0-github_workspace/Topo/research_experiments/rnj/run_source_ac288.py:262)、[新driver完整树普通初值](/D:/0-github_workspace/Topo/research_experiments/rnj/run_rx75_source_comparison.py:275)。

## 7. 执行与验证记录

本次后处理命令如下。原输出已存在；重跑时须为 `--output-dir` 选择新的空目录。

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -B research_experiments/rnj/refit_rx75_resolution.py `
  --job-dir 'outputs/source_ac288_benchmark_20261018/jobs/pengwah18__no_source__RNJ+WZZT' `
  --baseline-dir outputs/source_ac288_benchmark_20261018/jobs/pengwah18__no_source__NJ `
  --output-dir outputs/source_ac288_rx75_refit_20261018/no_source_hybrid `
  --network-dir outputs/source_ac288_20261018/pengwah18/network
```

`--network-dir`只供worker结束后的父进程物理评分，worker payload没有真网或源位置。脚本拒绝运行中job、不同输入的校准baseline以及覆盖已有目录。

```powershell
D:/apps/miniconda3/envs/Topo/python.exe -B -m pytest tests/test_refit_rx75_resolution.py -q `
  --basetemp D:/0-github_workspace/Topo/tmp/pytest_refit_rx75_budgetfix_20261018
```

最新受影响测试为 **6 passed**，包括阈值等号与绝对容差、root-stem收缩和singleton保留、288行Source对齐、运行中/覆盖/异输入拒绝、无源与非零Source两轮真实固定支持求解。新增预算回归覆盖“上次求解已认证、下一次求解开始前预算耗尽”：此时保留已认证checkpoint并标记incomplete。该修复面向未来中断状态。上述193项审计的producer源码哈希核对发生在补丁之前；已完成结果继续保留当次producer SHA及原文件，并未将其改写成修补后脚本的运行记录。补丁前的5个producer文件已按原记录SHA精确归档，见[archive_mapping.json](/D:/0-github_workspace/Topo/outputs/source_ac288_rx75_refit_20261018/no_source_hybrid/archive_mapping.json)；旧refit脚本的SHA为`4e65eb5601276703ac60a5fceb39f999a1e3c2a7b22c8c6a598ea1463b725c78`。追加归档前的34个既有结果文件逐项保持不变。

实现：[refit_rx75_resolution.py](/D:/0-github_workspace/Topo/research_experiments/rnj/refit_rx75_resolution.py)、[测试](/D:/0-github_workspace/Topo/tests/test_refit_rx75_resolution.py)。本报告的经验结论限于该无源算例：现有观测尺度能区分其短伪边，收缩后重新拟合有效。在线RX75约束与有源比较须按其各自冻结协议报告完整结果。
