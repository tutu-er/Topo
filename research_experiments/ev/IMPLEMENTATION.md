# 当前实现与可复用部分

状态：2026-10-19 封存。下表中的路径均相对于封存的 [mobile_ev_mle](../../archive/ev_methods_20261019/payload/research_experiments/mobile_ev_mle/)；依赖的旧 AC 组件一并保存于相邻的 `mobile_ev_probing/`。

## 实际流程

已到达 P/Q/V 与根总表 → WZZT/RNJ 或指定候选库 → 每候选历史 AC 拟合 → 预测未来普通负荷与同接口源 → EV 时段/地点/连续功率规划 → 真实网络完整 AC 含噪执行 → FIX 或 JNT 竞争候选拟合 → 保存点判别、校准保留集合、参数误差和费用。

`full_pipeline/` 已把实际候选生成、6/8末端整树评价和 EV 接起来。此前多数三/四末端批次使用按构造含真候选的有限库。`case33_candidate_ranking/` 则研究完整33母线的已知线路参数候选评分，未执行新的 EV 规划。它们的结论分别对应各自流程。

## 统计模型有两种实现口径

| 实现 | 估计变量、计量与输出 | 当前边界 |
|---|---|---|
| `likelihood.py::fit_candidate` | 固定候选下拟合跨时共享逐边 R/X、逐时真实 P/Q、同一物理接口的源；有独立校准读数时可纳入固定表计偏差。输出 `FitResult`：取得损失、参数、状态、停止信息 | 完整 Gaussian 量测模型的局部非线性拟合；偏差与源的可辨识性依赖校准/总表/遥测条件 |
| `ev_fixed_train_fit.py::fit_fixed_inputs`，`ev_systematic_refit.py`，`full_pipeline/` | P/Q 校准后直接作为 AC 输入，拟合电压与根总表输出。规划每候选固定一组训练 R/X；FIX 只调整新增源，JNT 用历史和新增数据共同拟合共享 R/X 与逐时源 | 这些批次使用条件输出损失，输入 P/Q 误差未继续传播；根电压、固定偏差按批次规定固定 |
| `ev_bayes_response_ac.py`，`ac_history_refresh/` | 从历史构造线路、P/Q、源的局部后验/预测矩，输出结构权重、预测均值和协方差 | 概率与积分精度需要独立检查；已有结果暴露响应区间欠覆盖和过度确定 |

输出损失统一可写成加权残差平方和，但不同文件使用 SSE 或半 SSE（乘 1/2）。跨批次比较须按原报告口径读取，不能直接合并损失数值。局部拟合值与求解器 success 一起保留；全局最大似然、全部拓扑排除和现场可靠性尚无完整证据。

## EV 规划目标与实现

| 方法 | 实现入口 | 优化什么 / 返回什么 |
|---|---|---|
| P-SQ | `g_precision/study.py::precision_task`，既有平方 MILP 模块 | 精度加权功率平方；返回合法 EV 计划、原目标可行值、分段代理界与费用。是当前强对照 |
| G-SQ / G-AQ | `g_precision/coefficients.py`、`g_precision/study.py`，`full_pipeline/solver.py` | G-SQ 加候选响应导数差，G-AQ 再加背景差与线性项；输出每候选对系数、计划及界 |
| 标量 L2 距离 | `ev_scalar_distance_milp.py`、`ev_fixed_train_design.py` | 冻结仿射响应下保留源 profile 的 L2 距离；弦 MILP 给上近似，实际可行计划/源子问题另记界 |
| AC 梯度 | `ev_fixed_gradient.py`、`ev_gradient_flow_design.py` | 完整 AC 的源 profile 距离及包络梯度；MILP 提案再用完整 AC 接受 |
| JP-GRAD / v4 | `block_joint_margin/`、`ev_mobile_v4_study.py` | 历史与假想未来共享线路竞争重拟合，计算优势/价格，再用完整 AC 分数接受并继承父计划 |
| B-RISK | `ev_bayes_small_milp.py`、`ev_bayes_response_milp.py` | Gaussian 预测风险或 Bhattacharyya 重叠代理；输出计划、预测风险、积分/操作失败与回退。完整协方差梯度尚未作为主 MILP 梯度接入 |
| 更早方法 | `adaptive_*`、`continuous_eig_*`、`ev_profile_*`、`ev_cross_*`、`ev_compact_*` 等 | 有限/连续先验积分、双线性代理、滚动与支持集的历史试验；按各冻结协议解释，文件级入口见目录表 |

以 G-AQ 为例，冻结候选 AC 响应后，其目标为

\[
\max_{u\in\mathcal U}\min_{a<b}\left[D^0_{ab}+\sum_{t,j}(b_{ab,tj}u_{tj}+w_{ab,tj}u_{tj}^2)\right],\quad
w_{ab,tj}=\tfrac12 g_{ab,tj}^{\mathsf T}\Sigma_t^{-1}g_{ab,tj}.
\]

`u` 是 kW 净充电功率，负值放电；`g` 是两候选的 AC 输出导数差，`Σ` 是声明的计量协方差。平方展开说明该分数为何含一次和二次项；用它选动作的前提是响应与预测已冻结。分段弦与求解器界只覆盖对应冻结代理，最终辨识仍由执行后的竞争拟合评价。

## 运行域、物理与评价

`ev_mobile_service_domain.py` 构建一个移动电池的时间展开线性运行域：接入/出行弧、每期最多一个接入点、充放电互斥、效率、SOC、驾驶耗能、接线时间、人员窗口、吞吐及终端电量任务。`ev_mobile_v4_domain.py` 构造固定纯充、固定双向、可移纯充、可移双向四个嵌套域，并保存合法父计划。允许移动包含原地不动。

负荷生成器与评价器保管真树、真参数和未来噪声；规划器只读取历史、公开设备域及因果预测。末端总 P/Q 已含 EV，AC 评价只计入一次。FIX/JNT 是最终推断口径，与规划方法名称分开。后验权重、代理距离、bootstrap 频率、局部损失差各按原定义保存。

## 可复用的数值工具

`ac_block_fit/` 保存批量逐时源、共享线路/局部源导数块、QR 消源、有界 Gauss–Newton、尺度对照、端点修整及 FP64 GPU 试验。TRF 保持旧批次默认；QR 的多候选验收尚未全过，恢复测试曾有一次未解释失败。物理 R/X 坐标加线性 X/R 锥约束的后续 GN 构造只完成推导。

`case33_quality_fixed_rx/` 与 `case33_candidate_ranking/` 保存已知逐边 R/X 下的无损 LD、含祖先损耗传播的 DF-I² 与完整 AC/EIV 评分，适合作近似误差和排序诊断。未知 R/X、多时段共享拟合及新 EV 方法的收益需重新验证。

封存后复用优先取 AC 前向/导数、运行域检查、配对评价、失败记录与证据恢复工具。新方法的目标、拟合器接入和统计校准须另行定义；旧报告中的建议不视为当前执行任务。
