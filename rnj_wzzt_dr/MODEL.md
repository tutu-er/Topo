# 从当前辨识器到 DR：一页接口

## 历史冻结

`fit_observed(training, validation, method='111')` 直接调用本目录原辨识器。训练输入决定候选和参数，验证观测只作最终选择。`freeze_history(result, validation)` 检查保存的 R/X、参数、原选择和观测指纹，保留整个 `frozen_candidates`，包括同拓扑不同参数点。活动拓扑按响应权重超过原 `zero_tol` 的内部支持定义，零辅助支持沿用原展示接口的口径。

第 j 个候选的预测是 `R_j P + X_j Q`；矩阵描述平方电压响应，并非原始线路阻抗。每时槽先拼 P、再拼 Q，节点顺序固定。记 `H_j=[R_j,X_j]`，以下平方范数覆盖新日所有时槽与节点。历史只包含原选择／验证集，训练集用于拟合及历史功率预测，不重复计入选择分数。

W 每步只求一个新支持并联合调权，整体是贪心路径。T/E 的整数证书覆盖各自声明的局部或有限候选域；原生 gap 与完整拓扑的全域最优、唯一性分开解释。

## 最终评分决定门槛

目标为 t、竞争者为 j，历史验证 SSE 为 `S_t^h,S_j^h`，历史加新日共有 N 个残差。共同尺度 `K=1/M_ref`，原无量纲并列容差 `epsilon_scaled=1e-12`，故物理 MSE 容差 `epsilon=M_ref×1e-12`。

\[
 M_j=\frac{S_j^h+\|Y-H_jP\|^2}{N},\qquad
 \delta_j^{phys}=N\epsilon-S_j^h+S_t^h.
\]

`M_j` 表示完整历史验证＋新日逐标量等权的 MSE。目标严格击败竞争者的充分条件是新日 SSE 差

\[
 L_j=\|Y-H_jP\|^2-\|Y-H_tP\|^2>\delta_j^{phys}.
\]

该式由 `M_j−M_t>epsilon` 直接移项，用于把最终选择规则转成 DR 门槛。`history.required_gap` 复用原 `selection_mse.required_new_sse_gap`。同拓扑不同参数胜出仍算该拓扑成功；固定目标点击败所有不同拓扑点给出充分条件。相同响应的并列按原支持数／序号处理，包括目标并非历史 winner 的情况。

## 条件误差及指数界

规划以目标候选正确为条件：`P=p_hat+B u+xi_P`，`Y=H_t P+e`。P 的各通道标准差为历史预测绝对峰值的 5%；同节点 P/Q 共享因子、节点间独立、时间 AR(.7)。e 独立于功率误差，其节点标准差取目标历史残差 RMS 和 `2sqrt(2)×voltage_sigma` 的较大者。`prepare_dr` 显式接收预测，误差参数写入报价收据；不会读取真实未来或网络线路。

令 s 为剩余电压误差的最大标准差，`z=(H_t−H_j)P/s`，`E=diag(Var(e)/s²)`。一个公共 s 仅换单位，保留未加权物理 MSE。固定 `lambda=1/4`、`A=diag(lambda(1−2lambda E_ii))`，C 为 z 的协方差，d(u) 为其均值。原两次高斯积分给出

\[
 G=I+2A^{1/2}CA^{1/2},\quad
 T_j(u)=d(u)^\top A^{1/2}G^{-1}A^{1/2}d(u)+\tfrac12\log\det G,
\]
\[
 \Pr_t[L_j/s^2\le\delta_j^{phys}/s^2]
 \le\min\{1,\exp(\lambda\delta_j^{phys}/s^2-T_j(u))\}.
\]

T 表示固定误差法则下的指数分离强度；积分精确，概率上界来自 Markov 不等式。`zero_screen` 复用原谱实现；`pair_gate` 复用原稠密积分，明确映射 `current=competitor, reference=target, ref=target`。两种实现的小维度交叉核验覆盖两个目标方向。

## 报价及实际执行

单对优化为 `min C(u)`，保留原设备域并加 `T_j(u)>=lambda delta_j+log 10`。原小时请求、投影、能量／截止、效率及费用图均保留；含分段整数变量时如实记为带整数的非凸 QCQP。单对入口保存可行上界、求解器下界、时间和状态；未证最小的报价仍是可行报价。

整库零动作使用各失败上界的 union bound，无需竞争事件独立。原设备零动作交付可行、费用 0，且费用域非负时，该点通过便得到最小费用 0。没有设备合同的零动作核算只报告概率充分界，不报告费用最优。

多竞争者非零优化仍待接通。预锁 `alpha_j>0, sum alpha_j<=.1` 后分别加 `T_j(u)>=lambda delta_j+log(1/alpha_j)` 可保持二次约束；本包能构造这些单对门槛，尚无联合设备图求解入口。直接把指数和放进优化一般会改变问题类别。

冻结报价后，原 `devices.draw/project/audit` 与 `compat/ac_bridge.py` 用于随机响应、设备截断和原 AC 观测。实际结果再用 `FrozenHistory.select(new)` 评分；评价中参数保持冻结。固定协方差尚未包含全部乘性响应／相关偏差／饱和；90% 条件界与实际 AC 成功率分开记录。
