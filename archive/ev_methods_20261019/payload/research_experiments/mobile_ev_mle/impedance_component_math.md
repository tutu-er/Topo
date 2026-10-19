# 按线路成分检查阻抗误差

本模块是评价器后处理。它只读取已经完整保存的实验批次，不调用拟合器、选择器、仿真器或阈值校准。输入 truth 必须带原有 evaluation_only=True 标记；输出逐边及逐历史记录均标注 truth_evaluator_only=True。真参数不会反馈给冻结策略。

## 1. 支撑映射与分类

用终端后代集合 \(s_e\) 标记每条边，而不假定拟合数组位置对应真值数组位置。拟合的 supports、r_pu、x_pu 是对齐向量，以支持集为键建立映射。若真值提供完整 edge_supports，按其显式顺序映射；当前 simulation.truth_metadata 只输出非平凡 clade，其权重保持 RadialCandidate.from_supports 的规范顺序。该规范先补全 full-set 根 stem 和全部单端叶边，再按 \((-|s|,s)\) 排序；模块明确保存这个输入契约。

拟合与真值完整支持集必须相同，否则标记导出无效，不把错误候选的阻抗冒充真拓扑参数估计。重复支持、向量长度不匹配和非有限权重均保留缺失原因。

分类规则为：

- full-set 支持：root_stem；
- 单终端支持：terminal_leaf；
- 非 full-set 的两个终端支持：cherry_internal；
- 其他内部支持：other_internal，预留给后续网络。

star 即使生成器 case 写 weak，也没有弱内部边，始终归入 star 层。不存在的边类记 not_applicable，不能输出伪造的零误差。

## 2. 每条边与每个独立历史

独立历史编号为 \(i\)，方法为 \(m\)，滚动轮次为 \(t\)。匹配真拓扑的一条边参数为 \(r_e,x_e\)，保存的估计为 \(\hat r_{imte},\hat x_{imte}\)。输出带符号误差及绝对误差：

\[
\delta^R_{imte}=\hat r_{imte}-r_e,\qquad
\delta^X_{imte}=\hat x_{imte}-x_e,\qquad
a^R_{imte}=|\delta^R_{imte}|,\quad a^X_{imte}=|\delta^X_{imte}|.
\]

真参数非零时，相对误差为

\[
\eta^R_{imte}=\delta^R_{imte}/r_e,\qquad
\eta^X_{imte}=\delta^X_{imte}/x_e.
\]

若真参数为零，相对误差未定义，报告缺失；不会任意加一个小数到分母。所有绝对误差单位均为 pu，不在未检查基准阻抗时转换为欧姆。

对该历史内一类边 \(C\)，先合成历史级 RMS：

\[
E^R_{imt,C}=
\sqrt{\frac1{|C|}\sum_{e\in C}(\delta^R_{imte})^2},\qquad
E^{R,\mathrm{rel}}_{imt,C}=
\sqrt{\frac1{|C|}\sum_{e\in C}(\eta^R_{imte})^2}.
\]

X 同理。合并 R/X 的统计量为

\[
E^{RX}_{imt,C}=
\sqrt{\frac{(E^R_{imt,C})^2+(E^X_{imt,C})^2}{2}}.
\]

相对合并量使用同样定义。对某一类边，若有必须的误差值缺失，则该历史该指标未定，不删除缺失边后悄悄降低 RMS。典型弱内部边可能只有 \(3\times10^{-5}\) pu：很小的绝对误差就会产生很大的相对误差。整体 relative RMSE 也可能被弱边的大误差主导，并非一定掩盖其量级；但它无法定位误差来自根、叶还是内部边，也不能区分弱/强网络层，因此各类分开报告。

## 3. 跨历史汇总与失败分母

固定方法、轮次、边类后，只对独立历史级 RMS 计算均值、中位数、5/25/50/75/95 分位数及最大值。三个叶边先在每个历史内合成，不能当作三个独立样本；滚动两轮也分别汇总，不能把同一历史变成两个独立历史。

每个汇总同时报告全历史分母、适用历史数量、可用历史数量、适用但缺失数量和不适用数量。漏真、真拟合未保存、真拟合未收敛、无效支持映射各自保留原因。所有数值均缺失时平均值为 None。

converged_summary 要求真候选 fit.success 和对应 record.success 同时成立。如果其他错误候选拟合失败，但真候选拟合成功，真参数误差仍可计算；另保存 whole_candidate_bank_fit_complete=False，不把它误当成整个辨识算法成功。

保存的未收敛解只有在已有 diagnostics 声明电压可行且 AC 方程及功率平衡残差均不超过 \(2\times10^{-10}\) 时，才进入 saved_feasible_unconverged_summary。它与收敛汇总严格分开。这个状态来自保存诊断，并非本模块重新验证 AC 可行性。其他已保存数值可留在逐边 CSV 中，但不并入上述两个汇总。

## 4. 如何解释阻抗误差

真拓扑的低似然损失不保证每条线路 R/X 准确。完整 AC 反射反例已经显示：在固定根功率因数比、同点逐时未知源、只测电压幅值的数据族内，两组不同的根 R/X 可以给出完全相同的计量均值；即使局部根参数 profile 秩二，全局解释仍然不唯一。因此非零阻抗误差不自动意味着似然求解器出了错。

弱内部边的巨大相对误差也可能来自响应差异小、源补偿、计量误差或工程边界，而非全部由优化失败造成。必须连同绝对误差、似然间隔、active bounds、源解释、profile 信息及失败记录分析。这里的任何误差分布都不是拓扑正确概率。

## 5. 使用方式

完整 evaluation_v3 后可运行模块 CLI，给出 input、out 路径；输出 impedance_edges.csv、impedance_history_components.csv 和 impedance_component_summary.json。D 的每个 round 自动分别汇总。输入 trial/config 与后处理源码都有 SHA-256；模块拒绝正在运行的未完成批次。
