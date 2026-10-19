# 共享功率平方图上界：独立原型数学与验证协议

目标是减少冻结上界 master 的平方图复制，仅更换建模表示。冻结 C/D 源码不改；当前只验证两时段、两站点合成 toy，不读取或重新优化实际 H。没有大算例速度、可识别性或最优性性能声明。原 `_secant_sum` 已将 B=0 的常数残差行累入常数，本原型的新压缩来自对功率坐标建一套共享平方图。

对 pair p、固定可行 nuisance 见证 z，设 r=c+A z，h=||Fz||²，u=(u_ti)。原冻结目标 f_pz(u)=||r+B u||²+h，Q(u)=min_p min_z f_pz(u)。共享模型仅用于一个 EV 的 operation_model 运行域 U。

若每个 B 残差行的精确非零列只属于一个时段，则跨时段列内积为零。单 EV 时间展开单位路径及 stay 充放电绑定使每时段最多一个站点净功率非零，因此同期 i≠j 时 u_ti u_tj=0。这两个条件一起给出

f_pz(u)=||r||²+h+2rᵀB u+Σ_j d_pj u_j²，d_pj=||B[:,j]||²≥0。

同期 B 列可以不正交；压缩依赖运行域排除它们同时激活。若同一残差行跨时段有任何非零（包括 1e−300），当前原型拒绝压缩，不按 near-zero 截断。即使某些混合行经内积抵消而正交，仍保守拒绝。多 EV 不支持；多 EV 可令同期交叉项非零。非块结构的原残差平方图或显式交叉上界可作后续后端，本原型未实现。

原 symmetric_ac_local.mean_and_jacobian 按 time 将 state Jacobian 写入零初始化 block；whitening 为逐元素 sigma；adaptive_history._future_for_center 按时间取 P/Q state columns，build_frozen_pairs 将控制 B 只填入未来残差行。这解释了当前路径预期满足精确时段支撑，但实际 H 矩阵尚未运行审查，每次新 master 必须检查实际 B 后才应用压缩。

对功率界 [l_j,h_j] 构造含 0 的分段节点 v_jk，signed bounds 使用偶数段、左右各一半；charge-only [0,P] 自动含 0。以 interval binaries 和相邻端点 weights 表示全局图 s_j=secant(u_j²)。一套 s_j 为全部 pairs、全部见证共享，故每条上界 row 仅为

T≤||r||²+h+2rᵀB u+Σ_j d_pj s_j。

平方 secant 在段 [a,b] 的误差是 (u−a)(b−u)∈[0,(b−a)²/4]。d≥0 保持上界方向。0 为节点保证未激活 u_j=0 时 s_j=0，因而逐 pair 误差界为一般 E_p=Σ_j d_pj Δ_j²/4，更紧的单 EV 界为 E_p^phys=Σ_t max_i(d_pti Δ_ti²/4)，Δ_j 是该坐标最大段宽。

令 W(u)=min_p min_{z∈Z_p} f_pz(u)，S(u) 为共享 secant 上界，Z_p 为有限可行见证库，则 Q(u)≤W(u)≤S(u)，且 0≤S(u)−W(u)≤max_p E_p^phys。最后一个误差界只衡量平方 secant，不能把有限见证近似误差 W−Q 也算进来。用旧后端 control_box/proxy_box_upper 的 qbar、scale=max(1,qbar)，以 T/scale 求解；加 T≤qbar 仍保持全域 Q 上界，但可能截断 secant surrogate 的松弛最大值。

最大化 master 的全局 best bound×scale 上界 max_U Q；incumbent 只是该松弛的可行值，不能当全域 Q 上界。无 best bound 时返回 unavailable，即使已有 incumbent。浮点计算增加 RHS numerical guard，但不宣称 interval arithmetic 证书。snapshot 保存模型矩阵、运行索引、power knots、误差界、qbar 与 scale，并以规范 JSON 给 SHA256。

共享图的额外变量数（所有功率坐标非固定）为 J(3m+1)+1：每坐标 2m weights、m interval binaries、1 square，再加目标 T；与 pair/见证/残差行数无关。J=24,m=8 时为 601 个新增变量，这是构造计数，不是已测算例速度或最终内存结果。约束仍随见证数增加，连续松弛强度和求解器搜索行为尚未验证。

验证协议：固定两时段、两站点，构造非正交同期列/精确互斥跨期支撑的两 pair，含历史常数行及一维 nuisance；独立枚举单 EV 支撑、正负功率和连续段网格，直接算残差平方检查分解与 uniform secant gap。用独立标量二次解析 profile 与离散运行集合检查 Q≤W≤S，执行 toy 运行 MILP 检查 best-bound 方向、物理可行性及 snapshot 稳定性；显式检验非块、tiny 非零、多 EV 拒绝、零节点与 inactive error、incumbent 无 bound 不可提升。

授权边界：当前只运行上述 toy。H-only 710009 的旧/新 upper 同 3 秒预算比较、或任何实际 H 性能结论，须由主任务另行授权；本阶段不运行。
