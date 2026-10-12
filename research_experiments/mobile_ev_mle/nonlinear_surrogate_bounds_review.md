# 非线性代理界推导的只读复核

复核对象为 nonlinear_surrogate_bounds_math.md。初版 SHA-256 为 e79c254f975414b829714737bcdc4b163b5bdb047fb8a4d205450dfbabaac71f；反馈后的复核版为 bdc3cee6a51d21237c719ad90c46a6521f887ec330d70308bd24fd00c5683856。未修改冻结代码、原数学文件或实验记录，未运行正式试验。

**结论：主要平方根界和 AC 子域下界方向正确。** 最新复核版已补齐我指出的近优点次优量、紧致域及实坐标/线段区域前提。以下给出可直接审查的推导及剩余的小补强。

## 1. 相同解释域上的内层界

固定 \(k,u\)，设

\[
d_R=\inf_{z\in Z_k}\|R_k(u,z)\|,\qquad
d_L=\inf_{z\in Z_k}\|L_k(u,z)\|.
\]

若 \(\|R_k-L_k\|\le\varepsilon_k\) 在整个同一域成立，则每个 z 有

\[
\|R_k\|\ge\|L_k\|-\varepsilon_k.
\]

对 z 取 inf 给 \(d_R\ge d_L-\varepsilon_k\)。取任意 \(\eta>0\) 的仿射近最小点，令 \(\|L_k(u,z_\eta)\|\le d_L+\eta\)，则

\[
d_R\le\|R_k(u,z_\eta)\|
\le d_L+\eta+\varepsilon_k.
\]

令 \(\eta\downarrow0\)，得到 \(d_R\le d_L+\varepsilon_k\)。于是

\[
|d_R-d_L|\le\varepsilon_k,\quad
[\max(0,d_L-\varepsilon_k)]^2\le q_k^{AC}
\le(d_L+\varepsilon_k)^2.
\]

不需要将 inf 错当作已被局部 AC 求解器取得的 min；紧致盒与连续函数可另保证理论 min 存在。平方根再平方的方向正确，不能把平方损失直接加减 \(\varepsilon_k\)。

若存在辅助 \(\|F_kz\|^2\)，建议显式定义
\(\widetilde R_k=(R_k,F_kz)\)、\(\widetilde L_k=(L_k,F_kz)\)，并对它们取平方范数。辅助项的差为零，原来的 \(\varepsilon_k\) 保持不变。当前版本“F 同样加入”意图正确，但显式增广能避免公式漏项。

## 2. min 竞争者及外层最优值

令 \(f(u)=\min_k d_{R,k}(u)\)、\(g(u)=\min_k d_{L,k}(u)\)，以及 \(\varepsilon=\max_k\varepsilon_k\)。逐点有

\[
g(u)-\varepsilon\le f(u)\le g(u)+\varepsilon.
\]

对相同 U 取 sup，得到

\[
|\sup_U f-\sup_U g|\le\varepsilon.
\]

非负量上平方根是单调连续函数，因此

\[
\sqrt{\sup_U\min_k q_k^{AC}}=\sup_U\min_k\sqrt{q_k^{AC}},
\]

仿射侧同理。由此得到原文的外层平方根界。若使用 max 记号，要有最优值存在；最新版本的非空紧致 U/Z、有限竞争集合、同一连续分支与连续残差可保证这一点。也可以统一用 sup，仍须保证优化值有限，否则 \(|\infty-\infty|\) 没有意义。

LB/UB 必须是真正针对同一冻结代理最优值的有效界。QP primal 上界、单动作成本上界和有限方向下界主问题的 best-bound 不能冒充全代理设计 UB。容差内 LB>UB 仍需保留数值方向冲突，不能通过平方根公式变成严格证书。

## 3. AC 可行子域

若 \(Z_k^{AC}(u)\subseteq Z_k\)，且残差误差界在 AC 子域成立，则

\[
\inf_{Z_k^{AC}(u)}\|R_k\|
\ge\inf_{Z_k^{AC}(u)}\|L_k\|-\varepsilon_k
\ge\inf_{Z_k}\|L_k\|-\varepsilon_k.
\]

所以用较大仿射域产生的下界仍有效。子域最优点可能比全域仿射最优点更差，不能沿用全域仿射 min 产生的上界。

最新版本已正确补充：对一个实际 AC 可行点 \(\hat z\)，直接计算实际成本可得 \(q_k^{AC}(u)\le\|R_k(u,\hat z)\|^2\)。若只用仿射成本，设 \(v_{\hat z}=\|L_k(u,\hat z)\|^2\)，则

\[
q_k^{AC}(u)\le(\sqrt{v_{\hat z}}+\varepsilon_k)^2.
\]

只有另有 \(v_{\hat z}\le q_k^{lin}(u)+\eta\) 的已证明次优界，才可写
\((\sqrt{q_k^{lin}(u)+\eta}+\varepsilon_k)^2\)。这一 \(\eta\) 不能省略。

建议再明确：如果 AC 子域为空，数学上的 inf 是 \(+\infty\)；但“数值优化未找出可行点”不证明子域为空。要把有限值的外层平方根界应用到物理子域，须补充非空/有限值前提，或限制到另行确认的设计域。一个已选动作的可行点也不能给所有动作的最大值上界。

外层统一上界的较弱充分条件是：对每个 u，至少有一个竞争对 \(k(u)\) 和其可行解释，其成本不超过一个已证明的统一 B；此时 \(\min_k q_k^{AC}(u)\le B\)，故外层值不超过 B。逐对、逐动作都给可行解释是更强的充分条件，但不是必要条件。缺失对不能删去以提高收益下界。

## 4. Taylor 和隐式 AC 条件

最新版本正确要求实坐标、包含每条 \(x_0\to x\) 线段的可解区域及同一连续分支。只在非凸 AC 可行子域抽样 Hessian，不保证线段上的余项界；端点可行也不保证中间可行或 Jacobian 可逆。必须有整体的 \(M_k\)，而非有限抽样最大值。

由于 AC 含共轭，\(G_V\) 必须是对 \((\operatorname{Re}V,\operatorname{Im}V)\) 的实 Jacobian。原文新增说明正确。对实方向 h，严格的二阶隐式式为

\[
V_h=-G_V^{-1}G_xh,
\]
\[
V_{hh}=-G_V^{-1}\bigl(
G_{VV}[V_h,V_h]+2G_{Vx}[V_h,h]+G_{xx}[h,h]\bigr).
\]

若区域内 \(\|G_V^{-1}\|\le\gamma\)、\(\|G_x\|\le a\)、三个二阶双线性算子的范数界为 \(b,c,d\)，则

\[
\|V_h\|\le\gamma a\|h\|,\quad
\|V_{hh}\|\le\gamma[b(\gamma a)^2+2c\gamma a+d]\|h\|^2.
\]

还须把完整计量残差视为 \(R(x)=H(V(x),x)\)，对 H 的导数给界；电压幅值及含 \(1/V\) 的量需要区域内电压模的正下界，白化要求协方差已知且正定。G 的界不能在未检查计量映射 H 的情况下直接当作 M。当前原文把有限 \(M_k\) 明确列作额外假设，所以定理本身未因此失效；补充这些式子会使“怎样取得 M”更完整。

所有坐标尺度要一致。若用 \(\tilde x=D^{-1}(x-x_0)\) 的欧氏半径，则 Hessian 界必须对应 \(R(x_0+D\tilde x)\)，不能用原 pu/kW 坐标的 Hessian 乘缩放半径。

## 5. 对本研究对象的范围

R 的两侧历史残差、未来均值差与 L 必须采用同一个参数映射：每侧同一线路参数同时出现在 H/N，同一节点 1 源接口、相同预测和白化；不能在未来单独再给一套线路。已有段落明确要求这些语义，方向正确。

这里的 q 是对称软历史设计准则的完整 AC 版本。它仍不是给历史计量重复计算两次独立 likelihood，也不是模型概率、proper EIG 或误拒绝控制定理。该误差界只把同一设计准则的仿射数值界转换成非线性数值界。

当前实际代码尚未取得 uniform \(\varepsilon_k\)。点值反例能够反驳未经误差补偿的 AC 下界宣称，却不能替代 uniform 证明。原文对此已写清楚。
