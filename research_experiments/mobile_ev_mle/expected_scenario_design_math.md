# 按预测权重平均的 EV 判别代理：下一轮数学原型

本文件是自行推导与小型原型验证。它没有改变已冻结的 C/D/stress 策略，也没有新增 AC 校准或评价试验。已实现部分仅接受事先给定的仿射代理与预测权重，通过操作网络 MILP 配置动作。预测概率模型、完整 AC 场景构造及正式策略仍待建立。

## 1. 对象、符号与目标

已有历史记为 \(H\)。控制 \(u\in\mathbb R^{Tn}\) 是一辆 EV 在 \(T\) 个未来时段、\(n\) 个接入点的有功功率，按时段优先排列，单位 kW。正式目标可取 \(T=8\)，解析算例取 \(T=2\)。同一个 \(u\) 作用于所有预测情形，不能先知道未来情形再选择动作。操作可行集 \(\mathcal U\) 同时包含地点、时间展开移动路径、接入、连续功率、SOC、效率、移动能耗及充电任务约束；操作模块当前无功恒为零，并按计划功率检查 SOC。

用 \(h\in\mathcal H\) 表示竞争模型，用 \(\omega\in\Omega\) 表示互斥的未来预测情形。一个 \(\omega\) 是整个未来路径，包括负荷、同物理点源及追踪增益，不是八个可任意拼接的逐时独立标签。权重 \(w_\omega\) 事先声明，满足

\[
w_\omega\ge0,\qquad \sum_{\omega\in\Omega}w_\omega=1.
\]

权重只能来自当前历史和声明的预测模型，不能读取未来观测或仿真真值。当前 toy 的权重是工程声明，未经过预测校准，不能把它们称为已验证的未来概率。

在冻结的局部区域，对每个 \((h,\omega)\) 给定有限矩阵和非空参数盒

\[
Z_{h\omega}=[\underline z_{h\omega},\overline z_{h\omega}],\qquad
q_{h\omega}(u)=\min_{z\in Z_{h\omega}}
\left\|c_{h\omega}+B_{h\omega}u+A_{h\omega}z\right\|^2+
\left\|F_{h\omega}z\right\|^2. \tag{1}
\]

若第一项残差有 \(m\) 行、解释调整有 \(p\) 个坐标、附加罚有 \(r\) 行，则
\(c\in\mathbb R^m,B\in\mathbb R^{m\times Tn},A\in\mathbb R^{m\times p},F\in\mathbb R^{r\times p}\)，
\(z,\underline z,\overline z\in\mathbb R^p\)。这些维度可随候选变化，但控制维度必须一致。残差先按声明计量协方差白化，所以 \(q\) 是无量纲平方和。

\(c\) 是展开点残差，\(B\) 是控制导数，\(A\) 是可调整竞争解释的导数，\(F\) 是另外声明的二次罚项。历史残差可以直接包含在 \(c,A\) 中；若已包含，就不能再在 \(F\) 中重复加入同一份历史成本。当前完整历史残差版本的 \(F\) 为空。程序只检查维度和数值，不替调用者证明其 AC 参数映射正确。

AC 场景构造时，\(z\) 应含一份逐边线路调整 \(\delta\beta_{h\omega}\)、历史状态调整和未来逐时同点源调整。该份 \(\beta_{h\omega}\) 同时作用于历史和该 \(\omega\) 的所有未来时段，不能逐时换线路。源始终由同一个物理接口进入 AC 方程；EV 只在末端总 P/Q 中加入一次。

本轮目标是

\[
\Phi(u)=\min_{h\in\mathcal H}\sum_{\omega\in\Omega}w_\omega q_{h\omega}(u),
\qquad \Phi^\star=\max_{u\in\mathcal U}\Phi(u). \tag{2}
\]

竞争拓扑 \(h\) 在平均之后选择，不能随着 \(\omega\) 偷换身份；其数值参数 \(z_{h\omega}\) 可以在各互斥未来情形下分别重新拟合。这是模拟“观测不同未来后会得到不同竞争解释”，并不是认为一条实际线路同时具有多套参数。只会执行一个未来情形，历史也只有一份。式 (2) 是假想试验下的加权设计罚，不是将同一历史重复当成多个独立数据样本的联合似然。

对于相同参数盒，独立逐情形重拟合满足

\[
\sum_\omega w_\omega\inf_z f_\omega(u,z)
\le \inf_z\sum_\omega w_\omega f_\omega(u,z). \tag{3}
\]

证明：对任意共同的 \(z\)，每项下确界都不超过该项在 \(z\) 的值；加权求和后再对共同 \(z\) 取下确界即可。前者允许未来数据出现后调整解释，通常比强行让所有互斥未来共用同一估计更低。它仍然不是参数后验积分。

旧的最坏情形目标为 \(\Psi(u)=\min_{h,\omega}q_{h\omega}(u)\)，一般有

\[
\Psi(u)\le\Phi(u),\qquad
\sum_\omega w_\omega\min_h q_{h\omega}(u)
\le \min_h\sum_\omega w_\omega q_{h\omega}(u). \tag{4}
\]

第二个不等式说明：将平均与竞争者最小值交换，会得到另一种、更低的目标。比如两个竞争者的两情形距离分别为 \((0,100)\)、\((100,0)\)，权重各一半，式 (2) 为 50，而逐情形先选竞争者再平均为 0。实现必须保留这层次。

## 2. 从范数方向得到平方距离的仿射下界

以下省略 \((h,\omega)\)。将两项残差堆叠为

\[
r(u,z)=\begin{bmatrix}c+Bu+Az\\Fz\end{bmatrix},\quad
C=\begin{bmatrix}A\\F\end{bmatrix},\quad
v=\begin{bmatrix}v_1\\v_2\end{bmatrix},\quad \|v\|\le1.
\]

对参数盒求线性函数的最小值，得到

\[
\ell_v(u)=\min_{z\in Z}v^\top r(u,z)
=v_1^\top(c+Bu)+
\sum_i\min\{g_i\underline z_i,g_i\overline z_i\},
\quad g=A^\top v_1+F^\top v_2. \tag{5}
\]

这是 box support-LP 的显式解。这里必须取最小值，不能取最大值，也不能用某一个可行竞争解释代替盒内最小值。

**下界。** 对式 (1) 的最优点 \(z^\star\)，
\(\ell_v(u)\le v^\top r(u,z^\star)\le\|r(u,z^\star)\|=\sqrt{q(u)}\)。
因此对任意有限 \(a\ge0\)，

\[
d_{v,a}(u):=2a\ell_v(u)-a^2
\le 2a\sqrt{q(u)}-a^2
=q(u)-(\sqrt{q(u)}-a)^2
\le q(u). \tag{6}
\]

\(d_{v,a}\) 是控制的仿射函数。即便 \(\ell_v\) 为负，推导仍成立；\(a=0\) 给出合法的零下界。负 \(a\) 会使第一步不等式反向，必须拒绝。

**精确 QP 点的接触。** 在 \(u^\star\) 精确解出盒 QP，记残差 \(r^\star\)、\(a=\|r^\star\|\)。若 \(a>0\)，取 \(v=r^\star/a\)。盒约束一阶条件为

\[
(r^\star)^\top C(z-z^\star)\ge0,\qquad z\in Z.
\]

所以 \(v^\top r(u^\star,z)\ge a\)，在 \(z^\star\) 取等号。式 (5) 的最小值正好是 \(a\)，由式 (6) 得
\(d_{v,a}(u^\star)=a^2=q(u^\star)\)。若 \(a=0\)，取 \(v=0,a=0\) 也接触。参数最优点不唯一不影响该推导。

数值 QP 只有可行点或开放 gap 时，切面仍有效，但不能声称精确接触。代码保存 QP primal/dual 界；support 下界另加向下浮点保护。零梯度点的零切面不能自行发现新动作，需上界主问题或新的可行控制提供展开点。

## 3. 平方距离下界 MILP

给定每对若干有效切面 \(d_{h\omega j}(u)=b_{h\omega j}+D_{h\omega j}u\)，并加入 \(d=0\)。操作集包含在有限外盒 \(\underline u\le u\le\overline u\) 中。

对每对，任取至少一个盒内固定可行解释 \(\widetilde z\)。其残差第 \(i\) 行在外控制盒的范围为 \([L_i,U_i]\)，所以

\[
q_{h\omega}(u)\le
\|c+Bu+A\widetilde z\|^2+\|F\widetilde z\|^2
\le\sum_i\max(L_i^2,U_i^2)+\|F\widetilde z\|^2.
\]

多个固定解释取这些常数界的最小值，得到有限 \(Q_{h\omega}\)。该界使用宽松外盒，未假设所有外盒控制都满足接入/SOC/路线约束。

令
\(\underline d_{h\omega j}=\min_{\underline u\le u\le\overline u}d_{h\omega j}(u)\)，
取 \(M_{h\omega j}=Q_{h\omega}-\underline d_{h\omega j}\)，数值实现再加向上保护。主问题为

\[
\begin{aligned}
\max\ &t\\
\text{s.t. }&u\in\mathcal U,\quad 0\le s_{h\omega}\le Q_{h\omega},\\
&\sum_j b_{h\omega j}^{\rm bin}=1,\quad b_{h\omega j}^{\rm bin}\in\{0,1\},\\
&s_{h\omega}\le d_{h\omega j}(u)+
M_{h\omega j}(1-b_{h\omega j}^{\rm bin}),\quad\forall h,\omega,j,\\
&0\le t\le\sum_\omega w_\omega s_{h\omega},\quad\forall h .
\end{aligned} \tag{7}
\]

当二进制为零时，右侧至少为 \(Q_{h\omega}\)，因此约束被有效关闭；当二进制为一时，选中该仿射下界。最大化时相当于每对取最大方向下界，再对竞争者取加权平均后的最小值。零切面保证 \(s=0\) 总可行，包括零权重情形。

式 (7) 任一操作可行 incumbent 都满足 \(t\le\Phi(u)\le\Phi^\star\)，因而提供设计下界。它的 MILP best-bound 只上界“有限切面模型的最优值”，不能作为完整代理的上界。

## 4. 固定解释与分段弦上界 MILP

对每对保留若干固定盒内可行解释 \(\widetilde z_j\)。它们的成本
\(f_j(u)=\|c+Bu+A\widetilde z_j\|^2+\|F\widetilde z_j\|^2\)
均大于等于 \(q(u)\)。

对一行残差 \(x\)，在覆盖其外盒范围的分段端点 \(x_k,x_{k+1}\) 间，平方函数的弦为

\[
\overline f_k(x)=(x_k+x_{k+1})x-x_kx_{k+1},\qquad
0\le\overline f_k(x)-x^2
=(x-x_k)(x_{k+1}-x)
\le\frac{(x_{k+1}-x_k)^2}{4}. \tag{8}
\]

用分段选择二进制和该段两个端点权重编码。全部残差行的弦相加，再加固定 \(\|F\widetilde z_j\|^2\)，得到 \(\overline f_j(u)\ge f_j(u)\ge q(u)\)。上界主问题令

\[
0\le s_{h\omega}\le Q_{h\omega},\qquad
s_{h\omega}\le\overline f_{h\omega j}(u)\ \ \forall j,\qquad
0\le t\le\sum_\omega w_\omega s_{h\omega}\ \ \forall h. \tag{9}
\]

仍最大化 \(t\)，操作约束照旧。对任意真实代理可行控制 \(u\)，令
\(s_{h\omega}=q_{h\omega}(u)\)、\(t=\Phi(u)\)，式 (9) 可行。所以上界 MILP 的全局最优值至少为 \(\Phi^\star\)，其有效 global best-bound 提供完整冻结代理上界。上界 MILP 的 incumbent 不能代替该 global best-bound。

分段误差上界为各行 \(\Delta_i^2/4\) 之和。但是固定解释可能离内层最优解释很远；仅细化分段不能消除这部分差距。需要在新控制处解 QP，增加固定解释和方向切面。有限预算结束仍有 gap 时，应保留开放 gap；本原型没有一般性的有限迭代精确收敛定理。

取已验证控制下的 QP score lower 或式 (7) incumbent 为 \(L\)，式 (9) global best-bound 与外盒常数界的较小者为 \(U\)。停止条件为

\[
U-L\le\varepsilon_{\rm abs}+
\varepsilon_{\rm rel}\max(1,|U|).
\]

所有这些界只属于固定矩阵、固定参数盒、声明场景和权重。重线性化后是新的代理，不能继续使用旧切面或旧证书；代码通过 pair fingerprint 绑定切面，并重算 support 拒绝伪造或用错参数盒的下界。

## 5. 两时段独立解析算例

一接入点，两时段各一小时，\(0\le u_0,u_1\le6\)，等电量任务 \(u_0+u_1=6\)。电池初始 30 kWh，范围 10–90 kWh，效率取 1，故这些控制满足本 toy 的 SOC。

情形一：一个竞争者，\(q_1(u)=u_0^2,q_2(u)=u_1^2\)，权重 \((0.8,0.2)\)。期望目标为 \(0.8u_0^2+0.2(6-u_0)^2\)，是区间上的凸二次函数，最大值位于端点，分别为 28.8 和 7.2，所以选 \((6,0)\)。旧最坏情形 \(\min(u_0^2,u_1^2)\) 在 \((3,3)\) 达到 9；不能说这一例所有最坏分数均为零。

情形二：\(q_1(u)=u_0^2,q_2(u)=0\)。旧最坏目标对所有动作都是 0，期望目标仍在 \((6,0)\) 达到 28.8。这说明一个零距离预测情形可以使最坏情形准则失去选择力，而有声明质量权重的平均准则仍能区分动作。该结论是代理层面的，不说明 EV 实际排错更好。

另有非零罚项核验：
\[
q(u)=\min_{|z|\le2}(u_0+z)^2+3z^2
=\tfrac34u_0^2,\quad 0\le u_0\le6.
\]
最优 \(z=-u_0/4\) 在盒内。在 \(u_0=4\) 的方向/幅值切面接触 12；整段稠密审计均不超过独立解析值。负幅值、非单位方向、把 box min support 错写成 max support、错误权重、缺失场景、不可行上界解释均有拒绝测试。零权重保留输入校验和 QP 分母，但不贡献目标。

已通过 14 项测试，包括上述解析值的 \(L\le\Phi^\star\le U\)、层级顺序、有效 big-M，以及三时段两地点中实际选择移动弧与连续功率的 SOC 独立检查。程序未枚举完整路线。toy 数值、配置、矩阵/源码哈希保存在 weighted_proxy_toy_evidence.json。

另在非零 \(F\) 搜索中观察到数值边界问题：MILP 返回原始控制 \(u_0=6.00000000090,u_1=-0.00000000090\)，独立操作检查的违反量低于声明容差，但在该原始点计算的 score lower 比外盒 score upper 高约 \(5.4\times10^{-9}\)。严格将其记为上下界冲突和计算失败，保留原控制及全部记录，不通过放宽目标 gap 把它写成成功。数学证明针对精确可行集；程序使用普通浮点求解器和可行性容差，尚未实现严格的操作控制修复/区间算术证书。初版未拦截该微小冲突的记录另存 weighted_proxy_toy_evidence_initial.json，最新版本保留 failure。这个计算问题不推翻式 (6)–(9) 的精确模型推导。

## 6. 期望距离不等于边缘 KL、EIG 或排错概率

即使 \(\omega\) 的权重来自正确预测分布，先知道 \(\omega\) 后平均的 KL 也一般不等于未观察 \(\omega\) 的边缘预测 KL。对共同权重和合法密度
\[
p(y)=\sum_\omega w_\omega p_\omega(y),\qquad
q(y)=\sum_\omega w_\omega q_\omega(y),
\]
逐点 log-sum 不等式积分给出
\[
\mathrm{KL}(p\|q)\le
\sum_\omega w_\omega\mathrm{KL}(p_\omega\|q_\omega). \tag{10}
\]
证明也可写成条件分解：右减左等于
\(\mathbb E_{Y\sim p}\mathrm{KL}(P(\omega\mid Y)\|Q(\omega\mid Y))\ge0\)。
直觉是：混合观测没有直接告诉实验者情形标签；一些已知标签后的差异会被混合抵消。

反例：各一半权重，\(p_1=N(-a,\sigma^2),p_2=N(a,\sigma^2)\)，而
\(q_1=N(a,\sigma^2),q_2=N(-a,\sigma^2)\)。两边缘混合分布完全相同，KL 为零；情形条件 KL 的平均为 \(2a^2/\sigma^2>0\)。所以条件平均距离不能作为边缘信息量的下界。

式 (10) 只有在已给定合法密度且权重共同的前提下成立。本原型每情形独立 profile 数值参数，包含软历史罚项，并未定义完整的参数后验和边缘预测密度。因此式 (2) 连“正确边缘 KL 的已验证上界”也不能直接声称，更不能称为 proper EIG、信息下界或排错概率。

proper EIG 需要对拓扑及其参数、源/负荷预测构造完整的 H 条件预测密度，再对未来观测积分。当前原型解决的是在声明仿射预测质量下配置动作的优化问题。它不修复已发现的 AC 非线性代理误差、历史多解释模式遗漏、共同源支持错误或未校准预测权重。

## 7. 下一轮可讨论、尚未实现的物理预测构造

可从 H-only 预测模型抽取整段未来负荷、追踪误差及同节点源路径作为 \(\omega\)。参考解释的源路径在该抽样情形中固定，竞争拓扑的线路与源仍按历史+新增响应重新拟合。这比同时将参考源和竞争源都在宽盒内任意最小化，更接近“未来将以何种质量出现”的设计问题；需要先建立预测支持、时间相关性、仪表误差与线路参数处理的合法概率模型。追踪误差存在时的 SOC 与任务可行性也需纳入相应物理构造，当前操作网络仍使用计划功率。

不能通过任意收窄源盒或人为提高有利情形权重制造分离。如果每个有正权重的情形都有低成本可行竞争解释，加权目标仍会小；如果均值预测各时段完全对称，也不能声称发现了最佳时段。若未来模型或参考权重随拓扑变化，共同权重假设需重审，不能直接沿用式 (10)。

阶段验收只包括：数学层级明确、仿射上下界方向正确、toy 的独立解析最优值位于证书区间、操作计划可执行、错误输入不静默删竞争者、源码与配置可复核。通过这些检查后，才考虑新的完整 AC 预测场景构造与独立试验；当前不将该原型加入 formal policy。
