# 弱支路完整 AC 信息界：独立复核

结论：核心构造和不等式可接受，未发现足以推翻 0.10715701486040206 界的代数或物理映射错误。该结论需保持以下条件和统计范围；建议纠正“完整 Gaussian 分布”的用语，并把共同控制器/latent law/路径约束写入定理，而非只留在后文说明。

本次只读检查数学稿、两个理论/AC 核验模块、已有测试及验证档案，并核对 C 项目的生成器、独立 R/X 的 ac_predict 和候选 support/B。用独立 Fraction 表达式重算得到

\[
\frac{3668918839949454447125}{34238718246575913269706}
=0.10715701486040206<0.11.
\]

现存数值验证记录的 C 源码哈希与现场文件一致。没有重复或扩大 72 路径的数值核验，没有修改主代码、策略或现有理论文件。

## 1. 物理与代数核查

1. **固定参数转换成立。** star 的两片叶边增加 \(2z_s\)，共用弱边删除，root stem 保留。这个映射在计量和控制前固定，同一组 star R/X 同时解释历史和全部未来；不是逐动作重新调线路。source 仍注入 root stem 下的节点 1，背景终端 P/Q 与源路径不变。
2. **reduced Z 恒等式的符号、系数正确。** cherry pair 子块为 diag 叶边加 \(z_s{\bf1}{\bf1}^{T}\)，star pair 子块为 diag 叶边加 \(2z_s I\)，相减为 \(z_s(e_i-e_j)(e_i-e_j)^T\)。四个保留坐标为 source1 及三个终端；无注入 junction 的删行/列是载流为零时的精确阻抗关系，不是近似 Kron 替代。source 行直接差为零不代表其电压或主表间接差为零。
3. **完整 AC 收缩界成立。** 所有 reduced 行的 \(\sum_k|Z_{jk}||S_k|\) 由
   \(C=.008(.885)+(.045+2\times.000044)(.445)=.02714416\)
   覆盖；cherry 弱边贡献 \(|z_s|(|S_i|+|S_j|)\le2Z_sM_{\max}\) 也被覆盖。球内 \(|V|\ge.9\)，不变性 \(C/.9<.1\)、收缩 \(K=C/.9^2<1\)。该 Lipschitz 估计也证明所有 loaded 坐标均至少 .9 的 AC 解唯一；它不排除其他低电压解。零注入 junction 重建的行和更小，完整模型电压域结论可保留。
4. **未充电端电流差界成立。** 单 EV、纯 P、最多一端非零使 pair 中能选一个未充电端 j。分解
   \(S_i/V_i-S_j/V_j=(S_i-S_j)/V_i+S_j(1/V_i-1/V_j)\)
   给出使用 \(M_b\) 的第二项；共用 stem/弱边在 pair 电压差中抵消。再用
   \(|V_i-V_j|\le Z_\ell|I_i-I_j|+\Delta Z_\ell M_b/.9\)
   得分母 \(3571/3600\)、L0/L1，与稿中一致。EV 在 pair 外时使用总 u 仍是合法放宽。
5. **高电压扰动界成立。** star 与 cherry 的固定点方程相减，直接 forcing 为 \(z_sd(I_i-I_j)\)，star 的收缩项产生 \(1/(1-K)\)，没有漏掉 source 间接反馈。
6. **Gaussian 主表系数正确。** 项目 master 为 root=1 时的 \(\overline{\sum I_k}\)，确含网损。相同负载下复电流差使
   \(|\Delta S_{\rm master}|\le M_{\rm tot}\delta/.9^2\)。
   同方差独立主表 P/Q 的二次型就是 \(|\Delta S_{\rm master}|^2/\sigma_{\rm master}^2\)，不再乘 2。电压计量有三个幅值通道，所以是 \(3/\sigma_V^2\)，不是四个 reduced 节点，也不是平方电压。终端 P/Q 条件均值相同；它们仍可能通过推断 latent path 帮助边缘辨识，但 revealing-L joint KL 上界已覆盖这种帮助。

没有将 v9 的固定 X/R outer-LP 当作独立 R/X 证明。当前物理代码和本构造确实使用独立逐边 R/X。

## 2. 必须说清的自适应过程条件

### 2.1 不是一般的联合 Gaussian 轨迹

给定确定的路径/参数及开环动作时，20 点计量联合为固定均值的 Gaussian 乘积。但随机 latent path 积分后通常为混合分布；自适应动作依赖旧计量时，整体观测/动作轨迹也通常不再是联合 Gaussian。

建议开头改为：

> 对每时刻条件计量服从指定 Gaussian 模型的完整观测/动作过程，在下述共同 latent law、共同因果控制器及路径功率/电量条件下，构造的 cherry/star 对满足 KL 不超过 0.10715701486040206。

这不改变 chain-rule 证明，但避免把条件 Gaussian 似然误当成边缘轨迹的多元 Gaussian。

### 2.2 共同 latent law 的基准空间

应明确公共 base variable 为 \(L=(\beta_{\rm cherry},S_{1:20}^{0},S_{s,1:20},\ldots)\)：

\[
\theta_{\rm cherry}(L)=\beta_{\rm cherry},\qquad
\theta_{\rm star}(L)=T(\beta_{\rm cherry}),
\]

其中 T 是实验开始前固定的 \(z_i\mapsto z_i+2z_s\) 转换。共同 \(\mu(dL)\) 指这个基准变量的同一个分布，并不意味着两种树有相同维度或相同数值的物理阻抗分布。两种模型的源/背景路径保持相同；该路径在控制前确定，且不被竞争解释在见到控制/计量后改写。

需要：噪声满足声明的逐时/通道独立 Gaussian 模型；共同随机化 R 与 L、计量噪声独立；背景/source 是本定理的恒功率外生过程，而非未建模的电压响应负载或依控制改变的 source law。更复杂过程可另建公共因果耦合，但不应直接套现有证明。

### 2.3 控制器和操作可行性

在给定相同已观测历史、已执行动作及同一随机化 R 时，两模型必须使用相同正常化动作核 \(\pi_t\)。由当前计量拟合出的模型、候选集或预测所驱动的共同算法可以符合此条件；直接读取真拓扑、模型特有 latent 状态或未来噪声的动作规则不能。

功率、最多一端充电、单位追踪、纯 P、前 12 点 u=0 与总 grid-energy 约束应逐路径成立（至少在比较的 joint law 下几乎处处）：

\[
0\le u_t\le.3,\qquad\sum_tu_t\le1.2.
\]

不能只凭有限样本平均电量、MILP success 或带数值容差的原始控制声称已满足该精确定理。实际使用时，数值越界须修复或显式扩大功率常数重新核算。

行驶、接入、SOC 等规则可以缩小控制集，但其状态/可行集合必须是两模型共同的已观测外生输入、动作历史或共享设备状态的函数。真拓扑依赖的不可观测物理 feasibility filter、模型特有中止/保护事件，或额外观测通道，可能改变动作核/轨迹信息，需明确建模。当前统一收缩域保证声明的高电压解存在；不能据此无条件覆盖任意更窄的电压安全阈值。

6 kWh 指 grid energy，不是电池净增能量。效率和行驶消耗改变 SOC/任务可行性，不自动增加该 grid budget。若为弥补行驶/效率而另加充电能量，路径能量常数也须相应改变。

### 2.4 chain rule 的正确完整写法

在以上条件下，joint law 可写为

\[
\mu(dL)\nu(dR)\prod_t
\pi_t(da_t\mid Y_{<t},a_{<t},R)\,
f_{M,t}(dY_t\mid L,a_t),
\]

且 f 为同协方差的条件 Gaussian。共同 latent/randomization/action 项的 log 比消失，因此

\[
D(P_{\rm cherry}^{L,R,\pi}\Vert P_{\rm star}^{L,R,\pi})
=\mathbb E_{\rm cherry}\sum_t
D(f_{{\rm cherry},t}(\cdot\mid L,a_t)\Vert
f_{{\rm star},t}(\cdot\mid L,a_t)).
\]

点态 AC 界及路径电量界控制此期待；再 marginalize L、R 得实际过程 KL 不增。两模型分别模拟时产生不同动作序列不会破坏证明。需要的相同动作是“同条件历史下的同动作核”，不是两次仿真逐路径取完全相同动作。

本界针对包含随机历史的完整 20 点过程；不能直接改称给定某条已实现 H 后的条件 KL 上界。两模型关于 L 的 H 后验可能不同。

## 3. TV 与检验结论的范围

使用保守 0.11，Pinsker 给 TV 不超过 0.23452078799；等先验二元检验平均错误率至少 0.38273960600。对随机化拒绝函数 \(\phi\in[0,1]\)，

\[
\mathbb E_{P_{\rm cherry}}\phi
\le\mathbb E_{P_{\rm star}}\phi+\operatorname{TV}(P_{\rm cherry},P_{\rm star}).
\]

所以 power 至多 alpha+TV 的必要前提是：null class 真正包含这个构造的 star 过程，且检验在该 null class 上 uniform size 不超过 alpha。若复合 null 的 size 是逐参数/路径统一控制，才可进一步混合得到该 Q 的 size；仅在某个 nominal mixture 上平均校准不够。

这里 alpha=.05 是示例条件，不是已证明的主实验门限性质。主研究的 nominal POP 中星型逐边参数范围/分布未必包含推送后的 star 叶边或其相关参数 law；物理拟合盒包含该参数不等于 nominal POP 校准覆盖它。现有稿已正确区分此范围，应继续保持。

这是一对物理过程的辨识限制；不是总体四拓扑准确率、某条历史的后验正确率、bootstrap 频率或 MLE 的最优性结论。它也不证明所有参数/强支路下 EV 都无效。

## 4. 文件层面的小修正

数学文件有三个实际 CR 字符，而非 CRLF 换行，位于单位公式的 rm 宏。建议用原始字符串写成

\[
\sum_tu_t\le
\frac{6\,\mathrm{kWh}}{20\,\mathrm{kW}\times0.25\,\mathrm h}=1.2 .
\]

这应在并入 LaTeX 报告前修正，避免把单位断行成不完整公式。独立精确常数及主表系数不受此文字错误影响。

总体建议：保留核心理论与现有有限数值核查；补上述过程条件和条件-Gaussian措辞后再用于报告。不需要继续扩大固定数值路径来支撑已经证明的 uniform 物理不等式；nominal mixture 校准应另按其真正参数/latent law研究。
