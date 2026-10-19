# 共享线路估计的分解结构与ADMM选择

这里的算法构造为自行推导；共识ADMM形式参考Boyd等的全文第7.1.1节。
当前已知R/X排序实验未实现ADMM，以下没有ADMM实测速度结论。

## 1. 先区分共享变量和逐时变量

令theta=(R,X)表示所有线路参数。case33有32条投入物理线路，故theta为64维；
编码为观测叶节点时引入的零阻抗虚拟边不作为待估线路。w_t表示t时段需估的状态，
可含64维真实P/Q和固定物理接口的二维源P/Q。独立计量误差下的半平方残差为f_t：

\[
\min_{\theta\in\Theta,\{w_t\}}\sum_{t=1}^T f_t(\theta,w_t).
\]

Theta是有物理依据的线路约束。若允许逐时变化的源，必须在各时段通过同一接口
注入。共享仪表偏差或跨时预测先验会额外耦合时段，不在最简模型中隐含加入。

固定theta且没有跨时约束时，各时段已完全独立：

\[
\min_{\{w_t\}}\sum_t f_t(\bar\theta,w_t)
=\sum_t\min_{w_t}f_t(\bar\theta,w_t).
\]

因此本轮已知R/X任务直接按候选和观测并行；加入ADMM会引入不必要的协调。

## 2. 未知R/X：优先消去逐时增量

一次Gauss–Newton迭代中，r_t为当前标准化残差；A_t为其对theta的导数；B_t为其对
w_t的导数。令d、v_t分别为线路和逐时状态增量，lambda为数值阻尼，D为缩放矩阵：

\[
\min_{d,\{v_t\}}\tfrac12\sum_t\|r_t+A_td+B_tv_t\|^2
 +\tfrac\lambda2\|D_\theta d\|^2
 +\tfrac\lambda2\sum_t\|D_tv_t\|^2,
\qquad \theta+d\in\Theta.
\]

阻尼用于稳定当前迭代，不宣称其提供新的统计信息。对每期把B_t与sqrt(lambda)D_t
纵向合并得到B_tilde，同时将r_t、A_t下方补零。对B_tilde做QR分解，Q_1张成其列
空间，Q_2张成正交补。最小化v_t后只剩

\[
\tfrac12\|Q_2^T(\widetilde r_t+\widetilde A_t d)\|^2.
\]

证明：将残差正交分成Q_1、Q_2两部分；可通过v_t消去Q_1部分，Q_2部分与v_t无关。
满列秩时通过三角回代取得v_t；秩亏用带主元QR/SVD识别秩或加入声明的阻尼。
不必显式形成大投影矩阵，可用QR变换直接处理残差和导数块。

各期QR可以并行，汇总后求64维共享线路约束QP，再逐期回代。96或288个时段增加
逐期计算和统计信息，共享线路维度仍为64。R/X盒及线性X/R限制直接留在共享QP。
若逐时状态有活动不等式边界，则需在相应主动集内消元或解局部约束QP，不能将
无约束增量事后裁剪当作精确消元。随后必须用原完整AC目标做线搜索/信赖域接受。

## 3. 如果采用ADMM，应放在凸子问题层

记phi_t(d_t)为上述消去局部变量后的凸二次函数，g(d)为共享阻尼与线路约束的指示
函数。复制线路增量，要求每组d_t与共享d一致：

\[
\min_{\{d_t\},d}\sum_t\phi_t(d_t)+g(d),\quad d_t=d.
\]

取rho>0、a_t为缩放乘子，共识ADMM迭代为

\[
d_t^{j+1}=\arg\min_v\{\phi_t(v)+\tfrac\rho2\|v-d^j+a_t^j\|^2\},
\]
\[
d^{j+1}=\arg\min_v\{g(v)+\tfrac\rho2\sum_t\|d_t^{j+1}-v+a_t^j\|^2\},
\qquad a_t^{j+1}=a_t^j+d_t^{j+1}-d^{j+1}.
\]

这是标准共识结构：各组独立求解后协调共享线路。若每时段复制64个参数，288期
会引入18432个线路副本；实际可按CPU/机器分组，而非每期一个副本。集中64维QP
本来很小时，共识迭代和罚参数选择可能比直接QR汇总更贵。

Boyd等第3.2.2节明确讨论ADMM达到高精度可能较慢。这里候选间弱分差要求较准的
损失，所以比较应采用相同目标、域、起点与终止精度，计入全部协调迭代、潮流、
导数及通信时间，并核对分数、R/X和候选排序。

若直接在非线性AC目标上运行ADMM，各局部步仍须求非线性拟合，并未消除主要的
潮流和导数开销。Hong等的非凸共识ADMM驻点结论依赖光滑性、罚参数及子问题条件；
需对当前AC求解分支、边界和精度逐项核对。它不提供原AC问题的全局最优保证。

## 建议顺序

1. 当前已知线路：直接按候选/时段并行，已实施。
2. 未知线路：优先完整AC导数+逐时QR消元+小共享QP，原求解器作回退。
3. 仅在共享步骤或多机数据布局成为实测瓶颈时，再比较GN子问题层的ADMM。

分解改善计算结构，无法补上数据对R/X补偿方向缺少的信息。速度与可辨识性需分别检查。

另一个可选结构是按馈线子区分解线路参数，复制接口电压和边界P/Q并约束一致。
它能保留电网空间稀疏性，但各子区仍有非线性AC方程，还增加接口协调。当前case33
的共享线路规模很小，先利用按时段结构更直接；子区ADMM留待网络规模或数据分布
确实要求时评测。

全文来源：

- Boyd等（2011），[Distributed Optimization and Statistical Learning via ADMM](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)，第3.2、7.1.1节。
- Hong、Luo、Razaviyayn，[Convergence Analysis of ADMM for a Family of Nonconvex Problems](https://arxiv.org/pdf/1410.1390)，Assumption A、Theorem 2.4。
