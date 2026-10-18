# 保留完整AC目标的分块求解

## 1. 同一个估计问题

T为时段数，n为电压表数量，m为候选编码边数量。theta表示共享线路的自由优化坐标；实现沿用R/alpha映射，将原R/X区间及X/R约束准确变为box。s_t=(P_t^src,Q_t^src)为同一接口当期的源。末端P/Q作为已到达输入，根电压和仪表校准偏差固定。

令h_t(theta,s_t)为完整AC预测的n个电压幅值及总表P/Q，y_t为对应读数，W_t为这些读数标准差的逆对角阵。定义

\[
r_t=W_t(h_t-y_t),\qquad F(\theta,s)=\frac12\sum_{t=1}^T\|r_t\|^2.
\]

FitResult.loss=2F；loglik=-F-1/2 sum log(2 pi sigma_i^2)。改变线性求解方法不会改变这个损失或归一化常数。这里仍是给定测得末端P/Q的条件输出模型，输入P/Q噪声传播不在本次数值改动内。

## 2. 固定线路时为何能分开

固定theta后，r_t只依赖s_t，各期源区间也互不耦合。因此

\[
\min_{s_1,\ldots,s_T}F(\theta,s)
=\sum_t\min_{s_t\in[l_t,u_t]}\frac12\|r_t(\theta,s_t)\|^2.
\]

证明：任何联合可行点的每项均不小于各项最小值；将各期最优点拼接得到反方向不等式。实际二维非线性局部算法不自动找到各项全局最小值，但分解本身准确。逐时失败及重启必须分别记录；“每行250次”与“批量250次”是不同预算。

## 3. 联合拟合如何消去源增量

记G_t=partial r_t/partial theta，S_t=partial r_t/partial s_t。S_t只有两列。使用正缩放a，令实际增量d=diag(a)eta。将G、S的列乘相应a后，仍简记为G、S；步界为L=(lower-x)/a、U=(upper-x)/a。每次求

\[
q(\eta)=\frac12\sum_t\|r_t+G_t\eta_\theta+S_t\eta_{s,t}\|^2
+\frac\lambda2\|\eta\|^2,\qquad L\le\eta\le U.
\]

lambda>0为数值阻尼，不进入最终AC统计损失。固定当前活动变量后，把其响应移到右端b_t；只对自由变量求解。设当期有q个自由源坐标（q=0、1、2），构造

\[
C_t=\begin{bmatrix}S_{t,F}\\\sqrt\lambda I_q\end{bmatrix},\quad
E_t=\begin{bmatrix}G_{t,F}\\0\end{bmatrix},\quad
d_t=\begin{bmatrix}b_t\\0\end{bmatrix}.
\]

QR分解C_t=Q_t[R_t;0]。同一正交变换作用于E_t、d_t，记上q行/剩余行为上、下两部分。自由源能恰好消去上部残差，故只需最小化下部残差，并保留线路的lambda阻尼：

\[
\min_{\eta_{\theta,F}}\sum_t\|\widetilde E_{t,\rm down}\eta_{\theta,F}
-\widetilde d_{t,\rm down}\|^2+\lambda\|\eta_{\theta,F}\|^2.
\]

求出共享线路增量后，通过R_t三角回代恢复当期自由源。QR正交变换保范数，这给出了与原自由子问题相同的最小值。lambda=0时源可能秩亏，使用SVD确定列空间与最小范数回代。整个过程不形成J^T J，也不创建各期源之间的全局零块。

## 4. 边界与内层验收

无约束自由解越界时，从当前可行步向该解走到首个边界，激活到界坐标并重求。得到自由可行解后，用完整子问题梯度g检查：自由坐标g=0，下界g>=0，上界g<=0；释放违反最大的活动坐标。预算耗尽时返回可行点和失败状态。

在边界坐标严格到界时，令v为去掉合法法向分量后的梯度：自由坐标v=g，下界v=min(g,0)，上界v=max(g,0)，固定坐标v=0。当lambda>0，q为lambda强凸函数，故

\[
0\le q(\eta)-q^*\le\frac{\|v\|^2}{2\lambda}.
\]

证明：选择box法锥中的n使v=g+n。强凸性及法锥定义给q(z)>=q(eta)+v^T(z-eta)+lambda/2||z-eta||^2；右端对z的无约束最小值为q(eta)-||v||^2/(2lambda)。此界仅针对当前阻尼凸子问题；在lambda很小时可能很松。lambda=0不报告这个界。

## 5. 完整AC决定是否接受

用线性模型、不含阻尼的损失下降作为预计改善，用重新计算完整AC所得下降作为实际改善。实际下降为正且比值达到预设门槛时接受；否则增大阻尼重求。源/线路box在步中处理，电压域仍沿用原程序的最终物理检查口径。

此求解器采用有界阻尼Gauss--Newton；原程序为SciPy TRF反射信赖域。两者的AC目标和可行参数域相同，迭代轨迹和局部终点可以不同，必须独立比较最终目标、梯度、物理状态和耗时。R/X信息不足的问题仍需由数据激励解决。

优化坐标的梯度还继承原映射的角点限制：当R=0且X区间塌缩时，partial X/partial alpha=0。此时R/alpha坐标的梯度通过，不自动认证物理R/X锥中的所有可行下降方向。报告只称“优化坐标的缩放投影梯度”，不称原物理域的KKT证书；本节前述凸子问题误差界仍准确成立。

## 6. 内层近似精度的有依据停机

设eta=0时的阻尼模型值为q0，已得可行步的模型值为q1，误差上界为e=||v||^2/(2lambda)。当q0>q1且e<=epsilon(q0-q1)时，该步到最优阻尼步的剩余模型收益至多是已实现收益的epsilon倍。实现的可选规则固定epsilon=1e-8；因此可在浮点KKT略未达到绝对门槛时接受已有精确度足够的近似内层步。报告仍保留KKT未通过，不把此规则称为外层AC的全局最优界。
