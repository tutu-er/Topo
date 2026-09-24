# 基于 Conformal Prediction 的拓扑集合辨识

## 1. 核心目标

把拓扑辨识从强制 Top-1 分类改成候选集合输出：

\[
\widehat{\mathcal C}_{\alpha}(x)
=\{\mathcal T_k:\mathcal T_k\text{ 与观测 }x\text{ 不显著冲突}\}.
\]

对每一种真实拓扑分别控制误排除概率：

\[
\Pr\{\mathcal T^\star\in\widehat{\mathcal C}_{\alpha}(X)
\mid\mathcal T^\star=\mathcal T_k\}\ge 1-\alpha.
\]

这是重复运行意义下的有限样本覆盖率，不是当前样本的贝叶斯后验概率。Conformal 校准本身不提高分类准确率；基础评分越有判别力，预测集合才越小。

## 2. 物理模型和非一致性分数

末端 Smart Meter 观测为

\[
X^{\rm SM}=\{P_{i,t},Q_{i,t},|V_{i,t}|:
i\in\mathcal N_{\rm SM},t=1,\ldots,T\}.
\]

隐藏节点注入、线路参数误差和噪声分别记为 \(Z,\Xi,\varepsilon\)。拓扑 \(\mathcal T_k\) 下：

\[
X^{\rm SM}=h_k(U,Z,\Xi)+\varepsilon.
\]

理想预测分布应边缘化隐藏变量：

\[
p_k(x\mid U)=\int p(x\mid\mathcal T_k,U,Z,\Xi)
p(Z,\Xi\mid U,\mathcal T_k)\,dZ\,d\Xi.
\]

当前原型使用标称潮流电压残差：

\[
A_k^{\rm SM}(x)=
\sum_{t,i\in\mathcal N_{\rm SM}}
\left(\frac{|V_{i,t}^{\rm obs}|-|\bar V_{i,t}^{(k)}|}
{\sigma_{\rm SM}}\right)^2.
\]

投稿版本建议使用边缘负对数似然。若
\(X\mid\mathcal T_k,U\sim\mathcal N(\mu_k(U),\Sigma_k(U))\)，则采用

\[
A_k(x)=r_k^\top\Sigma_k^{-1}r_k+\log\det\Sigma_k,
\qquad r_k=x-\mu_k(U).
\]

协方差应同时描述隐藏负荷、线路参数误差、量测噪声和跨节点相关性。

## 3. Class-conditional conformal 校准

对拓扑 \(k\) 的独立校准单元计算

\[
S_i^{(k)}=A(X_i^{(k)},\mathcal T_k),\quad i=1,\ldots,n_k.
\]

测试观测对拓扑 \(k\) 的 conformal p-value 为

\[
p_k(x)=
\frac{1+\sum_{i=1}^{n_k}
\mathbf1\{S_i^{(k)}\ge A(x,\mathcal T_k)\}}{n_k+1}.
\]

输出

\[
\widehat{\mathcal C}_{\alpha}(x)
=\{\mathcal T_k:p_k(x)>\alpha\}.
\]

若条件于拓扑 \(k\) 的校准样本和测试样本可交换，且评分模型在校准前已经冻结，则测试分数在 \(n_k+1\) 个分数中的秩均匀，从而得到类条件覆盖率。

分位数实现应显式取第

\[
m_k=\lceil(n_k+1)(1-\alpha)\rceil
\]

个次序统计量。若 \(m_k>n_k\)，阈值只能取正无穷。对 \(\alpha=0.05\)，每类至少需要 19 个真正独立的校准单元才有非平凡阈值。

## 4. 集合驱动的微型 PMU 信息获取

初始集合只有一个拓扑时不请求附加量测；集合包含多个拓扑时，当前原型使用最差成对相角分离度：

\[
D_j=\min_{a\ne b\in\widehat{\mathcal C}}
\frac{|\bar\theta_{a,j}-\bar\theta_{b,j}|}{s_{a,j}+s_{b,j}},
\qquad j^\star=\arg\max_jD_j.
\]

取得相角后更新

\[
A_k^+=A_k^{\rm SM}+
\left(\frac{y_{j^\star}^{\theta}-\bar\theta_{k,j^\star}}
{s_{k,j^\star}}\right)^2.
\]

更合理的目标是直接最小化预期集合大小：

\[
j^\star=\arg\min_{j\in S}
\mathbb E[|\widehat{\mathcal C}_{\alpha}^{+}(X,Y_j)|\mid X]
+\lambda c_j.
\]

其中 \(S\) 是已经安装的微型 PMU 集合，\(c_j\) 是通信或调用成本。

## 5. 严格的自适应校准流程

数据必须分为模型拟合、策略设计、最终校准和测试四部分：

1. 在拟合集上训练拓扑条件物理模型。
2. 在策略设计集上确定固定安装集合 \(S\) 和在线查询策略 \(\pi\)。
3. 冻结模型、安装位置和查询策略。
4. 对每个最终校准样本执行相同策略 \(j_i=\pi(X_i^{\rm SM})\)。
5. 以 Smart Meter 和查询结果共同构造端到端分数 \(A_k^\pi\)。
6. 仅使用最终校准集计算 conformal 阈值。
7. 在线执行同一策略并输出最终 conformal 集合，而不是未经校准的 Top-1。

只要完整策略在最终校准前冻结，且校准事件与测试事件可交换，自适应测量后的最终集合仍可获得相应覆盖率。

固定布置应先求解

\[
S^\star=\arg\min_{|S|\le M}
\widehat{\mathbb E}[|\widehat{\mathcal C}_{\alpha,S}(X)|]
+\lambda\operatorname{Cost}(S),
\]

再限制在线查询满足 \(j\in S^\star\)。安装集合的选择不能使用最终校准集。

## 6. 当前实现审计

当前实现位于 src/topoident/innovation_conformal.py，其作用是验证机制，不是完整理论实现：

- 校准和测试均来自同一个仿真器和同一参数扰动分布；
- 同一 draw 的多个运行点共享线路参数扰动，不能全部当作独立样本；
- 分数只使用独立同方差的电压幅值残差；
- 相角尺度采用启发式标准差组合；
- 查询节点可随样本任意变化，表示移动或按需传感，不是固定安装；
- 查询后的输出是 Top-1，没有端到端重新校准；
- 空集合替换为最小分数标签属于集合扩张，不降低覆盖，但不是原始 conformal 集合。

实际数据建议把一天、一次切换事件或完整时间窗口作为一个校准单元。存在季节或负荷迁移时，需要 weighted conformal 或滚动/分块校准，并报告每种拓扑、季节和负荷区间的经验覆盖率。

推荐名称为：**隐藏注入不确定性下，基于物理边缘似然和类条件 Conformal 校准的拓扑集合辨识与集合驱动微型 PMU 信息获取。**

该方法仍以有限拓扑库为标签空间。完全不依赖候选拓扑库的方法见 docs/topology_free_methods.md。
