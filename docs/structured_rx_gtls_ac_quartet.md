# 结构化 R/X、GTLS、AC 候选重排与 Quartet 显著性

## 1. 直接约束 R/X 的核心思路

设根节点到所有末端节点的候选树共有 m 条边。定义路径关联矩阵

\[
A_{ie}=\begin{cases}
1,& e\in \mathcal P(0,i),\\
0,& \text{otherwise},
\end{cases}
\qquad A\in\{0,1\}^{n\times m}.
\]

对平方电压 LinDistFlow，若边电阻、电抗的 p.u. 值分别为 \(r_e,x_e\)，则

\[
R=A\operatorname{diag}(2r)A^\top,
\qquad
X=A\operatorname{diag}(2x)A^\top,
\qquad r\ge0,\ x\ge0.
\]

这个分解同时给出以下性质：

1. 对称和逐元素非负；
2. 半正定；
3. \(R_{ii}\ge R_{ij}\)，\(X_{ii}\ge X_{ij}\)；
4. \(R_{ij}\) 等于 i、j 根路径公共部分，公共祖先越深，值越大；
5. \(d_{ij}=R_{ii}+R_{jj}-2R_{ij}\) 是树上的可加路径距离；
6. 每个 quartet 的三个距离和中，最大的两个相等。

当前 `matrix_constraints.py` 中的 symmetry、nonnegative、diagonal order 和 PSD 只是凸外近似。它们不能保证存在某一棵树和某组非负边权精确生成该矩阵。

### 1.1 为什么未知拓扑时非凸

若 A 已知，变量只有 r、x，优化

\[
\min_{r,x\ge0}\|Y-\Phi_R(A)r-\Phi_X(A)x\|_2^2
\]

是凸非负二次规划。本项目使用投影加速梯度求解。

若 A 未知，可行域是

\[
\mathcal C_{\rm tree}=\bigcup_{T\in\mathbb T_n}
\left\{A_T\operatorname{diag}(w)A_T^\top:w\ge0\right\}.
\]

这是多个凸锥的并集，一般非凸。four-point 条件同样包含“哪两个和相等”的离散选择。因此，直接把精确树可加性作为一个普通凸约束是不成立的。

本次实现采用有限候选分解：

1. 用外约束 R/X 产生少量 RNJ 候选树 \(T_k\)；
2. 对每个固定 \(A_{T_k}\) 独立求解凸的非负 r/x；
3. 用 held-out AC likelihood 选择候选；
4. 用 quartet 和拟合边权判断是否收缩冗余隐藏边。

这相当于在有限个树锥上做可验证的离散搜索。

## 2. P/Q 有噪声与 GTLS

记真实模型为

\[
Y^\star=X^\star B,
\]

但实际观测为

\[
X=X^\star+E_X,\qquad Y=Y^\star+E_Y.
\]

OLS 只最小化 \(\|Y-XB\|_F^2\)，等价于假设 \(E_X=0\)。在一维且噪声独立时，

\[
\mathbb E[\hat\beta_{\rm OLS}]
\approx
\beta\frac{\operatorname{var}(x^\star)}
{\operatorname{var}(x^\star)+\sigma_x^2},
\]

所以系数会向零衰减。

### 2.1 普通 TLS

普通 TLS 求解

\[
\min_{\Delta X,\Delta Y,B}
\|[\Delta X,\Delta Y]\|_F^2,
\quad
Y-\Delta Y=(X-\Delta X)B.
\]

它等价于寻找增广矩阵 \([X,Y]\) 的最佳低秩近似。在同方差、独立高斯误差下可以通过一次 SVD 得到全局解。虽然原始约束写法看起来是双线性的，但特殊的低秩结构使普通 TLS 有闭式谱解。

### 2.2 Generalized TLS 什么时候仍可全局求解

若每行增广误差具有共同、已知且可分离的列协方差 \(\Sigma\)，可以先白化

\[
C_w=[X,Y]\Sigma^{-1/2},
\]

再做 TLS/SVD 或广义特征值分解。这个特例仍可全局求解。

### 2.3 本项目的 GTLS 为什么通常非凸

本项目的 P/Q 是相对误差，方差随样本幅值变化；V 与 P/Q 噪声量级不同，并且还要施加对称、非负和树结构约束。消去潜在真值后，残差

\[
e(B)=Y-XB=E_Y-E_XB
\]

的协方差依赖 B：

\[
\Omega(B)=\Sigma_{YY}+B^\top\Sigma_{XX}B
-\Sigma_{YX}B-B^\top\Sigma_{XY}.
\]

对应负对数似然包含

\[
\sum_t e_t(B)^\top\Omega_t(B)^{-1}e_t(B)
+\log\det\Omega_t(B),
\]

一般是非凸的。再叠加未知拓扑的树锥并集后，不可能保持为单个普通凸问题。

推荐实现顺序是：

1. 有序约束 LS 初始化；
2. 固定 B 更新每个样本的有效协方差；
3. 固定权重，在每个候选树锥上求解加权非负二次规划；
4. 迭代到稳定；
5. 用 held-out AC likelihood 选择候选，而不是相信单次局部 GTLS 目标。

若需要全局下界，可进一步测试 structured TLS 的核范数/重加权核范数松弛，但它是近似，不等于原始 GTLS。

## 3. RNJ 候选与 AC likelihood

候选由 R、X、归一化 R/X 距离和多个 RNJ tolerance 产生。对固定候选树，令

\[
p_e(t)=\sum_{j:e\in\mathcal P(0,j)}p_j(t),
\qquad
q_e(t)=\sum_{j:e\in\mathcal P(0,j)}q_j(t).
\]

则节点 i 的线性平方压降为

\[
\Delta u_i(t)=
\sum_e A_{ie}\left[2r_ep_e(t)+2x_eq_e(t)\right].
\]

这对 r、x 是线性的，因此可先做非负拟合。随后把 \(r_e,x_e\) 转回物理 p.u./Ohm，在 held-out 场景运行 backward-forward-sweep AC 潮流，并计算

\[
\mathcal L_k=
\frac12\sum_{t,i}
\left(\frac{V_{i,t}^{\rm meas}-V_{i,t}^{\rm AC}(T_k,r_k,x_k)}
{\sigma_{V,i,t}}\right)^2
+\frac{k_k}{2}\log N.
\]

第二项是 BIC 型复杂度惩罚。它非常重要：如果允许新增边的阻抗拟合为零，过度细分的候选树与较简单树具有几乎相同的 AC 电压。没有复杂度惩罚时，初始 demo 的平均 F1 从 1.000 降到 0.8587；加入 BIC 后四个 5 场景/288 点算例均恢复为 1.000。

## 4. Quartet 显著性

对候选 clade C 及其补集，取 \(a,b\in C\)，\(c,d\notin C\)。定义

\[
g_{ab|cd}=\min\{d_{ac}+d_{bd},\ d_{ad}+d_{bc}\}
-(d_{ab}+d_{cd}).
\]

若 \(C|C^c\) 是正长度树边诱导的 split，则精确可加距离满足 \(g_{ab|cd}>0\)，且数值与该 quartet 中间边长度成正比。代码把已知根节点作为补集中的额外锚点，因此也能检验“除一个末端外的其余节点”这类 rooted clade。

对 circular-block 子样本重复估计距离，得到每个 clade 的 gap 分布。隐藏边只有同时满足以下条件才保留：

\[
Q_{1-\alpha}(g)>\delta,
\qquad
\Pr(g>\delta)\ge\pi,
\qquad
\hat r_e+\hat x_e>\eta\,\operatorname{median}_{f}(\hat r_f+\hat x_f).
\]

第三个条件用于删除“bootstrap 中很稳定，但只由系统性距离偏差产生”的零阻抗细分。实验中 quartet-only 会接受这类稳定假边；加入固定树拟合的边效应后可以正确收缩。

## 5. 当前验证结果

### 5.1 五场景、每场景 288 点

| case | RNJ baseline F1 | AC+BIC F1 | Quartet+edge F1 |
|---|---:|---:|---:|
| paper15 | 1.000 | 1.000 | 1.000 |
| soumalas11 | 1.000 | 1.000 | 1.000 |
| flynn16 | 1.000 | 1.000 | 1.000 |
| pengwah18 | 1.000 | 1.000 | 1.000 |

### 5.2 三场景、每场景 96 点

`pengwah18` 的固定 RX75/tolerance=0.24 RNJ 从 F1=0.9333 提升到 AC+BIC+quartet F1=1.000；其余三个算例保持 1.000。这个结果证明了候选集中存在真树时，AC 重排可以修正 RNJ 参数选择，但样本量仍小，不能替代后续大规模 sweep。

## 6. 实现位置

- `terminal_case33/pipeline/rnj_candidates.py` and `terminal_case33/pipeline/ac_likelihood.py`：候选生成、固定树非负边权拟合、AC likelihood。
- `terminal_case33/graph/quartet_significance.py`：four-point gap 和 block-bootstrap 置信度。
- `terminal_case33/pipeline/topology_validation.py`：BIC 重排、quartet 和零阻抗边收缩的组合接口。
- `experiments/run_validated_topology_demo.py`：四算例验证。
- `tests/test_ac_likelihood.py`、`tests/test_quartet_significance.py`：确定性测试。

## 7. 文献

- Buneman, *A Note on the Metric Properties of Trees*, Journal of Combinatorial Theory B, 1974。给出树距离 four-point 条件。
- Malioutov and Slavov, [Convex Total Least Squares](https://proceedings.mlr.press/v32/malioutov14.html), ICML 2014。说明普通 TLS 的 SVD 特例以及现实 structured TLS 需要松弛。
- Gao et al., [Robust Coupling Impedance-Based Topology Identification Method for Asynchronous LV Distribution Networks](https://ssrn.com/abstract=6750574), 2026 预印本。提出 CI 对称/层次性质和 symmetry-constrained regression；尚未同行评审。
