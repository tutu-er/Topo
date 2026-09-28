# 对称约束最小二乘：数学形式与代码对应

对应实现：`rnj_wzzt/estimation/constrained_least_squares.py`。

## 1. 原始矩阵问题

设共有 \(m\) 个样本、\(n\) 个终端：

\[
P,Q,Y\in\mathbb R^{m\times n},\qquad
A=[P\ Q],\qquad B=\begin{bmatrix}R\\X\end{bmatrix},\qquad
R,X\in\mathbb R^{n\times n}.
\]

函数接收 `design=A`、`target=Y`；`initial` 是两个对称矩阵的初值。
令 `output_transform` 为 \(T\in\mathbb R^{n\times n}\)，省略时取 \(T=I_n\)。求解

\[
\boxed{
\begin{aligned}
\min_{R,X}\quad &
\frac12\left\|(PR+QX-Y)T\right\|_F^2
+\frac\alpha2\left(\|R\|_F^2+\|X\|_F^2\right)\\
\text{s.t.}\quad &M=M^\top,\quad M_{ij}\ge0,
&&M\in\{R,X\},\\
&M_{ii}-M_{ij}\ge\delta_M,
&&M\in\{R,X\},\quad i\ne j.
\end{aligned}}
\]

其中 \(\alpha\ge0\)，固定裕量 \(\delta_R,\delta_X\ge0\) 由调用方传入。
`margins=None` 时去掉最后一组约束；只有一个终端时该组约束为空。
因 \(R,X\) 对称，写 \(PR+QX\) 与 \(PR^\top+QX^\top\) 等价。

**含义：**同一行的对角元素至少比每个非对角元素大一个固定裕量。
这不是 \(M_{ii}\ge\sum_{j\ne i}|M_{ij}|\) 的对角占优条件，
也不保证半正定或共同树拓扑。右变换 \(T\) 只作用于残差，正则项仍惩罚原始 \(R,X\)。

上层 `multiscenario.py` 对齐后直接堆叠各场景 \(P,Q,Y\)，其中
\(Y_{ti}=V_0(t)^2-V_i(t)^2\)，根电压已观测。不额外中心化或引入截距；
本文件求解给定矩阵问题，也不重新估计裕量。

## 2. 上三角变量与标准最小二乘

令 \(k=n(n+1)/2\)。对每个 \(i\le j\)，定义对称基矩阵

\[
E^{ij}=\begin{cases}
e_i e_i^\top,&i=j,\\
e_i e_j^\top+e_j e_i^\top,&i<j.
\end{cases}
\]

用正尺度 \(s_R,s_X\) 表示物理矩阵：

\[
R=s_R\sum_{i\le j}z_{R,ij}E^{ij},\qquad
X=s_X\sum_{i\le j}z_{X,ij}E^{ij},\qquad
z\in\mathbb R^{2k}.
\]

尺度由目标与两个设计块的 Frobenius 范数确定，并不取决于可能发散的无约束系数。
对于非零目标和设计，\(s_M=\|Y\|_F/\|A_M\|_F\)；设计块全零时分母改为
\(\|A\|_F\)，目标或整个设计全零时初始尺度取 1；有裕量时再取
\(s_M\leftarrow\max(s_M,\delta_M)\)。这些操作只改变坐标。

记 \(\operatorname{vec}_r\) 为与 NumPy `ravel()` 一致的逐行展开。
数据矩阵 \(F_0\) 的对应列为

\[
(F_0)_{R,ij}=s_R\operatorname{vec}_r(PE^{ij}T),\qquad
(F_0)_{X,ij}=s_X\operatorname{vec}_r(QE^{ij}T),\qquad
y_0=\operatorname{vec}_r(YT).
\]

代码直接填写这些列的非零位置，不构造大型基矩阵数组。

对称矩阵的非对角元素在 Frobenius 范数中出现两次：

\[
\|M\|_F^2=\sum_i M_{ii}^2+2\sum_{i<j}M_{ij}^2.
\]

因此令 \(D\) 为对角矩阵，其对应元素为
\(D_{M,ij}=s_M\sqrt{\alpha w_{ij}}\)，其中对角 \(w_{ii}=1\)，非对角 \(w_{ij}=2\)。
将 ridge 写成附加观测行：

\[
F=\frac1c\begin{bmatrix}F_0\\D\end{bmatrix},\qquad
y=\frac1c\begin{bmatrix}y_0\\0\end{bmatrix},\qquad
c=\begin{cases}\|y_0\|_2,&y_0\ne0,\\1,&y_0=0.\end{cases}
\]

当 \(\alpha=0\) 时直接省略 \(D\) 和对应零行。
除以同一个正常数 \(c\) 不改变最优解。

非负约束为 \(z\ge0\)，有序约束为
\(z_{M,ii}-z_{M,ij}\ge\delta_M/s_M\)。合并记为 \(Cz\ge d\)，得到

\[
\boxed{\min_z\ \frac12\|Fz-y\|_2^2\quad\text{s.t. }Cz\ge d.}
\]

## 3. 数值坐标变换

`_solve_qp` 只处理上述标准问题。记
\(H=F^\top F\)、\(\ell=-F^\top y\)。三种情况共用一次 `minimize` 调用。

- **数值正定：**若 \(\lambda_{\min}>10^{-12}\lambda_{\max}>0\)，且 Cholesky 分解成功，
  取 \(H=U^\top U\)、\(W=U^{-1}\)、\(a=W^\top\ell\)，令 \(u=Uz+a\)。
  此时 \(z=W(u-a)\)，目标等价于 \(\tfrac12\|u\|_2^2\) 加常数，
  约束变为 \(CWu\ge d+CWa\)。
- **近奇异或秩亏：**对 \(H=V\operatorname{diag}(\lambda_i)V^\top\)，令
  \(W=V\operatorname{diag}(\max(\lambda_i,10^{-6}\lambda_{\max})^{-1/2})\)，
  并取 \(z=Wu\)。直接计算原目标 \(\tfrac12\|FWu-y\|_2^2\)。
  特征值下限只用于坐标变换，不修改损失、不增加 ridge、不删去零空间。
- **\(H=0\)：**使用原坐标，即 \(W=I,a=0\)。

各分支统一使用 \(z=W(u-a)\)，再将每个约束及其右端除以该约束行的范数。
近奇异分支从残差计算梯度，避免展开二次式时的大数相减。

## 4. 可行性与检查边界

`make_feasible` 假设输入已对称，先截断负值，再按需要抬高对角元素。
它用于构造可行初值与清理舍入误差，**不是到约束集合的最近点投影**。
初值从修正后的输入和最小对角边界点中选择目标较小者。

求解失败、解含非有限值，或缩放变量下最大约束违反超过 \(10^{-8}\)，均抛出异常。
只有检查通过后才恢复物理矩阵并清理舍入误差。
SLSQP 成功状态和可行性检查是数值诊断，不是对任意输入的严格最优性证书。
当 \(\alpha=0\) 且数据存在不可辨识方向时，最优矩阵可能不唯一。

新增 `tests/test_constrained_least_squares.py` 直接从矩阵残差计算梯度，
检查非负乘子、活跃约束和 KKT 驻点条件，覆盖共线设计、奇异/非对称输出变换、
有/无 ridge、有/无有序约束；另检查单终端边界与非法模型参数。
当前解析解、单位缩放、已观测根基准和父项目兼容测试作为回归验证；
下列 2026-09-26 记录对应改成零截距前的历史模型，不能作为新主线效果结论。

2026-09-26 验证：改写前直接相关测试 68 项通过；改写后下列 212 项测试全部通过，
其中本次新增 24 项。运行环境为本地 Topo conda 环境。

~~~powershell
& D:\apps\miniconda3\envs\Topo\python.exe -B -m pytest -q -p no:cacheprovider rnj_wzzt_core/tests/test_constrained_least_squares.py rnj_wzzt_core/tests/test_multiscenario_regression.py rnj_wzzt_core/tests/test_common_mode_qp.py rnj_wzzt_core/tests/test_independent_estimation_properties.py rnj_wzzt_core/tests/test_matrix_constraint_repairs.py tests/test_matrix_constraints.py
~~~


## 5. 阅读顺序与函数职责

建议先读公开入口 `solve_symmetric_least_squares`，再读 `_solve_qp`，最后读
`make_feasible`。调用关系为：

~~~text
fit_projected_sensitivity                  上层：对齐并堆叠原始数据、确定固定裕量
  └─ solve_symmetric_least_squares         本文件：把矩阵模型编码为 F、y、C、d
       ├─ make_feasible                   构造两个可行候选初值
       ├─ _solve_qp                       变换坐标、调用 SLSQP、检查求解结果
       └─ make_feasible                   清理已通过检查的边界舍入误差
~~~

这里有三层变量，不能混用：

\[
\underbrace{R,X}_{\text{物理矩阵}}
\quad\longleftrightarrow\quad
\underbrace{z}_{\text{除以尺度后的上三角元素}}
\quad\longleftrightarrow\quad
\underbrace{u}_{\text{数值求解坐标}}.
\]

`output_transform` 是作用于残差输出方向的 \(T\)，改变它一般会改变拟合问题；
`_solve_qp` 内的 `transform` 是变量坐标变换 \(W\)，不改变拟合问题。
这两个名称相似，但作用对象和数学含义不同。

## 6. 输入、输出和变量总表

### 6.1 公开入口

记样本数为 \(m\)，终端数为 \(n\)，每个对称矩阵的独立元素数为
\(k=n(n+1)/2\)，总优化变量数为 \(p=2k=n(n+1)\)。
这里小写 \(p\) 是维数，大写 \(P\) 是有功数据矩阵。

| 参数 | 形状或类型 | 数学含义与约定 |
|---|---|---|
| `design` | \((m,2n)\) | \(A=[P\ Q]\)，前 \(n\) 列为 \(P\)，后 \(n\) 列为 \(Q\) |
| `target` | \((m,n)\) | \(Y\)，每行一个样本、每列一个终端的回归目标 |
| `initial` | \((2,n,n)\) | 两个已对称的初值矩阵，`initial[0]` 对应 \(R\)，`initial[1]` 对应 \(X\) |
| `margins` | `None` 或 \((2,)\) | 固定的 \((\delta_R,\delta_X)\)，不在迭代中更新 |
| `alpha` | 非负有限标量 | ridge 系数；为零时不添加正则行 |
| `max_iterations` | 正整数预算 | SLSQP 迭代上限，不要求一定执行这么多次 |
| `output_transform` | `None` 或 \((n,n)\) | 残差右变换 \(T\)，默认为 \(I_n\)，允许非对称或奇异 |

设计与目标的行必须对应同一组观测，两个设计块的终端顺序必须与目标列顺序一致。
上层负责场景对齐、数据有限性检查和堆叠；本函数显式检查 `alpha`、
`margins` 和 `output_transform` 的相应约定。`initial` 的对称性是输入前提，
`make_feasible` 本身不会把非对称输入对称化。

上层参数 `constraint_refine_iterations=0` 的含义是跳过本求解器，仅返回可行初值；
这不是要求直接向本函数传入零迭代预算。

返回值：

~~~python
blocks, diagnostics = solve_symmetric_least_squares(...)
r_matrix, x_matrix = blocks
~~~

`blocks` 的形状是 \((2,n,n)\)，存放恢复尺度后的物理矩阵；
`diagnostics` 是求解状态字典，不包含本函数重新计算的原始目标值。
上层 `fit_projected_sensitivity` 会另行从矩阵残差计算目标值，不拟合截距。

### 6.2 建模过程中的变量

| 变量 | 形状 | 含义 |
|---|---|---|
| `design_blocks` | 两个 \((m,n)\) 数组 | 拆分得到的 \(P,Q\) |
| `target_norm` | 标量 | \(\|Y\|_F\)，在右变换前计算 |
| `design_norm` | 标量 | \(\|[P\ Q]\|_F\)，设计块为零时的尺度回退分母 |
| `scales` | \((2,)\) | \(s_R,s_X\)，将物理矩阵换算为数值变量 |
| `ii, jj` | 各 \((k,)\) | 同一组上三角坐标的行、列下标 |
| `columns` | \((k,)\) | 当前 \(R\) 块或 \(X\) 块在整个变量向量中的位置 |
| `features` | 先 \((m,n,p)\)，后 \((L,p)\) | 每个变量对预测输出的线性系数；最终是标准问题中的 \(F\) |
| `y` | 最终 \((L,)\) | 右变换、展开、附加零行并归一化后的目标 |
| `normalizer` | 标量 | \(c=\|\operatorname{vec}_r(YT)\|_2\)，零目标时取 1 |
| `multiplicity` | \((k,)\) | 对角为 1，非对角为 2 |
| `ridge` | \((p,)\) | 正则矩阵 \(D\) 的对角元素 |
| `position` | \((n,n)\) 整数数组 | 把任意矩阵坐标 \((i,j)\) 映射到对应的上三角变量编号 |
| `rows` | 最终为 \(q\) 个长度 \(p\) 的行 | 约束矩阵 \(C\) 的各行 |
| `lower` | 最终 \((q,)\) | 约束右端 \(d\) |
| `candidates` | 两个 \((2,n,n)\) 数组 | 输入修正初值和最小对角边界点 |
| `starts` | 两个 \((p,)\) 向量 | 对应的缩放上三角初值 |
| `start` | \((p,)\) | 两个候选中损失较小的 \(z_0\) |
| `values` | \((p,)\) | 求解得到的 \(z\)，尚未恢复物理尺度 |
| `blocks` | \((2,n,n)\) | 恢复尺度和对称位置后的 \(R,X\) |

这里 \(L=mn\)（无 ridge），或 \(L=mn+p\)（有 ridge）。
约束行数为 \(q=p\)（无有序约束），或 \(q=p+2n(n-1)\)（有有序约束）。
`block` 是当前块编号 0 或 1；循环中的 `a` 是当前设计块 \(P\) 或 \(Q\)，
`scale` 是该块的尺度，`margin` 是该块除以尺度后的固定裕量。

## 7. 公开入口逐段解释

### 7.1 用数据确定变量尺度

~~~python
design_blocks = np.split(design, 2, axis=1)
target_norm = float(np.linalg.norm(target))
design_norm = float(np.linalg.norm(design))
scales = np.ones(2)
if target_norm > 0.0 and design_norm > 0.0:
    scales = np.array([target_norm / (np.linalg.norm(a) or design_norm) for a in design_blocks])
if margins is not None:
    scales = np.maximum(scales, margins)
~~~

如果典型的设计幅值很小，拟合相同目标就需要较大的矩阵元素。
比例 \(\|Y\|_F/\|P\|_F\) 或 \(\|Y\|_F/\|Q\|_F\) 用于提供粗略的单位尺度，
并不声称等于真实参数大小。`or design_norm` 表示：某个块的范数恰为零时，
使用整个设计的范数，避免除零。目标或整个设计全零时保留初始尺度 1。

`scales = np.maximum(scales, margins)` 还使 \(\delta_M/s_M\le1\)，
避免有序约束右端过大。该操作没有把裕量加到估计值上。

随后使用 \(M_{ij}=s_M z_{M,ij}\)。对可行域、正则项和最终输出都使用同一尺度，
所以不会改变原问题中 \(R,X\) 的相对惩罚权重。

### 7.2 用上三角变量保证对称性

~~~python
ii, jj = np.triu_indices(n)
k = len(ii)
~~~

以 \(n=2\) 为例，得到：

~~~text
ii = [0, 0, 1]
jj = [0, 1, 1]
k = 3
z = [z_R,00, z_R,01, z_R,11, z_X,00, z_X,01, z_X,11]
~~~

三个数就能表示一个对称矩阵：

\[
R=s_R
\begin{bmatrix}
z_{R,00}&z_{R,01}\\
z_{R,01}&z_{R,11}
\end{bmatrix}.
\]

因此无需给求解器增加 \(R_{01}=R_{10}\) 这类等式约束。
增加的只是恢复矩阵时的一次镜像赋值。

### 7.3 每一列特征对应一个矩阵元素的贡献

~~~python
features = np.zeros((len(target), n, 2 * k))
for block, (a, scale) in enumerate(zip(design_blocks, scales)):
    columns = block * k + np.arange(k)
    features[:, jj, columns] = scale * a[:, ii]
    features[:, ii, columns] = scale * a[:, jj]
~~~

`features[t,j,h]` 表示：第 \(h\) 个变量增加一单位时，第 \(t\) 个样本、
第 \(j\) 个输出的预测值增加多少。

因为
\[
(P R)_{tj}=\sum_i P_{ti}R_{ij},
\]
上三角变量 \(z_{R,ij}\) 对输出 \(j\) 的贡献是 \(s_RP_{ti}\)；
当 \(i\ne j\) 时，它还代表 \(R_{ji}\)，因此对输出 \(i\) 的贡献为 \(s_RP_{tj}\)。
这正是两次索引赋值。

对于 \(i=j\)，两次写入同一个位置、写入相同值；这里是赋值而非累加，
所以不会把对角贡献错误地翻倍。`X` 块完全相同，只需将 \(P\) 换成 \(Q\)。

更具体地，当 \(n=2,T=I\)，一个样本对应的两行特征为：

\[
F_{0,t}=
\begin{bmatrix}
s_RP_{t0}&s_RP_{t1}&0&s_XQ_{t0}&s_XQ_{t1}&0\\
0&s_RP_{t0}&s_RP_{t1}&0&s_XQ_{t0}&s_XQ_{t1}
\end{bmatrix}.
\]

它乘以上面的六维 \(z\)，恰好得到两个终端的预测值。
将每个样本的两行按顺序堆叠，就是后续展开得到的设计矩阵。

### 7.4 右变换与逐行展开

~~~python
y = target
if output_transform is not None:
    features = np.einsum("tjk,ji->tik", features, output_transform)
    y = target @ output_transform
features, y = features.reshape(target.size, 2 * k), y.ravel()
~~~

这条 `einsum` 的下标含义是：

\[
\text{new\_features}_{tih}
=\sum_j \text{features}_{tjh}T_{ji}.
\]

它对每个变量所产生的输出向量应用同一个右变换，
从而使线性残差恰好等于 \(\operatorname{vec}_r((PR+QX-Y)T)\)。
代码字符串中的 `k` 是求和记号里的变量轴名称，并不是在此处执行对整数 `k` 的运算。

`reshape` 和 `ravel` 都使用默认的逐行顺序。
例如 \(Y=\bigl[\begin{smallmatrix}y_{00}&y_{01}\\y_{10}&y_{11}\end{smallmatrix}\bigr]\)
展开为 \([y_{00},y_{01},y_{10},y_{11}]\)；
`features` 的行也按同一顺序排列。顺序一致是保持预测关系的必要条件。

### 7.5 把 ridge 写成附加的残差

~~~python
normalizer = float(np.linalg.norm(y)) or 1.0
if alpha > 0.0:
    multiplicity = np.where(ii == jj, 1.0, 2.0)
    ridge = (scales[:, None] * np.sqrt(alpha * multiplicity)).ravel()
    features = np.vstack([features, np.diag(ridge)])
    y = np.concatenate([y, np.zeros(2 * k)])
features, y = features / normalizer, y / normalizer
~~~

`scales[:, None]` 把形状 \((2,)\) 改成 \((2,1)\)，与 \((k,)\) 的因子广播，
得到 \((2,k)\) 的两组正则系数，然后按 \(R\) 在前、\(X\) 在后的顺序展开。

两终端时，\(R\) 块的 `multiplicity` 为 \([1,2,1]\)，
`ridge` 为 \([s_R\sqrt\alpha,s_R\sqrt{2\alpha},s_R\sqrt\alpha]\)。
于是新增三行的平方残差之和为

\[
\alpha s_R^2(z_{R,00}^2+2z_{R,01}^2+z_{R,11}^2)=\alpha\|R\|_F^2.
\]

两块一起加入就得到原来的正则项。这样后续目标和梯度可以统一使用
`residual = features @ z - y`，不必分别实现数据项和 ridge 项。

归一化后的标准目标是原矩阵目标的 \(1/c^2\) 倍。
`normalizer` 不改变最优解，但会改变目标函数的数值尺度。

### 7.6 构造所有线性不等式

~~~python
rows, lower = list(np.eye(2 * k)), [0.0] * (2 * k)
position = np.empty((n, n), dtype=int)
position[ii, jj] = position[jj, ii] = np.arange(k)
~~~

单位矩阵的第 \(h\) 行表达 \(z_h\ge0\)。
`position` 让同一个物理元素的两个对称坐标指向同一个变量编号；
当 \(n=2\) 时：

~~~text
position = [[0, 1],
            [1, 2]]
~~~

接下来给每个 \(M_{ii}-M_{ij}\ge\delta_M\) 增加一行：

~~~python
row = np.zeros(2 * k)
row[block * k + position[i, i]] = 1.0
row[block * k + position[i, j]] = -1.0
rows.append(row)
lower.append(margin)
~~~

此时循环中的 `margin` 已是 \(\delta_M/s_M\)。
例如 \(R_{00}-R_{01}\ge\delta_R\) 对应

\[
[1,-1,0,0,0,0]z\ge\delta_R/s_R,
\]

而 \(R_{11}-R_{10}\ge\delta_R\) 对应

\[
[0,-1,1,0,0,0]z\ge\delta_R/s_R.
\]

因此必须遍历所有 \(i\ne j\)，不能只遍历 \(i<j\)：
一个非对角元素必须分别小于它两端的对角元素。

### 7.7 选择初值和恢复输出

~~~python
candidates = [make_feasible(initial, margins), make_feasible(np.zeros_like(initial), margins)]
starts = [(blocks[:, ii, jj] / scales[:, None]).ravel() for blocks in candidates]
start = min(starts, key=lambda z: np.linalg.norm(features @ z - y))
~~~

第一个候选尽量保留输入估计，第二个候选只保留约束所要求的最小对角。
当 \(n>1\) 且有裕量时，第二个候选为 \(\delta_RI_n,\delta_XI_n\)；
无裕量或 \(n=1\) 时为零矩阵。
共线设计可能使无约束估计非常大，修正后的输入也可能有很差的损失，
所以再提供这个简单边界点。

比较的是已含 ridge 的完整残差范数。
因为平方和乘正常数都保持大小关系，这与比较完整目标一致。

求解结束后：

~~~python
blocks = np.zeros((2, n, n))
blocks[:, ii, jj] = values.reshape(2, k) * scales[:, None]
blocks[:, jj, ii] = blocks[:, ii, jj]
~~~

`values.reshape(2,k)` 把 \(R\) 与 \(X\) 的变量分开；
乘 `scales` 恢复物理量；第二次赋值复制到下三角。
因此最终输出始终使用同一个非对角参数的两个镜像位置。

## 8. `_solve_qp` 的代码和变量

### 8.1 标准问题与 Hessian

这里的参数 `target` 是前一函数传入的一维 \(y\)，
参数 `initial` 是一维 \(z_0\)，已经不是公开入口的目标矩阵或三维初值。

~~~python
hessian = features.T @ features
hessian = 0.5 * (hessian + hessian.T)
eigenvalues, eigenvectors = np.linalg.eigh(hessian)
~~~

对应

\[
f(z)=\frac12\|Fz-y\|_2^2,\qquad
\nabla f(z)=F^\top(Fz-y),\qquad
\nabla^2f(z)=H=F^\top F.
\]

对 \(H\) 与 \(H^\top\) 取平均只清理数值不对称。
理论上的 \(F^\top F\) 本就对称半正定。
`eigenvalues` 按升序排列，所以下标 0、-1 分别是最小、最大特征值。

| 局部变量 | 形状 | 含义 |
|---|---|---|
| `features` / `target` | \((L,p)\) / \((L,)\) | 已完成所有建模与归一化的 \(F,y\) |
| `constraints` / `lower` | \((q,p)\) / \((q,)\) | 原缩放变量上的 \(C,d\) |
| `initial` | \((p,)\) | 缩放上三角初值 \(z_0\) |
| `hessian` | \((p,p)\) | \(H=F^\top F\) |
| `eigenvalues` / `eigenvectors` | \((p,)\) / \((p,p)\) | \(\lambda,V\)，特征向量按列存放 |
| `upper` | `None` 或 \((p,p)\) | Cholesky 上三角因子 \(U\)，\(H=U^\top U\) |
| `transform` | \((p,p)\) | \(W\)，从求解坐标恢复 \(z\) 的线性变换 |
| `shift` | \((p,)\) | 平移量 \(a\)，满足 \(z=W(u-a)\) |
| `start` | \((p,)\) | 求解坐标的初值 \(u_0\) |
| `roots` | \((p,)\) | 谱预条件中的 \(\sqrt{\max(\lambda_i,10^{-6}\lambda_{\max})}\) |
| `transformed_features` | \((L,p)\) | \(FW\)，用于非 Cholesky 分支的残差计算 |
| `transformed_constraints` | \((q,p)\) | \(CW\) |
| `transformed_lower` | \((q,)\) | \(d+CWa\) |
| `row_norms` | \((q,)\) | \(CW\) 每一行的二范数 |
| `result.x` | \((p,)\) | SLSQP 返回的 \(u\) |
| `values` | \((p,)\) | 恢复后的 \(z=W(u-a)\) |
| `violation` | 非负标量 | 在 \(z\) 坐标上检查的最大不等式违反 |

### 8.2 Cholesky 分支为什么变成球形目标

~~~python
transform = solve_triangular(upper, np.eye(len(initial)), lower=False)
shift = -transform.T @ (features.T @ target)
start = upper @ initial + shift

def objective(u):
    return 0.5 * float(u @ u), u
~~~

`solve_triangular(U,I)` 求解 \(UW=I\)，因此 `transform` 是 \(W=U^{-1}\)；
`lower=False` 表示 \(U\) 为上三角矩阵。

设 \(\ell=-F^\top y\)，则 `shift` 为 \(a=W^\top\ell\)，`start` 为 \(u_0=Uz_0+a\)。
将二次目标配方：

\[
f(z)=\frac12z^\top Hz+\ell^\top z+\frac12y^\top y
=\frac12u^\top u+\frac12y^\top y-\frac12a^\top a.
\]

最后两项不依赖于 \(u\)，所以只优化 \(\tfrac12u^\top u\) 即可，
其梯度就是 \(u\)。从几何上看，原来不同方向伸缩差异很大的二次曲面
在这些坐标中变成球形，有利于数值求解。

`objective` 返回二元组“目标值、梯度”，与 `minimize(..., jac=True)` 对应。
不能把这一分支的 `result.fun` 直接当作原矩阵问题的目标值，因为它还省略了常数。

### 8.3 谱分支为什么不引入新的正则

~~~python
roots = np.sqrt(np.maximum(eigenvalues, 1e-6 * eigenvalues[-1]))
transform = eigenvectors / roots
start = roots * (eigenvectors.T @ initial)
~~~

NumPy 广播将 `eigenvectors` 的第 \(i\) 列除以 `roots[i]`，所以

\[
W=V\operatorname{diag}(1/\text{roots}_i),\qquad
z=Wu,\qquad u=\operatorname{diag}(\text{roots}_i)V^\top z.
\]

所有 `roots` 均为正，因此变换保持可逆；
即使 \(H\) 有零特征值，仍保留这些方向作为可优化变量。
特征值下限限制了坐标变换的放大程度。

实际目标仍通过下面的残差计算：

~~~python
transformed_features = features @ transform

def objective(u):
    residual = transformed_features @ u - target
    return 0.5 * float(residual @ residual), transformed_features.T @ residual
~~~

其梯度为 \((FW)^\top(FWu-y)\)，真实 Hessian 为 \(W^\top HW\)，
并未用取下限后的特征值替换 \(H\)。
这是“仅做预处理”和“向目标增加 ridge”之间的具体区别。

若 \(H=0\)，使用 \(W=I,a=0\) 并走同一段残差计算。
诊断名称 `original_rank_deficient` 在当前实现里对应这个全零 Hessian 分支；
一般的非零秩亏 Hessian 走 `spectral_preconditioned`。

### 8.4 为什么约束右端也要平移

~~~python
transformed_constraints = constraints @ transform
transformed_lower = lower + transformed_constraints @ shift
row_norms = np.linalg.norm(transformed_constraints, axis=1)
~~~

代入 \(z=W(u-a)\)：

\[
Cz\ge d
\iff CW(u-a)\ge d
\iff CWu\ge d+CWa.
\]

因此只改变左端而不改变右端会求解另一个问题。

每一行再除以自己的正范数，不改变不等式：
\[
c_i^\top u\ge b_i
\iff (c_i/\|c_i\|_2)^\top u\ge b_i/\|c_i\|_2.
\]
这使不同约束行的数值尺度更接近。这里原约束行非零，坐标变换可逆，
因此在精确算术下行范数为正。

### 8.5 求解器选项、检查和诊断

`method="SLSQP"` 使用支持线性约束的数值优化方法；
`LinearConstraint(..., lower, np.inf)` 表示只有下界的不等式。
`maxiter` 是迭代预算，`ftol=1e-12` 是停止判据的精度参数，
不能解释为每个估计元素的绝对误差都不超过 \(10^{-12}\)。

~~~python
values = transform @ (result.x - shift)
if not result.success or not np.all(np.isfinite(values)):
    raise RuntimeError(f"constrained R/X least squares failed: {result.message}")
violation = max(0.0, float(np.max(lower - constraints @ values)))
if violation > 1e-8:
    raise RuntimeError(f"constrained R/X least squares is infeasible: {violation:.6g}")
~~~

第一个检查针对求解状态与非有限解；
第二个检查针对所有原缩放约束，包括非负性和有序性：

\[
v=\max\left(0,\max_i[d_i-(Cz)_i]\right).
\]

这里的误差是在 \(z\) 坐标、约束按行归一化之前计算的，
不是物理矩阵元素的统一绝对误差。
例如 \(z_h\) 的误差 \(\varepsilon\) 恢复到块 \(M\) 时对应 \(s_M\varepsilon\)。

| 诊断字段 | 含义 |
|---|---|
| `method` | 当前求解方式，`SLSQP_convex_QP` |
| `coordinates` | 三种数值坐标分支的名称 |
| `success` | SLSQP 成功标志；本函数对失败直接抛出异常 |
| `iterations` | 实际迭代次数 `result.nit` |
| `message` | 求解器返回的终止说明 |
| `maximum_scaled_constraint_violation_before_cleanup` | 清理舍入误差前的 \(v\) |

## 9. `make_feasible` 为什么这样写

~~~python
result = np.maximum(blocks, 0.0)
if margins is not None and result.shape[1] > 1:
    for block, margin in zip(result, margins):
        off = block.copy()
        np.fill_diagonal(off, -np.inf)
        np.fill_diagonal(block, np.maximum(np.diag(block), off.max(axis=1) + margin))
return result
~~~

`result` 是截断负值后的新数组，不会就地修改调用方的 `blocks`。
循环中的 `block` 是 `result` 中当前矩阵的视图，`margin` 是该矩阵的物理裕量。
`off` 是该矩阵的副本，临时把对角置为负无穷，使行最大值只考虑非对角元素。
`np.diag(block)` 取得当前对角元素，`np.fill_diagonal` 则写回新的对角。

对应公式：

\[
M^+_{ij}=\max(M_{ij},0),\qquad
\widetilde M_{ii}
=\max\left(M^+_{ii},\max_{j\ne i}M^+_{ij}+\delta_M\right),\qquad
\widetilde M_{ij}=M^+_{ij}\ (i\ne j).
\]

当 \(n=1\) 时不存在非对角元素，也没有对应的有序约束，因此只需非负截断。

例如输入
\[
M=\begin{bmatrix}0&1\\1&0\end{bmatrix},\qquad\delta_M=0
\]
会得到全 1 矩阵；这个结果可行，却未必使回归损失最小。
因此不能用 `make_feasible` 替代约束最小二乘求解。

## 10. 可以手算并运行的两终端例子

令
\[
R_{\rm raw}=\begin{bmatrix}0&1\\1&0\end{bmatrix},\qquad X_{\rm raw}=I_2,
\qquad A=\begin{bmatrix}I_4\\-I_4\end{bmatrix},\qquad
Y=A\begin{bmatrix}R_{\rm raw}\\X_{\rm raw}\end{bmatrix}.
\]

取 \(\alpha=0,T=I,\delta_R=\delta_X=0\)。
由于 \(A^\top A=2I_4\)，目标等于
\(\|R-R_{\rm raw}\|_F^2+\|X-I_2\|_F^2\)。

由两终端的交换对称性及严格凸性，最优 \(R\) 可写为
\(R=\bigl[\begin{smallmatrix}d&t\\t&d\end{smallmatrix}\bigr]\)。
对 \(R\) 的优化化为
\[
\min_{d\ge t\ge0}2d^2+2(t-1)^2.
\]
固定 \(t\ge0\) 时最优 \(d=t\)，进而得到 \(d=t=1/2\)。因此
\[
R^*=\begin{bmatrix}1/2&1/2\\1/2&1/2\end{bmatrix},\qquad X^*=I_2,\qquad f^*=1.
\]

在 `rnj_wzzt_core` 目录下使用 Topo Python 环境运行：

~~~python
import numpy as np
from rnj_wzzt.estimation.constrained_least_squares import solve_symmetric_least_squares

design = np.vstack([np.eye(4), -np.eye(4)])
raw_r = np.array([[0.0, 1.0], [1.0, 0.0]])
target = design @ np.vstack([raw_r, np.eye(2)])

blocks, diagnostics = solve_symmetric_least_squares(
    design=design,
    target=target,
    initial=np.stack([np.eye(2), np.eye(2)]),
    margins=np.zeros(2),
    alpha=0.0,
    max_iterations=500,
)
r_hat, x_hat = blocks
objective = 0.5 * np.sum((design @ np.vstack(blocks) - target) ** 2)

np.testing.assert_allclose(r_hat, np.full((2, 2), 0.5), atol=2e-6)
np.testing.assert_allclose(x_hat, np.eye(2), atol=2e-6)
np.testing.assert_allclose(objective, 1.0, atol=2e-6)
print("R =", r_hat, "\nX =", x_hat, "\nobjective =", objective)
~~~

这个例子也说明：约束最小二乘会同时权衡“抬高对角”与“降低非对角”造成的拟合损失；
简单可行性修正只抬高对角，通常得不到同一个解。

本次说明核验：16 段源码摘录逐行匹配当前实现，公式与代码块分隔符检查通过；
上述示例已实际运行，得到全 0.5 的 R、单位矩阵 X 和约 1.0000000000000007 的目标值。
本次仅扩充文档，求解器源码 SHA-256 保持不变，未重复运行上一节记录的 212 项测试。
