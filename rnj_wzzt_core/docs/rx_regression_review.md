# RNJ 前多场景 R/X 回归：审查与修正

日期：2026-09-07。范围：rnj_wzzt_core 内 RNJ 前的平方损失回归及矩阵约束辅助函数。

## 结论

默认 ordered 路径已从“无约束最小二乘 + 反复可行性修正 + 手写梯度迭代”
改为“按场景中心化 + 固定约束凸二次规划”。继续使用 NumPy/SciPy，
不要求安装 Gurobi；Gurobi 仅用于本次独立数值对照。

R、X 仍是两个分别估计、共同参与电压回归的矩阵，损失仍是平方残差。
RNJ 后的 laminar L1-MILP 没有修改。

## 审查发现

1. **多个约束的循环修正不是交集最近点投影。**
   原 ordered 修正会受遍历顺序影响；继续做原梯度迭代也可能停在非最优可行点。
   对两终端解析例，正确最优 SSE 是 2，原 50/500 次迭代都得到约 2.1267217639。
2. **原单半空间投影的注释错误。**
   对称 Frobenius 距离中，同一个非对角值出现两次。
   最小化 a²+2b²、满足 a-b=v，解恰为 a=2v/3、b=-v/3。
   原代码的 2/3 与 1/3 更新是正确的；之前把它解释为过度修正的说法有误。
   已改正文档，保留该更新。
3. **原外层每次修正都重新计算裕量。**
   这会改变迭代所用约束域。现在从最初无约束估计确定 R、X 各自的裕量，并固定使用。
4. **场景截距列使原步长失衡。**
   P/Q 使用较小的标幺值，而场景指示列是 0/1；原步长被后者主导。
   中心化严格消去截距，避免该问题，也去掉反复更新截距的步骤。
5. **原拟合直接丢弃行索引。**
   时间标签相同但行顺序不同可能产生错误匹配。
   现在先按各场景 P 的时间索引、第一场景 P 的终端列顺序对齐。
   缺失或多余标签、重复索引、空数据、非有限数据和非法参数明确报错。
6. **矩阵约束辅助函数的近零判断错误。**
   极小正对角的中位数可能导致明显非零的非对角元被全部清零。
   现在按整个矩阵的绝对大小判断近零，保留常规尺度计算方式。

## 当前调用链

~~~text
pipeline._rx75_rnj_clades
  -> fit_projected_sensitivity
       1. 校验并对齐 P/Q/drop_target
       2. 每个场景分别中心化
       3. 无约束 LS（alpha>0 时为 ridge）确定初始化与固定裕量
       4. solve_symmetric_least_squares
            上三角变量编码 R/X 对称性
            非负界 + 固定的对角有序线性约束
            数值条件允许时做 Cholesky 坐标变换
            SciPy SLSQP 求解并检查状态与可行性
       5. 恢复场景截距、计算拟合指标
  -> sensitivity_geometry(..., "RX_75R_25X")
  -> rooted_neighbor_joining
~~~

原函数名和四个返回值保留，父项目兼容转发器继续指向同一份实现。
新增 constrained_least_squares.py 仅负责 QP 编码与数值求解，
multiscenario.py 负责场景数据、中心化、接口及诊断。

## 数学模型与等价简化

场景 s 的 P_s、Q_s、Y_s 均为 T_s × n 矩阵，R、X 为 n × n；
Y_s 是根节点与终端的平方电压差，主线另有去均值预处理。

原回归目标为

    min  1/2 Σ_s ||P_s Rᵀ + Q_s Xᵀ + 1 b_sᵀ - Y_s||_F²
         + alpha/2 (||R||_F² + ||X||_F²).

截距不参与正则。对每个场景消去最优截距：

    b_sᵀ = mean(Y_s) - mean(P_s) Rᵀ - mean(Q_s) Xᵀ.

因此，只需把 P_s、Q_s、Y_s 各自在场景内减去均值，求解

    min  1/2 Σ_s ||P_s^c Rᵀ + Q_s^c Xᵀ - Y_s^c||_F²
         + alpha/2 (||R||_F² + ||X||_F²).

当前主线 alpha=0。该中心化是原目标的等价消元，不是更换观测模型。

默认 ordered 对 R、X 各自施加：

    M = Mᵀ;
    M_ij >= 0;
    M_ii - M_ij >= epsilon_M, i != j.

epsilon_M = diagonal_margin_ratio × 初始矩阵尺度；
默认比例 1e-6，尺度取非负对称无约束估计的正对角中位数，
不存在正对角时取最大元素。裕量在整次拟合内不变。

## 数值求解与边界

上三角变量 z 使对称性天然成立；把回归写成

    min  1/2 zᵀ H z + lᵀ z + 常数,
    subject to z >= 0, C z >= d.

H 是半正定矩阵。若 alpha=0 且设计秩亏，不保证参数解唯一。

当 H 数值正定且最小/最大特征值之比大于 1e-12 时：

    H = Uᵀ U, W = U⁻¹, a = Wᵀ l,
    u = U z + a, z = W(u-a).

目标变为 1/2 ||u||² 加常数；线性约束一并变换，并按行范数缩放。
这只改变坐标，没有增加 ridge，也没有改变 R、X 的相对权重。
普通 SLSQP 在本次部分病态算例上迭代 1000 次仍未收敛，
而等价变换后四个算例均在 2 次迭代成功。

数值近奇异或秩亏时改用谱预条件。对 H=V diag(lambda) Vᵀ，构造
W=V diag(1/sqrt(max(lambda, 1e-6*lambda_max)))，将原变量写成 z=Wv。
特征值下限仅限制坐标变换的幅度；目标和梯度继续通过原始残差的变换计算，
没有用截断后的特征值替代原 Hessian，因此没有添加正则项或删除弱方向。
H 全零时保留原坐标。不能仅凭 Cholesky 是否报错判断数值秩。

变量尺度根据观测数据能量确定，避免高度共线时巨大的无约束系数破坏数值尺度；
从可行的 LS 初始化和最小对角边界点中选择真实目标较小者作为初始点。
求解器失败、非有限解、缩放后的约束违反大于 1e-8 时明确抛出异常。
仅在可行性检查通过后清理边界舍入误差。
这些是数值检查，不能表述为任意输入上的形式化最优性证书。

兼容参数与模式：

- constraint_refine_iterations 默认从 50 调为 500，含义为数值求解器迭代预算；
  不是固定执行 500 步。设为 0 明确表示仅返回可行初始化。
- basic 使用同一 QP，但不加对角有序约束。
- tree_covariance 保留单独的 PSD 可行性启发式兼容路径，
  不声称求得最优 SDP 解；当前 RNJ 主线不使用它。
- R² 与第四返回值（含场景指示列的原始设计条件数）保持原语义。
- 可传入 diagnostics={} 获取方法、坐标方式、迭代数、可行性违反、
  固定裕量、SSE、完整目标值、中心化设计条件数和截距。

ordered 不强制 PSD，不保证矩阵来自某棵树，
也不保证 R、X 对应相同的离散树拓扑。

## 本次四算例对照

使用原主线训练配置：三个场景，每场景 96 点，replicate=0，
P/Q 噪声 0.005、电压噪声 0.0002，RNJ 容差比例 0.16。

| 算例 | 修正前 SSE | 修正后 SSE | 修正前 RNJ F1 | 修正后 RNJ F1 |
|---|---:|---:|---:|---:|
| paper15 | 0.03917204 | 0.0007990152 | 1 | 1 |
| soumalas11 | 0.007587289 | 0.0005122692 | 1 | 1 |
| flynn16 | 0.05285358 | 0.0007877577 | 1 | 1 |
| pengwah18 | 0.05583255 | 0.0008607979 | 0.933333 | 1 |

四个实际新实现的 R/X 均已与独立 Gurobi QP 对照。
该配置下，pengwah18 去掉了原结果多出的终端集合 {311,312}。
这里是固定种子上的观测结果，不构成其他噪声/样本条件下的拓扑恢复保证；
本次没有重跑后续 L1-MILP。

## 验证与复现

数值测试覆盖解析活跃约束解、固定裕量、标签对齐、场景截距、
ridge、排列等变性、PSD边界、秩亏、单位缩放和失败处理。
最终验证记录：

- 核心目录 75 项测试、父项目兼容测试 123 项，共 198 项全部通过。
- 每算例完整样本拟合 1 次、循环块 bootstrap 拟合 100 次；共 404 次成功。
  四算例所有这些拟合均为 Cholesky 白化路径，SLSQP 2 次迭代。
- 最终四算例 R/X 与独立 Gurobi 参考的相对矩阵误差最大约 9.84e-10。
- 测试报告保存为 pytest_results.xml；实际拟合与 bootstrap JSON 记录源码 SHA-256。
- 数据、代码快照和前后矩阵均保留；差异文件为 source_changes.diff。


Topo 环境执行：

~~~powershell
conda activate Topo
python -B -m pytest -q -p no:cacheprovider rnj_wzzt_core/tests tests
~~~

独立 Gurobi 参考对照使用已有 poweropt 环境：

~~~powershell
& D:\apps\miniconda3\envs\poweropt\python.exe artifacts\rx_regression_review_20260907\qp_reference.py --whiten
~~~

证据目录：../../artifacts/rx_regression_review_20260907/

- before_multiscenario.py、before_matrix_constraints.py 与 SHA-256：修改前快照。
- before_benchmark.json、*_before.npz：修正前矩阵与结果。
- qp_reference.json：普通 SLSQP 与 Gurobi 对照及失败证据。
- qp_whiten_reference.json：等价坐标变换后的独立对照。
- actual_after_benchmark.json、*_actual_after.npz：实际新实现的结果与诊断。


- bootstrap100_benchmark.json：404 次实际求解诊断、边界块与重采样频率。
- actual_after_benchmark.py：在 Topo 环境重现实际代码和 bootstrap 验证。
- actual_after_benchmark.log：最后一次完整执行日志。
- pytest_results.xml、validation_summary.json：最终测试与源码版本摘要。
