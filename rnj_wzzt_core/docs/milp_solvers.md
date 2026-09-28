# 层状 L1-MILP：通用矩阵版与 gurobipy 版

只维护一套模型、两个求解入口：

| 入口 | 职责 |
|---|---|
| `laminar_l1_milp.py` | 数据与验证、通用矩阵模型、求解与前向路径；默认 SciPy/HiGHS |
| `gurobi_milp.py` | 用 gurobipy 求解同一标准矩阵模型 |

数据类型、验证规则和约束构造已合并到 `laminar_l1_milp.py`，按数据、校验、
矩阵与评价、模型构造、求解、前向路径分段。Gurobi 适配独立保留，MILP 运行代码
共两个文件，公开的 `solver="highs"` / `solver="gurobi"` 调用方式不变。

## Python 是否有统一接口

Python 没有内置的通用优化求解器协议。Pyomo 是统一建模层：用同一套变量、
目标和约束描述模型，再选择已安装的求解器。它不替代求解器本身，也不提供
Gurobi 许可证。可参阅 [Pyomo 的求解器说明](https://pyomo.readthedocs.io/en/stable/getting_started/solvers.html)。

本项目已经有独立的矩阵建模层，因此直接增加 Gurobi 适配器：

```text
laminar_l1_milp.py：构造同一套 c、A、变量界、约束界、整数标记
                          ↓
laminar_l1_milp.py：_run_milp(..., solver=...)
                          ↓
          highs                         gurobi
 scipy.optimize.milp             gurobi_milp.py → gurobipy
                          ↓
                统一诊断与原有路径选择
```

`solver="highs"` 保留默认行为。`solver="gurobi"` 同时切换新支撑搜索的 MILP
和固定支撑重拟合的 LP。RNJ 前的约束平方损失回归仍使用 SLSQP；
`--milp-solver` 只控制后续层状 L1 模型。

## 数学模型

每个场景 \(s\) 有 \(P_s,Q_s,Y_s\in\mathbb R^{T_s\times n}\)，每行是一个时刻。
\(Y_s\) 是已观测根节点相对末端的平方电压降：
\(Y_{s,ti}=V_{0,s}(t)^2-V_{i,s}(t)^2\)。同一时刻所有末端共用同一根观测。
主流程直接使用原始 P/Q/Y，不去均值，也不估计节点偏置。模型预测为

\[
\widehat Y_s=P_sR+Q_sX,
\qquad
R=\sum_{k=1}^K r_k z_kz_k^\top,\quad
X=\sum_{k=1}^K x_k z_kz_k^\top.
\]

\(z_k\in\{0,1\}^n\) 表示原子的终端支撑，\(r_k,x_k\ge0\) 是跨场景共享的权重；
这里 R/X 是平方电压降灵敏度，系数 2 已包含在权重中。观测噪声及 AC 线性化误差留在残差中。
固定所有 \(z_k\) 后拟合是 LP；每次扩展把一个新的 \(z\) 也作为变量，得到 MILP。
层状约束要求任意两个支撑不相交，或其中一个包含另一个。

引入逐元素非负残差上界 \(E_s\)，目标与拟合约束为

\[
\min\ \frac1N\sum_{s,t,i} E_{s,ti},\qquad
-E_s\le Y_s-\widehat Y_s\le E_s,\qquad
N=n\sum_s T_s.
\]

这就是平均绝对误差。残差上界、权重、支撑二进制变量和乘积线性化变量
共同组成一个长向量 \(v\)。模型构造器最后输出统一标准形：

\[
\boxed{\min_v c^\top v\quad
\text{s.t.}\quad \ell_A\le Av\le u_A,\quad
\ell_v\le v\le u_v,\quad v_j\in\mathbb Z\ (j\in\mathcal I).}
\]

生产模型的整数变量均限制在 \([0,1]\)。Gurobi 接收同一个标准形，
不重新推导或改变残差、层状相容以及权重上界。

## 接口和变量

三个公共入口均新增关键字参数 `solver="highs"` 或 `solver="gurobi"`：

- `solve_fixed_support_l1`：固定支撑，优化所有 R/X 权重。
- `solve_best_laminar_extension_l1`：搜索一个新支撑，同时重拟合旧权重。
- `fit_laminar_l1_sensitivity`：执行完整前向路径、界扩展、零权重清理与验证集选择。

`scenarios` 是字典序列，每个字典包含 `P_terminal`、`Q_terminal`、`drop_target`，
三者都是同形状 `(T_s, n)` 的二维数组或 DataFrame。DataFrame 按 P 的时间索引和
首个场景的终端标签对齐；重复、缺失或多余标签会报错。数组按位置对应。
`drop_target` 应事先用实际根电压观测构造；拟合器不再另外估计基准电压。

可以直接运行下面的小例子，不依赖 Pandas 或历史输出：

```python
import numpy as np
from rnj_wzzt.estimation.laminar_l1_milp import fit_laminar_l1_sensitivity

rng = np.random.default_rng(7)
P = rng.normal(size=(24, 2))
Q = rng.normal(size=(24, 2))
R = np.array([[0.5, 0.3], [0.3, 0.6]])
X = np.array([[0.3, 0.2], [0.2, 0.4]])
Y = P @ R + Q @ X
scenarios = [{"name": "example", "P_terminal": P,
              "Q_terminal": Q, "drop_target": Y}]

result = fit_laminar_l1_sensitivity(
    scenarios,
    initial_supports=[(0,), (1,)],
    r_upper_bound=2.0,
    x_upper_bound=2.0,
    solver="gurobi",
)
print(result.train_mae)
print(result.r_matrix)
print(result.path[-1].solver.to_dict())
```

`initial_supports` 是必须保留的结构，使用从 0 开始的终端位置。
主流程在原终端数据上，将全部叶边 singleton 和筛选后的 RNJ 支撑放入初始族。
不聚合 P/Q 或构造伪末端；固定的是支撑向量，已有与新增原子的 R/X 系数仍联合拟合。

设当前支撑族为 \(\mathcal F\)，本次新增支撑的搜索域直接写成

\[
z\in\{0,1\}^n,\qquad S=\{i:z_i=1\}\ne\varnothing,\qquad S\notin\mathcal F,
\qquad \forall C\in\mathcal F:\ S\cap C=\varnothing\ \text{或}\ S\subseteq C\ \text{或}\ C\subseteq S.
\]

无需预先枚举候选支撑。核心已删除白名单参数、整理函数及选择器变量/约束。
旧的 `allowed_supports`、`candidate_supports` 参数会被明确拒绝。
`normalize_supports` 仅校验已经给定的固定支撑族，不能代替上述 MILP 约束。

每次扩展 MILP 已联合优化已有与新增权重，正常步骤直接使用其输出参数，按实际
矩阵残差复核目标值，不再无条件重解相同固定族 LP。初始化和删去近零原子后的
条件重拟合仍使用 LP；改变权重界后仍重新开始前向路径。
提取系数时，仅将绝对 `1e-8` 容差内的边界浮点误差夹回合法区间；明显越界直接报错。

`evaluate_l1_matrices` 直接计算 `abs(Y - P @ R.T - Q @ X.T)`，不在评价数据上
校准任何参数。它返回总 MAE、时间块均值的标准误估计及块均值；后两者用于
一标准误差选模及诊断。训练和验证无需具有相同的场景名称或数量，只需终端一致。
旧 `fixed_intercepts` 参数及结果中的 `intercepts` 字段已删除，不提供自由偏置兼容模式。

`_run_milp` 将构造器转换为标准矩阵输入，再交给任一后端。
Gurobi 适配器使用与 SciPy 调用一致的关键字接口：
`solve_gurobi_milp(*, c, integrality, bounds, constraints, options)`。
这消除了两个后端各自解读构造器、转换数组和统计时间的重复逻辑。
设变量数为 \(p\)、矩阵行数为 \(m\)：

| 代码 | 数学含义 | 格式 |
|---|---|---|
| `c` | 目标系数 \(c\) | 长度 \(p\) 的浮点数组 |
| `bounds.lb / ub` | \(\ell_v,u_v\) | `Bounds` 中长度 \(p\) 的数组，可含无穷界 |
| `integrality` | 整数集合 \(\mathcal I\) | 长度 \(p\) 的数组，0 为连续、1 为整数 |
| `constraints.A` | \(A\) | `LinearConstraint` 中的 \(m\times p\) 稀疏矩阵 |
| `constraints.lb / ub` | \(\ell_A,u_A\) | 长度 \(m\) 的数组 |
| `options["time_limit"]` | 每次调用的求解时间上限 | 秒；无限制时省略该键 |
| `options["mip_rel_gap"]` | 请求的 MILP 相对间隙 | 非负浮点数，默认 0 |
| `options["mip_abs_gap"]` | 请求的 MILP 绝对间隙 | 当前固定为 `1e-10` |
| `options["presolve"]` | 是否启用预处理 | 布尔值 |
| `options["disp"]` | 是否输出 Gurobi 日志 | 布尔值 |

适配器依次完成以下操作：

1. 按需导入 `gurobipy`。未安装或许可证不可用时明确报错，不自动改用别的求解器。
2. 用 `addMVar` 建立 \(v\)，传入目标系数、变量界和变量类型。\([0,1]\) 整数标记为二进制。
3. 等式行建立一次 \(A_i v=\ell_{A,i}\)；其他行分别建立有限下界和有限上界。
   两端都无限的行不添加约束。双边行拆成两行，因此 Gurobi 内部约束数可能大于原始 \(m\)。
4. 调用 `optimize()`；仅在 `SolCount>0` 时读取解和目标。存在有效 MIP 下界时保留下界，
   即便当前没有可行解。
5. 转为与 SciPy 兼容的 `OptimizeResult`，再由原来的路径控制代码检查最优性证据。

Gurobi 代码使用[官方矩阵接口](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.addMConstr)。
`constraint_count` 记录共享模型的原始行数，便于跨后端比较。

## 状态与数值设置

两个求解器原始状态编号不同。例如 Gurobi 的 `2` 表示最优，而 SciPy 的 `2`
表示不可行，必须转换。`SolverDiagnostics.status` 始终使用统一编号，
`raw_status` 单独保存 Gurobi 原始编号，`solver` 记录实际后端。

| 统一 status | 含义 | Gurobi 示例 |
|---|---|---|
| 0 | 求解器报告最优 | `OPTIMAL=2` |
| 1 | 达到限制或中断 | `TIME_LIMIT=9` |
| 2 | 已证不可行 | `INFEASIBLE=3` |
| 3 | 无界 | `UNBOUNDED=5` |
| 4 | 其他或无法明确分类 | `INF_OR_UNBD=4`、`NUMERIC=12`、`SUBOPTIMAL=13` |

编号来源：[Gurobi 状态文档](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html)。
保留 HiGHS 的历史诊断格式；它的 `raw_status` 留空。

新后端设置单线程、可行性/整数可行性/LP 最优性容差均为 `1e-9`，
`MIPGapAbs=1e-10`，`MIPGap=mip_rel_gap`；启用预处理时使用 Gurobi 自动档。
关闭对偶约简以明确区分不可行和无界。环境和模型都通过上下文管理器释放。
`runtime_seconds` 包含适配器内的建模、环境启动和求解开销；`TimeLimit` 是求解器的
单次优化预算，不是整条前向路径的总时间限制。

MILP 仍需通过原来的严格证书检查：统一状态为 0、成功标记为真，并且绝对
原始/对偶间隙不大于 `1e-10`，或报告的相对间隙不大于 `1e-10`。
因此把请求的 `mip_rel_gap` 调大，不会自动放宽“精确扩展”的接受条件。
超时的可行解只保留目标和界作诊断，不输出为已证最优的新支撑。

这个证书只覆盖当前固定支撑族与权重界下的一次扩展；整个前向路径仍是贪心算法。

## 安装与运行

在 `rnj_wzzt_core` 目录下：

```powershell
python -m pip install -e ".[gurobi]"
python run.py --milp-solver gurobi --cases paper15 --output outputs/gurobi_paper15
```

安装 `gurobipy` 后仍需可用许可证。本次开发机器的 `poweropt` 环境已具备
Gurobi 13.0.1 和许可证，可从仓库根目录直接运行：

```powershell
& 'D:\apps\miniconda3\envs\poweropt\python.exe' rnj_wzzt_core/run.py --milp-solver gurobi --cases paper15 --output outputs/gurobi_paper15
```

主流程 Python 接口对应 `pipeline.run(..., milp_solver="gurobi")`。
`metrics.json` 的配置和 `milp_results.csv` 均记录 `milp_solver`；
CSV 中 `attempt_statuses` 为统一状态，`attempt_raw_statuses` 为原始状态。

## 验证方式

`tests/test_milp_solvers.py` 用独立的正负残差 LP 和所有合法支撑枚举作小规模参照，
另与 HiGHS 比较目标；退化问题比较损失和可行预测，不强求非唯一权重逐项相同。
还覆盖整数域、一般自由变量、单边/双边/等式行、不可行、无界、实际超时、完整路径、
支撑族已满、旧白名单参数拒绝、未安装可选包，以及非最优可行解不能成为精确扩展。

```powershell
python -m pytest -q tests/test_milp_solvers.py tests/test_independent_milp_oracle.py
```

缺少 `gurobipy` 时，实际 Gurobi 求解测试显示为跳过。装有包但许可证失效时测试失败，
不会把许可证故障伪装成通过或跳过。验证结果与实验设置保存在仓库的
`artifacts/gurobi_backend_20260927/`。
