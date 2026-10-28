# RNJ+WZZT 当前最佳已测策略：跨对话交接说明

日期：2026-10-28。工作区：`D:\0-github_workspace\Topo`。

## 1. 直接采用哪个版本

以当前算例库的平均拓扑簇 F1 为目标，采用 **`multistart_original_rule`**。它使用完整普通 RNJ 家族、联合 L1 拟合和有限池交叉簇替换，增加最多两个初值，最后保持原来的验证选模及弱支路收缩规则。

已测平均 F1 为 **0.9252397562**，原组合基线 `baseline25` 为 **0.9194371947**。相对基线，80 项中提升 5 项、持平 72 项、退化 3 项。这里“最佳”指截至本说明日期、同一算例矩阵中平均 F1 最高的已测版本。

使用时必须指定完整版本名。`multistart75` 是包含另一套收缩后评分和重估规则的对照版本，平均 F1 为 0.9032532051；它与当前最佳版本不同。

| 用途 | 精确版本名 | 当前定位 |
|---|---|---|
| 本库平均 F1 最高的已测方案 | `multistart_original_rule` | 增强版本 |
| 完整 RNJ 组合基线 | `baseline25` | 始终保留的对照 |
| 相同初值放宽时间 | `long75` | 80 项最终 F1 均与基线相同 |
| 收缩后评分、重估及 AC 门控 | 其他对照版本 | 保存结果，暂不并入增强版本 |

本说明是策略、配置和证据的交接入口；`best_strategy.json` 是可解析的策略规格，现有命令行尚未实现直接读取该 JSON 的通用 `--config` 入口。

## 2. 固定输入与信息边界

每组输入包括末端有功 `P`、无功 `Q`、电压幅值 `V` 和观测根电压 `root`。训练、验证、测试分开；列顺序及末端标签必须一致。支持集内部用从 0 开始的列索引，物理节点号通过 `labels` 映射。当前算例根节点标签为 1，且根不属于末端标签集合。

定义每个时刻的平方电压降：

\[
y_t=v_{0,t}^{2}\mathbf 1-v_t\odot v_t.
\]

它是根与末端的平方电压差，拟合和验证均使用这个量。P/Q、V/root 保留原量测文件的基准及净功率符号，包含反送。外部数据应先统一标幺基准、符号和列对应关系。

当前主分支用 **raw**：原始 P/Q/y 直接入模，不拟合截距。候选池构建仍使用 raw 和每日去均值两个分支；不能因为最终分支是 raw，就删掉去均值分支贡献的候选簇。

模拟真簇仅用于外层算例调参和事后评价。初值、候选池、每步修正、验证排序及最终选模读取训练/验证观测。历史 0.16 和 3% 等设置已允许面向本算例库的结果调参；本轮增强对照未再按逐实例真簇调参。

## 3. 参数表

| 参数 | 固定值或规则 |
|---|---|
| 最终分支 | `raw` |
| 候选池预处理分支 | `raw`, `demean`；demean 按三个物理日分别去均值 |
| 候选几何 | `R`, `RX75`, `X` |
| NJ/RNJ 构造阈值因子 | `0, 0.04, 0.08, 0.16, 0.24` |
| 候选池树生成器 | NJ 和 RNJ；两预处理×三几何×五阈值×两方法，去重前 60 棵树 |
| 最终 RNJ 模型库 | raw 分支的 15 个完整普通 RNJ 家族，各自联合 L1 拟合 |
| 必留叶支持 | 所有单元素支持，即使权重为零也保留 |
| 响应系数界 | 每个支持的 `0 <= r <= 2`, `0 <= x <= 2` |
| 零权内部支持删除 | `max(r,x) <= 1e-9` 时删除非叶支持 |
| 固定族 LP | Gurobi；`Threads=1`, `Method=1`, `TimeLimit=15` 秒 |
| LP 数值容差 | `FeasibilityTol=1e-9`, `OptimalityTol=1e-9` |
| LP 接受状态 | 原生 `Status=2`，并核验物理目标与约束 |
| 主初值 | `raw_RX75_0.16` 的完整 `native_family` |
| 主初值修正 | 最多 25 秒 |
| 额外初值 | 最多 2 个不同支持家族，各最多 25 秒 |
| 声明的结构搜索上限 | 最多 75 秒，按初值分别计时 |
| 路径时间检查 | 在候选之间检查；单次 LP 可能使实际时间略超名义预算 |
| 每条路径步数上限 | `3 * n`，n 为末端数 |
| 修正动作 | 加一个候选，同时删除所有与它交叉的旧支持；随后全族联合重估 |
| 接受增益 | 训练 MAE 下降严格大于 `1e-10 + 0.001 * current_loss` |
| 候选同分处理 | 固定候选顺序，损失改善超过 `1e-12` 才替换当前最佳 |
| 最终验证规则 | `validation_simple_3pct`，具体见第 4.6 节 |
| 最终分辨率 | `0.16 * calibration["median_root_depth"]` |
| 收缩强度数值容差 | `1e-10` |
| 最终收缩后再次拟合 | 关闭 |
| AC 拓扑选模门控 | 关闭；AC 用于另列的物理评价 |
| `mip_rel_gap` | 此有限池版本不适用：固定族子问题为连续 LP，结构搜索枚举有限候选 |

前置稠密 R/X 使用冻结版本的 `fit_projected_sensitivity`：`constraint_mode="ordered"`、`alpha=0`、`diagonal_margin_ratio=1e-6`，QP 迭代上限 5000。约束包括矩阵对称、非负及对角次序；前置矩阵提供几何信息，树结构由后续 RNJ/WZZT 形成。

## 4. 精确执行顺序

### 4.1 用训练量测建立有限候选池

对 raw 和每日去均值两分支分别估计前置稠密矩阵 `R_dense, X_dense`。

从矩阵 M 构造距离 `d_ij = M_ii + M_jj - 2*M_ij`，用全部严格正距离的平均值作为通道尺度；无正距离时取 1，数值下限为 `1e-12`。得到 `mu_r, mu_x` 后，定义：

\[
S_R=R_{dense}/\mu_r,\quad
S_X=X_{dense}/\mu_x,\quad
S_{RX75}=0.75S_R+0.25S_X.
\]

RX75 是归一化几何的 75/25 加权，不给 WZZT 的 r/x 强加比例。

对三种几何及五档阈值分别生成 NJ 和 RNJ。RNJ 的分组容差为 `factor * max(median(diag(S)), 1e-12)`；NJ 调用既有 `infer_classical_nj` 保持其归一化、Q 准则、收缩和同分规则。把全部后代末端簇并入候选池，加入所有叶支持及全末端支持，去重并按 `(簇大小, 索引元组)` 排序。

保留 raw 分支的 15 个普通 RNJ 原始家族作为最终模型库来源；初始化使用完整家族，保留所有叶。初始化中没有硬冻结的错误正权簇；正权簇也可通过后续交叉替换删除。

### 4.2 每个固定支持家族联合拟合

对家族 F 中支持 s，令 z_s 是对应末端的 0/1 向量。响应矩阵及预测为：

\[
R=\sum_{s\in F}r_s z_s z_s^T,\qquad
X=\sum_{s\in F}x_s z_s z_s^T,\qquad
\widehat y_t=Rp_t+Xq_t.
\]

在固定 F 上最小化所有训练时刻、末端的平均绝对残差：

\[
C(F)=\min_{0\le r_s,x_s\le2}
\frac{1}{Tn}\sum_{t,i}|y_{t,i}-\widehat y_{t,i}|.
\]

这是连续 LP。L1 指绝对残差；实现没有额外的系数稀疏正则项。r/x 是响应原子系数，不能直接作为欧姆阻抗使用。求解后删除双通道近零的非叶支持，并重新核算实际物理残差。

该响应是线性平方电压近似；算例量测由 AC 潮流生成，因此残差同时包含量测误差与近似误差。

### 4.3 主初值修正路径

从 `raw_RX75_0.16` 的完整 `native_family` 启动，先联合拟合，再最多搜索 25 秒。

每步依次考察池中非叶且尚未存在的支持 c。保留与 c 相容的旧支持，删除与 c 交叉的旧支持，再加入 c 并联合拟合。两个支持相容指不交、包含或被包含；有交集且互不包含才是交叉。叶与任意候选相容，因此叶始终保留。

从本轮已评价候选中选择训练损失最低者。改善超过参数表中的阈值时接受并保存路径点，否则停止。每条路径保存初始点及全部接受点；部分扫描因时间到而结束时，状态为 `pool_wall_limit`。

### 4.4 挑选最多两个额外初值

这一阶段只用 raw 的 15 个 RNJ 固定族拟合点。

1. 用第 4.7 节的 0.16 分辨率对每个点进行一次弱支路收缩。
2. 在收缩后的支持族上，重新用训练数据做联合 L1 拟合。
3. 计算这个重估模型在独立验证日上的平方电压降 MAE。
4. 按 `(验证 MAE, 重估后支持数, 原 RNJ 模型库顺序)` 排序。
5. 先把主初值的原始 `native_family` 放入已用集合，再按完整支持族去重，最多取两个新家族。

**这里的收缩后重估用于准备额外初值。** 它与把所有最终候选改成收缩后重估版本的 `post_refit` 对照处在不同位置。额外初值会作为新增路径的起点参与最终候选比较。

### 4.5 额外初值各运行最多 25 秒

每个额外初值沿用相同候选池、训练损失、交叉替换动作及接受阈值。保留新增路径的起点和全部接受点。若去重后不足两个不同家族，则只运行实际存在的初值。

最终用于选模的点按以下顺序拼接：

1. 原 raw RNJ 模型库及原主初值 25 秒路径，保持父批次 `RNJ_WZZT` 列表顺序；
2. 第一个额外初值的完整路径；
3. 第二个额外初值的完整路径。

### 4.6 在执行最终输出收缩之前选模

对进入上述候选列表的每个模型，按其当时的支持和权重计算独立验证 MAE。设最低值为 m，保留：

`validation_mae <= 1.03 * m + 1e-12`。

在这些模型中按 `(当前支持数, validation_mae, 候选列表索引)` 取最小者。支持数在**最终输出收缩之前**计算，包含保留的叶及可能的全末端支持。

额外初值已经在第 4.4 节收缩重估过；路径后续可以重新加入支持。这里的“收缩前”准确含义是尚未执行下一节统一的最终输出收缩。

保持这个选模顺序是当前最佳版本的重要组成部分。对所有候选先做最终收缩、再按其新验证分数统一重选，属于平均 F1 较低的 `final_score` / `multistart75` 对照。

### 4.7 只对所选模型执行最终弱支路收缩

固定使用训练 raw 稠密矩阵给出的校准量：

`a = 0.75 / mu_r`，`b = 0.25 / mu_x`，

`median_root_depth = median(diag(a*R_dense + b*X_dense))`。

非叶支持的强度为 `w_s = a*r_s + b*x_s`。保留所有叶；非叶仅在

`w_s + 1e-10 >= 0.16 * median_root_depth`

时保留。保留支持的 r/x 数值不变，最终输出后不再联合拟合、不再重新验证选模。校准来自前置 raw 稠密矩阵，路径中不随 WZZT 权重重算。

## 5. 时间、终止条件与证据层次

每条路径以局部墙钟时间和 `3*n` 步约束结束。固定族 LP 单次最多 15 秒，接受原生最优状态；检查每个训练 LP 的约束、边界和实际目标。

- `pool_local_optimum`：本轮完整扫描候选池，没有满足实际增益阈值的加入/交叉替换。
- `pool_wall_limit`：候选扫描未完成即达到路径时间预算。
- `step_limit`：达到该路径的步数上限。

这些状态覆盖已声明的有限候选池和局部动作。固定族 LP 的数值最优、候选邻域局部停点、拓扑恢复分别记录。

25 秒增至单初值 75 秒时，80 项最终 F1 全部不变；79 项局部停点，1 项仍限时，为 `flynn16/ideal_ac/288`。该限时项最终选择树库模型，F1 已为 1。

增强版声明的最多 75 秒是原 25 秒路径加两个 25 秒路径。实际批次共享 LP 缓存，并复用既有原路径。前置回归、模型库、初值准备及 AC 后处理另耗时间；现有局部搜索计时不代表独立部署的总运行时间。

路径预算在开始评价候选之前检查，当前 LP 可继续至其自身终止；因此 25/75 秒是名义搜索预算。增强实验控制器另有每组 300 秒的硬限制，它覆盖该组全部对照的执行，属于实验进程保护参数。

## 6. 已验证结果与必须保留的退化

四个网络为仓库构建的文献启发合成网络：paper15、soumalas11、flynn16、pengwah18。80 组为四网络×五条件×四数据量。条件为 `ideal_ac`、`reference`、`high_voltage_noise`、`higher_load`、`low_xr`，无额外隐藏源。

N 是固定三个物理日内的训练采样总量：24/48/96/288，分别每日取 8/16/32/96 点。训练 replicate=0，验证 replicate=2，测试 replicate=1；验证和测试各有三个完整日、共 288 点。原 NPZ 形状为日×96×末端，读入时选点、拼接。

| 总训练点数 | 原组合 `baseline25` | 最佳已测 `multistart_original_rule` |
|---:|---:|---:|
| 24 | 0.8326923077 | 0.8519230769 |
| 48 | 0.9265277778 | 0.9055530417 |
| 96 | 0.9324175824 | 0.9490384615 |
| 288 | 0.9861111111 | 0.9944444444 |
| 全部 80 项平均 | 0.9194371947 | 0.9252397562 |

精确恢复由 63/80 增至 65/80。R/X 平均相对 Frobenius 误差分别从 0.137698/0.399180 变为 0.120588/0.394586。独立测试 AC 相对电压 RMSE 从 0.0004696513 变为 0.0004672213。

相对矩阵误差以比例计，0.1 表示 10%；相对电压 RMSE=0.001 表示 0.1%。

增强版最终来源为树库 50 项、第一条额外路径 24 项、第二条额外路径 5 项、原修正路径 1 项。路径来源可能是路径起点；实际结构改变和新真簇应查完整支持差分及逐步记录。

三项退化均在 `high_voltage_noise`、48 点：

| 网络 | 基线 F1 | 增强版 F1 |
|---|---:|---:|
| paper15 | 0.8333333333 | 0.7692307692 |
| flynn16 | 0.5000000000 | 0.1666666667 |
| pengwah18 | 0.3750000000 | 0.3529411765 |

高电压噪声条件的 P/Q 相对噪声为 0.005，末端和根电压相对噪声为 0.001。其余具体条件见记录库的 `condition_json`。这些是当前固定输入库上的结果；尚未新增随机重复或未知网络的独立验证。

拓扑 F1 计分只包含大小在 2 到 n−1 之间的末端簇；叶和全末端根支撑不计入 F1，但可保留在模型中。最终收缩后模型用于结构和 R/X 评价。

AC 另作物理评价：只在训练日对所选拓扑共享阻抗作最多 12 次函数评价的局部重估，随后冻结参数评价测试日。AC 低误差和收敛状态不替代拓扑 F1。本轮 AC 验证门控平均 F1 为 0.9173538614，未并入最佳策略。

## 7. 其他对话如何直接调用已有结果

仓库根目录运行，Python 使用 `D:\apps\miniconda3\envs\Topo\python.exe`。

```python
from research_experiments.wzzt_budget.records import find, load, load_trace

catalog = r"D:\0-github_workspace\Topo\docs\wzzt_rnj_budget_20261028\catalog.sqlite"

# 查询某场景全部版本，也可加 variant 精确筛选。
rows = find(catalog, case="paper15", condition="high_voltage_noise", N=24)

m = load(
    catalog, case="paper15", condition="high_voltage_noise", N=24,
    variant="multistart_original_rule", with_observations=True,
)
R, X = m["R"], m["X"]
supports_by_node_label = m["support_labels"]
P, Q, V, root = m["train"]
P_val, Q_val, V_val, root_val = m["validation"]
P_test, Q_test, V_test, root_test = m["test"]
metrics_and_provenance = m["record"]

# 对照必须显式另读，避免按各实例真簇从版本中择优拼接。
baseline = load(
    catalog, case="paper15", condition="high_voltage_noise", N=24,
    variant="baseline25",
)

trace = load_trace(catalog, case="paper15", condition="high_voltage_noise", N=24)
steps = trace["searches"].get("multistart_0", {}).get("decisions", [])
```

`load(..., with_observations=True)` 校验原始量测文件哈希；`load_trace` 校验完整结果文件哈希。`load` 默认只加载模型与记录，R/X 会由支持和 r/x 重建。

记录库中有 80 组输入、七版本共 560 条模型。完整量测及约 80 MB 的原始逐候选结果留在运行输出目录，目录及哈希由记录库绑定；跨机器复制时需要同时迁移这些文件并处理绝对路径。

### 字段解释

| 记录 | 内容及使用方式 |
|---|---|
| `labels`, `supports`, `support_labels` | 列标签、列索引支持、映射后的物理标签支持 |
| `r`, `x`, `R`, `X` | 所选最终模型的响应系数与响应矩阵 |
| `record["selected_origin"]` | 树库、原路径或额外路径来源 |
| `record["search_stop"]` | 相关搜索状态，不能由它推断拓扑正确性 |
| `record["validation_mae"]` | **最终输出收缩后**重新测得的实际误差，供报告使用 |
| 原父模型库和未最终收缩路径点的 `validation_mae` | **选模时使用**的误差，检查排序应读取这些值 |
| `record["diagnostic_json"]` | 明确含事后真簇的假簇、遗漏及交叉阻挡标签 |
| `input_stats_json`, `condition_json` | 压降/PQ 波动摘要及噪声、负荷、X/R 场景设置 |
| `trace["searches"][...]["decisions"]` | 每步当前损失、完整扫描标记、逐候选新增/删除支持及目标值 |
| `trace["searches"][...]["path"]` | 含起点的全部接受模型，带支持和系数 |

当前最佳版本是后续分析器从已存路径派生的，因此原 `result.json` 的 `models` 字典只有五个初始对照版本。最佳模型从 catalog 或 `selected_models.json` 读取；需要重新派生时调用下一节的函数。

## 8. 实现入口与现有可用程度

| 环节 | 代码入口 |
|---|---|
| 候选池及前置矩阵 | `research_experiments/wzzt_retune/engine.py::build_bank` |
| 固定族 LP | 同文件 `PoolLP.fit` |
| 有限池交叉替换 | 同文件 `search` |
| 原选模及最终收缩 | `research_experiments/wzzt_retune/selection.py::select` |
| 额外初值准备及轨迹生成 | `research_experiments/wzzt_budget/worker.py::run` |
| **最佳版本的精确组装** | `research_experiments/wzzt_budget/variants.py::multistart_original_rule` |
| 记录读取 | `research_experiments/wzzt_budget/records.py::find/load/load_trace` |
| 统一评分与数据库 | `research_experiments/wzzt_budget/evaluate.py::evaluate` |
| 独立核验 | `research_experiments/wzzt_budget/audit.py`, `verify.py` |

从已有轨迹重新派生最佳模型：

```python
from research_experiments.wzzt_systematic.common import read, observed
from research_experiments.wzzt_budget.variants import multistart_original_rule

job = "paper15__high_voltage_noise__N24"
folder = rf"D:\0-github_workspace\Topo\outputs\wzzt_rnj_budget_20261028_full\jobs\{job}"
payload = read(folder + r"\payload.json")
result = read(folder + r"\result.json")
parent = read(payload["parent_result"])
point = multistart_original_rule(
    result, parent, observed(payload["train"], payload["N"]),
    observed(payload["validation"]),
)
```

当前已实现、已验证的是保存轨迹上的准确组装、评价和读取。`worker.run` 是依赖父批次结果的对照实验入口，同时计算若干消融；它不是只运行最佳策略、直接接收任意新 P/Q/V/root 的独立部署函数。处理新网络时，按第 4 节顺序组合现有函数，并单独验证新适配器，原基线继续保留。

## 9. 文件、环境与复现

| 用途 | 本地路径 |
|---|---|
| 本交接说明 | `docs/wzzt_rnj_best_strategy_20261028/README.md` |
| 可解析策略规格 | 同目录 `best_strategy.json` |
| 可复制的新对话开场文本 | 同目录 `handoff_prompt.md` |
| 最佳模型记录库 | `docs/wzzt_rnj_budget_20261028/catalog.sqlite` |
| 逐模型/变化/步骤索引 | 同目录 `per_job.csv`, `changed_models.csv`, `decision_summary.csv` |
| 最佳模型详细记录 | 同目录 `selected_models.json` |
| 最后一轮策略对照报告 | 同目录 `README.md` |
| 原组合调优和同源消融 | `docs/wzzt_retune_20261028/README.md`, `same_source_ablation.csv` |
| 完整增强实验输出和冻结源码 | `outputs/wzzt_rnj_budget_20261028_full/` |
| 父模型库和原 25 秒路径 | `outputs/wzzt_retune_20261028_round2/` |
| 原训练/测试数据 | `outputs/wzzt_systematic_20261027_round1/data/` |
| 历史好结果机制分析 | `docs/wzzt_retune_20261028/early_results_audit.md` |

上表相对路径均以 `D:\0-github_workspace\Topo` 为根。计算实现和结果的提交为 `02b705a4`、`a5f8f1ec`，任务分支为 `codex/source-ac96-benchmark-20261018`。现有工作区还含其他 AC 开发改动；历史运行以输出目录的 `source/` 或 `producer_source.zip` 为准。分析器代码闭包另存为 `analysis_source.zip`。

仅检查已有结果，使用：

```powershell
Set-Location D:/0-github_workspace/Topo
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_budget.records docs/wzzt_rnj_budget_20261028/catalog.sqlite --case paper15 --condition high_voltage_noise --N 24 --variant multistart_original_rule
```

核验已有交付内容的命令如下；它检查哈希、记录库及既有结果，不重新运行结构搜索：

```powershell
& D:/apps/miniconda3/envs/Topo/python.exe -B -m research_experiments.wzzt_budget.verify outputs/wzzt_rnj_budget_20261028_full docs/wzzt_rnj_budget_20261028
```

只有在新对话明确要重跑实验时，才使用原报告中的 `prepare -> run -> evaluate -> audit -> report -> verify` 流程，并指定新的输出目录。`prepare` 固定依赖现有父批次路径，任意新网络或新数据量需要独立适配。

原交付已经核验：80/80 输入、560/560 所选模型、2269 个不同物理模型、20235 条原生 LP 记录、15 个独立 HiGHS LP、791 步搜索；4 项单元检查通过，正式批次无失败或超时。

## 10. 给后续对话的使用约定

优先复用记录库；固定 `multistart_original_rule` 为本库最佳已测版本、`baseline25` 为对照。新实验在相同观测、数据分割、候选域和计分口径上比较，保留失败分母、每组退化、选模来源和停止原因。新的重复或网络沿用现有参数，另列重新调参的结果。数据、算法、运行和全树最优性四个层次分别表述。
