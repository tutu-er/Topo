# RX75-RNJ 与 L1-wzzT 独立核心

本目录是当前算法的唯一维护位置，可以单独复制、安装和运行，不导入父项目源码。
默认流程为：多场景约束最小二乘估计 R、X → RX75 共享路径分数 → RNJ 变体 →
循环分块 bootstrap → 可信边界子树收缩 → 有限候选层状 L1-MILP → 独立验证集选模型。

## 运行

在本目录执行：

```powershell
conda activate Topo
python -m pip install -e .
python run.py
```

也可以继续使用 `python -m pip install -r requirements.txt`，该文件引用同一项目依赖声明。
运行依赖仅为 NumPy、Pandas、SciPy、NetworkX；无需 Gurobi。

```powershell
python run.py --scenario-suite reference --cases paper15 --output outputs/reference_paper15 --time-limit 30
python run.py --scenario-suite legacy --output outputs/legacy_replay
python experiments/run_scenario_benchmark.py --repeats 3 --output outputs/scenario_benchmark
python -m pip install -e ".[test]"
python -m pytest -q
```

普通 `run.py` 默认使用 `reference` 套件并写入 `outputs/reference`：线路库倍率 1，三种不同工况，
根电压为均值 1.02 p.u.、背景波动尺度 0.003 p.u. 的相关过程，根表计相对误差标准差 0.02%。
仍运行四个案例 paper15、soumalas11、flynn16、pengwah18，每个案例使用 3×96 个训练样本、
独立验证样本和 100 次 bootstrap。`--time-limit` 是每次 MILP 扩展的时间上限，并非全流程预算。
新旧设置、独立控制的波动/量测/根可观测性以及压力测试见[算例设置](docs/scenario_design.md)。
场景核验脚本不启动 bootstrap 或 MILP；低数据量从完整 96 点曲线抽取。
高级研究参数由 `rnj_wzzt.cli.advanced_main` 统一解析，历史 `pipeline.main` 仍转发到它。
低层 Python `pipeline.run`、高级 CLI 与六位置参数 `_simulate_pool` 保持 legacy 默认，以便旧实验逐值重放；
新研究在这些入口应显式指定 `scenario_suite="reference"` 或 `--scenario-suite reference`。

## 源码职责

```text
run.py                         固定主线入口
rnj_wzzt/
├─ cli.py                      默认配置与高级参数解析
├─ pipeline.py                 回归、RNJ、收缩、候选补全的流程编排
├─ reporting.py                真值评分和 MILP 结果记录
├─ data/                       四个案例的网络及资源配置
├─ scenario/
│  ├─ settings.py              算例套件、根波动、观测与阻抗配置
│  ├─ simulation.py            AC 场景模拟、量测噪声及训练/验证样本
│  ├─ profiles.py              异质终端 P/Q 曲线
│  └─ validate_scenario.py     网络与终端观测假设检查
├─ models/                     网络对象、AC 潮流及理想 R/X 物理模型
├─ estimation/
│  ├─ preprocessing.py         平方电压降、时序变换及预处理配方
│  ├─ multiscenario.py          标签对齐、消截距、初始化及回归诊断
│  ├─ constrained_least_squares.py  对称/非负/有序约束的凸 QP
│  ├─ laminar_l1_milp.py        候选树原子的 L1 拟合、扩展及路径选择
│  ├─ matrix_constraints.py     矩阵修正、诊断及可选 PSD 兼容模式
│  └─ recipes.py               历史配方导入路径的薄兼容模块
└─ graph/
   ├─ sensitivity_geometry.py  R/X 距离组合与共享路径分数
   ├─ rooted_neighbor_joining.py  RNJ 变体的递归合并
   ├─ bootstrap.py             循环分块重采样和边界子树选择工具
   └─ rooted_hierarchy.py      根子树、伪终端收缩和展开
```

建议从 [结构分析与审查顺序](docs/code_structure.md) 开始。
R/X 数学模型与数值修正见 [回归审查报告](docs/rx_regression_review.md)。

公共模态、公共模态与双 ridge、当前回归的配对实验见
[公共模态对比结果](docs/common_mode_comparison.md)。实验入口为
`experiments/run_common_mode_comparison.py`，默认主线不随实验切换。

## 算法和兼容边界

RNJ 前的默认 R/X 回归使用平方损失与固定线性约束，是凸 QP；RNJ 后的 L1-MILP
使用绝对值损失与候选树结构。两者属于不同阶段。
有序约束或 PSD 修正不保证矩阵对应某棵树，bootstrap 频率也不是校准后的拓扑后验。

`fit_projected_sensitivity` 保留历史四返回值接口。`constraint_refine_iterations` 是求解器
迭代预算（默认 500），设为 0 时只返回可行初始化；`tree_covariance` 仍为可行性启发式。

父项目的两个主线命令及相应算法模块继续转发到这里。部分矩阵诊断、预处理和层次图工具
仍供父项目研究脚本使用，因而保留；父项目并非每个模块都采用转发。

根电压建模、低样本效果与限时联合求解见[根电压与低样本 MILP 审查](docs/root_voltage_low_sample_milp.md)。

## 根信息、结构消融与人工核查研究

本轮完整研究见[统一研究报告](docs/research_study_20260908.md)：固定4网络、36个低样本主条件和12个候选扩展条件，分别报告根部、末端、R/X、运行时间与失败。结果支持RNJ候选+wzzT的条件性优势；自动冻结/收缩并非本批最优配置，生产默认未据此更换。

- [根量测锚定模型与528个回归结果](docs/root_information_model.md)：准确根、噪根、无根量测及公共项/RX正则。
- [先验语义与人工核查排序](docs/prior_acquisition_design.md)：区分完整clade、末端兄弟、根分区和物理接线事实。
- `experiments/run_root_information_study.py`：独立调参/测试的根信息研究。
- `experiments/run_rooted_ablation.py --include-one-edit`：配对消融、固定搜索预算和完整诊断。

两个研究入口可指定`--output`到新目录；默认正式结果、图和原始数组的导航见统一研究报告。

## 独立测试与优势边界压力研究

新增独立数值参照、图与物理性质、MILP枚举和实验协议测试，覆盖最大128终端、弱支路、激励秩亏、异常观测、候选漏失及错误冻结。测试发现并修复负节点标签与自动隐藏节点编号冲突。

- [测试覆盖与复现协议](docs/independent_testing_20260910.md)
- [1508个压力条件的研究结果](docs/advantage_stress_results_20260910.md)
- [几何层分层审阅](docs/advantage_geometry_review.md)
- [优势的理论条件与反例](docs/advantage_theory_boundaries.md)

运行入口为 `experiments/run_advantage_stress.py --output outputs/my_new_stress_run`；生产默认不随本次压力实验更换。