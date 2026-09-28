# Terminal-only topology identification

本目录的主路径是 **固定 RX75-RNJ 末端分块 + L1-MILP 搜索相容的新支撑**。根节点电压已有观测；内部节点零注入且不可观。由含噪声的终端 P/Q/V 时序和公共根电压恢复可辨识线路（以下游终端 clade 表示）。

## 环境

使用 conda 环境 `Topo`（Python 3.11）。首次配置：

~~~powershell
conda activate Topo
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e rnj_wzzt_core
~~~

已有该环境时可直接 `conda activate Topo`。也可用 `conda env create -f environment.yml` 重建同名环境。

## 运行主流程

在本目录执行：

~~~powershell
conda activate Topo
python -m experiments.run_mainline
~~~

默认同时测试 paper15、soumalas11、flynn16、pengwah18，每个算例使用 3 个训练场景 × 96 点和独立的 3 × 96 验证数据。每次 MILP 扩展的时间上限为 1800 秒。结果写入 outputs/mainline/。

只运行部分算例：

~~~powershell
conda activate Topo
python -m experiments.run_mainline --cases paper15 flynn16
~~~

## 主算法

1. 用公共根电压观测构造 Y_i(t)=V_0(t)²−V_i(t)²，直接使用原始 P/Q/Y，不去均值或拟合末端截距。
2. 联合多场景估计对称、非负且对角占优有序的 R、X 灵敏度矩阵。
3. 分别归一化 R、X，构造 RX75 = 0.75 R + 0.25 X 的 shared-path 核；它不是 R/X 比值。
4. 用 RNJ 和 0.16 × median(root_depth) 容差重构完整候选层次。
5. 做 100 次 circular moving-block bootstrap（块长 4），只从完整 RNJ 树的 inclusion-minimal 非平凡 clade 中选择支持度不低于 0.75 的候选；最多固定 2 个互不相交的末端块。
6. 收缩可信末端块，聚合 P/Q，以 0.5 权重反嵌入伪末端电压；固定约化系统的全部 singleton 叶边原子。
7. L1-MILP 每次直接优化新的二进制支撑 z，在全部与当前 family 层状相容且不重复的支撑中搜索新 w z z^T 块，并 fully-corrective 重估所有非负 R/X 系数。
8. 只接受具有原始/对偶最优性证书的扩展；最终用独立验证集的一标准误差规则选择路径点。

真拓扑不参与支撑搜索或模型选择，只用于最终 precision、recall、F1 评价。获证的每一步 MILP 在当前 family 和系数界下是全局最优单次扩展；多步贪心路径不是固定 K 块的联合全局最优。

## 代码与结果结构

~~~text
Topo/
├─ rnj_wzzt_core/                              # 独立可运行、唯一权威实现
├─ experiments/run_mainline.py                 # 兼容入口，转发到核心目录
├─ terminal_case33/                            # 保留的研究与基线模块
├─ docs/mainline_algorithm.md                   # 算法、边界与结果说明
├─ docs/laminar_l1_milp_implementation_report.tex
├─ output/pdf/                                # 当前编译后的 PDF 报告
├─ artifacts/snapshots/                       # 可复核的冻结快照
├─ outputs/mainline/                            # 四算例可追溯结果
├─ tests/                                       # 单元与集成测试
└─ archive/                                     # 只读历史项目与旧构建产物
~~~

rnj_wzzt_core/ 可脱离其余项目源码独立运行，并且是上述核心算法的唯一权威源码。experiments/ 的两个主入口及 terminal_case33/ 中同名的 R/X、RNJ、收缩、AC 和 MILP 模块均为兼容转发层；其他文件用于消融、文献基线或历史研究。outputs/ 中除 mainline/ 外的目录是保留的历史实验数据。

## 历史白名单模式结果

下表来自先前限制 RNJ 白名单的运行，不代表当前默认自由搜索的重跑结果。
核心现已移除白名单模式；重放旧结果应使用相应运行保存的源码快照。

| case | 完整 RX75-RNJ F1 | 联合方法 F1 | MILP 时间（秒） |
|---|---:|---:|---:|
| paper15 | 1.0000 | 1.0000 | 94.92 |
| soumalas11 | 1.0000 | 1.0000 | 3.57 |
| flynn16 | 1.0000 | 1.0000 | 47.46 |
| pengwah18 | 0.9333 | 1.0000 | 142.82 |

原始指标和候选级证据见 outputs/mainline/metrics.json、rnj_summary.csv、rnj_boundary_candidates.csv、milp_results.csv。详细数学定义、终止条件和可证明边界见 docs/mainline_algorithm.md；完整 LaTeX 实现报告见 output/pdf/laminar_l1_milp_implementation_report.pdf。

## 测试

~~~powershell
conda activate Topo
.\scripts\test.ps1
~~~

依赖声明在 requirements.txt、environment.yml 和 pyproject.toml。运行与测试统一使用 conda 环境 Topo，不要再用 `py -3.12` 或 `.codex-rnj-deps/`。

## Git 提交

在仓库目录先预览，再创建本地提交：

~~~powershell
.\scripts\submit.ps1 -Preview
.\scripts\submit.ps1 -Message "描述本次修改"
~~~

默认会暂存所有未忽略的变更。只提交指定文件时，在预览和提交命令后均加 `-Paths @('README.md', 'scripts/submit.ps1')`；此模式要求暂存区原本为空。确认需要上传到 GitHub 时，在提交命令末尾显式加 `-Push`。`-Preview` 只读取状态，不修改暂存区。脚本会检查 `origin`、常见凭证文件与令牌格式，以及超过 25 MB 的新增或修改文件；生成结果和本地依赖遵循 `.gitignore`。
