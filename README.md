# Terminal-only topology identification

本目录的主路径是 **固定少量 RX75-RNJ 边界支撑 + L1-MILP 搜索相容的新支撑**。根节点电压已有观测；内部节点零注入且不可观。由含噪声的终端 P/Q/V 时序和公共根电压恢复可辨识线路（以下游终端 clade 表示）。

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

默认同时测试 paper15、soumalas11、flynn16、pengwah18，每个算例使用 3 个训练场景 × 96 点和独立的 3 × 96 验证数据。每次 MILP 扩展的时间上限为 1800 秒。默认使用 reference 场景，结果写入 outputs/reference/。

`rnj_wzzt.pipeline.run()` 统一定义计算流程和默认参数；CLI 解析用户选项并转发。
命令行、直接 Python 调用及历史入口默认均运行完整 RNJ＋MILP。
`--selection-only` 和 `--run-baseline` 分别显式启用仅 RNJ 筛选和额外纯 MILP 对照。

只运行部分算例：

~~~powershell
conda activate Topo
python -m experiments.run_mainline --cases paper15 flynn16
~~~

## 主算法

1. 用公共根电压观测构造 Y_i(t)=V_0(t)²−V_i(t)²，直接使用原始 P/Q/Y，不去均值或拟合末端截距。
2. 联合多场景估计对称、非负且对角有序的 R、X 灵敏度矩阵（对角元不小于同行非对角元加固定裕量）。
3. 分别归一化 R、X，构造 RX75 = 0.75 R + 0.25 X 的 shared-path 核；它不是 R/X 比值。
4. 用 RNJ 和 0.16 × median(root_depth) 容差重构完整候选层次。
5. 做 100 次 circular moving-block bootstrap（块长 4），只从完整 RNJ 树的 inclusion-minimal 非平凡 clade 中选择支持度不低于 0.75 的候选；最多固定 2 个互不相交的末端块。
6. 保留全部原始终端及其 P/Q/Y；将每个终端的 singleton 叶边支撑和筛选后的 RNJ 边界块放入 MILP 的固定初始支撑族，不聚合为伪终端。
7. L1-MILP 每次直接优化新的二进制支撑 z，在全部与当前 family 层状相容且不重复的支撑中搜索新 w z z^T 块，并 fully-corrective 重估所有非负 R/X 系数。
8. 只接受具有原始/对偶最优性证书的扩展；最终用独立验证集的一标准误差规则选择路径点。

真拓扑不参与支撑搜索或模型选择，只用于最终 precision、recall、F1 评价。获证的每一步 MILP 在当前 family 和系数界下是全局最优单次扩展；多步贪心路径不是固定 K 块的联合全局最优。

## 目录导航

| 目录 | 用途 |
|---|---|
| `rnj_wzzt_core/` | 独立安装和运行的 RNJ＋MILP 权威实现，内含核心测试、文档及旧入口的兼容转发 |
| `research_experiments/` | 从核心移出的 RNJ 聚合、替代模型、时序处理等研究代码与测试；`theft/` 提供独立偷电实验启动器 |
| `experiments/` | 仓库级运行入口及历史对照实验；`run_mainline.py` 转发到核心，其他脚本并不都属于当前主线 |
| `terminal_case33/` | 早期算例、基线和研究接口；与核心同名的部分模块是兼容转发 |
| `theft_wzzt/` | 保持原路径和校验协议的独立偷电研究工作区 |
| `configs/` | 算例和旧实验使用的 YAML 配置 |
| `data_external/` | 外部数据集及来源说明；部分原始数据按 `.gitignore` 留在本地 |
| `docs/` | 算法、实验、论文与文献文档；当前主线见 `mainline_algorithm.md` |
| `tests/` | 仓库级兼容及集成测试；核心测试另在 `rnj_wzzt_core/tests/` |
| `scripts/` | 测试、Git 提交、数据处理和研究分析脚本 |
| `artifacts/` | 报告、冻结快照和本地核验记录；大部分临时核验产物不入 Git |
| `outputs/` | 算法与实验的数值结果；默认主线使用 `outputs/reference/`，`outputs/mainline/` 是历史结果 |
| `output/` | LaTeX 编译和 PDF 成品，与数值实验结果分开 |
| `archive/` | 旧项目的可逆归档，不参与默认运行 |
| `tmp/` | 本地临时文件，不纳入 Git |

`.codex-rnj-deps/` 是旧本地依赖环境，`.pytest_cache/` 是测试缓存，
`terminal_load_only_case33.egg-info/` 是 Python 安装元数据，`.vscode/` 是编辑器设置；
这些目录均不属于项目源码提交范围。

rnj_wzzt_core/ 可脱离其余项目源码独立运行，并且是上述核心算法的唯一权威源码。experiments/ 的两个主入口及 terminal_case33/ 中同名的 R/X、RNJ、AC 和 MILP 模块均为兼容转发层。RNJ 聚合、替代几何、时序变换与独立对照集中在 [research_experiments](research_experiments/README.md)，旧研究入口保留转发。偷电实验通过新目录的独立启动器进入原工作区，保留其源码签名、校准和结果路径。outputs/ 中除 mainline/ 外的目录是保留的历史实验数据。

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
