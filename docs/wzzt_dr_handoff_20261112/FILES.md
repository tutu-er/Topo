# 文件索引与最短阅读路径

以下W表示当前工作树`C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo`，M表示主项目`D:/0-github_workspace/Topo`，P表示README列出的私有恢复根。`manifest.json`给出逐文件根目录、SHA-256与用途；本表只列最需要看的入口。

## 算法设计：先判断采用哪一版

| 文件/入口 | 用途及版本 |
|---|---|
| M/docs/wzzt_rnj_best_strategy_20261028/README.md、best_strategy.json | 旧联合最佳策略规格；JSON是说明，不是现成CLI配置。80组F1结果属于旧3%MAE/.16流程 |
| [W/dr_pool_v2/ADAPTER.md](../../research_experiments/wzzt_dr_selection/dr_pool_v2/ADAPTER.md)、[adapter.py](../../research_experiments/wzzt_dr_selection/dr_pool_v2/adapter.py) | `fit_observed(training, validation, output_dir, ...)`返回rnj_bank/baseline25/multistart_original_rule；已有任意观测接口，不要照搬旧文档中“尚无包装器” |
| [engine.py](../../research_experiments/wzzt_dr_selection/dr_pool_v2/engine.py)、[selection.py](../../research_experiments/wzzt_dr_selection/dr_pool_v2/selection.py) | 旧联合建池/共同L1调权/交叉删除/多初值，3%MAE选后.16收缩 |
| M/research_experiments/miqp_loss_benchmark_20261111/README.md、squared_backend.py | 后续MSE/MIQP研究，PoolQP与原搜索运行时后端；.24后训练QP。与旧15完整RNJ最终共选策略不同 |
| M/research_experiments/mse_exit_tuning_20261111/README.md、gate.py | `pure_mse_gate`：训练final-E接受M_old-M_raw>1e-10/K+.001M_old；不是最终DR验证门槛 |
| [MSE_SELECTION_PLAN.md](../../research_experiments/wzzt_dr_selection/dr_pricing/paper15_exponential/MSE_SELECTION_PLAN.md)、[selection_mse.py](../../research_experiments/wzzt_dr_selection/dr_pricing/paper15_exponential/selection_mse.py) | 当前DR严格物理MSE选模、完整历史＋新日权重1、`choose_mse/required_new_sse_gap` |

W/dr_pool_v2等简写均位于`research_experiments/wzzt_dr_selection/`之下。新窗口前面的算法设计若已有更新，应以其代码核对实际规则，保持这些版本为明确参照。

## DR：从候选到计划、实际观测

| 文件/入口 | 输入 → 输出 |
|---|---|
| [conditional_target/README.md](../../research_experiments/wzzt_dr_selection/dr_pricing/conditional_target/README.md)、[zero_screen.py](../../research_experiments/wzzt_dr_selection/dr_pricing/conditional_target/zero_screen.py) | 已存资格/历史/参数 → 逐对及整库零DR充分界。最新结果在同目录RESULTS/AUDIT/PROTOCOL |
| [mse_existing_cases/core.py](../../research_experiments/wzzt_dr_selection/dr_pricing/mse_existing_cases/core.py) | `load_history_case`读取原.24/QP库与历史哈希；`build_gate`计算物理MSE概率条件。目标/竞争者方向必须按MODEL显式设置 |
| [paper15_exponential/miqp_mse.py](../../research_experiments/wzzt_dr_selection/dr_pricing/paper15_exponential/miqp_mse.py) | `MSEGate/solve_factorized`：单对物理MSE指数条件、原设备图求解先例。原tau=.05是单独诊断，不能直接当最新最终门槛 |
| [vendor/problem.py](../../research_experiments/wzzt_dr_selection/dr_pricing/paper15_exponential/vendor/problem.py)、[vendor/solver.py](../../research_experiments/wzzt_dr_selection/dr_pricing/paper15_exponential/vendor/solver.py) | 原设备域、小时请求映射、状态/截止/费用及`build_device_graph`；非零优化复用这些，不写等价识别器 |
| [devices.py](../../research_experiments/wzzt_dr_selection/dr_heterogeneous_pilot/devices.py)、[library.json](../../research_experiments/wzzt_dr_selection/dr_heterogeneous_resources/library.json)、[case_allocations.json](../../research_experiments/wzzt_dr_selection/dr_heterogeneous_resources/case_allocations.json) | shift/curtail/EV/battery四类资源；`nominal/project/draw/audit`，相关连续响应、因果截断和费用。参数是合成设定 |
| [paper15_exponential/ac.py](../../research_experiments/wzzt_dr_selection/dr_pricing/paper15_exponential/ac.py) | paper15本机可迁移算例的物理适配；不是直接任意网络入口 |
| [mse_existing_cases/run.py](../../research_experiments/wzzt_dr_selection/dr_pricing/mse_existing_cases/run.py)、[ac_bridge.py](../../research_experiments/wzzt_dr_selection/dr_spatial_correlated/ac_bridge.py) | 原多网背景/设备/AC含噪执行链；只有冻结报价后才打开runtime/真实新日 |

资源README里仍有旧95%/L1文字，最新法则以conditional_target/PROTOCOL.json与MODEL为准。旧完成批次的run.py可能带all/actual命令，阅读索引不等于授权重复运行已完成实验。

## 理论和实验参考

- [relative_gate_paper/method_appendix.pdf](../../research_experiments/wzzt_dr_selection/dr_pricing/bayes_qcqp/relative_gate_paper/method_appendix.pdf)及同目录main.tex：条件高斯、指数界、积分与QCQP背景。其正文采用加权平方损失；当前未加权物理MSE以本交接MODEL和miqp_mse/core为准。
- [MIQP_REPORT.md](../../research_experiments/wzzt_dr_selection/dr_pricing/paper15_exponential/MIQP_REPORT.md)：paper15非零动作及原3%MAE的17/32结果、费用和求解界。
- [MSE_SELECTION_AUDIT.json](../../research_experiments/wzzt_dr_selection/dr_pricing/paper15_exponential/MSE_SELECTION_AUDIT.json)：旧观察严格MSE重评分核查，不是新独立实验。
- [mse_existing_cases/REPORT.md](../../research_experiments/wzzt_dr_selection/dr_pricing/mse_existing_cases/REPORT.md)：两组错误H1/不完整候选库的负对照与区域损失分解。
- [mse_reference_repair/REPORT.md](../../research_experiments/wzzt_dr_selection/dr_pricing/mse_reference_repair/REPORT.md)：一般Hbar替代分支、方向修正与Soumalas零费候选补充；非当前主模型。

## 数据与恢复

| 位置 | 状态与使用方法 |
|---|---|
| M/outputs/mse_exit_tuning_20261111/catalog.json、jobs/<job>/result.json | 主48组历史候选及冻结参数，40组合格。不要读取catalog指向的test来设计价格 |
| M/outputs/day96_t_benchmark_20261110/configuration.json、data/、networks/ | 旧训练96/验证288及原网络；图与负荷是原项目模拟算例，不当实证用户响应。复现按具体协议的源快照/哈希 |
| P/wzzt_conditional_zero_screen_v1/run_v1 | 40/188最新筛查，已complete；protocol/requests/rows、source_snapshot及两个独立审计齐全，无新AC |
| P/wzzt_mse_dr_diagnostic_v1/case_inventory.json、run_v1 | 原48资格清单、两错误H1的已完成36观察；不要重跑 |
| P/wzzt_reference_repair_v1/run_v1 | 两网各6新日、零价别名、候选补充与参考分支，已完成 |
| P/wzzt_paper15_miqp_gate_v1 | paper15已完成非零DR、32配对路径、原始报价和AC结果；保留原随机流/源快照，旧数据不可再称全新留出 |
| [STATUS.md](../../research_experiments/wzzt_dr_selection/STATUS.md)、[RECOVERY_INDEX.md](../../research_experiments/wzzt_dr_selection/RECOVERY_INDEX.md) | 全历史追溯；本交接README/MODEL是当前方向入口，旧段落是时间记录 |

私人轨迹、拟合数值、报价NPZ与完整恢复目录保持原位。已有记录含失败/限时/未决，读完成标记与结果，不只凭旧运行锁字段判定需要重启。
