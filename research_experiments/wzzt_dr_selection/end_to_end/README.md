# paper15 需求响应—WZZT 环境验收

当前目标是检查“历史预测 → 价格规划 → 随机需求响应 → AC 运行校验 → 实际 WZZT 拟合”是否按同一数据和预算约定运行。本批为一个配对实现、三个策略的环境验收；价格收益、完整恢复概率和方法创新均不由这一批证明。

代码入口为 `research_experiments.wzzt_dr_selection.end_to_end`，配置为本目录 `config.json`。已有的 WZZT 核心直接调用，不在价格搜索内部反复调用 WZZT。

## 数据与物理算例

| 项目 | 当前配置与来源 |
|---|---|
| 拓扑 | 复用仓库 paper15 的 reference 算例：15 个观测终端、隐藏内部节点无注入、阻抗倍率为 1。它是仓库合成参考馈线，不能写成原始论文实测馈线。 |
| 有功负荷 | iFlex Phase 2 的 Oslo / Os_Control 家庭实测小时用电。257 户符合固定资格，按种子 20261027 随机选定 45 户，每个终端聚合 3 户；选择不使用未来负荷值。 |
| 功率尺度 | 原始一小时电量 `Demand_kWh` 转为该小时平均 kW，三户直接求和，无幅值重缩放。主算例使用住宅正负荷，替代 paper15 原有混合负荷和 DER 曲线。 |
| 无功 | 复用 reference 中住宅功率因数曲线，由 P 合成 Q。iFlex 未提供本算例所需的实测 Q、V；Q 和 AC 电压均为模拟量。 |
| 菜单 | 使用原始完整 24 小时 `P_2/P_5/P_10/P_15` 奖励系数，单位 NOK/kWh；非事件小时价格也保留。另有严格零奖励菜单 `no_reward`。 |
| 规划响应 | `control_training_menu_factors.csv` 的训练期均值，按事件前、事件中、事件后三段展开。它们仍是暂定估计；现有校准未验证菜单专属均值相对共用响应均值具有额外预测优势。 |
| 实际响应池 | 留出期实验组相对自身历史的小时变化，再除以对应控制组的同类变化，并使用训练期固定三段归一化。保留整日 24 小时轨迹，不向规划均值重新居中。 |

原始来源索引为 iFlex 数据记录 8248802；原始文件及校准产物的 SHA-256 记录在 `data_manifest.json`。`privatebundle.pkl`、家庭 ID 映射和逐户原始文件留在本地数据缓存，不上传到代码仓库。

响应池应用到选定的控制组实际负荷，形成**半合成的价格反事实**。它包含真实整日波动，但不是这些家庭在所有菜单下同时观测到的个体因果响应。该版本未建立独立个体参与概率模型。

同一菜单只抽取一条整日聚合因子，并共同作用于该菜单下的终端；空间差异来自终端实际负荷与菜单分配。各菜单使用共同随机数配对抽取，三个策略共享同一菜单实现。它不是节点、小时或菜单之间独立同分布的采样。

## 时间划分与执行顺序

默认日期与历史已在 prepare 产物核验：

| 用途 | 数据范围 |
|---|---|
| 响应均值校准 | 截止 `2021-02-08` 之前；该日不含在训练中。 |
| 预测历史 | `2021-01-25`—`2021-01-29` 和 `2021-02-01`—`2021-02-05`，共 10 个工作日、240 个小时。规则跳过行动日前一日，并排除周末和指定节假日，所以最后一天是 **Feb 5**，不是 Feb 8。 |
| WZZT 历史拟合 | 上述历史最后 3 天，即 Feb 3、4、5，共 72 个小时。 |
| 行动日 | `2021-02-09`，24 小时；实际观测加入 WZZT 训练，总训练样本为 96 个时刻。 |
| 路径验证日 | `2021-02-10`，24 小时；重复已冻结价格分配，使用另一个实际负荷日和另一个响应池，不重新规划。 |
| 响应池原始日期 | `2021-02-09`—`2021-02-22`。每个菜单按完整源日期分为行动池和验证池，同一菜单的两个池不共享日期。 |

当前响应池为 13 个组日，行动池各菜单大小依次为 P2=2、P5=3、P10=1、P15=2；验证池为 P2=1、P5=2、P10=1、P15=1。日期分离针对每个菜单；两池仍可能共享家庭群体和跨菜单的日期环境。独立随机种子不等于统计独立样本，也不能由这些小池估计精确恢复概率。

运行顺序如下：

1. 从固定家庭映射生成十天历史 P，按住宅功率因数生成 Q，执行 AC 并生成历史计量实现。
2. 只用十天带计量误差的 P/Q，逐小时取均值，形成行动日预测。历史噪声实现同时用于价格设计和被选入的 WZZT 历史数据。
3. 只用预测 P/Q、训练期菜单均值和历史输入，枚举 65 个预声明分配：零奖励、四个全节点统一菜单、四个菜单在各单节点投放。默认选择普通 D-opt 输入设计分数最高的可行分配。该分数不是线路置信度，也不是新增的 WZZT 指标。
4. 冻结价格分配。三个策略分别为 `no_reward`、首个终端 P2 的 `fixed_P2_first_node`、`planned_d_opt`，共用预测、约束、响应抽样和对应日物理随机流。
5. 在实际行动日与验证日负荷上施加整日因子。无奖励时实际 P/Q 原样保留；有奖励时 P/Q 使用同一乘法因子，保留基线功率因数。
6. 运行 AC，报告功率平衡、电气方程、收敛与电压范围；再调用一次实际 WZZT 核心完成该策略的逐线路搜索和路径回选。真拓扑只在最后计算有效块族恢复时使用。

验证日用于 WZZT 的 1-SE 路径回选，所以该日电压误差应称为路径验证误差，不能称为未使用的独立测试误差。本版本未设置第三个完全保留的预测测试日。

## 约束、结算与配置

每终端聚合功率上限为 45 kW，来源是声明的 `3 家庭 × 15 kW` 合同容量；它不是 paper15 线路热稳定额定值。每个终端日能量变化的绝对值上限为基线日能量的 30%，不强制日能量守恒。

规划使用均值预测检查功率和能量。实现阶段相对该日真实无响应负荷检查功率和能量；超限结果保留原轨迹并记入分母，不裁剪、不重抽、不因失败更换家庭映射。

结算基线为冻结的行动日均值预测，两个模拟日共用该基线。每终端视为一个聚合结算账户，在完整 24 小时末才对累计有符号奖励取正部：

\[
\mathrm{payment}_i=\max\left\{0,\sum_{h=1}^{24}\pi_{hi}\bigl(b_{hi}-p_{hi}\bigr)\right\}.
\]

其中 \(\pi\) 为 NOK/kWh，\(b,p\) 为一小时平均 kW；本例每格为一小时，所得为 NOK。负小时奖励先抵扣正小时奖励，不能逐小时先截零。这是本例声明的日结算约定，不声称复现 iFlex 原实验末期全部结算。

每个 24 小时模拟期的预算均为 1000 NOK，验证期另有相同预算。规划采用保守支付界 \(\sum_{i,h}\pi_{hi}b_{hi}\le1000\)，由价格、基线和实际负荷均非负得到；它比平均响应下的支付更保守。固定 P2 和无奖励策略读取同一候选审计，遵守完全相同的约束。规划不可行的策略保留 `planning_infeasible` 记录，不绕过约束执行。

其他锁定默认值：

| 参数 | 值 |
|---|---|
| D-opt | 历史列 RMS 固定尺度，ridge=`1e-6`；历史使用 240 个时刻，无截距。 |
| P/Q 相对计量噪声 | 0.005 |
| 终端/根电压相对计量噪声 | 各 0.0002 |
| 物理根电压波动标准差 | 0.003 |
| 电压检查区间 | 0.9—1.1 pu |
| WZZT 初始化 | 终端单元素支撑；不调用 RNJ 初始化，调用生产 WZZT 核心。 |
| WZZT 求解器 | 配置文件默认 Gurobi，每次扩展时间上限 30 秒，要求 `mip_rel_gap=0`。 |
| 系数初始上界 | 2；最多扩界 4 次，实际扩界与求解状态写入记录。 |
| 支撑容量 | 默认 `2N−1=29`，含单元素初始支撑。 |
| 验证 | 每场景 4 块，使用生产代码的 1-SE 规则。 |
| 随机流 | 历史 41000+日序号；行动 42000；验证 43000；响应行动 44000；响应验证 45000。 |

`technical_flow_success` 只表示三个策略到达可评价的返回模型；必须同时查看 `fit_complete_count`、`fit_incomplete_count`、运行约束和失败数。未证明最优的扩展、超时及中断均保持未决状态。AC 收敛、电压通过与终端合同通过也不能合并写成所有线路安全已验证。

## 输入与输出格式

价格层接受 `24 × N` 的 NumPy P/Q 矩阵，P 单位 kW、Q 单位 kvar，列严格对应冻结终端顺序。价格均值因子为每菜单长度 24 的向量；实现响应池为 `S × 24 × 1` 的整日公共因子，抽样后扩为 `24 × N`。

物理层接受 pandas DataFrame：小时索引完整且无重复，列为排序后的终端编号，P/Q 时间和列完全一致。正 P/Q 表示消耗。默认住宅响应要求 P 非负。

WZZT 层接受带显式单位的观测字典：`P_terminal`、`Q_terminal` 使用 pu，`V_terminal` 和 `V_root_observed` 使用 pu；目标为同期测得的 `V_root_observed²−V_terminal²`，单位 pu²，不中心化、不增加拟合截距。runner 将物理结果经 `make_observed_scenario` 转换，不直接混用 kW 与 pu，也不静默取交集或丢弃时间行。

关键产物：

| 文件 | 含义 |
|---|---|
| `prepared_receipt.json` / `data_manifest.json` | 配置、原始输入与 bundle 哈希、日期划分、响应池来源。 |
| `privatebundle.pkl` / `private_household_mapping.json` | 本地数据与家庭映射；不提交。 |
| `plan.json` | 均值预测、全部 65 候选审计、选中分配、保守预算。 |
| `response_draws.json` | 两期整日抽样及源组日索引。 |
| `history/*.pkl` / `*.checks.json` | 历史实际物理/计量实现及检查。 |
| `strategies/<name>/action.pkl` / `validation.pkl` | 实现响应与 AC 结果，供恢复使用；含本地负荷数据。 |
| `strategies/<name>/wzzt.json` | 核心源码哈希、参数、逐步路径、求解器状态和模型。 |
| `strategies/<name>/result.json` | 本策略完整审计、恢复评价及失败信息。 |
| `run_state.json` / `summary.json` | 已完成索引，以及完整三策略分母和最终汇总。 |

## 正常入口命令

以下 PowerShell 命令均从仓库根目录执行。先核对目标输出目录中的 `runner.lock` 和进程，已有本批运行时不要重复启动或重新 prepare。

```powershell
$topoPython = 'D:\apps\miniconda3\envs\Topo\python.exe'
$flowData = 'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_data_calibration'
$flowConfig = '.\research_experiments\wzzt_dr_selection\end_to_end\config.json'
$flowCalibration = '.\research_experiments\wzzt_dr_selection\data_calibration'
$flowOutput = Join-Path $flowData 'flow_runs\paper15_iflex_v1'
```

通过正常测试脚本调用五组本任务测试。必须把 pytest 参数放入 `-PytestArgs` 数组，避免 PowerShell 把 `-o` 解释成脚本自身参数：

```powershell
& .\scripts\test.ps1 -PytestArgs @(
    'research_experiments/tests/test_wzzt_dr_adapter.py',
    'research_experiments/tests/test_wzzt_dr_data.py',
    'research_experiments/tests/test_wzzt_dr_price.py',
    'research_experiments/tests/test_wzzt_dr_physical.py',
    'research_experiments/tests/test_wzzt_dr_flow.py',
    '-q',
    '--basetemp', (Join-Path $flowData 'pytest_flow'),
    '-o', ('cache_dir=' + (Join-Path $flowData 'pytest_cache_flow'))
)
```

首次准备私有数据包：

```powershell
& $topoPython -m research_experiments.wzzt_dr_selection.end_to_end `
    --config $flowConfig --data-root $flowData `
    --calibration-dir $flowCalibration --output-dir $flowOutput --prepare-only
```

prepare 需要原始数据、校准产物及 pyreadr。代码优先复用已校验的控制组缓存；需要读取 RData 时可使用当前 Python 环境或数据缓存内匹配版本的 `deps_py311`。正式运行读取已准备 bundle，不反复解析大 RData。

首次正式运行：

```powershell
& $topoPython -m research_experiments.wzzt_dr_selection.end_to_end `
    --config $flowConfig --data-root $flowData `
    --calibration-dir $flowCalibration --output-dir $flowOutput
```

同一批恢复：

```powershell
& $topoPython -m research_experiments.wzzt_dr_selection.end_to_end `
    --config $flowConfig --data-root $flowData `
    --calibration-dir $flowCalibration --output-dir $flowOutput --resume
```

## 断点与结果边界

`runner.lock` 保存 PID、主机、配置哈希与所有者标识。存在锁时程序拒绝第二次进入；先检查该 PID 及实际命令，只有确认原进程已结束后才处理陈旧锁，不能删除仍在运行的锁。

恢复要求配置、代码、bundle 和菜单输入的指纹一致。已保存的历史、实现观测和完整策略结果直接复用；完整批次直接返回既有汇总。修改模型、预算、数据或代码后应使用新输出目录，保留旧证据。

恢复不会延长求解预算，也不会从部分 MILP 搜索树继续求解。若发现 `wzzt.json` 仍为 `running`，会把该次核心调用记为 `interrupted` 并保留断点，不自动重复求解。需要另行求解时必须声明新批次及原因。

本次新增42项和相关既有76项测试已通过。真实三策略流程调用3/3、运行约束通过3/3，WZZT完成1/3、首步超时未决2/3，完整恢复0/3。完整结果、分母及后续边界见 [REPORT.md](REPORT.md)。真实 `--resume` 已验证返回原汇总而不重新求解。
