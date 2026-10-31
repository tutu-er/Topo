# 连续相关 DR 的 SOCP / MISOCP 报价优化

本目录实现两个完整的报价求解器。90% 指事前声明的激励条件共同达标的概率；原 RNJ＋WZZT 在执行后接收真实 P/Q/V，另行评价最终选线。

- **SOCP**：固定合同类型，优化连续期望幅值与报价。
- **MISOCP**：为同一设备选择互斥的合同类型，同时优化连续幅值与报价。

两版使用相同共同因子随机模型、硬预算、设备约束和收益定义。不使用 Copula、VAE、二元参与率或人工节点分散要求。收益方向是显式输入；本实现没有把线性激励收益冒充 WZZT 的最终错漏。

## 运行

已核验环境：Python 3.11.15、NumPy 2.4.6、SciPy 1.17.1、Gurobi 13.0.3，需要有效的 Gurobi 许可证。不依赖 CVXPY，不改原项目默认依赖。仓库根目录执行：

```powershell
$py = 'D:\apps\miniconda3\envs\Topo\python.exe'
$entry = 'research_experiments.wzzt_dr_selection.dr_conic_pricing.cli'
& $py -m $entry example --output C:\scratch\dr_example.json
& $py -m $entry solve --problem C:\scratch\dr_example.json --output C:\scratch\dr_socp --mode socp --allowed '0,3'
& $py -m $entry solve --problem C:\scratch\dr_example.json --output C:\scratch\dr_misocp --mode misocp
```

要使用原资源声明构造的 96 时段 DR＋EV 输入，在 `example` 命令追加 `--resource-day`；其余求解命令不变。此例的 EV 保留连接窗口、SOC与80号时段边界前的服务任务。

示例是两个独立设备、每设备两个全天合同类型的合成资格例，不是用户行为校准或拓扑恢复试验。`--allowed` 使用零起始合同序号；存在互斥组时，SOCP 必须明确选定类型，每组至多一个，避免把同一设备的互斥合同当作独立可结算资源。MISOCP 自行选择类型，也可以不买。

输出目录必须尚不存在，生成：

| 文件 | 内容 |
|---|---|
| `protocol.json` | 规划前冻结的输入/源码哈希、环境、求解限制和独立采样种子 |
| `problem.json` | 实际生效的完整输入 |
| `optimization.json` | 两阶段状态、上下界、gap、幅值、报价、约束独立重算 |
| `validation.json` | 冻结后的独立完整随机路径激励/费用/设备评价，含配对无奖励参照 |
| `ended.json` | 完成或异常记录 |

输入与求解输出可能含真实预测或私有参数。实际应用的输出目录放在私有 scratch；只将白名单汇总加入 Git。程序不自动上传文件。

## Python 接口

```python
from research_experiments.wzzt_dr_selection.dr_conic_pricing import PriceProblem, solve_price
from research_experiments.wzzt_dr_selection.dr_conic_pricing.sampling import validate_frozen_plan

problem = PriceProblem.from_dict(input_dict)
fixed = solve_price(problem, mode='socp', allowed_contracts=[0, 3], time_limit=60.)
selected = solve_price(problem, mode='misocp', time_limit=60.)
if selected['feasible_incumbent']:
    result = validate_frozen_plan(problem, selected['amplitudes'], samples=20000, seed=41032)
```

第一阶段最大化保守的可靠激励收益下界。只有第一阶段报告最优且解通过重算，才在其收益容差内做第二阶段最小费用。第二阶段最小化费用平方根，与最小化非负费用等价，改善零报价附近的数值表现。共享总时限，超时与次优解保留原状态；没有可行解时幅值和报价为 `null`。二元激活与实际正幅值合同分别报告。

## 完整输入

所有数组有限，严格检查形状。`m` 是堆叠功率坐标数，`J` 合同数，`d` 共同因子数，`K` 激励方向数，`R` 设备线性不等式数。

| JSON 键 | 形状 | 含义 |
|---|---|---|
| `mu0` | m | 自然消费减新能源后的预测均值，通常 P[kW]/Q[kvar] |
| `background_factors` | m×d | 自然净负荷的共同因子载荷 F |
| `profiles` | m×J | 单位幅值的整段增量轨迹 H |
| `response_factors` | J×d | 相对响应随机性 D，每行 L1 范数≤1 |
| `elasticities` | J | 正的平均价格弹性 λ，x=λc |
| `reduction_hours` | J | 单位幅值的实际削减能量 e，单位 h，严格正 |
| `price_upper` | J | 单位削减能量报价上限 |
| `actual_amplitude_upper` | J | 实际幅值上限[kW] |
| `benefit_directions` | K×m | 事前固定的激励方向 g，必须注明含义和单位 |
| `benefit_offsets` | K | 收益常数 β；增量收益模式必须为零 |
| `risk_allocations` | K | 固定违约概率 ε，正且和≤1−confidence |
| `budget` | scalar | 整段实际激励支付硬上限 |
| `device_matrix`, `device_rhs` | R×J, R | 实际幅值的设备不等式 A·actual≤b；可为空 |
| `exclusive_groups` | list of lists | 同设备互斥类型的合同序号，组之间不重叠 |
| `contract_ids` | J strings | 合同唯一标识，可省略 |
| `confidence` | scalar | 默认 .9 |
| `benefit_mode` | string | `total` 或 `incremental`，默认总激励 |
| `metadata` | JSON object | 数据来源、单位、坐标顺序、物理服务说明 |

`mu0,F` 已包含新能源，不能再重复扣减。共享因子同时进入 F 和 D，自动保留新能源/自然负荷与 DR 的相关性。协方差由给定 F、D 和优化幅值产生，不是优化器任意选择的变量。假定参数可直接给定，但不能称已从 iFlex 校准。

## 复用原全天 DR/EV 资源

`resources.resource_contract_data(...)` 接收既有 `dr_day_modules.day_schedule.ResourceDay/DayModule`，输出上述物理轨迹、设备矩阵、支付系数和互斥组。再补充背景、收益、预算与风险输入即可创建 `PriceProblem`。

- 全部 96 时段保留，P/Q 分别按 `[时段,节点]` 展平后拼接。
- 复用设备功率、累计能量、SOC、可用窗和 EV 服务截止约束。
- 时间净能量/截止任务等式由每列轨迹原生满足；不依赖随机响应后修补。
- 每列对应一个独立计量的设备合同；多设备共用一个随机幅值的旧 spatial_module 应拆为独立合同，以 D 表达相关性。
- 同一设备的替代类型互斥，避免相反模块分别付费。多个独立设备可以位于同一节点。
- 资源基线是网络总负荷的组成部分，不再次加到总负荷。净注入为负合法。

`sampling.realize_factors(...)` 提供连续实际幅值、完整净功率与增量。`resources.increments_from_amplitudes(...)` 输出原 `node_increments` 形式；将增量加入已有自然负荷**一次**。`translate_net_load(...)` 解码物理 P/Q，不做隐含标幺变换。

## 接入原 RNJ＋WZZT

1. 报价前仅用历史和预测构造输入、冻结价格。
2. 独立产生整段响应与自然/新能源路径，使用既有 AC 流程或真实计量取得 P/Q/V/root；设备可行性不代替网络 AC 校验。
3. 用既有 `end_to_end.wzzt_adapter.make_observed_scenario` 封装实际观测。
4. `identification.fit_full_history(...)` 原样追加新观测，调用原 `dr_pool_v2.adapter.fit_observed`，保留完整历史、15 RNJ 族、原 W 路径、3% 验证与 .16 收缩。
5. 评价最终完整错漏、完整恢复、实际支付、失败/未决和结果来源。

本阶段实现并测试了交接函数，未新跑 AC 或真实 RNJ＋WZZT。`make_observed_scenario` 的输入 P/Q/V 为标幺量；不能直接把 kW 数组当标幺，也不能从线性激励分数伪造电压。

## 验证

```powershell
& $py -m pytest research_experiments/wzzt_dr_selection/dr_conic_pricing/tests -q
```

解析解、非零相关性、增量消去、完整类型枚举、盒顶点设备/预算核对、独立完整路径、不可行/超时和历史交接均有检查。见 `IMPLEMENTATION_REPORT.md` 和 `MODEL.md`。
