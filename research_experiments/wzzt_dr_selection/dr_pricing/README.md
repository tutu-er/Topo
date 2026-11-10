# 独立DR概率报价

把本目录复制到任何位置即可运行。运行依赖只有Python 3.10+、NumPy和SciPy，不导入Topo仓库、不读取旧实验路径、不使用pickle。核心只有3个文件，一个命令入口。

| 文件 | 作用 |
|---|---|
| `problem.py` | 读取固定候选及资源，设备控制、费用、梯度组合、实际残差选模和P/Q增量接口 |
| `probability.py` | 原三段高斯特征函数、概率积分及解析导数 |
| `solve.py` | 缩放费用SLSQP、有限可行性恢复、报价核验与冻结 |

数学见[MODEL.md](MODEL.md)，需要用户判断的设置见[REVIEW.md](REVIEW.md)。`results/`仅保留最近一次paper15的历史汇总，不能当本包或新权重的再次实验。

## 独立运行

```text
python -m pip install -r requirements.txt
python -B solve.py --case example --out run
```

`example/`是两个节点、两个可削减资源、2小时的纯合成安装示例。显式采用新日权重1；这没有修改paper15的历史权重。它用于验证代码可运行，不用于证明配网辨识收益。

输出为`quote.npz`（归一化请求x、名义kW交付mean_u、奖励单价unit_reward）、`summary.json`、配置/源码指纹、完整迭代和冻结凭据。重复运行要使用新输出目录，防止覆盖已完成或失败记录。

只核验冻结报价，不进行优化

```text
python -B solve.py --case CASE_DIR --quote quote.npz --check-only --out CHECK_DIR
```

paper15的完整输入和合同仍私有，迁移包位于私有`wzzt_dr_portable_v1/paper15_ready_v2`。该包明确标记`approved_for_new_optimization=false`，只允许回放，待用户决定权重等设置后再规划。它不是从旧路径自动加载的数据。

## 接入其他辨识器

任何辨识器只要能提供下述冻结线性响应，即可准备输入；本包不重新实现RNJ、WZZT或AC。

1. **拟合阶段**：只用历史生成两个候选、固定参数、历史验证损失、原支撑数、输出块族和历史资格。
2. **规划阶段**：将日前预测及参考响应预计算为`baseline_coarse/fine`和`effect_coarse/fine`，连同资源能力写成`case.json + arrays.npz`。
3. **执行阶段**：外部设备响应器生成实际交付。`aggregate_delivery`把资源kW映射到节点kW/kvar增量，外部模拟器只把增量添加一次；基线不重复加入。
4. **评价阶段**：用冻结参数和实际P/Q/V计算两候选残差，调用`score_residuals`，再由外部辨识器按预冻结输出块族统计错漏。真拓扑不进入规划包。

```python
from problem import load_problem, aggregate_delivery

problem = load_problem("CASE_DIR")
point = problem.evaluate(problem.warm)  # 费用、概率、全部请求梯度

# actual_delivery为外部设备实现的资源×时段kW；节点索引从0开始。
dp_kw, dq_kvar = aggregate_delivery(
    actual_delivery, resource_node_index, q_over_p, node_count)

# 两个数组均是外部冻结模型对同一实际观测的残差，形状T×N。
selection = problem.score_residuals(coarse_residual, fine_residual)

# 也可直接传实际功率通道、观测电压平方降及规划前冻结的响应矩阵。
selection = problem.score_observations(
    actual_power_channels, actual_voltage_drop,
    frozen_response_coarse, frozen_response_fine)
```

上面后三项变量由调用方提供，不由优化器猜测。需要标幺输入时由调用方按声明的基准换算。私有paper15包的`physical_adapter.npz/json`保留原资源映射、无功比和200kW/kvar基准。

对原AC接口，执行器仍可用原项目`dr_heterogeneous_pilot.driver.execute`；取其scene中的实际P/Q、根表和末端电压，按残差单位构造`actual_voltage_drop`，再使用本包的`score_observations`。不要调用旧driver的硬编码权重2选模分支。这样规划门槛与实际评价始终来自同一case配置。AC求解器和辨识器是外部依赖，不属于本独立报价包。

若辨识器输出冻结R/X，预计算可以直接使用下式。设参考响应为R₀/X₀，历史剩余电压均值为c_ref，预测P/Q先换成标幺；s为残差单位换算系数，P₀/Q₀为物理基准，n(r)和γᵣ为资源节点与kvar/kW比。

\[
a^0_{k,t}=c_t^{\rm ref}+s[(R_0-R_k)\bar p_t+(X_0-X_k)\bar q_t],\quad
G_{k,jr}=s\left[\frac{(R_0-R_k)_{j,n(r)}}{P_0}+\frac{\gamma_r(X_0-X_k)_{j,n(r)}}{Q_0}\right].
\]

paper15使用s=10⁶、P₀=Q₀=200。c_ref只取未被上述功率解释的历史剩余均值，不再叠加一次同一负荷预测误差。其他预测器可替换这些输入，但若要把预测和交付随机性放进规划，当前单元独立共同电压模型须另改，不能只改一个字段便声称已支持。

## 输入合同

`case.json`的完整可运行模板见`example/case.json`。必须显式填写时间步长、每决策槽数、单位、候选信息和以下选模字段；**新日权重没有默认值**。

```text
selection:
  history_count, new_count, new_day_weight
  historical_loss_coarse, historical_loss_fine
  relative_band, absolute_tolerance
  candidate_ids, support_counts, historically_eligible, candidate_count
```

目前只支持两个历史合格候选、粗族原支撑数少于细族、正验证带、完整新日槽数匹配。超过两候选、重新拟合参数、改用其他损失或非独立残差时会拒绝或须另接评价规则，不能把该对的过线概率称为全体拓扑保证。

历史损失和新观测残差必须采用同一单位、同一观测节点集合，计数使用同一“时段”口径。软件能核对数组和声明，不能从两个均值倒推出原始数据来源是否正确。

`arrays.npz`只允许数值数组，`N`为观测节点数、`R`为资源数、`T`为物理槽数、`H=T/decision_slots`。

| 数组键 | 形状与含义 |
|---|---|
| `baseline_coarse`, `baseline_fine` | T×N，固定自然残差均值，同一残差单位 |
| `effect_coarse`, `effect_fine` | N×R，每kW交付对残差的影响 |
| `sigma` | T×N，共同剩余电压创新标准差，非负 |
| `resource_lower_kw`, `resource_upper_kw` | R×T，负/正功率能力，包含零 |
| `resource_cost` | R×T，正满响应奖励系数 |
| `resource_efficiency` | R，能量转换效率，(0,1] |
| `resource_energy_lower_kwh`, `resource_energy_upper_kwh` | R，能量偏移界，可为无穷，必须包含零 |
| `resource_reset`, `resource_terminal` | R×T，独立服务段开始/末端归零标记 |
| `warm_requests` | 可选R×H请求起点；省略则零起点，无可行性承诺 |
| `resource_future_lower_kwh`, `resource_future_upper_kwh` | 可选R×T剩余能力，移植原控制器时可原样提供，否则按任务段推导 |

reset不能抹掉未完成能量，前一任务须已关闭且间隔没有非零可调能力。所有信息边界字段是调用方声明，形状/数值校验不替代数据来源审查。

## 结果如何判断

`summary.json`分别报告`device_feasible`、`numerical_resolved`、`probability_target_met`。`qualified`要求三者同时通过。数值未决、未达目标、设备未决分别保留，局部失败不当不可行证明，`global_optimal`保持false。

当前概率只对给定预测均值和名义交付后的独立电压创新积分。功率/响应随机性、共享根表误差和真实AC需要调用方独立评价。小规模输入也可能比1440单元难积分，可显式配置`numerics.order/audit_orders`并保留未决，不能裁剪概率以伪造达标。

测试需额外安装pytest，命令为`python -m pytest -q test_problem.py test_solve.py`。迁移核验和清理恢复见`VALIDATION.json`及`../archive/dr_continuous_independent_20261110.md`。

## 节点范围诊断

2026-11-10的[101–105限制试验](studies/node_scope_report.md)区分直接候选差异、共同残差位置和可撤销动作，并报告冻结报价后的原AC结果。`studies/node_scope.py`只准备/核算独立输入，优化仍调用上述`solve.py`；[预锁协议](studies/node_scope_protocol.json)与聚合结果保留。该单因素试验显式沿用旧权重2，不修改默认规则或将固定节点标签用于其他算例。原AC执行器仍是外部接口，不引入核心依赖。
