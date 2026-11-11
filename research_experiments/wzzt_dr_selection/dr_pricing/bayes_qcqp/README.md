# 平方残差证据的二次DR规划

当前相对损失版本集中在单文件[exponential_dr.py](exponential_dr.py)，代码开头包含符号、假设、公式、输入输出和接入说明，对应[IEEE方法短稿](relative_gate_paper/README.md)。固定λ=1/(4κ)，计算W/h，最小化费用并要求d(u)ᵀWd(u)≥h；保留指数充分条件，不增加另一种概率近似。直接执行`python exponential_dr.py`运行纯合成安装例。

`build_gate`预计算条件，`solve_continuous`求解二次费用和线性可行域的QCQP，`probability_bound`输出概率下界，`observed_loss`重算固定模型的实际平方损失。原小时设备直接把gate传给已有`solver.solve(problem, gate, gate.h, out)`，不复制设备图或辨识器。25项独立数学/求解/迁移检查通过；当前κ>1版本没有新增paper15、AC或最终恢复实验，与下列旧fixed_f后验结果分开。

新增[固定F概率分支](FIXED_F_REPORT.md)：分别实现平均后验和后验过线率的充分二次条件，完整推导见[LaTeX](fixed_f_theory/main.tex)。原期望log odds模式保留，不能把其指标解释为下面的新概率保证。

```text
python -B fixed_f_run.py plan --case example --settings example/fixed_f.json --mode mean --out fixed_mean
python -B fixed_f_run.py plan --case example --settings example/fixed_f.json --mode probability --out fixed_probability
python -B fixed_f_run.py sample --case example --settings example/fixed_f.json --plan mean=fixed_mean --out fixed_sample
```

固定分支只增加`fixed_f.py`与`fixed_f_run.py`两个计算文件，复用原设备求解器；`fixed_f_arrays.npz`明确给完整时空白化差协方差，分解一次。`mean`要求E[qF|F]≥qstar，`probability`要求Pr(qF≥eta|F)≥beta；F只表示固定背景误差载荷，名义DR按确定交付规划。原两槽例前者全局费用13.164913，后者充分条件已证明不可行；满幅动作的精确概率却可行，详见报告的保守性核验。paper15两模式均全局费用0，32旧AC观察复用，没有新增DR恢复收益。

这里实现用户提出的“高斯似然 → 后验门槛 → 平方残差差 → 取期望 → 非凸二次约束”。两个候选及其R/X在历史拟合后冻结，复用上层原设备模型，不重新实现RNJ/WZZT。

**取期望得到的是期望对数证据约束。它不等价于90%的后验过线概率。** 实现同时给出一个名义功率、闭合双候选下的精确概率对照，便于检查这一区别。

- [MODEL.md](MODEL.md)：完整符号、推导、设备约束、求解界和适用条件。
- [LaTeX理论文档](loss_conventions/README.md)：如何确定损失口径、L1/L2基础、未知尺度相对门槛及协方差的完整推导；未知尺度尚未实现。
- [REPORT.md](REPORT.md)：解析示例、paper15和保存AC观测的实际结果。
- [PROTOCOL.json](PROTOCOL.json)：事前设置；[RESULTS.json](RESULTS.json)：无轨迹的聚合结果。

## 运行

保留父目录`problem.py`与`solve.py`。NumPy、SciPy之外，非零优化需要可用的Gurobi许可证；本轮使用Gurobi 13.0.3。软件不读取许可文件内容，也不导出许可证。

```text
python -m pip install -r requirements.txt
python -B run.py plan --case example --settings example/bayes.json --mode expected_either --out run_expected
python -B run.py sample --case example --settings example/bayes.json --quote run_expected/quote.npz --out check_expected
python -B run.py plan --case example --settings example/bayes.json --mode pair_event_nominal --out run_nominal
```

以上命令从本目录运行。输出目录须为新目录；已有成功、失败和超时记录均保留。`sample`要求规划冻结凭据，核对报价、输入及计算源码哈希。报价冻结后修改源码，应使用原快照回放，不能悄悄换模型继续评价。

`example/`只有一资源、两时段的公开合成数据，具有解析最优值。它用于安装与数学核验，不是实证配网算例。paper15的输入、报价和AC轨迹留在私有恢复目录。

## 只有三个新增计算文件

| 文件 | 输入 | 输出与作用 |
|---|---|---|
| `core.py` | 两候选残差均值、响应系数、似然标准差、响应方差 | 后验、期望证据的二次系数、名义分离和解析概率 |
| `solver.py` | 父层设备及费用、二次系数、门槛 | 原小时控制器的精确混合整数图、Gurobi求解、上下界、原设备回放核验 |
| `run.py` | 原case与显式`bayes.json` | 两方向规划、冻结、独立Gaussian诊断；不读实际未来或真拓扑 |

主模式`expected_either`分别求解F、C两个证据方向，取核验通过的最低费用解。零报价若可行，非负费用直接证明全局最优0，不伪造一次求解器运行。非零问题采用`NonConvex=2`；完整设备图含离散变量，因此为MIQCP。状态、可行报价、下界、gap和超时分别保存。

`pair_event_nominal`只在**名义确定交付、两个冻结Gaussian候选之一为生成模型**的条件下，要求正确候选后验过线率达到目标。它没有自动包括随机设备执行和AC误差，不能借此替主模式声称概率保证。

Gurobi的任意二次约束与`NonConvex=2`空间分支处理，参见[官方约束文档](https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/constraints.html)及[参数文档](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameter:NonConvex)。本轮已实测许可证和全局界，不仅检查包是否可导入。

## 新输入

父层`case.json + arrays.npz`继续提供冻结两候选、原小时请求、资源功率、效率、能量、截止任务和费用。`bayes.json`额外显式提供：

| 字段 | 含义 |
|---|---|
| `likelihood` | 当前只接受`fixed_common_diagonal_gaussian` |
| `base_prior` | C、F顺序的两个正且归一化的基础先验 |
| `history_rows`, `history_log_bayes_factor` | 同一Gaussian似然从历史验证观测计算的行数、对数证据 |
| `history_receipt` | 历史证据计算凭据的路径和SHA-256；有历史时必需 |
| `posterior_threshold` | 后验门槛η，位于(0.5,1) |
| `event_probability_target` | 仅名义对照使用的达成率β |
| `arrays_file` | 数值`bayes_arrays.npz`的位置 |
| `solver` | 每方向时间、gap、线程数、随机种子 |

数值文件至少含`likelihood_sd[T,N]`。可提供`response_relative_sd[R,T]`及两候选的额外背景残差方差。额外方差须对应尚未计入的误差；独立采样接口遇到非零背景方差会要求联合因子，而不从两个边际方差猜测联合响应。

新证据按观测求和，历史和新日均一次计入；**不套用父case的MAE、3%带或新日权重2**。父层字段只供旧规则并列评价。当前后验仅覆盖这个固定拟合候选对，未对参数积分，也未处理候选筛选后的概率校准。

## 如何评价

1. 优化只用历史与日前预测，冻结报价。
2. `sample`独立生成规划Gaussian模型的响应与共同电压误差，核验二次期望、后验过线率和名义公式。它不调用AC。
3. 外部原执行器按真实功率、设备截断及P/Q/V量测流程产生AC观测；对同一观测计算冻结候选残差，用`core.posterior_pair`评分。保存选择后再读取真支持统计错漏。
4. 新Bayesian判据与原3%规则分开报告。更换选模判据带来的改善，不计作DR收益。本轮新报价为零，已核对与保存零合同完全相同，故直接复用32条已有AC观测，没有重复运行相同物理场景。

当前26项测试通过。paper15主模式给出全局费用0；另一个更强名义对照120秒未找到报价，保留未决。结果没有证明本轮产生新的付费DR收益。
