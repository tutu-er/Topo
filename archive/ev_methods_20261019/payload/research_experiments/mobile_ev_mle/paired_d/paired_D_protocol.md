# continuation_v2 execution protocol

The original eight-hour cutoff remains recorded in runs_v1 and is not changed.
The user authorized continuation at 2026-10-13 01:54 UTC. This new execution
extends only the no-new-history and final checkpoint deadline to
2026-10-13 03:00 UTC. Work here is outside the original eight-hour budget.

The statistics, calibration seeds 940001-940032, evaluation seeds
950001-950048, 32/48 requested counts, candidate bank, two methods, H14 R/X
initialization, common fit seed, budgets, risk allocation, failure retention,
and paired comparison rules are unchanged. The two pilot seeds are
941101/941102; repeating them checks this integrated runner and adds no new
independent pilot units. Pilot observations never enter formal denominators.

The original runs_v1/frozen.json, pilot and protocol are immutable. This
continuation has its own freeze, output and source hashes. Evaluation can
start only after all 32 calibration histories have been attempted and the
new calibration thresholds have been written and frozen. Partial batches
and failures remain recorded; neither counts nor deadlines will be padded.
Thread count is one and Windows process priority must be BelowNormal.


--- Original protocol retained below ---

# 冻结 D 的 open-loop / rolling 配对补充方案

状态：**制定方案与无真实历史的契约测试；未运行生成器、AC、拟合或新正式批次。** 原 D 64+96 与原报告结果保持原样。此方案使用新冻结wrapper之后生成的fresh独立历史，增加一个对照，检验“执行首两格后反馈重规划”相对于“执行同一首轮八格计划”的作用；不重新挑动作，不依据真值或未来数据选择方案。

pilot种子941101/941102，不纳入正式分母。只有pilot平均时间≤45s且契约通过，才按预声明预算运行新calibration32（940001–940032）、evaluation48（950001–950048）；pilot>45s则保留pilot/失败与预算检查点，不扩正式批次。00:15 UTC后不开始新历史；已经开始的历史完整结束，01:00 UTC总预算留检查点。pilot前根代理审查方案/代码/测试与冻结哈希；本文件不授权生成actual H。

## 1. 两个已选动作与识别对象

每个fresh独立历史 i 都调用原已冻结 `rolling_trial` 取得完整影子轨迹，H0有12点。令该fresh trace已保存首轮实际八格计划为 u_i^1（读取 `selections[0].actual_plan`，包括原操作回退，不读搜索器中未执行的方案），实际 rolling 全轨迹为 u_i^R（读取 `realized_plan`）。定义 openloop u_i^O=u_i^1。两个动作在新的共同拟合/离线真值评分之前都已固定。

原策略首轮只根据 H0 选择 u_i^1，实际首两格为 a_i=u_i^1[0:2]。因此必须验证 u_i^R[0:2]=u_i^O[0:2]=a_i，包括 P+/P-/净P、route/site、行驶能耗和SOC。第二轮 rolling 使用 H1=H0+Y_i^{1:2} 与继承公共状态，余六格可以改变。openloop 余六格直接沿原计划执行，不读取 H1 改动作。

目标是冻结策略及声明模拟总体下的**配对算法结果差**，不是全局MLE、实际现场因果收益或全拓扑正确概率。由于模拟是逐时稳态、外生负荷/源/计量噪声不受EV反馈影响，可用共同随机数构造两种动作的同路径潜在观测。两方法共同净网电任务6kWh、初始SOC/site/access/效率/Pmax与行驶耗能规则相同；实际路线及其额外行驶成本允许不同，并单独报告。

## 2. 可重复的外生路径和重放闸门

读取完整冻结 D policy，先生成并归档fresh影子trace；随后调用 `_evaluation_path(seed, policy, None)` 重建同一外生path，只由 evaluator 持有返回的path/truth。`simulation.generate` 已预存全部逐时noise，`observe(path, slice, action)` 只切片并加动作，内部没有新的随机抽样，因此同一seed和同一政策参数给共同背景、源、R/X与噪声。

先使用原实际 rolling 动作，按原来的 **2+6分块** `observe` 重建 H0、Y1、Y2 和 H20；与原 NPZ 的每一通道核对，再构造 openloop。分块保持一致可避免向量化 AC 收敛停止因批量大小改变而产生微小差异。

- p/q/root 与已执行u须逐元素精确相等；历史12点和首两格两方法必须相同。
- v/master_P/master_Q优先字节一致；若浮点平台仅有舍入差，允许预声明绝对最大误差2e-10 pu，并记录每通道最大差及所有不一致位置。此容差对应当前AC方程/平衡核查合同，不在观察误差后自动扩大。
- 保存 generator/observe/current-loaded-code哈希、运行库版本和每条fresh JSON/NPZ SHA-256；原控制器依赖必须匹配原冻结 D 代码契约，fresh seed不与原D/C/pilot重叠。
- 一旦源/策略/档案或重放不匹配，保留失败证据并停为核查检查点。未经解释不得把不同路径叫配对，不跳过失败seed凑分母。

实际 openloop 也按2+6分块观察。首两格重用经校验的共同 Y1；剩六格只在 action 参数上与 rolling 不同。端点总P/Q中的EV只加一次。

## 3. route/SOC 验收

把归档 `EVPlan.as_dict` 还原成 `EVPlan(arcs, arc_values=ones, p_plus_kw, p_minus_kw, u_kw, energy_kwh)`。`as_dict` 不保存 arc_values；这里只按已选整数弧还原，不解新的路线。原config由完整 `initial_ev_config` 还原，access转bool数组。

先 `check_plan` 原首轮完整八格与原rolling完整轨迹，均须在2e-6容差内满足操作约束。`execute_prefix(first_plan, original_config, 2)` 与原round1.executed逐项比较：网电、throughput、SOC、site、行驶耗能。使用 `remaining_config` 验证后六格初始SOC/site、剩余电任务、access切片、terminal要求和throughput承接。

允许的travel_slots固定为1，因此执行边界必须到达节点；若归档出现尚在途中，记录失败，不把它标成到达，不改路线。openloop原计划的suffix在同一继承状态下必须可执行；已原先检验完整八格不替代这项接口一致性检查。每次驾驶能耗只在出发格扣一次。两种最终实际电压分别报告；参考预测通过不等于全部真实路径安全。

## 4. 共同数值评价初值

完整四候选顺序为 star/clade_01/clade_02/clade_12。每个候选g的共同R/X初值取原round1的H14局部fit：

```python
initial[g.name] = {"r": np.array(firstfit["r_pu"]),
                   "x": np.array(firstfit["x_pu"])}
common_fit_seed = row["policy_seed"] + 2200
fits = fit_bank(bank, H20_method, model(3), initial=initial,
                seed=common_fit_seed, multistart=2, max_nfev=300)
```

`run_study.fit_bank` 给予候选索引 j 的 seed 是 common_fit_seed+j；两个方法相同。`likelihood.fit_candidate` 支持只传R/X的mapping。**不传原rolling-final fit，不传真R/X，不把完整firstfit对象作为initial。** 省略p/q/source字段后，latent初值按各自总计量及同一bias/master规则构建；共有H14部分因此相同，后六点只因实际动作观测不同而异。相同随机restart不意味着整个初始向量相同，但消除了方法专用warm-start规则。

两方法候选、共享线路/源的物理域、计量误差、拟合边界、重启数、nfev和停止规则相同。每个方法在不同seed交替先后拟合（例如奇偶seed），顺序规则在冻结前声明。必要firstfit缺失/不成功/损失无效时，不悄悄换默认初值；该历史保留为共同初始化不可用，两项校准分数∞，最终门控abstain。若某个新final fit失败，该方法abstain，历史仍在全分母；共同校准分数∞。所有完整latent拟合状态保存，可独立回算取得的AC目标。

## 5. 新门限和失败保留

令 L_i^{1,g} 为原共同H14拟合损失，L_i^{R,g}、L_i^{O,g} 为此次共同初值的H20损失，真实候选h_i只由 evaluator 用于离线评分。

S_i=max_{a∈{1,R,O}}(L_i^{a,h_i}−min_g L_i^{a,g})，A_i=max_{a∈{1,R,O}} L_i^{a,h_i}。

每条独立历史给一对分数。任何必要first/final fit失败、缺归档或漏真，都将两项分数设∞；不从32/48分母中删除。新配对cal32阈值分别为第32次序统计量，相对/绝对预算各.05，严格超过时拒绝。不复用 C 或原 D 阈值。首轮共享项只加入max一次，不把它复制为两个独立样本。n=32时阈值为样本最大值，任何一条必要轨迹失败都使两阈值∞；该保守性与有限power必须公开。

若新score规则及全部数值政策事前固定、cal/eval轨迹可交换，则每个规则秩界1/33，控制共同首轮与两方法final的联合误拒绝，两个规则并集界2/33≈.060606。名义预算仍.1，离散秩使实际界更保守。它依然是随机校准+新轨迹共同概率的边际结论，不是条件alpha-spending。

**旧数据只作背景：** 当前原D样本及部分评分已生成并被查看；在此后提出新score不能仅靠新hash把旧64/96重称预声明formal。当前方案已经改为新wrapper冻结后生成fresh独立32/48，旧D阈值/评分不进入新calibration。若另做旧64/96回溯配对，只能单独标retrospective，不能与本fresh分母混合。

## 6. 固定评价端点和成本

主端点预声明为 final算法唯一正确保留（必要拟合成功且真实电压安全；原实际选择回退另标）。次端点：任一共同首轮/该方法final真候选误拒绝、最终未决/错误单候选/空集、错误候选排除比例、signed margin=min_wrong L−L_true、逐边R/X相对误差、选择失败/共同初值失败/final fit失败/物理失败、route驾驶能耗及计算成本。

所有二元端点按同一独立历史配对：报告 improved/worsened/tied、配对差、保守精确区间及McNemar；失败方法的成功事件为false，原分母不删除。连续差只汇总可用配对，同时公开缺失配对数/完整分母，不能从成功子集推断总体收益。预声明主端点避免从多项统计结果中事后挑显著性；其余区间为描述性，若要作多端点确认结论需另行多重性规则。

成本分开：已保存的first选择成本双方共有；rolling额外第二轮选择成本只归rolling，openloop没有新增选址。新final共同拟合成本、重放核查成本单列，不夹进旧D计时。实际AC函数进入次数包含失败；不把returned-fit诊断当全部调用数。

## 7. 读取架构及待授权实施顺序

1. `load_frozen_input`：只读原D frozen/config以绑定原控制器；加载新paired frozen manifest及固定fresh seed集合。
2. `fresh_shadow_trace`：只在pilot获明确授权后调用原`rolling_trial`（或其原计数包装），保存fresh原始JSON/NPZ与输入SHA；不修改原D目录。
3. `decode_saved_actions`：只读选择归档还原first8与realized8；禁止入口接收truth/未来计量。
4. `audit_common_prefix_and_operations`：纯归档数值、route/SOC承接核验。
5. `evaluator_replay_path`：独立持有path/truth，重放原观测一致性闸门，再生成counterfactual openloop。
6. `build_common_initial`：只读共同H14四候选fit的R/X。
7. `fit_pair`：两方法同预算/seed最终全AC拟合，保存全部失败、latent状态和调用计数。
8. `calibrate_pair`：只有32条均已尝试且策略/数据合同固定时形成新阈值；失败以∞保留。
9. `evaluate_pair`：48条完整请求分母、固定门限、配对端点；不再影响已选动作。

实施前还须新冻结wrapper/统计器和真正loaded依赖hash、fresh输入archive hashes、预算/deadline、失败规则及输出schema，使用新policy_id。先经根代理明确授权pilot；pilot只核查重放、共同prefix、初值和记录契约，不凭结果挑动作或统计规则。正式32/48需在核查通过与根代理授权后开始。本次没有开始上述实际运行步骤。

## 8. 已实现的 runner 与仍须 pilot 验证的接口

`paired_runner.py` 已实现lazy evaluator facade、不可覆盖freeze、fresh完整影子trace保存、NPZ重放核对、原操作检查器/前缀解释器复核、R/X-only共同初始化、全fit-bank调用计数、错误/源漂移检查点、固定deadline批次/resume、新的paired collector及保存状态Gaussian目标回放。写入被限制在 D_paired_plan 子树，不能指向原D目录。collector先从保存loss/success重算三bank score再校准，防止仅信任汇总字段。

`test_paired_contracts.py`、`test_paired_runner.py` 使用纯数组与mock facade，未导入原项目模块、未调用真实生成器、AC或拟合。它们能核查读取流程/失败规则与统计记账，不能证明真正运行接口或优化收敛。实际项目接口只能由获授权的pilot验证；当前未运行actual pilot。

冻结及运行时在本目录作cwd，输出示例为 `runs_v1/frozen.json`、`runs_v1/pilot`、`runs_v1/calibration`、`runs_v1/evaluation` 和 `runs_v1/audit`。待根代理完成代码/数学审查并授权pilot后，运行：

```text
python paired_runner.py --freeze runs_v1/frozen.json
python paired_runner.py --phase pilot --manifest runs_v1/frozen.json --output runs_v1/pilot
```

pilot若满足声明的≤45s及所有必要拟合/重放/操作合同，正式阶段仍由根代理决定何时启动：

```text
python paired_runner.py --phase calibration --manifest runs_v1/frozen.json --output runs_v1/calibration
python paired_runner.py --phase evaluation --manifest runs_v1/frozen.json --output runs_v1/evaluation
python paired_runner.py --collect --calibration runs_v1/calibration --evaluation runs_v1/evaluation --output runs_v1/audit
```

科学Python须为 `D:/apps/miniconda3/envs/Topo/python.exe`。新生成只写本staging，不改变原C/D。测试的mock frozen文件位于明确命名的pytest子目录，不得把它们当实际freeze或actual H档案。主实际freeze应在代码、协议和纯测试最终稳定后创建，之后修改任何绑定源码/协议会使resume拒绝。
