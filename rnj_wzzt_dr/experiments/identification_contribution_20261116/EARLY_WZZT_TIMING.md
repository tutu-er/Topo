# 早期 WZZT 的耗时、初值贡献与停止原因

核查日期：2026-11-16。只读取原协议、执行源码快照和已保存结果；没有重新拟合、求解或运行潮流。

**早期确实出现过单步耗尽预算，也长期存在“预试初值已经很好，整数求解主要用来缩小界差”的情况。但这两件事没有同时发生在所有版本上。** 10 月 27 日的 60 秒单步实验有明确超时；10 月 31 日 ordinary 的强 RNJ 初值多数没有再被原生搜索改善，但没有一次达到 TimeLimit；长预算 pure WZZT 则几乎每步都实际改善初值。11 月的主 240 输入也没有近期 8 输入中普遍出现的单步耗尽预算现象。

本文的 main108 按原 `rnj_four_methods_20261031/catalog.json` 中 `panel == "main"`，并与协议 `jobs` 取交集。旧 120 中另 12 个输入不纳入下表；它们与已经退出默认评估的 pressure160 是不同目录、不同清单。本文不使用 pressure160 支持效果或参数结论。

## 1. 各批实际花了多久

时间单位均为秒。“原生时间”相加原记录的求解事件时间；整流程还包含前处理、固定结构拟合、候选评分等。表内 Status 2 只表示求解器达到该次设置的容差，证书对象是当前子问题。

|版本及输入数|协议：单步 / 整体预算|实际整数求解次数|TimeLimit 次数|最大单次原生时间|原生时间合计|整流程合计 / 每输入均值|
|---|---:|---:|---:|---:|---:|---:|
|10/27 pure WZZT，80|60 / 180 结构搜索|664|20|60.093|4,714.153|5,802.141 / 72.527|
|10/27 RNJ 小块 + WZZT，80|60 / 180 结构搜索|667|13|60.077|4,390.578|6,341.231 / 79.265|
|10/31 ordinary，main108|30 / 300|257，另 13 次直接判结构完整|0|28.090|1,408.274|1,485.854 / 13.758|
|10/31 long pure WZZT，main108|3,600 / 14,400|1,034|0|111.168|9,510.413|9,759.440 / 90.365|
|11/01 matched before，main108|3,600 / 14,400|531|0|2,850.126|27,102.924|27,461.892 / 254.277|
|11/01 matched after，main108，含恢复分支|3,600 / 14,400|723|0|2,863.017|53,836.010|54,307.188 / 502.844|
|11/12 main240，MSE `000`|3,600 / 14,400|1,923|0|134.278|13,572.740|14,195.969 / 59.150|
|11/12 main240，MSE `100`|3,600 / 14,400|1,924|0|118.154|12,247.167|13,092.463 / 54.552|
|11/12 main240，MSE `101`|3,600 / 14,400|2,164|0|142.583|15,370.927|16,281.829 / 67.841|
|11/12 main240，MSE `111`|3,600 / 14,400|1,640|0|668.094|20,184.698|21,024.038 / 87.600|

计时口径保留原字段：10/27 使用 `elapsed_seconds`，ordinary 使用 `elapsed_seconds`，long pure 使用 `cumulative_seconds`，matched 和 main240 使用 `total_seconds`。10/27 的 180 秒仅是结构搜索预算，不能直接拿总时间与它相减认定超时；后处理和启动开销另外存在。上述每输入均值可用于了解各批运行规模，但不同数据量、目标函数、算法阶段和机器负载的跨批比值不是加速实验。

10/27 的 160 个有效结果采用两条已记录修复结果替换对应程序错误的首次结果，原错误仍保留；另四条控制器中断记录保存在历史目录，不加入完成记录的时间合计。这里统计的是有效完成尝试，不是连同失败重试在内的全部历史 CPU 消耗。主108及 main240 的上述方法结果均齐全。

## 2. 长预算 matched 的时间主要去了哪里

matched 的 `before` / `after` 是包括重建、退出和恢复的完整流程。不能把它们的总耗时全部称为 W 扩展耗时，更不能把 after 总耗时当成后来 `zero_only_own` 单独执行成本。

|main108 阶段|before：次数 / 原生秒|after：次数 / 原生秒|after 最大单次秒|
|---|---:|---:|---:|
|主路径 W 扩展|326 / 2,969.974|344 / 3,282.621|95.262|
|主路径小块联合重建 T|97 / 531.423|118 / 700.755|36.740|
|主路径最终重选 E|108 / 23,601.527|108 / 24,066.038|2,863.017|
|恢复分支 W/T|0 / 0|89 / 534.556|36.893|
|恢复分支 E|0 / 0|64 / 25,252.040|2,444.294|
|合计|531 / 27,102.924|723 / 53,836.010|2,863.017|

after 在 32/108 输入上运行了 release 和 masked 两个恢复分支。因此 64 次额外 E 占用了约 7.01 小时；全部 E 合计 49,318.078 秒，占该批原生时间约 91.6%。W 并没有在这里一轮用完 3,600 秒。最长完整输入为 before 2,874.178 秒、after 7,609.983 秒，均低于 14,400 秒总上限。

## 3. 哪些结果来自初值，哪些来自原生搜索

有保存实际 MIP-start 审计值的批次，逐事件计算

\[
g_{\mathrm{native}}=f(\text{实际传入的初值})-f(\text{原生返回且通过审计的点}).
\]

两个值必须来自同一个已编码模型。这里用 `warm_start.objective` 和 `audit.objective`；E 记录没有后一字段时用 `final_objective`。以绝对差 `1e-10` 区分改善和数值相同。这个计数衡量训练目标的原生搜索贡献，并不表示拓扑一定改善；目标相同也不要求所有系数逐值相同。

|main108 阶段|可比较实际初值的次数|原生目标改善 > 1e-10|目标数值相同|比初值更差 < −1e-10|
|---|---:|---:|---:|---:|
|ordinary W|257|32|225|0|
|long pure WZZT|1,034|1,033|1|0|
|matched before W|326|95|231|0|
|matched before T|97|92|5|0|
|matched before E|108|63|45|0|
|matched after W|344|101|243|0|
|matched after T|118|104|14|0|
|matched after 主路径 E|108|62|46|0|
|matched after 恢复分支 W/T|89|57|32|0|
|matched after 恢复分支 E|64|45|19|0|

ordinary 的 225 次目标相同的原生求解合计 1,248.184 秒，占该批原生时间约 88.6%；32 次确有改善的求解合计 160.091 秒。**强 RNJ 预试点以后再花时间确认或缩小界差，早期已经存在。** 但 long pure 没有 RNJ 结构池、整树初值或单连接预试，1,033/1,034 次都改善了实际初值；这部分确有大量原生结构搜索贡献。

原快照给出了机制：

- ordinary `source/research_experiments/joint_five/search.py:14–45`：先对兼容池中的连接固定结构拟合，取最佳 `warm`，再把它完整写入扩展 MILP；`best = warm`，只有原生点更好才替换。只要预试点相对当前路径过门槛，即使原生搜索没有改善它，也能接受这一步。原始调用 `run.py` 给的是 total=300、per_mip=30、relative=.001；不是当前 MSE `100`。
- long pure `source/research_experiments/rnj_union_long/model.py:136–168`：只从单叶和全节点公共项起步，`any_extension(..., ())` 不用 RNJ 池；每步把当前点作为可行初值，再求未知新连接。`:52–100` 保存实际初值、返回点、目标、界、事件与时间。
- 10/27 `source/research_experiments/wzzt_solver_tuning/bounded_solver.py:27–54` 每步重新建立原生模型，没有设置 `Start`。因此不能把其超时归因于“保留一个已经很好的 RNJ 数值热启动”。该批只用 RNJ 小块改变结构起点。
- main240 的 MSE 快照保存了 `before_score`、原生返回点和闭合后目标，但没有保存实际 MIP-start 的完整目标及预试耗时。这里不能把 `before_score - after_score` 全算成原生贡献；它还包含预试和后续固定结构拟合的作用。

## 4. 超时、未决和“无增益”不是同一个字段

10/27 pure 的状态为 498 次 Status 11、146 次 Status 2、20 次 Status 9；RNJ+WZZT 为 514、140、13。Status 11 来自改善上下界满足规则后的主动 callback 停止，不能记作 TimeLimit。有效结果分别为 60/80、67/80 complete，20/80、13/80 incomplete；未决结果没有从分母删掉。

ordinary 主108 的停止原因：76 `bound_excludes_gain`、13 `structurally_complete`、19 `no_gain_unresolved`。虽然所有真正启动的 MILP 都是 Status 2，仍有 19 条没有用保存的界证明所需的无增益结论。求解器的相对/绝对容差与算法的接受门槛是两层条件。

long pure 主108：107 `bound_excludes_gain`、1 `structurally_complete`。matched 主路径 before 为 97 `joint_tail_gap_met`、11 `coarse_no_gain_or_unresolved`；after 为 95、13。after 会在零权释放后继续阶段，所以主路径有 118 次 T，而不是每输入最多一次。

main240 的 `000`、`100`、`101` 都有 240 个 `no_gain_or_unresolved`。这是源码在没有接受新的 W 步骤时使用的合并标签（同哈希 `algorithm.py:304`），而上述原生求解全部 Status 2。应报告“路径停止于无接受增益，标签未细分”，不能据标签把 240 个都解释成超时或全部数学未决。`111` 则有 214 `tail_eligible` 和 26 `no_gain_or_unresolved`。

10/27 还有一个必须避开的旧字段问题：worker 在保存事件时把 `info['bound']` 覆盖成系数上界 2、4 等（原 `worker.py:127`），不能拿该字段当目标下界。原始 callback 的 lower/gain_possible 等另有记录；本表不使用被覆盖的 `bound` 判断最优性。

## 5. 与近期 120 秒 / 10 秒单步诊断怎样对应

近期源为 `outputs/identification_stepcap_20261116_v1/evaluation/report.json`，8 个输入是两网络、各 4 次历史噪声重复；训练为 8 天，不能与旧 N=24/48/96 的 MAE 主108或 train64 的 MSE main240当成同难度计时。

|近期方法|W 次数 / 接受|TimeLimit|原生秒|预试秒|原生改善实际初值次数|停止|
|---|---:|---:|---:|---:|---:|---|
|原 `100`，per_mip=120|18 / 18|8|853.398|8.877|1|8/8 forward_budget|
|同配置仅 per_mip=10|58 / 50|58|582.421|63.638|0|8/8 no_gain_or_unresolved|

原 120 秒设置下 Pengwah 的四个输入各只进行一次 W：预试约 1.29–1.51 秒，原生约 106.16–106.41 秒，没有原生改善，末端 gap 仍为约 34.88%–53.24%。缩短单步后能够推进更多路径点，流程均时由 108.113 秒降至 81.501 秒，在这 8 个输入上最终验证及累计全支持恢复从 0/8 到 8/8；但同样 .08 收缩的单 RNJ 参考本身也是 8/8，均时约 .314 秒。因此本轮能支持“预算分配妨碍了完成路径”，尚不能用这 8 例证明额外整数搜索优于同处理 RNJ。

近期另外两步出现的 −.003、−.003 假负原生增益，来自审计对实际编码初值计数与清理后活动支持计数混用；根任务正在独立修正这两个记录的表示。数学模型和传入 Start 不因此改变。历史普通 ordinary 是未加支持数量罚的 MAE 模型，没有这个特定的 .003 计数差。历史 matched 的有罚版本已经显式保存 `protected_count_cost`（`rnj_count/engine.py:96–100`），原生 objective 不含这一固定项，`decoded_score` 含固定项；直接跨字段相减同样会产生人为差值。本核查统一比较同模型的 warm/audit，未观察到超过 1e-10 的负原生增益。

main240 的 MSE 编码也有“保留旧 proper 数量 + 新 active”的固定计数（同哈希 `optimization.py:281–288`）。这是旧源码已有的表示方式；因为当时没保存实际 Start 目标，不能宣称近期发现的两次计数误报已经在旧 240 中发生，也不能宣称已排除。

## 6. 原始来源、版本与复算方式

所有私有路径均相对仓库 `D:/0-github_workspace/Topo`。本次核查了下列协议、相关源码 SHA 和原结果字段，未拿当前修改后的代码替代旧执行版本。

|记录根|协议 SHA256|本次结果范围|
|---|---|---|
|`outputs/wzzt_systematic_20261027_round1`|`8d028fe6b22ad1c74d872129b19a84417c54f62c814fb997dfed6a52df8d0c8d`|协议160个 WZZT/RNJ+WZZT任务；对应 repairs 优先，否则 jobs|
|`outputs/rnj_four_methods_20261031`|`97c43d6d8076964b4e56fb2b32208988f89e02fa4c83e6e634780dca3f22c658`|`evaluation/main*/ordinary/result.json`，108份|
|`outputs/rnj_union_long_20261031`|`64e70cfbd16c6c7cd6a293f27b585dac5516a4fe4c0cdd0614a9f45e752b9456`|`jobs/main*/pure_wzzt/result.json`，108份|
|`outputs/iterative_exit_matched120_20261101`|`10e1eca6efe506f55f41118cf5af03293745f6b2d6dd6a3172fdb7d70b2ba4d7`|`jobs/main*/{before,after}/result.json`，216份|
|`topology_identification_standalone/outputs/large96_main_20261112_001`|`ceac935650e280105c507e26ce780191fde4340cdd3a8fd3ba29028f8a6a8fc4`|`runs/input*/{000,100,101,111}/result.json`，960份|

与各自协议/manifest一致的主要源码：ordinary `joint_five/search.py` 为 `f03085a2a21b5563ff2621c8f695c5fcd17300415d01fbc805b4f11c537933dd`；long pure `rnj_union_long/model.py` 为 `42d1ee995669fe54f50d9920a6648d92d3616833609af573042c8cae5cd934d3`；matched after `rnj_count/engine.py` 为 `6154ca0c42b370dee5eebdac0d266d52c6dafce824fd04fd34277295fd234fee`。10/27 原/修复 `bounded_solver.py` 同为 `7be797d84c0a121982308053a6aa50616fd2f45cd9b38d1233217ea9ae45eabe`；原/修复 worker分别为 `a85293e4a92a224aafb8d6ab3d29296d386991c8f42a00a6ff1b9acea3686994`、`8c9db0b2a7b4cac798a4491ccddf987502dc589e93fc3f08445b2232a4b065a6`，分别匹配原冻结清单和修复清单。

main240 的 `optimization.py` SHA 为 `c11b1034a03694838abc05349616f0930065c4db47e24aa2ec2d2bb774ebf244`、`algorithm.py` 为 `06152967c0e939be11cd51e3fbf592f162655c9b92e032f9b28aaa59b4a4e20d`。本次阅读的是 `outputs/history10_rx_20261116_v1/source_snapshot/rnj_wzzt_dr/identification/` 中与该协议逐字同 SHA 的副本；没有假设目录较新就代表算法不同。

结果集合校验：对结果路径排序，构造 `[{"path": 相对记录根的POSIX路径, "sha256": 文件SHA}, ...]`，使用 `json.dumps(..., sort_keys=True, separators=(',', ':'))` 后取 SHA256。按上表顺序得到：

```text
160: 5f4b8a285b82dab318856e53fcc607122e5fc6e1cd1aa6c5e7cd3e010a335794
108: 06aa08fe2c38f9799e0379b061c91d8dbd08fe1f3e56b49490d8920809dd945b
108: 69ac8521ea598a61bee899f2529d75af9e1e53ec58e8cafb47eef5c7e306dd96
216: c8efa28e14403836df6af36b913418b0365958eee376619ce5d7213d0b4227c6
960: 1d2c818415b23bc75a109e0ad92f4baeefa9d8a243d609729410faa51ba7e5cf
```

逐层复算字段：ordinary/long pure 为 `trajectory.attempts`；matched 主路径为 `prefix.attempts`，主 E 为 `branches.baseline.exit_event`，额外分支为 `branches.release/masked.trajectory.events` 及各自 `exit_event`，不能再重复累加 baseline.trajectory；main240 为包装层 `result.events[*].solver`。近期报告 SHA 为 `d0f15297a639b6971bceebaa98e7981d9a55e2e222faba9bf8a786a243caa962`，其中初值表示修正应以随后独立收据为准，不覆盖原记录。

本核查支持的直接改进方向是：按阶段分别记录预试、实际 Start、原生返回和闭合后值；对强初值设置合理单步预算，并单列证明上下界花费的时间。现有证据不支持“早期 WZZT 的搜索一直都没有作用”，也不支持把所有历史慢例归成同一个计数问题。
