# 匹配 RNJ 阈值调参：2026-09-10 事后探索性补充

**这项补充是在查看正式结果后新增的公平性诊断，不属于预注册实验，也不构成对新假设的独立确认。** 它回答一个具体问题：主实验的 NJ 使用验证集选择四个阈值，而默认 RNJ 固定使用 0.16；两者结果差多少来自这种调参机会不一致？

所有数据沿用主运行 `advantage_stress_20260910` 的 `inputs.npz` 和 `metadata.json`，没有重新生成观测、增加数据或改变原始结果。覆盖原计划的 **72 个 synthetic 条件和 36 个 AC 条件，共 108 个条件**。补充运行在主运行完成后启动，以免影响主实验的限时求解。

[补充协议](../outputs/advantage_stress_20260910/matched_tuning/protocol.json)、[逐条件结果](../outputs/advantage_stress_20260910/matched_tuning/results.csv)、[配对差值](../outputs/advantage_stress_20260910/matched_tuning/paired_differences.csv)、[阈值计数](../outputs/advantage_stress_20260910/matched_tuning/threshold_counts.csv)、[完成记录](../outputs/advantage_stress_20260910/matched_tuning/completion.json)。

## 匹配协议与核验

使用与主实验 NJ 相同的 RX75 回归和原端固定树 L1 拟合。RNJ 候选阈值为根深度中位数乘 **0、0.04、0.08、0.16**；每个候选只用 train 拟合 R/X 与场景截距，再用 validation MAE 选择。验证时保持训练截距，不根据验证样本重估。精确平局选择较小阈值，与主实验 NJ 的规则相同。

选择函数只接收 train、validation、终端标签、观测根标签与每候选求解预算。所选阈值和模型固定以后，才传入 test 和真拓扑做最终评估。四个候选的验证损失、结构、求解状态与耗时均保存在对应 `matched_tuning/jobs/<job_id>/selection.json`；选定参数保存在 `selected_model.npz`。

实际完成 **108/108 条件，0 条件失败；432/432 候选成功，432 个固定树 LP 均满足现有求解最优性证书检查**。这些证书只表明该固定树参数拟合达到其优化目标，不保证真拓扑正确。

为核验缓存恢复和参数拟合口径，对每个条件的 **0.16 候选**，比较其 validation MAE 与主实验 `rnj_fixed_tree_lp` 已记录的 validation MAE：

```text
条件数：108
max |补充 RNJ(tau=0.16) validation MAE - 主实验固定树 LP validation MAE| = 0.0
```

逐条件损失完全一致，比仅检查总体均值更强。这项核验支持观测数组、标签与顺序、回归、默认 RNJ 结构、原端 L1 拟合、验证截距口径相同。另有 [16 个 mock 协议测试](../tests/test_matched_rnj_tuning.py)，覆盖缓存恢复、失败保留、选择不接触 test/truth、平局规则以及主运行未完成时禁止计算。

## synthetic 与 AC 给出相反结果

下表是逐条件 rooted-clade F1 的等权均值。默认 RNJ 与默认 RNJ 固定树 LP 的结构相同，因此 F1 相同；预测误差比较则优先使用固定树 LP，以控制参数拟合方式。

| 数据层 | 条件数 | 默认 RNJ | 匹配调参 RNJ | 主实验验证调参 NJ |
|---|---:|---:|---:|---:|
| synthetic | 72 | 0.841468254 | **0.982533670** | 0.977904040 |
| AC | 36 | **0.723805802** | 0.668703634 | 0.691729467 |

| 配对差值方向 | 平均 F1 差 | 胜 / 平 / 负 |
|---|---:|---|
| synthetic：匹配 RNJ − 默认 RNJ | +0.141065416 | 21 / 51 / 0 |
| synthetic：匹配 RNJ − 验证 NJ | +0.004629630 | 4 / 67 / 1 |
| AC：匹配 RNJ − 默认 RNJ | -0.055102168 | 3 / 15 / 18 |
| AC：匹配 RNJ − 验证 NJ | -0.023025833 | 10 / 12 / 14 |

synthetic 的精确恢复数从默认 **48/72** 提高到 **65/72**，与验证 NJ 的 65/72 相同。AC 的精确恢复数为默认 1/36、匹配 RNJ 0/36、验证 NJ 4/36。因此，这项补充不能得出“给 RNJ 调参后就普遍更优”。

## 合成层：主要差距确实来自固定阈值

synthetic 的 72 个条件中，**71 个选择 0，1 个选择 0.16，没有选择 0.04 或 0.08**。唯一选择 0.16 的条件属于 outliers。各压力的平均 F1 为：

| 压力，每组9例 | 默认 RNJ | 匹配 RNJ | 验证 NJ |
|---|---:|---:|---:|
| reference | 1.000000000 | 1.000000000 | 1.000000000 |
| low_sample | 0.989898990 | 1.000000000 | 1.000000000 |
| high_noise | 0.989898990 | 1.000000000 | 1.000000000 |
| pq_collinear | 1.000000000 | 1.000000000 | 1.000000000 |
| terminal_correlated | 0.989898990 | 1.000000000 | 0.989898990 |
| common_noise | 0.957575758 | 1.000000000 | 1.000000000 |
| outliers | 0.772727273 | 0.860269360 | 0.833333333 |
| weak_internal | 0.031746032 | **1.000000000** | 1.000000000 |

weak_internal 是最强的机制证据：默认阈值损失几乎全部内部结构，匹配调参后全部恢复；它不是 RNJ 排序本身无法恢复这些短内部边。调参后的 RNJ 与 NJ 在 67/72 条件打平，原先 0.841 对 0.978 的大差距不能解释为 NJ 对这个合成库有固有算法优势。

这也不证明零阈值可以普遍替代当前默认值。这里的 synthetic 只覆盖三种二叉树族、8 个终端；主实验 geometry 已显示，含噪多分叉与星形树在零阈值下容易被错误细分。P/Q 共线时恢复拓扑也不等于 R 和 X 分别可识别。

## AC 层：预测 MAE 与结构 F1 是不同目标

AC 的阈值选择数为：0 有 **10** 例，0.04 有 **6** 例，0.08 有 **5** 例，0.16 有 **15** 例。三个压力下，匹配调参 RNJ 的平均 F1 都低于默认 RNJ：

| AC压力，每组12例 | 默认 RNJ | 匹配 RNJ | 验证 NJ |
|---|---:|---:|---:|
| reference | 0.893554131 | 0.808025996 | 0.825584181 |
| low_sample | 0.694027236 | 0.647261283 | 0.676413831 |
| high_noise | 0.583836038 | 0.550823622 | 0.573190389 |

然而，在相同原端固定树 LP 参数拟合口径下，匹配 RNJ 相对默认固定树 LP 的**平均独立 test MAE 差为 -4.527092777×10^-6**，即平均预测误差下降；与此同时 F1 下降 0.055102168。MAE 单位对应平方电压降目标，不能将该数值直接写成电压百分比。

这不是测试集参与选参导致的泄漏。它表明，在当前 AC 生成、回归近似和噪声条件下，能改善电压降预测的结构复杂度变化，未必改善真实 clade 恢复。验证 MAE 是可部署的选择信号，但没有获得“最小验证 MAE 等价于最大拓扑 F1”的保证。

相对验证 NJ，匹配 RNJ 的 AC 平均 test MAE 差也略低（-1.198022391×10^-6），而 F1 平均低 0.023025833。这里同样需要分别报告预测与拓扑目标，不能只挑一个评价指标宣布优越性。

## 证据限制

- 这是查看原结果后增加的探索性消融，复用同一个测试库。下一轮独立、预先固定协议的实验才能检验这些新观察是否稳定。
- 配对区间保存在 [paired_comparisons.csv](../outputs/advantage_stress_20260910/matched_tuning/paired_comparisons.csv)，按照原 tree/AC replicate 整群重采样。每个 synthetic 压力只有9个群、AC压力只有12个群；它们是该固定库的描述性区间。AC 三个压力中，匹配 RNJ−验证 NJ 的 F1 差区间都包含零，不支持显著或普遍的排序宣称。
- 求解策略、单候选预算及返回解接受方式与原 NJ 相同，证书状态另外保存。没有使用真值过滤候选，也没有因结果不好追加阈值。
- 主评价是结构 clade 集，即使某个拟合边权为零也保留结构；正权 clade 另行保存。零权分支具有结构歧义。
- 耗时包含 RX75 回归及全部四候选的重建、拟合和验证，排除磁盘与报告；补充计算发生在较晚运行中，机器并发负载不同，因此它不是严格控制的运行速度对比。

## 严格复现：使用运行时源码快照

主实验结束后，工作区的实验审计元数据代码发生过修改。**当前工作区源码哈希与主运行协议不再完全相同，直接在 live 目录复跑会被严格哈希门禁拒绝。此门禁保留，不能通过删检查或跳过哈希来复现。**

主运行时源码保存在 [source_at_run.zip](../outputs/advantage_stress_20260910/source_at_run.zip)。已逐项验证：归档中 **37/37 个源码文件**与主运行 `protocol.json/source_sha256` 一致，零缺失、零不匹配；归档顶层就是 `rnj_wzzt/` 与 `experiments/`。

实际执行的补充脚本另已逐字节备份为 [supplement_script.py](../outputs/advantage_stress_20260910/matched_tuning/supplement_script.py)。其 SHA-256 与补充 `protocol.json/script_sha256` 一致：

```text
8d35463cf967e38bf573a04029e8b9b5517e23fffb5b60e7e680537fde4de24e
```

在 PowerShell 中执行以下命令。两个 replay 路径必须尚不存在；命令不会覆盖原始数据或结果。这里给出复现步骤，本次文档收尾没有再次启动计算。

```powershell
$sourceRun = 'D:\0-github_workspace\Topo\rnj_wzzt_core\outputs\advantage_stress_20260910'
$replayCore = 'D:\0-github_workspace\Topo\rnj_wzzt_core\outputs\advantage_stress_20260910_replay_source'
$replayOutput = 'D:\0-github_workspace\Topo\rnj_wzzt_core\outputs\advantage_stress_20260910_matched_replay'
$pythonExe = 'C:\Users\23761\AppData\Local\Programs\Python\Python312\python.exe'

if (Test-Path -LiteralPath $replayCore) { throw '请选择尚不存在的 replay 源码目录' }
if (Test-Path -LiteralPath $replayOutput) { throw '请选择尚不存在的 replay 结果目录' }

Expand-Archive -LiteralPath (Join-Path $sourceRun 'source_at_run.zip') -DestinationPath $replayCore
New-Item -ItemType Directory -Path (Join-Path $replayCore 'scripts') | Out-Null
Copy-Item -LiteralPath (Join-Path $sourceRun 'matched_tuning\supplement_script.py') -Destination (Join-Path $replayCore 'scripts\run_matched_rnj_tuning.py')

$env:PYTHONPATH = 'D:\0-github_workspace\Topo\.codex-rnj-deps'
& $pythonExe (Join-Path $replayCore 'scripts\run_matched_rnj_tuning.py') --source $sourceRun --output $replayOutput
if ($LASTEXITCODE -ne 0) { throw '复现未成功，请查看上述错误；保留哈希门禁' }
```

脚本会以解压目录为核心代码根目录，重新计算全部源码哈希，与原运行协议比对后才拟合；数据继续读取原 `inputs.npz`，新结果写到独立 replay 目录。每次运行仍会保存新协议、候选与失败明细以及完成记录。软件/求解器版本、硬件和限时环境变化可能影响数值舍入与耗时；本次归档保证的是代码与输入可核验，而不是对任何环境承诺逐位相同输出。

