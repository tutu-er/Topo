# 正式评价 12 历史：便携交付与离线复查

本目录保存已冻结策略的正式评价交付。独立单位是历史：12 个历史、每历史 11 方法，共 132 方法条件；任务、预测情景和 QP 数不增加历史样本量。恢复和复查只读取保存状态，不重新拟合或选择动作。

## 交付内容与已完成验证

| 文件 | 恢复后的文件数 | 内容 |
|---|---:|---|
| `checks.tar.xz` | 6 | `freeze.json`、72 源字节快照、源预检、完整原 raw 的 AC/操作与 QP 审查报告、审查覆盖及哈希核对 |
| `analysis.tar.xz` | 7 | 5 张逐历史 CSV，以及 `summary.json`、`summary.md` |
| `evaluation_slim.tar.xz` | 16 | 12 份显式 slim trace、freeze、72 源字节快照、交付清单及省略范围 |

三包合计 31,613,148 字节。全部已恢复到新的目录，6/6、7/7、16/16 文件与归档前的原压缩字节完全一致；细节分别见三个 `*_verification.json`。恢复后的 slim 已通过 12/12 历史的实际 AC/操作复查，见 `restored_slim_review.json`。从 raw 与 slim 生成的 5 张 CSV 共 14,690 行逐字节相同：方法 132、配对 168、候选 1,056、任务 13,032、主问题 302；见 `delivery_validation.json`。

完整 raw 共 733,260,981 字节，仅保留在原 D 阶段目录，未改写、删除或加入此交付包。原路径、压缩文件大小及 SHA-256 保存在恢复后的 `delivery_manifest.json` 和本目录 `delivery_validation.json`；其他机器无需存在这些 D 路径。已完成的完整 raw AC/QP 审查保存在 `checks.tar.xz`，重新执行相同范围的完整审查仍需原 raw。

## 在任意检出路径恢复

以下命令均从 **Topo 仓库根目录**运行；`python` 指向已安装项目依赖的 Python 环境，路径相对于当前仓库，不依赖原机器的 D 盘。使用包含交付工具和科学源的完整检出，不把本目录单独作为 Python 项目运行。目标目录和 `--report` 文件必须尚不存在；重复验证时换一个新目录名。

```text
python -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_evaluation_v2/checks.tar.xz replay_evaluation/checks --require-exact --report replay_evaluation/checks_restore.json
python -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_evaluation_v2/analysis.tar.xz replay_evaluation/analysis --require-exact --report replay_evaluation/analysis_restore.json
python -m research_experiments.mobile_ev_mle.ev_original_evidence_archive restore research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_evaluation_v2/evaluation_slim.tar.xz replay_evaluation/slim --require-exact --report replay_evaluation/slim_restore.json
```

这是带内部清单和编解码记录的专用归档，应通过上述工具恢复，普通 tar 解包只得到内部载荷。工具检查每份载荷和恢复文件的哈希。`--require-exact`要求原压缩字节也一致；不同 Python/zlib 运行时可能出现载荷相同而 gzip 字节不同，此时命令返回非零并在报告中区分两种一致性。本次验证使用 CPython 3.11.15、zlib 1.3.2。

## 核对 72 源与换行

先进行只读检查，显式指定刚恢复的评价快照，避免工具默认使用开发包：

```text
python -m research_experiments.mobile_ev_mle.ev_compact_source_restore --directory replay_evaluation/checks --source-root research_experiments/mobile_ev_mle --repository-root . --report replay_evaluation/source_audit_before.json
```

检查要求 72 个源码文件已经存在于当前检出，并验证快照字节、外部 freeze 和当前文件。Git 的换行处理可能使内容相同而字节哈希不同。已有全新 Git 对象导出验证记录在相邻的 `../fresh_git_stage3_776280b_validation/`：31/72 原字节一致、41/72 仅换行不同，显式恢复后 72/72 完全一致；这是开发交付上的恢复工具验证，评价包另有本目录的精确恢复记录。

若只读报告确认 `content_changed_count=0`、`missing_count=0`，且差异仅为换行，可在用于复查的独立检出中执行：

```text
python -m research_experiments.mobile_ev_mle.ev_compact_source_restore --directory replay_evaluation/checks --source-root research_experiments/mobile_ev_mle --repository-root . --restore-line-endings --report replay_evaluation/source_audit_restored.json
```

该选项只恢复经过整批预检的换行差异，保留快照中的原始字节；真实内容变化或缺失会阻止整批恢复。源码快照作为数据核验，不自动执行。科学源全部匹配后再进行下面的复查。

## 复查保存的实际 AC、操作与分析表

```text
python -m research_experiments.mobile_ev_mle.ev_compact_slim_review --directory replay_evaluation/slim --output replay_evaluation/slim_review.json
python -m research_experiments.mobile_ev_mle.ev_compact_analysis --input evaluation=replay_evaluation/slim --output replay_evaluation/reanalysis
```

第一条应核对报告中的 `success=true`、`available_independent_histories=12`、`audited_histories=12` 及空 `review_errors`。第二条在 `replay_evaluation/reanalysis/evaluation/` 生成 5 张 CSV 和汇总；可逐字节对照 `replay_evaluation/analysis/` 中的同名 CSV。分析器为交付外圈工具，其版本哈希保存在 `delivery_manifest.json`；更换分析器版本后，应解释新旧输出差异，不修改归档内的原分析记录。

slim 保留历史、真值与计量精度，全部实际 noisy/noiseless 计量及拟合、操作路径与资源账本、H-only 各任务分数和失败、各轮求解界/间隙/规模/耗时、逐动作 QP 区间和状态，以及每轮最佳 QP 的完整参数和残差。slim 省略完整预测拟合向量、非最优 QP 参数/残差向量、重复 LP 向量和完整见证库。因此上面的离线复查覆盖保存的实际 AC/操作与分析算术，不能仅凭 slim 独立重建完整 QP 或所有全域见证；这些阶段的完整保存状态在本地 raw，既有复核报告在 checks 包。

统计阈值和决策另见相邻的 `../risk_calibration_before_evaluation.json.gz`、`../risk_evaluation_v2.json.gz` 及 `../risk_final_independent_audit.json`。这三份文件不在上述三包内。校准与评价沿用同一完整 calibration 对象；统计结论适用于所声明的模拟分布，取得似然、设计分数及其排名分别保留原含义。

无需原 D raw 的统计算术复查命令为：

```text
python -m research_experiments.mobile_ev_mle.ev_compact_risk_records_replay --before research_experiments/mobile_ev_mle/evidence/ev_compact/risk_calibration_before_evaluation.json.gz --final research_experiments/mobile_ev_mle/evidence/ev_compact/risk_evaluation_v2.json.gz --output replay_evaluation/risk_records_only_replay.json --readme research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_evaluation_v2/README.md
```

它只读取两份小记录，独立重建单方法和两主族的完整校准对象及全部决策，检查 bundle/calibration 哈希、校准对象不变及历史分离。该命令不访问记录中的 raw 或旧审查路径，也不验证 raw 的计量或拟合。真实36/12记录的便携重放证明见本目录 `risk_records_only_replay.json`；它绑定本版 README。旧 `README_validation.json` 保留更新前 README 的验证及哈希，作为历史记录。
