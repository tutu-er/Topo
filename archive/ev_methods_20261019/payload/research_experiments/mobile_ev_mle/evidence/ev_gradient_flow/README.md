# 完整EV梯度流程证据

结论及公式见上两级目录的 `ev_gradient_flow_results.md`、`ev_gradient_flow_math.md`；恢复、重算和重跑命令见 `ev_gradient_flow_reproduce.md`。

- `raw.tar.xz`：开发v1 pilot、开发v2、36校准、16独立评价的68个原文件，包含实际科学源码快照。
- `raw.tar.xz.manifest.json`、`pack.json`、`restore.json`：原字节清单与已执行的精确恢复记录。
- `restored_replay.json`：从恢复目录重新计算阈值和评价，两个结果文件逐字节一致。
- `formal_summary.md`、`formal_summary.json.gz`、`thresholds.json.gz`、`evaluation.json.gz`：完整正式统计、方法内校准与逐历史判断。
- `negative_cases.json.gz`：保留的失败实例；开发汇总和核查记录另存。
- `*_review.json.gz`：独立AC重建、固定线路、操作和价格差分核查，包含全部失败分母。
- `launch_manifest.json`、`canonical_source_verification.json`：实际启动配置及97份科学源码字节对应。
- `threshold_saved_before_evaluation.json`：先保存校准阈值再读取评价的记录。
- `unit_tests.xml`：127项相关测试，0失败。
- `delivery_manifest.json`：本目录各文件SHA256、尺寸及主要验证计数；清单自身不循环纳入哈希。

所有计量为声明模拟场景生成的数据。单个独立历史是统计单位；同一历史的模型对、动作和重复核查不增加样本量。两候选方法仅在规划时取训练前二，最终仍检查全部四候选。归档不自动执行其中源码。
