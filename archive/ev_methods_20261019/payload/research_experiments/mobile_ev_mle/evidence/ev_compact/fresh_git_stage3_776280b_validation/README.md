# 阶段 3：Git 对象导出与离线审查实测

验证提交：`776280b6845bd132b53a7bb8a2cba90db752fad6`。

代码全部通过该提交的 `git ls-tree` 与 `git cat-file --batch` 导出，没有从工作树复制代码。每个对象的 Git blob SHA-1 和导出字节 SHA-256 已核对。新目录为 `D:/0-github_workspace/Topo/outputs/ev_compact_stage_20261015/fresh_stage3_776280b`，包含 416 文件、28,469,389 字节；其中有 72 冻结科学源及 ensemble v2 完整 4 历史 slim 包，不含原 raw 或校准/评价数据。

## 实测步骤

1. `export_git_stage3_objects.py` 从 Git 对象导出代码、必要协议与开发交付数据，保存逐文件导出清单。
2. 在新导出目录执行默认 `ev_compact_source_restore`：31/72 已一致，41/72 仅换行不同，真实内容变化及缺失均为 0。随后独立核对所有 416 个原导出文件仍与 Git 对象字节一致，证明默认检查没有修改它们。
3. 显式执行 `ev_compact_source_restore --restore-line-endings`：仅新 D 目录的 41 个文件恢复原换行，72/72 与冻结快照完全一致；其余 375 个导出文件与原 Git blob 保持一致。C 工作树未修改。
4. 从新目录执行 `ev_compact_slim_review`，完整 ensemble 4/4 历史、44/44 方法行通过。复算 415 个历史/实际 AC 拟合状态、44 个实际未来物理响应、158 个操作账目及 52 项继承分数。预测部分只校核 3648 个拟合摘要及 2736 个任务摘要。
5. 独立 `python -I verify_fresh_stage3_export.py` 导入核查：58 个项目模块全部来自新 Git 对象目录，无旧工作树模块，无 `gurobipy` 导入。

源核对和审查入口分别为：

```text
python -m research_experiments.mobile_ev_mle.ev_compact_source_restore --report source_audit_before.json
python -m research_experiments.mobile_ev_mle.ev_compact_source_restore --restore-line-endings --report source_audit_restored.json
python -m research_experiments.mobile_ev_mle.ev_compact_slim_review --directory research_experiments/mobile_ev_mle/evidence/ev_compact/delivery_development_v2/development_ensemble_v2 --output fresh_slim_review.json
```

这些命令从新导出根运行，报告使用新文件名。导出器和最终核查器的源码也随此证据保留；其宿主路径写在脚本常量中，复用到另一机器需显式指定对应仓库及空的新目录。

## 文件及结论范围

`git_object_export_manifest.json` 绑定全部原始 Git 对象；`source_audit_before.json` 和 `source_audit_restored.json` 保存换行预检/恢复；`fresh_slim_review.json` 保存离线实际 AC/操作审查；`fresh_git_object_validation.json` 绑定工具、运行环境、导入来源及全部报告哈希。

没有新拟合、优化或策略选择。预测完整状态、QP 和全域见证证书沿用 slim 审查的省略范围。该结果验证提交中的代码与交付数据足以离线重放**已保存的实际 AC/操作范围**，不将 slim 包作为完整 raw 的替代。
