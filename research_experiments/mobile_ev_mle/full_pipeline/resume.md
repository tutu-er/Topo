# 本批已完成：复现入口

本批136个条件已全部结束，主分析、物理审计、完整树F1、候选覆盖与证据恢复均已完成。不要重复启动controller或把失败样本重新抽取到本批。

1. 先读report.md及checkpoint.json。
2. 按REPRODUCE.md核对两个ZIP及还原统计。执行副本的完整raw保持不变。
3. 若开始后续科学研究，另行登记开发数据、候选生成/拟合改动及验收条件，再冻结新的评价。不得使用本批评价调参后仍沿用本批冻结身份。

当前失败已经保留：弱支路漏真、候选截断、单初值联合拟合不可用、移动双向域求解间隙未闭合，以及8末端校准不足导致未决。报告区分这些原因，未把未解决研究写成方法成功。

源码根C:/Users/23761/.codex/worktrees/36a1/Topo，分支codex/ev-train-validation-design，科学实现dfe24edf。执行副本D:/0-github_workspace/Topo/outputs/ev_full_pipeline_stage_20261018；outputs/full_pipeline_v1为已完成输出。原PDF和pyproject修改保持不动。
