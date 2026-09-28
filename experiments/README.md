# Experiment entrypoints

## 当前主线入口

```powershell
python -m experiments.run_mainline
```

此入口转发到 `rnj_wzzt_core/rnj_wzzt/` 的正式流程：有序约束 R/X 最小二乘、
RX75/RNJ、bootstrap 边界筛选、原终端支撑 MILP 补全和独立验证集选模。
本目录其余脚本用于历史基线和研究对照；例如
`run_unified_topology_pipeline_sweep` 包含 NJ/RG 聚合、GTLS 候选后验、AC 排序与 quartet 验证，
是独立研究流程。正式算法定义和默认参数以核心目录为准。

## Aggregation ablation

```powershell
python -m experiments.run_two_level_aggregation_ablation `
  --cases paper15 soumalas11 flynn16 pengwah18 `
  --data-regimes sparse medium `
  --noise-regimes nominal high `
  --trials 2 --quartet-replicates 8 `
  --nj-bootstrap-replicates 8 --paired-nj-ablation
```

This compares the common non-aggregated candidate pool with NJ/RG-guided
aggregate/decompose candidates under the same AC-generated measurements and
the same physical ranker.

## Latent-tree baselines

```powershell
python -m experiments.run_latent_tree_stability_aggregation
python -m experiments.run_complete_rooted_topology_f1
```

These entrypoints compare NJ, RG, and RNJ hidden-tree reconstruction and report
rooted terminal-clade metrics. The disabled terminal-equivalent MST code is not
part of the physical hidden-tree pipeline.

## Complete deterministic baseline

```powershell
python -m experiments.run_complete_baseline
```

The baseline uses daily demeaning, one Ordered symmetric/nonnegative R/X fit,
the RX75 additive distance, and known-root RNJ. It does not use aggregation,
GTLS, AC reranking, or quartet postprocessing.
