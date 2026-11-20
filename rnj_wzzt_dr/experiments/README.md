# Experiments

The latest head-to-head studies use the same four reference networks and a common 11-day profile window.

| Study | Question | Recorded outcome |
| --- | --- | --- |
| [WZZT training loss and measurement grid](wzzt_long_grid_20261117/README.md) | How do MSE and MAE compare under matched validation, budgets, and data? | 1,249 successful fits and 23 failures across 1,272 tasks |
| [W-stage ablation](w_importance_20261117/README.md) | What changes when W uses only the RNJ pool or is skipped? | 528 completed tasks, with a paired comparison in each panel |

Each study has its own runner, evaluator, protocol, and tests. Full reproduction uses the locked inputs and original run archives described in the [data and archive guide](../ARCHIVE.md). Earlier research folders retain their original records and report languages. See the [research history](../HISTORY.md) for version-specific references.
