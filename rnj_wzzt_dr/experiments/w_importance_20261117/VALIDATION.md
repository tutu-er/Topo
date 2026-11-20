# Pre-run Implementation Checks

Before dispatch, the study checked the following behaviors.

- In eight small solves across three and four terminals, both losses, and methods `100` and `101`, the default full mode reproduced the prior path, parameters, objectives, and frozen candidate bank.
- `pool_only` excluded the unknown-support extension call. `skip` skipped W trials and retained the baseline E pool and reference rule. The explicit E allowance reached the solver and remained limited by the structural deadline.
- The three worker variants fitted and froze the same small three-terminal observation set without loading truth or new-day inputs.
- The scheduler preserved all 528 planned task slots, rotated the variant order, and waited for the parent run to finish.
- The pre-dispatch regression recorded 131 passes, 2 optional solver skips, and 47 passing subtests. The final independent report tests recorded 14 passes.

The [completed ablation](README.md) reports the research outcomes. These checks cover the implementation and interface.
