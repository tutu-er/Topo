# Frozen AC288 runtime source

`source_ac288_72af_20261018.zip` contains the 34 original Python files of the
approved `72af` working-tree package. It includes the terminal-group and Source
implementation and its required untracked `observed_input.py` and
`solver_budgets.py`. These bytes include uncommitted changes on top of Git HEAD
`0d5ae1d81c5cb9324ae9f3160a4d50273e539b9f`; that commit alone does not reproduce
this runtime.

The archive was made from the already frozen
`outputs/source_ac288_runtime_20261018/source_snapshot/rnj_wzzt`, rather than from
the live worktree. The original source location was
`C:/Users/23761/.codex/worktrees/72af/Topo/rnj_wzzt_core/rnj_wzzt`.
The sidecar manifest records every source SHA256, the archive SHA256, the original
runtime-manifest SHA256, and the observed-runtime hashes. The original runtime
manifest remains unchanged:
`575bc61ba620ad1ad12ea2b8383ee7ece8810756a9a74ba4ff0fc34d82c37e1d`.

Archive SHA256:
`ced63e0ad09c5d2be805be680d4a89b98cde9173e20caf665f88986ebcb24028`.
Members have a fixed order, timestamp and file mode. The archive contains Python
sources only, with no caches, experiment data, or credentials.

`prepare_source_ac288_runtime.ensure_runtime(output_directory)` reads this
repository archive by default, checks the archive and per-file hashes, and reads
members directly in memory without extracting paths. It then creates the source
snapshot and frozen runtime, adding only the passive observer. Removing that
observer must reproduce the original whole-module AST. An explicit
`source_package=Path(...)` remains available to compare a local working-tree
package against the archived bytes. An existing valid runtime is checked and
returned without rewriting its manifest or source files.

From a fresh checkout:

```powershell
python -B research_experiments/rnj/prepare_source_ac288_runtime.py --output-dir outputs/source_ac288_runtime_20261018
```

Preparing and verifying a runtime do not call an estimator or a solver. Loading
the runtime requires the environment dependencies listed by its manifest.
