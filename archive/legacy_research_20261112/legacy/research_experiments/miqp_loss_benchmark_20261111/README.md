# Squared-loss / MIQP comparison on the fixed day96 panel

This isolated experiment changes the training fit from MAE to scaled MSE. It
reuses the 48 observation files, RNJ geometry, candidate pool, initial families,
calibration and terminal order frozen in `outputs/day96_t_benchmark_20261110`.
It does not regenerate observations or modify the L1 source snapshot.

For each input, a bounded QP refits the same complete RNJ plus singleton/full
reference family. If its physical mean squared error is `f0_mse` and the parent
MAE reference is `f0_mae`, set `K=f0_mae/f0_mse`. All training decisions minimize
`K*MSE + .003*f0_mae*(proper internal support count)`. Thus the reference
normalized penalty has the same strength as the L1 comparison. SSE and MSE
have the same unpenalized minimizers; changing their scale without adjusting
the count penalty would change the experiment.

Both variants keep small RNJ seed preselection, original one-support WZZT,
immediate numerical-zero release, one final exit, and output contraction/refit.
`noT` continues the original step at the tail. `T003` uses the existing joint
model once every unresolved region has at most five direct blocks. Its count
coefficient remains `.003`. RNJ seed deletion scores and all warm-family fits
are also recomputed using squared loss; the geometric RNJ initialization is
fixed, while the two chosen small seeds may differ from the MAE run.
The warm families are exactly the support families supplied in the frozen
parent bank, which had already removed some zero-weight LP supports. This does
not reconstruct every unpruned raw RNJ hierarchy. The frozen candidate pool and
initial family are unchanged; the structural solver can select their lines.

The primary final selector remains strict validation **MAE** on the `.24`
contracted, QP-refitted own-path library. A secondary strict `K*MSE` selector is
reported on exactly that fitted library, with the original absolute `1e-12`
tie gate. It performs no additional fits. No complete-RNJ output fallback is
used. Truth and test observations are reserved for postanalysis.

## Frozen sources, budgets and accounting

`freeze` verifies and copies every parent Python source byte (253 files in the
current parent protocol), then adds this package. Runtime patching is explicit
through `squared_backend.install(K)`; each common worker and method worker saves
the actual alias/hash manifest and dependency versions. It never edits a parent
file in either the original or copied frozen source tree.
An added package marker and explicit package-path guard bind native imports to
the new frozen snapshot. Every runtime alias is checked against its protocol
path and content hash before search. Separate before/after receipts check all
loaded `research_experiments.*` and `rnj_wzzt.*` files, including the core MILP
builder, QP backend and candidate exporter, against the frozen source manifest.

There are 96 configurations: 48 inputs times `noT/T003`. Each configuration has
a 14,400-second wall budget including the charged shared parent preprocessing,
the new reference/warm QPs, the own structural search, output fitting and
export. Parent preprocessing includes its original MAE reference and LP warm
fits; those costs are retained even though none of their fitted coefficients
is reused. Per fixed-family QP: 300 seconds. Per original extension MIQP:
3,600 seconds. Prefix: at most 7,200 seconds and half the remaining structural
allowance. Final output reserve: 600 seconds. Structural solves keep the
existing relative gap `1e-4`, absolute gap and feasibility settings. Up to eight
workers use one solver thread each. Nominal cap sum is 384 method-hours; shared
preprocessing is charged separately to each configuration for fair standalone
costs. Actual batch elapsed time also depends on concurrent CPU use.

Common preprocessing is shared physically across both methods and charged in
both reported method totals. Completed results are reused on resume. Failed
and missing configurations retain their denominator. An interrupted unknown
worker is not automatically rerun with a fresh budget; it requires an explicit
attempt/budget audit. Atomic JSON writes retry Windows reader-lock conflicts.

The four cases reuse four global seed groups. Confidence intervals must resample
those four groups while carrying all cases and noise levels together; 16
case-seed histories are not 16 independent random seeds.

## Frozen-parameter candidate export

The uncapped pool includes every raw own-path point, the singleton model,
every `.24` output-refitted model, and explicit coefficient-identical anchors
for the actual MAE and MSE selections. Identical support/weight pairs merge;
the same family with different coefficients remains distinct. Export scoring
uses actual historical MAE to preserve the downstream DR scoring contract.
Only detached export copies set `point.objective=training MAE`; native result
points continue to store `objective=K*MSE`. Each row also stores physical MSE,
SSE, native scaled loss and the declared fitting loss. All coefficients stay
frozen. The `.16` no-refit topology map in the portable pool is separate from
the native `.24` contraction/refit selection and is recorded explicitly.

This preserves the absolute-error historical score interface. A downstream
probability calculation is valid only under its declared noise model and still
needs the explicit bridge-to-DR input adapter. No end-to-end DR solve is part
of this benchmark.

## Commands

Run from `D:\0-github_workspace\Topo` with the project Python environment:

```powershell
$py = 'D:/apps/miniconda3/envs/Topo/python.exe'
$out = 'D:/0-github_workspace/Topo/outputs/miqp_loss_benchmark_20261111'
& $py -B -m research_experiments.miqp_loss_benchmark_20261111.run --help
& $py -B -m research_experiments.miqp_loss_benchmark_20261111.run freeze $out --workers 8
& $py -B -m research_experiments.miqp_loss_benchmark_20261111.run pilot $out --workers 8
& $py -B -m research_experiments.miqp_loss_benchmark_20261111.run status $out
& $py -B -m research_experiments.miqp_loss_benchmark_20261111.run all $out --workers 8
```

The pilot is fixed before viewing new results: all four cases, bundle 1000,
voltage noise `.0005`, two methods, eight configurations. Validate this pilot
before the 96-configuration batch. The implementation-stage check is AST/CLI
and synthetic export validation only; no actual pilot is started by writing
this package. Independent postanalysis lives in `analyze.py` and separately
checks QP losses, bounds, selection, topology recovery and parent hashes.
