# Compensation-guided DR: first bounded test, 2026-10-30

**The implemented rule did not outperform same-budget simple exploration.**
DR reduced errors compared with equal additional observations without reward,
but the guidance itself did not add value. No production defaults were changed.

## Implemented rule

The planner runs historical finite-pool WZZT to its stopping conditions and uses
its selected uncontracted family F. Candidate G_C is obtained by deleting blocks
crossing C and adding C. Within the original independent r/x domain [0,2], it
finds minimum/maximum new-block effects on a historical near-optimal face. Both
training and validation are constrained; all weights are jointly free, including
zero. The face thresholds come from the existing acceptance and 3% rules.

The finite endpoint predictions are deduplicated and equally weighted as an
explicit heuristic. For each executed training state, F, G_C and complete native
RNJ candidates are fitted jointly on old plus forecast data. One accepted step,
original 3% selection and output contraction give predicted final block error.
The planner integrates independent training/validation participation and whole-
event signs exactly, then enumerates 16 price menus. It does not maximize a
gradient norm or input rank, and does not run a new MILP for every price.

This is a one-step planning approximation with historical calibration. Actual
evaluation rebuilds the current observed candidate bank/calibration and runs
the full unchanged research WZZT paths. See [METHOD.md](METHOD.md) for the full
definition and its finite-model limitations.

The rule selected terminals102/103, both price1 in both training rounds, with
the same prices repeated for independent validation events. The two nodes use
opposite shift directions, each subject to stochastic acceptance. Five menus
tied at predicted error .025; expected payment .8 versus1.2 made this pair unique.
The no-reward predicted error was .4. These are finite-model planning scores,
not actual recovery probabilities. Expected acceptance was .4 for every menu.

## Locked test and complete denominators

- Three-terminal linear mechanism case; no new AC or empirical DR claim.
- Same original288 training/288 validation rows; final480/480; original natural
  stream, .2kW/.07kvar shift, four events and worst payment1.6 retained.
- Voltage relative Gaussian SD .0002; P/Q SD .005. These remain synthetic SDs,
  not instrument accuracy classes.
- Zero internal block once; weak.02 and strong.6 at all three pair locations:
  seven different true models, with positive true leaf blocks retained.
- Two independent future measurement/behavior streams, shared natural future.
  Five policies give70 mapped records,47 unique observed fits,141 distinct
  method outputs and210 mapped method outputs. They are not70 independent samples.
- All70 records completed, with zero failures, unresolved outcomes or partial
  searches. Total2335 LP:1825 history/profile/planning,510 actual identification;
  elapsed592.64s, within the locked3000LP/900s limit.

## Final results

Each policy has14 counterfactual model-stream records. Error totals count every
missing and extra block, including weak and leaf blocks.

| Policy | W raw errors | W final errors | W final full recovery | RNJ + optional root final errors | Same-method full recovery |
|---|---:|---:|---:|---:|---:|
| Compensation guidance |18|22|0/14|14|3/14|
| No reward, same new observations |29|29|0/14|29|1/14|
| Random repeated single, price2 |22|22|0/14|15|3/14|
| Random different-node rotation, price2 |21|21|0/14|14|3/14|
| Random simultaneous opposed pair, price1 |15|19|1/14|14|3/14|

Guided W versus zero:7 improved,6 equal,1 worse; versus rotation:0 improved,
13 equal,1 worse; versus random pair:1 improved,9 equal,4 worse. The optional-
root method has the same per-condition final errors for guided, rotation and
pair. This does not establish W-specific guidance value.

Per *one four-event deployment* actual payments were:

| Stream | Guided | Zero | Single | Rotation | Pair |
|---|---:|---:|---:|---:|---:|
| r0 |1.0|0|1.6|1.2|1.0|
| r1 |1.2|0|1.6|1.2|1.2|

All paid menus have worst payment1.6. Each deployment is reused across seven
counterfactual models, so their payment sum is not seven actual expenditures.
In r1 the random pair chose exactly the guided menu; seven rows are identical
observations/output, not independent guidance comparisons.

## Where potential recovery was lost

Guided W has a complete correct saved candidate in10/14 records. All ten lie
within the original3% validation band. In nine records, the selector chooses a
smaller declared family. The remaining record selects a correct raw family,
then contraction removes its true common root. Final full recovery is0/14.

The raw18 to final22 errors are four additional missing true roots in r1: the
zero-internal case and all three weak cases. Their estimated contraction
strengths are about.0721–.0752, below thresholds.1153–.1171. No false-block
removal offsets these four misses. This is an output-rule effect, not merely
an increase/decrease in a gradient or rank.

## A concrete planning limitation

The historical selected F is leaf-only, and its pool contains only internal23
and the full root. Thus this batch has no planning crossing deletion, and the
one-step planner contains only leaf, leaf+23 and leaf+root families.

The original historical W path already saved a root+23 family at point16:
training loss about.00042251969 is below the face cap.00042455114; validation
loss.00043560249 is below.00044867057. The planning definition excluded that
available joint explanation because it started from the simplified final F
and allowed one addition. The five finite endpoint models therefore did not
represent the available multi-block joint explanation. This is evidence from
old history, not a family added using test truth.

The failed test rejects the advantage claim for this endpoint/one-step rule.
It does not reject all compensation-based DR or DR itself. A possible next
mathematical correction is to retain complete historical path families in the
planning domain. Whether it changes prices and actual results is unverified.
No model, seed, threshold, noise or budget was adjusted after these results;
no second performance batch was started.

## Verification and recovery

26 pure tests passed. Independent audits replayed all observation losses,
controls, payments, source/data bindings, original selection/contraction and
2335 LP receipts:12,176 actual checks plus230 binding/selection checks passed.
Two independent planning audits reproduced all16 menu scores; largest reported
score difference was2.22e-16. These audit counts are not statistical samples.

Public, explicitly aggregate-only files are under
`../evidence/compensation_design_v1/`:protocol_summary,results_summary,audit_summary.
They contain declared settings, random seeds, hashes and aggregate counts only.
The full private archive export was rejected by automatic review because its
JSON field-level exclusion of derived trajectories had not been established.
It was not executed. No broad private archive was uploaded.

Private recovery locations, relative to WZZT_RESEARCH_CACHE:

- `wzzt_dr_compensation_v1`:frozen protocol/source snapshot, historical/profile
  fits, all conditional planning fits, locked actions,70 evaluation rows,
 47 actual fit receipts, execution receipts and observed packets.
- `wzzt_dr_compensation_audit_v1`:actual independent replay and supplemental
  freeze/selection audit.
- `dr_compensation_method_audit_v1`:historical faces, independent menu replay
  and conclusion checks, each with source and hashes.
- `dr_compensation_price_audit_stage`:second price/probability/cost replay.

All runs have ended. Do not re-run or overwrite the frozen run. Implementation
commit is0c5c8714; full private protocol SHA-256 is
`86f3993d47f71287718c3444e2895427f6924306b5075a18715637bc4c72b25f`.
