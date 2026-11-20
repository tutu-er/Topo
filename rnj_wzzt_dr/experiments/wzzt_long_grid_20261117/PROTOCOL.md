# Matched Study Protocol for WZZT Training Loss and Data Conditions

This English edition describes the executed November 17, 2026 design. The original protocol and the budget amendment are pinned in the [source snapshot](https://github.com/tutu-er/Topo/tree/e66aae90cf49093483ac3dc8ece2e8d411296a99/rnj_wzzt_dr/experiments/wzzt_long_grid_20261117). Use those original bytes to verify the historical run.

## Matched estimator

Method `101` uses RNJ guidance, W, and E, with T disabled. Both training losses use the same preliminary R/X geometry and RNJ pool, the same coefficient bounds of [0, 2], the same internal support penalty of 0.003, and the same output response floor of 0.08. The fixed RNJ reference family is refitted under each loss for its training normalization. The validation selector uses strict normalized MSE for both. The evaluator also ranks each saved bank by normalized MAE to isolate the effect of changing the selection rule.

RNJ1 takes one fixed RNJ family. RNJ15 fits the distinct families from the three distance weightings and five tolerances. Both baselines share the coefficient bounds, base supports, singleton protection, and output contraction. The [execution amendment](EXECUTION_ADDENDUM.md) matches the RNJ baselines' outer budgets and reserves to the corresponding `101` panel.

## Observations

The four feeders are Paper15, Flynn16, Soumalas11, and Pengwah18. The input signals come from 18 paired FeederBW P/Q profiles with the fixed January 4 mapping and common positive scales for signed P and Q. The aligned window begins January 21, 2024 at 08:00 Central European Time. Two 1,056-point AC cases are reused and two are prepared for this window. True supports, physical R/X and AC states are used during preparation and post-fit scoring. The worker loads training and validation observations only.

- Training uses the last 48, 96, 192, 384, or 768 quarter-hour points before index 768. The common validation interval is [768, 960), and the new-day interval is [960, 1056).
- Simulated multiplicative Q and P errors have standard deviations of 0%, 0.5%, 1%, or 2%. The terminal and root voltage meters use 0.05% in the main grid. The original P/Q measurement error is unknown.
- Two noise repetitions share the physical window. Methods and nested sample lengths share standard noise draws. Training, validation, and new-day streams are distinct. The recorded seed is `2026111701`.
- Sixteen root-meter controls use 96 training points, 0.5% added P/Q noise, and root-meter error levels of 0% and 0.1%. The main grid supplies the 0.05% midpoint.
- Sixteen sampling controls use 48, 96, 192, or 384 evenly spaced points from the first 768 rows, at 0.5% P/Q noise and repeat 0. All panels share the validation and evaluation windows.

## Tasks and budgets

| Panel | Inputs | Tasks | Total / per W solve / output reserve (s) |
| --- | ---: | ---: | --- |
| Long, `101` and both RNJ baselines | 16 | 96 | 14,400 / 600 / 600 |
| Main grid, `101` and both RNJ baselines | 160 | 960 | 600 / 10 / 60 |
| `100` with RNJ guidance and W | 8 | 16 | 600 / 10 / 60 |
| `000` with W and no RNJ support guidance | 4 | 8 | 3,600 / 300 / 300 |
| Root-meter controls | 16 | 96 | 600 / 10 / 60 |
| Fixed-window sampling controls | 16 | 96 | 600 / 10 / 60 |

The 16 long inputs use both 96 and 768 points, 0.5% and 2% P/Q noise, and repeat 0. They are part of the main grid. Across the batch there are 1,272 method tasks, 440 of which include integer structure search. The runner attempts both losses for every planned method and input.

The per-step cap applies to the unknown support solve in W. E uses the remaining structural allowance. The fixed-support fit has a 300-second soft allowance, with the native limit passed to the MAE LP and a return-time check for the bounded least-squares fit. Native relative and absolute gap settings are `1e-4` and `1e-7`. Up to three single-threaded workers run in parallel. Dispatch stops after 96 hours, and in-progress tasks retain their original allowances.

## Evaluation

For each frozen bank the evaluator computes exact internal and full-support recovery, support precision, recall and F1, true support coverage, and response-matrix errors. It scores validation, new day, and validation plus new day separately. Near-root, large-downstream, common, and singleton supports are reported separately. Newly observed data contribute residuals to scoring, with weights fixed.

Reports include planned and completed denominators, failures, time limits, and same-input gains and losses. They also record how much of the improvement comes from the solver start, the native solver return, or the subsequent refit and output contraction. Solver bounds cover their recorded subproblems. Networks, dates, grid points, and error repetitions are paired and share underlying physical conditions.

The predeclared decision compared the tradeoff in exact recovery and runtime. A switch to MAE required no loss in recovery and at least a halving of runtime. A five-percentage-point gain for MSE with acceptable runtime supported keeping MSE. The reported comparison remains specific to the tested inputs and budgets.
