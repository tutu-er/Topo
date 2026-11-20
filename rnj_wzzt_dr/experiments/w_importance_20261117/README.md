# W-Stage Ablation

This paired study ran three fresh versions of method `101` on the same inputs. `full` runs the pool trials and native W search before E. `pool_only` keeps the sequential fixed-support trials and then runs E. `skip` goes from the base model directly to E. All variants use MSE, identical validation selection, and the same maximum E time within each panel.

## Completed results

All 528 tasks finished and passed independent scoring checks. The table counts exact internal and full-support recovery for the validation winner. The 16 long-budget inputs are a subset of the 160 main-grid inputs.

| Panel | W mode | Exact internal | Exact full support | Mean runtime (s) |
| --- | --- | ---: | ---: | ---: |
| Main grid | `full` | 136 / 160 | 132 / 160 | 129.04 |
| Main grid | `pool_only` | 133 / 160 | 129 / 160 | 68.69 |
| Main grid | `skip` | 112 / 160 | 112 / 160 | 62.37 |
| Long budget | `full` | 16 / 16 | 16 / 16 | 865.85 |
| Long budget | `pool_only` | 16 / 16 | 16 / 16 | 62.09 |
| Long budget | `skip` | 14 / 16 | 14 / 16 | 58.02 |

On the main grid, switching from `full` to `pool_only` gained four full recoveries and lost seven. The median paired runtime ratio was 0.235. Switching from `pool_only` to `skip` lost 17 full recoveries and gained none, with a median runtime ratio of 0.913. On the long panel, `pool_only` recovered the same full supports as `full` at a median paired runtime ratio of 0.087. Removing W changed E's input pool and warm start as well as the final candidate bank. The [protocol](PROTOCOL.md) explains how the report attributes each part of the change.

The original summary is `data/frozen_studies/w_importance_20261117_v1/report/report.json`. The [validation note](VALIDATION.md) describes the separate pre-run checks.

## Run a single input

~~~sh
python -m rnj_wzzt_dr fit --train data/train.npz --validation data/validation.npz --method 101 --loss mse --w-mode pool_only --preset grid --e-seconds 270 --output outputs/pool_only_fit
~~~

Use `--w-mode full` or `--w-mode skip` with a fresh output path for a paired comparison. The regular CLI defaults to `full` and uses the remaining structural allowance for E when you omit the explicit cap.

## Run the study

Use a prepared parent [loss-study run](../wzzt_long_grid_20261117/README.md) with its original input files and run metadata. Preparation checks the source hashes and creates its own task and code snapshot. The controller waits until the parent finishes dispatching and all parent tasks have reached a terminal state.

~~~sh
python -m rnj_wzzt_dr.experiments.w_importance_20261117.run prepare --parent outputs/loss_study --out outputs/w_ablation
python -m rnj_wzzt_dr.experiments.w_importance_20261117.run run --out outputs/w_ablation
python -m rnj_wzzt_dr.experiments.w_importance_20261117.run status --out outputs/w_ablation
python -m rnj_wzzt_dr.experiments.w_importance_20261117.run report --out outputs/w_ablation
~~~

A `STOP` file stops further dispatch. Completed, failed, and interrupted attempts retain their files. Each panel gets an independent report with the planned denominator, completion status, paired outcomes, time limits, and source attribution.
