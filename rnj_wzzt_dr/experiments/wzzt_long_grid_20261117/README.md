# WZZT Training Loss and Measurement Grid

This study compares MSE/MIQP and MAE/MILP training under the same strict validation MSE rule. It also compares method `101`, which uses RNJ guidance, W, and E, with the RNJ1 single-family and RNJ15 multiple-family baselines. All three use the same output contraction and singleton protection.

## Completed results

The frozen report records 1,249 successful fits and 23 failures across all 1,272 planned tasks, with zero independent scoring audit errors. The table shows exact recovery by the validation winner for method `101`. Full recovery includes internal, terminal, and common upstream supports.

| Panel | Training loss | Finished / planned | Exact internal | Exact full support |
| --- | --- | ---: | ---: | ---: |
| Main grid | MSE | 154 / 160 | 130 / 160 | 128 / 160 |
| Main grid | MAE | 154 / 160 | 135 / 160 | 125 / 160 |
| Long budget | MSE | 16 / 16 | 16 / 16 | 16 / 16 |
| Long budget | MAE | 14 / 16 | 14 / 16 | 14 / 16 |

Among the 148 main-grid inputs on which both fits finished, switching from MAE to MSE gained nine exact internal recoveries and lost thirteen. It gained eleven exact full recoveries and lost nine. The median paired MSE-to-MAE runtime ratio was 0.451. On the 14 completed long-budget pairs, both losses recovered every full support, and the median runtime ratio was 0.433. These panels share physical networks and input dates.

The main grid spans four networks, 48 to 768 training points, four added P/Q error levels, and two noise repeats. Validation uses 192 points and the new day has 96. The long-budget panel reuses 16 of the 160 main-grid inputs. An additional 32 inputs test root-meter error and sampling over a fixed time span. See the [protocol](PROTOCOL.md), [data qualification](DATA_VALIDATION.md), and [root-support interpretation](ROOT_IDENTIFIABILITY.md).

The archived evidence is under `data/frozen_studies/wzzt_long_grid_20261117_v2/`. The main result file is `report/report.json`. The [early paired comparison](EARLY_PAIRED.md) and [startup record](RUN_STATUS.md) document intermediate checks. MSE remains the default training loss.

## Run one fit

~~~sh
python -m rnj_wzzt_dr fit --train data/train.npz --validation data/validation.npz --method 101 --loss mse --preset grid --output outputs/fit_mse
python -m rnj_wzzt_dr fit --train data/train.npz --validation data/validation.npz --method 101 --loss mae --preset long --output outputs/fit_mae
~~~

Both commands save the fitted bank for the same score and demand-response interfaces. See the [project README](../../../README.md) for input arrays.

## Prepare and run a batch

The batch requires the original hashed P/Q archive, case context, and prior AC caches in the locations described in the [data plan](DATA_PLAN.md). Install the solver and physical extras and the research dependencies from the root `requirements.txt`. From the repository root, with fresh output paths, run the following commands.

~~~sh
python -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.data prepare --protocol rnj_wzzt_dr/experiments/wzzt_long_grid_20261117/PROTOCOL.md --out outputs/loss_inputs
python -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run prepare --data outputs/loss_inputs --out outputs/loss_study
python -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run check-workers --out outputs/loss_study
python -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run run --out outputs/loss_study
python -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run status --out outputs/loss_study
python -m rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run report --out outputs/loss_study
~~~

Preparation records source, input, and task hashes. The controller runs from the saved source snapshot and preserves attempt records. A `STOP` file in the run directory stops new dispatch while current tasks finish. If the controller exits unexpectedly, keep its lock and partial records until you have verified that its processes have exited. Pending tasks and recorded failures are included in the final plan counts. Use the original archive and its source snapshot for sealed-run replay.
