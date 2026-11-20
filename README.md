# Distribution Network Topology Identification and Conditional Demand Response

This project infers the observable structure of a radial distribution network from terminal active power, reactive power, and voltage measurements. It fits candidate topologies and their voltage-response matrices on training data, selects a model on held-out validation data, and saves the entire candidate set for later scoring and demand-response (DR) planning.

This English edition is based on the November 18, 2026 source snapshot `e66aae90`. The earlier experimental records under `experiments/` retain their original publication language.

## Install and check

Use Python 3.10 or newer. NumPy and SciPy support the fixed-topology fits, frozen scoring, and zero-action screen. Install Gurobi with a license that covers the model size for integer structure search or a nonzero DR quote.

~~~sh
python -m pip install -e ".[test]"
python -m rnj_wzzt_dr doctor
python -m rnj_wzzt_dr check
~~~

For solver workflows and the feeder simulation dependencies, use the following commands.

~~~sh
python -m pip install -e ".[solver,physical,test]"
python -m rnj_wzzt_dr check --solver
~~~

The script `python -I -B rnj_wzzt_dr/run.py --help` is also available from a source checkout. The self-checks use small synthetic cases.

## Input files

Use NumPy `.npz` files with the same terminal labels and column order in every split. Longer histories and multiple days can be supplied by stacking rows in time order.

| Array | Shape | Meaning |
| --- | --- | --- |
| `labels` | `(n,)` | Unique integer identifiers for the measured terminals |
| `P`, `Q` | `(T, n)` | Signed net powers in per unit on a common base. Positive P is consumption and positive Q is inductive demand. |
| `V`, `root` | `(T, n)` and `(T,)` | Positive terminal and root voltage magnitudes in per unit |
| `Y` | `(T, n)` | Optional squared-voltage drop, `root[:, None]**2 - V**2`. Supply this in place of `V` and `root`. |

Use separate training and validation files. The training file determines candidate supports and weights. The validation file ranks the saved candidates by mean squared error (MSE). A support is the set of measured terminals downstream of a branch. Branches that have the same downstream set share one effective response in this model.

## Fit and score

Replace the input paths with your own files. Each command requires a fresh output path.

~~~sh
python -m rnj_wzzt_dr fit --train data/train.npz --validation data/validation.npz --method 101 --loss mse --preset grid --output outputs/fit
python -m rnj_wzzt_dr score --result outputs/fit/result.json --history data/validation.npz --new data/new_day.npz --output outputs/score.json
~~~

`result.json` records the selected index and the complete `frozen_candidates` bank, including the support sets and resistance and reactance response matrices (`R` and `X`). Supply the original validation file as `--history`. The `score` command ranks the frozen bank using validation and new-day residuals together. Each scalar residual has equal weight.

| Method | Search |
| --- | --- |
| `rnj`, `nj` | Fixed-tree baselines from Recursive Neighbor Joining and Neighbor Joining |
| `000` | W alone, with a shared training normalization |
| `100` | RNJ guidance followed by W |
| `101` | RNJ guidance, W, and E |
| `111` | RNJ guidance, W, T, and E. This is the CLI default. |

W adds one support at a time. With RNJ guidance enabled, it first runs fixed-structure trials on compatible supports from the RNJ pool. It then searches for an unknown support and refits the weights. T rebuilds a small neighborhood jointly. E reselects supports from a finite pool and can remove connections accepted earlier.

MSE uses quadratic programming (QP) for fixed supports and mixed-integer quadratic programming (MIQP) for structure search. `--loss mae` uses mean absolute error (MAE), linear programming (LP), and mixed-integer linear programming (MILP). Both modes select by validation MSE. The default output step removes weak non-singleton supports using a training-only relative response threshold of `0.08`, then refits the remaining weights on training data. Set `--internal-floor 0` to disable this step.

| Preset | Total budget (s) | Per W search (s) | Reserved for output (s) |
| --- | ---: | ---: | ---: |
| `quick` | 300 | 30 | 30 |
| `grid` | 600 | 10 | 60 |
| `long` | 14,400 | 600 | 600 |

`--seconds` sets the total budget. `--step-seconds` limits one unknown-support solve in W. T and E use their remaining stage budgets. `--e-seconds` can further cap E. `--w-mode pool_only` uses fixed-structure trials within W. `--w-mode skip` bypasses W. The default is `full`.

`progress.json` and `events.jsonl` record search progress. `checkpoint.json` records accepted candidate points. Solver objectives and bounds apply to the reported subproblem. Check the solve status and the warm-start record when interpreting a returned point.

## Conditional demand response

Prepare a separate `forecast.npz` before observing the new day. Its `labels` must match the fit, and its `power` array has shape `(new_slots, 2*n)`, with all P columns followed by all Q columns in per unit. Supply a zero-based index into the complete candidate bank and a voltage measurement error standard deviation in per unit.

~~~sh
python -m rnj_wzzt_dr zero --result outputs/fit/result.json --history data/validation.npz --forecast data/forecast.npz --target 0 --voltage-sigma 0.0005 --output outputs/zero.json
~~~

The bound is conditional on the target model and the specified error distribution. Add `--case` to use a bundled device allocation when its labels, time slots, and power base match the forecast. With a matching allocation, `plan` can price zero action and can optimize a nonzero quote when exactly one competing candidate remains. Multi-competitor nonzero pricing is available only in separate research implementations. Run `python -m rnj_wzzt_dr plan --help` for arguments.

## Latest studies

- The paired loss study finished with 1,249 successful fits and 23 failures out of 1,272 planned tasks. See the sample-size and measurement-error panels, exact recovery counts, and timing in the [loss study](rnj_wzzt_dr/experiments/wzzt_long_grid_20261117/README.md).
- The W ablation completed all 528 tasks across three W modes with a shared cap on E. See which recoveries were gained or lost and how long the variants took in the [W ablation](rnj_wzzt_dr/experiments/w_importance_20261117/README.md).

The package includes case tables for four radial networks and example device allocations. The studies require their original, hash-checked observation archives under `data/`. Use your own observations to run the main CLI. The fixed-tree `--method rnj` workflow runs with NumPy and SciPy. Keep measured data, fitted parameters, quotes, and detailed search traces in private `data/` or `outputs/` directories.

## Source layout

| Location | Contents |
| --- | --- |
| [identification](rnj_wzzt_dr/identification/) | RNJ guidance, support search, fitting, and output contraction |
| [dr](rnj_wzzt_dr/dr/) | Conditional bounds, device models, and single-competitor pricing |
| [physical](rnj_wzzt_dr/physical/) | AC feeder simulator and four network tables |
| [experiments](rnj_wzzt_dr/experiments/) | Study runners, evaluators, tests, and original research records |
| [tests](rnj_wzzt_dr/tests/) | Regression tests for the supported interfaces |

See the [model and scoring rules](rnj_wzzt_dr/MODEL.md), [validation notes](rnj_wzzt_dr/VALIDATION.md), and [data and archive layout](rnj_wzzt_dr/ARCHIVE.md).
