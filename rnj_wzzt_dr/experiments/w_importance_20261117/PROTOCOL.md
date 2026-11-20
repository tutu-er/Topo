# Matched Protocol for Removing the W Stage

The study compares three fresh MSE versions of method `101` on inputs from the prior loss study. The original [locked protocol](https://github.com/tutu-er/Topo/blob/e66aae90cf49093483ac3dc8ece2e8d411296a99/rnj_wzzt_dr/experiments/w_importance_20261117/PROTOCOL.md) was set before this batch was fitted, after review of earlier results on the same inputs.

| Variant | Sequential pool QP trials | Unknown-support MIQP | E reconstruction |
| --- | --- | --- | --- |
| `full` | Yes | Yes | Yes |
| `pool_only` | Yes | No | Yes |
| `skip` | No | No | Yes, from the base model |

The objective, support penalty of 0.003, coefficient range [0, 2], zero-weight cleanup, relative response floor of 0.08, and strict validation MSE are shared. All branches use the same raw RNJ support pool, fixed reference family, base supports, and two-terminal supports. A W path can add supports to the E pool. The E warm-start rule compares the current feasible point with the same fixed reference.

## Inputs and budgets

The grid uses the 160 primary inputs from the paired loss study. Four networks use five training lengths, four levels of additional P/Q measurement noise, and two repeats, with common 192-point validation and 96-point new-day windows. The long panel reuses 16 of those inputs. Each input in each panel runs all three treatments, yielding 480 grid tasks and 48 long tasks. True supports and the new day enter only the evaluator.

| Panel | Total (s) | Per W solve (s) | E maximum (s) | Output reserve (s) | Fixed-support soft cap (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long | 14,400 | 600 | 6,900 | 600 | 300 |
| Main grid | 600 | 10 | 270 | 60 | 300 |

The E allowance is also bounded by remaining structural time. Each treatment shares the same maximum E allowance, so omitting W cannot increase that maximum. The variant order rotates by input. Gurobi uses a single thread, relative gap `1e-4`, and absolute gap `1e-7`. Up to three workers start only after the parent batch has finished. The 96-hour dispatch window begins at the start of this batch, and interrupted fits retain their original attempt records.

## Evaluation and attribution

Report gains, losses, and ties on exact internal and full-support recovery, then per-support errors, R/X response error, coverage, runtime and failed or timed-out calls. Compare the validation and cumulative winners separately. Record the E solve status and actual allowance and the support changes caused by output contraction.

The comparison from `full` to `pool_only` tests removing W's unknown-support solve. The comparison from `pool_only` to `skip` tests removing the sequential trials. Changes can propagate through E's domain, initialization, and the candidate bank. Compare the actual warm start and native return before assigning a gain to a solver. Re-evaluating a saved W prefix describes that path under its original budget.

Each worker saves the normal model and timing receipts or preserves partial records on failure. An independent evaluation reconstructs scores, checks planned denominators, and records failures and time limits alongside paired results.
