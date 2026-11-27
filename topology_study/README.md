# Partial topology identification and cost-aware probing (2026-11-27)

One directory with everything needed to review and run the study: the method's code, the readers of the cases, the
development runs and checks, the protocol, the results so far, the plan and the paper draft. It replaces the scattered
pieces under `rnj_wzzt_dr/` for review; those stay unchanged (CHANGES.md).

## Status in one table

| Part | Status |
| --- | --- |
| Method (draft v17) | Algorithm 1 reports a feasible tree and grouping labels relative to an explicit candidate pool and score margin. Algorithm 2 ranks candidate probing actions. |
| Identification, current code | Four synthetic inputs with four metered leaves. All 24 conclusive final labels agreed with independent enumeration within the corresponding final pool. See [the audit](experiments/pool_audit/README.md). |
| Identification, historical code | The 240-case run predates the large-feeder solver fix. The later development and real-feeder runs used an earlier implementation of pricing and stopping. See [historical results](RESULTS.md). |
| Probing | Simulated days only. The monetary conversion ψ and the policy's cost–coverage tradeoff await evaluation. |
| Registered result | Only the predecessor's sealed test (holds; RESULTS.md §1). Its test set is spent |
| Decisions needed | PLAN.md §3 |

## Reading order

1. **METHOD.md**: the method in plain words, and where each part is in the code.
2. **PROTOCOL.md**: the cases, settings, measurements and rules.
3. **RESULTS.md**: what the development runs show so far.
4. **PLAN.md**: next steps and the decisions needed.
5. **PROPOSALS.md**: rules under test, not adopted (when probing starts, what it targets, stop rules).
6. [Paper draft v17](paper/procedure.tex) gives the formal claims and algorithms.
7. **CHANGES.md**: what moved here from where, and what changed in the code.
8. The code: `ptid/`.

## Layout

```
topology_study/
  README.md  METHOD.md  PROTOCOL.md  RESULTS.md  PLAN.md  PROPOSALS.md  CHANGES.md
  paper/       procedure.tex (draft v17), procedure_preview.html
  experiments/pool_audit/    four synthetic inputs, audit report and recorded JSON
  ptid/        the Python package (numpy, scipy, gurobipy only)
    method/    sets, model, builders, miqp, search, labels, anchors, identify (Algorithm 1), probe (Algorithm 2)
    cases/     paths, physics, library (development cases), real (real feeders), world (probing day), manifest_dev.json
    runs/      identify_cases (runs, summary, compare), probe_cases, checks, pool_audit
    tests/     test_method, test_score_bounds, test_pool_pricing
```

## Running

Run from `topology_study` with Python, NumPy, SciPy and a usable Gurobi license. The four-input audit used the local
`poweropt` environment with Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3 and Gurobi 13.0.1.

```
python -m ptid.tests.test_method                      # 12 checks, no case files needed
python -m ptid.tests.test_method --data               # + 3 checks on the study's cases
python -m ptid.tests.test_method --legacy             # + 2 checks against the registered predecessor (needs the repo)
python -B -m ptid.tests.test_score_bounds             # 6 regressions on bounds, cutoffs and feasible flip witnesses
python -B -m ptid.tests.test_pool_pricing             # 4 regressions on pool retention, pricing and stopping
python -B -m ptid.runs.pool_audit --out pool_audit_rerun.json  # 4 synthetic inputs, independent enumeration

python -m ptid.runs.identify_cases run --set dev --one-twin --workers 8    # 72 development cases
python -m ptid.runs.identify_cases run --set real --workers 8              # 66 real-feeder inputs
python -m ptid.runs.identify_cases summary --set dev
python -m ptid.runs.identify_cases compare OLD.jsonl NEW.jsonl

python -m ptid.runs.probe_cases --names dev_small_003_head --variant T0,T2 --quick
python -m ptid.runs.checks extra-lines
```

Identification runs resume by case name. Choose a separate `--file` for a new code version or configuration.

## Data

The synthetic audit can run without external case files. Its [recorded results](experiments/pool_audit/results.json)
accompany the code. Larger case files and historical batch outputs live under `../outputs/` by default
(`PTID_OUTPUTS` overrides this path).

| What | Where | Count |
| --- | --- | --- |
| development cases | `outputs/case_library_20261125/dev/` | 288, each checked against `ptid/cases/manifest_dev.json` |
| real-feeder inputs | `outputs/hourly_probing_dev_20261120/eulv/inputs/` | 66 (33 EU LV feeders, start days 0 and 6) |
| this study's runs | `outputs/topology_study/` | written by `ptid.runs` |

The sealed test cases (`outputs/case_library_20261125/test/`) are never read by this code. The case generator is not
here: it is registered and frozen in `rnj_wzzt_dr/experiments/case_library_20261125/`, and the hash check ties every
case read here to it.
