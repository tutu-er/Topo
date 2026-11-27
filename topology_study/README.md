# Partial topology identification and cost-aware probing: the case study, revised (2026-11-27)

One directory with everything needed to review and run the study: the method's code, the readers of the cases, the
development runs and checks, the protocol, the results so far, the plan and the paper draft. It replaces the scattered
pieces under `rnj_wzzt_dr/` for review; those stay unchanged (CHANGES.md).

## Status in one table

| Part | Status |
| --- | --- |
| Method (draft v16) | Algorithm 1 (identification with labels) and Algorithm 2 (probing) implemented |
| Identification | Development results only. A solver fault on 47-60 meter feeders was found and fixed on 2026-11-26/27 (RESULTS.md §3); the runs with the fix are in RESULTS.md §4 |
| Probing | Simulated days only; ψ is a placeholder; not evaluated |
| Registered result | Only the predecessor's sealed test (holds; RESULTS.md §1). Its test set is spent |
| Decisions needed | PLAN.md §3 |

## Reading order

1. **METHOD.md**: the method in plain words, and where each part is in the code.
2. **PROTOCOL.md**: the cases, settings, measurements and rules.
3. **RESULTS.md**: what the development runs show so far.
4. **PLAN.md**: next steps and the decisions needed.
5. **PROPOSALS.md**: rules under test, not adopted (when probing starts, what it targets, stop rules).
6. **paper/procedure.tex**: the paper draft (v16). `paper/procedure_preview.html` is a browser view of the same text.
7. **CHANGES.md**: what moved here from where, and what changed in the code.
8. The code: `ptid/`.

## Layout

```
topology_study/
  README.md  METHOD.md  PROTOCOL.md  RESULTS.md  PLAN.md  PROPOSALS.md  CHANGES.md
  paper/       procedure.tex (draft v16), procedure_preview.html
  ptid/        the Python package (numpy, scipy, gurobipy only)
    method/    sets, model, builders, miqp, search, labels, anchors, identify (Algorithm 1), probe (Algorithm 2)
    cases/     paths, physics, library (development cases), real (real feeders), world (probing day), manifest_dev.json
    runs/      identify_cases (runs, summary, compare), probe_cases, checks
    tests/     test_method
```

## Running

Environment: Python 3.10 with numpy, scipy and gurobipy (Gurobi 13, licensed); on this machine
`conda run -n Topo310`. Run everything from this directory (`cd topology_study`).

```
python -m ptid.tests.test_method                      # 12 checks, no case files needed
python -m ptid.tests.test_method --data               # + 3 checks on the study's cases
python -m ptid.tests.test_method --legacy             # + 2 checks against the registered predecessor (needs the repo)

python -m ptid.runs.identify_cases run --set dev --one-twin --workers 8    # 72 development cases
python -m ptid.runs.identify_cases run --set real --workers 8              # 66 real-feeder inputs
python -m ptid.runs.identify_cases summary --set dev
python -m ptid.runs.identify_cases compare OLD.jsonl NEW.jsonl

python -m ptid.runs.probe_cases --names dev_small_003_head --variant T0,T2 --quick
python -m ptid.runs.checks extra-lines
```

`run` is resumable: it skips cases already in its output file.

## Data

Case files and results are not in Git. By default they are under `../outputs/` (set `PTID_OUTPUTS` to change it):

| What | Where | Count |
| --- | --- | --- |
| development cases | `outputs/case_library_20261125/dev/` | 288, each checked against `ptid/cases/manifest_dev.json` |
| real-feeder inputs | `outputs/hourly_probing_dev_20261120/eulv/inputs/` | 66 (33 EU LV feeders, start days 0 and 6) |
| this study's runs | `outputs/topology_study/` | written by `ptid.runs` |

The sealed test cases (`outputs/case_library_20261125/test/`) are never read by this code. The case generator is not
here: it is registered and frozen in `rnj_wzzt_dr/experiments/case_library_20261125/`, and the hash check ties every
case read here to it.
