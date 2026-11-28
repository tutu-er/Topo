# Partial topology identification and cost-aware probing (2026-11-27)

One directory with everything needed to review and run the study: the method's code, the readers of the cases, the
development runs and checks, the protocol, the results so far, the plan and the paper draft. It replaces the scattered
pieces under `rnj_wzzt_dr/` for review; those stay unchanged (CHANGES.md).

## 实验入口与结果清单

从仓库根目录运行 `python -B topology_study/experiments/run.py --help` 查看实验清单。进入 `topology_study` 后，运行 `python -B experiments/run.py <实验名> --help` 可查看参数。入口调用 `ptid/runs/` 中的同一份实现。相对路径以研究目录为基准。计算使用装有 NumPy、SciPy、gurobipy 并配置了 Gurobi 许可证的 Python 环境。

| 查看或复现的内容 | 实验名 | 最终结果 |
| --- | --- | --- |
| 四网被动辨识、时窗与噪声、末日调度 | `offline`、`sensitivity`、`probe`、`probe-summary` | [四网汇总、文件清单与复现说明](experiments/four_feeders/README.md) |
| small 至 huge 开发库恢复、五法比较与放宽求解时间 | `random-restore`、`random`、`random-budget` | [逐网比较及原始限时记录](experiments/four_feeders/random_comparison.json)、[延时复算说明](experiments/four_feeders/README.md) |
| 根侧漏线的逐阶段归因与相邻负荷变体复查 | `root-ablation`、`root-variants` | [根侧消融](experiments/root_ablation/README.md) |
| 24 与 36 表计的宽分叉算例 | `wide-feeders` | [生成参数和消融结果](experiments/wide_feeders/README.md) |
| 固定锚点后的单条加线或换线，含 60 表固定接点穷举 | `line-exchange`、`exchange-reference` | [21 网配对与固定接点核验](experiments/line_exchange/README.md) |
| 小例穷举、节点对照、基线自检 | `pool-audit`、`node-pairs`、`node`、`baselines-check` | [候选池核验](experiments/pool_audit/README.md)、[节点对照](experiments/node_pairs/README.md)、[固定节点结果](experiments/four_feeders/README.md) |

例如，`python -B experiments/run.py root-ablation` 查看根侧漏线归因。`python -B experiments/run.py offline --cases paper15 --jobs C` 生成日初辨识，再运行 `python -B experiments/run.py probe --case paper15 --policies P0 P3` 可比较被动和滚动调度。已发布的记录存于 `experiments/` 的各实验目录。新运行的四网回执、缓存和详细轨迹默认写入本地 `outputs/`。

**论文算例的展示顺序**

1. 用预定馈线上的五法对照报告整体准确率。对每种方法列出返回最终树的分母、正确分组、漏检和误增。[整体精度](experiments/four_feeders/README.md)
2. 用自然漏线解释各阶段的作用。沿同一条物理线追踪 NJ、RNJ、局部搜索热启动、Tree MIQP 原始整数解和 Node MIQP 的输出。记录还包括评分排除真线和宽节点误增的例子。[逐线归因](experiments/root_ablation/README.md)
3. 用同一锚定树配对检验宽节点搜索能否发现候选池外的线路。24 和 36 表算例各含一条待找回的近根共享段，另各配一张去掉共享段的对照网。4 秒档仅 24 表算例找回目标。将单次时限放宽至 60 秒后，两张含共享段的网均恢复 13/13 个真分组，零误增。[宽节点消融与两档时间](experiments/wide_feeders/README.md)

## Status in one table

| Part | Status |
| --- | --- |
| Method (draft v19) | Tree MIQP selects lines in the pool. Node MIQP searches for a line at a wide junction. Flip tests score groupings and supply feasible rivals. |
| Identification, current code | The passive method recovered all 23 identifiable groups on the four named feeders. Increasing the time limits raised coverage on the verified random feeders from 9/21 to 21/21. Mean grouping F1 was 0.622 on all 21, compared with 0.608 for the first Tree MIQP warm start and 0.399 for RNJ. See [the paired comparison](experiments/four_feeders/README.md). |
| Node refinement | Local search had already recovered the four named trees. With a controlled missing shared line, Node MIQP recovered the target at junctions with 16 and 22 estimated children. See [the paired experiment](experiments/wide_feeders/README.md). |
| Identification, historical code | The 240-case run predates the large-feeder solver fix. The later development and real-feeder runs used an earlier implementation of pricing and stopping. See [historical results](RESULTS.md). |
| Probing | All 28 simulated policy trajectories had zero orders at the registered prices. See [the dispatch results](experiments/four_feeders/probe_results.md). |
| Registered result | Only the predecessor's sealed test (holds; RESULTS.md §1). Its test set is spent |
| Decisions needed | PLAN.md §3 |

## Reading order

1. **METHOD.md**: the method in plain words, and where each part is in the code.
2. **PROTOCOL.md**: the cases, settings, measurements and rules.
3. **RESULTS.md**: what the development runs show so far.
4. **PLAN.md**: next steps and the decisions needed.
5. **PROPOSALS.md**: rules under test, not adopted (when probing starts, what it targets, stop rules).
6. [Paper draft v19](paper/procedure.tex) gives the formal claims and algorithms.
7. **CHANGES.md**: what moved here from where, and what changed in the code.
8. The code: `ptid/`.

## Layout

```
topology_study/
  README.md  METHOD.md  PROTOCOL.md  RESULTS.md  PLAN.md  PROPOSALS.md  CHANGES.md
  paper/       procedure.tex (current draft), procedure_preview.html (historical snapshot)
  experiments/ run.py, experiment design, final reports, figures and supporting records
  ptid/        the Python package (numpy, scipy, gurobipy only)
    method/    sets, model, builders, miqp, search, labels, anchors, identify (Algorithm 1), probe (Algorithm 2)
    cases/     paths, physics, library (development cases), real (real feeders), world (probing day), manifest_dev.json
    runs/      experiment implementations called by experiments/run.py
    tests/     test_method, test_score_bounds, test_pool_pricing, test_probe_rivals
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
