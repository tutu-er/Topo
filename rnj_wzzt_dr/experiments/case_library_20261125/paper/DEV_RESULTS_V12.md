# Algorithm 1 of draft v12 on the development twins (2026-11-26)

**Run.** `dev_v8.py run --max-n 25`: the 240 development twins with at most 25 meters, with the PCC (the gate
decides). Code frozen at ef9ec75 (a separate checkout, so later edits could not touch the run). Eight workers, about
1 h 45 min, 0 errors. The 48 twins with 26 to 60 meters are running now; this note covers the 240.

## Against the registered predecessor (same cases, both with the PCC)

The predecessor's numbers are S2 of `final_source.jsonl`. The truth is the identifiable truth.

| Cases | F1, v12 | F1, predecessor | v12 better / equal / worse |
| --- | --- | --- | --- |
| named feeders, 4 x 48 | 0.924 to 0.931 | 0.926 to 0.933 | 0 / 189 / 3 |
| random small and medium, 4 x 12 | 0.729 to 0.778 | 0.715 to 0.778 | 3 / 45 / 0 |
| all 240 | 0.890 | 0.890 | 3 / 234 / 3 |

- The trees are the same in 234 of 240 cases.
- What v12 adds is the labels and the load gate.
- It is slower: median 57 s per case against 9 s, maximum 4,040 s, because of the flip tests.
- **Gate:** it modelled the load in 180 of the 180 cases with a load and in 0 of the 60 without.
- **Stops:** 220 runs stopped by the stopping rule (Lemma 3) and 20 at the round limit.

## Labels

| Groupings of the tree | True | False |
| --- | --- | --- |
| decided | 1,475 | 3 (0.2%, in 3 cases) |
| open | 137 | 35 |
| unsettled (flip test at its time limit) | 82 | 6 |

| Open groupings outside the tree | True | False |
| --- | --- | --- |
| nested in the tree (one more line) | 235 | 350 |
| crossing the tree | 0 | 9 |

- **Decided labels are reliable:** 0.2% of them are wrong, against about 1% if read as a Bayes factor of 100. This is
  much better than the predecessor's local margins on the sealed set: only 78% of lines with a margin of 4.6 to 10
  were true (`RESULTS_SEALED.md`).
- **Open is not a "no":** 80% of the open groupings inside the tree are true lines.
- **The trees miss 434 true groupings.** 235 of them (54%) are labelled open, as lines nested in the tree.

## What this says for the proposals (`PROPOSALS_UNDER_TEST.md`)

- **SR1 passes** on these 240: the true tree is flagged by the S1 F-test in 0 cases (criterion: at most 2%). The leader
  is flagged in 1.
- **SR3, start rule:**
  - "No target" (every grouping decided, nothing open outside) happened in 25 cases.
  - None of the 25 has a wrong decision; 22 have a perfect tree.
  - The other 3 are the three twins of `flynn16_street_l6`. Each misses 4 true lines with no open alternative,
    probably lines the data cannot see; this still needs a look.
  - Decided groupings are wrong in 0.2% overall (criterion: at most 2%).
- **SR4:** pricing was complete in 240 of 240 cases, so S2 never blocks here and this set cannot test it.
- **TR2 and TR4, target rule:**
  - Open nested groupings average 2.4 per case, and 40% of them are true lines the tree lacks: 235 lines in 240 cases.
  - The TR4 criterion for T1 ("at most 1 such line per 50 cases") fails by far. So T1 would drop the targets most
    worth probing, unless probing cannot recover them under T0 or T2 either. That needs probing runs.
  - The simulated day on `dev_small_003_head` already showed one: (8, 9), true and nested, was T0's first target.

## The rewritten builders (`identification_v8/builders.py`, 2026-11-26)

The nine starting trees were rewritten inside the package, replacing `candidates.py` and its use of `identification/`.
On all 240 twins with at most 25 meters, with the load modelled exactly when the gate says so, every one of the nine
trees is identical to the predecessor's.

- One recursive-grouping tree differed at first, from a rounding difference of 1e-16 in how the load's profiles were
  projected out. The rewrite now does those operations in the predecessor's order.
- Algorithm 1's whole output on two cases is identical to the frozen ef9ec75.
- The 48 larger twins (26 to 60 meters) give identical trees too, so all nine builders match the predecessor on all
  288 twins. The rewrite is also faster there: 441 s against 788 s in total.

## Wide nodes and the node MIQP (draft v14, 2026-11-26)

With the old switch (enumerate up to 12 children, node MIQP beyond), the node MIQP never ran on the 248 twins finished so
far (the 240 small to medium twins and the first 8 huge ones). Widest node per tree:

| Twins | Found trees | True trees |
|---|---|---|
| star, street (192) | 3 to 4 children | 2 to 4 |
| small, medium (48) | 3 to 8 | 3 to 7 |
| huge, first 8 | 7 to 11 (two trees with 11) | 5 to 6 |

A found tree with a node wider than any branching point of the feeder is missing lines there. Pricing now enumerates
nodes up to K = 9 children, the same K as the anchored universe, and uses the node MIQP beyond. Checked on the final trees
of the two huge twins with an 11-child node (`check_wide_nodes.py`; pricing only):

| Twin | Pricing by enumeration | Pricing with the node MIQP | Node MIQP at the 11-child node |
|---|---|---|---|
| dev_huge_004_head | 73 s | 11 s, same best line and gain | optimal in 8 s |
| dev_huge_004_none | 63 s, no line pays | 13 s, no line pays | optimal in 9 s |

Both are exact, so results change only if a node MIQP stops at its 60 s limit, which is reported as an unproved node.
The running large-twin dev run uses the frozen checkout with the old switch.

## Solver fix (draft v15, 2026-11-26)

The run of the 48 larger twins was stopped at 10 of 48. Huge twins (47 to 60 meters) took 2.6 to 3.6 hours each, about
80% of it in flip tests that ended at their limit. Looking into why showed three problems.

**1. False solver claims.**
- All 11 anchored MIQPs that reported "optimal" on the huge twins had a bound above a tree in their own search space,
  by up to 82,000 score units.
- The Gurobi log of `dev_huge_004_head`'s first tree shows the root relaxation declared "infeasible" while Gurobi held a
  feasible solution, which it then called optimal.
- Cause: tolerances forced to 1e-9 (NumericFocus 2) on a badly scaled problem. The matrix coefficients spanned 1e-13 to
  2e3. The per-meter bounds are valid: the run's own tree satisfies all 120 of them.
- On the huge twins (old code), 4 of 35 decided groupings were wrong, against 3 of 1,478 below 26 meters. Whether these
  came from false proofs or from pools that lack the true lines is not separated.

First-tree MIQP on `dev_huge_006_none`, 240 s, started from the 40 s local-search tree. A claim is false if its bound is
above the best tree known (the converged local search, score 5,661.4):

| Setting | Outcome | Bound below the best known tree |
|---|---|---|
| tolerances forced to 1e-9 (the old code) | "optimal" in 10 s: false | none (bound above it by 69) |
| forced 1e-9 with NumericFocus 3 | consistent | 241.5 |
| Gurobi's defaults | consistent | 222.4 |
| defaults, coefficients scaled to their columns | consistent | 171.5 |
| forced 1e-9, coefficients scaled | consistent | 165.4 |

With scaled coefficients, `dev_huge_004_head`'s root relaxation solved in 19 to 79 s (unscaled: "infeasible" or
"numerical trouble"). Its claims were consistent at both tolerance settings, with bounds 518 to 525 units below the best
known tree.

Other formulations, not adopted:
- **Residual per meter and hour** (eq. 9 written directly; range 3e-7 to 1e2): also a false optimum at forced
  tolerances on `dev_huge_004_head`, and its root relaxation did not finish in 240 s on `dev_huge_006_none`.
- **MILP with cuts below the same quadratic:** not built. Its relaxation cannot be tighter than the quadratic's own.

**2. Local search stopped by its time limit.** At 40 s per start, the search made one or two steps. On the first pool,
the local-search tree was far from a local optimum:

| Twin | Paying single swaps left | Useless lines in the tree | To convergence | Score gained |
|---|---|---|---|---|
| dev_huge_006_none | 28 | 0 | 8 steps, 52 s | 69 |
| dev_huge_004_head | 104 | 35 (each saves its price) | 25 steps, 346 s | 398 |

In 240 s the MIQP did not find even the 69-unit improvement.

**3. No proof is possible at this size.** Even when consistent, the MIQP's bound stays 158 to 525 score units below the
best tree, so flip tests on 47-60 meter feeders can rarely decide a grouping. There, the trees rest on the local search.

**The fix:**
- Gurobi's default tolerances;
- coefficients scaled to the size of their columns;
- local search to convergence before and after the MIQP, so every candidate line's value is checked and no single
  removal or swap pays;
- every MIQP always started from the converged tree;
- every claim checked against the trees known. A bound above a known tree's score, or an "optimum" that one move still
  improves, is a numerical failure: no bound, nothing decided.

First-tree step with the fix (local search, MIQP, local search again):

| Twin | Score (run's 40 s pass, then fixed) | MIQP | Bound below the tree | Time |
|---|---|---|---|---|
| dev_huge_006_none | 5,730.4, then 5,661.4 | time limit, consistent | 158 | 687 s |
| dev_huge_004_head | 7,332.3, then 6,934.7 | time limit, consistent | 506 | 980 s |

Those two rows are without the scaling. With the scaling too (the committed code):

| Twin | Score | MIQP | Root relaxation | Bound below the tree | Time |
|---|---|---|---|---|---|
| dev_huge_006_none | 5,661.4 | time limit, consistent | solved in 8 s | 177 | 680 s |
| dev_huge_004_head | 6,934.7 | time limit, consistent | solved in 26 s | 525 | 943 s |

The scaling does not reliably tighten the bound (177 here against 158 unscaled, both with the per-meter bounds). Its value
is robustness: no false claims even at forced tolerances, and a root relaxation that always solves.
