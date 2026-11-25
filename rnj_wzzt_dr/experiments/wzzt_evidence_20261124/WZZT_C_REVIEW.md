# WZZT-C (the explicit combination) against WZZT-R: code, procedure, first results, open design questions

Status: **prototype for your review** (2026-11-25). Development inputs only, nothing registered or frozen, nothing
committed. Code: [`identification/combo.py`](../../identification/combo.py) (WZZT-C, about 170 lines),
[`run_combo.py`](run_combo.py) (runs and tables), [`test_combo.py`](test_combo.py) (4 tests, including the root
search against brute force). WZZT-R: [`identification/revised.py`](../../identification/revised.py), procedure in
[`METHOD_WZZT_R.md`](METHOD_WZZT_R.md).

## 1. The two procedures side by side

Both use the same model, residual operator M (meter offsets, GLS slot term), noise level σ̂² (saturated fit) and
BIC score J(F) = SSE/σ̂² + ln N · p_s · k(F). They differ only in how the tree is searched.

| Step | WZZT-R (frozen) | WZZT-C (prototype) |
| --- | --- | --- |
| Matrix for the distance builder | P-only ordered fit under M (SciPy SLSQP in the frozen runs) | same fit, Gurobi barrier (`combo.setup`, line 30) |
| Distance builder | RNJ at tolerances 0, 0.04, 0.08, 0.16 → candidate **pool** | RNJ at tolerance 0 only (`fit`, line 68; `builder=` also takes `nj`, `rg`) |
| Terminal lines | proposed by the pool and four generators, **added one at a time** by W when they lower J | the builder's lines with ≤ n/2 meters are candidates (`terminal_blocks`, line 48); the score keeps or drops them inside one exact selection |
| Terminal blocks | none (W starts from the star) | the largest terminal candidates; meters outside them are single-meter blocks (k = 2–8 here) |
| Root lines | proposed by the pool, sibling pairs, residual RNJ, residual clusters and the screen (all unions of child blocks, coefficients fixed); added one at a time, **greedy** | **every union of two or more blocks** (`root_unions`, line 58; at most 2^k − k − 2 = 246) |
| Decision | W: refit each candidate exactly, add the best, stop when none lowers J; E: portfolio search over a finite pool | one exact pool MIQP (`revised.select_miqp`, line 116) over terminal candidates + root unions, warm-started by a local search and given its value as cutoff: **the proven best laminar family in that set** |
| Completeness check | W: one unknown-support MIQP when no candidate gains; E: MIQP certificate over the pool | node-by-node MIQP over the whole tree (`miqp_ext.certify_by_nodes`, line 131): any compatible line anywhere; a paying line is added and the check repeats (up to `repairs` = 10) |
| Greedy steps | yes (W) | only in the repair loop |
| Time (130 development inputs) | median 85 s | median 1.8 s (max 75 s) |

## 2. First results (130 Phase 2 development inputs; `run_combo.py table --tag dev`)

| | WZZT-C | WZZT-R |
| --- | --- | --- |
| Synthetic exact recovery (64 inputs) | 28 | **32** (only by C: 1, only by R: 5) |
| Real supported lines found (254) | 192 | **228** |
| Precision | 78.0% | **91.3%** |
| Recall, near-root lines (176) | 60.8% | **69.9%** |
| Recall, terminal-side lines (676) | 65.5% | **68.3%** |
| Lower score J on the same data | 2 inputs | **49 inputs** (79 ties) |
| Certified (no compatible line pays, all node MIQPs proved) | 130 / 130 | (E confirmed 122 / 130 in Phase 2) |

**Why WZZT-C loses.** WZZT-R's tree scores better in 49 inputs. In **37** of them a line of WZZT-R's tree
*crosses one of the builder's blocks*: the block wrongly joins meters that the better tree separates, and no root
union can split a block. In the other **12**, a line of WZZT-R's tree fits the blocks but the builder never
proposed it (the repair loop adds lines one at a time and stopped in a worse certified tree). Both trees are
certified, so these are two different local optima; the blocks decide which one WZZT-C can reach.

**Consensus blocks do not fix it.** Blocks on which RNJ, NJ and RG all agree still have a WZZT-R line crossing
them in 43 of 130 inputs (37 with a 0.35 n size limit): all three builders read the same matrix and make the same
grouping errors.

**What the builders are good at** (P-only matrix, exploratory, `RESULTS_MIQP.md`): on long terminal lines the best
builder's recall equals WZZT-R's (Phase 3 real 46.0% against 46.0%), but two in three of the lines it outputs are
false (precision 32–48%). So builders are good at *proposing* terminal lines, not at *deciding* them; their blocks
inherit the false lines.

## 3. The open design questions

### Q1. How to confirm root lines?

What "confirmed" can mean, from weakest to strongest:
1. **Kept by the score**: the line lowers J by more than its price in the exact selection (both methods).
2. **Optimal in its set**: the exact pool MIQP proves no laminar family in the candidate set scores better
   (WZZT-C's root search; WZZT-R's E certificate).
3. **No compatible alternative**: the node-by-node MIQP proves no single compatible line anywhere pays its price
   (WZZT-C's final check; seconds up to 63 meters).
4. **Evidence margin per line** (not yet used in either): for each root line, the score loss when it is removed
   (dJ), and the best alternative grouping at its node by the forced node MIQP. A root line is confirmed when its
   own evidence exceeds the price *and* beats every alternative grouping of the same blocks by a margin.

Options for the search itself: (a) exhaustive over unions of blocks (current; needs k ≤ about 9); (b) a
multi-line MIQP over blocks for larger k (a new formulation: binary block membership per line slot, laminarity
between slots); (c) top-down node MIQPs (exact steps, greedy order: the original MIQP-only W, `R_wmiqp`, which was
not more accurate than the generators).

### Q2. When and how to construct terminal lines?

- **When.**
  - *Before the root (current).* Fast, small root problem; but block errors are unrecoverable (37 of 49 losses).
  - *After the root (top-down).* Root lines first on the whole feeder (wide, slow: provable only to about 29
    meters at the star), then terminal lines inside each root block.
  - *Alternating.* Terminal blocks → root search → re-derive blocks from the result (allowing a block to be
    dropped or split by the certificate) → repeat until stable.
- **How (which source).**
  - *RNJ* on the P-only matrix (current), at one or several tolerances.
  - *A clustering method.* Two candidates in the code base: WZZT-R's residual clustering (average linkage on the
    correlation of each block's residual, `revised.gen_residual_clusters`), or clustering of the meters' voltage
    responses directly. **Which cluster method do you mean?**
  - *Union of sources* (RNJ + clustering + NJ/RG) as candidates, with blocks taken only from lines that several
    sources agree on, or that the score validates.
- **Which lines become blocks.**
  - *Builder blocks (current):* the builder's largest terminal lines.
  - *Score-validated blocks (implemented as `blocks_from="validated"`, not yet tested):* the exact pool MIQP first
    selects among the builder's terminal lines alone; only the lines it keeps form blocks.
  - *Size limit:* n/2 now; a smaller limit gives smaller, more reliable blocks but more of them (k up to 21 with
    consensus blocks at 0.35 n), which needs Q1(b).

## 3b. Added 2026-11-25: clustering sources and a defined block rule (code in `combo.py`)

**What every fit and MIQP minimises (WZZT-R and WZZT-C alike).** The error of the squared-voltage drops,
y_t,i = V0,t² − V_i,t², after the residual operator M: ‖M(y − Σ_C (r_C a_C,t + x_C b_C,t) z_C)‖², with a_C and b_C
the active and reactive load downstream of line C. Both r_C and x_C are fitted (`select_miqp`:
`res == B[:, :K] @ r + B[:, K:] @ x − uy`; `miqp_ext`: Φ = [P Q]). "P-only" appears in one place only: the matrix
that the distance builders read to *propose* groupings. With a fixed power factor, Q = tan φ · P, so a fit of y on
P alone returns R_eff = R + tan φ · X, which has exactly the same tree (same supports, weights r_C + tan φ · x_C);
the P-and-Q matrix gave the builders about half the recall (Amendment D). The price counts 1 parameter per line
when Q moves with P (r and x are then not separately identifiable), 2 otherwise.

**Sources of terminal lines** (each gives one hierarchy; terminal candidates are its lines with ≤ n/2 meters):

| Source | Input | Method |
| --- | --- | --- |
| `rnj`, `nj`, `rg` | P-only matrix (shared paths) | RNJ, NJ, recursive grouping (tolerance 0) |
| `upgma` | path distance d_ij = R_ii + R_jj − 2R_ij | average linkage |
| `ward` | √d_ij | Ward linkage |
| `rrows` | rows of the matrix (sensitivity profiles) | correlation distance, average linkage |
| `volt` | raw V² drops, per-meter demeaned | correlation distance, average linkage |
| `volt_m` | V² drops after M (offsets, GLS slot term removed) | correlation distance, average linkage |
| `resid` | V² residual of the star fit under M | correlation distance, average linkage |

**How the blocks for the root MIQP are determined** (`blocks_from`):
1. `builder`: the largest terminal lines of one source (the first prototype).
2. `validated`: the exact pool MIQP selects among *all* sources' terminal candidates; the score keeps only lines
   that pay their price. The blocks are the largest kept lines.
3. `margin` (proposed rule): as 2, then a kept line becomes a block only if the data prefer it by at least m BIC
   units both to removing it and to *every* candidate from any source that crosses it (replacing it, with the kept
   lines that candidate crosses removed). A line that fails is opened, and its kept sub-lines are tested instead.
   m = 6 (strong evidence on the usual BIC scale; WZZT-R uses the same 6 for its alternatives), lowered to 4, 2, 0
   only as far as needed to leave at most k_max = 9 blocks. Rationale: a block is fixed only where the data rule
   out the competing groupings that the other sources propose; everything else is left to the exact root search.

Evaluation on the 130 development inputs: each source's terminal recall and precision, and for every block rule
the number of wrong blocks (a block that a true line crosses), the share of true lines still reachable, and k
(`run_combo.py sources`); then full WZZT-C runs against WZZT-R (`run_combo.py run --sources ... --blocks ...`).

### Results, step 1: sources and block rules against the truth (130 development inputs, `sources_dev_table.txt`)

852 true internal lines (676 terminal side, 176 near the root). A block is *wrong* when a true line crosses it;
*reachable* true lines are those compatible with every block (only these can still be found).

| Method | Terminal recall | Precision | Wrong blocks (inputs) | Reachable, all / near root | Blocks k, median (max) |
| --- | --- | --- | --- | --- | --- |
| RNJ | 68.5% | 35.3% | 119 of 367 (53) | 78.4% / 60.8% | 3 (8) |
| NJ | 68.0% | 34.7% | 116 of 371 (52) | 79.2% / 56.8% | 3 (8) |
| RG | 39.6% | 33.8% | 111 of 337 (62) | 72.3% / 58.5% | 5 (14) |
| volt (raw V²) | 54.6% | 31.3% | 113 of 345 (60) | 80.4% / **76.1%** | 5 (13) |
| volt_m (V² after M) | 58.1% | 30.3% | 119 of 326 (72) | 78.1% / 60.8% | 4 (9) |
| resid (star residual) | 20.4% | 9.8% | 244 of 364 (103) | 54.6% / 31.2% | 3 (7) |
| rrows (matrix rows) | 59.0% | 29.0% | 138 of 366 (63) | 71.1% / 34.1% | 3 (7) |
| upgma | 49.4% | 24.9% | 157 of 378 (73) | 63.3% / 21.0% | 3 (8) |
| ward | 51.6% | 24.9% | 149 of 384 (70) | 64.9% / 17.6% | 3 (7) |
| **Union of all nine** | **84.2%** | 12.3% | – | – | – |
| **Validated** (score selects from the union) | 65.5% | **78.7%** | 59 of 312 (37) | 87.8% / 70.5% | 4 (12) |
| **Margin, m = 6** | | | **43 of 281 (28)** | **91.0% / 77.3%** | 5 (17) |
| Margin, m lowered to keep k ≤ 9 | | | 53 of 291 (33) | 88.6% / 73.9% | 5 (12) |

- Every single source makes wrong blocks on 52–103 of 130 inputs; about a third of RNJ's multi-meter blocks are
  wrong. No single source is safe for fixing blocks.
- The sources are complementary: their union holds 84% of the true terminal lines (best single source 68.5%).
- Letting the score choose among them (validated) halves the wrong blocks (119 → 59) and raises precision to 79%.
- The margin rule removes more (43 wrong blocks on 28 inputs at m = 6) and keeps 77% of the true root lines
  reachable, but needs more blocks (up to 17); keeping k ≤ 9 forced a lower m on 12 inputs.
- The margin is a calibrated confidence: selected lines with margin ≥ 20 BIC units are true 85.6% of the time,
  6–20: 74.6%, 2–6: 65.7%, 0–2: 57.9%.
- Raw-voltage clustering (`volt`) keeps root lines reachable best (76%) despite lower recall; residual clustering
  and the path-distance clusterings (upgma, ward) damage the root side.

## 3c. The block criterion with one parameter (proposed 2026-11-25)

**Units.** WZZT's score in its reporting units is J = SSE/(2σ̂²) + ½ · (parameters) · ln N, half the BIC, i.e. the
negative log of the BIC approximation to the marginal likelihood. A difference ΔJ between two trees is therefore an
approximate log Bayes factor.

**Criterion (parameter m, an evidence margin in log-Bayes-factor units).**
1. *Candidates:* every line proposed by any source (RNJ, NJ, RG and the six clusterings). Sources only propose.
2. *Best tree among the candidates:* T*, the exact optimum of J over all laminar families of candidates (pool MIQP).
3. *Evidence of a kept line C:* Δ(C) = min over its alternatives of J(alternative) − J(T*). The alternatives are
   (a) T* without C, and (b) for every candidate D that crosses C, T* with D instead of C (and without the kept
   lines D crosses).
4. *Blocks:* from the largest kept lines down, C is fixed as a block when Δ(C) ≥ m; a line that fails is opened and
   its kept sub-lines are tested. Meters in no block are single-meter blocks. The exact root search then works on
   the blocks.

**Meaning of m.** The data favour the block over its strongest competing grouping by a Bayes factor of at least
e^m; with equal prior odds the block's probability against that competitor is at least e^m/(1+e^m): m = 2.2 for
90%, 4.6 for 99%, 6.9 for 99.9%. An equivalent form uses all alternatives at once: π(C) = 1/(1 + Σ e^(−ΔJ)) ≥ π₀.

**Why it is not a heuristic.** The decision uses only the model's own likelihood and price. The sources enter only
as alternatives, and more sources can only make the test stricter. The one free choice is m.

**What remains to set.** (a) The value of m: the BIC probabilities assume the model is right, and the linearised
model is not exact, so m is read off the development trade-off below and then frozen for confirmation. (b) Whether
"terminal" keeps its size limit (at most n/2 meters) or the evidence alone decides at every level (no limit): both
are swept.

### Results: the sweep over m (130 development inputs, `evidence_dev_sweep.txt`)

**Decide terminal lines together with root lines.** T* is now the exact best tree over *all* candidate lines (any
size), not over terminal candidates alone. Alone it is already close to WZZT-R: synthetic exact 30 against 32,
real supported lines 214 against 228, precision 89.5% against 91.3%, proved optimal on all 130 inputs. Choosing
terminal lines without the root lines (step 1's "validated") had forced wrong terminal lines to absorb the missing
root effects: 59 wrong blocks against 4–9 now.

**Calibration** (kept lines of T*, all sizes: share that are true lines): margin < 2: 67%; 2–4.6: 71%;
4.6–6.9: 82%; 6.9–10: 84%; 10–20: 86%; 20–50: 94%; ≥ 50: 100% (202 lines). The margin orders the lines well, but
the BIC probabilities are optimistic (a margin of 4.6 means 99% under the model, 82% observed).

| m | BF e^m | No size limit: wrong blocks (inputs) | reachable all / root | k median, max | k ≤ 9 | Limit n/2: wrong blocks (inputs) | k ≤ 9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 4 of 263 (4) | 99.1% / 100% | 3, 10 | 129 | 9 of 275 (7) | 108 |
| 2 | 7 | 3 of 261 (3) | 99.2% / 100% | 3, 11 | 127 | 7 of 268 (6) | 106 |
| **4.6** | **100** | **1 of 254 (1)** | **99.5% / 100%** | **3, 11** | **127** | 5 of 255 (4) | 103 |
| 6.9 | 1,000 | 2 of 252 (2) | 99.2% / 98.9% | 3, 11 | 127 | 5 of 246 (4) | 100 |
| 10 | 2·10⁴ | 3 of 246 (3) | 99.1% / 99.4% | 3, 13 | 125 | 3 of 232 (3) | 97 |
| 20 | 5·10⁸ | 1 of 227 (1) | 99.9% / 99.4% | 4, 15 | 119 | 0 of 201 (0) | 87 |
| 50 | 5·10²¹ | 0 of 156 (0) | 100% / 100% | 7, 18 | 90 | 0 of 136 (0) | 71 |

- Without the size limit the evidence alone decides where blocks are (confident lines near the root become blocks
  too), wrong blocks are fewer at every m up to 10, and the root search stays small (k ≤ 9 on 127 of 130 at m = 4.6).
  So the n/2 limit can be dropped, leaving **m as the only parameter**.
- **Recommended m = 4.6** (Bayes factor 100, "99% against the strongest competing grouping" under the model): 1
  wrong block in 130 inputs, 99.5% of true lines and every true root line still reachable, at most 11 blocks.
  Larger m buys no further safety until m ≈ 50, where k grows past 9 on 40 inputs.
- To be fixed before confirmation: m (chosen here on development inputs) and k_max = 9 (a computing limit, not a
  modelling choice: above it the root search is skipped and the certificate's repairs work on T*).

## 3d. All versions so far (130 development inputs; WZZT-R as frozen, SLSQP guidance)

| Version | Synthetic exact (64) | Real supported (254) | Precision | Score vs WZZT-R: better / worse / tie | Median time | Certified |
| --- | --- | --- | --- | --- | --- | --- |
| **WZZT-R** | 32 | **228** | **91.3%** | – | 85 s | E confirmed 122 |
| **WZZT-C, evidence blocks, m = 4.6, all 9 sources** | **33** | 223 | 89.4% | 3 / 17 / 110 | **5.9 s** | **130** |
| WZZT-C, evidence blocks, m = 0 | **33** | 222 | 89.1% | 3 / 18 / 109 | 5.4 s | 130 |
| WZZT-C, evidence blocks, m = 15 | **33** | 223 | 89.2% | 4 / 17 / 109 | 4.4 s | 130 (root search skipped on 9: k > 9) |
| T* alone (exact choice over the sources' lines) | 30 | 214 | 89.5% | – | – | proved 130 |
| WZZT-C, margin rule on terminal lines only | **33** | 199 / 247 | 86.1% | 4 / 28 / 95 | 8.5 s | 126 of 127 |
| WZZT-C, first prototype (RNJ blocks) | 28 | 192 | 78.0% | 2 / 49 / 79 | 1.8 s | 130 |

- WZZT-R is still the most accurate on the real feeders, by 5 supported lines (2%) and 1.9 points of precision;
  the evidence-block WZZT-C recovers one more synthetic tree, is 14 times faster and certified on every input.
- Wrong blocks no longer cost anything: in all 17 inputs where WZZT-R's tree scores better, its extra lines fit
  WZZT-C's blocks. They are lines no source proposed, which the one-line-at-a-time repair did not reach.
- m = 0 and m = 4.6 give almost the same trees here (the joint selection already removes most wrong lines); m
  matters for the safety of the blocks (wrong blocks 4 against 1) more than for the final accuracy.
- All versions were developed on these inputs; a fair verdict needs fresh inputs and a registration.

## 4. How to change and test

- Parameters of `combo.fit`: `builder` (rnj/nj/rg), `tolerance`, `limit` (block size limit as a share of n),
  `k_max`, `blocks_from` (builder/validated), `repairs`, time limits.
- Run on the development inputs and compare with WZZT-R:

      python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_combo run --tag <name> [--blocks validated]
      python -m rnj_wzzt_dr.experiments.wzzt_evidence_20261124.run_combo table --tag <name>

  (about 10 minutes with 2 workers; set `PYTHONPATH` to the repository root.)
- Tests: `pytest rnj_wzzt_dr/experiments/wzzt_evidence_20261124/test_combo.py`.
- Any version we want to claim needs its own registration and fresh inputs (the transformer networks are still
  unexamined line by line).
