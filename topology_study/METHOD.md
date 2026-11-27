# The method in plain words

This explains what the code does and why, without the proofs. The formal version is `paper/procedure.tex` (draft
v16); the section and equation numbers below are the draft's. The code is in `ptid/method/`.

## 1. The task

A low-voltage feeder runs from a transformer (the point of common coupling, PCC) to n customers with smart meters.
Every hour each meter reports its active and reactive power (p, q) and its voltage V; the PCC reports its voltage and,
where metered, its power. We want the feeder's topology: which customers share which line. Some of it the data decide,
some they do not. The method says which part is decided, how sure it is, and, optionally, buys small demand-response
"blocks" (probing) to decide more.

There may be one load without a meter (street lights, a pump, unmetered homes). Its power is known from the PCC
balance (PCC power minus the meters and the losses), but it has no voltage reading.

## 2. Topology as sets of meters

A line is described by its **support**: the set of meters (and possibly the unmetered load) it feeds. The supports of
a tree are **laminar**: any two are nested or disjoint. So a topology is a laminar family of meter sets. Two lines
**cross** when they overlap without one containing the other; crossing lines cannot both be in a tree.

With an unmetered load s, the same group of meters M can appear as the line M (load elsewhere) or M ∪ {s} (load below
it). Both say the same about the meters, so labels are about **groupings** of meters; where the load sits is reported
separately.

## 3. The model and the score (Sections I and III)

Linearised power flow: the drop of a meter's squared voltage, y_i = V0² − V_i², is a sum over the lines on its path of
(r × the power through the line + x × the reactive power through it). Each line has its own r and x, both ≥ 0.
Meter offsets are removed (each meter's mean), and the shared PCC voltage error is weighted out (generalised least
squares).

The **score** of a tree F is

    H(F) = RSS(F) / (2 σ̂²) + β k(F),    β = ½ p_s ln N − λ

- RSS(F): the residual of F's best fit (coefficients bounded to [0, 2] pu);
- σ̂²: the noise variance, from the **saturated model** (every pair of meters, and every meter with the load, gets its
  own free coefficient; every tree is a special case of it, so σ̂² does not depend on any tree);
- k(F): the number of lines; β: the price of a line. ½ p_s ln N is the BIC price (p_s = 2 parameters per line, N = hours
  × meters); λ = 1.96 is a structural prior: real LV feeders rarely have nodes with more than two children
  (odds 0.141 on 84 real feeders, λ = ln(1/0.141)).

Score differences are log posterior odds. A margin of **m = ln 100 ≈ 4.6** score units means odds of 100 to 1.

**Why the score finds true lines (Theorem 1).** Removing a true line costs the part of the true voltage signal that the
remaining lines cannot reproduce; that cost grows with the number of hours. Adding a line that is not there gains only
the noise along the directions it adds (about one χ² unit per parameter) and pays β, so it lowers the score only with
probability about e^λ / N (with 8 days of hourly data: about 1 in 430 for 16 meters, 1 in 1,600 for 60). This is the classical reason BIC selects the true model, stated for
one data set and bounded fits.

## 4. Is there an unmetered load? (Section II)

Before any fitting, the PCC balance decides it: the unmetered share ς = (mean unmetered power) / (mean metered power).
A load is modelled only if ς ≥ 1%. Below that, the load is too small to place (a line with and without it would fit
equally), so it is treated as absent. On the development cases the statistic separates cleanly (no load: at most
0.66%; a 2% load: at least 1.53%).

## 5. Algorithm 1: the tree and its labels (Section IV)

1. **Starting trees.** Nine quick tree builders (neighbour joining, recursive grouping, clustering and others,
   Appendix B) each give a tree. Every cluster of every builder tree becomes a **candidate line** (the pool). With a
   load, every candidate appears with and without the load.
2. **First tree (`solve`).**
   - *Local search:* from the empty tree and from each builder tree, repeatedly make the best single move: remove a
     line, or add a pool line and drop the lines it crosses. The best start is searched **to convergence**: at the end,
     no single removal or swap of any candidate line improves the score. (Every candidate line's value is checked.)
   - *Tree MIQP:* a mixed-integer quadratic program picks the best tree over the whole pool: one yes/no variable per
     candidate line, crossing lines exclusive, a line's r and x zero unless chosen. It starts from the local-search tree.
     Its coefficients are scaled to the size of their columns, and Gurobi runs at its default tolerances (see
     RESULTS.md, "Solver fix", for why).
   - *Checked claims:* every result is compared with the trees already known. A lower bound above a known tree's score,
     or an "optimum" that one move still improves, is impossible: the solve is recorded as a **numerical failure**,
     gives no bound and decides nothing.
   - The better tree is searched to convergence again.
3. **Labels.** For each grouping M of the current tree, ask: what is the best tree that decides M the other way?
   - *Local flips* (cheap): remove M's lines, or swap in a crossing candidate. A flipped tree within m of the tree makes
     M **open**. Local flips can never prove M decided (they only give upper bounds).
   - *Flip test* (the MIQP with M's lines removed from the pool, and the tree's score plus m as cutoff): if the solver
     proves no flipped tree within m exists, M is **decided**; if it finds one, M is **open** (that tree is the rival);
     if the time runs out first, M is **unsettled**.
4. **Anchors and the anchored search.** Decided groupings are **anchors**. Given the anchors, every tree that keeps them
   uses only unions of a few "units" inside each anchor (Proposition 4). When an anchor has at most K = 9 units, all
   those unions are offered; larger nodes keep the pool's own lines. The tree MIQP then searches this anchored universe
   with every anchor required.
5. **Pricing.** At each node of the tree, is there a single new line (a union of some of the node's children) that
   lowers the score? Nodes with at most K children are checked by fitting every union; wider nodes by a small MIQP (the
   node MIQP), which is exact too. Paying lines are added and the tree is re-solved.
6. **Rounds.** Steps 3 to 5 repeat (at most R = 2 anchored searches) until the stopping rule of Lemma 3 holds (same
   anchors, nothing added by pricing, the last MIQP proved optimal) or the round limit is reached. The last labels are
   those of the final tree and pool.
7. **Outputs.** The tree; each grouping's label with its gap interval and rival; the certificate's scope (single-line
   pricing complete, a paying line left, or incomplete; whether the anchored search was solved to optimality); with a
   load: its path, its admissible places within m, and how strongly the voltages show it (the evidence interval).

**What a label promises (Proposition 2).** "Decided" means: every tree within m of the best (over this pool and model)
agrees with the tree on M. "Open" means a flipped tree within m was found. Labels are about the model, the data and the
pool, not the truth; how often decided labels are wrong is measured on the development cases.

## 6. Algorithm 2: one day of probing (Section V)

The operator can buy signed active-power blocks at chosen meters for an hour (from batteries and EV chargers through
an aggregator), within price, budget and voltage-band limits.

- **Targets.** At the start of the day: the open groupings, and the load's place when its second-best place is within
  m, whose rival tree responds differently from the leader to some feasible block.
- **Value of a block.** For each target, a block changes how differently the leader and its rival predict the next
  hour's voltages: the separation q. If the leader is true, the evidence the hour adds is Gaussian with mean q/2 and
  variance q (Lemma 4). The value counts the expected evidence up to the target's remaining deficit:
  κ(δ, q) = E min(δ, Z). The block bought maximises J = ψ Σ [κ(with block) − κ(without)] − cost, if J > 0.
- **Trust gate.** Blocks are bought only when the leader fits about as well as the meters allow.
- **Gates.** Every hour: refit (Algorithm 1, steps 3 to 6, on all data so far), relabel every target (targets outside
  the tree by their flip test), choose the block for the next hour. Stop when every target is decided at two
  consecutive gates; a last gate after hour 23 uses the last block's readings.

ψ (money per unit of evidence) is a placeholder until the cost-coverage study (PLAN.md).

## 7. Where each part is in the code

| Draft | What | Code |
| --- | --- | --- |
| eqs 1-5 | model, balance, gate, score, saturated model | `method/model.py` |
| Appendix B | nine starting trees, candidate pool | `method/builders.py` |
| eq. 9 | tree MIQP, per-meter bounds, checked claims | `method/miqp.py` |
| eq. 10, Lemma 2 | local search; pricing (enumeration, node MIQP) | `method/search.py` |
| Def. 1, Lemma 1, eqs 6-8 | local flips, flip tests, labels | `method/labels.py` |
| Def. 2, Prop. 4, eq. 11 | units and anchored universe | `method/anchors.py` |
| Algorithm 1, eq. 12 | the whole identification; load places and evidence | `method/identify.py` |
| Algorithm 2, eqs 13-15 | probing | `method/probe.py` |
| Section I | supports, laminar families | `method/sets.py` |

## 8. Known limits

- On feeders of 47 to 60 meters the MIQP's bound stays far below the best tree (hundreds of score units), so flip tests
  there end unsettled and the trees rest on the converged local search. The output says so.
- Labels hold for the model; the model is linear, balanced and single-phase, with one unmetered load at most.
- Certificates cover single-line additions only: two lines can pay together when neither pays alone (Remark 1).
- The gate ignores loads below 1% of the metered power; its threshold fits one meter class.
- ψ and the trust gate's level are not yet calibrated; κ is not monotone in q for small deficits (PROPOSALS.md, §7).
