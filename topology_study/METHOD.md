# The method in plain words

The formal version is [paper/procedure.tex](paper/procedure.tex). The equation numbers in this overview refer to
the v17 derivation. The code is in `ptid/method/`.

## 1. The task

A low-voltage feeder runs from a transformer (the point of common coupling, PCC) to n customers with smart meters.
Every hour each meter reports its active and reactive power (p, q) and its voltage V; the PCC reports its voltage and,
where metered, its power. We want to know which customers share which line. Given the measurements, scoring model
and a pool of possible lines, the method tests whether
competitive trees agree on a grouping. The optional probing policy buys one-hour demand-response blocks.

There may be one load without a meter (street lights, a pump, unmetered homes). Its power is estimated from the PCC
balance using the measured loads and an approximation of feeder losses. The load has no voltage reading.

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
- k(F) counts optional internal lines. Each costs β score units. The number of observations is N = hours × meters.
  The code sets the penalty parameter p_s to 2 when reactive power varies sufficiently independently of active power
  at a majority of the meters, and to 1 otherwise. Each line still has two fitted coefficients.
- λ = 1.96 is an empirical structural weight derived from branching counts on 84 feeders.

The preselected margin **m = 4.6** compares tree scores. Within a candidate pool a flip changes the grouping under
test and allows the rest of the tree to change. Its evidence is a feasible competing tree or a valid bound on the best
competing score. The empirical error rate among decided groupings is measured separately against reference topologies.

**How a missing line affects fit (Theorem 1).** Under the linear model, a rival that omits a true line loses the part
of the noiseless signal that its remaining predictors cannot represent. Distinct variations in customer demand can
make that loss grow across hours. If the rival reproduces the signal, the bound gives no positive separation.
For a redundant line the theorem bounds the potential fit improvement by noise in the added directions under its
stated coefficient conditions. The score also charges β for the line.

## 4. Is there an unmetered load? (Section II)

Before fitting, calculate the estimated share ς = (mean unmetered power) / (mean metered power).
The identification model includes the load when ς ≥ 1%. This modelling rule was chosen for the study's meter class.
The largest share in the no-load development cases was 0.66%, and the smallest in the 2% load cases was 1.53%.

## 5. Algorithm 1: the tree and its labels (Section IV)

1. **Starting trees.** Nine quick tree builders (neighbour joining, recursive grouping, clustering and others,
   Appendix B) each give a tree. Every cluster of every builder tree becomes a **candidate line** (the pool). With a
   load, every candidate appears with and without the load.
2. **First tree (`solve`).**
   - *Local search:* from the empty tree and from each builder tree, repeatedly make the best single move: remove a
     line, or add a pool line and drop the lines it crosses. The best start is searched **to convergence**: at the end,
     no single removal or swap of any candidate line improves the score. (Every candidate line's value is checked.)
   - A mixed-integer quadratic program (MIQP) searches the pool from the local-search tree. It uses one yes/no variable
     per candidate line, excludes crossings and sets a line's r and x to zero unless it is chosen.
     Its coefficients are scaled to the size of their columns, and Gurobi runs at its default tolerances (see
     RESULTS.md, "Solver fix", for why).
   - *Checked claims.* A lower bound has to be consistent with known feasible trees in the domain it covers.
     A conflicting bound or a claimed optimum improved by a feasible move makes the solver claim unusable.
   - The better tree is searched to convergence again.
3. **Labels.** For each grouping M of the current tree, ask: what is the best tree that decides M the other way?
   - *Local flips* (cheap): remove M's lines, or swap in a crossing candidate. A flipped tree within m of the tree makes
     M **open**. Local flips can never prove M decided (they only give upper bounds).
   - The flip MIQP excludes all lines expressing M and compares the best remaining score to H(T) + m for the current
     tree T. A valid exclusion gives **decided**, a feasible rival within the margin gives **open**, and an unresolved
     comparison gives **unsettled**. A time-limited solve can still supply a witness or a sufficient lower bound.
   - Per-meter tightening and an objective cutoff can restrict the solve to low-scoring solutions. The reported
     lower bound is valid over the original pool with the flip imposed. If a restricted search proves no flip
     reaches the threshold, it establishes threshold exclusion. A larger lower bound from that search is capped
     at the score level its restrictions protect.
4. **Anchors and the anchored search.** Decided groupings are **anchors**. Given the anchors, every tree that keeps them
   uses only unions of a few "units" inside each anchor (Proposition 4). When an anchor has at most K = 9 units, all
   those unions are offered; larger nodes keep the pool's own lines. The tree MIQP then searches this anchored universe
   with every anchor required. A better refitted flip rival within this domain that respects all its anchors supplies
   an extra starting tree. A competing line that crosses an anchor and is absent from the pool can remain undiscovered.
5. **Pricing.** Each node offers lines formed by joining a proper subset of its children. At most K children permits
   enumeration. At a wider node, a feasible improving union can be added immediately. A valid lower bound can rule
   out any improvement exceeding the pricing tolerance. If neither is available, that node is unproved. Every added
   line is retained in the cumulative pool, including those removed from the tree by a subsequent solve.
6. **Rounds.** The convergence rule requires unchanged anchors, no added line, completed single-line pricing with
   no outstanding improvement, and a closed gap for the preceding anchored solve at the stated tolerance. Otherwise
   the method continues to the configured limit (default R = 2). The final labels use the final tree and cumulative
   pool. At termination an eligible lower-scoring flip rival may replace the tree once. The procedure relabels
   that tree, records any further improving rivals and retires the old single-line certificate.
7. **Outputs.** The result contains the feasible tree, the cumulative pool, the evaluated groupings' labels and
   gap intervals, and any recorded rival. It identifies the final restricted solve, whether its gap closed to
   tolerance, and the status of single-line pricing. For an unmetered load it also reports its path and the
   score of each placement relative to the best on the same meter tree. The optional evidence interval
   compares the best fits with and without a load in their specified search domains.

**What a label promises (Proposition 2).** Every tree within m of the best score in the pool agrees with a decided
grouping of T. An open grouping has a rival within m of T. To establish disagreement in the near-optimal set itself,
both trees must lie in it. For example, a valid lower bound L for the same pool without anchors certifies this if
both trees score at most L + m.

## 6. Algorithm 2: one day of probing (Section V)

The operator can buy signed active-power blocks at chosen meters for an hour (from batteries and EV chargers through
an aggregator), within price, budget and voltage-band limits.

- **Targets.** Use open groupings and, where applicable, the lowest-scoring tested load path distinct from the
  fitted path. That placement is open if its score is at most the fitted tree's score plus m. A target's rival
  must respond differently from the leader to an available block.
- **Value of a block.** For each target, q measures how far apart the two fitted predictions are for the next hour.
  Holding both fits and their common covariance fixed, if observations follow the leading tree's assumed Gaussian
  distribution, the predictive log-likelihood ratio has mean q/2 and variance q (Lemma 4). This calculation ranks
  actions before the next observation. The value counts the predicted evidence up to the target's remaining deficit:
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

- In historical 47–60 meter cases, MIQP bounds remained hundreds of score units below the best feasible tree.
  Many flip tests reached their time limit and left the grouping unsettled.
- Labels hold for the model; the model is linear, balanced and single-phase, with one unmetered load at most.
- Single-line pricing checks additions to the current tree. Two lines can pay together when neither pays alone
  (Remark 1). The anchored master and the grouping flips have their own candidate domains.
- The gate ignores loads below 1% of the metered power; its threshold fits one meter class.
- ψ and the trust gate's level are not yet calibrated; κ is not monotone in q for small deficits (PROPOSALS.md, §7).
