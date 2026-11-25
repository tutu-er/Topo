# WZZT-R: the current procedure (as frozen by rule D2, 2026-11-24)

Code: `rnj_wzzt_dr/identification/revised.py` (`fit`, defaults of `RevisedConfig`); the scalable MIQP forms are in
`rnj_wzzt_dr/identification/miqp_ext.py`. This note states what the method does now, step by step, and which step
handles which part of the tree. The design idea it serves (the user's, from the start): **distance-based tree
building (NJ, RNJ, RG) is reliable for the lines near the terminals; the whole-map fit and its MIQP are needed for
the lines near the root.** WZZT-R combines the two.

## 1. Data and model

- n meters (terminals), T time slots. Measured per slot t: active and reactive consumption p_t, q_t (length n) and
  the squared-voltage drop y_t,i = V0,t² − V_i,t² (V0 = root, i.e. transformer secondary, reading).
- The tree is a laminar family F of supports (sets of meters downstream of a line). Line C has resistance r_C ≥ 0
  and reactance x_C ≥ 0 (nuisance parameters; the factor 2 of the linearised equation is included).
- Linearised model, one line at a time:

      y_t = Σ_{C∈F} (r_C a_C,t + x_C b_C,t) z_C + o + s_t 1 + e_t,
      a_C,t = z_Cᵀ p_t (load downstream of C), b_C,t = z_Cᵀ q_t.

  So R = Σ r_C z_C z_Cᵀ and X = Σ x_C z_C z_Cᵀ, and line C moves the voltage of all |C| meters downstream of it by
  r_C a_C,t + x_C b_C,t.
- Nuisance terms are removed by one linear operator M = D ⊗ S_c applied to every row:
  - D: a free offset o_i per meter (time demeaning), for meter gain bias;
  - S_c = I − c 11ᵀ/n per slot: the root reading's own error, which every y_i shares. Default `source="gls"`:
    c = γ, the GLS weight for a root meter as accurate as a household meter.
  - **Not yet included:** the user's source term (one unmetered load whose power comes from the PCC meter minus
    the terminal meters, acting along its root-to-source path). It is the next model extension.

## 2. Score (L2 with a sample-size-aware price)

    J(F) = ||M(y − ŷ_F)||² / σ̂²  +  ln(N) · p_s · k(F),   N = nT,

- ŷ_F: the bounded least-squares fit of family F (0 ≤ r_C, x_C ≤ bound);
- σ̂²: noise level from the saturated symmetric fit (every R_ij, X_ij free) under M;
- k(F): internal lines; p_s: identifiable parameters per line (1 if every meter's Q moves with its P within the
  meter class, else 2).

The fit error is L2 (squared); the earlier source-term prototype used L1. The score measures each line by its
**effect on all the voltages it moves**, r_C² · |C| · (variation of the downstream load), not by its own entry in
the R/X matrix.

## 3. Procedure

**Stage R0, guidance: distance-based tree building (terminal side).**
1. Fit the best-fitting symmetric R matrix to P only under M, with every entry free except the order rule
   R_ii ≥ R_ij (a convex QP; currently SciPy SLSQP, Gurobi option added, see section 5).
2. Build RNJ trees from it at tolerances 0, 0.04, 0.08, 0.16 (relative to the median root-to-meter path). Their
   lines form the candidate pool; the 0.16 tree is the reference.

RNJ merges any line shorter than the tolerance into its parent, so the pool is strong on long lines near the
terminals and weak on short lines near the root.

**Stage W, top-down growth by the whole-map score (root side first).**
Start from the star (every meter directly under the root). At each step:
1. Candidates: the pool lines compatible with the current tree, plus four generators that work on the current
   residual: sibling pairs; RNJ on the residual; residual clustering; and the **screen**, which ranks every
   union of two or more child blocks at every node with at most 20 children (2^k − k − 2 unions per node).
2. Refit every candidate exactly and add the one that lowers J the most. This is greedy: it adds the one line
   that helps most and never goes back to change that choice.
3. Drop lines whose fitted r and x became zero, when that does not worsen J.
4. Stop when no candidate lowers J. Then the exact unknown-support MIQP checks that no line compatible with the
   tree (not only the candidates) lowers J (120 s limit).

Because the star is the start and the score weighs the whole effect of a line, the first lines added are the big
groups near the root; the terminal lines come later, mostly from the RNJ pool.

**Stage E, search over the finite pool, certified by an MIQP.**
1. Universe: the pool, every line seen on the W path, a frontier of unions of up to 5 child blocks, all pairs.
2. A portfolio (beam search, local search, 20 random starts) from the W result and every RNJ tree.
3. Certificate: an MIQP over the universe proves no family in it scores better (cold start with another seed, the
   best known value as cutoff, tightened bounds: deviation D1).

**Output:** the best family, the W path, and alternatives within 6 BIC units.

## 4. Who handles what, and the MIQP's role today

| Part of the tree | Proposed by | Decided by |
| --- | --- | --- |
| Lines near the terminals (long, small groups) | RNJ pool | whole-map score (W refit) |
| Lines near the root (short, large groups) | screen and sibling generators (unions of child blocks) | whole-map score (W refit) |
| Proof that nothing compatible is better | — | MIQP (W final check, E certificate; node-by-node form for large networks) |

So in WZZT-R as frozen, the combination already exists, but the MIQP certifies rather than searches: the screen
enumerates unions of child blocks with old coefficients fixed, and the refit decides. The node-by-node MIQP
(`miqp_ext.certify_by_nodes`) is the exact version of the screen: it searches every union of child blocks at a
node with all coefficients refitted.

## 5. Solvers

- MIQPs: Gurobi (gurobipy 13).
- Ordered matrix fits (R0 guidance and the residual-RNJ generator): Gurobi barrier since deviation D4
  (`RevisedConfig.qp_solver = "gurobi"`); SciPy SLSQP in all runs before D4 (`qp_solver="slsqp"` reproduces them).
  Same optimum on all 163 inputs checked; ties in the flat optimum can break differently (PROTOCOL_PHASE2 D4).
- Bounded least squares of a fixed family (`bounded_ls`): SciPy NNLS/BVLS; it failed with LinAlgError on 2 Phase 3
  inputs (counted against WZZT-R). To be replaced by a Gurobi QP after the confirmation.
