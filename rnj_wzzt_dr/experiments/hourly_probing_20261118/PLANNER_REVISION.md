# Planner revision: a certified miss bound in place of the union-bound tail (draft, 2026-11-22)

**Status:** proposal for the open decision "Planner guarantee" (PROTOCOL §11). It replaces the active-set rule of manuscript Section IV-C/D (protocol v3 §5). Sections 1–9 give the certified planner and its first test; Section 10 a stronger greedy; Section 11 a correction to the scoring (GLS for the shared root-meter error). Equivalence, the confirmation rule and the device model are unchanged.

## 1. Why the v3 planner is idle

The v3 planner targets an active set A of at most three rivals. Every other non-equivalent rival goes into a tail, and a plan exists only if the tail mass stays below δ: `Σ_{j∉A} c_j < δ`, with `c_j = t^{-1/2} e^{-E_j/2}` and `t = α/(1−α) = 1/19`. A single tail rival already breaks this when

$$
c_j \ge \delta \iff E_j \le \ln\tfrac1t + 2\ln\tfrac1\delta = \ln 19 + 2\ln 10 \approx 7.5 .
$$

So the planner buys nothing as soon as a fourth non-equivalent rival lies within 7.5 BIC units. With the revised bank (up to 150 candidates) this held at every triggered gate: 862 of 862 in the development runs (89% for this reason, 11% because a rival was beyond reach). The tail rule is valid, but it spends the miss probability δ in a rigid way.

## 2. Setting

At gate g the planner seals a block schedule `u` for the next `H_c` hours (2 h). Notation as in the manuscript:
- `L` is the BIC leader and `R` the set of rivals that are not operationally equivalent to it (19); `E_j = J_j − J_L ≥ 0`.
- **Confirmation** at the end of the window means `Σ_{j∈R} e^{−E_j^+} ≤ t`, with `E_j^+` the BIC gap after the window.
- **Separation of rival j:**

  $$
  q_j(u) = \Delta\mu_j(u)^\top \bar S_j^{-1} \Delta\mu_j(u),\qquad
  \Delta\mu_j(u) = \mu_L(u)-\mu_j(u),\qquad
  \bar S_j = \sigma^2 I + \tfrac12\Phi_L\Sigma_L\Phi_L^\top + \tfrac12\Phi_j\Sigma_j\Phi_j^\top ,
  $$

  that is, (15a) with ω = 1/2. `Δμ_j` is affine in `u` (operating-point sensitivities), so `q_j` is a convex quadratic in `u`.
- **Penalty drift.** `β_j = ½(p_j − p_L) ln(N^+/N)` is the deterministic change of the BIC penalty as the sample grows from `N` to `N^+` rows.

**Assumptions.**
- (A1) Under H_L (the leader is the truth), the window's measurements are Gaussian around `μ_L(u)` with covariance `S̄_j`.
- (A2) Parameters are held at their current fits during the window. The hourly refits make this approximate; `Σ_k` carries their uncertainty.

## 3. The certified miss bound

**Lemma 1 (Bhattacharyya increment).** Under (A1)–(A2), the log-likelihood-ratio increment of L over j in the window is `Λ_j ~ N(q_j/2, q_j)`. Hence, for λ ∈ [0, 1],

$$
\mathbb E\!\left[e^{-\lambda \Lambda_j}\right] = e^{-\lambda(1-\lambda)\,q_j/2},
\qquad\text{in particular}\qquad
\mathbb E\!\left[e^{-\Lambda_j/2}\right] = e^{-q_j/8}.
$$

At λ = 1 the expectation is 1: the rival's odds are a martingale under the truth.

If the two hypotheses have different covariances, the exact Bhattacharyya coefficient carries an extra factor `exp(−½ ln det S̄ + ¼ ln det S_L + ¼ ln det S_j) ≤ 1`, so `e^{−q_j/8}` remains an upper bound. Dropping that log-determinant term is the same step as in manuscript Section IV-D.

**Theorem 1 (certified miss probability).** With `E_j^+ = E_j + β_j + Λ_j`,

$$
\boxed{\;
\Pr\!\Big(\textstyle\sum_{j\in R} e^{-E_j^+} > t \;\Big|\; H_L\Big)
\;\le\;
B(u) \;=\; \frac{1}{\sqrt t}\sum_{j\in R} \exp\!\Big(-\psi_j(u)\Big),
\qquad
\psi_j(u) = \frac{E_j+\beta_j}{2} + \frac{q_j(u)}{8}.
\;}
$$

*Proof.* Square roots are subadditive, so `Σ_j x_j > t` implies `Σ_j √x_j > √t`. By Markov's inequality and Lemma 1,

$$
\Pr\Big(\sum_j e^{-E_j^+} > t\Big) \le \Pr\Big(\sum_j e^{-E_j^+/2} > \sqrt t\Big) \le \frac{1}{\sqrt t}\sum_j e^{-(E_j+\beta_j)/2}\, \mathbb E\big[e^{-\Lambda_j/2}\big]. \qquad\square
$$

No independence between rivals is needed. Every rival enters with the weight it deserves: a rival far behind contributes `e^{−E_j/2}`, which vanishes on its own.

**Corollary 1 (v3 is one allocation of the same budget).** `B(u) ≤ δ` holds if and only if there are `δ_j ≥ 0` with `Σ_j δ_j = δ` and

$$
q_j(u) \;\ge\; 8\Big[\tfrac12\big(\ln\tfrac1t - E_j-\beta_j\big) + \ln\tfrac1{\delta_j}\Big] = 8\gamma_j\big|_{\omega=1/2} = \eta_j(\delta_j).
$$

This is exactly the v3 condition `q_j ≥ η_j`. The v3 planner fixes the allocation (an equal share `(δ − R_tail)/|A|` on A, and `q_j = 0` on the tail) instead of letting the program choose it. The optimal allocation is `δ_j = t^{-1/2} e^{−ψ_j(u)}`, the rival's own term in B.

## 4. Two readings of B

Write `ln B(u) + ½ ln t = log Σ_j e^{−ψ_j(u)} = −smin_j ψ_j(u)`, a smooth minimum (log-sum-exp). Minimising B therefore means raising the smallest `ψ_j`, smoothly:

- **Weak excitation (`q_j ≪ 8`, the R1 regime).** To first order

  $$
  B(u) \approx B(0) - \tfrac18 \sum_j \varpi_j\, \big(q_j(u)-q_j(0)\big),\qquad \varpi_j = t^{-1/2} e^{-\psi_j(0)} ,
  $$

  so the planner maximises a separation weighted by the rivals' square-root posterior odds: a Bayesian average design.
- **Strong excitation.** B is dominated by the worst rival, and minimising B tends to `max_u min_j (E_j + β_j + q_j(u)/4)`: a max–min design that stops paying for rivals that are already separated.

The same objective thus interpolates between an average-case design and a worst-rival design, depending on how much flexibility the hour offers.

**Relation to a committee heuristic.** "Query by committee" maximises `Σ_k w_k ‖μ_k(u) − μ̄(u)‖²` with posterior weights `w_k ∝ e^{−E_k}`. That is the weak-excitation reading with three differences:
- weights `e^{−E}` instead of `e^{−E/2}`;
- no whitening by `S̄_j`, so no parameter uncertainty;
- no saturation.

In weak-flexibility hours the two should choose similar blocks. They should differ when some rivals are already separated or when parameter uncertainty differs between rivals.

## 5. The planner: one program with an exact penalty

$$
\boxed{\;
u_h^\star \in \arg\min_{u\in\mathcal U_h}\; c_h(u) + V\,\big[B(u)-\delta\big]_+
\;}
$$

- `𝒰_h` is the excitation window of the hour (headroom and energy, robust voltage, transformer, unbalance allowance, linearisation remainder; hourly and daily caps). `c_h(u)` is the contract cost (3).
- `V` (CNY) is the **price of certainty**: what a timely confirmation is worth, per unit of miss probability. Default: the daily cap, 50 CNY.
- **Guarantee mode.** If some feasible `u` has `B(u) ≤ δ`, the hinge is an exact penalty: for `V` above the constraint's multiplier, the solution is the cheapest block meeting the certified guarantee, as in v3.
- **Best-effort mode.** If no feasible `u` reaches δ, the program minimises `c_h(u) + V·B(u)`. It buys only while a unit of cost lowers the miss bound by at least `1/V`, and it never buys when no block lowers the bound.
- **Certificate.** Every sealed block reports `B(u*)`. When `B(u*) < 1` this is a certified upper bound (under A1–A2) on the probability of failing to confirm within `H_c` hours. When `B(u*) ≥ 1` the bound is vacuous, and the program works as the weak-excitation design of Section 4.

**Solution method.** For a dictionary shape `s` and level κ, `Δμ_j = d_j + κ g_{js}`, so

$$
q_j(\kappa) = a_j + 2 b_{js}\,\kappa + c_{js}\,\kappa^2 ,
$$

with `a_j = d_jᵀS̄_j⁻¹d_j`, `b_{js} = g_{js}ᵀS̄_j⁻¹d_j` and `c_{js} = g_{js}ᵀS̄_j⁻¹g_{js}`. One Cholesky factor per rival and gate serves every shape and level, so the search over shapes × levels is a one-dimensional scan per shape (development code: `Gate.plan_certified`). `S̄_j` is evaluated at the forecast operating point.

## 6. Limiting the guarantee to the nearest rivals (Occam window)

**Proposition 2 (far rivals need no probing).** Let `π_j(τ) = (N(τ)/N(g))^{(p_L − p_j)_+/2} ≥ 1` be the penalty drift of a rival with fewer parameters than the leader (`π_j ≡ 1` otherwise). Under (A1)–(A2), `Z_j(τ) = e^{−E_j(τ)}/π_j(τ)` is a nonnegative supermartingale under H_L: its likelihood-ratio part is a martingale (Lemma 1 at λ = 1), and the remaining penalty factor never increases. For any set `W ⊆ R`, the outside mass `M_out(τ) = Σ_{j∉W} Z_j(τ)` is then a nonnegative supermartingale too, and Ville's inequality gives

$$
\Pr\Big(\sup_{\tau\ge g} M_{\text{out}}(\tau) \ge \frac{M_{\text{out}}(g)}{\delta_{\text{out}}}\Big) \le \delta_{\text{out}} .
$$

With `π̄ = max_j π_j` at the end of the day, the outside rivals keep their combined odds `Σ_{j∉W} e^{−E_j(τ)} ≤ π̄ M_out(g)/δ_out` for the rest of the day, without any probing. It remains to certify the inside:

$$
B_W(u) = \frac{1}{\sqrt{t_W}}\sum_{j\in W} e^{-\psi_j(u)} \le \delta-\delta_{\text{out}},
\qquad
t_W = t - \frac{\bar\pi\, M_{\text{out}}(g)}{\delta_{\text{out}}} > 0 .
$$

`W` is the smallest set of nearest rivals with `π̄ M_out(g) ≤ δ_out t/2`, so `t_W ≥ t/2`. Theorem 1 is the case `W = R`. The development code uses the full sum over the 20 nearest rivals and holds the rest at `q_j = 0`, which is conservative.

## 7. The trust gate (decided 2026-11-22)

Probe only when both hold:
1. The bank's closure completed: no cap bound, and no candidate within the margin is left unexpanded.
2. The leader is adequate:

   $$
   \rho_L = \frac{\mathrm{SSE}_L}{(N - p_L - n)\,\sigma_y^2} \le 1 + z_{0.99}\sqrt{2/N_{\text{eff}}},
   \qquad
   \sigma_y^2 = 4\sigma_V^2\big(V^4+V_0^4\big).
   $$

   `σ_V` is the voltage-meter class, and `N_eff` is the number of time slots, which is conservative because the root error is shared by every terminal. In the development runs, with offsets, `ρ_L` had a median of 0.98–1.01 and never exceeded 1.16, so the development threshold is 1.2.

The gate matters because probing sharpens whichever candidate leads. When the truth is missing, best-effort probing raised wrong confirmations from 44% to 76% (DEVELOPMENT_RUNS §12). Theorem 1 is conditional on H_L, and the gate is the check on that condition that the data can provide. Near-equivalent wrong leaders pass it; their voltage error stayed below 0.2%.

*Fresh-seed result (DEVELOPMENT_RUNS §18, 2026-11-23): the gate is not enough.* With the truth removed, the strong greedy still confirmed a wrong topology in more runs than passive (64% against 60%, 37 to 17, p = 0.009), and the gate let probing run on 55% of truth-removed days against 58% of natural ones. At the gates where it probed, ρ_L had the same distribution with and without the truth (median 0.97; ROC area 0.48–0.52). A near-twin of the truth fits the meters as well as the truth, so no threshold on ρ_L can separate the two cases. The extra wrong confirmations were near-twins, at most 0.14% off in voltage on short lines and 0.27% under meter stress.

*A prediction check does not help either (development run, tag `preq`).* Each candidate, fitted before hour g−1, predicts that hour's readings; the statistic r'(I + A G⁻¹ A')⁻¹ r / σ² has expectation equal to the number of readings when the candidate is right, and is summed separately over probed and passive hours. Under the truth it is calibrated (ratio 1.00), but with and without the truth the leader's ratio has the same distribution (ROC area 0.47–0.58). The reason is power, not scaling: the twin's excess E ≈ 3–10 over a day is spread over 60–120 readings, whose χ² sum has a standard deviation of 11–15. A test of the leader alone cannot see it; only a comparison with an alternative inside the bank can. So a missing truth must be detected by building the missing alternatives (re-closing the bank around the leader with the probe data), not by testing the leader's fit.

*Condition 1 is the part that works (DEVELOPMENT_RUNS §19).* When the bank construction itself missed the truth, its closure hit a cap in 51 of 62 inputs, so the gate forbade probing and probing changed nothing there. Every truth-removed run that probing turned wrong came from a completed bank, a state that arose naturally in 2.4% of probe-eligible inputs. Condition 2 never acted (ρ_L ≤ 1.05 throughout).

*Decision (2026-11-23, user).* The gate is kept as it is, the selector is locked, and the blind spot (a completed bank without the truth) is disclosed with its estimated size, about 0.1 point of extra wrong confirmations against 11 points of extra correct ones (PROTOCOL §11). Re-closing around the leader stays future work.

## 8. Empirical test (DEVELOPMENT_RUNS §13)

In a pre-registered comparison over four cases (214 paired ambiguous runs), the certified planner did significantly better than passive (20 gained, 1 lost) and than random blocks of the same cost (17 gained, 1 lost). It did not beat a query-by-committee rule at the same hours and budget (6 gained, 3 lost, p = 0.51), and it cost 1.4–1.6 times as much in three of the four cases.

Under household-scale flexibility the certificate is vacuous in nearly every hour, so the planner works in the weak-excitation regime of Section 4, where the committee rule approximates it. The formulas remain useful for three things:
- explaining why a guarantee is infeasible at this flexibility (Corollary 1);
- reporting a certificate when flexibility is ample;
- a principled stopping rule (buy only while the bound falls by at least cost/V).

They do not, on this evidence, choose better blocks than a simple targeted rule.

## 10. A stronger greedy: one-step Bayes-risk minimisation

**What "greedy" means here.** A greedy planner decides one hour at a time. At each gate it buys the single block with the best immediate value, given everything measured so far, and does not plan later hours. The crude greedy of §13 (`greedy_P`) maximised the predicted voltage gap between the leader and its nearest rival at full headroom. The strong greedy below keeps the one-hour-at-a-time structure but optimises a quantity with a meaning: a bound on the probability of naming the wrong topology class, traded against the money spent.

### 10.1 Quantities at gate g (all computed by `Gate.plan_bayes_greedy`)

- **Data model.** Each terminal i measures `y_i = V_0² − V_i²` every 15 min. A candidate h predicts `μ_h(u) = μ_h + M_h u` for the next hour, where `μ_h` is its prediction under the forecast background load and `M_h` its operating-point sensitivity to a block `u ∈ ℝⁿ` (kW per terminal, held for the hour).
- **GLS whitening.** Every terminal subtracts the same root reading, so each slot's errors share a component. `W` subtracts γ times the cross-terminal mean in every slot, with `γ = 1 − 1/√(1+nρ)` (§11). All differences below are whitened: `Wx`.
- **Posterior weights.** `J_h = SSE_h/(2σ̂²) + ½ p_h ln N` is the BIC of candidate h refitted on all data so far (whitened SSE; `p_h` counts the n meter offsets). `E_h = J_h − J_L`, and `w_h = e^{−E_h} / Σ_k e^{−E_k}`.
- **Operational classes.** Two candidates are operationally equivalent when their full-day predicted voltages differ by at most `ε_V = 5·10⁻⁴` pu at every terminal and slot (manuscript (19)). The leader's class `C_L` is exactly the set the stopping rule treats as equivalent to L. The other candidates with `E_h < 25` are grouped around representatives by the same test, at most `M = 10` classes. Class c has weight

  $$W_c = \sum_{h \in c} w_h \quad(\text{renormalised over the kept classes}),$$

  and its representative `r(c)` (the member with the largest weight) supplies its predictions.
- **Separation of two classes.**

  $$
  q_{cc'}(u) = \big(W\Delta\mu_{cc'}(u)\big)^{\!\top} \bar S_{cc'}^{-1}\, \big(W\Delta\mu_{cc'}(u)\big),\qquad
  \Delta\mu_{cc'}(u) = \mu_{r(c)}(u) - \mu_{r(c')}(u),
  $$
  $$
  \bar S_{cc'} = \sigma^2 I + \tfrac12 \Phi_{r(c)}\Sigma_{r(c)}\Phi_{r(c)}^{\top} + \tfrac12 \Phi_{r(c')}\Sigma_{r(c')}\Phi_{r(c')}^{\top},
  $$

  over the next hour's 4 slots × n terminals. `Φ_h Σ_h Φ_hᵀ` is the prediction variance from candidate h's parameter uncertainty (whitened design `Φ_h`, covariance `Σ_h` of its R/X fit), so unknown impedances count as noise rather than signal. For `u = κ s`, `q_{cc'}(κ s) = a + 2bκ + cκ²` with `c ≥ 0`.
- **Cost of a block (realistic DR settlement, DEVELOPMENT_RUNS §16).** For one hour,

  $$c(u) = \pi_\downarrow \sum_i u_i^- + \pi_\uparrow \sum_i u_i^+ \quad\text{CNY},\qquad \pi_\downarrow = 4,\ \pi_\uparrow = 1.2\ \text{CNY/kWh},$$

  with `u_i^- = max(−u_i, 0)` the reduction (battery discharge or paused EV charging) and `u_i^+` the increase (extra charging) at terminal i. The v3 draft contract `c(u) = Σ_i π_i u_i²/ū_i` is kept as an option (`cost_model="contract"`).
- **Admissible blocks.** `𝒰_g = {κ s : s ∈ 𝒟, 0 ≤ κ ≤ κ̄(s)}`. The dictionary 𝒟 has single-home blocks `±e_i` and group blocks on the terminals whose paths differ between the leader and its three nearest rivals. The ceiling κ̄(s) is the smallest of: enrolled headroom and battery energy, the voltage band 0.96–1.035 pu held robustly over the plausible candidates (z = 2.33 under an 8% forecast error), the transformer rating, the single-phase unbalance limit, the linearisation limit, and the budget (20 CNY in the hour, and the day's remaining budget spread over the remaining hours).

### 10.2 The risk bound and the rule

**Lemma 2 (Bhattacharyya bound on the class-level misidentification risk).** Treat each operational class as one hypothesis with prior `W_c`; its members predict the same voltages within ε_V, so they are not separable in operation. After the next hour's data, name the MAP class. Then

$$
\Pr(\text{wrong class}) \;\le\; U(u) \;=\; \sum_{c<c'} \sqrt{W_c W_{c'}}\;\exp\!\Big(-\frac{q_{cc'}(u)}{8}\Big).
$$

*Proof.* Let `a_c(y)` be the posterior of class c after data y and `m(y)` the MAP class. The error mass at y is `1 − a_m ≤ Σ_{c<c'} min(a_c, a_{c'})`, because each class other than m appears in a pair with m, and there `min = a_c`. Integrating against the predictive density gives `Σ_{c<c'} ∫ min(W_c p_c, W_{c'} p_{c'}) ≤ Σ_{c<c'} √(W_c W_{c'}) ∫ √(p_c p_{c'})`, using `min(x, z) ≤ √(xz)`. For Gaussians with covariances `S_c`, `S_{c'}` the Bhattacharyya coefficient is `exp(−q/8 − ½ ln(det S̄ / √(det S_c det S_{c'})))` with `S̄` their average, and the log-determinant term is non-negative, so the coefficient is at most `e^{−q_{cc'}/8}`. □

**The strong greedy (`bgreedy_P`).** At every gate where the leader is not yet confirmed and the trust gate passes, solve

$$
\boxed{\;u_g^\star \in \arg\min_{u \in \mathcal U_g}\; c(u) + V\,U(u)\;}
$$

with V the price of certainty (CNY per unit of risk): V = 1,000 CNY in DEVELOPMENT_RUNS §16, about one field verification visit; `bgreedy_lowV_P` uses V = 200 CNY. Equivalently, **buy block u only if the risk it removes is worth its price**,

$$
V\,\big(U(0) - U(u)\big) \;>\; c(u),
$$

and among such blocks take the one with the largest net value. If no block qualifies, buy nothing this hour. In code: every shape s ∈ 𝒟, every level κ on a 24-point grid of `(0, κ̄(s)]`, one Cholesky factor per class pair, about 20 s per day-long run.

### 10.3 Why this is better than the simpler rules

Compared with the crude greedy and with query by committee, it:
- counts only pairs of different operational classes, so it never pays to separate topologies that give the same voltages;
- gives each class its whole posterior mass, so a leader with many equivalent twins still faces its real rivals (the 2026-11-22 fix, DEVELOPMENT_RUNS §16);
- weights pairs by `√(W_c W_{c'})`, so it attends to every plausible pair, not only the leader against one rival;
- whitens for the shared root-meter error (§11) and counts parameter uncertainty as noise;
- saturates (`e^{−q/8}`), so it moves on from pairs that are already separated;
- prices every kWh against the risk it removes, so it stops by itself.

**Proposition 3 (why greedy is near-optimal).** Fix a menu of candidate blocks and freeze sensitivities and weights. Suppose separations add over executed blocks, `q_{cc'}(A) = q_{cc'}(0) + Σ_{b∈A} δq_{cc'}(b)` with `δq ≥ 0`. Then the risk reduction

$$
F(A) = U(\varnothing) - U(A) = \sum_{c<c'} \sqrt{W_c W_{c'}}\Big(e^{-q_{cc'}(0)/8} - e^{-q_{cc'}(A)/8}\Big)
$$

is a non-negative sum of concave functions of modular functions, hence **monotone submodular** in the set A of executed blocks.
- Greedy selection is therefore within a factor `1 − 1/e` of the best schedule of the same length (Nemhauser, Wolsey and Fisher, 1978).
- Under a cost budget, cost-benefit greedy combined with the best single block achieves a constant factor (Krause and Guestrin, 2005).

Two approximations separate this from the executed policy.
- Separations of different hours add only approximately. The meter noise is independent across slots, but the parameter-uncertainty term `ΦΣΦᵀ` couples the hours, and the cross term `2bκ` with the background difference can be negative for a badly chosen sign (the greedy never picks such a block, because it would raise U).
- The hourly refit changes W and the sensitivities, which makes the policy adaptive. Near-optimality of adaptive greedy is proven for the closely related EC² objective (Golovin, Krause and Ray, 2010), not for U itself.

**Empirical record.**
- DEVELOPMENT_RUNS §15 (fresh seeds, contract prices, defective member choice): the strong greedy matched the committee and certified planners in accuracy (231 of 256 runs correct, against 232 and 233) and was the cheapest (0.47 CNY a day). Part of that thrift came from the defect: it stopped buying at 716 short-line gates where it saw only the leader's twins.
- DEVELOPMENT_RUNS §16 (hard cases, linear DR prices, fixed version, 550 ambiguous runs): the strong greedy beat passive 62 to 2 and random blocks 55 to 6, matched the certified planner (15 to 17) and beat the committee rule (29 to 12). V = 200 CNY was as accurate as V = 1,000 (9 to 9) at 70% of the cost. These numbers predate the battery-state fix and are about 3 points optimistic.
- DEVELOPMENT_RUNS §17 (V sweep, all fixes, the same 550 runs): every V from 100 to 2,000 CNY was 37–39% correct (passive 29%) and worth 62–72 CNY per run at 1,000 CNY per correct identification. From V ≈ 500 the probed hours saturate at 3.2 a day, as the V → ∞ limit predicts (the buy rule then passes whenever U falls, and a larger V only enlarges the block). The registered rule picks V = 2,000; V = 100 is not significantly worse at less than half the cost.
- DEVELOPMENT_RUNS §18 (fresh seeds 65–96, all fixes, 532 ambiguous runs): at V = 2,000 the strong greedy beat passive 61 to 2 (43% against 32%) and random blocks 47 to 8, matched the certified planner (21 to 15) and had the largest net value (88 CNY per run). With the truth removed it confirmed a near-twin more often than passive (37 to 17), so the registered lock is not confirmed (§7). V = 2,000 beat V = 100 on the same inputs (38 to 14, p = 0.001), so the flat §17 curve was underpowered: a larger V buys larger blocks and more probed hours, and these pay off. Above V = 2,000 they stop paying: V = 100,000 on the §16 inputs was as accurate (214 against 215 of 550) at a higher cost. Against the random arm with its headroom cap fixed, the strong greedy wins 50 to 8. Locked on 2026-11-23 at V = 2,000 by the user's decision, with T1's failure disclosed (§7).

**Relations.**
- The certified bound B of §3 is the leader-centric special case: only the pairs (L, j), weighted against the leader.
- Query by committee is the weak-separation, unwhitened, class-blind limit: `Σ w_h w_h' ‖Δμ‖²` in place of `Σ √(W_c W_c') q_cc'/8`.

## 11. Correction: GLS scoring for the shared root-meter error

Every terminal's `y_i = V0² − V_i²` subtracts the same root reading. The measurement error of one time slot therefore has covariance `c(I + ρ 11ᵀ)`, with `ρ = V0⁴/V⁴ ≈ 1`; in the true model's residuals 50–52% of the variance was shared. The scorer had treated terminals as independent, which inflated σ̂² about twofold and roughly halved the evidence about internal topology.

The exact correction is generalised least squares. The inverse square root of the slot covariance is `I − γ 11ᵀ/n` with `γ = 1 − 1/√(1 + nρ)`: subtract the fraction γ of the cross-terminal mean from every slot (`model.whiten_slots`, unit-tested against full GLS). It commutes with the per-meter offsets and applies to the refit, the BIC, the adequacy ratio (whose denominator becomes `4σ_V²V⁴`) and every separation `q`.

On the main case's ambiguous inputs it raised the passive leader's accuracy from 20 to 28 of 28 and the runner-up gap from 2.3 to 4.5 BIC units (DEVELOPMENT_RUNS §14).

## 12. What the formulas do not claim

- B is conditional on the leader being the truth and on (A1)–(A2). It is not a probability that the confirmed topology is correct.
- Under scenario R1 the certificate is usually vacuous (`B ≥ 1`): the separations bought in one hour are far below what the bound needs. The planner then works as the information-weighted design of Section 4, and its value has to be shown empirically against simpler rules (DEVELOPMENT_RUNS §13).
- The program over the dictionary is solved by scanning; the minimiser over all of `𝒰_h` is not certified globally optimal.
