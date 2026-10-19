# Independent review of nominal weak-branch information bounds

Review date: 2026-10-13. Objects: the initial V1 `nominal_weak_information_math.md` and the proposed V2 that reuses the existing uniform `.107157 < .11` AC/KL bound and adds the nominal-prior TV bridge. This is a read-only mathematical/source audit; no AC cases, solvers, policies, calibration runs, or empirical error-rate estimation were executed. The only computation was exact rational arithmetic for the stated constants, plus certified rational bounds for the logarithm/square-root consequences. Frozen C/D and the research worktree were not edited.

**Verdict: V2 is mathematically valid for the declared ideal continuous nominal prior and a common admissible causal controller.** The old `.107157 < .11` proof already covers every nominal independent R/X box; only its auxiliary-to-nominal prior bridge was missing. Describing the old proof as valid only at fixed nominal lines is incorrect. V1's `.194216 < .2` derivation is algebraically valid but looser and unnecessary. One V2 display requires careful rounding: JS is at most `.185431279937...`, so a strict `.18543` upper bound would be false; `.185432` is valid. The implementation limitations below should be retained when connecting the theorem to C.

## Evidence and scope

Current source reads: `simulation.py` (nominal impedances, iid backgrounds/source, independent Gaussian meters); `adaptive_milp_study.py` (topology mass .25 each, strong probability .5, independent R/X scaling, history-only selector); `likelihood.py` (constant-power AC and actual root master power); `ev_operations_milp.py` (one EV, 6 kW, 6 kWh grid energy, pure P); `weak_branch_information.py` (exact constants and explicit `nominal_domain_checks`). The supplied `weak_branch_information_math.md` and `continuous_ac_inverse_math.md` were read as mathematical definitions; their historical experiment results were not reused as evidence.

The target theorem concerns 12 passive historical observations and 8 future observations from **one actual execution of one policy**, under nominal calibrated bias/root 1, source fixed at node 1, charge only, tracking gain 1, at most one nonzero terminal command, `0 <= u_t <= .3`, and pathwise grid energy at most 6 kWh with 20 kW base and .25 h slots. All three weak cherries and every independent nominal R/X scaling in [.8,1.2] are covered by the old KL bound and by the new TV bridge. Strong-cherry branch impedances are not covered by the weak-branch contraction constant; this is unnecessary for the four-topology risk reduction because their error contribution is simply discarded.

## Mathematical checks

1. **Reduced AC identity.** In loaded-node coordinates (node 1 plus three terminals), the pair block of the weak system is `diag(z_i,z_j)+z_s 11^T` beyond the common root. The auxiliary star pair block is `diag(z_i+2z_s,z_j+2z_s)`. Their difference is exactly `z_s [[1,-1],[-1,1]] = z_s d d^T`, where d has zero source coordinate and terminal coordinates +1/-1. Zero-load internal-junction elimination is exact. Source voltage and master power can change indirectly and are correctly bounded rather than cancelled.

2. **Existing uniform contraction.** `nominal_domain_checks` uses exact squared complex moduli, including full independent R/X extrema: root `.006^2+.0042^2 < .008^2`; weak edge `.000036^2+.000024^2 < .000044^2`; leaf `.0408^2+.018^2 < .045^2`; pair difference `.0184^2+.0092^2 < .022^2`. Background `.136^2+.047^2 < .145^2`, source `.135^2+.05^2 < .15^2`, and pair contrast `2*.008^2 < .012^2` also hold. Hence `Mtotal=.885`, `Mmax=.445`, and the common row bound is `C=.008*.885+(.045+2*.000044)*.445=.02714416`. It satisfies `C/.9 < .1` and `K=C/.9^2=.0335113086419753 < 1`. This is uniform over the nominal boxes, not a fixed-nominal-point proof. The zero-load weak junction reconstructs inside the same ball. Lower-voltage equilibria are not enumerated.

3. **Existing sharper current contrast.** In each pair, choose j to be the uncharged endpoint; when neither endpoint is charged, choose either. This relabeling changes only the sign of the rank-one contrast, not its norm or any physical parameter. Then `|S_j| <= Mb=.145` and `|I_j| <= Mb/.9`. The identity `V_i-V_j=-z_i(I_i-I_j)+(z_j-z_i)I_j` gives `|V_i-V_j| <= .045 L+.022 Mb/.9`. Combining it with `L <= (.012+u_t)/.9+Mb |V_i-V_j|/.9^2` yields `L_den=1-Mb*.045/.9^2=3571/3600`, `L0=203651/14462550`, `L1=4000/3571`. No u-dependent denominator is required. At most one charging terminal is the decisive premise; the EV may be outside the pair.

4. **Existing voltage/master difference.** The rank-one perturbation acts only on `I_i-I_j`; contraction of the auxiliary star gives `delta_t <= .000044*(L0+L1*u_t)/(1-K)`. Equal complex nodal powers and root 1 give `|Delta S0| <= (sum |S_k|)/.9^2*delta_t <= Gamma*delta_t`, including the unmetered source, with `Gamma=.885/.9^2=59/54`. Three terminal voltage amplitudes obey the same delta bound. For equal master P/Q noise, their Gaussian contribution is `(|Delta P|^2+|Delta Q|^2)/sigmaM^2 = |Delta S0|^2/sigmaM^2`; there is no extra factor 2. The exact 20-point KL computation is `.107157014860402064... < .11`.

5. **Causal KL chain and latent integration.** First augment both processes with identical latent root/leaf/shared-edge parameters, full background/source paths, and independent controller randomization. Conditioned on these and a common past, the action kernel is identical and contributes zero KL; the fresh measurement kernel is Gaussian with the stated covariance. The pathwise power/energy inequalities bound the sum of conditional KL terms. Integration of the latent coordinates is data processing and decreases KL. The marginal adaptive mixture need not be Gaussian. This argument requires noise-independent, nonanticipating control and no extra truth-dependent feedback or abort channel.

6. **Auxiliary-to-nominal bridge.** Conditional on z_s, the four independently uniform leaf R/X coordinates are translated by `(2r_s,2x_s)` on each of the two leaves. All shifts are smaller than their box widths; equal-size uniform-box TV is exactly one minus the overlap-volume fraction. Root, remaining leaf, backgrounds/source and controller randomization have common laws. Attaching an independent z_s to nominal star, conditioning, and applying the common star AC/controller kernel gives TV no larger than the box-TV bound. The dependence between the two translated leaves after integrating z_s is harmless. Auxiliary star is correctly not asserted to have the nominal star prior.

7. **Binary risk and JS.** Equal-prior forced binary error is at least `(1-TV)/2`. For `m=(p+q)/2` and `t=p/(p+q)`, concavity gives `h(t) >= log(2)*(1-|2t-1|)`. Consequently `JS <= log(2)*TV`. Chain decomposition bounds the population average `I(label;future|history)` in the same equal-prior binary experiment. It cannot bound every realized-history conditional information or the four-label mutual information.

8. **Full nominal POP risk.** Star's generated law is unaffected by the independent strong/weak flag, since star has no internal shared edge, so its complete mass is .25. Each weak clade has mass `.25*.5=.125`. Their normalized mixture has weights 1/3 and TV to star at most epsilon by convexity. Map a forced four-topology output to “star” iff it outputs star. Every star mistake is a group mistake; every weak group mistake is a topology mistake. Therefore total topology risk is at least `integral min(.25 pstar,.375 pweakmix) >= .25 integral min(pstar,pweakmix) = .25*(1-TV)`. The .375 strong mass adds only nonnegative risk. The stated lower bound therefore follows.

## Exact constants

The companion `nominal_weak_review.json` stores exact fractions, high-precision decimals, input SHA-256 hashes, and strict rational assertions for both V2 and the initial V1. Adopted V2 values:

| Quantity | Reviewed value |
|---|---:|
| AC row bound C | .02714416 |
| K | .03351130864197530864 |
| Current denominator | 3571/3600 |
| L0 | .01408126506044922922 |
| L1 | 1.120134416129935592 |
| a | 6.410583674695593115e-7 |
| b | .00005099481737387423016 |
| Gamma | 59/54 |
| Gaussian weight | 209527091.9067215364 |
| KL upper bound | .10715701486040206426 < .11 |
| Pair 01 box-TV upper bound | .03199242510263929619 < .033 |
| Pair 02 box-TV upper bound | .03029047189610389610 < .033 |
| Pair 12 box-TV upper bound | .02803682809049773756 < .033 |
| epsilon = sqrt(.055)+.033 | .26752078799117147773 < .267521 |
| Equal binary error lower bound | .36623960600441426114 > .3662395 |
| epsilon log(2), nat | .18543127993725538509 < .185432 |
| Four-topology POP error lower bound | .18311980300220713057 > .18311975 |

The existing KL bound is exactly `3668918839949454447125 / 34238718246575913269706`. Decimal strings alone were not used to certify strict inequalities. `sqrt(.055) < .234521` follows by exact squaring. For `x=.693148`, the positive exponential Taylor partial sum through degree 9 is strictly greater than 2 by exact fraction comparison, so `log(2)<x`. Their outward product is `.267521*.693148=.185431646108 < .185432`.

V1 independently reproduces KL `.19421669348476018197 < .2`, epsilon `.34922776601683793320`, equal binary error `.32538611699158103340`, JS `.24206624138781949541`, and full POP error `.16269305849579051670`. These are correct but are not adopted in V2. The V1 draft's assertion that the old KL proof covered only fixed nominal lines must be corrected; it is contradicted by the old exact source checks.

## Clarifications required for an implementation claim

These do not overturn the displayed theorem; they delimit how it can be attached to C.

- When keeping V1 for provenance, explicitly distinguish Euclidean complex modulus from the `p+q` triangle envelope: the `.012` background contrast is certified by `2*.008^2 < .012^2`, not by `.008+.008`. V2 already uses the sharper Euclidean envelopes.
- Specify that every edge's R/X parameters are independent uniform coordinates **sampled once and shared across all 20 time points**; c, eP, eQ and source have the declared independent uniform per-time laws. Define d's source coordinate as zero to remove terminal-index ambiguity.
- The complete observed transcript should contain exactly the meters and the device/control state available to the common causal controller, including recorded actions/randomization if they are used in classification. It must not contain evaluator truth arrays, true source/impedance values, or hidden-state failure signals. A controller may use all four nominal candidates and does not need oracle access to the auxiliary star.
- `adaptive_milp_study.py` evaluates several plans on common future noise and saves counterfactual arrays. Concatenating those arrays is a different observation experiment: common-noise contrasts can cancel noise and the stated Gaussian KL bound does not cover that bundle. Apply the result separately to a single executed policy trajectory.
- Public-history-dependent aborts can be treated as common postprocessing of the full Gaussian transcript and therefore preserve the bounds. Conditioning on completed runs or silently excluding failures need not preserve them. `Observations` rejects nonpositive measured voltage, an event with nonzero probability under ideal Gaussian noise; retain an absorbing/public abort definition or state that the theorem concerns the full raw-Gaussian ideal model rather than a success-conditioned dataset.
- The theorem assumes exact pathwise caps. The implementation's accepted feasibility tolerances (e.g. `_plan_action` uses `check_plan(..., tolerance=2e-6)`) do not by themselves prove literal `u_t<=.3` and energy<=6. An implementation check should either establish the exact executed caps or enlarge the analytic envelopes by the documented tolerances before asserting an implementation-level universal bound.
- Write “equal-prior binary population average conditional information”. The full POP classification lower bound is a separate weighted-risk statement and does not turn the equal-prior JS calculation into a four-topology EIG ceiling.

No experiment was requested or performed by this review. Uniform validity rests on the analytic envelopes and exact inequalities above, not finite endpoint samples or estimated actual rates.
