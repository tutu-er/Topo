# Continuous AC inverse and the probability coordinate change

Saved before implementation, 2026-10-13. This is an independent derivation for the declared nominal synthetic model, not a literature theorem or a field prior. No actual future record or evaluation truth is used.

## 1. Model and coordinates

There are three constant-power terminal consumptions, a calibrated slack voltage of 1 pu, and one unmetered constant-power consumption at node 1 immediately below the root edge. The four candidate families are the star and the three cherries. R and X of every edge are independent shared-in-time parameters, in project canonical edge order. A cherry has one internal shared edge, two terminal edges below it, and one terminal edge directly below node 1.

At a time point use terminal total complex consumptions `s_i=P_i+j Q_i` and actual slack/master complex power `S0=P0+j Q0` as latent coordinates. The background prior is applied to `P_i-u_i`, so the known EV consumption is added exactly once. The inverse reconstructs node-1 source consumption, terminal voltage magnitudes, and `J=d(source_P,source_Q)/d(P0,Q0)`. Quantities are per unit.

The nominal background is `c~U[-.012,.012]`, `eP_i,eQ_i~U[-.004,.004]` independently, `pbase_i=.12+c+eP_i`, `qbase_i=.04+.25c+eQ_i`. The independent source prior is `source_P~U[.065,.135]`, `source_Q~U[.020,.050]`. All are iid across time conditional on the shared topology and edge parameters. Allowed actions here have at most one nonzero terminal EV P per time, `0<=u<=.3`, and zero EV Q. Hence `.104<=P_i<=.436`, `.033<=Q_i<=.047`, `sum P_i<=.708`, `sum Q_i<=.141`.

The proof below uses a conservative superset of the nominal impedance prior: `0<=r0<=.006`, `0<=x0<=.0042`, `0<=rs<=.0216`, `0<=xs<=.0108`, and `0<=ri<=.0408`, `0<=xi<=.018`. Canonical nominal strong shared values from simulation are `.012,.007`, whose scaled maxima `.0144,.0084` are smaller. Weak shared values are also covered. The larger shared limits are used only to make the proof robust to the already-declared envelope. This proof does not cover larger actions, reactive EV injection, negative net loads, or wider impedance/source priors.

## 2. Exact high-voltage inverse

Let `z0=r0+j x0`. Since the slack voltage is real 1, root-edge current is `conj(S0)`, node-1 voltage is `V1=1-z0*conj(S0)`, and

`tau=|V1|^2=1-2(r0 P0+x0 Q0)+|z0|^2(P0^2+Q0^2)`.

For a terminal receiving complex power `s` at the end of an edge `z` from a parent of squared magnitude `t`, put `k=|s|^2`, `h=Re(z conj(s))`, `A=t-2h`, `D=A^2-4|z|^2 k`. On the high-voltage branch,

`tleaf=(A+sqrt(D))/2`,

`l=|I|^2=2k/(A+sqrt(D))`,

`Sflow=s+z*l`,

`dl/dt=-l/sqrt(D)`.

The stable rational form avoids cancellation when the load is small. The zero-load limit is `l=0`, `tleaf=t`. `D>0`, `A>0`, and positive receiving voltage are explicit validity conditions. Differentiating `k=tleaf*l` or the quadratic gives the last identity. Choosing the plus square root is a branch definition, not a numerical convergence claim.

For a star `Sdown(t)=sum_i Sflow_i(t)` and `Sdown'(t)=sum_i z_i*l_i'(t)`.

For a cherry, at junction squared voltage `u`, define `L(u)=Sflow_a(u)+Sflow_b(u)`. The shared-edge equation is

`G(u)=u+2 Re(zs conj(L(u)))+|zs|^2 |L(u)|^2/u=t`.

Its derivative is

`G'(u)=1+2 Re(zs conj(L'(u)))+|zs|^2[2 Re(conj(L(u))*L'(u))/u-|L(u)|^2/u^2]`.

The shared-edge sending power is `B(u)=L(u)+zs |L(u)|^2/u`. Thus `du/dt=1/G'(u)` and

`dB/dt=[L'(u)+zs(2 Re(conj(L(u))*L'(u))/u-|L(u)|^2/u^2)]/G'(u)`.

Add the separately computed third terminal flow to obtain `Sdown(t)` and its derivative.

Root-edge receiving power is `S0-z0|S0|^2`. Therefore the reconstructed source is

`Fsrc(S0)=S0-z0|S0|^2-Sdown(tau(S0))`.

In real coordinates, with `d=(Re Sdown', Im Sdown')^T`, `a=|z0|^2`,

`grad tau=(-2r0+2aP0, -2x0+2aQ0)`

and

`J=I-2(r0,x0)^T(P0,Q0)-d*(grad tau)`.

The requested density Jacobian is `abs(det J)`, not its reciprocal. It maps master coordinates back to source coordinates.

## 3. All declared prior inputs have a unique high-voltage solution inside the rectangle

This section is a mathematical uniform bound; random checks are not used as a proof. Use rounded-outward envelopes `|z0|<.0074`, `|zs|<.0242`, `|zi|<.0446`, one EV-loaded terminal `|s|<.439`, each other terminal `|s|<.144`, and `|source|<.144`. In the project bus-impedance fixed-point equation `V=1-Zbus conj(S/V)`, the weighted maximum row sum is bounded by

`W<=.0074*.871+.0242*.583+.0446*.439=.0401334<.041`.

The term for the shared path is absent for the star and can only get smaller when the EV is outside the cherry. On the closed complex sup-norm ball `|V-1|<=.05`, all `|V|>=.95`. The AC map takes the ball into itself since `W/.95<.044<.05`; its Lipschitz constant is at most `W/.95^2<.046<1`. Banach's theorem gives a solution and convergence from all-ones. Any two solutions whose voltage magnitudes are at least `.9` obey the direct difference bound `||V-W||<=.041/.9^2 ||V-W||`; they are therefore identical, even though that magnitude domain is not convex.

All consumptions and R/X are nonnegative. From conservation, slack P/Q are nonnegative and

`P0<=.135+.708+.006*(.871/.95)^2+.0216*(.583/.95)^2+.0408*(.439^2+2*.144^2)/.95^2<.867<1`,

`Q0<=.050+.141+.0042*(.871/.95)^2+.0108*(.583/.95)^2+.018*(.439^2+2*.144^2)/.95^2<.204<.5`.

Thus every declared source/background/impedance/action prior input maps into the explicit rectangle `R=[0,1]x[0,.5]`, with every AC voltage magnitude at least `.95`. The generative model selects this unique high-voltage solution. Possible low-voltage mathematical roots are not part of that model.

## 4. The inverse is well defined and injective on the entire rectangle

On `R`, root squared voltage obeys `tau>=1-2(.006+.0042*.5)=.9838`. No upper bound on tau is needed; this keeps the proof valid for the conservative superset permitting zero individual R/X components as well as the nominal positive ranges.

At any potential cherry junction `u>=.94`, the EV-loaded leaf has `k<.193`, `h<.0187`, `A>.9026`, `sqrt(D)>.9017`, `tleaf>.9021`, `l<.214`, and `|l'|<.238`. A base-only leaf has `k<.021`, `h<.0064`, `A>.9272`, `sqrt(D)>.9271`, `l<.02266`, and `|l'|<.0245`. The largest two-leaf cherry occurs when one of its leaves has the EV; then

`Re L<.581656`, `Im L<.09826`, `|L|<.590`, `|L'|<.01171`.

Consequently `G(.94)<.968<.9838<=tau`, while `G(tau)>=tau`. Moreover for every `u>=.94` under consideration,

`G'(u)>=1-2*.0242*.01171-.000584*(2*.590*.01171/.94+.590^2/.94^2)>.9991`.

There is exactly one junction root in `[.94,tau]`; its high shared-edge branch is confirmed by `u>|zs|*|L|` (`.94>.0242*.590`). Every terminal has `|V|>sqrt(.9021)>.949`. The star branch is still further from its discriminant boundary.

The shared sending-power derivative is bounded by `.0217`. The directly attached leaf derivative is bounded by `.0446*.238<.0107`. Hence for either star or cherry `|Sdown'|<.033`. For all `S0 in R`,

`||J-I||_infinity <= 2*.006*(1+.5)+.033*(2*(.006+.0042)+2*.000054*(1+.5)) < .019 <1`.

Because `R` is convex, integrating this derivative along any segment shows `Fsrc-identity` is `.019`-Lipschitz in the real infinity norm. Thus

`||Fsrc(x)-Fsrc(y)||_infinity >= (1-.019)||x-y||_infinity`,

and the inverse source map is globally injective on `R`. Its determinant is strictly positive: `I+t(J-I)` is nonsingular for `0<=t<=1`, and determinant cannot cross zero from the identity. This is a uniform branch and change-of-variables certificate for the stated envelope, not just a pointwise finite-Jacobian check.

Every source prior point is reached by the forward solution from Section 3, and the inverse reproduces it by high-voltage uniqueness. Restricting the transformed integral to `R` therefore loses no prior mass. No multiple-branch count is required.

## 5. Proper prior densities and proposal cancellation

For a given terminal background vector `(p,q)`, intersect all these c intervals:

`[-.012,.012]`,

`[p_i-.12-.004,p_i-.12+.004]`,

`[4*(q_i-.04-.004),4*(q_i-.04+.004)]`.

If their intersection length is `L_c`, the exact marginal six-dimensional density is

`p_background(p,q)=L_c/(.024*.008^6)`.

There is no additional `.25` Jacobian: this expression integrates the original conditional density with respect to c. Empty intersections give zero density. The source density is `1/(.070*.030)` inside its declared rectangle and zero outside.

For fixed edge parameters and topology the full coordinate map leaves terminal P/Q unchanged. Its Jacobian is block triangular, so its determinant is `det J` despite the dependence of source on terminal P/Q. At each time the transformed density is

`p_background(P-u,Q)*p_source(Fsrc(P0,Q0))*abs(det J)`

on the master rectangle and zero outside. Independence across time gives a product over time; shared edge/topology prior is integrated only once.

Proposal terminal P/Q independently from Gaussians centered at their observed bias-corrected readings with exactly their known measurement standard deviations; proposal master P/Q likewise around their actual observed readings. These normalized proposal densities equal the normalized P/Q and master measurement likelihoods exactly, so they cancel in the importance ratio. The remaining integrand is voltage likelihood times background prior times source prior times `abs(det J)` and the explicit master-rectangle indicator. This cancellation does not permit omitting proposal density for any edge-parameter/history-fitted proposal, and does not turn an MLE into a prior. The calibrated bias and known root appear once. WZZT is not an additional independent measurement likelihood.

## 6. Implementation acceptance and limitations

The planned engine accepts arbitrary NumPy-broadcastable leading axes with final edge or terminal axis. It returns the terminal magnitudes, inferred source, signed/absolute determinant, real 2x2 Jacobian, and separate numerical/high-branch validity masks. The marginal-prior helpers support the same leading axes. A vectorized forward engine provides declared-prior hypothetical generation without actual future data.

Acceptance requires comparison to project `likelihood.ac_predict`, independent real central finite differences of the source inverse, forward/inverse round trips, support boundaries/background-density normalization, rectangular high-branch checks, and recording code hashes/runtime/seeds. Finite tests verify implementation accuracy; Sections 3-4, rather than those tests, justify uniform properness. Importance-weight stability and predictive-EIG precision remain separate sampler obligations. Changes beyond the proven envelope require a new proof or explicitly narrower probability model.

Implementation verification completed in staging: 20 pytest checks passed. Eight independent topology/strong-weak groups with 37 states each gave maximum inverse voltage error `9.94e-14`, source error `2.48e-15`, real finite-Jacobian error `8.07e-11`, and independently computed project-forward reciprocal-Jacobian error `7.29e-11`. A transformed source-prior normalization sanity check on 400/800 midpoint grids gave star mass `1.001110/1.000278` and cherry mass `1.001410/1.000101`; the finite quadrature supports implementation consistency and is not the uniform proof. Runtime, seeds, exact residuals, and SHA-256 identities are in `continuous_ac_inverse_validation.json`; the reproducible driver is `validate_continuous_ac_inverse.py`.
