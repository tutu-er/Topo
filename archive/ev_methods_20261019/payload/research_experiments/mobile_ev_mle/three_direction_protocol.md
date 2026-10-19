# Three EV design directions: frozen staged protocol

Date: 2026-10-12. This extension is separate from the existing 80-trial study.
The initial question is whether three explicit criteria choose useful, feasible
actions in a fully auditable small composite AC model, and whether that choice
survives enlargement to the existing continuous AC fit.

## Order and gates

1. Complete notation, derivations and counterexamples for symmetric distance,
   rejection power and Bayesian topology information. State the different
   uncertainty assumptions before comparing results.
2. Verify each direction on independent analytic Gaussian examples, in that
   order. Establish symmetry/monotonicity, finite Monte Carlo test validity,
   and the entropy/information identity respectively.
3. Run all three, in that order, on the same three-terminal finite composite AC
   family. Exhaustive finite searches are verification references, not claims
   that the continuous action/parameter problem has been solved.
4. Refit the selected actions with the existing complete continuous AC fitter,
   retaining free separate R/X and per-time terminal totals/common source.
   Report transfer loss gaps and parameter errors separately. Finite-family
   testing thresholds do not transfer to this larger family.
5. Extend continuous design and network size only after the information,
   calibration, numerical and uncertainty-domain checks justify expansion.
   A weak unresolved branch, zero symmetric distance, prior sensitivity, or
   inconsistent finite-versus-continuous ranking is an outcome, not a reason
   to silently change the test or remove a candidate.

## Common physical and meter model

Reuse likelihood.ac_predict (balanced radial constant-power AC). A line vector
is shared across H and N. Terminal totals are baseline plus the executed EV
action once. Hidden source P/Q varies in time and always enters node1 beneath
the common trunk. P/Q/V and master P/Q have calibrated fixed offsets and known
independent Gaussian errors. All designs receive historical meters, public
candidate/atom metadata, feasible actions and independent simulation seeds.
Evaluation truth and realized future noise are never selector arguments.

## Finite AC audit family (external, proper and predeclared)

Two reduced candidate topologies: clade {0,1} and clade {0,2}. Each has 24 atoms:
two independent R scales (.85,1.15), two independent X scales (.85,1.15), three
internal-branch strengths (weak, medium, strong), and two predeclared joint
background/source paths. All paths have 12 historical and 8 future snapshots.
Nominal trunk R/X=.005/.0035, leaf R=.028+.003*i and X=.011+.002*i,
internal R/X=(.00003,.00002),(.006,.004),(.012,.007) pu. The shared scale
restriction is an explicitly smaller line family than the continuous fitter.
Latent path definitions and all masses are fixed before seeing history.
Default topology mass is 1/2; atoms are uniform within topology. Sensitivity
changes topology odds and weak-branch mass without changing the support.
No hard historical cutoff is applied to Bayesian updating or to null tests.

Action library reuses 11 charging schedules: 8 quarter-hour slots, 6 kWh,
6 kW maximum, 60 kWh battery, efficiency .95, SOC [.1,.9], initial SOC .4,
initial site0, travel 1 slot/.2 kWh, site2 inaccessible in the first 2 slots.
Q=0. No-EV is an equal-duration, zero-energy control. Random action uses an
independent historical-side seed. Best constant is selected by each criterion
from the feasible constants; it never uses evaluation responses.

## Direction 1: symmetric historical support

Within each topology retain atoms whose historical loss is at most its minimum
plus c=12. Sensitivity c=0,4,36 is descriptive, not a confidence level. For each
action minimize the whitened future mean distance over both retained sets.
Report closest atom pair, support sizes and paired gain versus no-EV, fixed
constant and best constant. Larger support cannot increase exact distance.
Do not let independently free future terminal totals erase the action.
The unrestricted continuous hard-history local problem is QCQP/SOCP after
linearization; a soft historical-penalty version has a QP inner problem.
Nonlinear local inner values are upper estimates of minimum separation.

## Direction 2: direct rejection power

For candidate h, Z_h is joint H+N best finite-family loss minus the overall best
finite-family loss. Every minimization reselects atom line/source coordinates,
but retains one atom across H and N. Null simulations include every atom of h.
For a null atom use the maximum Z_h over the entire feasible action library,
with common Gaussian noise across actions. Independent calibration ranks give
p_atom=(1+number of calibration maxima >= observed Z)/(B+1); p_h is their
maximum. This controls false rejection conditional on the observed H for any
true atom in the declared finite family; it is not uniform over the continuous
line/source domain. Keeping p_h>alpha yields a compatible candidate set.

Training uses separate random noise to estimate the probability of rejecting
the wrong candidate while retaining the generator. Optimize the smaller of the
two topology-specific mean powers over their historical-support atoms, and
also report per-atom worst power. Fresh final calibration is independent of
training and selection. Alpha=.1, calibration B=99, training 64 samples per
topology. Evaluate 512 fresh future noises per observed history; record
binomial intervals and MC uncertainty, not topology-correctness probabilities.

## Direction 3: topology information gain

Update the positive external atom prior with the complete historical Gaussian
likelihood using log-sum-exp. Sum atom weights to obtain topology mass. Future
predictive density is a Gaussian mixture with the same posterior atom weights;
R/X is shared rather than redrawn for N. Maximize the expected log ratio of
topology-conditional predictive density to their mixture, equivalently the
expected reduction of topology entropy. Marginalization is exact in this
finite model; only the outer Gaussian integral is Monte Carlo.
Use stratified/common noise for action comparison; compare increasing sample
counts (128,512,2048) and three independent integration seeds. Do not clip small
negative estimates. Report standard errors, entropy upper bound, posterior
atom concentration and explicit prior sensitivity. Model information and line
parameter information are different targets.

## Evaluation, provenance and stopping expansion

Predeclared histories: 66001..66003 strong and 67001..67003 weak, alternating
the two true topologies. The evaluation driver chooses a known atom solely to
generate observations and evaluate; methods see no atom index or future noise.
Use common future noise for controls and all designs, with independent streams
for history, training, calibration and evaluation. Also retain one off-grid
line/source and one fixed voltage-bias stress test as model-domain diagnostics.
Continuous transfer fits use two starts/max_nfev150 initially; failures remain
in the denominator as failures, not rejections. Record achieved local loss,
loglik, R/X, source errors, physical residuals, restarts, time and hashes.
Existing formal-study evidence and thresholds are preserved.

All source and result JSON remain under research_experiments/mobile_ev_mle/.
Compact evidence, mathematical derivations and the original report are updated
and checked. Commit only relevant paths and normally push the research branch.

## Sources and attribution

Dette--Titoff (2009), sections 2 and 4.2: rival-refit T/KL optimization.
Harman--Mueller (2020), sections 2.1--2.2: symmetric linearized mean-surface
distance with flexible nominal parameter sets. Hainy et al. (2022), section 2:
Bayesian model discrimination utilities. This project's AC/history extension,
finite-null rank proof and staged validation are independently derived.
