# Topology Identification and Conditional Demand Response Model

## Observed response

For each time slot, the model predicts the squared-voltage drop `Y = root² − V²` from per-unit active and reactive power. A support `C` is a downstream set of measured terminals. Its indicator vector is $z_C$. Its two nonnegative coefficients $r_C$ and $x_C$ contribute to the response matrices.

$$
\widehat Y_t = R P_t + X Q_t,\qquad
R = \sum_C r_C z_C z_C^\top,\qquad
X = \sum_C x_C z_C z_C^\top.
$$

A branch affects all terminals in its support and responds to their total downstream power. $R$ and $X$ are squared-voltage response matrices. Their coefficients use the power and voltage bases of the input.

## Training and structural search

`fit_observed(train, validation, method="111")` uses training data for candidate generation and parameter fitting. Recursive Neighbor Joining (RNJ) provides a deduplicated pool from three distance weightings and five tolerances. With RNJ guidance enabled, the pool supplies support proposals for W and the domain for E. One fixed RNJ family with distance weighting 0.75 and tolerance 0.16 supplies the loss reference for every method and can initialize E when RNJ guidance is enabled.

W starts at a model with singleton supports and a common support. At each step it fits each compatible pool support that the time allowance permits, keeping the best feasible start. It then searches for one unknown support while refitting all weights. The accepted point can come from a completed pool trial. T jointly rebuilds a small neighborhood when its maximum number of direct blocks is at most five. E can add or remove internal supports in a finite pool formed from the guidance pool, the accepted path, T's proposals, and all two-terminal supports. E can start from either the current point or the fixed reference when the reference lies in its domain.

The training criterion divides prediction loss by the corresponding loss of the fixed reference tree on training data, then adds a penalty for internal supports. With mean squared error (MSE), E solves

$$
\min\ \frac{\|Y-\widehat Y\|_F^2}{N M_{\rm ref}} + 0.003\sum_C a_C,
\qquad 0 \le r_C,x_C \le 2a_C,\qquad a_C\in\{0,1\}.
$$

Here $N$ counts scalar training residuals, $M_{\rm ref}$ is the reference training MSE, and $a_C$ selects internal support $C$. The selected supports are nested or disjoint. For each containing support $D$ in the pool and for the full terminal set, the number of selected proper internal subsets of $D$ is at most $|D|-2$. The singleton and common coefficients have bounds $[0,2]$ and are outside the internal count penalty.

With `--loss mae`, the training loss becomes $\sum_{t,i}|Y_{ti}-\widehat Y_{ti}|/(N A_{\rm ref})$, where $A_{\rm ref}$ is the mean absolute error (MAE) of the refitted reference family. Each residual has an auxiliary variable $d_{ti}$ subject to $-d_{ti}\le Y_{ti}-\widehat Y_{ti}\le d_{ti}$. Minimizing their sum gives linear programs for fixed supports and mixed-integer linear programs for structure search. The same support constraints, coefficient bounds, and validation MSE apply to both training losses.

The fit records the current objective, the objective at the solver start, the solver return, and the objective after fixed-support refitting. For W, the solver start counts old supports that still occupy active slots even when their coefficients start at zero. The record identifies gains from pool trials, integer search, and refitting. Native bounds and gaps cover the declared single-step or finite-pool models. W follows a greedy sequence of accepted supports.

## Output contraction and validation

Before ranking the candidates, the output step processes the accepted path and the singleton-only protection model. It checks non-singleton supports, including a common upstream support, using the training calibration.

$$
s_C=\frac{a r_C+b x_C}{m},\qquad
a=\frac{0.75}{\mu_R},\quad b=\frac{0.25}{\mu_X},\quad
m=\operatorname{median}\operatorname{diag}(aR_0+bX_0).
$$

Here $R_0,X_0$ are the preliminary training response matrices and $\mu_R,\mu_X$ are the means of their positive pairwise distances. The ratio compares one branch's response with a typical root-to-terminal response. The default threshold is 0.08. The procedure removes the weakest support below the threshold, refits the remaining coefficients on training data, and repeats until the candidate passes. Singleton supports are exempt.

Validation MSE is divided by $M_{\rm ref}$ for selection. Within the normalized tie tolerance $10^{-12}$, the selector uses the internal support count, then the score and original candidate index. `freeze_history(result, validation)` verifies the validation fingerprint, parameter fingerprints, R/X matrices, original selection, and applicable contraction receipts. It retains distinct parameter points from the same topology. Older banks without an output-floor setting retain their original semantics.

An active internal support has more than one terminal and fewer than all terminals, with a weight above the zero tolerance. Exact recovery of all active supports also checks terminal and common supports.

## New-day score and the pairwise target

Let $S_j^h$ be candidate $j$'s sum of squared errors (SSE) on the original validation data. Write $H_j=[R_j,X_j]$. Stack each future time slot's P values followed by its Q values in $p$. Apply $H_j$ separately to each slot and stack the predicted drops. Use $Y$ for the vector of observed future voltage drops. The updated score is the MSE over all $N_{\rm all}$ scalar residuals across validation and the new day.

$$
M_j = \frac{S_j^h + \|Y-H_jp\|^2}{N_{\rm all}}.
$$

For target $t$, let $\epsilon=M_{\rm ref}\,10^{-12}$. A sufficient condition for $M_j - M_t>\epsilon$ is

$$
L_j=\|Y-H_jp\|^2-\|Y-H_tp\|^2 >
\delta_j,\qquad
\delta_j=N_{\rm all}\epsilon-S_j^h+S_t^h.
$$

This threshold translates the final score into a required new-day SSE difference. Beating every candidate from a different active topology is sufficient for recovering the target topology. Equal-response ties follow the original support-count and index ordering. Scoring uses the saved coefficients.

## Conditional error model and probability bound

The planning interface assumes the target response is correct. It uses $p=\widehat p+B u+\xi_p$ and $Y=H_t p+e$, with supplied forecast $\widehat p$, control $u$, and device map $B$. The Gaussian power-error standard deviation of each channel defaults to 5% of its absolute forecast peak. P and Q at one node share a factor, different nodes are independent, and the default time correlation is the autoregressive law with parameter 0.7. The independent voltage-drop residual uses a nodewise standard deviation given by the larger of historical residual RMS and $2\sqrt{2}$ times the supplied voltage measurement standard deviation.

Let $s$ be the largest voltage-drop residual standard deviation. Let $d(u)$ and $C$ be the mean and covariance of $z=(H_t-H_j)p/s$, and let $D$ be the diagonal covariance of $e/s$. With $\lambda=1/4$ and $A=\lambda(I-2\lambda D)$, the Gaussian integral gives

$$
G=I+2A^{1/2}CA^{1/2},\qquad
T_j(u)=d(u)^\top A^{1/2} G^{-1} A^{1/2} d(u)
       +\tfrac12\log\det G .
$$

Markov's inequality gives the conditional pairwise failure bound

$$
\Pr_t[L_j\le\delta_j]\ \le\
\min\{1,\ \exp(\lambda\delta_j/s^2 - T_j(u))\}.
$$

The scale $s$ changes the units of the calculation while keeping the unweighted MSE selection rule. The bound uses the specified error law. Empirical AC performance with realized device actions is a separate measurement.

## Zero-action screen and pricing

`zero_screen` combines the pairwise failure bounds with a union bound, which permits dependence between the pairwise events. With a device allocation, feasible zero delivery at zero cost is a minimum-fee solution when the aggregate failure bound is at most 10% and feasible costs are nonnegative. A screen without devices reports the probability bound.

When zero action falls short and there is one competitor, `plan` uses the retained device optimizer. It minimizes cost subject to the device constraints and $T_j(u)\ge\lambda\delta_j/s^2+\log 10$. It checks delivery and the bound against the returned schedule and records the feasible cost, lower bound, and solver status. This problem is a nonconvex quadratically constrained quadratic program and can include integer device variables. The generic planner supports this single-competitor case. Multi-competitor nonzero procurement is handled by separate research scripts.
