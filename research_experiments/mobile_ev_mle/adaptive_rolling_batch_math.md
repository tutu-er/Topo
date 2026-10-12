# Two-round shadow-policy batch contract

One statistical unit is one independently generated latent AC path and its
complete numerical-policy trace. Let H0 contain the twelve initial meters.
The first decision is u1=f(H0, public EV state); only its first two slots are
executed. Their total P/Q/V and master meters form H1=H0 concatenated with Y1.
The second decision is u2=f(H1, carried EV state), and its six slots form
H2=H1 concatenated with Y2. Both decisions use the full star/three-cherry audit
bank. Truth, unexecuted future states and future noise are evaluator-only.
All common-source paths act through the same physical junction node 1.

EV terminal power enters the AC load once. With declared tracking gain one,
the actually executed kW prefix is recorded. H1/H2 likelihoods use the total
meters unchanged. Forecasting subtracts known actual EV input from historical
total P only; uncertain commands cannot be treated as exact realized input.
The second operational target is 6 kWh minus net grid energy already executed,
not a new 6 kWh task. Battery carry follows

E(t+1)=E(t)+dt*(eta_charge*P_plus(t)-P_minus(t)/eta_discharge)-D(t),

where driving energy D is deducted once in the departure slot. Position,
access, remaining throughput and terminal conditions also carry. The pilot
requires one-slot allowed travel. Integer execution boundaries therefore end
at a node; a longer in-transit arc is explicitly recorded and rejected by
the minimal replanner, never relabelled an arrival.

For achieved local Gaussian losses L[r,g] on H1 and H2, define the whole-trace
relative and absolute truth scores

S_rel=max(r=1,2) (L[r,g_true]-min_g L[r,g]),
S_abs=max(r=1,2) L[r,g_true].

Every required candidate/round fit must finish with a valid finite objective.
Otherwise both scores are +infinity and complete=False. A later recovery does
not erase an earlier failure. These are statistics of the frozen numerical
policy, not exact global-MLE statistics or topology posterior probabilities.
The audit class covers this declared four-topology generator; this does not
prove coverage of arbitrary distribution networks.

Independent calibration histories sample this same two-round policy and
population. A one-round threshold is incompatible. With total alpha=.1 and
equal budgets, each upper order statistic has rank
k=ceil((n+1)*(1-.05)). A rank beyond n gives infinity; n>=19 permits a finite
rank but failures can still make the threshold infinite. The rank guarantee
is marginal over the declared independent-history population and frozen
policy, not uniform over all continuous impedances and sources. The pilot
schedule is declared before calibration: if mean pilot time is at most 45 s,
64 calibration and 96 evaluation histories; otherwise 32 and 64. Finite
sample power and failed histories remain reported. Pilot data do not enter
the formal calibration/evaluation denominator.

The batch header binds the rolling policy, initial device constraints,
carry requirements, actual loaded source hashes and numerical runtime. A
resume must match the full header. Phase, count and seed range remain explicit.
Deadline checks occur before a new independent history; an in-flight history
finishes. A partially attempted requested count is a budget checkpoint, not
a completed study. A changed source during a history is preserved as pending
evidence and excluded from completed histories.

The standard collector rows use method rolling_refit_milp and rounds 1/2.
Original rounds and selections are retained. NPZ states have keys
round_{r}__{candidate}__fit_{p,q,source_p,source_q}; their matching measured
joint records use round_{r}_joint_{p,q,v,master_p,master_q,root_v}. Replay
needs those saved states and achieved R/X, never generator truth. AC counters
count every research ac_predict entry, including exceptions. Separate
selection and full fit-bank wrappers capture failed candidates whose solver
objects were never returned. The remaining calls are meter-generation
evaluations. These counts are not flow iterations or legacy WZZT work.

Frozen proxy bounds, numerical QP gaps and actual nonlinear rollback are
different evidence. AC rollback retains every pair and evaluates its affine
witness in the original equations; neither its score nor its minimum over
pairs computes the nonlinear-domain infimum. Taylor mismatch remains model
approximation evidence and cannot be promoted into a separation certificate.
