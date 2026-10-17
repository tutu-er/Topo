"""Prospective settings; independent evaluation is not used for tuning."""
CONFIG = dict(
    version='g_precision_v1', horizon=8, dt_hours=.25, base_kw=20.,
    development_seeds=[2031021000, 2031021001],
    evaluation_seeds=list(range(2031022000, 2031022008)),
    fit_multistart=1, fit_max_nfev=400, fit_tol=1e-10,
    iterations=2, segments=4, max_segments=16, solver_seconds=2.,
    total_seconds_per_domain=12., absolute_gap=1e-5, relative_gap=.01,
    backend='gurobi', aggregation='worst', clean_diagnostic=True,
    methods=['OBS', 'CC', 'HL', 'P-SQ', 'G-SQ', 'G-AQ'],
    source_forecast='history-only harmonic regression; fixed within planning',
    lines='one main fit per candidate; fixed within planning; JNT refits in evaluation',
    probability_calibrated=False, evaluation_tuned=False,
    full_route_enumeration=False, truth_in_selector=False,
)
