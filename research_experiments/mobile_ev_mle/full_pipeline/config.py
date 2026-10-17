"""Prospective settings; values are sealed before calibration/evaluation."""
CONFIG = dict(
    version='full_pipeline_v1', base_commit='6d57f155',
    H=96, T=12, dt_hours=.25, base_kw=20., base_kv=.4,
    development_clusters=6, calibration_clusters=12, evaluation_clusters=16,
    terminal_counts=[6,8], source_modes=['none','DER'],
    methods=['OBS','CC','HL','RAND','P-SQ/F±','G-SQ/F±','G-AQ/F±','P-SQ','G-SQ','G-AQ'],
    pure_charge_ablation_clusters=[0,1,2,3],
    seed_bases=dict(development=204001000,calibration=204010000,evaluation=204020000),
    passive=dict(max_candidates=8,bootstrap_replicates=3,block_length=4,
        wzzt_time_limit=.5,wzzt_max_atoms=None,max_repairs=8),
    post_max_new_candidates=4, post_candidate_update=True,
    fit_multistart=1,fit_max_nfev=250,fit_tol=1e-8,
    iterations=2,segments=4,max_segments=16,solver_seconds=2.,
    total_seconds_per_domain=12.,relative_gap=.01,absolute_gap=1e-5,backend='gurobi',
    alpha=.1,statistical_bootstrap=2000,statistical_seed=20409999,
    total_precision_per_channel=30.,clean_development_only=True,
    source_interface=1, bias='independently calibrated known',root='known',
    power_tracking='exact',input_PQ_noise_propagated=False,
    max_workers=12, per_worker_threads=1,
    topology_scope='whole reduced rooted tree on all measured terminals; serial degree-two nodes contracted',
    raw_csv_sha256='eca7dee5ac6e53373cd0209bc05ba3383cd860bfa5d5e995f74fa78db8857c58',
)
CALENDAR = [
    dict(name='training',start_utc='2014-12-11',end_utc='2016-03-19'),
    dict(name='development',start_utc='2016-03-19',end_utc='2016-07-01'),
    dict(name='calibration',start_utc='2016-07-01',end_utc='2017-01-01'),
    dict(name='evaluation',start_utc='2017-01-01',end_utc='2017-02-02'),
]
