"""Complete, independent cases for the line-uncertainty EV study.

All truth and future arrays belong to the evaluation driver.  ``planning_inputs``
returns only observed T/V records, public meters and the unfiltered four-model
bank.  No EV is included in these background trajectories.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np

from . import ev_mobile_cases_v2 as assets
from . import ev_mobile_v4_cases as ordinary
from . import ev_original_study as base
from . import likelihood as lk
from .ev_fixed_train_fit import subset_model, subset_observations
from .simulation import model


BANKS = ('n3', 'n4')
TRAINING_COUNTS = (32, 96)
EXCITATIONS = ('ordinary', 'correlated')
INNOVATIONS = {'nominal': 1., 'triple': 3.}
REGIMES = ('regular', 'shift')
EVIDENCE_COUNT = 20
HORIZON = 8
SEED_BASES = dict(development=2026110100, calibration=2026111000, evaluation=2026112000)
CASE_CONFIG = dict(version='bayes_response_complete_cases_v1', banks=list(BANKS),
    families_per_bank=4, training_counts=list(TRAINING_COUNTS), evidence_count=EVIDENCE_COUNT,
    horizon=HORIZON, dt_hours=.25, excitations=list(EXCITATIONS),
    innovation_multipliers=dict(INNOVATIONS), regimes=list(REGIMES),
    calibration_replicates_per_stratum=1, evaluation_replicates_per_stratum=2,
    seed_bases=dict(SEED_BASES), base_kw=base.BASE_KW, base_kv=base.BASE_KV,
    source_interface=1, total_precision_per_channel=30.,
    statistical_unit='one independently seeded complete physical history',
    paired_regimes_share_history_and_meter_noise=True, EV_included=False)
GENERATOR_CONFIG = deepcopy(ordinary.GENERATOR_CONFIG)
GENERATOR_CONFIG.pop('unannounced_shift_start', None)
GENERATOR_CONFIG.update(version='ordinary_or_correlated_harmonic_AR_complete_v1',
    correlated_AR_common_variance_fraction=.95, correlated_phase_jitter_radians=.08,
    shift_start='H+20', innovation_multiplier_scope='all training, evidence and future process innovations')
STRATA = tuple(dict(stratum_id=i, bank=bank, family_index=family, H=H,
    excitation=excitation, innovation=innovation, innovation_multiplier=INNOVATIONS[innovation])
    for i, (bank, family, H, excitation, innovation) in enumerate(product(
        BANKS, range(4), TRAINING_COUNTS, EXCITATIONS, INNOVATIONS)))


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def specifications(phase):
    """16 development, 64 calibration, 128 new independent evaluation histories."""
    if phase not in SEED_BASES:
        raise ValueError('development, calibration or evaluation phase required')
    if phase == 'development':
        chosen = []
        # First eight include both banks, both H and star/one-branch truths.
        for family, hi, bi in product(range(4), range(2), range(2)):
            key = dict(bank=BANKS[bi], family_index=family, H=TRAINING_COUNTS[hi],
                excitation=EXCITATIONS[(family+hi+bi) % 2],
                innovation=tuple(INNOVATIONS)[(family+bi) % 2])
            chosen.append(next(s for s in STRATA if all(s[k] == v for k, v in key.items())))
        repetitions = 1
    else:
        chosen = STRATA
        repetitions = 1 if phase == 'calibration' else 2
    result = []
    for stratum in chosen:
        for replicate in range(repetitions):
            index = len(result)
            result.append(dict(deepcopy(stratum), phase=phase, index=index,
                seed=SEED_BASES[phase]+index, replicate=replicate,
                legacy_audit=bool(phase == 'evaluation' and replicate == 0
                    and stratum['excitation'] == 'ordinary' and stratum['innovation'] == 'nominal')))
    return result


def _ar(rng, count, width, sd, phi):
    values = np.empty((count, width)); values[0] = sd*rng.standard_normal(width)
    innovation = sd*np.sqrt(1-phi**2)*rng.standard_normal((count-1, width))
    for t in range(1, count):
        values[t] = phi*values[t-1]+innovation[t-1]
    return values


def _load_AR(rng, count, n, sd, phi, correlated):
    if not correlated:
        return _ar(rng, count, n, sd, phi)
    common = _ar(rng, count, 1, sd, phi)
    independent = _ar(rng, count, n, sd, phi)
    rho = GENERATOR_CONFIG['correlated_AR_common_variance_fraction']
    return np.sqrt(rho)*common+np.sqrt(1-rho)*independent


def generate_trajectory(seed, n, H, excitation='ordinary', innovation='nominal', regime='regular'):
    seed, n, H = _integer(seed, 'seed'), _integer(n, 'n', 1), _integer(H, 'H', 1)
    if n not in (3, 4) or H not in TRAINING_COUNTS or excitation not in EXCITATIONS or innovation not in INNOVATIONS or regime not in REGIMES:
        raise ValueError('declared n3/n4, H32/H96, excitation, innovation and regime required')
    count, start = H+EVIDENCE_COUNT+HORIZON, H+EVIDENCE_COUNT
    names = ('phases', 'correlation', 'p', 'ratio', 'source_p', 'source_q', 'shift')
    sequences = np.random.SeedSequence(seed).spawn(len(names))
    seeds = {k: int(v.generate_state(1)[0]) for k, v in zip(names, sequences)}
    rng = {k: np.random.default_rng(v) for k, v in seeds.items()}
    cfg, t, scale = GENERATOR_CONFIG, np.arange(count, dtype=float), INNOVATIONS[innovation]
    p_phase = rng['phases'].uniform(0, 2*np.pi, (3, n))
    ratio_phase = rng['phases'].uniform(0, 2*np.pi, (2, n))
    source_phase = rng['phases'].uniform(0, 2*np.pi, 4)
    means = np.asarray(cfg['p_mean_kw'][:n])+rng['phases'].uniform(-.03, .03, n)
    correlated = excitation == 'correlated'
    if correlated:
        jitter = float(cfg['correlated_phase_jitter_radians'])
        p_phase = rng['correlation'].uniform(0, 2*np.pi, (3, 1))+rng['correlation'].uniform(-jitter, jitter, (3, n))
        ratio_phase = rng['correlation'].uniform(0, 2*np.pi, (2, 1))+rng['correlation'].uniform(-jitter, jitter, (2, n))
    modulation = np.zeros((count, n))
    for amplitude, period, phase in zip(cfg['p_fractional_amplitudes'], cfg['p_periods_slots'], p_phase):
        modulation += amplitude*np.sin(2*np.pi*t[:, None]/period+phase)
    p_raw = means*(1+modulation)+_load_AR(rng['p'], count, n, scale*cfg['p_innovation_stationary_sd_kw'], .65, correlated)
    ratio_raw = np.full((count, n), cfg['q_over_p_mean'])
    for amplitude, period, phase in zip(cfg['q_over_p_amplitudes'], cfg['q_over_p_periods_slots'], ratio_phase):
        ratio_raw += amplitude*np.sin(2*np.pi*t[:, None]/period+phase)
    ratio_raw += _load_AR(rng['ratio'], count, n, scale*cfg['q_over_p_stationary_sd'], .65, correlated)
    sp_raw, sq_raw = np.full(count, cfg['source_p_mean_kw']), np.full(count, cfg['source_q_mean_kvar'])
    for amplitude, period, phase in zip(cfg['source_p_amplitudes_kw'], cfg['source_p_periods_slots'], source_phase[:2]):
        sp_raw += amplitude*np.sin(2*np.pi*t/period+phase)
    for amplitude, period, phase in zip(cfg['source_q_amplitudes_kvar'], cfg['source_q_periods_slots'], source_phase[2:]):
        sq_raw += amplitude*np.sin(2*np.pi*t/period+phase)
    sp_raw += _ar(rng['source_p'], count, 1, scale*cfg['source_p_stationary_sd_kw'], .7)[:, 0]
    sq_raw += _ar(rng['source_q'], count, 1, scale*cfg['source_q_stationary_sd_kvar'], .6)[:, 0]
    shift_phase = rng['shift'].uniform(0, 2*np.pi, (2, n))
    source_shift = rng['shift'].uniform(0, 2*np.pi, 2)
    if regime == 'shift':
        h = np.arange(HORIZON, dtype=float); rows = slice(start, None)
        p_raw[rows] += np.asarray(cfg['shift_p_level_kw'][:n])+cfg['shift_p_amplitude_kw']*np.sin(2*np.pi*h[:, None]/cfg['shift_p_period_slots']+shift_phase[0])
        ratio_raw[rows] += np.asarray(cfg['shift_q_over_p_level'][:n])+cfg['shift_q_over_p_amplitude']*np.sin(2*np.pi*h[:, None]/cfg['shift_q_over_p_period_slots']+shift_phase[1])
        sp_raw[rows] += cfg['shift_source_p_level_kw']+cfg['shift_source_p_amplitude_kw']*np.sin(2*np.pi*h/cfg['shift_p_period_slots']+source_shift[0])
        sq_raw[rows] += cfg['shift_source_q_level_kvar']+cfg['shift_source_q_amplitude_kvar']*np.sin(2*np.pi*h/32+source_shift[1])
    p, ratio = np.maximum(p_raw, .05), np.clip(ratio_raw, *cfg['q_over_p_bounds'])
    sp, sq = np.maximum(sp_raw, 0.), np.maximum(sq_raw, 0.)
    return dict(p_kw=p, q_kvar=p*ratio, q_over_p=ratio, source_p_kw=sp,
        source_q_kvar=sq, root_v=np.ones(count), metadata=dict(config=deepcopy(cfg), seed=seed,
            child_seeds=seeds, n=n, H=H, evidence_count=EVIDENCE_COUNT, horizon=HORIZON,
            count=count, future_start=start, time_origin=0, excitation=excitation,
            innovation=innovation, innovation_multiplier=scale, regime=regime,
            stationary_innovation_scales=dict(p_kw=scale*cfg['p_innovation_stationary_sd_kw'],
                q_over_p=scale*cfg['q_over_p_stationary_sd'], source_p_kw=scale*cfg['source_p_stationary_sd_kw'],
                source_q_kvar=scale*cfg['source_q_stationary_sd_kvar']),
            clips=dict(p=int(np.count_nonzero(p != p_raw)), q_over_p=int(np.count_nonzero(ratio != ratio_raw)),
                source_p=int(np.count_nonzero(sp != sp_raw)), source_q=int(np.count_nonzero(sq != sq_raw))),
            source_interface=1, EV_included=False, generation_only=True, field_data=False))


def make_case(spec, regime='regular'):
    """Generate exactly H+20+8 rows; only evaluation code receives this object."""
    spec = deepcopy(spec); bank, H = spec['bank'], _integer(spec['H'], 'H', 1)
    seed, family = _integer(spec['seed'], 'seed'), _integer(spec['family_index'], 'family_index')
    if bank not in BANKS or family not in range(4) or H not in TRAINING_COUNTS:
        raise ValueError('declared four-candidate n3/n4 and H32/H96 required')
    excitation, innovation = spec['excitation'], spec['innovation']
    if innovation not in INNOVATIONS or ('innovation_multiplier' in spec and spec['innovation_multiplier'] != INNOVATIONS[innovation]):
        raise ValueError('innovation label/multiplier disagree')
    case_id = 'remote8' if bank == 'n3' else 'twinblock12'
    cfg = assets.case_config(case_id, training_points=96)
    cfg.update(version=CASE_CONFIG['version'], training_points=H, validation_points=EVIDENCE_COUNT,
        evidence_points=EVIDENCE_COUNT, horizon=HORIZON, future_start=H+EVIDENCE_COUNT,
        observed_before_EV=H+EVIDENCE_COUNT, n_terminals=3 if bank == 'n3' else 4,
        forecast_gap=0, public_voltage_repeat_counts=[30./HORIZON]*HORIZON,
        future_metering_integration_seconds_per_slot=3000./HORIZON,
        whole_training_sequence=H, complete_trajectory_count=H+EVIDENCE_COUNT+HORIZON)
    candidate, r, x, seeds, asset_rows = assets._assets(seed, case_id, family)
    bg = generate_trajectory(seeds['background'], candidate.n, H, excitation, innovation, regime)
    p, q, sp, sq = (bg[k]/base.BASE_KW for k in ('p_kw', 'q_kvar', 'source_p_kw', 'source_q_kvar'))
    root, meter, count = bg['root_v'], model(candidate.n), len(p)
    sequences = np.random.SeedSequence(seeds['noise']).spawn(len(base.CHANNELS))
    white = {key: np.random.default_rng(s).standard_normal((count, candidate.n) if key in ('p', 'q', 'v') else count)
             for key, s in zip(base.CHANNELS, sequences)}
    physical = lk.ac_predict(candidate, r, x, p, q, sp, sq, root)
    full = base.observation(physical, meter, white)
    train_rows, evidence_rows = slice(0, H), slice(H, H+EVIDENCE_COUNT)
    train, evidence = subset_observations(full, train_rows), subset_observations(full, evidence_rows)
    tm, em = subset_model(meter, full, train_rows), subset_model(meter, full, evidence_rows)
    precision = 30./HORIZON
    fm = replace(meter, v_sigma=np.broadcast_to(meter.v_sigma/np.sqrt(precision), (HORIZON, candidate.n)),
        **{key+'_sigma': getattr(meter, key+'_sigma')/np.sqrt(precision) for key in ('p', 'q', 'master_p', 'master_q')})
    return dict(spec=spec, case_id=case_id, seed=seed, family_index=family, regime=regime,
        config=cfg, train=train, evidence=evidence, validation=evidence, tm=tm, em=em, vm=em, fm=fm,
        p=p, q=q, source_p=sp, source_q=sq, root_v=root, white=white,
        future_start=H+EVIDENCE_COUNT, base_kw=base.BASE_KW, assets=asset_rows,
        truth=dict(candidate=candidate, r=r, x=x), bank=assets.candidate_bank(case_id),
        provenance=dict(data_kind='new_complete_synthetic_LV_line_uncertainty_cases',
            case_config=deepcopy(CASE_CONFIG), asset=deepcopy(base.ASSET), assets=asset_rows,
            source_interface=1, seeds=seeds, background=bg['metadata'],
            H=H, V=EVIDENCE_COUNT, T=HORIZON, count=count, future_start=H+EVIDENCE_COUNT,
            base_kw=base.BASE_KW, base_kv=base.BASE_KV, impedance_base_ohm=base.ZBASE,
            candidate_scope='four predeclared structures, no truth filtering',
            total_effective_precision=30., precision_uniform_in_time=True,
            metering=dict(per_slot_weight=precision, total_weight_per_channel=30.,
                integration_seconds_per_slot=3000./HORIZON, integration_seconds_total=3000.),
            field_PQV_available=False, true_labels_used_only_for_generation_and_evaluation=True,
            paired_regime_scope='identical first H+20 physical/meter rows, assets and standardized noise',
            EV_included=False), truth_used_only_for_generation_and_evaluation=True)


def planning_inputs(case):
    """Explicit observed-only build_context input boundary; fits added by caller."""
    return dict(training=case['train'], evidence=case['evidence'], train_meter=case['tm'],
        evidence_meter=case['em'], future_meter=case['fm'], candidates=case['bank'],
        horizon=HORIZON, base_kw=case['base_kw'])
