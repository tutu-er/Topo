"""Declared physical LV cases for testing an EV's spatial value.

``remote8`` preserves the old three-terminal, four-candidate family.  Its
8/32/96-point training subsets have the same endpoint, assets and future.
``twinblock12`` has four terminals and four predeclared candidates: neither,
either, or both of the disjoint (0,1) and (2,3) internal branches.  This finite
bank is not the set of all four-terminal radial topologies.

All sites are accessible; one move uses one driving slot, one setup slot,
0.2 kWh, and 30 worker minutes.  The larger window does not add metering
information for free: every future channel has total effective precision
weight 30, uniformly distributed over the window.  These weights describe
Gaussian integration precision, not integer numbers of independent samples.

Truth is used only by ``make_case`` and ``observe_future``.  Operations depend
only on a declared case ID and public device data; forecasts receive training
observations and fitted training models only.  Frozen systematic-v1 files are
imported but never changed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import numpy as np

from . import ev_fixed_train_design as fixed
from . import ev_original_study as base
from . import ev_systematic_cases as old
from . import likelihood as lk
from .ev_dynamic_loads import generate_background, fit_forecast
from .ev_fixed_train_fit import subset_observations, subset_model
from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from .simulation import model


CASES = ('remote8', 'twinblock12')
FAMILIES = {
    'remote8': ((), ((0, 1),), ((0, 2),), ((1, 2),)),
    'twinblock12': ((), ((0, 1),), ((2, 3),), ((0, 1), (2, 3))),
}
DIMENSIONS = {'remote8': (8, 3), 'twinblock12': (12, 4)}
TRAINING_ENDPOINT = 96
FUTURE_START = 100


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + ' outside declared range')
    return value


def _dimensions(case_id):
    if case_id not in CASES:
        raise ValueError('unknown declared mobile-v2 case')
    return DIMENSIONS[case_id]


def candidate_bank(case_id):
    """An unfiltered public bank, independent of the generating true label."""
    _, n = _dimensions(case_id)
    return [lk.RadialCandidate.from_supports(
        'star' if not family else 'clade_' + '_'.join(''.join(map(str, s)) for s in family),
        n, family) for family in FAMILIES[case_id]]


def case_config(case_id, *, training_points=None):
    T, n = _dimensions(case_id)
    H = 96 if training_points is None else _integer(training_points, 'training_points', 1)
    if (case_id == 'remote8' and H not in (8, 32, 96)) or (case_id == 'twinblock12' and H != 96):
        raise ValueError('remote8 supports H=8/32/96; twinblock12 requires H=96')
    cfg = old.case_config('ordinary96')
    cfg.update(case_id=case_id, version='mobile_spatial_cases_v2', n_terminals=n,
        training_points=H, horizon=T, future_start=FUTURE_START,
        public_voltage_repeat_counts=[30. / T] * T,
        future_sampling_budget_per_channel=30.,
        sampling_weight_interpretation='uniform effective Gaussian integration precision; not integer sample counts',
        future_metering_integration_seconds_per_unit=100.,
        future_metering_integration_seconds_per_slot=3000. / T,
        future_metering_integration_seconds_total=3000.,
        candidate_bank_scope='four predeclared local structures, no truth-based filtering',
        whole_training_sequence=TRAINING_ENDPOINT,
        source_interface=1, PQ_inputs_fixed=True, root_fixed=True,
        operations=dict(initial_site=0, all_sites_accessible=True, max_moves=1,
            travel_slots=1, setup_slots=1, drive_kwh=.2,
            driver_budget_minutes=30., task_battery_gain_kwh=2.85,
            throughput_kwh=7.5, extra_information_from_more_slots=False))
    # No new narrower impedance box or X/R cone is introduced.
    return cfg


def _assets(seed, case_id, family_index):
    """Same published cable constants and synthetic lengths as systematic-v1."""
    _, n = _dimensions(case_id)
    streams = np.random.SeedSequence(seed).spawn(5)
    seeds = {k: int(s.generate_state(1)[0]) for k, s in
             zip(('asset', 'background', 'source', 'noise', 'policy'), streams)}
    family = FAMILIES[case_id][family_index]
    candidate = lk.RadialCandidate.from_supports('evaluation_truth', n, family)
    rng = np.random.default_rng(seeds['asset'])
    terminal_lengths = (.080, .100, .120, .100)
    rr, xx, records = [], [], []
    for support in candidate.supports:
        if len(support) == n:
            kind, length = 2, .040
        elif len(support) == 1:
            kind, length = 0, terminal_lengths[support[0]]
        else:
            kind, length = 2, .080
        length *= rng.uniform(.95, 1.05)
        rr.append(base.ASSET['R_ohm_km'][kind] * length / base.ZBASE)
        xx.append(base.ASSET['X_ohm_km'][kind] * length / base.ZBASE)
        records.append(dict(support=tuple(support), cable_mm2=(50, 120, 150)[kind],
            length_km=float(length)))
    return candidate, np.asarray(rr), np.asarray(xx), seeds, records


def make_case(case_id, seed, family_index, *, training_points=None):
    """Generate truth/meters; selectors must receive only the allowed fields."""
    seed = _integer(seed, 'seed', 0)
    fi = _integer(family_index, 'family_index', 0, 3)
    cfg = case_config(case_id, training_points=training_points)
    H, T, n = cfg['training_points'], cfg['horizon'], cfg['n_terminals']
    candidate, r, x, seeds, assets = _assets(seed, case_id, fi)
    N = FUTURE_START + T
    background = generate_background(seeds['background'], N, n)
    p, q = background['p_kw'] / base.BASE_KW, background['q_kvar'] / base.BASE_KW
    sp = background['source_p_kw'] / base.BASE_KW
    sq = background['source_q_kvar'] / base.BASE_KW
    root = background['root_v']
    meter = model(n)
    sequences = np.random.SeedSequence(seeds['noise']).spawn(len(base.CHANNELS))
    white = {k: np.random.default_rng(s).standard_normal((N, n) if k in ('p', 'q', 'v') else N)
             for k, s in zip(base.CHANNELS, sequences)}
    physical = lk.ac_predict(candidate, r, x, p, q, sp, sq, root)
    full = base.observation(physical, meter, white)
    training_rows = slice(TRAINING_ENDPOINT - H, TRAINING_ENDPOINT)
    validation_rows = slice(TRAINING_ENDPOINT, FUTURE_START)
    train = subset_observations(full, training_rows)
    validation = subset_observations(full, validation_rows)
    tm = subset_model(meter, full, training_rows)
    vm = subset_model(meter, full, validation_rows)
    effective_weight = 30. / T
    fm = replace(meter,
        v_sigma=np.broadcast_to(meter.v_sigma / np.sqrt(effective_weight), (T, n)),
        **{k + '_sigma': getattr(meter, k + '_sigma') / np.sqrt(effective_weight)
           for k in ('p', 'q', 'master_p', 'master_q')})
    provenance = dict(data_kind='declared_synthetic_LV_spatial_value_case',
        asset=deepcopy(base.ASSET), assets=assets, asset_seeds=seeds,
        background=background['metadata'], source_interface=1,
        source_interface_meaning='one common physical interface upstream of all terminal subtrees',
        candidate_scope=cfg['candidate_bank_scope'],
        metering=dict(total_effective_precision_weight_per_channel=30.,
            per_slot_effective_weight=effective_weight,
            integration_seconds_per_unit=100.,
            integration_seconds_per_slot=3000. / T,
            integration_seconds_total=3000.,
            weights_uniform_in_time=True, weights_are_integer_sample_counts=False),
        true_labels_used_only_for_generation_and_evaluation=True,
        field_PQV_available=False)
    return dict(case_id=case_id, seed=seed, family_index=fi, config=cfg,
        train=train, validation=validation, tm=tm, vm=vm, fm=fm,
        p=p, q=q, source_p=sp, source_q=sq, root_v=root, white=white,
        future_start=FUTURE_START, base_kw=base.BASE_KW,
        truth=dict(candidate=candidate, r=r, x=x), provenance=provenance,
        bank=candidate_bank(case_id), truth_used_only_for_generation_and_evaluation=True)


def forecast_for(tr, tm, fits, cfg):
    """Training-only point forecast; no simulator, validation or truth inputs."""
    return fit_forecast(tr, tm, fits, horizon=cfg['horizon'], gap=4,
        base_kw=cfg['base_kw'], kind=cfg['forecast'])


def service_domain(case_id, T=None, n=None, *, mobile):
    """Public operations domain; no topology or measurement arguments."""
    declared_T, declared_n = _dimensions(case_id)
    if T is not None and T != declared_T:
        raise ValueError('horizon does not match declared mobile-v2 case')
    if n is not None and n != declared_n:
        raise ValueError('terminal count does not match declared mobile-v2 case')
    c = fixed.CONFIG
    return build_mobile_service_domain(MobileServiceConfig(
        n_slots=declared_T, n_sites=declared_n, dt_hours=c['dt_hours'],
        p_charge_max_kw=c['command_power_kw'], p_discharge_max_kw=c['command_power_kw'],
        allow_discharge=True, battery_capacity_kwh=c['capacity_kwh'],
        initial_soc=c['initial_energy_kwh'] / c['capacity_kwh'],
        terminal_energy_kwh=c['terminal_energy_kwh'],
        eta_charge=c['eta_charge'], eta_discharge=c['eta_discharge'],
        initial_site=0, initial_connected=True, mobile=mobile, access=True,
        travel_slots=1, setup_slots=1, travel_energy_kwh=.2,
        max_moves=1, max_throughput_kwh=c['throughput_kwh'],
        travel_cost=15., setup_cost=15., driver_budget=30.,
        cost_units='worker_minutes', min_arrival_throughput_kwh=c['min_arrival_throughput_kwh']))


def observe_future(case, power, *, clean=False):
    """Generate future meters; EV is inserted once into terminal total P."""
    T, n = case['config']['horizon'], case['train'].n
    power = np.asarray(power, float)
    if power.shape != (T, n) or not np.isfinite(power).all():
        raise ValueError('power must be finite with shape (horizon, terminal count)')
    start = case['future_start']
    rows = slice(start, start + T)
    truth = case['truth']
    predicted = lk.ac_predict(truth['candidate'], truth['r'], truth['x'],
        case['p'][rows] + power / case['base_kw'], case['q'][rows],
        case['source_p'][rows], case['source_q'][rows], case['root_v'][rows])
    white = {k: np.zeros_like(values[rows]) if clean else values[rows]
             for k, values in case['white'].items()}
    return base.observation(predicted, case['fm'], white)
