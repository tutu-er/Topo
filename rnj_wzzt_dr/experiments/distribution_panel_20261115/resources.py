"""Historical resource catalogue for the new multi-network distribution study.

Only registration at the history start, frozen historical candidates and the
192 training P/Q samples are used. No old case-specific node floor is inherited.
The two 5-kvar locations are selected anew from historical candidate differences.
This module neither fits candidates nor solves resource or network optimizations.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ... import io
from ..measured_pq_flex_20261114.domain import catalog


CONTRACT_TIERS = (1., 2., 3., 5., 7.5, 10., 15., 20.)
EQUIPMENT_TIERS = (3., 5., 10., 15., 20., 30.)
RESERVED_FRACTION = .8
PARTICIPATION_FRACTION = .1
TARGET_MASS = .99
REGISTRATION_FIELDS = ('PV_systems_kW', 'batteries_kW', 'EV_chargers_kW',
                       'hot_water_tanks_kW')


def tier(value, catalogue):
    """Smallest predeclared capacity at least value; never silently cap it."""
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError('Capacity must be finite and nonnegative')
    for capacity in catalogue:
        if value <= capacity + 1e-12:
            return float(capacity)
    raise ValueError('Required capacity exceeds predeclared catalogue')


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    # Explicit numeric representation avoids object-pointer/endianness hashes.
    array = np.ascontiguousarray(value, dtype='<f8')
    return hashlib.sha256(json.dumps(list(array.shape)).encode()+array.tobytes()).hexdigest()


def _history_context(context, labels, start_date):
    rows = context['node_mapping']
    if [int(r['label']) for r in rows] != labels:
        raise ValueError('Training labels and resource mappings differ in identity/order')
    if len({int(r['feeder_id']) for r in rows}) != len(rows):
        raise ValueError('Independent source feeders required; duplicated feeder mapping')
    registrations = {}
    for row in context['resource_metadata_at_history_start']:
        feeder = int(row['feeder'])
        if feeder in registrations:
            raise ValueError('Ambiguous registration: duplicate feeder rows')
        registrations[feeder] = row
    mapping, selected = [], []
    for row in rows:
        feeder, scale = int(row['feeder_id']), float(row['scale'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('P and Q require the same positive finite source scale')
        registration = registrations[feeder]
        registration_date = str(registration['date'])
        if date.fromisoformat(registration_date[:10]) > start_date:
            raise ValueError('Registration after history start would leak future availability')
        values = {key: float(registration.get(key, 0.) or 0.) for key in REGISTRATION_FIELDS}
        if not np.isfinite(list(values.values())).all() or min(values.values()) < 0:
            raise ValueError('Registered capacities must be finite and nonnegative')
        mapping.append(dict(label=int(row['label']), feeder_id=feeder, scale=scale))
        selected.append(dict(feeder=feeder, date=registration_date, **values))
    return dict(node_mapping=mapping, resource_metadata_at_history_start=selected)


def _candidate_inputs(score, labels):
    bank = score['native_bank']
    if not bank:
        raise ValueError('Frozen native candidate bank is empty')
    if 'labels' in score and list(map(int, score['labels'])) != labels:
        raise ValueError('Frozen candidate and training label orders differ')
    H = np.asarray([point['H'] for point in bank], float)
    n = len(labels)
    if H.shape != (len(bank), n, 2*n) or not np.isfinite(H).all():
        raise ValueError('Expected finite native response matrices of shape N by 2N')
    families = [str(point['topology_id']) for point in bank]
    representatives = [dict(index=int(row['index']), probability=float(row['probability']))
                       for row in score['models']]
    probability = np.asarray([r['probability'] for r in representatives], float)
    indices = [r['index'] for r in representatives]
    if (not indices or len(set(indices)) != len(indices)
            or any(i < 0 or i >= len(bank) for i in indices)
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or not np.isclose(probability.sum(), 1., rtol=0., atol=1e-8)):
        raise ValueError('Invalid frozen representative probabilities or native indices')
    if len({families[i] for i in indices}) != len(indices):
        raise ValueError('Representative list must contain one point per topology family')
    order = sorted(range(len(indices)), key=lambda i: (-probability[i], i))
    cumulative, cutoff = 0., None
    for i in order:
        cumulative += probability[i]
        cutoff = probability[i]
        if cumulative >= TARGET_MASS:
            break
    targets = [i for i in order if probability[i] >= cutoff-1e-15]
    native_targets = [indices[i] for i in targets]
    pairs = sorted({tuple(sorted((i, j))) for i in native_targets for j in range(len(bank))
                    if families[i] != families[j]})
    return H, dict(representatives=representatives, native_family_ids=families,
                   target_representatives=targets, target_native_indices=native_targets,
                   native_pairs=[list(pair) for pair in pairs])


def choose_focus(energy, eligible, labels):
    """Maximize the weakest candidate-pair Q-column sum over at most two nodes."""
    energy = np.asarray(energy, float)
    eligible = sorted(eligible, key=lambda i: labels[i])
    if not eligible or energy.shape[0] == 0:
        return dict(node_indices=[], node_labels=[], combinations=0,
                    reason='no_foreign_topology' if energy.shape[0] == 0 else 'no_5kw_eligible_site',
                    minimum_pair_sensitivity=None, pair_sensitivities=[])
    best, count = None, 0
    for nodes in itertools.combinations(eligible, min(2, len(eligible))):
        values = energy[:, nodes].sum(axis=1)
        key = (float(values.min()), float(values.sum()), tuple(-labels[i] for i in nodes))
        if best is None or key > best[0]:
            best = key, nodes, values
        count += 1
    key, nodes, values = best
    return dict(node_indices=list(nodes), node_labels=[labels[i] for i in nodes],
                combinations=count, reason='historical_maximin_Q_response',
                minimum_pair_sensitivity=float(values.min()), pair_sensitivities=values.tolist())


def build_resources(entry, score, context, train, base_kw):
    """Return a JSON-ready catalogue receipt; caller must require status='ready'.

    train is a mapping containing P/Q/labels (192 rows, per unit); extra entries
    such as Y/V are ignored. entry supplies id/start_date and optional history
    file references only. Caller saves this receipt next to the sealed inputs.
    A catalogue overflow or invalid/missing input is retained as unresolved,
    never converted into an artificially capped resource.
    """
    began = perf_counter()
    result = dict(status='unresolved', id=str(entry.get('id', 'unknown')),
        rule_version='historical_q95_maximin_v1', p_resources=[], q_resources=[], nodes=[],
        problems=[], source=io.ref(__file__), catalogue_source=io.ref(Path(catalog.__code__.co_filename)),
        information_boundary='Historical registration, 192 training P/Q, historical frozen candidate H and representative weights only',
        rule=dict(P_fraction=PARTICIPATION_FRACTION, P_top_counts=dict(pv=3, ev=2, thermal=2),
            Q_eligibility='positive scaled registered PV plus battery kW', focus_eligibility_kw=5.,
            focus_count_at_most=2, focus_floor_kvar=5., other_floor_kvar=1., target_mass=TARGET_MASS,
            training_Q_quantile=.95, contract_tiers_kvar=list(CONTRACT_TIERS),
            equipment_tiers_kvar=list(EQUIPMENT_TIERS), maximum_reserved_fraction=RESERVED_FRACTION,
            Q_window=[0, 24], Q_activation_yuan_per_kvarh=.05),
        thermal_rating_scope='No network rating inferred. Power base is normalization, not transformer nameplate; branch/transformer thermal checks require separate rated data.',
        equipment_scope='Additional independent bidirectional compensators are a declared scenario, not measured installed SVG or PV inverter headroom. Q95 is a size statistic, not an availability probability.',
        pricing_scope='Physical domain activation fees only; separate billing adds predeclared daily reserved-capacity prices.',
        history_references={key: deepcopy(entry[key]) for key in ('score','resource_context') if key in entry})
    if 'train.npz' in entry.get('files', {}):
        result['history_references']['training'] = deepcopy(entry['files']['train.npz'])
    try:
        base_kw = float(base_kw)
        if not np.isfinite(base_kw) or base_kw <= 0:
            raise ValueError('Positive finite physical P/Q normalization base required')
        labels = list(map(int, np.asarray(train['labels']).tolist()))
        if not labels or len(set(labels)) != len(labels):
            raise ValueError('Training terminal labels must be unique')
        P, Q = np.asarray(train['P'], float), np.asarray(train['Q'], float)
        if P.shape != (192, len(labels)) or Q.shape != P.shape or not np.isfinite(P).all() or not np.isfinite(Q).all():
            raise ValueError('Expected finite matching 192-row training P/Q arrays')
        start = date.fromisoformat(str(entry['start_date'])[:10])
        historical = _history_context(context, labels, start)
        H, candidates = _candidate_inputs(score, labels)
        result.update(base_kw_kvar=base_kw, training_points=192, labels=labels, history_start_date=start.isoformat(),
            input_hashes=dict(P_pu_float64=_array_hash(P), Q_pu_float64=_array_hash(Q), labels=_json_hash(labels),
                base_kw=_json_hash(base_kw), history_start_date=_json_hash(start.isoformat()),
                historical_registration_and_mapping=_json_hash(historical),
                frozen_H_float64=_array_hash(H), candidate_families_weights_targets=_json_hash(candidates)))
        registrations = {row['feeder']: row for row in historical['resource_metadata_at_history_start']}
        q95 = np.quantile(np.abs(Q*base_kw), .95, axis=0)
        registered = np.asarray([(registrations[row['feeder_id']]['PV_systems_kW']
                                 + registrations[row['feeder_id']]['batteries_kW'])*row['scale']
                                for row in historical['node_mapping']], float)
        pairs = candidates['native_pairs']
        # The old 96/(2*sigma2) normalization is common to all pairs and sites;
        # omitting it leaves the exact maximin/tie choice unchanged.
        energy = np.asarray([np.square((H[i]-H[j])[:, len(labels):]).sum(axis=0)/base_kw**2
                             for i, j in pairs], float).reshape(len(pairs), len(labels))
        focus = choose_focus(energy, np.flatnonzero(registered >= 5.).tolist(), labels)
        result['placement'] = dict(candidates, **focus,
            sensitivity_units='Sum of squared per-unit output response per kvar squared; common 96/(2*sigma2) multiplier omitted without changing placement',
            input_scope='Frozen historical candidates and historical registration; no outcome or new-day power used')
        result['p_resources'] = catalog(historical, PARTICIPATION_FRACTION)
        for k, mapping in enumerate(historical['node_mapping']):
            label = labels[k]; participating = bool(registered[k] > 0.)
            is_focus = label in focus['node_labels']
            row = dict(node_label=label, feeder_id=mapping['feeder_id'], source_to_network_scale=mapping['scale'],
                registration_date=registrations[mapping['feeder_id']]['date'], scaled_PV_battery_kw=float(registered[k]),
                participating=participating, focus=is_focus, training_abs_Q95_kvar=float(q95[k]),
                base_contract_kvar=(5. if is_focus else 1.) if participating else 0.,
                contract_kvar=0., equipment_rating_kvar=0.)
            if participating:
                try:
                    sized = tier(q95[k], CONTRACT_TIERS)
                    capacity = max(row['base_contract_kvar'], sized)
                    equipment = tier(capacity/RESERVED_FRACTION, EQUIPMENT_TIERS)
                    row.update(Q95_tier_kvar=sized, contract_kvar=capacity, equipment_rating_kvar=equipment,
                               capacity_driver='focus_floor' if is_focus and 5. > sized else 'training_Q95_tier')
                    result['q_resources'].append(dict(id=f'svg_{label}', node_label=label, kind='reactive',
                        capacity_kvar=capacity, available_clock_hours=[0, 24],
                        fee_positive_yuan_per_kvarh=.05, fee_negative_yuan_per_kvarh=.05,
                        reservation_fee_yuan_per_kvar_day=0., feeder_id=mapping['feeder_id'],
                        capacity_origin='max(new historical maximin focus floor or1, training absoluteQ95 rounded up)',
                        assumed_equipment_rating_kvar=equipment, equipment_reservation_fraction_limit=RESERVED_FRACTION,
                        equipment_type='additional independent bidirectional converter compensator',
                        remote_signed_Q_control='Assumed service contract, not certified by registration',
                        all_day_service='Independent energized compensator; no PV night-headroom inference'))
                except ValueError as exc:
                    row['error'] = str(exc)
                    result['problems'].append(dict(stage='Q_capacity', node_label=label, error=str(exc)))
            result['nodes'].append(row)
        result['totals'] = dict(P_devices=len(result['p_resources']), Q_devices=len(result['q_resources']),
            P_contract_kw=float(sum(r['contract_kw'] for r in result['p_resources'])),
            Q_contract_kvar=float(sum(r['capacity_kvar'] for r in result['q_resources'])),
            assumed_Q_equipment_kvar=float(sum(r['assumed_equipment_rating_kvar'] for r in result['q_resources'])))
        if not result['problems']:
            result['status'] = 'ready'
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        result['problems'].append(dict(stage='historical_input_or_legacy_P_domain', error=str(exc)))
    result['configuration_sha256'] = _json_hash(dict(p_resources=result['p_resources'], q_resources=result['q_resources']))
    result['construction_seconds'] = perf_counter()-began
    result['created_utc'] = datetime.now(timezone.utc).isoformat()
    return result
