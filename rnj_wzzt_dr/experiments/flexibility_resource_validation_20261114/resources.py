"""Size additional Q service contracts from training-only load statistics.

The old 23 kvar contract is retained as a reference and a lower capacity floor.
New equipment is a declared engineering scenario, not an inferred inventory.
No candidate response, true line parameter, forecast or execution is read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from ... import io


REFERENCE = 'registered_plus1'
UPDATED = 'registered_q95_36'
CONTRACT_TIERS = (1., 2., 3., 5., 7.5, 10., 15., 20.)
EQUIPMENT_TIERS = (3., 5., 10., 15., 20., 30.)
RESERVED_FRACTION = .8
BASE_KVA = 200.


def tier(value, tiers):
    if not np.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative size required')
    for capacity in tiers:
        if value <= capacity + 1e-12:
            return float(capacity)
    raise ValueError('Required capacity exceeds declared equipment catalogue')


def checked(ref):
    if io.ref(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('Changed source: '+ref['path'])
    return Path(ref['path'])


def prepare(previous_run, out):
    previous_run, out = Path(previous_run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Keep previous resource specifications')
    previous_complete = io.read(previous_run/'complete.json')
    if previous_complete['status'] != 'complete':
        raise ValueError('Expected a completed reference run')
    protocol = io.read(previous_run/'protocol.json')
    entries = [e for e in io.read(checked(protocol['inputs'])) if e['resource_variant'] == REFERENCE]
    if not entries:
        raise ValueError('Reference Q contract absent')
    paired, fresh, audits = {}, {}, []
    for entry in entries:
        context = io.read(checked(entry['resource_context']))
        with np.load(checked(entry['files']['train.npz']), allow_pickle=False) as saved:
            labels = list(map(int, saved['labels']))
            p, q = saved['P']*BASE_KVA, saved['Q']*BASE_KVA
        if p.shape != (192, len(labels)) or q.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError('Expected the original 192 training P/Q samples')
        mappings = {int(row['label']): row for row in context['node_mapping']}
        registrations = {int(row['feeder']): row for row in context['resource_metadata_at_history_start']}
        old = deepcopy(entry['q_resources'])
        old_by_node = {int(row['node_label']): row for row in old}
        if len(old_by_node) != len(old):
            raise ValueError('Duplicate Q node')
        q95 = np.quantile(np.abs(q), .95, axis=0)
        new, rows = [], []
        for resource in old:
            label = int(resource['node_label']); k = labels.index(label)
            mapping = mappings[label]; reg = registrations[int(mapping['feeder_id'])]
            registered = float(reg.get('PV_systems_kW', 0.) or 0.) + float(reg.get('batteries_kW', 0.) or 0.)
            if registered*float(mapping['scale']) <= 0:
                raise ValueError('Reference participant lacks original eligibility')
            capacity = max(float(resource['capacity_kvar']), tier(float(q95[k]), CONTRACT_TIERS))
            installed = tier(capacity/RESERVED_FRACTION, EQUIPMENT_TIERS)
            r = deepcopy(resource)
            r.update(capacity_kvar=capacity,
                capacity_origin='Training abs(Q) 95th percentile rounded up to fixed contract tiers; never reduce existing contract',
                equipment_type='additional independent bidirectional converter compensator',
                assumed_equipment_rating_kvar=installed,
                equipment_reservation_fraction_limit=RESERVED_FRACTION,
                remote_signed_Q_control='Explicit assumed service contract; vendor rating alone does not certify command interface',
                all_day_service='Independent energized compensator, not unverified PV inverter night headroom')
            if capacity > RESERVED_FRACTION*installed+1e-12:
                raise ArithmeticError('Insufficient assumed equipment capacity')
            new.append(r)
        new_by_node = {int(row['node_label']): row for row in new}
        for k, label in enumerate(labels):
            mapping = mappings[label]; reg = registrations[int(mapping['feeder_id'])]
            chosen = new_by_node.get(label)
            rows.append(dict(node_label=label,
                scaled_registered_PV_kw=float(reg.get('PV_systems_kW', 0.) or 0.)*float(mapping['scale']),
                scaled_registered_battery_kw=float(reg.get('batteries_kW', 0.) or 0.)*float(mapping['scale']),
                training_abs_Q95_kvar=float(q95[k]), training_abs_Q_max_kvar=float(np.abs(q[:, k]).max()),
                training_S95_kva=float(np.quantile(np.hypot(p[:, k], q[:, k]), .95)),
                previous_contract_kvar=0. if label not in old_by_node else float(old_by_node[label]['capacity_kvar']),
                updated_contract_kvar=0. if chosen is None else chosen['capacity_kvar'],
                assumed_equipment_rating_kvar=0. if chosen is None else chosen['assumed_equipment_rating_kvar'],
                participation_basis='historical PV/battery registration; additional Q service assumed' if chosen else 'no contract under retained participation rule; installation is physically possible'))
        identity = entry['original_id']
        paired[identity] = {UPDATED: new}
        fresh[identity] = {REFERENCE: old, UPDATED: new}
        audits.append(dict(id=identity, resource_context=entry['resource_context'], training=entry['files']['train.npz'],
            source_nodes=rows, resource_ids_in_order=[r['id'] for r in new],
            total_previous_kvar=sum(r['capacity_kvar'] for r in old),
            total_updated_kvar=sum(r['capacity_kvar'] for r in new),
            total_assumed_equipment_kvar=sum(r['assumed_equipment_rating_kvar'] for r in new),
            old_resource_domain_contained=True,
            unchanged_ids_order_prices=True))
        if [(r['id'], r['node_label']) for r in old] != [(r['id'], r['node_label']) for r in new]:
            raise ArithmeticError('Device identities would change paired response streams')
    policy = dict(source=io.ref(__file__), previous_run=str(previous_run), previous_protocol=io.ref(previous_run/'protocol.json'),
        previous_complete=io.ref(previous_run/'complete.json'), contract_tiers_kvar=list(CONTRACT_TIERS),
        engineering_equipment_tiers_kvar=list(EQUIPMENT_TIERS), maximum_reserved_fraction=RESERVED_FRACTION,
        rule='q_new=max(q_old, tier(quantile(abs(training_Q), .95))); equipment=tier(q_new/.8)',
        units='Historical P/kW, Q/kvar and S/kVA from fixed 200 kVA normalization; equipment ratings are separately assumed',
        information_used='Historical resource registration and 192 training P/Q observations only',
        outcome_or_true_topology_used=False, forecast_used=False,
        interpretation='Engineering scenario: additional independent bidirectional Q contracts; no claim of measured installed SVG inventory',
        P_contracts='Unchanged seven contracts, windows, energy constraints, prices and resource order',
        prices='Same illustrative activation prices as reference; added hardware, reservation and auxiliary energy unpriced',
        availability='24 h reserved independent compensators; signed remote dispatch is a declared contract assumption',
        product_rating_sources=[
            'https://smone.pl/produkty/sm-asvg-3-04-4l-cw',
            'https://smone.pl/files/SVG%20ASVG%20Manual%2020251212v1.6.7.pdf'])
    out.mkdir(parents=True)
    io.write(out/'paired_placements.json', dict(policy, configurations_by_input=paired))
    io.write(out/'fresh_placements.json', dict(policy, configurations_by_input=fresh))
    io.write(out/'node_audit.json', dict(policy, rows=audits))
    receipt = dict(status='complete', source=io.ref(__file__), node_audit=io.ref(out/'node_audit.json'),
        paired=io.ref(out/'paired_placements.json'), fresh=io.ref(out/'fresh_placements.json'),
        totals=[{k:a[k] for k in ('id','total_previous_kvar','total_updated_kvar','total_assumed_equipment_kvar')} for a in audits])
    io.write(out/'complete.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', type=Path, default=Path('outputs/flexibility_reaudit_long180_20261114_v1'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.previous_run, args.out)['totals'])
