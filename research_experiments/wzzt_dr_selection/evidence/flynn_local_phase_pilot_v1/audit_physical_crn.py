"""Independent saved-data physical/CRN audit. No project or solver imports.

Prepare may read frozen plans/protocol only. Run refuses all future data reads
until ended.json AND fits_frozen.json exist at the pilot root.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pickle
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(root, folder):
    folder.mkdir(parents=True, exist_ok=True)
    protocol_path = folder / 'audit_protocol.json'
    if protocol_path.exists():
        raise FileExistsError('frozen audit protocol retained')
    p = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    plan = json.loads((root / 'plans.json').read_text(encoding='utf-8'))
    assert sha(root / 'plans.json') == p['plan_sha256']
    assert p['price_optimization'] is False
    value = dict(id='flynn_pilot_physical_crn_audit_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), pilot_root=str(root), pilot_protocol_sha256=sha(root / 'protocol.json'),
        plans_sha256=sha(root / 'plans.json'), policies=p['policies'], labels=plan['labels'],
        expected_physical_event_records=32, expected_AC_timepoint_records=3072,
        guard='ended.json and fits_frozen.json must both exist before every future/truth load',
        tests=['frozen receipt/artifact hashes', 'four ordered observed events',
            'natural P/Q/root CRN across all policies', 'mask law and no redraw',
            'signed whole task P/Q increments', 'power/energy/settlement',
            'common standardized P/Q/V noise and common root observation',
            'same/opposed identical prices/participants/cost', 'actual balanced vs partial counts'],
        limits=dict(new_AC_calls=0, new_identifier_calls=0, new_solver_calls=0),
        tolerance=dict(power_kw=1e-10, standardized_noise=5e-10),
        stage='frozen_before_future_or_truth_read')
    write(protocol_path, value)
    print(json.dumps(dict(status='audit_prepared', source_sha256=value['source_sha256'])))


def run(root, folder):
    import numpy as np
    protocol_path = folder / 'audit_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    assert sha(__file__) == protocol['source_sha256'], 'audit source drift'
    assert sha(root / 'protocol.json') == protocol['pilot_protocol_sha256'], 'pilot protocol drift'
    assert sha(root / 'plans.json') == protocol['plans_sha256'], 'plans drift'
    # Do not move this guard below any future receipt/packet/truth read.
    if not (root / 'ended.json').exists() or not (root / 'fits_frozen.json').exists():
        raise RuntimeError('pilot is not ended and all fit attempts frozen; no future/truth data read')
    out = folder / 'receipt.json'
    if out.exists():
        raise FileExistsError('terminal audit receipt retained; do not overwrite')
    p = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    plan = json.loads((root / 'plans.json').read_text(encoding='utf-8'))
    checks, cases, file_hashes = [], [], {}
    maxima = dict(power_kw=0., reactive_kvar=0., energy_kWh=0., standardized_noise_difference=0.)
    def check(name, good, details=None):
        checks.append(dict(name=name, passed=bool(good), details=details))
    def same(name, a, b, *, atol=1e-10):
        a, b = np.asarray(a), np.asarray(b)
        good = a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=0.)
        error = float(np.max(abs(a - b), initial=0.)) if a.shape == b.shape else None
        check(name, good, error)
        return error
    def read_json(path):
        if not (root / 'ended.json').exists() or not (root / 'fits_frozen.json').exists():
            raise RuntimeError('fit freeze marker removed')
        file_hashes[str(path)] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def load(path):
        if not (root / 'ended.json').exists() or not (root / 'fits_frozen.json').exists():
            raise RuntimeError('fit freeze marker removed')
        file_hashes[str(path)] = sha(path)
        with path.open('rb') as stream:
            return pickle.load(stream)
    try:
        markers = {name: sha(root / name) for name in ('ended.json', 'fits_frozen.json')}
        write(folder / 'execution_binding.json', dict(audit_protocol_sha256=sha(protocol_path),
            marker_sha256=markers, execution_utc=datetime.now(timezone.utc).isoformat()))
        labels = list(plan['labels']); n = len(labels); base = float(plan['base_kw'])
        check('base_and_labels', base == 200. and labels == list(range(201, 217)))
        check('sigma_unchanged', p['pq_sigma'] == .005 and p['voltage_sigma'] == .0002)
        policies = list(p['policies'])
        natural_reference, noise_reference, root_reference = {}, {}, {}
        policy_data = {}
        for policy in policies:
            directory = root / policy / 'physical'
            path = directory / 'events_receipt.json'
            if not path.exists():
                cases.append(dict(policy=policy, status='not_started_or_no_receipt', events_audited=0))
                continue
            receipt = read_json(path)
            prices = np.asarray(plan['plans'][policy], int)
            directions = np.asarray(plan['directions'][policy], int)
            template = plan['templates'][policy]
            worst = .4 * float(prices.sum())
            check(policy + '.price_shape', prices.shape == (2, n) and np.isin(prices, [0, 1, 2]).all())
            check(policy + '.budget', worst <= 1.6 + 1e-12)
            check(policy + '.worst_payment', abs(float(receipt.get('worst_payment', worst)) - worst) <= 1e-12)
            for name, ref in receipt.get('private_files', {}).items():
                artifact = directory / name
                check(policy + '.artifact.' + name, artifact.exists() and sha(artifact) == ref['sha256'])
            if not (directory / 'packet.pkl').exists() or not (directory / 'physical_truth.pkl').exists():
                cases.append(dict(policy=policy, status=receipt.get('status'), events_audited=0,
                                  actual_payment=receipt.get('actual_payment'), reason='no complete physical artifacts'))
                continue
            packet = load(directory / 'packet.pkl')
            truth = load(directory / 'physical_truth.pkl')
            event_truth = {int(t['event']): t for t in truth['events']}
            event_record = {int(t['event']): t for t in receipt.get('event_records', [])}
            event_scene = {}
            for role in ('training', 'validation'):
                ids = packet['metadata'][role + '_event_ids']
                check(policy + '.' + role + '_event_count', len(ids) == len(packet[role]))
                for event, scene in zip(ids, packet[role]):
                    event_scene[int(event)] = scene
            expected_ids = [0, 1, 2, 3]
            if receipt.get('status') == 'complete':
                check(policy + '.four_complete_events', sorted(event_truth) == expected_ids
                      and sorted(event_scene) == expected_ids and sorted(event_record) == expected_ids)
            check(policy + '.observed_packet_only', packet['metadata']['observed_only'] is True
                  and 'truth' not in packet and 'physical_network' not in packet)
            check(policy + '.packet_bases', packet['metadata']['base_kw'] == base
                  and packet['metadata']['base_mva'] == .2 and packet['metadata']['base_kv'] == .4
                  and packet['metadata']['root_label'] == 1)
            committed, known_paid, unknown = 0., 0., 0
            events = []
            for event in range(4):
                record = event_record.get(event)
                if record is None:
                    events.append(dict(event=event, status='missing_record'))
                    continue
                price = prices[event % 2]
                z = np.asarray(record['accepted_participants'], bool)
                u = np.random.default_rng(np.random.SeedSequence([int(p['participation_seed']), int(p['path']), event])).random(n)
                expected_z = u < np.asarray([0., .5, .75])[price]
                check(policy + f'.e{event}.mask_law', np.array_equal(z, expected_z))
                check(policy + f'.e{event}.mask_retained', np.array_equal(z, np.asarray(record['participants'], bool)))
                fee = .2 * float(price @ z)
                committed += fee
                check(policy + f'.e{event}.commitment', abs(record['committed_payment'] - fee) <= 1e-12)
                if record.get('actual_payment') is None:
                    unknown += 1
                else:
                    known_paid += float(record['actual_payment'])
                if event not in event_truth or event not in event_scene:
                    events.append(dict(event=event, status=record['status'], actual_payment=record.get('actual_payment')))
                    continue
                t, scene = event_truth[event], event_scene[event]
                true_p, true_q, true_v, root_true = [np.asarray(t[k], float) for k in ('P_pu', 'Q_pu', 'V_pu', 'root_pu')]
                natural_p, natural_q = [np.asarray(t[k], float) for k in ('natural_P_pu', 'natural_Q_pu')]
                check(policy + f'.e{event}.physical_shapes', true_p.shape == true_q.shape == true_v.shape == (96, n)
                      and natural_p.shape == natural_q.shape == (96, n) and root_true.shape == (96,))
                orientation = int(plan['orientations_by_event'][event])
                event_direction = directions[event % 2] * orientation
                same(policy + f'.e{event}.recorded_directions', record['directions'], event_direction, atol=0.)
                wave = np.r_[np.full(4, -.2), np.full(4, .2)]
                expected = np.zeros((96, n))
                expected[76:84] = wave[:, None] * event_direction[None, :] * z[None, :]
                if template == 'alternating_opposed':
                    raise ValueError('this locked pilot does not declare alternating profiles')
                delta_p, delta_q = (true_p - natural_p) * base, (true_q - natural_q) * base
                ep = same(policy + f'.e{event}.signed_P_delta', delta_p, expected)
                eq = same(policy + f'.e{event}.signed_Q_delta', delta_q, .35 * expected)
                maxima['power_kw'] = max(maxima['power_kw'], ep or 0.)
                maxima['reactive_kvar'] = max(maxima['reactive_kvar'], eq or 0.)
                net = delta_p.sum(axis=0) * .25
                reduced = np.maximum(-delta_p, 0.).sum(axis=0) * .25
                added = np.maximum(delta_p, 0.).sum(axis=0) * .25
                er = same(policy + f'.e{event}.zero_net_energy', net, np.zeros(n))
                maxima['energy_kWh'] = max(maxima['energy_kWh'], er or 0.)
                same(policy + f'.e{event}.delivered_reduction', reduced, .2 * z)
                same(policy + f'.e{event}.delivered_rebound', added, .2 * z)
                same(policy + f'.e{event}.receipt_reduction', record['delivered_reduction_kWh'], reduced)
                same(policy + f'.e{event}.receipt_rebound', record['delivered_rebound_kWh'], added)
                check(policy + f'.e{event}.settlement', record['actual_payment'] is not None
                      and abs(float(record['actual_payment']) - fee) <= 1e-12)
                device_power = .2 + expected[76:84]
                device_energy = np.cumsum(device_power, axis=0) * .25
                check(policy + f'.e{event}.device_power_energy', device_power.min() >= -1e-12
                      and device_power.max() <= .4 + 1e-12 and device_energy.min() >= -1e-12
                      and device_energy.max() <= .4 + 1e-12 and np.max(abs(device_energy[-1] - .4)) <= 1e-12)
                observed = [scene[k].to_numpy(dtype=float) for k in ('P_terminal', 'Q_terminal', 'V_terminal')]
                root_observed = scene['V_root_observed'].to_numpy(dtype=float)
                check(policy + f'.e{event}.observed_shape_labels', all(x.shape == (96, n) for x in observed)
                      and list(scene['P_terminal'].columns) == labels and root_observed.shape == (96,))
                noises = [(m - t_) / (sigma * np.maximum(abs(t_), 1e-12))
                          for m, t_, sigma in zip(observed, (true_p, true_q, true_v), (.005, .005, .0002))]
                meter_rng = np.random.default_rng(int(record['profile_seed']) + 2000)
                for channel, noise in zip(('P', 'Q', 'V'), noises):
                    expected_noise = meter_rng.standard_normal((96, n))
                    same(policy + f'.e{event}.nominal_noise_' + channel, noise, expected_noise, atol=5e-10)
                replicate = int(p['replicate_base']) + 10 * int(p['path']) + event
                root_seed = 70_000_019 + 100_003 * replicate + int(p['event_profiles'][event])
                root_noise = (root_observed - root_true) / (.0002 * np.maximum(abs(root_true), 1e-12))
                same(policy + f'.e{event}.root_noise_seed', root_noise,
                     np.random.default_rng(root_seed).standard_normal(96), atol=5e-10)
                if event not in natural_reference:
                    natural_reference[event] = (natural_p, natural_q, root_true)
                    noise_reference[event] = noises
                    root_reference[event] = root_observed
                else:
                    for channel, current, reference in zip(('P', 'Q', 'root'),
                            (natural_p, natural_q, root_true), natural_reference[event]):
                        same(policy + f'.e{event}.natural_CRN_' + channel, current, reference, atol=0.)
                    for channel, current, reference in zip(('P', 'Q', 'V'), noises, noise_reference[event]):
                        error = same(policy + f'.e{event}.standardized_noise_CRN_' + channel,
                                     current, reference, atol=5e-10)
                        maxima['standardized_noise_difference'] = max(maxima['standardized_noise_difference'], error or 0.)
                    same(policy + f'.e{event}.root_observation_CRN', root_observed, root_reference[event], atol=0.)
                pair_labels = plan['active_labels_by_row'][policy][event % 2]
                pair = [labels.index(v) for v in pair_labels]
                active_count = int(z[pair].sum())
                full_pair = active_count == 2
                nonzero = bool(np.max(abs(delta_p), initial=0.) > 1e-10)
                balanced = bool(nonzero and np.max(abs(delta_p.sum(axis=1)), initial=0.) <= 1e-10)
                events.append(dict(event=event, status=record['status'], pair_labels=pair_labels,
                    accepted_count_in_pair=active_count, full_pair=full_pair,
                    nonzero_balanced_PQ=balanced, actual_payment=record['actual_payment'],
                    energy_returned=True, standardized_noise_checked=True))
            check(policy + '.committed_total', abs(receipt['committed_payment'] - committed) <= 1e-12)
            check(policy + '.known_paid_total', abs(receipt['known_actual_payment'] - known_paid) <= 1e-12)
            check(policy + '.unknown_payment_count', receipt['unknown_payment_events'] == unknown)
            if not unknown:
                check(policy + '.actual_payment_total', abs(receipt['actual_payment'] - known_paid) <= 1e-12)
            else:
                check(policy + '.unknown_payment_retained', receipt['actual_payment'] is None)
            policy_data[policy] = dict(receipt=receipt, record=event_record, prices=prices, events=events)
            cases.append(dict(policy=policy, status=receipt['status'], events_audited=sum(e.get('standardized_noise_checked', False) for e in events),
                              committed_payment=committed, actual_payment=receipt['actual_payment'], events=events))
        if 'same_local' in policy_data and 'opposed_local' in policy_data:
            a, b = policy_data['same_local'], policy_data['opposed_local']
            same('phase_pair.identical_prices', a['prices'], b['prices'], atol=0.)
            for event in range(4):
                if event in a['record'] and event in b['record']:
                    ar, br = a['record'][event], b['record'][event]
                    same(f'phase_pair.e{event}.identical_participation', ar['accepted_participants'], br['accepted_participants'], atol=0.)
                    check(f'phase_pair.e{event}.identical_cost', ar['actual_payment'] == br['actual_payment']
                          and ar['committed_payment'] == br['committed_payment'])
            check('phase_pair.identical_total_cost', a['receipt']['actual_payment'] == b['receipt']['actual_payment'])
        failed = [v for v in checks if not v['passed']]
        complete = sum(c['status'] == 'complete' for c in cases)
        result = dict(status='passed' if not failed else 'failed', checks_total=len(checks),
            checks_passed=len(checks) - len(failed), failed=failed, checks=checks,
            planned_policies=len(policies), complete_physical_policies=complete,
            physical_event_records_audited=sum(c['events_audited'] for c in cases),
            AC_timepoint_records_audited=96 * sum(c['events_audited'] for c in cases),
            unique_natural_events=len(natural_reference), cases=cases, maxima=maxima,
            file_sha256=file_hashes, marker_sha256=markers, audit_source_sha256=sha(__file__),
            audit_protocol_sha256=sha(protocol_path), new_AC_calls=0, new_identifier_calls=0,
            new_solver_calls=0, new_future_generated=False,
            limitations=['one shared natural path; not independent recovery samples',
                         'saved AC equation receipts were not independently resolved or graph-replayed',
                         'synthetic device contract, no empirical availability calibration'])
        write(out, result)
        summary = {k: result[k] for k in ('status', 'checks_total', 'checks_passed', 'planned_policies',
            'complete_physical_policies', 'physical_event_records_audited', 'AC_timepoint_records_audited',
            'unique_natural_events', 'maxima', 'new_AC_calls', 'new_identifier_calls', 'new_solver_calls')}
        summary['same_opposed_actual_cost_equal'] = all(c['passed'] for c in checks if c['name'].startswith('phase_pair.'))
        summary['private_receipt_sha256'] = sha(out)
        summary['cases'] = cases
        write(folder / 'summary.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}))
    except BaseException as exc:
        write(out, dict(status='audit_error', checks_total=len(checks), checks=checks,
            exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
            new_AC_calls=0, new_identifier_calls=0, new_solver_calls=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = args.folder.resolve() if args.folder else root / 'audit_v1/physical'
    (prepare if args.mode == 'prepare' else run)(root, folder)
