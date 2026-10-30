"""Independent observed-only audit; no project helper, solver or future import.

Recomputes saved qualification with NumPy and explicit contract arithmetic.
This is an audit after reading its recorded result, not a new pre-registered
performance experiment. All observations and detailed failure cells stay private.
"""
from pathlib import Path
from itertools import product
import hashlib
import json
import pickle
import platform
import sys
import traceback
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit_root = Path(sys.argv[1]).resolve()
    parent = audit_root.parent
    protocol_path = audit_root / 'audit_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    output = audit_root / 'independent_receipt.json'
    if output.exists():
        raise FileExistsError('audit result cannot be overwritten')
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)
    try:
        check('audit_source_frozen', sha(Path(__file__)) == protocol['source_sha256'])
        for name, expected in protocol['packet_bindings'].items():
            check('binding_' + name, sha(parent / name) == expected)
        original_protocol = json.loads((parent / 'qualification_protocol.json').read_text(encoding='utf-8'))
        recorded = json.loads((parent / 'qualification_result.json').read_text(encoding='utf-8'))
        check('producer_protocol_bound', sha(parent / 'qualification_protocol.json') == recorded['protocol_sha256'])
        with (parent / 'history_and_forecast.pkl').open('rb') as stream:
            packet = pickle.load(stream)
        check('only_history_and_forecast_in_packet', set(packet) == {'history', 'forecast'})
        history, forecast = packet['history'], packet['forecast']
        labels = list(range(201, 217))
        check('labels_exactly_201_to216', history['labels'] == labels == forecast['labels'] == recorded['terminal_labels'])
        check('root_is_one_not_a_terminal', history['root_label'] == recorded['root_label'] == 1 and 1 not in labels)
        check('power_and_voltage_bases', history['base_mva'] == .2 and history['base_kv'] == .4 and history['base_kw'] == 200.)
        metadata_path = Path(original_protocol['source_metadata']['path'])
        check('source_metadata_hash', sha(metadata_path) == original_protocol['source_metadata']['sha256'])
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        check('metadata_case_root_and_bases', metadata['source']['case_name'] == 'flynn_balanced_lv_16'
              and metadata['root_bus'] == 1 and metadata['base_mva'] == .2 and metadata['base_kv'] == .4)
        check('P_increment_conversion', .2 / 200. == recorded['increment_P_pu'] == .001)
        check('Q_increment_conversion', abs(.07 / 200. - recorded['increment_Q_pu']) < 1e-18)
        daily = []
        for split, role, offset in [('training', 'train', 0), ('validation', 'validation', 3)]:
            scenes = history[split]
            check(split + '_three_complete_days', len(scenes) == 3)
            source = original_protocol['inputs'][role]
            path = Path(source['path'])
            check(role + '_observed_filename', path.name == role + '_observed.npz')
            check(role + '_original_content_hash', sha(path) == source['catalog_sha256'])
            # Reopen only the two whitelisted past observed files; no truth or test.
            with np.load(path, allow_pickle=False) as archive:
                check(role + '_observed_fields_only', set(archive.files) == {'labels', 'P', 'Q', 'V', 'root'})
                check(role + '_source_labels', np.array_equal(archive['labels'], labels))
                arrays = {k: np.asarray(archive[k]) for k in ('P', 'Q', 'V', 'root')}
            check(role + '_source_shape', all(arrays[k].shape == (3, 96, 16) for k in ('P', 'Q', 'V'))
                  and arrays['root'].shape == (3, 96))
            for day, scene in enumerate(scenes):
                index = scene['P_terminal'].index
                expected_dates = (np.datetime64('2000-01-01T00:00:00', 'ns')
                                  + np.timedelta64(offset + day, 'D')
                                  + np.arange(96) * np.timedelta64(15, 'm'))
                check(f'{role}_day{day}_synthetic_15min_dates', np.array_equal(index.to_numpy(), expected_dates))
                for field, key in [('P_terminal', 'P'), ('Q_terminal', 'Q'), ('V_terminal', 'V'), ('V_root_observed', 'root')]:
                    value = scene[field]
                    check(f'{role}_day{day}_{key}_index', value.index.equals(index))
                    check(f'{role}_day{day}_{key}_equals_source', np.array_equal(value.to_numpy(), arrays[key][day]))
                    check(f'{role}_day{day}_{key}_finite', np.isfinite(value.to_numpy()).all())
                    if key != 'root':
                        check(f'{role}_day{day}_{key}_terminal_order', value.columns.tolist() == labels)
                voltage = scene['V_terminal'].to_numpy()
                root_voltage = scene['V_root_observed'].to_numpy()
                check(f'{role}_day{day}_positive_voltage', (voltage > 0).all() and (root_voltage > 0).all())
                check(f'{role}_day{day}_squared_drop_target', np.array_equal(scene['drop_target'].to_numpy(), root_voltage[:, None] ** 2 - voltage ** 2))
                check(f'{role}_day{day}_units', scene['units'] == dict(P_terminal='pu', Q_terminal='pu', V_terminal='pu', V_root_observed='pu', drop_target='pu2'))
                daily.append((f'{role}_day{day}', scene['P_terminal'].to_numpy(), scene['Q_terminal'].to_numpy()))
            check(split + '_288_rows', sum(len(scene['P_terminal']) for scene in scenes) == 288)
        check('train_validation_dates_disjoint', history['training'][-1]['P_terminal'].index[-1] < history['validation'][0]['P_terminal'].index[0])
        observed_P = np.stack([value[1] for value in daily])
        observed_Q = np.stack([value[2] for value in daily])
        check('forecast_P_same_slot_six_day_mean', np.array_equal(forecast['P_pu'], observed_P.mean(axis=0)))
        check('forecast_Q_same_slot_six_day_mean', np.array_equal(forecast['Q_pu'], observed_Q.mean(axis=0)))
        check('forecast_grid_and_base', forecast['historical_days'] == 6 and forecast['slots'] == 96
              and forecast['dt_hours'] == .25 and forecast['base_kw'] == 200.)
        daily.append(('forecast', np.asarray(forecast['P_pu']), np.asarray(forecast['Q_pu'])))
        window = np.arange(76, 84)
        wave = np.array([-.2] * 4 + [.2] * 4)
        available = np.ones((7, 2, 16), dtype=bool)
        failures = []
        for day_idx, (name, P, Q) in enumerate(daily):
            p_kw, q_kvar = P[window] * 200., Q[window] * 200.
            residual_ok = (p_kw >= .2 - 1e-12).all(axis=0) & (q_kvar >= .07 - 1e-12).all(axis=0)
            for sign_idx, sign in enumerate((1, -1)):
                task_power = .2 + sign * wave
                task_energy = np.cumsum(task_power) * .25
                check(f'{name}_sign{sign}_power_bounds', min(task_power) >= -1e-12 and max(task_power) <= .4 + 1e-12)
                check(f'{name}_sign{sign}_energy_bounds_and_terminal', min(task_energy) >= -1e-12 and max(task_energy) <= .4 + 1e-12 and abs(task_energy[-1] - .4) <= 1e-12)
                check(f'{name}_sign{sign}_net_shift_zero', abs((sign * wave).sum() * .25) < 1e-12)
                available[day_idx, sign_idx] = residual_ok
                for node in np.flatnonzero(~residual_ok):
                    failures.append({'day': name, 'sign': sign, 'label': labels[node],
                        'P_min_kw': float(p_kw[:, node].min()), 'Q_min_kvar': float(q_kvar[:, node].min()),
                        'P_below_baseline_slots': [int(window[k]) for k in np.flatnonzero(p_kw[:, node] < .2 - 1e-12)],
                        'Q_below_baseline_slots': [int(window[k]) for k in np.flatnonzero(q_kvar[:, node] < .07 - 1e-12)]})
        forecast_nodes = available[-1].all(axis=0)
        all_nodes = available.all(axis=(0, 1))
        check('forecast_16_of16', int(forecast_nodes.sum()) == recorded['forecast_qualified_nodes'] == 16)
        check('history_and_forecast_15_of16', int(all_nodes.sum()) == recorded['history_and_forecast_qualified_nodes'] == 15)
        check('only203_fails_history', [label for label, ok in zip(labels, all_nodes) if not ok] == [203])
        for node in recorded['nodes']:
            pos = labels.index(node['label'])
            check(f'node{node["label"]}_recorded_forecast_flag', bool(forecast_nodes[pos]) == node['forecast_both_signs'])
            check(f'node{node["label"]}_recorded_history_flag', bool(available[:6, :, pos].all()) == node['historical_all6_both_signs'])
            p_min = float(observed_P[:, window, pos].min() * 200.)
            q_min = float(observed_Q[:, window, pos].min() * 200.)
            check(f'node{node["label"]}_recorded_minima', abs(p_min - node['min_observed_window_P_kw']) < 1e-12 and abs(q_min - node['min_observed_window_Q_kvar']) < 1e-12)
        for side in ('left', 'right'):
            anchor = original_protocol['anchors'][side]
            candidates = sorted(set(original_protocol['regions'][side]) - set(original_protocol['disputed_triples'][side]))
            check(side + '_anchor_and_all_partners_qualified', all_nodes[labels.index(anchor)] and len(candidates) == 5 and all(all_nodes[labels.index(v)] for v in candidates))
            check(side + '_recorded_partner_list', candidates == recorded['local_roles'][side]['qualified_counterparts'])
        prices = [list(v) for v in product((0, 1, 2), repeat=4) if sum(v) <= 4]
        check('exactly50_price_menus', len(prices) == recorded['price_menus_per_fixed_two_local_pairs'] == 50)
        check('four_event_worst_payment', abs(.4 * sum([1, 1, 1, 1]) - 1.6) < 1e-12)
        check('four_event_expected_low_pair_payment', abs(.4 * 4 * .5 - .8) < 1e-12)
        check('four_event_expected_high_single_rotation_payment', abs(.4 * 4 * .75 - 1.2) < 1e-12)
        check('shifted_energy_per_device_point2', abs(np.maximum(-wave, 0).sum() * .25 - .2) < 1e-12)
        check('whole_event_local_pair_pu_balance', np.allclose(wave / 200. + (-wave) / 200., 0., atol=0.)
              and np.allclose(.35 * wave / 200. + .35 * (-wave) / 200., 0., atol=0.))
        receipt = {'status': 'passed', 'scope': 'independent observed-history and declared-device qualification only',
            'checks': checks, 'check_count': len(checks), 'node_sign_day_cells_recomputed': int(available.size),
            'cell_pass_count': int(available.sum()), 'cell_fail_count': int((~available).sum()),
            'forecast_qualified': int(forecast_nodes.sum()), 'all_past_and_forecast_qualified': int(all_nodes.sum()),
            'failed_labels': [203], 'failure_details': failures,
            'two_training_local_pairs_both_complete_probability': '1/16 under independent p=.5',
            'four_events_all_pair_participants_complete_probability': '1/256 under independent p=.5',
            'source_sha256': sha(Path(__file__)), 'audit_protocol_sha256': sha(protocol_path),
            'original_protocol_sha256': sha(parent / 'qualification_protocol.json'),
            'packet_sha256': sha(parent / 'history_and_forecast.pkl'),
            'historical_NPZ_reopened_for_identity_check': 2, 'new_distinct_data_files': 0,
            'forbidden_future_truth_graph_reads': 0, 'project_functions_imported_or_called': False,
            'solver_calls': 0, 'identifier_calls': 0, 'physical_simulations': 0,
            'environment': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
            'boundary': 'Synthetic demand-only residual model; does not establish field availability, feeder feasibility, true topology or DR recovery.'}
    except Exception as exc:
        receipt = {'status': 'failed', 'checks': checks, 'error_type': type(exc).__name__,
                   'error': str(exc), 'traceback': traceback.format_exc(), 'source_sha256': sha(Path(__file__)),
                   'audit_protocol_sha256': sha(protocol_path)}
    output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('status', 'check_count', 'cell_pass_count', 'cell_fail_count', 'failed_labels') if k in receipt}))
    if receipt['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
