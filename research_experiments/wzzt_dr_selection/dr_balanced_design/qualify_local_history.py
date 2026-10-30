"""One locked historical qualification; no fitting, graph or future access.

Reuse existing observed loader, forecast, synthetic-device and price helpers.
This is an input audit, not a new DR controller or identifier.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
from datetime import datetime, timezone

REPO = Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
BASE = REPO / 'research_experiments/wzzt_dr_selection'
JOB = 'main__flynn16__b7__reference__v0002__N288'
METADATA = Path(r'D:\0-github_workspace\Topo\outputs\wzzt_paper_20261029_full\networks\flynn16\reference\metadata.json')
METADATA_SHA = '37344b520c0277b9052e1901789b99c7287e3cd061cfb21999fcbf0e811d8432'
REGIONS = {'left': [201, 202, 203, 204, 205, 206, 213, 215],
           'right': [207, 208, 209, 210, 211, 212, 214, 216]}
TRIPLES = {'left': [201, 202, 203], 'right': [207, 208, 209]}
ANCHORS = {'left': 201, 'right': 209}
sys.path.insert(0, str(REPO))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sources():
    files = [Path(__file__), BASE / 'dr_competition_bridge/physical_bridge.py',
             BASE / 'dr_coded_selection/coded_physical.py', BASE / 'dr_competition_bridge/pricing_tables.py']
    return {str(p): sha(p) for p in files}


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'qualification_protocol.json'
    if path.exists():
        raise FileExistsError('prior qualification protocol retained')
    inventory = BASE / 'evidence/scenario_review_v1/source_inventory.json'
    row = next(v for v in json.loads(inventory.read_text(encoding='utf-8'))['rows'] if v['job_id'] == JOB)
    inputs = {role: {k: row[role][k] for k in ('path', 'catalog_sha256')} for role in ('train', 'validation')}
    assert inputs['train']['catalog_sha256'] == 'c7b96b7e32bddeb010a160c39c0ac2a6fc2ee7d849429494b1313198caa18d53'
    assert inputs['validation']['catalog_sha256'] == '0cfdb833eda9dcb4968eefab7de9759562d9db64625137e71e5707f107c465df'
    assert sha(METADATA) == METADATA_SHA
    protocol = dict(id='flynn_local_history_qualification_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources(), inventory_path=str(inventory), inventory_sha256=sha(inventory), job_id=JOB,
        source_metadata=dict(path=str(METADATA), sha256=METADATA_SHA, case_name='flynn_balanced_lv_16'),
        inputs=inputs, expected_case='flynn16', expected_labels=list(range(201, 217)),
        expected_root=1, base_kw=200., expected_base_kv=.4, expected_base_mva=.2,
        regions=REGIONS, disputed_triples=TRIPLES, anchors=ANCHORS,
        device=dict(baseline_kw=.2, baseline_kvar=.07, power_upper_kw=.4, energy_terminal_kWh=.4,
                    interval_hours=.25, efficiency=1., empirical_calibration=False),
        qualification_rule='both task signs pass forecast and all6 historical device windows; no replacement of failed anchors',
        endpoint_rule='list all qualified anchor/outside-triple pairs within each declared historical region; no endpoint or price selected',
        allowed_reads=['locked source metadata', 'catalog metadata', 'train_observed.npz', 'validation_observed.npz'],
        forbidden_reads=['test_observed.npz', 'true_network', 'diagnostics.json', 'actual future'],
        limits=dict(real_observed_NPZ=2, LP_calls=0, identifier_calls=0, AC_calls=0),
        status='frozen_before_reading_observed_values')
    write(path, protocol)
    print(json.dumps(dict(status='frozen', protocol_sha256=sha(path))))


def run(root):
    import numpy as np
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import load_history, forecast_mean
    from research_experiments.wzzt_dr_selection.dr_coded_selection.coded_physical import device_precheck, waveforms
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.pricing_tables import enumerate_menus
    protocol_path = root / 'qualification_protocol.json'
    p = json.loads(protocol_path.read_text(encoding='utf-8'))
    assert p['source_sha256'] == sources(), 'qualification source drift'
    assert sha(p['inventory_path']) == p['inventory_sha256'], 'inventory drift'
    if (root / 'qualification_result.json').exists() or (root / 'qualification_ended.json').exists():
        raise FileExistsError('terminal qualification retained; do not rerun')
    lock = root / 'qualification.lock'
    with lock.open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), protocol_sha256=sha(protocol_path)), f)
    try:
        row = next(v for v in json.loads(Path(p['inventory_path']).read_text(encoding='utf-8'))['rows'] if v['job_id'] == JOB)
        for role in ('train', 'validation'):
            assert all(row[role][k] == v for k, v in p['inputs'][role].items())
            path = Path(row[role]['path'])
            assert path.parent.name == 'v0002' and path.parent.parent.name == 'reference'
            assert path.parent.parent.parent.name == 'b7' and path.parent.parent.parent.parent.name == 'flynn16'
        history = load_history(row, expected_case=p['expected_case'], base_kw=p['base_kw'], source_metadata=p['source_metadata'])
        labels = history['labels']
        assert labels == p['expected_labels'] and len(labels) == 16
        assert history['root_label'] == p['expected_root'] and history['base_mva'] == p['expected_base_mva']
        assert history['base_kv'] == p['expected_base_kv']
        forecast = forecast_mean(history)
        scenes = history['training'] + history['validation']
        n = len(labels)
        runs = []
        for index, scene in enumerate(scenes + [None]):
            power = forecast['P_pu'] if scene is None else scene['P_terminal'].to_numpy()
            reactive = forecast['Q_pu'] if scene is None else scene['Q_terminal'].to_numpy()
            for sign in (1, -1):
                delta = waveforms('same', np.full(n, sign), [])
                out = device_precheck(power, reactive, np.ones(n, bool), delta, base_kw=p['base_kw'])
                runs.append(dict(source='forecast' if scene is None else 'historical_day_' + str(index),
                                 sign=sign, **out))
        node_rows = []
        for i, label in enumerate(labels):
            hist = [r for r in runs if r['source'] != 'forecast']
            future_mean = [r for r in runs if r['source'] == 'forecast']
            node_rows.append(dict(label=label,
                forecast_both_signs=all(r['node_status'][i] == 'available' for r in future_mean),
                historical_all6_both_signs=all(r['node_status'][i] == 'available' for r in hist),
                min_observed_window_P_kw=min(r['min_natural_window_P_kw'][i] for r in hist),
                min_observed_window_Q_kvar=min(r['min_natural_window_Q_kvar'][i] for r in hist),
                forecast_window_P_min_kw=future_mean[0]['min_natural_window_P_kw'][i],
                forecast_window_Q_min_kvar=future_mean[0]['min_natural_window_Q_kvar'][i]))
        good = {r['label'] for r in node_rows if r['forecast_both_signs'] and r['historical_all6_both_signs']}
        pairs = {}
        for side in ('left', 'right'):
            anchor = ANCHORS[side]
            others = sorted(set(REGIONS[side]) - set(TRIPLES[side]))
            pairs[side] = dict(anchor=anchor, anchor_qualified=anchor in good,
                counterpart_candidates=others, qualified_counterparts=[v for v in others if v in good],
                admissible_pairs=[[anchor, v] for v in others if anchor in good and v in good])
        menus = enumerate_menus(2)
        assert menus.shape == (50, 2, 2) and (menus.sum(axis=(1, 2)) <= 4).all()
        result = dict(status='complete', job_id=JOB, protocol_sha256=sha(protocol_path),
            source_sha256=sources(), history_receipt=history['receipt'], terminal_labels=labels,
            root_label=history['root_label'], base_mva=history['base_mva'], base_kv=history['base_kv'],
            base_kw=history['base_kw'], increment_P_pu=.2 / p['base_kw'], increment_Q_pu=.07 / p['base_kw'],
            training_rows=288, validation_rows=288, original_PQ_channels_retained=True,
            forecast='same-slot mean of six independent synthetic historical days; no real-calendar holdout claim',
            device_evaluations=len(runs), node_sign_day_cells=len(runs) * n,
            nodes=node_rows, local_roles=pairs, forecast_qualified_nodes=sum(v['forecast_both_signs'] for v in node_rows),
            history_and_forecast_qualified_nodes=len(good), price_menus_per_fixed_two_local_pairs=50,
            selected_price=None, selected_pair=None, LP_calls=0, identifier_calls=0, AC_calls=0,
            observed_NPZ_loaded=2, test_or_truth_loaded=False, actual_future_read=False,
            field_device_feasibility='unknown; compatibility with declared synthetic device only',
            feeder_capacity_verified=False, network_tree_used=False,
            environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__))
        # Full observed and forecast trajectories remain in the private root.
        with (root / 'history_and_forecast.pkl').open('xb') as stream:
            pickle.dump(dict(history=history, forecast=forecast), stream)
        write(root / 'qualification_result.json', result)
        write(root / 'qualification_ended.json', dict(status='complete', private_packet_sha256=sha(root / 'history_and_forecast.pkl')))
        print(json.dumps({k: result[k] for k in ('status', 'terminal_labels', 'training_rows', 'validation_rows',
            'increment_P_pu', 'increment_Q_pu', 'forecast_qualified_nodes', 'history_and_forecast_qualified_nodes', 'local_roles')}))
    except BaseException as exc:
        write(root / 'qualification_ended.json', dict(status='failed', exception_type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)(args.root)
