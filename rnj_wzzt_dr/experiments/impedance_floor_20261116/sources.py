"""Frozen 56-current / 400-legacy inputs; evaluation is a separate interface.

No fitting, generators, AC, or truth-based eligibility is performed here. The
400 single-day inputs have no independent new day. Missing cells stay in the
inventory and fail explicitly when loaded, rather than changing denominators.
"""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from ...observations import Observation, fingerprint, load

ROOT = Path(__file__).resolve().parents[3]
GROUPS = {
    'history288': ('outputs/history288_pq_noise_data_20261115_v1', 40,
                   'cc9c8db90a1be721d99bb98faead6c12a0cc28d8bbb9c9fd2ad1b91c0a9f231b'),
    'history10': ('outputs/history10_rx_20261116_v1', 16,
                  '33884afd579eec798fe94127f1bc39dee6721a15ce486afd156309734eef07bc'),
    'large96_main': ('topology_identification_standalone/outputs/large96_main_20261112_001', 240,
                     'ceac935650e280105c507e26ce780191fde4340cdd3a8fd3ba29028f8a6a8fc4'),
    'large96_pressure': ('topology_identification_standalone/outputs/large96_pressure_20261112_001', 160,
                         '20877c32c732d5083c2765908ccd37701251bc23bf40a23f6aa5a49de2016ec8'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(reference):
    if not isinstance(reference, dict) or not reference.get('sha256'):
        raise ValueError('Missing frozen reference')
    path = Path(reference['path'])
    if ref(path)['sha256'] != reference['sha256']:
        raise ValueError('Frozen source changed: ' + str(path))
    return path


def _snapshot_sources(root, directory, protocol):
    """Verify executed snapshots, not today's possibly extended source files."""
    if 'source_sha256' in protocol:
        refs = [dict(path=str(directory/'runtime'/name), sha256=sha)
                for name, sha in protocol['source_sha256'].items()]
    else:
        refs = []
        for source in protocol['sources']:
            relative = Path(source['path']).relative_to(root)
            refs.append(dict(path=str(directory/'source_snapshot'/relative), sha256=source['sha256']))
    for reference in refs:
        checked(reference)
    return refs


def _case_refs(root, directory, protocol, case):
    if 'evaluation_case_sha256' in protocol:
        return {name: dict(path=str(directory/'evaluation_cases'/case/name),
                          sha256=protocol['evaluation_case_sha256'][case+'/'+name])
                for name in ('branches.csv', 'metadata.json')}
    result = {}
    for name in ('branches.csv', 'metadata.json'):
        suffix = '/cases/' + case + '/' + name
        source = next(r for r in protocol['sources'] if r['path'].replace('\\', '/').endswith(suffix))
        result[name] = dict(path=str(directory/'source_snapshot'/Path(source['path']).relative_to(root)),
                            sha256=source['sha256'])
    return result


def _base_entry(group, task, root, directory, protocol, protocol_ref, snapshots):
    case = task['case']; case_refs = _case_refs(root, directory, protocol, case)
    meta = read(checked(case_refs['metadata.json']))  # units only, no branch parameters
    condition = task.get('profile', 'PQ_noise' if group == 'history288' else 'training_days')
    condition_value = task.get('level', task.get('pq_relative_sd', task.get('train_days')))
    return dict(id=group+'__'+task['id'], original_id=task['id'], case=case, case_key=case,
                source_group=group, group=group, protocol_ref=protocol_ref,
                source_condition=condition, condition_value=condition_value,
                phase_condition=f'{group}|{condition}|{condition_value}',
                executed_source_refs=snapshots, evaluation_case_refs=case_refs,
                base_kw=1000*float(meta['base_mva']),
                zbase_ohm=float(meta['base_kv'])**2/float(meta['base_mva']),
                history_repeat=task.get('history_repeat'), train_days=task.get('train_days'),
                pq_relative_sd=task.get('pq_relative_sd'), profile=task.get('profile', 'measured_net_PQ'),
                level=task.get('level'), availability='available', load_errors=[])


def _inventory_group(root, selected_group):
    root = Path(root).resolve(); entries = []
    if selected_group.startswith('large96_'):
        broad = read(root/'outputs/astra_inventory_20261114/mainline400.json')
        broad_lookup = {r['id']: r for r in broad}
        if len(broad) != 400 or len(broad_lookup) != 400:
            raise ValueError('Original 400-entry manifest is incomplete or duplicated')
    for group, (relative, planned, sha) in GROUPS.items():
        if group != selected_group:
            continue
        directory = root/relative; pref = dict(path=str(directory/'protocol.json'), sha256=sha)
        protocol = read(checked(pref)); snapshots = _snapshot_sources(root, directory, protocol)
        if group == 'history288':
            complete = read(directory/'complete.json')
            index = read(checked(complete['input_index']))
            by_id = {r['id']: r for r in index['inputs']}; tasks = protocol['tasks']
            # Same physical trees: use the later sealed response matrices only
            # in load_evaluation, after verifying their branch provenance.
            h10 = root/GROUPS['history10'][0]
            h10_prepared = read(h10/'prepared.json')
            h10_protocol = read(checked(h10_prepared['protocol']))
            if h10_prepared['protocol']['sha256'] != GROUPS['history10'][2]:
                raise ValueError('Cross-world physical-response provenance mismatch')
            h10_worlds = {w['case']: w for w in read(checked(h10_prepared['worlds']))}
        elif group == 'history10':
            prepared = read(directory/'prepared.json')
            if prepared['protocol'] != pref:
                # Equivalent canonical paths are allowed; the digest is binding.
                if checked(prepared['protocol']).resolve() != checked(pref).resolve():
                    raise ValueError('History10 prepare/protocol mismatch')
            worlds = {r['case']: r for r in read(checked(prepared['worlds']))}
            tasks = protocol['tasks']
        else:
            cohort = group.removeprefix('large96_')
            tasks = [dict(id=f'{cohort}__{job}__101', job=job, **read(directory/'evaluation_map.json')[job])
                     for job in sorted(protocol['inputs'])]
            seal = read(directory/'selection_seal.json')
            complete = read(directory/'completion.json')
            if complete['selection_seal_sha256'] != ref(directory/'selection_seal.json')['sha256']:
                raise ValueError('Legacy completion/selection seal mismatch')
            if seal['protocol_sha256'] != sha:
                raise ValueError('Legacy selection used a different protocol')
        if len(tasks) != planned:
            raise ValueError(f'{group}: expected {planned} planned tasks, got {len(tasks)}')
        for task in tasks:
            entry = _base_entry(group, task, root, directory, protocol, pref, snapshots)
            try:
                if group == 'history288':
                    source = by_id[task['id']]; files = source['files']
                    entry.update(labels=source['labels'], native_ref=files['101_native.json'],
                                 train_ref=files['train.npz'], validation_ref=files['validation.npz'],
                                 new_day_ref=files['new_day.npz'], train_rows=[0, 192], validation_rows=[0, 96],
                                 new_day_rows=[0, 96], source_row_indices=dict(train=[0, 192], validation=[192, 288], new_day=[288, 384]),
                                 candidate_seal_ref=files['candidates_sealed.json'], native_wrapper_key=None,
                                 source_index_ref=complete['input_index'], physical_history_group=source['original_id'])
                    h10_refs = _case_refs(root, h10, h10_protocol, task['case'])
                    if h10_refs['branches.csv']['sha256'] != entry['evaluation_case_refs']['branches.csv']['sha256']:
                        raise ValueError('History288 and stored Hstar have different physical branches')
                    entry.update(Hstar_ref=h10_worlds[task['case']]['files']['Hstar.npz'],
                                 Hstar_provenance_refs=[h10_prepared['protocol'],h10_prepared['worlds']])
                elif group == 'history10':
                    folder = directory/'fits'/task['id']; sealed = read(folder/'candidates_sealed.json')
                    world = worlds[task['case']]; days = int(task['train_days'])
                    entry.update(labels=world['labels'], native_ref=sealed['native'], train_ref=sealed['train'],
                                 validation_ref=sealed['validation'], new_day_ref=ref(folder/'new_day.npz'),
                                 train_rows=[0, 96*days], validation_rows=[0, 192], new_day_rows=[0, 96],
                                 source_row_indices=dict(train=[768-96*days, 768], validation=[768, 960], new_day=[960, 1056]),
                                 candidate_seal_ref=ref(folder/'candidates_sealed.json'), native_wrapper_key=None,
                                 source_index_ref=prepared['worlds'], physical_history_group=world['id'],
                                 Hstar_ref=world['files']['Hstar.npz'])
                else:
                    source = broad_lookup[task['id']]; request = source['request']; info = protocol['inputs'][task['job']]
                    if request['method'] != '101' or request['case'] != task['case']:
                        raise ValueError('Legacy ID/method/case mapping mismatch')
                    native_ref = dict(path=str(directory/'runs'/task['job']/'101/result.json'),
                                      sha256=seal['result_sha256'][f'runs/{task["job"]}/101/result.json'])
                    observed = dict(path=str(directory/'runtime'/info['path']), sha256=info['sha256'])
                    if native_ref != request['result'] or observed != request['observed']:
                        if checked(native_ref) != checked(request['result']) or checked(observed) != checked(request['observed']):
                            raise ValueError('Legacy inventory differs from owning seal')
                    entry.update(labels=info['labels'], native_ref=native_ref, train_ref=observed,
                                 validation_ref=observed, new_day_ref=None, train_rows=[0, 64], validation_rows=[64, 96],
                                 new_day_rows=None, source_row_indices=dict(train=[0, 64], validation=[64, 96], new_day=None),
                                 train_fingerprint=info['training_sha256'], validation_fingerprint=info['selection_sha256'],
                                 candidate_seal_ref=ref(directory/'selection_seal.json'), native_wrapper_key='result',
                                 legacy_truth_ref=ref(directory/'evaluation'/f'{task["job"]}.json'),
                                 source_index_ref=ref(root/'outputs/astra_inventory_20261114/mainline400.json'),
                                 physical_history_group=f'{task["cohort"]}|{task["case"]}|{task["replicate"]}',
                                 replicate=task['replicate'], voltage_sigma=task['voltage_sigma'])
                entry['training_points'] = entry['train_rows'][1]-entry['train_rows'][0]
                entry['validation_points'] = entry['validation_rows'][1]-entry['validation_rows'][0]
                entry['new_points'] = 0 if entry['new_day_rows'] is None else entry['new_day_rows'][1]-entry['new_day_rows'][0]
                for key in ('native_ref', 'train_ref', 'validation_ref', 'candidate_seal_ref'):
                    checked(entry[key])
            except Exception as error:
                entry.update(availability='unavailable', load_errors=[f'{type(error).__name__}: {error}'])
            entries.append(entry)
    return entries


def _failed_group_entries(root, group, error):
    """Untrusted metadata may name failed slots, never make them eligible."""
    relative, planned, _ = GROUPS[group]; tasks = []
    try:
        raw = read(Path(root)/relative/'protocol.json')
        if group.startswith('large96_'):
            cohort = group.removeprefix('large96_')
            tasks = [dict(id=f'{cohort}__{job}__101') for job in sorted(raw['inputs'])]
        else:
            tasks = raw['tasks']
    except Exception:
        pass
    if len(tasks) != planned or len({t['id'] for t in tasks}) != planned:
        tasks = [dict(id=f'unavailable_slot{i:03d}') for i in range(planned)]
    return [dict(id=group+'__'+task['id'], original_id=task['id'], source_group=group, group=group,
                 case=task.get('case'), case_key=task.get('case'), availability='unavailable',
                 load_errors=[f'{type(error).__name__}: {error}'],
                 metadata_scope='Unverified naming only; source validation failed; cannot fit') for task in tasks]


def inventory(root=ROOT):
    """Return all 456 slots, retaining cell and whole-source failures explicitly.

    This function does not open truth tables or held-out observation arrays.
    Every source must validate before load_training can make it eligible.
    """
    entries = []
    for group in GROUPS:
        try:
            rows = _inventory_group(root, group)
            if len(rows) != GROUPS[group][1]:
                raise ValueError('Original group count differs from its planned denominator')
            entries.extend(rows)
        except Exception as error:
            entries.extend(_failed_group_entries(root, group, error))
    planned = sum(v[1] for v in GROUPS.values())
    if len(entries) != planned or len({r['id'] for r in entries}) != planned:
        raise ValueError('The planned input ledger is not complete and unique')
    return entries


def slice_observation(reference, rows, labels):
    whole = load(checked(reference)); start, stop = map(int, rows)
    if tuple(labels) != whole.labels or not 0 <= start < stop <= whole.points:
        raise ValueError('Observation row range or terminal order mismatch')
    return Observation(whole.p[start:stop], whole.q[start:stop], whole.y[start:stop], whole.labels,
                       None if whole.voltage is None else whole.voltage[start:stop],
                       None if whole.root is None else whole.root[start:stop])


def load_training(entry):
    """Load only sealed native parameters and historical observed data."""
    if entry.get('availability') != 'available':
        raise ValueError('Planned input unavailable: '+str(entry.get('load_errors')))
    checked(entry['protocol_ref']); sealed = read(checked(entry['candidate_seal_ref']))
    payload = read(checked(entry['native_ref']))
    key = entry['native_wrapper_key']; native = payload[key] if key else payload
    if key:
        protocol = read(checked(entry['protocol_ref']))
        if payload['status'] != 'fitted' or payload['protocol_sha256'] != entry['protocol_ref']['sha256']:
            raise ValueError('Legacy worker is not a sealed successful fit')
        if payload['source_sha256_before'] != protocol['source_sha256'] or payload['source_sha256_after'] != protocol['source_sha256']:
            raise ValueError('Legacy worker source hashes differ')
        if payload['input_sha256'] != entry['train_ref']['sha256'] or entry['train_ref'] != entry['validation_ref']:
            raise ValueError('Legacy worker used a different observed day')
    else:
        if sealed['native']['sha256'] != entry['native_ref']['sha256']:
            raise ValueError('Candidate seal/native mismatch')
        if sealed.get('truth_used') or sealed.get('new_day_used') or sealed.get('new_day_values_used'):
            raise ValueError('Candidate seal reports use of excluded evaluation information')
        for name in ('train','validation'):
            sr = sealed['training']['ref'] if name == 'train' and 'training' in sealed else sealed[name]
            sr = sr['ref'] if 'ref' in sr else sr
            if sr['sha256'] != entry[name+'_ref']['sha256'] or checked(sr).resolve() != checked(entry[name+'_ref']).resolve():
                raise ValueError('Candidate seal/historical observation mismatch: '+name)
    if native.get('truth_used') or native.get('test_used'):
        raise ValueError('Source identifier used excluded evaluation information')
    train = slice_observation(entry['train_ref'], entry['train_rows'], entry['labels'])
    valid = slice_observation(entry['validation_ref'], entry['validation_rows'], entry['labels'])
    for name, obs in (('train', train), ('validation', valid)):
        handoff_key = 'training_sha256' if name == 'train' else 'validation_sha256'
        expected = entry.get(name+'_fingerprint', native.get('handoff',{}).get(handoff_key))
        if expected and fingerprint(obs) != expected:
            raise ValueError('Legacy historical split fingerprint mismatch: '+name)
    cfg = native['config']; rows = native['frozen_candidates']; indices = [r['index'] for r in rows]
    if not rows or len(indices) != len(set(indices)) or native['selected']['index'] not in indices:
        raise ValueError('Native candidate IDs are missing or duplicated')
    if native.get('selection_metric') != 'validation_mse':
        raise ValueError('Unsupported original selection metric: '+str(native.get('selection_metric')))
    return dict(native=deepcopy(native), train=train, validation=valid,
                reference_mse=float(native['reference_mse']), validation_tie=float(cfg['validation_tie']),
                zero_tol=float(cfg['zero_tol']), bound=float(cfg['response_bound']),
                zbase_ohm=float(entry['zbase_ohm']), expected_selected_index=native['selected']['index'],
                source_refs=[entry[k] for k in ('protocol_ref','candidate_seal_ref','native_ref','train_ref','validation_ref')])


def support_truth(branches, labels, root_bus):
    """All closed physical branches, including zero/weak impedance branches."""
    children = {}; parent = {}
    for edge in branches:
        if str(edge.get('status', 'True')).lower() not in ('true', '1'):
            continue
        a, b = int(edge['from_bus']), int(edge['to_bus'])
        if b in parent or b == root_bus:
            raise ValueError('Physical network is not a rooted directed tree')
        parent[b] = a; children.setdefault(a, []).append(b)
    label_index = {int(label): i for i,label in enumerate(labels)}; memo = {}; visiting = set()
    def descendants(node):
        if node in visiting:
            raise ValueError('Cycle in physical network')
        if node in memo:
            return memo[node]
        visiting.add(node)
        result = {label_index[node]} if node in label_index else set()
        for child in children.get(node, []): result.update(descendants(child))
        visiting.remove(node); memo[node] = tuple(sorted(result)); return memo[node]
    if descendants(root_bus) != tuple(range(len(labels))) or set(parent)-set(memo):
        raise ValueError('Physical tree is disconnected from measured terminals')
    supports = {descendants(node) for node in parent if descendants(node)}
    return [list(c) for c in sorted(supports, key=lambda c:(len(c),c))]


def load_evaluation(entry):
    """Evaluator only: call after the new floor-library/selection seal exists.

    Pressure experiments modified impedance values. Their full support geometry
    remains available, but this adapter deliberately does not report a baseline
    R/X matrix as the modified true matrix.
    """
    refs = entry['evaluation_case_refs']; meta = read(checked(refs['metadata.json']))
    with checked(refs['branches.csv']).open(encoding='utf-8', newline='') as stream:
        edges = list(csv.DictReader(stream))
    truth = support_truth(edges, entry['labels'], int(meta['root_bus']))
    source_refs = list(refs.values())
    if entry.get('legacy_truth_ref'):
        legacy = read(checked(entry['legacy_truth_ref'])); source_refs.append(entry['legacy_truth_ref'])
        proper = {tuple(c) for c in truth if 1 < len(c) < len(entry['labels'])}
        if legacy['labels'] != entry['labels'] or proper != {tuple(c) for c in legacy['proper_clades']}:
            raise ValueError('Full physical support disagrees with original internal-topology truth')
    new = None
    if entry.get('new_day_ref'):
        new = slice_observation(entry['new_day_ref'], entry['new_day_rows'], entry['labels'])
        source_refs.append(entry['new_day_ref'])
    hstar = None
    if entry.get('Hstar_ref'):
        with np.load(checked(entry['Hstar_ref']), allow_pickle=False) as z:
            hstar = np.asarray(z['Hstar'], float)
            if hstar.shape != (len(entry['labels']), 2*len(entry['labels'])):
                raise ValueError('Stored physical response dimensions differ')
            if list(map(int,z['labels'])) != entry['labels']:
                raise ValueError('Stored physical response terminal ordering differs')
            if {tuple(c) for c in json.loads(str(z['support_json']))} != {tuple(c) for c in truth}:
                raise ValueError('Stored physical response support differs from sealed branches')
        source_refs.append(entry['Hstar_ref'])
        for reference in entry.get('Hstar_provenance_refs',[]):
            checked(reference); source_refs.append(reference)
    return dict(truth_supports=truth, truth_proper_supports=[c for c in truth if 1<len(c)<len(entry['labels'])],
                new_day=new, source_refs=source_refs, Hstar=hstar, true_response_available=hstar is not None,
                true_response_scope='sealed history10 Hstar' if hstar is not None else 'unavailable; support-only evaluation',
                new_day_scope='independent saved new day' if new is not None else 'none: legacy day used only 64 train + 32 validation')
