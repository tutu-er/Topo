"""Prepare fixed-scale eleven-day worlds; no identification or procurement.

Source qualification uses only dates, finite flags and valid-minute counts.
The physical response matrix is an offline diagnostic artifact, never a fit seed.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np
import pandas as pd

from ... import io
from ...observations import Observation
from ..measured_pq_20261114 import physical
from ..measured_pq_flex_20261114 import full_ac

ROOT = Path(__file__).resolve().parents[3]
CASES = ('soumalas11', 'pengwah18')
START_DATE = '2024-01-21'
QUALITY_START = '2024-01-04'
QUALITY_DATES = 31
POINTS = 11*96
HISTORY_REPEATS = 4
NOISE_SEED = 2026111601
PQ_SD = .005
VOLTAGE_SD = .0005
ROOT_PHYSICAL_SD = .003
LAWS = ('cleanlinear', 'cleanAC', 'Vonly', 'all')


def checked(ref):
    path = Path(ref['path']).resolve()
    if io.ref(path)['sha256'] != ref['sha256']:
        raise ValueError('Changed frozen input: '+str(path))
    return path


def indices():
    return dict(train_indices={'2': list(range(576, 768)), '4': list(range(384, 768)),
                               '8': list(range(768))},
                validation_indices=list(range(768, 960)),
                new_day_indices=list(range(960, POINTS)))


def wanted_times(start_date=START_DATE):
    start = pd.Timestamp(start_date+'T08:00:00', tz='Europe/Berlin')
    stop = start+pd.DateOffset(days=11)
    if stop.tz_convert('UTC')-start.tz_convert('UTC') != pd.Timedelta(days=11):
        raise ValueError('Eleven-day local window crosses DST')
    return pd.date_range(start.tz_convert('UTC'), periods=POINTS, freq='15min')


def quality_rows(times, finite, counts, feeder_ids):
    """The qualification boundary accepts flags, never numerical P/Q."""
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    finite, counts = np.asarray(finite), np.asarray(counts)
    if finite.dtype != np.bool_ or finite.shape != counts.shape:
        raise ValueError('Pass matching finite flags and valid-minute counts')
    if finite.shape != (len(times), len(feeder_ids)) or times.has_duplicates or times.isna().any():
        raise ValueError('Invalid source quality axes')
    if list(map(int, feeder_ids[:18])) != list(range(1, 19)):
        raise ValueError('Require unchanged first18 source feeder IDs')
    rows = []
    for offset in range(QUALITY_DATES):
        start = (pd.Timestamp(QUALITY_START)+pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        want = wanted_times(start); ix = times.get_indexer(want); found = ix >= 0
        cases = {}
        for case, n in zip(CASES, (11, 18)):
            bad = np.ones((POINTS, n), bool)
            bad[found] = ~finite[ix[found], :n] | (counts[ix[found], :n] < 12)
            cases[case] = dict(complete=not bool(bad.any()), incomplete_node_time_bins=int(bad.sum()),
                incomplete_time_rows=int(bad.any(axis=1).sum()),
                missing_timestamp_rows=int((~found).sum()),
                by_day=[int(bad[k*96:(k+1)*96].sum()) for k in range(11)],
                incomplete_source_feeders=[int(feeder_ids[j]) for j in np.where(bad.any(axis=0))[0]])
        rows.append(dict(start_date=start, cases=cases, both_complete=all(c['complete'] for c in cases.values())))
    return rows


def extract_profile(case, archive, context, *, start_date=START_DATE):
    """Reuse old per-node scales exactly; do not estimate from the new window."""
    mapping = context['node_mapping']; n = len(mapping)
    expected_n = dict(zip(CASES, (11, 18)))[case]
    if n != expected_n or list(map(int, archive['feeder_ids'][:n])) != [int(r['feeder_id']) for r in mapping]:
        raise ValueError('Old source-to-terminal mapping differs')
    labels = tuple(int(r['label']) for r in mapping)
    scale = np.asarray([r['scale'] for r in mapping], float)
    if len(set(labels)) != n or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid frozen terminal scales')
    times = pd.DatetimeIndex(pd.to_datetime(archive['timestamps_utc'], utc=True))
    if times.has_duplicates or times.isna().any():
        raise ValueError('Invalid archive time axis')
    want = wanted_times(start_date); ix = times.get_indexer(want)
    if np.any(ix < 0):
        raise ValueError('Selected eleven-day window is absent')
    p, q = np.asarray(archive['p_kw'][ix, :n], float), np.asarray(archive['q_kvar'][ix, :n], float)
    count = archive['valid_minutes'][ix, :n]
    if np.any(count < 12) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Incomplete selected window; no interpolation or date replacement')
    return dict(case=case, labels=labels, p_kw=p*scale, q_kvar=q*scale,
        timestamps=want.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(),
        metadata=dict(local_timezone='Europe/Berlin', start_date=start_date,
            local_start=str(want[0].tz_convert('Europe/Berlin')),
            local_stop=str((want[-1]+pd.Timedelta(minutes=15)).tz_convert('Europe/Berlin')),
            source_valid_minutes_min=int(count.min()), node_mapping=deepcopy(mapping),
            scales=scale.tolist(), scale_origin='Frozen original Jan04 first192 training P RMS',
            scales_refitted=False, extra_device_baseline_added=False))


def standard_noise(n, case, history_repeat, *, seed=NOISE_SEED):
    if case not in CASES or history_repeat not in range(HISTORY_REPEATS) or n < 1:
        raise ValueError('Invalid locked noise coordinates')
    return {name: np.random.default_rng(np.random.SeedSequence(
            [int(seed), 1610, CASES.index(case), int(history_repeat), component])).normal(
                size=(POINTS,) if name == 'root' else (POINTS, n))
            for name, component in (('P', 2), ('Q', 3), ('V', 4), ('root', 5))}


def observations(clean, noise, law='all', Hstar=None):
    """Materialize a full observed law; all histories share exactly its rows."""
    if law not in LAWS:
        raise ValueError('Unknown diagnostic observation law')
    labels = tuple(map(int, clean['labels'])); n = len(labels)
    values = {k: np.asarray(clean[k], float).copy() for k in ('P', 'Q', 'V', 'root')}
    for key, value in values.items():
        shape = (POINTS,) if key == 'root' else (POINTS, n)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid clean eleven-day '+key)
        z = np.asarray(noise[key], float)
        if z.shape != shape or not np.isfinite(z).all():
            raise ValueError('Invalid standard noise '+key)
    if law == 'cleanlinear':
        h = np.asarray(Hstar, float)
        if h.shape != (n, 2*n) or not np.isfinite(h).all():
            raise ValueError('Offline linear law requires the original physical Hstar')
        y = np.c_[values['P'], values['Q']]@h.T
        v2 = values['root'][:, None]**2-y
        if np.any(v2 <= 0):
            raise ValueError('Linear diagnostic produces nonpositive squared voltage')
        return Observation(values['P'], values['Q'], y, labels, np.sqrt(v2), values['root'])
    for key in values:
        sd = (PQ_SD if key in ('P', 'Q') else VOLTAGE_SD) if law == 'all' else (
            VOLTAGE_SD if law == 'Vonly' and key in ('V', 'root') else 0.)
        values[key] *= 1+sd*np.asarray(noise[key])
    if np.any(values['root'] <= 0) or np.any(values['V'] <= 0):
        raise ValueError('Nonpositive observed voltage')
    return Observation(values['P'], values['Q'], values['root'][:, None]**2-values['V']**2,
                       labels, values['V'], values['root'])


def slice_observations(observed, world, train_days):
    if observed.points != POINTS or str(train_days) not in ('2', '4', '8'):
        raise ValueError('Expected full eleven-day observation and declared training length')
    result = {}
    for name, ix in (('train', world['train_indices'][str(train_days)]),
                     ('validation', world['validation_indices']), ('new_day', world['new_day_indices'])):
        ix = np.asarray(ix, int)
        result[name] = Observation(observed.p[ix], observed.q[ix], observed.y[ix], observed.labels,
            None if observed.voltage is None else observed.voltage[ix],
            None if observed.root is None else observed.root[ix])
    return result


def _physical_response(network, labels):
    order, parent, incoming = physical.ac_backend._radial_order(network)
    descendants = {bus: set() for bus in order}
    for i, bus in enumerate(labels):
        descendants[bus].add(i)
    R = np.zeros((len(labels), len(labels))); X = R.copy()
    zbase = network['base_kv']**2/network['base_mva']
    for bus in reversed(order[1:]):
        z = np.asarray([i in descendants[bus] for i in range(len(labels))], float)
        R += 2*incoming[bus]['r_ohm']/zbase*np.outer(z, z)
        X += 2*incoming[bus]['x_ohm']/zbase*np.outer(z, z)
        descendants[parent[bus]].update(descendants[bus])
    return np.c_[R, X]


def generate_clean(profile):
    """Original AC equations, original network and root law, now1056 rows."""
    case = profile['case']; network = physical.ac_backend.load_case(case)
    labels = tuple(b['bus_id'] for b in network['buses'] if b['is_terminal'])
    if labels != tuple(profile['labels']):
        raise ValueError('Original network labels/order differ')
    network_hash = physical._network_hash(network)
    p, q = profile['p_kw']/network['base_kw'], profile['q_kvar']/network['base_kw']
    local = pd.DatetimeIndex(pd.to_datetime(profile['timestamps'], utc=True)).tz_convert('Europe/Berlin')
    hour = local.hour.to_numpy()+local.minute.to_numpy()/60
    rng = np.random.default_rng(np.random.SeedSequence([NOISE_SEED, 1611, CASES.index(case), 1]))
    root = 1.02+ROOT_PHYSICAL_SD*(.7*np.sin(2*np.pi*hour/24+.3)+.3*rng.normal(size=POINTS))
    voltage, original = physical.ac_backend._ac_voltages(network, labels, p, q, root,
        tolerance=1e-10, max_iterations=200)
    state, checks = full_ac.solve(network, labels, p, q, root)
    checks['max_original_solver_difference_pu'] = float(np.max(abs(voltage-state['terminal_voltage_pu'])))
    if checks['max_original_solver_difference_pu'] > 1e-12 or not np.isfinite(voltage).all() or np.any(voltage <= 0):
        raise ArithmeticError('Full AC/original AC disagreement or invalid voltage')
    if physical._network_hash(network) != network_hash:
        raise ArithmeticError('Physical network mutated')
    clean = dict(P=p, Q=q, V=voltage, root=root, labels=np.asarray(labels),
                 timestamps=np.asarray(profile['timestamps']))
    state['timestamps_utc'] = clean['timestamps']
    receipt = dict(original_network_sha256=network_hash, base_kw=network['base_kw'],
        original_AC=original, checks=checks, points=POINTS, root_physical_sigma=ROOT_PHYSICAL_SD,
        root_seed_entropy=[NOISE_SEED, 1611, CASES.index(case), 1],
        outside_095_105_cells=int(((voltage < .95) | (voltage > 1.05)).sum()),
        outside_090_110_cells=int(((voltage < .90) | (voltage > 1.10)).sum()),
        thermal_branch_limits_checked=False, voltage_violations_retained=True,
        no_extra_device_baseline=True, no_impedance_or_power_adjustment=True)
    return clean, state, _physical_response(network, labels), receipt


def _sources():
    paths = {Path(__file__).resolve(), Path(__file__).with_name('__init__.py'),
             Path(io.__file__).resolve(), Path(physical.__file__).resolve(),
             Path(full_ac.__file__).resolve(), Path(physical.ac_backend.__file__).resolve()}
    from ... import observations as obs_module
    paths.add(Path(obs_module.__file__).resolve())
    for case in CASES:
        paths.update(physical.ac_backend.ROOT/'cases'/case/name for name in ('metadata.json', 'buses.csv', 'branches.csv'))
    return sorted(paths)


def prepare_worlds(out, source_inputs_path, archive_path, protocol_ref):
    """Execute only after the external protocol is locked; preserve partial roots."""
    out = Path(out).resolve(); source_inputs_path = Path(source_inputs_path).resolve()
    archive_path = Path(archive_path).resolve()
    protocol_ref = io.ref(protocol_ref) if isinstance(protocol_ref, (str, Path)) else protocol_ref
    checked(protocol_ref)
    if out.exists():
        raise FileExistsError('Preserve completed/partial worlds; do not reset preparation: '+str(out))
    if shutil.disk_usage(out.parent).free < 2*2**30:
        raise RuntimeError('Less than2GiB free workspace space')
    archive_receipt = io.read(archive_path.with_name('archive.json'))
    checked(dict(path=str(archive_path), sha256=archive_receipt['archive']['sha256']))
    entries = io.read(source_inputs_path)
    if isinstance(entries, dict):
        entries = entries['inputs']
    chosen = []
    for case in CASES:
        found = sorted((e for e in entries if e['case'] == case and e['start_date'] == QUALITY_START), key=lambda e:e['id'])
        if not found:
            raise ValueError('Missing old Jan04 scale source: '+case)
        chosen.append(found[0])
    refs = [io.ref(source_inputs_path), io.ref(archive_path), io.ref(archive_path.with_name('archive.json')), protocol_ref]
    for old in chosen:
        for ref in (old['resource_context'], old['physical_cache']['original'], old['equipment_spec']):
            checked(ref); refs.append(ref)
    sources = [io.ref(path) for path in _sources()]
    out.mkdir(parents=True)
    for source in sources:
        path = Path(source['path']); destination = out/'source_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
    protocol = dict(status='prepared', created_utc=datetime.now(timezone.utc).isoformat(),
        external_protocol=protocol_ref, sources=sources, inputs=refs, start_date=START_DATE,
        noise_seed=NOISE_SEED, noise_entropy='[seed,1610,case_index,history_repeat,component]',
        root_entropy='[seed,1611,case_index,1]', history_repetitions=HISTORY_REPEATS,
        PQ_relative_sd=PQ_SD, V_root_relative_sd=VOLTAGE_SD, root_physical_sigma=ROOT_PHYSICAL_SD,
        laws=LAWS, diagnostic_four_day_only=True, qualification_only='finite flags/counts/time; no model or recovery filtering',
        environment={**io.environment(), 'pandas':pd.__version__}, **indices())
    io.write(out/'protocol.json', protocol)
    worlds = []; failures = []; began = perf_counter()
    with np.load(archive_path, allow_pickle=False) as archive:
        quality = quality_rows(archive['timestamps_utc'], np.isfinite(archive['p_kw']) & np.isfinite(archive['q_kvar']),
                               archive['valid_minutes'], archive['feeder_ids'])
        io.write(out/'qualification.json', dict(rows=quality, archive=io.ref(archive_path), locked_start=START_DATE,
            first_common_complete=next((r['start_date'] for r in quality if r['both_complete']), None)))
        if next((r['start_date'] for r in quality if r['both_complete']), None) != START_DATE:
            raise ValueError('Locked first complete common window no longer matches; no automatic date change')
        for old in chosen:
            case = old['case']; folder = out/'worlds'/case; folder.mkdir(parents=True)
            io.write(folder/'started.json', dict(case=case, created_utc=datetime.now(timezone.utc).isoformat()))
            try:
                context = io.read(checked(old['resource_context']))
                if context.get('start_date') != QUALITY_START:
                    raise ValueError('Frozen scales must originate in Jan04 history')
                if context.get('source_archive_sha256') != io.ref(archive_path)['sha256']:
                    raise ValueError('Frozen scale source archive differs')
                profile = extract_profile(case, archive, context)
                clean, state, Hstar, physical_receipt = generate_clean(profile)
                if physical_receipt['original_network_sha256'] != old['physical_cache']['network_hash']:
                    raise ValueError('Original physical network changed')
                np.savez_compressed(folder/'clean.npz', **clean)
                np.savez_compressed(folder/'full_ac.npz', **state)
                truth_supports = physical.ac_backend.case_truth(case, clean['labels'])
                np.savez_compressed(folder/'Hstar.npz', Hstar=Hstar, labels=clean['labels'],
                    support_json=np.asarray(json.dumps(truth_supports)))
                io.write(folder/'physical_receipt.json', physical_receipt)
                io.write(folder/'profile_metadata.json', profile['metadata'])
                rc = deepcopy(context)
                rc.update(start_date=START_DATE, local_start=profile['metadata']['local_start'],
                    scales_frozen_from=old['resource_context'], resource_registration_frozen_from_date=QUALITY_START,
                    resource_use='Old equipment retained for response-column diagnostics only; no new procurement',
                    source_archive_sha256=io.ref(archive_path)['sha256'])
                io.write(folder/'resource_context.json', rc)
                noise_refs = {}
                for h in range(HISTORY_REPEATS):
                    noise = standard_noise(len(clean['labels']), case, h)
                    path = folder/f'noise_h{h:02d}.npz'; np.savez_compressed(path, **noise)
                    noise_refs[str(h)] = io.ref(path)
                world = dict(case=case, id=case+'__'+START_DATE, labels=list(profile['labels']),
                    start_date=START_DATE, base_kw=physical_receipt['base_kw'], old_entry=deepcopy(old),
                    old_candidates_reusable=False, resource_context=io.ref(folder/'resource_context.json'),
                    equipment_spec=old['equipment_spec'], noise_by_repeat=noise_refs,
                    physical_cache=dict(original=io.ref(folder/'clean.npz'), cache=io.ref(folder/'full_ac.npz'),
                        network_hash=physical_receipt['original_network_sha256'], points=POINTS,
                        checks=physical_receipt['checks']),
                    files={name:io.ref(folder/name) for name in ('clean.npz','Hstar.npz','physical_receipt.json','profile_metadata.json')},
                    **indices())
                world['files'].update({f'noise_h{h:02d}.npz':noise_refs[str(h)] for h in range(HISTORY_REPEATS)})
                io.write(folder/'complete.json', dict(status='complete', world=world))
                worlds.append(world)
            except Exception as exc:
                failure = dict(case=case, error=repr(exc), traceback=traceback.format_exc())
                io.write(folder/'failed.json', failure); failures.append(failure)
    source_unchanged = all(io.ref(s['path'])['sha256'] == s['sha256'] for s in sources)
    inputs_unchanged = all(io.ref(s['path'])['sha256'] == s['sha256'] for s in refs)
    io.write(out/'worlds.json', worlds)
    io.write(out/'failures.json', failures)
    io.write(out/'complete.json', dict(status='complete' if not failures else 'complete_with_failures',
        planned_worlds=2, completed_worlds=len(worlds), failures=len(failures),
        worlds=io.ref(out/'worlds.json'), qualification=io.ref(out/'qualification.json'),
        source_unchanged=source_unchanged, inputs_unchanged=inputs_unchanged,
        seconds=perf_counter()-began, fits=0, procurement_quotes=0))
    if not source_unchanged or not inputs_unchanged:
        raise ArithmeticError('Preparation source/input changed during execution')
    return worlds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-inputs', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    worlds = prepare_worlds(args.out, args.source_inputs, args.archive, io.ref(args.protocol))
    print(dict(out=str(args.out), completed_worlds=len(worlds), fits=0))
