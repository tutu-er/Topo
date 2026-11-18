"""Fixed-scale Jan21 eleven-day observations; no identification or procurement.

Only Paper/Flynn require new AC preparation. Soumalas/Pengwah reuse their
hashed 1056-point physical caches. Keep primary, root-noise and uniform-time
sampling panels separate (160 + 16 + 16 observation sets).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np
import pandas as pd

from ... import io
from ...observations import Observation, fingerprint
from ..history10_rx_20261116 import data as previous
from ..measured_pq_20261114 import physical
from ..measured_pq_flex_20261114 import full_ac

ROOT = Path(__file__).resolve().parents[3]
CASES = ('paper15', 'flynn16', 'soumalas11', 'pengwah18')
COUNTS = dict(zip(CASES, (15, 16, 11, 18)))
TRAIN_LENGTHS = (48, 96, 192, 384, 768)
PQ_LEVELS = (0., .005, .01, .02)
REPEATS = (0, 1)
POINTS, TRAIN_STOP, VALIDATION_STOP = 1056, 768, 960
START_DATE, OLD_DATE = '2024-01-21', '2024-01-04'
NOISE_SEED, VOLTAGE_SD = 2026111701, .0005
ARCHIVE = ROOT/'data/measured_pq_sources_20261114/paired_pq_15min.npz'
OLD_WORLDS = ROOT/'data/history10_rx_20261116_v1/worlds.json'
CONTEXT_ROOT = ROOT/'data/distribution_panel_confirmation_data_20261115_v1/confirmation'
ARCHIVE_SHA = '45c5556c96cb792043ca40ad8e0b2df11f24efb44423c164c381f566cf010d80'
OLD_WORLDS_SHA = '2fb7f208d9c465e63ea24cd07cb5da5bf354660ae197861e44c9b9e9b39c86b1'
CONTEXT_SHA = dict(zip(CASES, (
    'dc5bb88a578b47fe61ddf175a6c8349413e1bd2903898d27661fcb3852370778',
    '0e846eba6657d27449977cdb047da5fcc7c78811a6c9703b9e2814fac2467a2d',
    '1c9f14b86a4c6a83add1be964af3347c24a3050917743eeec6c070fbfc33f9df',
    '997fd663b782b56208582be9fded7cecc78b3b3c2279810cae522ca0e0b3fffe')))


def checked(ref):
    path = Path(ref['path']).resolve()
    if io.ref(path)['sha256'] != ref['sha256']:
        raise ValueError('Changed frozen source: '+str(path))
    return path


def arrays(ref):
    with np.load(checked(ref), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def split_indices(n, sample_mode='consecutive'):
    if isinstance(n, bool) or n not in TRAIN_LENGTHS:
        raise ValueError('Undeclared training length')
    if sample_mode == 'consecutive':
        train = list(range(TRAIN_STOP-int(n), TRAIN_STOP))
    elif sample_mode == 'uniform8d':
        train = np.rint(np.linspace(0, TRAIN_STOP-1, int(n))).astype(int).tolist()
        if len(set(train)) != n:
            raise ArithmeticError('Uniform sampling produced duplicate indices')
    else:
        raise ValueError('Undeclared sampling mode')
    return dict(train=train,
                validation=list(range(TRAIN_STOP, VALIDATION_STOP)),
                new_day=list(range(VALIDATION_STOP, POINTS)))


def coordinates():
    result = []
    for case in CASES:
        for n in TRAIN_LENGTHS:
            for sd in PQ_LEVELS:
                for rep in REPEATS:
                    tag = format(sd, 'g').replace('.', 'p')
                    result.append(dict(id=f'{case}__2024-01-21__N{n:03d}__pq{tag}__h{rep:02d}',
                        case=case, start_date=START_DATE, training_rows=n, validation_rows=192,
                        new_rows=96, pq_relative_sd=sd, history_repeat=rep,
                        N=n, pq_sd=sd, repeat=rep, panel='primary', sample_mode='consecutive',
                        root_relative_sd=VOLTAGE_SD,
                        long_selected=n in (96, 768) and sd in (.005, .02) and rep == 0))
    return result


def root_coordinates():
    rows=[]
    for base in coordinates():
        if base['training_rows']==96 and base['pq_relative_sd']==.005:
            for sd,tag in ((0.,'0000'),(.001,'0010')):
                rows.append({**base, 'id':base['id']+'__root'+tag, 'panel':'root_noise',
                    'root_relative_sd':sd, 'long_selected':False, 'paired_primary_id':base['id']})
    return rows


def sampling_coordinates():
    return [{**base, 'id':base['id']+'__uniform8d', 'panel':'sampling',
             'sample_mode':'uniform8d', 'long_selected':False,
             'paired_primary_id':base['id']}
            for base in coordinates() if base['training_rows'] != 768
            and base['pq_relative_sd'] == .005 and base['history_repeat'] == 0]


def extract_profile(case, archive, context):
    """Use the Jan04 positive scale for both signed channels without refitting."""
    if case not in CASES or context.get('case') != case or context.get('start_date') != OLD_DATE:
        raise ValueError('Wrong frozen case/date context')
    mapping = context['node_mapping']; n = COUNTS[case]
    if len(mapping) != n or context.get('source_archive_sha256') != ARCHIVE_SHA:
        raise ValueError('Wrong frozen mapping/archive')
    feeder_ids = np.asarray(archive['feeder_ids'])
    if not np.array_equal(feeder_ids[:18], np.arange(1, 19)):
        raise ValueError('Expected unchanged18 distinct source feeders')
    if [int(v['feeder_id']) for v in mapping] != list(range(1, n+1)):
        raise ValueError('Changed terminal/source mapping')
    labels = tuple(int(v['label']) for v in mapping)
    scales = np.asarray([v['scale'] for v in mapping], float)
    if len(set(labels)) != n or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Invalid labels or positive common scales')
    times = pd.DatetimeIndex(pd.to_datetime(archive['timestamps_utc'], utc=True))
    if times.has_duplicates or times.isna().any() or not times.is_monotonic_increasing:
        raise ValueError('Invalid source timestamps')
    want = previous.wanted_times(START_DATE); ix = times.get_indexer(want)
    if np.any(ix < 0):
        raise ValueError('Locked1056 timestamps missing; no replacement window')
    p, q = (np.asarray(archive[key][ix, :n], float) for key in ('p_kw', 'q_kvar'))
    counts = np.asarray(archive['valid_minutes'][ix, :n])
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(counts).all() or np.any(counts < 12):
        raise ValueError('Incomplete selected window; no interpolation')
    return dict(case=case, labels=labels, p_kw=p*scales, q_kvar=q*scales,
        timestamps=want.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(),
        metadata=dict(start_date=START_DATE, local_timezone='Europe/Berlin',
            local_start=str(want[0].tz_convert('Europe/Berlin')),
            local_stop=str((want[-1]+pd.Timedelta(minutes=15)).tz_convert('Europe/Berlin')),
            node_mapping=deepcopy(mapping), scales=scales.tolist(), scale_origin=OLD_DATE,
            scales_refitted=False, source_min_valid_minutes=int(counts.min()),
            source_bins_below15=int((counts < 15).sum()), source_indices=ix.tolist(),
            missing_policy='reject; no interpolation or date replacement',
            extra_device_baseline_added=False, source_PQ_meter_error='unknown'))


def standard_noise(n, case, repeat, *, seed=NOISE_SEED):
    """Independent train/validation/evaluation streams; lengths/levels absent."""
    if case not in CASES or repeat not in REPEATS or isinstance(repeat, bool) or n < 1:
        raise ValueError('Invalid fixed noise coordinates')
    return {key: np.concatenate([np.random.default_rng(np.random.SeedSequence(
        [int(seed), 1717, CASES.index(case), int(repeat), split, component])).normal(
            size=(length,) if key == 'root' else (length, n))
        for split, length in enumerate((768, 192, 96))], axis=0)
        for key, component in (('P', 2), ('Q', 3), ('V', 4), ('root', 5))}


def observe(clean, noise, pq_sd, root_sd=VOLTAGE_SD):
    if pq_sd not in PQ_LEVELS:
        raise ValueError('Undeclared P/Q noise standard deviation')
    if root_sd not in (0., VOLTAGE_SD, .001):
        raise ValueError('Undeclared root-meter standard deviation')
    labels = tuple(map(int, clean['labels'])); values = {}
    for key in ('P', 'Q', 'V', 'root'):
        value, z = np.asarray(clean[key], float), np.asarray(noise[key], float)
        shape = (POINTS,) if key == 'root' else (POINTS, len(labels))
        if value.shape != shape or z.shape != shape or not np.isfinite(value).all() or not np.isfinite(z).all():
            raise ValueError('Invalid physical/noise '+key)
        sd = pq_sd if key in ('P','Q') else root_sd if key=='root' else VOLTAGE_SD
        values[key] = value*(1+sd*z)
    if np.any(values['V'] <= 0) or np.any(values['root'] <= 0):
        raise ValueError('Nonpositive observed voltage')
    return Observation(values['P'], values['Q'], values['root'][:, None]**2-values['V']**2,
                       labels, values['V'], values['root'])


def _save_observation(path, obs, ix):
    ix = np.asarray(ix, int)
    sliced = Observation(obs.p[ix], obs.q[ix], obs.y[ix], obs.labels)
    np.savez_compressed(path, P=sliced.p, Q=sliced.q, Y=sliced.y, labels=np.asarray(sliced.labels))
    return dict(ref=io.ref(path), fingerprint=fingerprint(sliced), rows=len(ix))


def _check_clean(profile, clean, state, base):
    if tuple(map(int, clean['labels'])) != profile['labels']:
        raise ValueError('Cached terminal order differs')
    for key, target in (('P', profile['p_kw']/base), ('Q', profile['q_kvar']/base),
                        ('timestamps', np.asarray(profile['timestamps']))):
        if not np.array_equal(clean[key], target):
            raise ValueError('Cached '+key+' differs from fixed source/scales/window')
    if not np.array_equal(clean['V'], state['terminal_voltage_pu']) or not np.array_equal(clean['root'], state['root_voltage_pu']):
        raise ValueError('Cached full AC differs from clean physical channels')
    if not all(np.isfinite(clean[k]).all() for k in ('P', 'Q', 'V', 'root')):
        raise ValueError('Nonfinite cached physical channels')
    if np.any(clean['V'] <= 0) or np.any(clean['root'] <= 0):
        raise ValueError('Nonpositive cached voltage')


def _generate(profile, network):
    """Only new Paper/Flynn caches; unchanged original AC and root law."""
    case = profile['case']
    if case not in ('paper15', 'flynn16'):
        raise ValueError('Sou/Peng must reuse the sealed physical world')
    labels = profile['labels']; base = network['base_kw']
    p, q = profile['p_kw']/base, profile['q_kvar']/base
    clock = pd.DatetimeIndex(pd.to_datetime(profile['timestamps'], utc=True)).tz_convert('Europe/Berlin')
    hour = clock.hour.to_numpy()+clock.minute.to_numpy()/60
    entropy = [previous.NOISE_SEED, 1611, {'paper15':2, 'flynn16':3}[case], 1]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    root = 1.02+.003*(.7*np.sin(2*np.pi*hour/24+.3)+.3*rng.normal(size=POINTS))
    v, original = physical.ac_backend._ac_voltages(network, labels, p, q, root,
                                                 tolerance=1e-10, max_iterations=200)
    state, checks = full_ac.solve(network, labels, p, q, root)
    difference = float(np.max(abs(v-state['terminal_voltage_pu'])))
    if difference > 1e-12:
        raise ArithmeticError('Original and full AC disagree')
    checks['max_original_solver_difference_pu'] = difference
    clean = dict(P=p, Q=q, V=v, root=root, labels=np.asarray(labels),
                 timestamps=np.asarray(profile['timestamps']))
    state['timestamps_utc'] = clean['timestamps']
    _check_clean(profile, clean, state, base)
    receipt = dict(original_network_sha256=physical._network_hash(network),
        base_kw=base, power_base_kva=base, points=POINTS, original_AC=original, checks=checks,
        root_physical_sigma=.003, root_seed_entropy=entropy,
        outside_095_105_cells=int(((v < .95) | (v > 1.05)).sum()),
        outside_090_110_cells=int(((v < .90) | (v > 1.10)).sum()),
        thermal_branch_limits_checked=False, voltage_violations_retained=True,
        no_extra_device_baseline=True, no_impedance_or_power_adjustment=True)
    return clean, state, previous._physical_response(network, labels), receipt


def _source_paths():
    from ... import observations as obs_module
    paths = {Path(__file__), Path(previous.__file__), Path(io.__file__), Path(obs_module.__file__),
             Path(physical.__file__), Path(full_ac.__file__), Path(physical.ac_backend.__file__)}
    for path in tuple(paths):
        if path.with_name('__init__.py').exists():
            paths.add(path.with_name('__init__.py'))
    for case in CASES:
        paths.update(physical.ac_backend.ROOT/'cases'/case/name
                     for name in ('metadata.json', 'buses.csv', 'branches.csv'))
    return sorted(path.resolve() for path in paths)


def prepare(out, *, archive=ARCHIVE, old_worlds=OLD_WORLDS,
            context_root=CONTEXT_ROOT, protocol=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve prior/partial preparation: '+str(out))
    archive_ref = dict(path=str(Path(archive).resolve()), sha256=ARCHIVE_SHA)
    old_ref = dict(path=str(Path(old_worlds).resolve()), sha256=OLD_WORLDS_SHA)
    checked(archive_ref); old = {w['case']:w for w in io.read(checked(old_ref))}
    archive_manifest = io.ref(Path(archive).with_name('archive.json'))
    if io.read(checked(archive_manifest))['archive']['sha256'] != ARCHIVE_SHA:
        raise ValueError('Archive manifest differs')
    contexts = {case:dict(path=str(Path(context_root)/f'{case}__{OLD_DATE}'/'resource_context.json'),
                         sha256=CONTEXT_SHA[case]) for case in CASES}
    for ref in contexts.values(): checked(ref)
    inputs = [archive_ref, old_ref, archive_manifest, *contexts.values()]
    if protocol is not None:
        external = io.ref(protocol); checked(external); inputs.append(external)
    else:
        external = None
    for case in ('soumalas11', 'pengwah18'):
        w = old[case]
        for ref in [w['physical_cache']['original'], w['physical_cache']['cache'],
                    *[w['files'][k] for k in ('Hstar.npz', 'physical_receipt.json', 'profile_metadata.json')]]:
            checked(ref); inputs.append(ref)
    sources = [io.ref(p) for p in _source_paths()]
    out.mkdir(parents=True)
    for source in sources:
        path = Path(source['path']); destination = out/'source_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
    io.write(out/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        external_protocol=external, sources=sources, inputs=inputs, planned_worlds=4,
        planned_grid_inputs=160, planned_long_inputs=16, planned_root_inputs=16,
        planned_sampling_inputs=16, planned_observations=192, cases=CASES, training_rows=TRAIN_LENGTHS,
        PQ_relative_sd=PQ_LEVELS, V_root_relative_sd=VOLTAGE_SD, repeats=REPEATS,
        start_date=START_DATE, common_validation=[768,960], common_evaluation=[960,1056],
        noise_seed=NOISE_SEED, noise_entropy='[seed,1717,case_index,repeat,split,component]',
        noise_splits={'0':[0,768], '1':[768,960], '2':[960,1056]},
        root_meter_panel=dict(training_rows=96, PQ_relative_sd=.005,
            root_relative_sd=[0.,.001], primary_midpoint_reused=.0005, terminal_relative_sd=.0005),
        sampling_panel=dict(training_rows=[48,96,192,384], PQ_relative_sd=.005, repeat=0,
            index_rule='np.rint(np.linspace(0,767,N)).astype(int)',
            sample_mode='uniform8d', full768_reused_from_primary=True),
        physical_cache_policy='reuse Sou/Peng exactly; original AC for Paper/Flynn only',
        scale_policy='frozen Jan04 scales; no refit, no device baseline, signed measured P/Q',
        source_meter_error='unknown; sd=0 means no additional P/Q error',
        environment={**io.environment(), 'pandas':pd.__version__}))
    began = perf_counter(); worlds = []; failures = []; grid = []; quality = []
    with np.load(checked(archive_ref), allow_pickle=False) as source:
        for case in CASES:
            folder = out/'worlds'/case; folder.mkdir(parents=True)
            io.write(folder/'started.json', dict(case=case, started_utc=datetime.now(timezone.utc).isoformat()))
            try:
                context = io.read(checked(contexts[case])); profile = extract_profile(case, source, context)
                quality.append(dict(case=case, status='complete', **profile['metadata']))
                network = physical.ac_backend.load_case(case); network_hash = physical._network_hash(network)
                labels = tuple(b['bus_id'] for b in network['buses'] if b['is_terminal'])
                if labels != profile['labels']:
                    raise ValueError('Original network terminal order changed')
                assets = {name:io.ref(physical.ac_backend.ROOT/'cases'/case/name)
                          for name in ('metadata.json', 'buses.csv', 'branches.csv')}
                if case in ('soumalas11', 'pengwah18'):
                    prior = old[case]; clean = arrays(prior['physical_cache']['original'])
                    state = arrays(prior['physical_cache']['cache'])
                    _check_clean(profile, clean, state, network['base_kw'])
                    if prior['physical_cache']['network_hash'] != network_hash:
                        raise ValueError('Cached physical network differs')
                    physical_receipt = io.read(checked(prior['files']['physical_receipt.json']))
                    physical_refs = dict(clean=prior['physical_cache']['original'],
                        full_ac=prior['physical_cache']['cache'], Hstar=prior['files']['Hstar.npz'],
                        receipt=prior['files']['physical_receipt.json'])
                    physical_origin = 'reused_frozen_history10'
                else:
                    clean, state, hstar, physical_receipt = _generate(profile, network)
                    np.savez_compressed(folder/'clean.npz', **clean)
                    np.savez_compressed(folder/'full_ac.npz', **state)
                    np.savez_compressed(folder/'Hstar.npz', Hstar=hstar, labels=np.asarray(labels))
                    io.write(folder/'physical_receipt.json', physical_receipt)
                    physical_refs = {key:io.ref(folder/name) for key,name in
                        (('clean','clean.npz'),('full_ac','full_ac.npz'),('Hstar','Hstar.npz'),('receipt','physical_receipt.json'))}
                    physical_origin = 'new_original_AC_cache'
                if physical._network_hash(network) != network_hash:
                    raise ArithmeticError('Network mutated')
                io.write(folder/'profile_metadata.json', profile['metadata'])
                noise_refs = {}
                for rep in REPEATS:
                    path = folder/f'noise_h{rep:02d}.npz'
                    np.savez_compressed(path, **standard_noise(len(labels), case, rep))
                    noise_refs[str(rep)] = io.ref(path)
                world = dict(id=case+'__'+START_DATE, case=case, start_date=START_DATE,
                    labels=list(labels), base_kw=network['base_kw'], power_base_kva=network['base_kw'],
                    zbase_ohm=network['base_kv']**2/network['base_mva'], original_context=contexts[case],
                    resource_context=contexts[case], profile_metadata=io.ref(folder/'profile_metadata.json'),
                    physical_origin=physical_origin, physical=physical_refs, noise_by_repeat=noise_refs,
                    network_hash=network_hash, evaluation_case_refs=assets,
                    physical_checks=physical_receipt['checks'], meter_relative_sd=VOLTAGE_SD,
                    physical_root_sigma=.003, observation_source_has_unknown_meter_error=True)
                io.write(folder/'world.json', world); world_ref=io.ref(folder/'world.json'); worlds.append(world)
                for rep in REPEATS:
                    z = arrays(noise_refs[str(rep)])
                    observed_cache={}
                    for coordinate in (c for c in coordinates()+root_coordinates()+sampling_coordinates()
                                       if c['case']==case and c['history_repeat']==rep):
                        sd,root_sd=coordinate['pq_relative_sd'],coordinate['root_relative_sd']
                        key=(sd,root_sd)
                        if key not in observed_cache:
                            observed_cache[key]=observe(clean,z,sd,root_sd)
                        obs=observed_cache[key]
                        dest=out/'inputs'/coordinate['id']; dest.mkdir(parents=True)
                        ix=split_indices(coordinate['training_rows'],coordinate['sample_mode'])
                        files={}; fingerprints={}
                        for name, rows in ix.items():
                            detail=_save_observation(dest/(name+'.npz'), obs, rows)
                            files[name+'.npz']=detail['ref']; fingerprints[name]=detail['fingerprint']
                        lineage=dict(source_world=world_ref, standard_noise=noise_refs[str(rep)],
                            PQ_relative_sd=sd, V_relative_sd=VOLTAGE_SD, root_relative_sd=root_sd,
                            sample_mode=coordinate['sample_mode'],
                            sampling={name:dict(indices=rows, timestamps=[profile['timestamps'][t] for t in rows])
                                      for name,rows in ix.items()}, fingerprints=fingerprints,
                            fitter_columns=['P','Q','Y','labels'], no_true_parameters_in_fit_npz=True)
                        io.write(dest/'observations_sealed.json', lineage)
                        evaluation=dict(world=world_ref, physical=physical_refs, network_hash=network_hash,
                            evaluation_case_refs=assets, Hstar_ref=physical_refs['Hstar'],
                            evaluation_indices=ix['new_day'], evaluation_observation=files['new_day.npz'],
                            scope='Evaluator only; open after candidate fitting and sealing')
                        io.write(dest/'evaluation.json', evaluation)
                        row={**coordinate, 'status':'complete', 'labels':list(labels),
                            'base_kw':network['base_kw'], 'power_base_kva':network['base_kw'],
                            'zbase_ohm':world['zbase_ohm'], 'files':files, 'fingerprints':fingerprints,
                            'source_world':world_ref, 'world_ref':world_ref,
                            'Hstar_ref':physical_refs['Hstar'], 'standard_noise':noise_refs[str(rep)],
                            'observations_sealed':io.ref(dest/'observations_sealed.json'),
                            'evaluation_ref':io.ref(dest/'evaluation.json'), 'meter_relative_sd':VOLTAGE_SD}
                        io.write(dest/'complete.json', row); grid.append(row)
                io.write(folder/'complete.json', dict(status='complete', world=world_ref))
            except Exception as exc:
                failure=dict(case=case, error=repr(exc), traceback=traceback.format_exc())
                io.write(folder/'failed.json', failure); failures.append(failure)
    completed={row['id']:row for row in grid}
    ledger=[completed.get(c['id'], {**c, 'status':'failed',
        'reason':'world/input preparation failed; see failures.json'})
        for c in coordinates()+root_coordinates()+sampling_coordinates()]
    primary=[r for r in ledger if r['panel']=='primary'];root_panel=[r for r in ledger if r['panel']=='root_noise']
    sampling_panel=[r for r in ledger if r['panel']=='sampling']
    unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in sources)
    inputs_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in inputs)
    io.write(out/'worlds.json', worlds); io.write(out/'inputs.json', primary)
    io.write(out/'root_inputs.json', root_panel);io.write(out/'attempts.json',ledger)
    io.write(out/'sampling_inputs.json', sampling_panel)
    io.write(out/'long_inputs.json', [r for r in primary if r['long_selected']])
    io.write(out/'qualification.json', dict(locked_start=START_DATE, rows=quality,
        selection_rule='Previously locked Jan21 common complete source window; no outcome filtering'))
    io.write(out/'failures.json', failures)
    result=dict(status='complete' if not failures and len(grid)==192 else 'complete_with_failures',
        protocol=io.ref(out/'protocol.json'), worlds=io.ref(out/'worlds.json'), inputs=io.ref(out/'inputs.json'),
        long_inputs=io.ref(out/'long_inputs.json'), root_inputs=io.ref(out/'root_inputs.json'),
        sampling_inputs=io.ref(out/'sampling_inputs.json'),
        attempts=io.ref(out/'attempts.json'), qualification=io.ref(out/'qualification.json'),
        failures=io.ref(out/'failures.json'), planned_worlds=4, completed_worlds=len(worlds),
        planned_inputs=192, completed_inputs=len(grid), failed_inputs=192-len(grid),
        planned_primary_inputs=160, completed_primary_inputs=sum(r['status']=='complete' for r in primary),
        failed_primary_inputs=sum(r['status']!='complete' for r in primary),
        planned_root_inputs=16, completed_root_inputs=sum(r['status']=='complete' for r in root_panel),
        failed_root_inputs=sum(r['status']!='complete' for r in root_panel),
        planned_sampling_inputs=16, completed_sampling_inputs=sum(r['status']=='complete' for r in sampling_panel),
        failed_sampling_inputs=sum(r['status']!='complete' for r in sampling_panel),
        planned_long_inputs=16, source_unchanged=unchanged, inputs_unchanged=inputs_unchanged,
        seconds=perf_counter()-began, identification_fits=0, procurement_quotes=0)
    io.write(out/'complete.json', result)
    if not unchanged or not inputs_unchanged:
        raise ArithmeticError('Source/input changed during data preparation')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command', required=True)
    p=sub.add_parser('prepare'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--archive', type=Path, default=ARCHIVE)
    p.add_argument('--old-worlds', type=Path, default=OLD_WORLDS)
    p.add_argument('--context-root', type=Path, default=CONTEXT_ROOT)
    p.add_argument('--protocol', type=Path)
    args=parser.parse_args()
    result=prepare(args.out, archive=args.archive, old_worlds=args.old_worlds,
                   context_root=args.context_root, protocol=args.protocol)
    print({k:result[k] for k in ('status','completed_worlds','completed_inputs','failed_inputs','seconds')})


if __name__ == '__main__': main()
