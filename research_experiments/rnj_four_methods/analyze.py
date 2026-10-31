"""Post-search scoring and independent numerical audit for all 480 records."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from .run import ROOT, METHODS, LABELS, read, write, sha
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile, markdown
from research_experiments.joint_five.analyze import size_errors
from research_experiments.local_rebuild.model import canonical, compatible
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed
from rnj_wzzt.estimation import laminar_l1_milp as core

PRIOR_FITS = {}


def initialize_profiles(output):
    global PRIOR_FITS
    output = Path(output)
    p = read(output / 'protocol.json')
    for name, digest in p['prior_audit_hashes'].items():
        assert sha(output / 'previous_audits' / name) == digest
    for name in ('long_independent_LP.json', 'base_independent_LP.json'):
        for row in read(output / 'previous_audits' / name):
            assert row['status'] == 0 and row['violation'] <= 1e-7
            key = row['job'], canonical(row['family'])
            if key in PRIOR_FITS:
                assert abs(PRIOR_FITS[key]['objective'] - row['objective']) < 2e-7
            PRIOR_FITS[key] = {k: row[k] for k in ('objective', 'violation', 'status')}


def selection_stage(record, common, choice):
    origin = choice['origin']
    if origin != 'path':
        return origin
    t = record['trajectory']
    stages = ['initial'] + [a['kind'] for a in t['attempts'] if a['accepted']]
    if record.get('exit') and record['exit']['accepted']:
        stages.append('exit')
    assert len(stages) == len(t['path']) + int(bool(record.get('exit') and record['exit']['accepted']))
    return stages[choice['index'] - len(common['bank'])]


def verify_job(args):
    output, job = args
    output = Path(output)
    p = read(output / 'protocol.json')
    meta = read(output / 'catalog.json')[job]
    common_file = output / 'common' / job / 'result.json'
    common = read(common_file)
    assert sha(common_file) == p['common_hashes'][job]
    assert {s: sha(meta[s]) for s in ('train', 'validation', 'test')} == p['input_hashes'][job]
    data = {s: prepare(observed(meta[s], meta['N'] if s == 'train' else None)) for s in ('train', 'validation', 'test')}
    labels = common['labels']
    n = len(labels)
    truth = physical_truth(Path(meta['truth_network']), labels)
    pos = {v: i for i, v in enumerate(labels)}
    true = {tuple(sorted(pos[v] for v in C)) for C in truth['clades']}
    signature = lambda point: frozenset(tuple(C) for C, r, x in zip(point['supports'], point['r'], point['x'])
                                       if 1 < len(C) < n and max(r, x) > 1e-9)
    rnj_signatures = {signature(point) for point in common['bank']}
    rows, events, sizes, profiles = [], [], [], []
    checks = 2
    cache = {}
    for method in METHODS:
        file = output / 'evaluation' / job / method / 'result.json'
        info = dict(job=job, panel=meta['panel'], case=meta['case'], method=method,
                    bundle=meta['bundle'], N=meta['N'], voltage_sigma=meta['voltage_sigma'])
        if not file.exists():
            for resolution in p['resolutions']:
                for view in ('bank', 'path', 'prefix_bank', 'prefix_path', 'endpoint', 'prefix_endpoint'):
                    rows.append(dict(**info, resolution=resolution, view=view, model_available=False,
                                     clade_f1=0., exact=False, origin='failure'))
            events.append(dict(**info, returned=False))
            continue
        record = read(file)
        provenance = read(file.with_name('provenance.json'))
        assert sha(file) == provenance['result_sha256']
        assert record['protocol_sha256'] == provenance['original_protocol_sha256']
        assert record['common_sha256'] == p['common_hashes'][job] == provenance['common_sha256']
        assert not record['truth_used'] and not record['test_opened']
        if provenance['origin'] == 'fresh':
            assert record['protocol_sha256'] == sha(output / 'protocol.json') and record['limits'] == p['limits']
        else:
            original = 'previous_long_protocol.json' if provenance['origin'] == 'previous_long' else 'previous_base_protocol.json'
            assert record['protocol_sha256'] == sha(output / original)
        checks += 5
        t = record['trajectory']
        q, event = record.get('exit'), record.get('event')
        if method == 'ordinary':
            assert not t['pre_joint'] and not t['tail_joint'] and q is None
            assert all(a['kind'] == 'wzzt_extension' for a in t['attempts'])
        if method == 'small3_original_exit':
            assert t['strategy'] == 'original' and all(a['kind'] != 'coarse_wzzt' for a in t['attempts'])
        penalty = 0. if method == 'ordinary' else .003*t['reference_MAE']
        criterion = lambda point: point['objective'] + penalty*sum(1 < len(C) < n for C in point['supports'])
        for before, after in zip(t['path'], t['path'][1:]):
            assert criterion(after) <= criterion(before) + 1e-8
            checks += 1
        if q:
            assert q['audit']['valid'] and q['score'] <= q['before_score'] + 1e-8
            available = [q['warm_score'], *[v['objective'] for v in event.get('events', []) if v['at'] <= p['limits']['total']]]
            assert q['raw_score'] <= min(available) + 1e-8
            if q['accepted']:
                assert q['incumbent_at'] <= p['limits']['total']
            if q['bound'] is not None:
                assert q['bound'] <= min(q['score'], min(criterion(v) for v in common['bank'])) + 1e-8
            checks += 5
        basic = [a for a in t['attempts'] if a['kind'] in ('wzzt_extension', 'coarse_wzzt')]
        trials = [v for a in t['attempts'] for v in (a['trials'] if a['kind'] == 'coarse_wzzt' else [a])]
        basic_trials = [v for a in basic for v in (a['trials'] if a['kind'] == 'coarse_wzzt' else [a])]
        for trial in trials:
            if trial.get('status') is not None:
                assert trial['warm_start']['valid'] and trial['bound_consistent_with_known_upper']
                checks += 2
        pool = tuple(map(tuple, event['pool'] if event else common['pool']))
        seeds = [tuple(C) for C in t.get('seeding', {}).get('seeds', [])]
        events.append(dict(**info, returned=True, reuse_origin=provenance['origin'],
                           basic_calls=len(basic), basic_MIPs=sum(v.get('status') is not None for v in basic_trials),
                           basic_accepted=sum(a['accepted'] for a in basic),
                           basic_timeouts=sum(v.get('status') == 9 for v in basic_trials),
                           pre_calls=sum(a['kind'] == 'rnj_pre_joint' for a in t['attempts']),
                           tail_calls=sum(a['kind'] in ('joint_tail', 'wzzt_tail_joint') for a in t['attempts']),
                           prefix_seconds=t['seconds'], exit_seconds=event['seconds'] if event else 0.,
                           elapsed_seconds=record['elapsed_seconds'], exit_accepted=q['accepted'] if q else False,
                           exit_status=event.get('status') if event else None, pool_count=event['pool_count'] if event else None,
                           all_true_in_exit_pool=true <= set(pool) if event else None,
                           missing_small=sum(len(C) <= 5 for C in true-set(pool)) if event else None,
                           missing_large=sum(len(C) > 5 for C in true-set(pool)) if event else None,
                           seeds=len(seeds), seed_false=sum(C not in true for C in seeds),
                           seed_cross=sum(any(not compatible(C, D) for D in true) for C in seeds), stop_reason=t['stop_reason']))
        for resolution in p['resolutions']:
            key = f'strict_{resolution:g}'
            for view, point, choice in [
                ('bank', record['choices'][key]['bank']['point'], record['choices'][key]['bank']),
                ('path', record['choices'][key]['path']['point'], record['choices'][key]['path']),
                ('prefix_bank', record['prefix_choices'][key]['bank']['point'], record['prefix_choices'][key]['bank']),
                ('prefix_path', record['prefix_choices'][key]['path']['point'], record['prefix_choices'][key]['path']),
                ('endpoint', q['point'] if q else t['point'], {}), ('prefix_endpoint', t['point'], {})]:
                family = canonical(point['supports'])
                assert core.is_laminar_family(family)
                assert len(point['r']) == len(point['x']) == len(point['supports'])
                weights = np.r_[point['r'], point['x']]
                assert np.isfinite(weights).all() and weights.min(initial=0.) >= -1e-8 and weights.max(initial=0.) <= 2.+1e-8
                metrics = score(point, data, truth, labels, pool)
                if family not in cache:
                    previous = PRIOR_FITS.get((job, family))
                    profile = previous if previous is not None else independent_profile(data['train'], family)
                    cache[family] = profile
                    profiles.append(dict(job=job, family=family, reused_independent_fit=previous is not None, **profile))
                assert abs(cache[family]['objective']-point['objective']) < 2e-7
                row = dict(**info, resolution=resolution, view=view, origin=choice.get('origin', view),
                           topology_in_common_RNJ_bank=signature(point) in rnj_signatures,
                           selected_stage=selection_stage(record, common, choice) if view == 'bank' else view, **metrics)
                rows.append(row)
                sizes.extend(dict(**info, resolution=resolution, view=view, **s) for s in size_errors(point, true, n))
                checks += 5
    return dict(rows=rows, events=events, sizes=sizes, profiles=profiles, checks=checks)


def analyze(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    complete = read(output / 'completion.json')
    for name, digest in p['source_hashes'].items():
        assert sha(output / 'source' / name) == digest
    for name, digest in p['previous_protocol_hashes'].items():
        assert sha(output / name) == digest
    # Preservation includes the earlier summaries and all three solver archives.
    previous = Path(p['previous_delivery_manifest']).parent
    assert sha(Path(p['previous_delivery_manifest'])) == p['previous_delivery_manifest_sha256']
    for name, item in p['preserved_delivery_hashes'].items():
        assert sha(previous / name) == item['sha256'], name
    rows, events, sizes, profiles, checks = [], [], [], [], 0
    with ProcessPoolExecutor(max_workers=8, initializer=initialize_profiles, initargs=(str(output),)) as executor:
        for i, result in enumerate(executor.map(verify_job, [(str(output), j) for j in p['jobs']]), 1):
            rows += result['rows']; events += result['events']; sizes += result['sizes']; profiles += result['profiles']; checks += result['checks']
            if i % 20 == 0:
                print('four-methods verified', i, '/120', flush=True)
    dest = output / 'analysis'
    dest.mkdir(exist_ok=True)
    frame, timing = pd.DataFrame(rows), pd.DataFrame(events)
    assert len(frame) == 480*12 and len(timing) == 480
    frame.to_csv(dest / 'per_model.csv', index=False)
    timing.to_csv(dest / 'events.csv', index=False)
    pd.DataFrame(sizes).to_csv(dest / 'size_errors_per_input.csv', index=False)
    groups = ['panel', 'resolution', 'view', 'method']
    aggregate = dict(inputs=('job', 'size'), available=('model_available', 'sum'), F1=('clade_f1', 'mean'), exact=('exact', 'sum'),
                     internal=('internal', 'mean'), train=('train_mae', 'mean'), validation=('validation_mae', 'mean'), test=('test_mae', 'mean'))
    frame.groupby(groups).agg(**aggregate).to_csv(dest / 'summary.csv')
    frame.groupby([*groups, 'case']).agg(**aggregate).to_csv(dest / 'cases.csv')
    frame.groupby([*groups, 'N', 'voltage_sigma']).agg(**aggregate).to_csv(dest / 'conditions.csv')
    pd.DataFrame(sizes).groupby([*groups, 'size'])[['TP', 'FP', 'FN']].sum().to_csv(dest / 'size_errors.csv')
    timing.groupby(['panel', 'method']).agg(inputs=('job', 'size'), returned=('returned', 'sum'),
        basic_inputs=('basic_MIPs', lambda s: int((s > 0).sum())), basic_MIPs=('basic_MIPs', 'sum'),
        basic_timeouts=('basic_timeouts', 'sum'), tail_inputs=('tail_calls', lambda s: int((s > 0).sum())),
        exit_gap_met=('exit_status', lambda s: int((s == 2).sum())), exit_timeouts=('exit_status', lambda s: int((s == 9).sum())),
        prefix_seconds=('prefix_seconds', 'median'), exit_seconds=('exit_seconds', 'median'), elapsed_seconds=('elapsed_seconds', 'median'),
        pool_count=('pool_count', 'mean'), all_true_in_exit_pool=('all_true_in_exit_pool', 'sum')).to_csv(dest / 'timing_summary.csv')
    pairs = []
    for (panel, resolution, view), g in frame.groupby(['panel', 'resolution', 'view']):
        for a, b in combinations(METHODS, 2):
            aa, bb = g[g.method == a].set_index('job'), g[g.method == b].set_index('job')
            delta = bb.clade_f1-aa.clade_f1
            pairs.append(dict(panel=panel, resolution=resolution, view=view, reference=a, method=b, inputs=len(delta), delta=float(delta.mean()),
                              wins=int((delta > 1e-10).sum()), ties=int((abs(delta) <= 1e-10).sum()), losses=int((delta < -1e-10).sum()),
                              new_exact=int((bb.exact & ~aa.exact).sum()), lost_exact=int((~bb.exact & aa.exact).sum())))
    pd.DataFrame(pairs).to_csv(dest / 'paired.csv', index=False)
    frame[frame['view'] == 'bank'].groupby(['panel', 'resolution', 'method', 'origin', 'selected_stage']).size().rename('inputs').to_csv(dest / 'selection_origins.csv')
    write(dest / 'independent_LP.json', profiles)
    audit = dict(passed=complete['returned'] == 480, planned=480, returned=complete['returned'], inputs=120, checks=checks,
                 independent_families=len(profiles), newly_solved_LP=sum(not r['reused_independent_fit'] for r in profiles),
                 reused_verified_LP=sum(r['reused_independent_fit'] for r in profiles), preserved_previous_files=len(p['preserved_delivery_hashes']),
                 failures=[r for r in complete['methods'] if r['failures']], analyzer_sha256=sha(Path(__file__)))
    write(dest / 'verification.json', audit)
    delivery = ROOT / 'docs/rnj_four_methods_20261031'
    for file in dest.iterdir():
        if file.suffix == '.csv' or file.name == 'verification.json':
            shutil.copy2(file, delivery / file.name)
    for name in ('protocol.json', 'completion.json'):
        shutil.copy2(output / name, delivery / name)
    summary = pd.read_csv(dest / 'summary.csv')
    lines = ['# 四方法完整长时限对照', '', '480条记录：264条经哈希核验复用，216条本轮新增；主108输入、压力12输入，四网络。',
             '普通版使用近期完整RNJ初值、原无计数惩罚的逐步WZZT；扩展版保留既有0.003 f0惩罚。共同结构预算300秒、单步MILP/LP30秒；不重新调参。',
             '', '## 最终候选库输出', '', markdown(summary[summary['view'] == 'bank']),
             '', '## 仅自身路径', '', markdown(summary[summary['view'] == 'path']),
             '', '## 调用、耗时与退出状态', '', markdown(pd.read_csv(dest / 'timing_summary.csv')),
             '', '普通版并非生产bootstrap两边界块配置；最终库仍允许选择公共完整RNJ。全部指标与分母保留；达到求解gap只证明相应候选域。详见PLAN.md和逐输入表。', '']
    (delivery / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    with ZipFile(delivery / 'records.zip', 'w', ZIP_DEFLATED) as zipped:
        for folder in ('source', 'evaluation', 'common', 'previous_audits', 'analysis'):
            for file in (output / folder).rglob('*'):
                if file.is_file() and file.suffix in ('.py', '.json', '.log'):
                    zipped.write(file, file.relative_to(output))
        for name in ('protocol.json', 'completion.json', 'catalog.json', 'previous_long_protocol.json', 'previous_base_protocol.json'):
            zipped.write(output / name, name)
        zipped.write(Path(__file__), 'actual_driver/analyze.py')
    print(summary[(summary['view'] == 'bank') & (summary.resolution == .24)].to_string(index=False), flush=True)
    return audit


if __name__ == '__main__':
    import argparse
    from time import sleep
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    output = Path(args.path).resolve()
    try:
        if args.wait:
            while not (output / 'completion.json').exists():
                sleep(2.)
        write(output / 'state.json', dict(stage='independent verification', running=True, planned=480))
        audit = analyze(output)
        write(output / 'state.json', dict(stage='complete', running=False, passed=audit['passed'],
                                         completed_records=audit['returned'], planned=480))
    except Exception:
        write(output / 'state.json', dict(stage='verification needs attention', running=False, failed=True,
                                         traceback=traceback.format_exc()))
        raise
