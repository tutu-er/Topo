"""Long-study independent LP, deadline, topology and simplicity audit."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import sleep
import shutil
import traceback
import numpy as np
import pandas as pd
from .run import ROOT, FOUR, PORT, METHODS, read, write, sha
from .model import criterion
from research_experiments.local_rebuild.model import canonical, compatible
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile, markdown
from research_experiments.wzzt_paper.common import observed
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from rnj_wzzt.estimation import laminar_l1_milp as core

PRIOR = {}


def initialize(output):
    global PRIOR
    output = Path(output); p = read(output / 'protocol.json')
    for name, digest in p['previous_hashes'].items(): assert sha(output / 'previous' / name) == digest
    for file in ('four_profiles.json', 'portfolio_profiles.json'):
        for row in read(output / 'previous' / file):
            assert row['status'] == 0 and row['violation'] <= 1e-7
            key = row['job'], canonical(row['family'])
            if key in PRIOR: assert abs(PRIOR[key]['objective']-row['objective']) < 2e-7
            PRIOR[key] = {k: row[k] for k in ('objective', 'status', 'violation')}


def verify_job(args):
    directory, job = args; output = Path(directory); p = read(output / 'protocol.json'); meta = read(output / 'catalog.json')[job]
    assert {s: sha(meta[s]) for s in ('train', 'validation', 'test')} == p['input_hashes'][job]
    pool_file = output / 'pools' / job / 'result.json'; assert sha(pool_file) == p['pool_hashes'][job]
    pooled = read(pool_file); common_file = FOUR / 'common' / job / 'result.json'; common = read(common_file)
    labels = common['labels']; n = len(labels); positions = {v: i for i, v in enumerate(labels)}
    truth = physical_truth(Path(meta['truth_network']), labels)
    true = {tuple(sorted(positions[v] for v in C)) for C in truth['clades']}
    data = {s: prepare(observed(meta[s], meta['N'] if s == 'train' else None)) for s in ('train', 'validation', 'test')}
    rows, events, LP, oracle, sizes = [], [], [], [], []
    cache, checks = {}, 2
    def profile(family):
        family = canonical(family)
        if family not in cache:
            prior = PRIOR.get((job, family)); value = prior if prior is not None else independent_profile(data['train'], family)
            cache[family] = value; LP.append(dict(job=job, family=family, reused=prior is not None, **value))
        return cache[family]
    true_family = canonical([*true, *((i,) for i in range(n)), tuple(range(n))])
    true_loss = profile(true_family)['objective']
    sig = lambda point: {tuple(C) for C, r, x in zip(point['supports'], point['r'], point['x']) if 1 < len(C) < n and max(r, x) > 1e-9}
    info = dict(job=job, panel=meta['panel'], case=meta['case'], bundle=meta['bundle'], N=meta['N'], voltage_sigma=meta['voltage_sigma'])
    for method in METHODS:
        file = output / 'jobs' / job / method / 'result.json'
        if not file.exists():
            for resolution in p['resolutions']:
                for view in ('selected', 'endpoint', 'raw'):
                    rows.append(dict(**info, method=method, resolution=resolution, view=view, model_available=False, clade_f1=0., exact=False, source='failure'))
            events.append(dict(**info, method=method, returned=False)); continue
        r = read(file)
        assert r['protocol_sha256'] == sha(output / 'protocol.json') and not r['truth_used'] and not r['test_opened']
        assert r['input_hashes'] == {s: p['input_hashes'][job][s] for s in ('train', 'validation')}
        assert not r['LP_failures']
        checks += 3
        joint, t = r['joint'], r['trajectory']
        if method.startswith('union_'):
            assert r['pool_hash'] == p['pool_hashes'][job] and canonical(joint['pool']) == canonical(pooled['pool'])
            assert r['source_generation_recorded'] == pooled['upstream_recorded_seconds']
            assert r['penalty'] == (0. if method == 'union_plain' else .003*pooled['reference'])
        else:
            assert r['source_generation_recorded'] == 0.
            assert r['front']['tree_generators'] == (0 if method == 'pure_wzzt' else 2)
        if t:
            assert t['no_RNJ_NJ_supports'] and canonical(t['initial_supports']) == canonical([*((i,) for i in range(n)), tuple(range(n))])
            assert len(t['path']) == 1+sum(a['accepted'] for a in t['attempts'])
            for a, b in zip(t['path'], t['path'][1:]): assert b['objective'] <= a['objective']+1e-8
            trials = t['attempts']
        else:
            assert r['penalty'] >= 0.
            if method == 'simple_count':
                assert canonical(joint['pool']) == canonical(C for family in r['front']['families'] for C in family)
            assert joint['score'] <= criterion(joint['warm'], n, r['penalty'])+1e-8
            trials = [joint['event']]
        for trial in trials:
            assert trial['warm_start']['valid'] and trial['audit']['valid']
            assert trial['objective'] <= trial['warm_start']['objective']+1e-8
            if trial['bound'] is not None: assert trial['bound'] <= trial['objective']+1e-8
            if t:
                assert trial['after'] <= trial['objective']+1e-8
                if trial['bound'] is not None: assert trial['bound'] <= trial['after']+1e-8
                assert trial['accepted'] == (trial['gain'] > trial['threshold'])
            checks += 5
        if joint:
            candidate_set = set(canonical(joint['pool'])); covered = true <= candidate_set
            true_J = true_loss+r['penalty']*len(true)
            if covered and joint['event']['bound'] is not None:
                assert joint['event']['bound'] <= true_J+2e-7
            oracle.append(dict(**info, method=method, true_in_pool=covered, true_MAE=true_loss, true_J=true_J,
                predicted_J=joint['score'], predicted_MAE=r['raw_endpoint']['objective'],
                raw_wrong=sig(r['raw_endpoint']) != true,
                wrong_beats_true=sig(r['raw_endpoint']) != true and joint['score'] < true_J-2e-7,
                wrong_dominates_fit_and_count=sig(r['raw_endpoint']) != true and r['raw_endpoint']['objective'] < true_loss-2e-7 and len(sig(r['raw_endpoint'])) <= len(true),
                truth_better_but_unresolved=covered and joint['score'] > true_J+2e-7,
                pool_missing_small=sum(len(C) <= 5 for C in true-candidate_set), pool_missing_large=sum(len(C) > 5 for C in true-candidate_set)))
        else:
            provisional = {tuple(C) for C in r['raw_endpoint']['supports'] if 1 < len(C) < n}
            covered = None
        event = dict(**info, method=method, returned=True, upstream_seconds=r['source_generation_recorded'], fresh_seconds=r['fresh_seconds'],
            cumulative_seconds=r['cumulative_seconds'], structural_and_front=r['search_and_front_seconds'], post_seconds=r['post_seconds'],
            total_cap=r['declared_total_cap'], cap_exceeded=r['total_cap_exceeded'],
            MIP_calls=len(trials), gap_met=sum(e['status'] == 2 for e in trials), MIP_timeouts=sum(e['status'] == 9 for e in trials),
            max_MIP_seconds=max((e['seconds'] for e in trials), default=0.),
            last_status=trials[-1]['status'] if trials else None,
            last_relative_gap=trials[-1].get('relative_gap') if trials else None,
            pool_count=joint['internal_candidates'] if joint else None, all_true_in_pool=covered,
            pure_blocked_true=sum(any(not compatible(C, D) for D in provisional) for C in true) if t else None,
            stop_reason=t['stop_reason'] if t else 'single_joint_solve',
            raw_internal=len(sig(r['raw_endpoint'])), front_seconds=r['front']['seconds'] if r['front'] else None)
        events.append(event)
        for resolution in p['resolutions']:
            block = r['choices'][str(resolution)]
            for view, point, choice in [('selected', block['selected']['point'], block['selected']),
                                        ('endpoint', block['endpoint']['point'], block['endpoint']), ('raw', r['raw_endpoint'], {})]:
                family = canonical(point['supports']); weights = np.r_[point['r'], point['x']]
                assert core.is_laminar_family(family) and len(point['r']) == len(point['x']) == len(family)
                assert np.isfinite(weights).all() and weights.min(initial=0.) >= -1e-8 and weights.max(initial=0.) <= 2.+1e-8
                metrics = score(point, data, truth, labels, ())
                for key in ('candidate_pool_true_recall', 'novel_true', 'novel_false'): metrics.pop(key, None)
                assert abs(profile(family)['objective']-point['objective']) < 2e-7
                if choice: assert abs(metrics['validation_mae']-choice['validation']) < 1e-10
                rows.append(dict(**info, method=method, resolution=resolution, view=view,
                                 source='pure_path' if t and choice.get('index', 0) < len(t['path']) else 'star' if t else 'joint', **metrics))
                pred = sig(point)
                for size in ('small', 'large'):
                    a = {C for C in pred if (len(C) <= 5) == (size == 'small')}; b = {C for C in true if (len(C) <= 5) == (size == 'small')}
                    sizes.append(dict(**info, method=method, resolution=resolution, view=view, size=size, TP=len(a&b), FP=len(a-b), FN=len(b-a)))
                checks += 5
    return dict(rows=rows, events=events, profiles=LP, oracle=oracle, sizes=sizes, checks=checks)


def analyze(output):
    output = Path(output); p = read(output / 'protocol.json'); complete = read(output / 'completion.json')
    for name, digest in p['source_hashes'].items(): assert sha(output / 'source' / name) == digest
    preserved = 0
    for file, record in p['previous_manifests'].items():
        assert sha(file) == record['sha256']
        for name, item in record['files'].items(): assert sha(Path(file).parent / name) == item['sha256']; preserved += 1
    rows, events, LP, oracle, sizes, checks = [], [], [], [], [], 0
    with ProcessPoolExecutor(max_workers=8, initializer=initialize, initargs=(str(output),)) as executor:
        for i, result in enumerate(executor.map(verify_job, [(str(output), j) for j in p['jobs']]), 1):
            rows += result['rows']; events += result['events']; LP += result['profiles']; oracle += result['oracle']; sizes += result['sizes']; checks += result['checks']
            if i % 20 == 0: print('long union verified', i, '/120', flush=True)
    dest = output / 'analysis'; dest.mkdir(exist_ok=True)
    fresh = pd.DataFrame(rows); parent = pd.read_csv(output / 'previous/per_model.csv'); parent['view'] = 'selected'
    joined = pd.concat([parent, fresh], ignore_index=True); joined.to_csv(dest / 'per_model.csv', index=False)
    ev = pd.DataFrame(events); ev.to_csv(dest / 'events.csv', index=False)
    groups = ['panel', 'resolution', 'view', 'method']
    agg = dict(inputs=('job', 'size'), available=('model_available', 'sum'), F1=('clade_f1', 'mean'), exact=('exact', 'sum'),
               internal=('internal', 'mean'), train=('train_mae', 'mean'), validation=('validation_mae', 'mean'), test=('test_mae', 'mean'))
    joined.groupby(groups).agg(**agg).to_csv(dest / 'summary.csv')
    joined.groupby([*groups, 'case']).agg(**agg).to_csv(dest / 'cases.csv')
    joined.groupby([*groups, 'N', 'voltage_sigma']).agg(**agg).to_csv(dest / 'conditions.csv')
    pd.DataFrame(sizes).groupby([*groups, 'size'])[['TP', 'FP', 'FN']].sum().to_csv(dest / 'size_errors.csv')
    ev.groupby(['panel', 'method']).agg(inputs=('job', 'size'), returned=('returned', 'sum'), upstream=('upstream_seconds', 'median'),
        fresh=('fresh_seconds', 'median'), total=('cumulative_seconds', 'median'), post=('post_seconds', 'median'),
        calls=('MIP_calls', 'sum'), gap_met=('gap_met', 'sum'), timeouts=('MIP_timeouts', 'sum'), pool_count=('pool_count', 'mean'),
        all_true_in_pool=('all_true_in_pool', 'sum'), budget_exceeded=('cap_exceeded', 'sum')).to_csv(dest / 'timing_summary.csv')
    o = pd.DataFrame(oracle); o.to_csv(dest / 'objective_diagnostics.csv', index=False)
    o.groupby(['panel', 'method']).agg(inputs=('job', 'size'), true_in_pool=('true_in_pool', 'sum'), raw_wrong=('raw_wrong', 'sum'),
        wrong_beats_true=('wrong_beats_true', 'sum'), pareto_wrong=('wrong_dominates_fit_and_count', 'sum'),
        truth_better_unresolved=('truth_better_but_unresolved', 'sum'), missing_small=('pool_missing_small', 'sum'),
        missing_large=('pool_missing_large', 'sum')).to_csv(dest / 'objective_summary.csv')
    pairs = []
    selected = joined[joined['view'] == 'selected']
    for (panel, resolution), g in selected.groupby(['panel', 'resolution']):
        for method in METHODS:
            for reference in ('NJ_matched', 'ordinary', 'full_rnj_exit', 'small3_original_exit', 'two_strict', 'union_count'):
                if method == reference: continue
                a = g[g.method == method].set_index('job'); b = g[g.method == reference].set_index('job'); d = a.clade_f1-b.clade_f1
                pairs.append(dict(panel=panel, resolution=resolution, method=method, reference=reference, inputs=len(d), delta=float(d.mean()),
                    wins=int((d > 1e-10).sum()), ties=int((abs(d) <= 1e-10).sum()), losses=int((d < -1e-10).sum()),
                    new_exact=int((a.exact & ~b.exact).sum()), lost_exact=int((~a.exact & b.exact).sum())))
    pd.DataFrame(pairs).to_csv(dest / 'paired.csv', index=False)
    old24 = {r['job'] for r in read(FOUR / 'protocol.json')['reused_records'] if r['method'] == 'small3_original_exit'}
    joined['scope'] = ['prior_main24' if j in old24 else 'new_main84' if panel == 'main' else 'stress12' for j, panel in zip(joined.job, joined.panel)]
    joined.groupby(['scope', 'resolution', 'view', 'method']).agg(**agg).to_csv(dest / 'scopes.csv')
    write(dest / 'independent_LP.json', LP)
    audit = dict(passed=complete['returned'] == 480 and not complete['failures'], planned=480, returned=complete['returned'], checks=checks,
        profiles=len(LP), newly_solved_LP=sum(not r['reused'] for r in LP), reused_LP=sum(r['reused'] for r in LP),
        previous_files_preserved=preserved, failures=complete['failures'], cap_exceeded=int(ev.cap_exceeded.sum()), analyzer_sha256=sha(Path(__file__)))
    write(dest / 'verification.json', audit)
    delivery = ROOT / 'docs/rnj_union_long_20261031'
    for file in dest.iterdir():
        if file.suffix == '.csv' or file.name == 'verification.json': shutil.copy2(file, delivery / file.name)
    for name in ('protocol.json', 'completion.json'): shutil.copy2(output / name, delivery / name)
    summary = pd.read_csv(dest / 'summary.csv')
    text = ['# 长时限联合重构、简化模型与纯WZZT', '', '同120输入；旧方案保留，新方法总账上限14400秒，纯WZZT单步3600秒。',
        '完整配置和不同预算口径见PLAN.md；本表仅汇总，不把未完成或gap状态当作全局拓扑证明。', '',
        '## 最终输出（0.24）', '', markdown(summary[(summary.resolution == .24) & (summary['view'] == 'selected')]), '',
        '## 新方法时间与求解状态', '', markdown(pd.read_csv(dest / 'timing_summary.csv')), '',
        '## 候选覆盖与原始训练目标诊断', '', markdown(pd.read_csv(dest / 'objective_summary.csv')), '',
        '真实家族的独立LP只在评分时计算；未参与优化、候选生成或选模。复用生成器累计耗时不是新测端到端重跑。', '']
    (delivery / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    print(summary[(summary.panel == 'main') & (summary.resolution == .24) & (summary['view'] == 'selected')].to_string(index=False), flush=True)
    write(output / 'state.json', dict(stage='complete', running=False, passed=audit['passed'], returned=complete['returned'], planned=480))
    return audit


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('path'); parser.add_argument('--wait', action='store_true'); args = parser.parse_args()
    output = Path(args.path).resolve()
    try:
        if args.wait:
            while not (output / 'completion.json').exists(): sleep(2.)
        write(output / 'state.json', dict(stage='independent verification', running=True, planned=480))
        analyze(output)
    except Exception:
        write(output / 'state.json', dict(stage='verification needs attention', running=False, error=traceback.format_exc()))
        raise
