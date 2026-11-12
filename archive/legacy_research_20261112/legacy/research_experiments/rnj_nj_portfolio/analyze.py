"""Independent score/selector checks and finite-portfolio complementarity audit."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import shutil
from time import sleep
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from .run import ROOT, NEW_METHODS, BASES, read, write, sha
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile, markdown
from research_experiments.local_rebuild.model import canonical
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth, clade_metrics
from research_experiments.wzzt_paper.common import observed
from rnj_wzzt.estimation import laminar_l1_milp as core

PRIOR = {}


def initialize(output):
    global PRIOR
    output = Path(output)
    p = read(output / 'protocol.json')
    for name, digest in p['previous_hashes'].items():
        assert sha(output / 'previous' / name) == digest
    for row in read(output / 'previous/independent_LP.json'):
        assert row['status'] == 0 and row['violation'] <= 1e-7
        PRIOR[(row['job'], canonical(row['family']))] = {k: row[k] for k in ('objective', 'status', 'violation')}


def selector_reference(rows, rule):
    """Separate implementation of the declared selection inequality."""
    best_index = min(range(len(rows)), key=lambda i: (rows[i]['validation'], rows[i]['internal'], i))
    best = rows[best_index]
    eligible = []
    for i, row in enumerate(rows):
        if rule == 'strict':
            limit = 1e-12
        else:
            delta = np.asarray(row['blocks']) - np.asarray(best['blocks'])
            limit = max(.001*best['validation'], float(delta.std(ddof=1)/np.sqrt(len(delta)))) + 1e-12
        if row['validation']-best['validation'] <= limit:
            eligible.append(i)
    return min(eligible, key=lambda i: (rows[i]['internal'], rows[i]['validation'], i))


def verify_job(args):
    output, job = args
    output = Path(output)
    p = read(output / 'protocol.json')
    meta = read(output / 'catalog.json')[job]
    result_file = output / 'jobs' / job / 'result.json'
    record = read(result_file)
    assert record['protocol_sha256'] == sha(output / 'protocol.json')
    assert not record['truth_used'] and not record['test_opened'] and record['new_structural_MILP'] == 0
    assert record['reference_hashes'] == p['references'][job]
    for ref in [p['references'][job]['common'], *p['references'][job]['bases'].values()]:
        assert sha(ref['path']) == ref['sha256']
    assert {s: sha(meta[s]) for s in ('train', 'validation', 'test')} == p['input_hashes'][job]
    common = read(p['references'][job]['common']['path'])
    labels = common['labels']; n = len(labels)
    truth = physical_truth(Path(meta['truth_network']), labels)
    data = {s: prepare(observed(meta[s], meta['N'] if s == 'train' else None)) for s in ('train', 'validation', 'test')}
    rows, oracles, profiles = [], [], []
    cache, scored = {}, {}
    checks = 8
    def evaluate(point, native=False):
        nonlocal checks
        family = canonical(point['supports'])
        assert core.is_laminar_family(family)
        assert len(point['r']) == len(point['x']) == len(point['supports'])
        weights = np.r_[point['r'], point['x']]
        assert np.isfinite(weights).all() and weights.min(initial=0.) >= -1e-8 and weights.max(initial=0.) <= 2.+1e-8
        metrics = score(point, data, truth, labels, ())
        for key in ('candidate_pool_true_recall', 'novel_true', 'novel_false'):
            metrics.pop(key, None)
        if native:
            predicted = {frozenset(labels[i] for i in C) for C in family if 1 < len(C) < n}
            metrics.update(clade_metrics(predicted, truth['clades'], labels), exact=predicted == truth['clades'], internal=len(predicted))
        if family not in cache:
            previous = PRIOR.get((job, family))
            cache[family] = previous if previous is not None else independent_profile(data['train'], family)
            profiles.append(dict(job=job, family=family, reused=previous is not None, **cache[family]))
        assert abs(cache[family]['objective'] - point['objective']) < 2e-7
        checks += 5
        return metrics
    for resolution in p['resolutions']:
        block = record['choices'][str(resolution)]
        info = dict(job=job, panel=meta['panel'], case=meta['case'], bundle=meta['bundle'], N=meta['N'],
                    voltage_sigma=meta['voltage_sigma'], resolution=resolution)
        candidates = block['candidates']
        for method, candidate in candidates.items():
            metrics = evaluate(candidate['point'])
            assert abs(metrics['validation_mae']-candidate['validation']) < 1e-10
            assert abs(np.mean(candidate['blocks'])-candidate['validation']) < 1e-12
            assert len(candidate['blocks']) == 24
            scored[method] = metrics
            checks += 3
        for method, q in block['methods'].items():
            metrics = evaluate(q['point'], native=method == 'NJ_native')
            assert abs(metrics['validation_mae']-q['validation']) < 1e-10
            if method.startswith(('two_', 'three_')):
                domain = [candidates[m] for m in BASES] + ([candidates['NJ_matched']] if method.startswith('three_') else [])
                expected = selector_reference(domain, 'strict' if method.endswith('strict') else 'paired')
                assert q['index'] == expected and q['source'] == domain[expected]['source']
                assert q['point'] == domain[expected]['point']
                if method.endswith('strict'):
                    assert q['validation'] <= min(c['validation'] for c in domain) + 1e-12
                checks += 4
            rows.append(dict(**info, method=method, selected_source=q['source'], **metrics))
        for name, members in [('two', BASES), ('three', (*BASES, 'NJ_matched'))]:
            best_f1 = max(scored[m]['clade_f1'] for m in members)
            possible_exact = any(scored[m]['exact'] for m in members)
            oracles.append(dict(**info, domain=name, F1=best_f1, exact=possible_exact,
                                deployed=False, description='post-hoc finite-candidate oracle only'))
    event = dict(job=job, panel=meta['panel'], case=meta['case'], marginal_seconds=record['marginal_seconds'],
                 recorded_two_pipeline_seconds=record['recorded_two_pipeline_seconds'], fixed_LP=len(record['LP']),
                 LP_failures=len(record['LP_failures']), new_structural_MILP=0)
    assert not record['LP_failures']
    return dict(rows=rows, oracles=oracles, profiles=profiles, checks=checks, event=event)


def analyze(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    complete = read(output / 'completion.json')
    assert complete['returned'] == 120 and not complete['failures']
    for name, digest in p['source_hashes'].items():
        assert sha(output / 'source' / name) == digest
    previous = Path(p['parent_manifest']).parent
    assert sha(Path(p['parent_manifest'])) == p['parent_manifest_sha256']
    preserved = read(Path(p['parent_manifest']))['files']
    for name, item in preserved.items():
        assert sha(previous / name) == item['sha256']
    rows, oracles, profiles, events, checks = [], [], [], [], 0
    with ProcessPoolExecutor(max_workers=8, initializer=initialize, initargs=(str(output),)) as executor:
        for i, r in enumerate(executor.map(verify_job, [(str(output), j) for j in p['jobs']]), 1):
            rows += r['rows']; oracles += r['oracles']; profiles += r['profiles']; events.append(r['event']); checks += r['checks']
            if i % 20 == 0: print('NJ/portfolio verified', i, '/120', flush=True)
    dest = output / 'analysis'; dest.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    parent = pd.read_csv(output / 'previous/per_model.csv')
    parent = parent[parent['view'] == 'bank'].copy()
    parent['selected_source'] = parent['method']
    joined = pd.concat([parent, frame], ignore_index=True)
    assert len(frame) == 120*6*2 and len(joined) == 120*10*2
    joined.to_csv(dest / 'per_model.csv', index=False)
    aggregate = dict(inputs=('job', 'size'), F1=('clade_f1', 'mean'), exact=('exact', 'sum'), internal=('internal', 'mean'),
                     train=('train_mae', 'mean'), validation=('validation_mae', 'mean'), test=('test_mae', 'mean'))
    groups = ['panel', 'resolution', 'method']
    joined.groupby(groups).agg(**aggregate).to_csv(dest / 'summary.csv')
    joined.groupby([*groups, 'case']).agg(**aggregate).to_csv(dest / 'cases.csv')
    joined.groupby([*groups, 'N', 'voltage_sigma']).agg(**aggregate).to_csv(dest / 'conditions.csv')
    frame.groupby([*groups, 'selected_source']).size().rename('inputs').to_csv(dest / 'sources.csv')
    oracle = pd.DataFrame(oracles)
    oracle.to_csv(dest / 'oracle_per_input.csv', index=False)
    oracle.groupby(['panel', 'resolution', 'domain']).agg(inputs=('job', 'size'), F1=('F1', 'mean'), exact=('exact', 'sum')).to_csv(dest / 'oracle_summary.csv')
    pd.DataFrame(events).to_csv(dest / 'timing.csv', index=False)
    pd.DataFrame(events).groupby('panel').agg(inputs=('job', 'size'), marginal_seconds=('marginal_seconds', 'median'),
         recorded_two_pipeline_seconds=('recorded_two_pipeline_seconds', 'median'), new_structural_MILP=('new_structural_MILP', 'sum')).to_csv(dest / 'timing_summary.csv')
    pairs = []
    references = ['NJ_native', 'NJ_matched', 'full_rnj_exit', 'small3_original_exit']
    for (panel, resolution), g in joined.groupby(['panel', 'resolution']):
        for method in ('two_strict', 'two_paired', 'three_strict', 'three_paired', 'full_rnj_exit', 'small3_original_exit'):
            for reference in references:
                if method == reference: continue
                a = g[g.method == method].set_index('job'); b = g[g.method == reference].set_index('job')
                d = a.clade_f1-b.clade_f1
                pairs.append(dict(panel=panel, resolution=resolution, method=method, reference=reference, inputs=len(d),
                                  delta=float(d.mean()), wins=int((d > 1e-10).sum()), ties=int((abs(d) <= 1e-10).sum()),
                                  losses=int((d < -1e-10).sum()), new_exact=int((a.exact & ~b.exact).sum()), lost_exact=int((~a.exact & b.exact).sum())))
    pd.DataFrame(pairs).to_csv(dest / 'paired.csv', index=False)
    old24 = set(read(ROOT / 'outputs/rnj_four_methods_20261031/previous_base_protocol.json')['jobs']) if (ROOT / 'outputs/rnj_four_methods_20261031/previous_base_protocol.json').exists() else set()
    # The four-method package records old24 in its report; use that immutable scope list instead of selecting by outcome.
    old_scopes = pd.read_csv(previous / 'prior_new_scopes.csv')
    assert set(old_scopes.scope) == {'prior_main24', 'new_main84', 'new_stress12'}
    if not old24:
        prior_records = read(ROOT / 'outputs/rnj_four_methods_20261031/protocol.json')['reused_records']
        old24 = {r['job'] for r in prior_records if r['method'] == 'small3_original_exit'}
    joined['scope'] = ['prior_main24' if j in old24 else 'new_main84' if panel == 'main' else 'new_stress12' for j, panel in zip(joined.job, joined.panel)]
    joined.groupby(['scope', 'resolution', 'method']).agg(**aggregate).to_csv(dest / 'scopes.csv')
    write(dest / 'independent_LP.json', profiles)
    audit = dict(passed=True, jobs=120, new_structural_MILP=0, checks=checks, independent_families=len(profiles),
                 new_independent_LP=sum(not r['reused'] for r in profiles), reused_independent_LP=sum(r['reused'] for r in profiles),
                 preserved_parent_files=len(preserved), failures=[], analyzer_sha256=sha(Path(__file__)))
    write(dest / 'verification.json', audit)
    delivery = ROOT / 'docs/rnj_nj_portfolio_20261031'
    for file in dest.iterdir():
        if file.suffix == '.csv' or file.name == 'verification.json': shutil.copy2(file, delivery / file.name)
    for name in ('protocol.json', 'completion.json'): shutil.copy2(output / name, delivery / name)
    summary = pd.read_csv(dest / 'summary.csv')
    lines = ['# NJ基线与完整树验证组合', '', '同一120输入；NJ和选择新增120作业，无新结构MILP。旧四方法按原记录对照。',
             'NJ_native保留固定0.16几何拓扑；NJ_matched使用相同0.24/0.16响应收缩与L1重拟合。组合只选完整预测树；strict为主规则，paired为既有一标准误差敏感性。', '',
             '## 统一0.24口径', '', markdown(summary[summary.resolution == .24]), '',
             '## 有限候选事后互补性参照', '', markdown(pd.read_csv(dest / 'oracle_summary.csv')), '',
             '该参照使用真拓扑逐输入取最好候选，未参与部署选择，不是可实现方法或全局拓扑界。', '',
             '## 选中来源', '', markdown(pd.read_csv(dest / 'sources.csv').query('resolution == 0.24')), '',
             '## 计算成本', '', markdown(pd.read_csv(dest / 'timing_summary.csv')), '',
             'marginal_seconds为共用前端后的NJ解码、固定LP与全部选择的实测额外耗时；recorded_two_pipeline_seconds为两条原搜索流程的记录耗时之和，不是新测端到端组合时间。部署需运行两条结构路径，顺序结构上限300＋300秒，不能视为同一300秒预算。', '']
    (delivery / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    with ZipFile(delivery / 'records.zip', 'w', ZIP_DEFLATED) as z:
        for folder in ('source', 'jobs', 'previous', 'analysis'):
            for file in (output / folder).rglob('*'):
                if file.is_file() and file.suffix in ('.py', '.json', '.csv', '.log'): z.write(file, file.relative_to(output))
        for name in ('protocol.json', 'completion.json', 'catalog.json'): z.write(output / name, name)
        z.write(Path(__file__), 'actual_driver/analyze.py')
    write(output / 'state.json', dict(stage='complete', running=False, passed=True, jobs=120))
    print(summary[(summary.panel == 'main') & (summary.resolution == .24)].to_string(index=False), flush=True)
    return audit


if __name__ == '__main__':
    import argparse
    import traceback
    parser = argparse.ArgumentParser(); parser.add_argument('path'); parser.add_argument('--wait', action='store_true'); args = parser.parse_args()
    output = Path(args.path).resolve()
    try:
        if args.wait:
            while not (output / 'completion.json').exists(): sleep(2.)
        write(output / 'state.json', dict(stage='independent verification', running=True))
        analyze(output)
    except Exception:
        write(output / 'state.json', dict(stage='verification needs attention', running=False, error=traceback.format_exc()))
        raise
