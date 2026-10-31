"""Exploratory generator-deletion ablation, fixed after the primary study.

Keep the same RNJ bank and NJ tree; remove only WZZT-generated candidates.
Use the identical frozen worker/MILP and coefficient, without a parameter grid.
"""
from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import subprocess
from time import sleep, perf_counter
import pandas as pd
import numpy as np
from .run import ROOT, FOUR, PORT, read, write, sha
from research_experiments.local_rebuild.model import canonical
from research_experiments.rebuild_benchmark.engine import prepare
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile
from research_experiments.wzzt_paper.common import observed
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from rnj_wzzt.estimation import laminar_l1_milp as core


def run(output):
    output = Path(output); parent = read(output / 'protocol.json'); catalog = read(output / 'catalog.json')
    folder = output / 'bank_ablation'; protocol = folder / 'protocol.json'
    if not protocol.exists():
        hashes = {}
        for job in parent['jobs']:
            common = read(FOUR / 'common' / job / 'result.json'); nj = read(PORT / 'jobs' / job / 'result.json')
            old = read(output / 'pools' / job / 'result.json')
            pool = canonical([*common['initial'], *nj['NJ_family'], *(C for point in common['bank'] for C in point['supports'])])
            assert set(pool) <= set(canonical(old['pool']))
            file = folder / 'pools' / job / 'result.json'
            write(file, dict(pool=pool, warm=common['bank'], calibration=common['calibration'], reference=old['reference'],
                upstream_recorded_seconds=common['seconds']+nj['marginal_seconds'], no_WZZT_generator=True))
            hashes[job] = sha(file)
        write(protocol, dict(created_utc=datetime.now(timezone.utc).isoformat(), added_after_primary_results=True,
            purpose='delete only WZZT-generated supports, retaining same training RNJ bank and NJ; fixed .003 penalty',
            primary_protocol_sha256=sha(output / 'protocol.json'), jobs=parent['jobs'], pools=hashes,
            worker_source_sha256=sha(output / 'source/research_experiments/rnj_union_long/run.py'), limits=parent['limits']))
    p = read(protocol); tasks = []
    for job in p['jobs']:
        result = folder / 'jobs' / job / 'result.json'
        if result.exists() or result.with_name('error.json').exists(): continue
        meta = catalog[job]
        tasks.append(dict(job=job, method='union_bank', N=meta['N'], train=meta['train'], validation=meta['validation'],
            input_hashes={s: parent['input_hashes'][job][s] for s in ('train', 'validation')}, limits=p['limits'],
            result=str(result), protocol_sha256=sha(protocol), pool_file=str(folder / 'pools' / job / 'result.json'), pool_sha256=p['pools'][job]))
    env = os.environ.copy(); env['PYTHONPATH'] = str(output / 'source')+os.pathsep+str(output / 'source/rnj_wzzt_core')
    active, done = {}, 0
    while tasks or active:
        while tasks and len(active) < 8:
            task = tasks.pop(0); result = Path(task['result']); payload = result.with_name('payload.json'); write(payload, task)
            log = result.with_name('worker.log').open('w', encoding='utf-8')
            proc = subprocess.Popen([sys.executable, '-B', '-m', 'research_experiments.rnj_union_long.run', 'worker', str(payload)],
                cwd=output / 'source', env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active[task['job']] = proc, log, result
        for job, (proc, log, result) in list(active.items()):
            if proc.poll() is not None:
                log.close(); del active[job]; done += 1
                if not result.exists() and not result.with_name('error.json').exists(): write(result.with_name('error.json'), dict(exit_code=proc.returncode))
                if done % 20 == 0: print('bank-only ablation', done, '/120', flush=True)
        if active: sleep(.5)
    failed = [j for j in p['jobs'] if not (folder / 'jobs' / j / 'result.json').exists()]
    write(folder / 'completion.json', dict(planned=120, returned=120-len(failed), failures=failed))


def analyze(output):
    output = Path(output); folder = output / 'bank_ablation'; p = read(folder / 'protocol.json'); catalog = read(output / 'catalog.json')
    assert read(folder / 'completion.json')['returned'] == 120
    main = read(output / 'protocol.json')
    assert sha(output / 'protocol.json') == p['primary_protocol_sha256']
    assert sha(output / 'source/research_experiments/rnj_union_long/run.py') == p['worker_source_sha256']
    previous = read(output / 'analysis/independent_LP.json')
    cache = {(q['job'], canonical(q['family'])): {k: q[k] for k in ('objective', 'status', 'violation')} for q in previous}
    rows, events, profiles = [], [], []; checks = 0
    for job in p['jobs']:
        r = read(folder / 'jobs' / job / 'result.json'); pooled = read(folder / 'pools' / job / 'result.json')
        assert r['protocol_sha256'] == sha(folder / 'protocol.json') and r['pool_hash'] == p['pools'][job]
        assert sha(folder / 'pools' / job / 'result.json') == p['pools'][job]
        assert not r['truth_used'] and not r['test_opened'] and not r['LP_failures']
        assert canonical(r['joint']['pool']) == canonical(pooled['pool'])
        assert r['joint']['score'] <= r['joint']['event']['warm_start']['objective']+1e-8
        assert r['joint']['event']['audit']['valid']
        if r['joint']['event']['bound'] is not None: assert r['joint']['event']['bound'] <= r['joint']['score']+1e-8
        meta = catalog[job]; data = {s: prepare(observed(meta[s], meta['N'] if s == 'train' else None)) for s in ('train', 'validation', 'test')}
        assert {s: sha(meta[s]) for s in ('train', 'validation', 'test')} == main['input_hashes'][job]
        labels = list(observed(meta['train'], meta['N'])[0].columns); truth = physical_truth(Path(meta['truth_network']), labels)
        positions = {v: i for i, v in enumerate(labels)}; true = {tuple(sorted(positions[v] for v in C)) for C in truth['clades']}
        checks += 5
        for resolution in (.16, .24):
            point = r['choices'][str(resolution)]['selected']['point']; family = canonical(point['supports'])
            assert core.is_laminar_family(family)
            key = job, family; reused = key in cache
            if not reused: cache[key] = independent_profile(data['train'], family)
            assert abs(cache[key]['objective']-point['objective']) < 2e-7
            profiles.append(dict(job=job, family=family, reused=reused, **cache[key]))
            weights = np.r_[point['r'], point['x']]; assert weights.min(initial=0.) >= -1e-8 and weights.max(initial=0.) <= 2.+1e-8
            metric = score(point, data, truth, labels, ())
            assert abs(metric['validation_mae']-r['choices'][str(resolution)]['selected']['validation']) < 1e-10
            rows.append(dict(job=job, panel=meta['panel'], case=meta['case'], resolution=resolution, method='union_bank', exploratory_after_primary=True, **metric))
            checks += 4
        events.append(dict(job=job, panel=meta['panel'], pool_count=r['joint']['internal_candidates'], true_in_pool=true <= set(canonical(pooled['pool'])),
            upstream=r['source_generation_recorded'], fresh=r['fresh_seconds'], total=r['cumulative_seconds'], status=r['joint']['event']['status']))
    dest = output / 'analysis'; e = pd.DataFrame(events); f = pd.DataFrame(rows)
    f.to_csv(dest / 'bank_ablation_per_input.csv', index=False); e.to_csv(dest / 'bank_ablation_events.csv', index=False)
    f.groupby(['panel', 'resolution']).agg(inputs=('job', 'size'), F1=('clade_f1', 'mean'), exact=('exact', 'sum'),
        train=('train_mae', 'mean'), validation=('validation_mae', 'mean'), test=('test_mae', 'mean')).to_csv(dest / 'bank_ablation_summary.csv')
    e.groupby('panel').agg(inputs=('job', 'size'), pool=('pool_count', 'mean'), all_true_in_pool=('true_in_pool', 'sum'),
        fresh=('fresh', 'median'), total=('total', 'median'), gap_met=('status', lambda q: int((q == 2).sum()))).to_csv(dest / 'bank_ablation_cost.csv')
    write(dest / 'bank_ablation_verification.json', dict(passed=True, jobs=120, checks=checks, profiles=len(profiles),
        new_LP=sum(not q['reused'] for q in profiles), source_sha256=sha(Path(__file__)), exploratory=True))
    write(dest / 'bank_ablation_profiles.json', profiles)
    delivery = ROOT / 'docs/rnj_union_long_20261031'
    for name in ('bank_ablation_per_input.csv', 'bank_ablation_events.csv', 'bank_ablation_summary.csv', 'bank_ablation_cost.csv', 'bank_ablation_verification.json'):
        import shutil
        shutil.copy2(dest / name, delivery / name)
    print(pd.read_csv(dest / 'bank_ablation_summary.csv').to_string(index=False))
    print(pd.read_csv(dest / 'bank_ablation_cost.csv').to_string(index=False))


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve()); analyze(Path(sys.argv[1]).resolve())
