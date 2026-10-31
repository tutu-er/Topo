"""Cost ledger, reproducible archives and final manifest; no model fitting."""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import shutil
import pandas as pd
from .run import ROOT, FOUR, PORT, read, write, sha


def ledger(output):
    output = Path(output); protocol = read(output / 'protocol.json')
    current = pd.read_csv(output / 'analysis/events.csv')
    old = pd.read_csv(ROOT / 'docs/rnj_four_methods_20261031/events.csv')
    portfolio = pd.read_csv(ROOT / 'docs/rnj_nj_portfolio_20261031/timing.csv').set_index('job')
    common = {j: read(FOUR / 'common' / j / 'result.json')['seconds'] for j in protocol['jobs']}
    entries = []
    for row in old.to_dict('records'):
        entries.append(dict(job=row['job'], panel=row['panel'], method=row['method'], total_seconds=row['elapsed_seconds']+common[row['job']],
            fresh_seconds=0., budget=300., budget_scope='historical structure cap; frontend and closure extra',
            timing_kind='recorded common frontend plus original pipeline'))
    for job in protocol['jobs']:
        panel = read(output / 'catalog.json')[job]['panel']; q = portfolio.loc[job]
        for method in ('two_strict', 'two_paired', 'three_strict', 'three_paired'):
            entries.append(dict(job=job, panel=panel, method=method,
                total_seconds=q.recorded_two_pipeline_seconds+common[job]+q.marginal_seconds,
                fresh_seconds=0., budget=600., budget_scope='historical two structure paths; frontend and closure extra',
                timing_kind='recorded two pipelines plus shared frontend and portfolio overhead'))
        for method in ('NJ_native', 'NJ_matched'):
            entries.append(dict(job=job, panel=panel, method=method, total_seconds=common[job]+q.marginal_seconds,
                fresh_seconds=0., budget=None, budget_scope='no structural MILP; previous LP30s/fit',
                timing_kind='shared frontend plus jointly recorded NJ/selector cost; not standalone NJ timing'))
    for row in current.to_dict('records'):
        entries.append(dict(job=row['job'], panel=row['panel'], method=row['method'], total_seconds=row.get('cumulative_seconds'),
            fresh_seconds=row.get('fresh_seconds'), budget=14400., budget_scope='new total recorded-stage cap including charged upstream',
            timing_kind='recorded upstream plus measured new stages' if row.get('upstream_seconds', 0) else 'fresh measured algorithm stages'))
    frame = pd.DataFrame(entries)
    frame.to_csv(output / 'analysis/cost_per_input.csv', index=False)
    frame.groupby(['panel', 'method']).agg(inputs=('job', 'size'), total_median=('total_seconds', 'median'),
        total_p90=('total_seconds', lambda s: s.quantile(.9)), total_max=('total_seconds', 'max'),
        fresh_median=('fresh_seconds', 'median'), declared_cap=('budget', 'first')).to_csv(output / 'analysis/cost_summary.csv')
    dest = ROOT / 'docs/rnj_union_long_20261031'
    for name in ('cost_per_input.csv', 'cost_summary.csv'): shutil.copy2(output / 'analysis' / name, dest / name)
    print(pd.read_csv(dest / 'cost_summary.csv').query('panel == "main"').to_string(index=False))


def comparison(output):
    """One row per method/panel, including the requested all-120 denominator."""
    output = Path(output); dest = ROOT / 'docs/rnj_union_long_20261031'
    primary = pd.read_csv(output / 'analysis/per_model.csv')
    primary = primary.query('view == "selected" and resolution == .24').copy()
    primary['exploratory_after_primary'] = False
    extra = [pd.read_csv(output / 'analysis' / name).query('resolution == .24').copy()
             for name in ('bank_ablation_per_input.csv', 'validation_diagnostic_per_input.csv')]
    frames = pd.concat([primary, *extra], ignore_index=True)
    costs = pd.read_csv(output / 'analysis/cost_per_input.csv')
    bank = pd.read_csv(output / 'analysis/bank_ablation_events.csv').rename(
        columns={'total': 'total_seconds', 'fresh': 'fresh_seconds'})
    bank['method'] = 'union_bank'
    guard = costs.query('method == "union_count"').copy()
    overhead = extra[1].set_index('job')['extra_selection_seconds']
    guard['total_seconds'] += guard['job'].map(overhead)
    guard['fresh_seconds'] += guard['job'].map(overhead)
    guard['method'] = 'union_count_validated'
    costs = pd.concat([costs, bank, guard], ignore_index=True)
    frames = frames.merge(costs[['job', 'method', 'total_seconds', 'fresh_seconds']],
                          on=['job', 'method'], validate='one_to_one')
    records = []
    for scope in ('all120', 'main', 'stress'):
        part = frames if scope == 'all120' else frames[frames['panel'] == scope]
        for method, group in part.groupby('method'):
            assert len(group) == {'all120': 120, 'main': 108, 'stress': 12}[scope]
            records.append(dict(scope=scope, method=method, inputs=len(group),
                F1=group['clade_f1'].mean(), exact=int(group['exact'].sum()),
                recovery_rate=group['exact'].mean(), total_median=group['total_seconds'].median(),
                total_sum_seconds=group['total_seconds'].sum(),
                fresh_sum_seconds=group['fresh_seconds'].sum(),
                exploratory=bool(group['exploratory_after_primary'].any())))
    result = pd.DataFrame(records)
    result.to_csv(output / 'analysis/comparison.csv', index=False)
    shutil.copy2(output / 'analysis/comparison.csv', dest / 'comparison.csv')
    batch = read(output / 'analysis/batch_budget.json')
    write(output / 'analysis/budget_reconciliation.json', dict(
        primary_jobs=480, exploratory_structural_jobs=120, unique_inputs=120,
        per_job_recorded_cap_seconds=14400, nominal_caps_sum_seconds=600*14400,
        primary_fresh_stage_seconds=batch['worker_stage_seconds_sum'],
        exploratory_fresh_stage_seconds=float(bank['fresh_seconds'].sum()),
        unique_reused_generation_seconds=batch['reused_generation_seconds_once'],
        unique_recorded_components_seconds=batch['cumulative_components_once']+float(bank['fresh_seconds'].sum()),
        not_in_algorithm_timing='Python process startup, independent result audits and artifact packaging',
        note='Parallel worker walls summed, not CPU seconds or elapsed batch wall. Bank ablation upstream already present in primary reused components.'))
    shutil.copy2(output / 'analysis/budget_reconciliation.json', dest / 'budget_reconciliation.json')
    print(result.query('scope == "all120"').to_string(index=False))


def package(output):
    output = Path(output); p = read(output / 'protocol.json'); audit = read(output / 'analysis/verification.json')
    assert audit['passed'] and audit['returned'] == 480 and not audit['failures']
    assert read(output / 'state.json')['stage'] == 'complete'
    extra = read(output / 'analysis/bank_ablation_verification.json')
    assert extra['passed'] and extra['jobs'] == 120 and extra['exploratory']
    assert extra['source_sha256'] == sha(Path(__file__).with_name('bank_ablation.py'))
    for job in p['jobs']:
        receipt = read(output / 'bank_ablation/jobs' / job / 'result.json')
        assert not receipt['total_cap_exceeded'] and receipt['joint']['event']['status'] == 2
    dest = ROOT / 'docs/rnj_union_long_20261031'
    for file, record in p['previous_manifests'].items():
        assert sha(file) == record['sha256']
        for name, item in record['files'].items(): assert sha(Path(file).parent / name) == item['sha256']
    for job, refs in p['reference_hashes'].items():
        for path, digest in refs.items(): assert sha(path) == digest
    shutil.copy2(output / 'state.json', dest / 'state.json')
    archive = dest / 'records.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as zipped:
        for folder in ('source', 'pools', 'jobs', 'previous', 'analysis', 'bank_ablation'):
            for file in (output / folder).rglob('*'):
                if file.is_file() and file.suffix in ('.py', '.json', '.log', '.csv'): zipped.write(file, file.relative_to(output))
        for name in ('protocol.json', 'completion.json', 'catalog.json', 'state.json'): zipped.write(output / name, name)
        for file in Path(__file__).parent.glob('*.py'): zipped.write(file, 'actual_driver/'+file.name)
        for name in ('PLAN.md', 'MODEL.md', 'README.md', 'CONCLUSIONS.md', 'ADDENDUM.md', 'RECOVERY.md', 'implementation_check.json'):
            if (dest / name).exists(): zipped.write(dest / name, 'reports/'+name)
        zipped.write(ROOT / 'research_experiments/tests/test_rnj_union_long.py', 'tests/test_rnj_union_long.py')
    with ZipFile(archive) as zipped:
        assert zipped.testzip() is None and len(zipped.namelist()) == len(set(zipped.namelist()))
        assert hashlib.sha256(zipped.read('actual_driver/analyze.py')).hexdigest() == audit['analyzer_sha256']
        for name, digest in p['source_hashes'].items(): assert hashlib.sha256(zipped.read('source/'+name)).hexdigest() == digest
    write(dest / 'delivery_audit.json', dict(passed=True, returned=480, exploratory_ablation_returned=120,
        previous_files_preserved=audit['previous_files_preserved'],
        budget_exceeded=audit['cap_exceeded'], archive_sha256=sha(archive), analyzer_sha256=audit['analyzer_sha256'],
        note='Total costs are recorded component sums; old candidate generators were reused, not rerun. Frozen optimizer source and actual post-analysis drivers are both retained.'))
    write(dest / 'manifest.json', dict(files={f.relative_to(dest).as_posix(): dict(bytes=f.stat().st_size, sha256=sha(f))
        for f in sorted(dest.rglob('*')) if f.is_file() and f.name != 'manifest.json'}))
    print('Long union/pure-WZZT delivery verified')


if __name__ == '__main__':
    import sys
    {'ledger': ledger, 'comparison': comparison, 'package': package}[sys.argv[1]](Path(sys.argv[2]).resolve())
