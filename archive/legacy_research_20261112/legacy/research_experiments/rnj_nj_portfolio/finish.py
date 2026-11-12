"""Post-hoc support-coverage diagnostics and verified archive delivery."""
from __future__ import annotations
from pathlib import Path
import hashlib
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from .run import ROOT, read, write, sha
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth


def diagnose(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    frame = pd.read_csv(output / 'analysis/per_model.csv')
    catalog = read(output / 'catalog.json')
    rows = []
    topology_changes = 0
    for job in p['jobs']:
        common = read(p['references'][job]['common']['path'])
        labels = common['labels']; n = len(labels); positions = {v: i for i, v in enumerate(labels)}
        true = {tuple(sorted(positions[v] for v in C)) for C in physical_truth(Path(catalog[job]['truth_network']), labels)['clades']}
        record = read(output / 'jobs' / job / 'result.json')
        # Use terminal count, not the number of supports, when excluding the full set.
        sig = lambda point: {tuple(C) for C, r, x in zip(point['supports'], point['r'], point['x']) if 1 < len(C) < n and max(r, x) > 1e-9}
        for resolution in p['resolutions']:
            block = record['choices'][str(resolution)]
            if resolution == .24:
                topology_changes += sig(block['methods']['two_strict']['point']) != sig(block['methods']['three_strict']['point'])
            candidates = block['candidates']
            for domain, names in [('two', ('full_rnj_exit', 'small3_original_exit')),
                                  ('three', ('full_rnj_exit', 'small3_original_exit', 'NJ_matched'))]:
                supports = [sig(candidates[m]['point']) for m in names]
                union = set().union(*supports)
                any_exact = any(s == true for s in supports)
                rows.append(dict(job=job, panel=catalog[job]['panel'], resolution=resolution, domain=domain,
                                 any_tree_exact=any_exact, all_true_in_support_union=true <= union,
                                 neither_exact_but_union_covers=not any_exact and true <= union,
                                 missing_small=sum(len(C) <= 5 for C in true-union), missing_large=sum(len(C) > 5 for C in true-union)))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'analysis/union_coverage_per_input.csv', index=False)
    summary = table.groupby(['panel', 'resolution', 'domain']).sum(numeric_only=True)
    summary.to_csv(output / 'analysis/union_coverage.csv')
    q = frame[frame.resolution == .24]
    two = q[q.method == 'two_strict'].set_index('job'); three = q[q.method == 'three_strict'].set_index('job')
    check = dict(two_three_F1_changed_inputs=int(((two.clade_f1-three.clade_f1).abs() > 1e-10).sum()),
                 two_three_source_changed_inputs=int((two.selected_source != three.selected_source).sum()),
                 two_three_topology_changed_inputs=int(topology_changes),
                 truth_used_only_for_offline_diagnostics=True, actual_joint_support_rebuild_performed=False)
    write(output / 'analysis/post_diagnostics.json', check)
    dest = ROOT / 'docs/rnj_nj_portfolio_20261031'
    for name in ('union_coverage.csv', 'union_coverage_per_input.csv', 'post_diagnostics.json'):
        shutil.copy2(output / 'analysis' / name, dest / name)
    print(summary.to_string())


def package(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    audit = read(output / 'analysis/verification.json')
    assert audit['passed'] and not audit['failures'] and audit['jobs'] == 120
    assert read(output / 'state.json')['stage'] == 'complete'
    dest = ROOT / 'docs/rnj_nj_portfolio_20261031'
    previous = Path(p['parent_manifest']).parent
    for name, item in read(previous / 'manifest.json')['files'].items():
        assert sha(previous / name) == item['sha256']
    for name, item in read(previous / 'scoring_data_manifest.json')['truth_files'].items():
        assert sha(name) == item['sha256']
    for job, refs in p['references'].items():
        for name, reference in [('common', refs['common']), *refs['bases'].items()]:
            assert sha(reference['path']) == reference['sha256']
            target = output / 'parent_records' / job / (name+'.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference['path'], target)
    with ZipFile(dest / 'records.zip', 'a', ZIP_DEFLATED) as zipped:
        assert 'actual_driver/finish.py' not in zipped.namelist(), 'Rebuild the base analysis archive before packaging again.'
        for file in (output / 'parent_records').rglob('*.json'):
            zipped.write(file, file.relative_to(output))
        for name in ('union_coverage.csv', 'union_coverage_per_input.csv', 'post_diagnostics.json'):
            zipped.write(output / 'analysis' / name, f'analysis/{name}')
        for name in ('PLAN.md', 'README.md', 'CONCLUSIONS.md', 'implementation_check.json'):
            if (dest / name).exists(): zipped.write(dest / name, f'reports/{name}')
        zipped.write(Path(__file__), 'actual_driver/finish.py')
        zipped.write(ROOT / 'research_experiments/tests/test_rnj_nj_portfolio.py', 'tests/test_rnj_nj_portfolio.py')
    with ZipFile(dest / 'records.zip') as zipped:
        assert zipped.testzip() is None and len(zipped.namelist()) == len(set(zipped.namelist()))
        assert hashlib.sha256(zipped.read('actual_driver/analyze.py')).hexdigest() == audit['analyzer_sha256']
        for name, digest in p['source_hashes'].items():
            assert hashlib.sha256(zipped.read('source/'+name)).hexdigest() == digest
    shutil.copy2(output / 'state.json', dest / 'state.json')
    write(dest / 'delivery_audit.json', dict(passed=True, jobs=120, new_structural_MILP=0, parent_files_preserved=audit['preserved_parent_files'],
          archive_sha256=sha(dest / 'records.zip'), finish_sha256=sha(Path(__file__)), analyzer_sha256=audit['analyzer_sha256']))
    write(dest / 'manifest.json', dict(files={f.relative_to(dest).as_posix(): dict(bytes=f.stat().st_size, sha256=sha(f))
          for f in sorted(dest.rglob('*')) if f.is_file() and f.name != 'manifest.json'}))
    print('NJ/portfolio delivery verified')


if __name__ == '__main__':
    import sys
    (diagnose if sys.argv[1] == 'diagnose' else package)(Path(sys.argv[2]).resolve())
