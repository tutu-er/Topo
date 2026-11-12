"""Package completed evidence and exact historical-source overlays; no solves."""
from __future__ import annotations
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
from .run import ROOT, PARENT, read, write, sha


def source_recovery(output):
    recovery = {}
    for kind in ('long', 'base'):
        p = read(output / f'previous_{kind}_protocol.json')
        mappings = {}
        for name, digest in p['source_hashes'].items():
            existing = output / 'source' / name
            if existing.exists() and sha(existing) == digest:
                location = existing.relative_to(output).as_posix()
            else:
                old = PARENT / f'{kind}_source' / name
                assert sha(old) == digest, (kind, name)
                dest = output / 'previous_sources' / kind / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, dest)
                location = dest.relative_to(output).as_posix()
            mappings[name] = dict(sha256=digest, archived_path=location)
        recovery[kind] = mappings
    write(output / 'source_recovery.json', recovery)
    return recovery


def package(output):
    output = Path(output)
    dest = ROOT / 'docs/rnj_four_methods_20261031'
    p = read(output / 'protocol.json')
    audit = read(output / 'analysis/verification.json')
    state = read(output / 'state.json')
    assert audit['passed'] and audit['returned'] == 480 and not audit['failures']
    assert state['stage'] == 'complete' and state['passed'] and not state['running']
    recovery = source_recovery(output)
    reused = 0
    for file in (output / 'evaluation').glob('*/*/result.json'):
        provenance = read(file.with_name('provenance.json'))
        assert sha(file) == provenance['result_sha256']
        reused += provenance['origin'] != 'fresh'
    assert reused == 264
    previous = Path(p['previous_delivery_manifest']).parent
    for name, item in p['preserved_delivery_hashes'].items():
        assert sha(previous / name) == item['sha256']
    for name in ('state.json', 'source_recovery.json'):
        shutil.copy2(output / name, dest / name)
    with ZipFile(dest / 'records.zip', 'a', ZIP_DEFLATED) as zipped:
        assert 'source_recovery.json' not in zipped.namelist(), 'Do not append twice; recreate the analysis archive first.'
        for file in (output / 'previous_sources').rglob('*.py'):
            zipped.write(file, file.relative_to(output))
        for file in (output / 'analysis').glob('*.csv'):
            zipped.write(file, file.relative_to(output))
        for name in ('PLAN.md', 'METHODS.md', 'ANALYSIS.md', 'CONCLUSIONS.md', 'README.md', 'RECOVERY.md', 'implementation_check.json'):
            if (dest / name).exists():
                zipped.write(dest / name, f'reports/{name}')
        for name in ('test_rnj_four_methods.py', 'test_rnj_small_seed.py', 'test_joint_exit.py'):
            zipped.write(ROOT / 'research_experiments/tests' / name, f'tests/{name}')
        zipped.write(output / 'source_recovery.json', 'source_recovery.json')
        zipped.write(Path(__file__), 'actual_driver/delivery.py')
        if (dest / 'report_provenance.json').exists():
            assert sha(Path(__file__).with_name('report.py')) == read(dest / 'report_provenance.json')['source_sha256']
            zipped.write(Path(__file__).with_name('report.py'), 'actual_driver/report.py')
            zipped.write(dest / 'report_provenance.json', 'analysis/report_provenance.json')
            for name in ('prior_new_scopes.csv', 'all120.csv', 'mechanism_checks.json', 'scoring_data_manifest.json'):
                zipped.write(dest / name, f'analysis/{name}')
    with ZipFile(dest / 'records.zip') as zipped:
        assert zipped.testzip() is None
        assert hashlib.sha256(zipped.read('actual_driver/analyze.py')).hexdigest() == audit['analyzer_sha256']
        for mapping in recovery.values():
            for item in mapping.values():
                assert hashlib.sha256(zipped.read(item['archived_path'])).hexdigest() == item['sha256']
        members = len(zipped.namelist())
    write(dest / 'delivery_audit.json', dict(passed=True, records=480, reused=264, fresh=216,
        preserved_previous_files=len(p['preserved_delivery_hashes']), archive_members=members,
        archive_sha256=sha(dest / 'records.zip'), analyzer_sha256=audit['analyzer_sha256'],
        note='Historical runs retain their original protocols and exact source recovery maps; no new optimizer calls in delivery.'))
    write(dest / 'manifest.json', dict(files={f.relative_to(dest).as_posix(): dict(bytes=f.stat().st_size, sha256=sha(f))
        for f in sorted(dest.rglob('*')) if f.is_file() and f.name != 'manifest.json'}))
    print('DELIVERY VERIFIED', audit['returned'], 'records;', members, 'archive members')


if __name__ == '__main__':
    import sys
    package(Path(sys.argv[1]).resolve())
