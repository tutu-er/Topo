"""Verify preserved deliveries and package completed, post-freeze diagnostics.

This performs no solver calls and does not alter frozen experiment records.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'docs/rnj_small_seed_20261031'
OUTPUT = ROOT / 'outputs/rnj_small_seed_20261031'
OLD = ROOT / 'docs/joint_exit_20261031'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def package():
    old_manifest = read(OLD / 'manifest.json')
    for name, item in old_manifest['files'].items():
        file = OLD / name
        assert file.stat().st_size == item['bytes'] and sha(file) == item['sha256'], name
    assert sha(OLD / 'manifest.json') == read(DEST / 'baseline_preservation.json')['old_manifest_sha256']

    completions = {}
    for stage, expected in [('development', 64), ('evaluation', 168), ('warm', 24), ('long', 240), ('base', 24)]:
        receipt = read(OUTPUT / f'{stage}_completion.json')
        assert receipt['planned'] == receipt['returned'] == expected and not receipt['failures'], stage
        completions[stage] = receipt
    for state in ('pipeline', 'base'):
        receipt = read(OUTPUT / f'{state}_state.json')
        assert receipt == dict(stage='complete', running=False, passed=True), state
        shutil.copy2(OUTPUT / f'{state}_state.json', DEST / f'{state}_state.json')

    archives = {}
    for archive, driver, receipt_name in [
        ('records.zip', 'analyze.py', 'verification.json'),
        ('long_records.zip', 'longanalyze.py', 'long_verification.json'),
        ('base_records.zip', 'baseanalyze.py', 'base_verification.json'),
    ]:
        receipt = read(DEST / receipt_name)
        assert receipt['passed'] and not receipt['failures'], receipt_name
        with ZipFile(DEST / archive) as zipped:
            assert zipped.testzip() is None, archive
            actual = hashlib.sha256(zipped.read(f'actual_driver/{driver}')).hexdigest()
            assert actual == receipt['source_hash'], archive
            archives[archive] = dict(members=len(zipped.namelist()), sha256=sha(DEST / archive),
                                     executed_analyzer_sha256=actual)

    # This old manifest is intentionally retained as a historical checkpoint.
    checkpoint = read(DEST / 'short_manifest.json')
    changed = [name for name, item in checkpoint['files'].items() if sha(DEST / name) != item['sha256']]
    assert set(changed) <= {'PLAN.md', 'METHOD_INDEX.md', 'README.md'}, changed
    attribution = read(DEST / 'attribution_provenance.json')
    assert attribution['source_hash'] == sha(Path(__file__).with_name('attribution.py'))
    files = ['role_diagnostics.csv', 'role_summary.csv', 'solver_completion_subset.csv',
             'selection_origins.csv', 'oracle_objective.csv', 'oracle_objective_summary.csv',
             'base_paired.csv', 'selected_stage_sources.csv', 'attribution_provenance.json']
    old_output = ROOT / 'outputs/joint_exit_20261031'
    reused = ['analysis/information_run_audit.json', 'analysis/information_independent_LP.csv',
              'information_protocol.json', 'information_completion.json',
              'information_verification_protocol.json']
    with ZipFile(DEST / 'attribution_records.zip', 'w', ZIP_DEFLATED, compresslevel=6) as zipped:
        for name in files:
            zipped.write(DEST / name, f'analysis/{name}')
        for name in ('attribution.py', 'delivery.py'):
            zipped.write(Path(__file__).with_name(name), f'actual_driver/{name}')
        for name in reused:
            zipped.write(old_output / name, f'reused_oracle/{name}')
    with ZipFile(DEST / 'attribution_records.zip') as zipped:
        assert zipped.testzip() is None

    write(DEST / 'delivery_audit.json', dict(
        passed=True, preserved_old_files=len(old_manifest['files']),
        old_manifest_sha256=sha(OLD / 'manifest.json'), completions=completions, archives=archives,
        historical_short_checkpoint='249c840e', changed_guidance_since_short_checkpoint=changed,
        attribution_source_sha256=attribution['source_hash'],
        attribution_archive_sha256=sha(DEST / 'attribution_records.zip'),
        note='Only guidance files changed since the short checkpoint. Raw observations and full solver logs remain in local outputs; source/data hashes and compact records are delivered.'
    ))
    write(DEST / 'delivery_manifest.json', dict(
        scope='Final completed short, long, original-foundation and post-freeze attribution delivery',
        files={f.relative_to(DEST).as_posix(): dict(bytes=f.stat().st_size, sha256=sha(f))
               for f in sorted(DEST.rglob('*')) if f.is_file() and f.name != 'delivery_manifest.json'}
    ))
    print(json.dumps(dict(passed=True, old_files=len(old_manifest['files']),
                          archives=len(archives) + 1, files=len(read(DEST / 'delivery_manifest.json')['files']))))


if __name__ == '__main__':
    package()
