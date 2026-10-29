"""Export completed synthetic DR evidence without running any experiment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export(run_root, repo_root, output, postprocess_root):
    run_root, repo_root, output, postprocess_root = map(Path, (run_root, repo_root, output, postprocess_root))
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run_root / 'protocol.json').read_text(encoding='utf-8'))
    audit = json.loads((run_root / 'audit_final_corrected.json').read_text(encoding='utf-8'))
    assert audit['all_passed'] and audit['recorded_cells'] == 713 and audit['unique_fits'] == 98
    receipt_files = ['protocol.json', 'planning.json', 'policies.json', 'test_draws.json',
                     'model_self_check.json', 'summary.json', 'analysis.json',
                     'audit_first8.json', 'audit_final.json', 'audit_final_corrected.json',
                     'mechanism_diagnosis.json']
    for name in receipt_files:
        shutil.copyfile(run_root / name, output / name)
    entries = {}
    for category, pattern in [('record', 'records/*.json'), ('input', 'inputs/*.json'),
                              ('fit_result', 'fits/*/result.json'), ('checkpoint', 'fits/*/checkpoint.json')]:
        for path in sorted(run_root.glob(pattern)):
            entries['run/' + path.relative_to(run_root).as_posix()] = (path, category)
    for original, expected in protocol['source_sha256'].items():
        path = Path(original)
        assert digest(path.read_bytes()) == expected, str(path)
        entries['frozen_source/' + path.relative_to(repo_root).as_posix()] = (path, 'frozen_source')
    helper = Path(audit['supplemental_evaluation_helper']['path'])
    assert digest(helper.read_bytes()) == audit['supplemental_evaluation_helper']['current_sha256']
    entries['frozen_source/' + helper.relative_to(repo_root).as_posix()] = (helper, 'supplemental_helper')
    for name in ['analyze_results.py', 'mechanism_diagnosis.py', 'audit.py', 'export_evidence.py']:
        entries['postprocess_source/' + name] = (postprocess_root / name, 'postprocess_source')
    original_audit = postprocess_root / 'audit_original.py'
    assert digest(original_audit.read_bytes()) == json.loads((run_root / 'audit_final.json').read_text())['audit_source_sha256']
    entries['postprocess_source/audit_original.py'] = (original_audit, 'postprocess_source')
    records = []
    archive = output / 'synthetic_run.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, (path, category) in sorted(entries.items()):
            data = path.read_bytes()
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            records.append({'archive_path': name, 'category': category, 'sha256': digest(data), 'bytes': len(data)})
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in records:
            assert digest(z.read(row['archive_path'])) == row['sha256']
    counts = {category: sum(row['category'] == category for row in records) for category in sorted({r['category'] for r in records})}
    assert [counts[k] for k in ['record', 'input', 'fit_result', 'checkpoint']] == [713, 98, 98, 98]
    manifest = {'scope': 'Complete bounded synthetic run only; no household records or new computation.',
                'implementation_commit': '872818e3', 'export_source_sha256': digest(Path(__file__).read_bytes()),
                'archive_counts': counts, 'archive_members': records,
                'files': {path.name: {'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size}
                          for path in sorted(output.iterdir()) if path.is_file() and path.name != 'manifest.json'},
                'readback_all_passed': True}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'archive_counts': counts, 'archive_bytes': archive.stat().st_size, 'readback_all_passed': True}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for flag in ['run-root', 'repo-root', 'output', 'postprocess-root']:
        p.add_argument('--' + flag, required=True, type=Path)
    a = p.parse_args()
    export(a.run_root, a.repo_root, a.output, a.postprocess_root)
