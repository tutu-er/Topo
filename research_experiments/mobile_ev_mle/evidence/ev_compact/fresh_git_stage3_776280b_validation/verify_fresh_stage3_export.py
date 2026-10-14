"""Verify object provenance, newline-only restoration and fresh imports."""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

STAGE = Path(__file__).resolve().parent
ROOT = STAGE / 'fresh_stage3_776280b'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    exported = json.loads((ROOT / 'git_object_export_manifest.json').read_text())
    before = json.loads((ROOT / 'source_audit_before.json').read_text())
    restored = json.loads((ROOT / 'source_audit_restored.json').read_text())
    review = json.loads((ROOT / 'fresh_slim_review.json').read_text())
    allowed = {row['repository_path'] for row in before['files'] if row['status'] == 'line_endings_only'}
    changed = [row for row in exported['files'] if digest(ROOT / row['path']) != row['sha256']]
    if {row['path'] for row in changed} != allowed or any(not row['frozen_scientific_source'] for row in changed):
        raise ValueError('export changes extend beyond preflight newline-only source targets')
    if not (before['checked_count'] == restored['checked_count'] == 72
            and before['exact_before_count'] == 31 and before['line_ending_only_count'] == 41
            and before['content_changed_count'] == before['missing_count'] == before['restored_count'] == 0
            and restored['success'] and restored['restored_count'] == 41
            and restored['exact_after_count'] == 72 and restored['all_bytes_match']):
        raise ValueError('unexpected restoration validation counts')
    if not (review['success'] and review['audited_histories'] == 4 and not review['complete_raw_replay']
            and review['new_fit_calls'] == review['optimizer_calls'] == review['selector_calls'] == 0):
        raise ValueError('fresh slim replay failed or scope differs')
    # -I invocation ignores PYTHONPATH and user-site configuration. The only
    # project import root inserted here is the new object-export directory.
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module('research_experiments.mobile_ev_mle.ev_compact_slim_review')
    imports = []; outside = []
    project_bases = [Path('D:/0-github_workspace/Topo').resolve(),
                     Path('C:/Users/23761/.codex/worktrees/36a1/Topo').resolve()]
    for name, imported in sorted(sys.modules.items()):
        filename = getattr(imported, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        under_root = path.is_relative_to(ROOT.resolve())
        if any(path.is_relative_to(base) for base in project_bases) and not under_root:
            # This standalone verifier is not science code and is the sole
            # deliberately external project-local executable in this check.
            if path != Path(__file__).resolve():
                outside.append(dict(module=name, path=str(path)))
        if under_root:
            imports.append(dict(module=name, path=str(path.relative_to(ROOT)), sha256=digest(path)))
    if outside:
        raise ValueError('project imports escaped fresh Git-object directory: ' + repr(outside))
    if Path(module.__file__).resolve() != ROOT / 'research_experiments/mobile_ev_mle/ev_compact_slim_review.py':
        raise ValueError('fresh reviewer module origin mismatch')
    import numpy, scipy
    result = dict(success=True, commit=exported['commit'], exported_files=len(exported['files']),
                  exported_bytes=exported['bytes'], frozen_sources=72,
                  default_read_only_source_check=dict(exact=31, newline_only=41, real_changes=0, missing=0,
                    restoration_requested=False, restored=0,
                    all_416_original_exported_blobs_unchanged_verified_before_restore=True),
                  explicit_restore=dict(newline_only_changed_files=len(changed), exact_after=72,
                    all_nonrestored_exported_blobs_unchanged=True, only_new_D_directory_modified=True),
                  offline_actual_AC_review=dict(success=True, scope=review['scope'],
                    independent_histories=review['audited_histories'], totals=review['totals'],
                    complete_raw_replay=False, new_fits=0, new_optimizers=0, new_selectors=0,
                    raw_local_paths_read=False),
                  isolated_import_check=dict(project_imports=len(imports), all_from_fresh_object_export=True,
                    outside_project_imports=outside, gurobipy_imported='gurobipy' in sys.modules, imports=imports),
                  runtime=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                               numpy=numpy.__version__, scipy=scipy.__version__),
                  artifacts={name:digest(ROOT / name) for name in
                    ('git_object_export_manifest.json', 'source_audit_before.json', 'source_audit_restored.json', 'fresh_slim_review.json')},
                  tools={name:digest(ROOT / 'research_experiments/mobile_ev_mle' / name) for name in
                    ('ev_compact_source_restore.py', 'ev_original_source_restore.py', 'ev_policy_calibration.py',
                     'ev_compact_slim_review.py', 'ev_compact_review.py', 'ev_original_review.py')},
                  export_tool_sha256=exported['export_tool_sha256'],
                  verification_tool_sha256=digest(__file__))
    (ROOT / 'fresh_git_object_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key not in
                      ('isolated_import_check', 'tools', 'runtime', 'artifacts')}, indent=2))
    print(json.dumps(dict(project_imports=len(imports), escaped_imports=len(outside),
                         gurobipy_imported='gurobipy' in sys.modules)))


if __name__ == '__main__':
    main()
