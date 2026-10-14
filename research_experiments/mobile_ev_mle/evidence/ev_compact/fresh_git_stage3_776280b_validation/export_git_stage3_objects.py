"""Export only committed objects for the stage-3 offline replay check.

Reads the named C repository through git's object database. No checkout file
is copied, and the C repository is not modified.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = '776280b6845bd132b53a7bb8a2cba90db752fad6'
REPOSITORY = Path('C:/Users/23761/.codex/worktrees/36a1/Topo')
STAGE = Path(__file__).resolve().parent
TARGET = STAGE / 'fresh_stage3_776280b'
MODULE = 'research_experiments/mobile_ev_mle'
DELIVERY = MODULE + '/evidence/ev_compact/delivery_development_v2/development_ensemble_v2'
CODE_ROOTS = (MODULE + '/', 'research_experiments/mobile_ev_probing/', 'rnj_wzzt_core/')


def git(*args, input=None):
    return subprocess.run(['git', '-C', str(REPOSITORY), *args], input=input,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def main():
    if TARGET.resolve().parent != STAGE.resolve() or TARGET.exists():
        raise ValueError('requires a new named directory inside the isolated D stage')
    commit = git('rev-parse', COMMIT + '^{commit}').decode().strip()
    if commit != COMMIT:
        raise ValueError('unexpected committed object')
    freeze = json.loads(git('show', COMMIT + ':' + DELIVERY + '/freeze.json'))
    tree = {}
    for item in git('ls-tree', '-r', '-l', '-z', COMMIT).split(b'\0'):
        if not item:
            continue
        meta, name = item.split(b'\t', 1)
        mode, kind, oid, size = meta.decode().split()
        path = name.decode('utf-8')
        tree[path] = dict(mode=mode, kind=kind, git_blob=oid, bytes=int(size))
    selected = {path for path in tree if path.endswith('.py')
                and (path.startswith(CODE_ROOTS) or path == 'research_experiments/__init__.py')
                and '/evidence/' not in path}
    frozen_paths = set()
    for name in freeze['source_sha256']:
        if name.startswith('../mobile_ev_probing/'):
            path = 'research_experiments/mobile_ev_probing/' + name.split('/', 2)[2]
        elif name.startswith('rnj_wzzt_core/'):
            path = name
        else:
            if '..' in name.split('/') or name.startswith('/') or '\\' in name or ':' in name:
                raise ValueError('noncanonical frozen source name')
            path = MODULE + '/' + name
        frozen_paths.add(path)
    if len(frozen_paths) != 72:
        raise ValueError('exactly 72 frozen source targets required')
    selected |= frozen_paths
    evidence = {'freeze.json', 'source_snapshot.json.gz', 'delivery_manifest.json', 'DELIVERY_SCOPE.md'}
    evidence |= {f'trial_{seed}.slim.json.gz' for seed in freeze['seeds']}
    selected |= {DELIVERY + '/' + name for name in evidence}
    missing = selected - set(tree)
    if missing:
        raise ValueError('required committed paths absent: ' + repr(sorted(missing)))
    paths = sorted(selected)
    if any(tree[path]['kind'] != 'blob' or tree[path]['mode'] not in ('100644', '100755') for path in paths):
        raise ValueError('only regular committed blob objects may be exported')
    batch = git('cat-file', '--batch', input=('\n'.join(tree[path]['git_blob'] for path in paths) + '\n').encode())
    TARGET.mkdir()
    offset = 0; rows = []
    for path in paths:
        end = batch.index(b'\n', offset)
        oid, kind, size = batch[offset:end].decode().split(); size = int(size)
        data = batch[end + 1:end + 1 + size]; offset = end + 2 + size
        row = tree[path]
        if (oid != row['git_blob'] or kind != 'blob' or size != row['bytes'] or len(data) != size
                or hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + data).hexdigest() != oid):
            raise ValueError('Git object byte verification failed: ' + path)
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe committed export path')
        destination = TARGET / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        rows.append(dict(path=path, **row, sha256=hashlib.sha256(data).hexdigest(),
                         frozen_scientific_source=path in frozen_paths,
                         declaration='exact git cat-file committed blob bytes'))
    if offset != len(batch):
        raise ValueError('unexpected bytes after Git object batch')
    result = dict(commit=COMMIT, repository_object_database=str(REPOSITORY), target=str(TARGET),
                  export_method='git ls-tree + git cat-file --batch; never checkout file copy',
                  git_version=git('--version').decode().strip(), exported_files=len(rows),
                  frozen_scientific_sources=len(frozen_paths), bytes=sum(row['bytes'] for row in rows),
                  evidence_seeds=freeze['seeds'], original_raw_exported=False,
                  calibration_or_evaluation_exported=False,
                  export_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), files=rows)
    (TARGET / 'git_object_export_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
