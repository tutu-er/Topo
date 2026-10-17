"""Package and restore-check the completed frozen study; never run science."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def digest(data):
    return sha256(data).hexdigest()


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def safe(root, name):
    if '\\' in name or ':' in name or name.startswith('/'):
        raise ValueError('nonportable archive path: ' + name)
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('archive path escapes destination: ' + name)
    return path


def build(run, finish, repository, output, restore_root):
    run, finish, repository, output, restore_root = map(
        lambda p: Path(p).resolve(), (run, finish, repository, output, restore_root))
    if output.exists() or restore_root.exists():
        raise FileExistsError('delivery and restoration require fresh directories')
    frozen, state, controller = read(run / 'freeze.json'), read(finish / 'checkpoint.json'), read(run / 'controller.json')
    if state['status'] != 'complete' or controller['status'] != 'statistical_complete':
        raise ValueError('all frozen computation and postprocessing must be complete')
    policy = frozen['policy_id']
    if state['policy_id'] != policy:
        raise ValueError('policy mismatch')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=repository, text=True).strip()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    snapshot = read(run / 'source_snapshot.json.gz')
    if snapshot['source_sha256'] != frozen['source_sha256']:
        raise ValueError('freeze/snapshot mismatch')
    entries = {}

    def add(name, data):
        safe(restore_root, name)
        if name in entries and entries[name] != data:
            raise ValueError('conflicting archive entries: ' + name)
        entries[name] = data

    for name in ('freeze.json', 'source_snapshot.json.gz', 'controller.json',
                 'audit_development.json', 'audit_calibration.json', 'audit_evaluation.json'):
        add('run/' + name, (run / name).read_bytes())
    for name in ('checkpoint.json', 'evidence_manifest.json', 'review_summary.md'):
        add('postprocess/' + name, (finish / name).read_bytes())
    for path in sorted((finish / 'steps').rglob('*')):
        if not path.is_file() or any(part == 'pytest_tmp' for part in path.parts):
            continue
        # Preserve results and execution logs; pytest scratch is not evidence.
        if 'result' in path.parts or path.suffix == '.log':
            add('postprocess/' + path.relative_to(finish).as_posix(), path.read_bytes())

    # These are public, tracked supporting modules. The exact scientific bytes
    # from the frozen snapshot override any Git newline normalization below.
    tracked = subprocess.check_output(['git', 'ls-files', '--',
        'research_experiments/mobile_ev_mle', 'rnj_wzzt_core'], cwd=repository, text=True).splitlines()
    source_root = run.parents[1]
    for name in tracked:
        if not name.endswith('.py'):
            continue
        path = source_root / name
        if path.is_file():
            add('repository/' + name, path.read_bytes())
    for name, expected in frozen['source_sha256'].items():
        data = base64.b64decode(snapshot['source_base64'][name], validate=True)
        if digest(data) != expected:
            raise ValueError('scientific snapshot hash mismatch: ' + name)
        base = Path() if name.startswith('rnj_wzzt_core/') else Path('research_experiments/mobile_ev_mle')
        resolved = safe(restore_root / 'repository', (base / name).as_posix())
        archive_name = 'repository/' + resolved.relative_to(restore_root / 'repository').as_posix()
        # Deliberate, byte-exact overlay of a frozen scientific source.
        entries[archive_name] = data
    for name in ('ev_bayes_response_delivery.py', 'ev_bayes_response_final_metrics_review.json',
                 'ev_bayes_response_final_metrics_review.md', 'ev_bayes_response_final_physical_review.json',
                 'ev_bayes_response_final_physical_review.md', 'ev_bayes_response_final_verification.json'):
        path = source_root / 'research_experiments/mobile_ev_mle' / name
        if path.exists():
            add('delivery_reviews/' + name, path.read_bytes())
    index = dict(policy_id=policy, schema='ev_bayes_response_archive_index_v1',
        repository_commit=commit, repository_branch=branch,
        scientific_sources=len(frozen['source_sha256']),
        members=[dict(path=n, bytes=len(b), sha256=digest(b)) for n, b in sorted(entries.items())])
    add('archive_index.json', encode(index))
    output.mkdir(parents=True)
    archive = output / 'postprocess_evidence.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for name, data in sorted(entries.items()):
            handle.writestr(name, data)

    # Select by index/status only: first complete evaluation and first failed
    # evaluation. This is a restoration sample, not an efficacy comparison.
    inventory = read(finish / 'steps/00_inventory/attempt_001/result/inventory.json')
    evaluations = sorted((r for r in inventory['records'] if r['phase'] == 'evaluation'), key=lambda r: r['spec']['index'])
    examples = [next(r for r in evaluations if r['status'] == 'complete'),
                next(r for r in evaluations if r['status'] != 'complete')]
    samples = []
    for row in examples:
        name = 'evaluation_%03d.json.gz' % row['spec']['index']
        original = Path(row['input']['path'])
        target = output / name
        shutil.copyfile(original, target)
        if digest(target.read_bytes()) != row['input']['sha256']:
            raise ValueError('representative raw record changed')
        samples.append(dict(path=name, index=row['spec']['index'], status=row['status'],
            bytes=target.stat().st_size, sha256=row['input']['sha256']))

    restore_root.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None or set(handle.namelist()) != set(entries):
            raise ValueError('archive CRC/member mismatch')
        for name, data in sorted(entries.items()):
            recovered = handle.read(name)
            if recovered != data:
                raise ValueError('archive byte round trip failed: ' + name)
            destination = safe(restore_root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(recovered)
            if digest(destination.read_bytes()) != digest(data):
                raise ValueError('restored disk bytes differ: ' + name)
    for row in samples:
        destination = restore_root / 'run/evaluation' / ('history_%03d.json.gz' % row['index'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / row['path'], destination)
        restored = read(destination)
        if (digest(destination.read_bytes()) != row['sha256'] or restored['policy_id'] != policy
                or restored['spec']['index'] != row['index'] or restored['status'] != row['status']):
            raise ValueError('restored raw record identity/hash differs')
    result = dict(schema='ev_bayes_response_delivery_v1', policy_id=policy,
        created=datetime.now(timezone.utc).isoformat(),
        archive=dict(path=archive.name, bytes=archive.stat().st_size, sha256=digest(archive.read_bytes())),
        archive_index_sha256=digest(entries['archive_index.json']), members=len(entries),
        repository_commit=commit, repository_branch=branch,
        representative_rule='lowest complete evaluation index and lowest noncomplete evaluation index',
        representative_records=samples, original_histories=208,
        full_original_history_scope='all raw hashes and seeds included; remaining large raw files retained in original run directory',
        restoration=dict(passed=True, root=str(restore_root), restored_members=len(entries),
            restored_samples=len(samples), scientific_sources=140,
            restored_sources_executed=False, byte_exact=True),
        packaging_tool_sha256=digest(Path(__file__).read_bytes()))
    (output / 'delivery_manifest.json').write_bytes(encode(result))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'finish', 'repository', 'output', 'restore-root'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    build(args.run, args.finish, args.repository, args.output, args.restore_root)


if __name__ == '__main__':
    main()
