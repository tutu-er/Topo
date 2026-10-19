"""Verify the sealed EV payload or copy it into a new empty restore directory."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inside(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('path escapes intended root: ' + relative)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify', action='store_true')
    group.add_argument('--restore', type=Path)
    parser.add_argument('--all-local', action='store_true', help='require/include originally untracked files too')
    args = parser.parse_args()
    with gzip.open(HERE / 'MANIFEST.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    errors = []
    missing_local = []
    checked = 0
    for row in records:
        p = inside(HERE, row['sealed_relative'])
        if not p.is_file():
            if row['tracked'] == 'true' or args.all_local:
                errors.append('missing ' + row['original_relative'])
            else:
                missing_local.append(row['original_relative'])
            continue
        if p.stat().st_size != int(row['bytes']) or digest(p) != row['sha256']:
            errors.append('hash mismatch ' + row['original_relative'])
        checked += 1
    if errors:
        raise SystemExit(json.dumps({'errors': errors, 'checked': checked}, ensure_ascii=False, indent=2))
    result = {'status': 'verified', 'checked': checked, 'missing_local_only': len(missing_local)}
    if args.restore:
        dest = args.restore.resolve()
        if dest == Path(dest.anchor) or dest == HERE or dest.is_relative_to(HERE) or HERE.is_relative_to(dest):
            raise ValueError('restore target must be a separate new directory')
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            raise ValueError('restore target must be new or empty')
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for row in records:
            if row['tracked'] != 'true' and not args.all_local:
                continue
            source = inside(HERE, row['sealed_relative'])
            target = inside(dest, row['original_relative'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError('refusing to replace ' + str(target))
            shutil.copy2(source, target)
            if digest(target) != row['sha256']:
                raise ValueError('restore hash mismatch ' + str(target))
            copied += 1
        result.update(status='restored', destination=str(dest), copied=copied, starts_scientific_work=False)
        (dest / 'RESTORE_RECEIPT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
