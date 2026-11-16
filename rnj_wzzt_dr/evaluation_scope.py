"""Default catalog for new research evaluations; sealed experiments stay intact.

Only source-group membership is filtered. A failed or unavailable active entry
remains in the planned denominator. Explicit historical access labels retired
entries without changing the source inventory or its existing status fields.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_suffix('.json')


def _configuration():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if config.get('schema') != 'research_evaluation_scope_v1':
        raise ValueError('Unsupported research scope schema')
    active, retired = config.get('active_groups'), config.get('retired_groups')
    if not isinstance(active, dict) or not isinstance(retired, dict) or not active:
        raise ValueError('Active and retired source groups must be explicit mappings')
    if set(active) & set(retired):
        raise ValueError('A source group cannot be both active and retired')
    for group, count in {**active, **retired}.items():
        if not isinstance(group, str) or not group or type(count) is not int or count < 1:
            raise ValueError('Source group counts must be positive integers')
    if (config.get('active_count') != sum(active.values()) or
            config.get('historical_count') != sum(active.values()) + sum(retired.values())):
        raise ValueError('Scope totals disagree with source-group counts')
    source = config.get('source_inventory', {})
    if not isinstance(source.get('path'), str) or not re.fullmatch('[0-9a-f]{64}', source.get('sha256', '')):
        raise ValueError('A source inventory path and SHA-256 are required')
    return config


def _select(entries, include_retired, config):
    if type(include_retired) is not bool:
        raise TypeError('include_retired must be an explicit boolean')
    selected, seen = [], set()
    known = set(config['active_groups']) | set(config['retired_groups'])
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError('Catalog entries must be dictionaries')
        identifier, group = entry.get('id'), entry.get('source_group')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Catalog entries require unique nonempty IDs')
        seen.add(identifier)
        if group not in known:
            raise ValueError(f'Unknown source group for {identifier}: {group!r}')
        status = 'active' if group in config['active_groups'] else 'retired'
        if status == 'retired' and not include_retired:
            continue
        copied = deepcopy(entry)
        copied['evaluation_scope'] = dict(
            scope_id=config['scope_id'], status=status,
            included_in_default=status == 'active',
            source_group=group,
            source_inventory_sha256=config['source_inventory']['sha256'])
        selected.append(copied)
    return selected


def select_entries(entries, include_retired=False):
    """Copy and label a catalog/subset, excluding retired groups by default.

    Unknown groups and duplicate IDs are errors, including in excluded entries.
    Existing outcome/status/availability fields never determine inclusion.
    """
    return _select(entries, include_retired, _configuration())


def load_catalog(include_retired=False):
    """Read the hash-locked historical inventory; default 296, explicit 456.

    Group counts are checked before filtering, so missing entries cannot silently
    reduce either the active or historical planned denominator.
    """
    config = _configuration()
    relative = Path(config['source_inventory']['path'])
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Source inventory must stay inside the workspace')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['source_inventory']['sha256']:
        raise ValueError('Historical source inventory SHA-256 changed')
    entries = json.loads(raw)
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise ValueError('Historical source inventory must be a list of entries')
    selected = _select(entries, include_retired, config)
    counts = Counter(e.get('source_group') for e in entries)
    if dict(counts) != {**config['active_groups'], **config['retired_groups']}:
        raise ValueError(f'Historical source-group counts changed: {dict(counts)}')
    expected = config['historical_count'] if include_retired else config['active_count']
    if len(selected) != expected:
        raise ValueError('Selected catalog count differs from the locked scope')
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-retired', action='store_true', help='Explicitly read the historical 456-entry scope')
    parser.add_argument('--out', type=Path, help='Write the copied catalog to a new JSON file; never overwrite')
    args = parser.parse_args(argv)
    entries = load_catalog(include_retired=args.include_retired)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(entries, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(count=len(entries), include_retired=args.include_retired,
                         groups=dict(Counter(e['source_group'] for e in entries))), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
