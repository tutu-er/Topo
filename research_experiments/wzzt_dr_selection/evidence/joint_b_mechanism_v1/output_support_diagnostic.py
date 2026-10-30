"""Classify saved original-rule outputs; do not filter or refit candidates."""
from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(sys.argv[1]).resolve()
    protocol_path = root / 'output_support_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    assert sha(Path(__file__)) == protocol['source_sha256']
    for item in protocol['inputs'].values():
        assert sha(Path(item['path'])) == item['sha256']
    output = root / 'output_support_diagnostic.json'
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(Path(protocol['inputs']['audit']['path']).read_text(encoding='utf-8'))['records']
    raw = json.loads(Path(protocol['inputs']['raw_qualification']['path']).read_text(encoding='utf-8'))['rows']
    assert len(audit) == len(raw) == 160
    by_id = {v['job_id']: v for v in raw}
    rows = []
    for record in audit:
        families = [frozenset(tuple(sorted(s)) for s in f) for f in record['contracted_families']]
        assert len(set(families)) == record['eligible_contracted_families']
        n = len(set().union(*(set(s) for s in next(iter(families)))))
        disputed = set.union(*(set(f) for f in families)) - set.intersection(*(set(f) for f in families))
        internal = sorted(s for s in disputed if 2 <= len(s) < n)
        simultaneous = []
        for c, d in combinations(internal, 2):
            if set(c) & set(d):
                continue
            if any(c in f and d in f for f in families):
                simultaneous.append([list(c), list(d)])
        rows.append({'job_id': record['job_id'], 'voltage_sigma': record['voltage_sigma'],
            'raw_two_block_opportunity': by_id[record['job_id']]['two_disjoint_disputes'],
            'different_original_rule_outputs': len(families) > 1,
            'output_family_count': len(families), 'proper_disputed_supports': [list(s) for s in internal],
            'two_disjoint_output_disputes': bool(simultaneous), 'disjoint_pairs': simultaneous})
    strata = {}
    for sigma in sorted({v['voltage_sigma'] for v in rows}):
        part = [v for v in rows if v['voltage_sigma'] == sigma]
        strata[str(sigma)] = {'denominator': len(part),
            'raw_two_block_opportunity': sum(v['raw_two_block_opportunity'] for v in part),
            'different_original_rule_outputs': sum(v['different_original_rule_outputs'] for v in part),
            'two_disjoint_output_disputes': sum(v['two_disjoint_output_disputes'] for v in part)}
    result = {'status': 'complete', 'rows': rows, 'by_voltage_sigma': strata,
        'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(protocol_path),
        'LP_calls': 0, 'identifier_calls': 0, 'NPZ_files_loaded': 0,
        'topology_truth_read': False, 'new_future_data_read': False,
        'limitation': 'Original-rule output shapes only; agreement does not establish truth or absence of useful DR. Raw weak/zero candidates remain retained.'}
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'by_sigma': strata, 'LP_calls': 0}))


if __name__ == '__main__':
    main()
