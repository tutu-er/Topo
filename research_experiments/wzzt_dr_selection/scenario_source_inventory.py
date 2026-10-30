"""Freeze a complete existing panel using metadata; never load NPZ values."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Existing inventory retained; do not overwrite')
    docs = args.source_repo/'docs/wzzt_paper_20261029'
    protocol_path, catalog_path = docs/'protocol.json', docs/'datasets.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    # The inclusion rule uses only pre-existing panel/sample labels, never outcomes.
    chosen = [r for r in catalog if r['panel'] == 'main' and r['N'] == 288]
    cases = ('paper15', 'soumalas11', 'flynn16', 'pengwah18')
    noise = (0.0, 0.0002, 0.0005, 0.001)
    expected = {(c, b, v) for c in cases for b in range(10) for v in noise}
    keys = {(r['case'], r['bundle'], r['voltage_sigma']) for r in chosen}
    assert len(chosen) == len(keys) == 160 and keys == expected
    assert all(r['condition'] == 'reference' for r in chosen)
    assert all(r['job_id'] in protocol['jobs'] for r in chosen)
    rows = []
    for row in sorted(chosen, key=lambda r:r['job_id']):
        entry = {k:row[k] for k in ('job_id', 'case', 'bundle', 'N', 'condition', 'voltage_sigma', 'pq_sigma')}
        entry['terminal_count'] = len(json.loads(row['labels_json']))
        entry['declared_data_error'] = row.get('data_error')
        for role in ('train', 'validation', 'test'):
            path = Path(row[role+'_path'])
            entry[role] = dict(path=str(path), exists=path.is_file(),
                               catalog_sha256=row.get(role+'_sha256'),
                               content_hash_verified=False, values_loaded=False)
        result_path = Path(row['result_path'])
        entry['existing_result'] = dict(path=str(result_path), exists=result_path.is_file(),
                                       catalog_sha256=row.get('result_sha256'), contents_loaded=False)
        rows.append(entry)
    counter = lambda key:dict(sorted(Counter(r[key] for r in rows).items()))
    summary = dict(selected_conditions=len(rows), by_case=counter('case'),
                   by_terminal_count=counter('terminal_count'), by_voltage_sigma=counter('voltage_sigma'),
                   complete_observation_file_sets=sum(all(r[role]['exists'] for role in ('train','validation','test')) for r in rows),
                   existing_result_files=sum(r['existing_result']['exists'] for r in rows),
                   declared_data_errors=sum(r['declared_data_error'] is not None for r in rows))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Metadata-only source qualification; no structural ambiguity/recovery finding',
                  inclusion_rule='all pre-existing main panel N288; no result/true-block/noise-performance filtering',
                  source_sha256={str(p):sha(p) for p in (protocol_path,catalog_path,Path(__file__))},
                  summary=summary, rows=rows,
                  data_domain='already-seen synthetic networks and load/DER profiles; not iFlex or field validation',
                  independent_unit='10 original bundle blocks; 4 cases and 4 noise settings paired within them',
                  declared_history='3 physical days x 96 points in train; validation/test independent replicates, not a real chronological sequence',
                  pending=['verify observation bytes/shapes before use', 'inspect complete historical candidate outputs under a declared algorithm version', 'define chronological forecasting and physical DR adapter before any performance batch'],
                  solver_calls=0, physical_calls=0, NPZ_values_loaded=False, truth_files_loaded=False,
                  existing_result_contents_loaded=False, new_future_trajectories_read=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
