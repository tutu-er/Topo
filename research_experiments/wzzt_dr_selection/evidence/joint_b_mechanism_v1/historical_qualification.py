"""Saved-candidate structural eligibility only; no scores, fits or truth.

This does not prove a continuous compensation direction or device availability.
It inventories a specific two-disjoint-block opportunity before any DR trial.
"""
from pathlib import Path
from itertools import combinations
from collections import Counter
import gzip
import hashlib
import json
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(supports):
    return frozenset(tuple(sorted(v)) for v in supports)


def main():
    root = Path(sys.argv[1]).resolve()
    protocol_path = root / 'qualification_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    assert digest(Path(__file__)) == protocol['source_sha256']
    inputs = protocol['inputs']
    for item in inputs.values():
        assert digest(Path(item['path'])) == item['sha256'], 'input drift'
    output = root / 'qualification_result.json'
    if output.exists():
        raise FileExistsError('do not overwrite completed qualification')
    extracted = json.loads(gzip.decompress(Path(inputs['candidates']['path']).read_bytes()))['records']
    audits = json.loads(Path(inputs['audit']['path']).read_text(encoding='utf-8'))['records']
    by_id = {v['job_id']: v for v in audits}
    assert len(extracted) == len(audits) == len(by_id) == 160
    rows = []
    for record in sorted(extracted, key=lambda v: v['job_id']):
        audit = by_id[record['job_id']]
        assert audit['status'] == 'complete'
        assert record['original_result_sha256'] == audit['result_sha256']
        labels = record['labels']
        n = len(labels)
        eligible = audit['eligible_point_indices']
        families = {i: canon(record['base_points'][i]['supports']) for i in eligible}
        witnessed = {}
        for i, family in families.items():
            internal = sorted(v for v in family if 2 <= len(v) < n)
            for c, d in combinations(internal, 2):
                if set(c) & set(d):
                    continue
                no_c = [j for j, f in families.items() if c not in f]
                no_d = [j for j, f in families.items() if d not in f]
                if not no_c or not no_d:
                    continue
                both = sorted(set(no_c) & set(no_d))
                key = (c, d)
                witnessed.setdefault(key, {'C': [labels[k] for k in c],
                    'D': [labels[k] for k in d], 'joint_point': i,
                    'point_omitting_C': no_c[0], 'point_omitting_D': no_d[0],
                    'one_point_omitting_both': both[0] if both else None,
                    'all_cross_region_endpoint_pairs': [[labels[a], labels[b]] for a in c for b in d],
                    'unactuated_terminal_per_block_possible': True})
        opportunities = list(witnessed.values())
        simultaneous = [v for v in opportunities if v['one_point_omitting_both'] is not None]
        rows.append({'job_id': record['job_id'], 'case': audit['case'],
            'bundle': audit['bundle'], 'voltage_sigma': audit['voltage_sigma'], 'N': audit['N'],
            'eligible_points': len(eligible), 'eligible_declared_families': len(set(families.values())),
            'two_disjoint_disputes': bool(opportunities),
            'both_omitted_in_one_saved_alternative': bool(simultaneous),
            'opportunity_count': len(opportunities), 'opportunities': opportunities,
            'status': 'structural_opportunity_only' if opportunities else 'no_opportunity_for_this_two_block_pattern',
            'continuous_compensation_checked': False, 'device_availability_checked': False,
            'true_topology_used': False, 'response_effectiveness_checked': False})
    strata = {}
    for sigma in sorted({v['voltage_sigma'] for v in rows}):
        part = [v for v in rows if v['voltage_sigma'] == sigma]
        strata[str(sigma)] = {'denominator': len(part),
            'two_disjoint_disputes': sum(v['two_disjoint_disputes'] for v in part),
            'both_omitted_in_one_alternative': sum(v['both_omitted_in_one_saved_alternative'] for v in part)}
    primary = [v for v in rows if v['voltage_sigma'] == protocol['primary_voltage_sigma']]
    assert len(primary) == 40
    result = {'status': 'complete', 'source_sha256': digest(Path(__file__)),
        'protocol_sha256': digest(protocol_path), 'rows': rows, 'by_voltage_sigma': strata,
        'primary_denominator': len(primary),
        'primary_opportunity_count': sum(v['two_disjoint_disputes'] for v in primary),
        'aggregate_statuses': dict(Counter(v['status'] for v in rows)),
        'LP_calls': 0, 'identifier_calls': 0, 'NPZ_files_loaded': 0,
        'new_future_data_read': False, 'topology_truth_read': False,
        'selected_case': None, 'selected_pair': None,
        'limitations': ['finite saved candidates, not a full parameter domain',
            'a missing saved support is not a proved omission completion',
            'no numerical compensation or action effect was inferred',
            'no device, price optimization or final-output guarantee']}
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'denominator': len(rows), 'by_sigma': strata,
                      'statuses': result['aggregate_statuses'], 'LP_calls': 0}))


if __name__ == '__main__':
    main()
