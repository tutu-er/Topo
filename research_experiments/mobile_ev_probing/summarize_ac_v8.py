"""Derive descriptive policy counts from frozen AC observations and certificates."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    folder = HERE / 'results_v8' / 'ac_formal'
    raw = folder / 'results.json'
    data = json.loads(raw.read_text(encoding='utf-8'))
    audit = json.loads((folder / 'artifact_audit.json').read_text(encoding='utf-8'))
    assert audit['passed'] and data['source_hash_unchanged']
    rows = []
    for episode in data['episodes']:
        design = episode['design']
        skip = design['status'] == 'conditional_singleton_skip'
        single = 'site1_none' if skip else design['selection']['selected']
        mobile = 'site1_none' if skip else design['mobile_selection']['selected']
        actions = {'no_EV': 'site1_none', 'selected_single': single,
                   'matched_constant': single.replace('endpoint', 'constant'),
                   'fixed_site1': 'site1_none' if skip else 'site1_endpoint',
                   'selected_mobile': mobile,
                   'fixed_mobile12': 'site1_none' if skip else 'mobile12'}
        outcomes = {}
        for policy, action in actions.items():
            outcomes[policy] = {}
            for sharing in ('path', 'output'):
                records = [v for v in episode['records']
                           if v['action'] == action and v['sharing'] == sharing]
                assert len(records) == 3
                truth = [v for v in records if v['candidate'] == episode['truth']][0]
                assert truth['result']['status'] != 'excluded'
                wrong = [v for v in records if v['candidate'] != episode['truth']]
                outcomes[policy][sharing] = {
                    'wrong_excluded': sum(v['result']['status'] == 'excluded' for v in wrong),
                    'minimum_wrong_lower': min(v['result']['lower'] for v in wrong),
                    'minimum_wrong_upper': min(v['result']['upper'] for v in wrong)}
        rows.append(dict(truth=episode['truth'], seed=episode['seed'],
                         history=episode['history'], profile=episode['profile'],
                         skip_EV=skip, actions=actions, outcomes=outcomes))
    groups = {}
    for name, subset in [('all', rows)] + [
            (f'{history}_{profile}', [r for r in rows if r['history'] == history and r['profile'] == profile])
            for history, profile in [('flat', 'symmetric'), ('rich', 'symmetric'), ('flat', 'weak')]]:
        groups[name] = {'n': len(subset), 'skip_EV': sum(r['skip_EV'] for r in subset)}
        for policy in rows[0]['actions']:
            groups[name][policy] = {
                sharing: {'fully_separated': sum(r['outcomes'][policy][sharing]['wrong_excluded'] == 2 for r in subset),
                          'mean_min_wrong_lower': sum(r['outcomes'][policy][sharing]['minimum_wrong_lower'] for r in subset)/len(subset)}
                for sharing in ('path', 'output')}
    truths = [json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*_truth.json')]
    result = dict(input_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                  groups=groups, rows=rows, audit_counts=audit['counts'],
                  max_certificate_gap=audit['max_recorded_certificate_gap'],
                  voltage_range=[min(t['min_voltage'] for t in truths), max(t['max_voltage'] for t in truths)],
                  max_AC_residual=max(t['max_ac_residual'] for t in truths),
                  max_DistFlow_error=max(t['max_distflow_error'] for t in truths),
                  scope='Descriptive finite synthetic cases; excluded rivals plus retained truth is conditional candidate-set separation, not interval AC existence certification.',
                  output_ablation_scope='Actions and stopping decisions are supplied by the path controller; output results are same-data ablations, not an independently optimized output controller.')
    (folder / 'policy_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
