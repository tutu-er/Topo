"""Portable, records-only replay; never opens a saved raw or review path."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .ev_compact_risk_independent_audit import (
    PRIMARY, canonical_hash, read_json, reconstruct_calibration,
    reconstruct_evaluation, projected_records,
)


def replay_reports(before, final):
    checks = {}
    b, f = before['standard_records'], final['standard_records']
    for label, bundle in (('before', b), ('final', f)):
        checks[label + '_bundle_hash'] = canonical_hash({k: v for k, v in bundle.items()
            if k != 'bundle_id'}) == bundle['bundle_id']
    checks['same_spec'] = b['spec'] == f['spec']
    checks['same_calibration_records'] = b['calibration'] == f['calibration']
    checks['same_planned_ids'] = all(b[k] == f[k] for k in
        ('planned_calibration_ids', 'planned_evaluation_ids'))
    summaries = {}
    for phase in ('calibration', 'evaluation'):
        rows = f[phase]
        ids = [str(r['history_id']) for r in rows]
        units = [str(r['independent_unit_id']) for r in rows]
        checks[phase + '_unique_complete_ids'] = (len(set(ids)) == len(rows)
            and len(set(units)) == len(rows) and set(ids) == set(f['planned_' + phase + '_ids']))
    checks['disjoint_histories'] = not ({r['history_id'] for r in f['calibration']}
        & {r['history_id'] for r in f['evaluation']})
    checks['disjoint_units'] = not ({r['independent_unit_id'] for r in f['calibration']}
        & {r['independent_unit_id'] for r in f['evaluation']})
    for label, methods, familywise in (
            ('individual_policies', f['spec']['methods'], False),
            ('primary_family', list(PRIMARY), True)):
        old, saved = before[label]['calibration'], final[label]['calibration']
        checks[label + '_same_complete_calibration'] = old == saved
        checks[label + '_calibration_identity'] = saved['calibration_id'] == canonical_hash(
            {k: v for k, v in saved.items() if k != 'calibration_id'})
        spec = dict(f['spec'], methods=methods, familywise=familywise)
        cr = f['calibration'] if not familywise else projected_records(f['calibration'], methods)
        er = f['evaluation'] if not familywise else projected_records(f['evaluation'], methods)
        cal = reconstruct_calibration(cr, spec, f['planned_calibration_ids'])
        ev = reconstruct_evaluation(er, cal)
        checks[label + '_whole_calibration_replay'] = cal == saved
        checks[label + '_whole_final_evaluation_replay'] = ev == final[label]['evaluation']
        old_er = b['evaluation'] if not familywise else projected_records(b['evaluation'], methods)
        checks[label + '_whole_before_evaluation_replay'] = (
            reconstruct_evaluation(old_er, cal) == before[label]['evaluation'])
        summaries[label] = dict(calibration_id=cal['calibration_id'],
            evaluation_totals=ev['totals'], methods=ev['methods'],
            calibration_sha256=canonical_hash(cal), evaluation_sha256=canonical_hash(ev))
    return dict(schema='ev_compact_risk_records_only_replay_v1',
        success=all(checks.values()), checks=checks,
        errors=[k for k, v in checks.items() if not v],
        calibration_histories=len(f['calibration']), evaluation_histories=len(f['evaluation']),
        summary=summaries, raw_files_validated=False, raw_or_review_paths_accessed=False,
        scope='small standard_records content, fixed calibration identities and decision arithmetic only',
        new_fit_calls=0, new_selector_calls=0, topology_probability=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--final', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--readme', type=Path)
    args = parser.parse_args()
    result = replay_reports(read_json(args.before), read_json(args.final))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result['input_sha256'] = {str(p): digest(p) for p in (args.before, args.final)}
    result['tool_sha256'] = digest(Path(__file__))
    import inspect
    result['independent_auditor_sha256'] = digest(Path(inspect.getfile(reconstruct_calibration)))
    if args.readme:
        result['readme_sha256'] = digest(args.readme)
    result['actual_CLI_arguments'] = dict(before=str(args.before), final=str(args.final),
                                         output=str(args.output), readme=str(args.readme) if args.readme else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                                      allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('success', 'errors', 'calibration_histories',
        'evaluation_histories', 'raw_or_review_paths_accessed')}, ensure_ascii=False))
    raise SystemExit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
