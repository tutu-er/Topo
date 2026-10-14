"""Outside-freeze pilot gate and saved-policy calibration/evaluation export."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from . import gradient_price_study as io
from . import ev_policy_calibration as cal

METHODS=('none','constant','quality_fixed','quality_mobile','random_mobile',
    'charge_fixed','v2g_fixed','charge_mobile','v2g_mobile',
    'refined_v2g_mobile','capped_v2g_mobile')
PRIMARY=('refined_v2g_mobile','capped_v2g_mobile')
DECLARED_CANDIDATES=('clade_01','clade_02','clade_12','star')


def trials(directory):
    return [io.read_json(p) for p in sorted(Path(directory).glob('trial_*.json.gz'))]


def checked_trials(directory,freeze,phase):
    rows=trials(directory)
    if freeze.get('phase')!=phase:raise ValueError('unexpected evidence phase')
    ids=[r['seed'] for r in rows]
    if len(ids)!=len(set(ids)) or not set(ids)<=set(freeze['seeds']):
        raise ValueError('duplicate or unplanned independent history')
    if any(r.get('phase')!=phase for r in rows):raise ValueError('mixed trial phases')
    return rows


def pilot_gate(directory,review):
    directory=Path(directory);freeze=io.read_json(directory/'freeze.json')
    rows=checked_trials(directory,freeze,'development')
    reasons=[]
    if len(rows)!=len(freeze['seeds']):reasons.append('development histories incomplete')
    if not review.get('success'):reasons.append('independent replay failed or missing')
    if review.get('policy_id')!=freeze['policy_id']:reasons.append('review policy identity')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob('trial_*.json.gz'))}
    if review.get('trial_sha256')!=hashes:reasons.append('review input bytes differ')
    if sorted(review.get('seeds',[]))!=sorted(r['seed'] for r in rows):reasons.append('review seed coverage')
    if sorted(r['seed'] for r in rows)!=sorted(freeze['seeds']):reasons.append('seed coverage')
    for row in rows:
        selection=row['selection']
        if row.get('policy_id')!=freeze['policy_id']:reasons.append('policy identity')
        if selection.get('random_used_by_primary') is not False:reasons.append('random entered primary')
        if selection.get('uses_truth') or selection.get('uses_future_noise'):reasons.append('information leak')
        if row.get('historical_failures'):reasons.append('historical fit failure')
        if set(r['method'] for r in row['rows'])!=set(METHODS):reasons.append('method denominator')
        for name in ('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile',*PRIMARY):
            method=selection['methods'].get(name,{})
            if not method.get('inheritance_protected'):reasons.append('incumbent protection '+name)
    return dict(passed=not reasons,reasons=sorted(set(reasons)),policy_id=freeze['policy_id'],
        development_seeds=freeze['seeds'],source_sha256=freeze['source_sha256'],
        scope='verified implementation and replay; effectiveness evaluated on new histories',
        random_used_by_primary=False,independent_review=review)


def risk_report(calibration_directory,evaluation_directory,*,alpha=.1):
    cd,ed=Path(calibration_directory),Path(evaluation_directory)
    cf,ef=io.read_json(cd/'freeze.json'),io.read_json(ed/'freeze.json')
    if cal.freeze_identity(cf)!=cal.freeze_identity(ef):raise ValueError('policy versions differ')
    cr,er=checked_trials(cd,cf,'calibration'),checked_trials(ed,ef,'evaluation')
    names=DECLARED_CANDIDATES
    answer=dict(reference='declared synthetic cable/length, load/source/noise distribution',
        planned_calibration=len(cf['seeds']),planned_evaluation=len(ef['seeds']),
        actual_calibration=len(cr),actual_evaluation=len(er),alpha=alpha,
        simulations_are_not_field_coverage=True,topology_probability=False)
    for label,methods,familywise in (('individual_policies',METHODS,False),
                                     ('primary_family',PRIMARY,True)):
        spec=cal.spec_from_freeze(cf,reference_id='declared_lv_asset_reference_v1',
            candidates=names,methods=methods,alpha=alpha,familywise=familywise)
        cdata=[cal.adapt_saved_trial(r,cf,spec) for r in cr]
        edata=[cal.adapt_saved_trial(r,ef,spec) for r in er]
        calibration=cal.calibrate(cdata,spec,expected_history_ids=[str(r['seed']) for r in cr],
            planned_history_ids=[str(x) for x in cf['seeds']])
        evaluation=cal.evaluate(edata,calibration,expected_history_ids=[str(r['seed']) for r in er])
        answer[label]=dict(calibration=calibration,evaluation=evaluation)
    return answer


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    g=sub.add_parser('gate');g.add_argument('--directory',type=Path,required=True)
    g.add_argument('--review',type=Path,required=True);g.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('risk');r.add_argument('--calibration',type=Path,required=True)
    r.add_argument('--evaluation',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=(pilot_gate(args.directory,io.read_json(args.review)) if args.command=='gate'
            else risk_report(args.calibration,args.evaluation))
    io.write_json(args.output,result,compressed=args.output.suffix=='.gz')
    print('passed='+str(result.get('passed')) if args.command=='gate' else
        'calibration='+str(result['actual_calibration'])+' evaluation='+str(result['actual_evaluation']))


if __name__=='__main__':main()
