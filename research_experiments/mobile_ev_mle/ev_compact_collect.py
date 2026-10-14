"""Outside-freeze pilot gate and saved-policy calibration/evaluation export."""
from __future__ import annotations
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
from . import gradient_price_study as io
from . import ev_policy_calibration as cal

METHODS=('none','constant','quality_fixed','quality_mobile','random_mobile',
    'charge_fixed','v2g_fixed','charge_mobile','v2g_mobile',
    'refined_v2g_mobile','capped_v2g_mobile')
PRIMARY=('refined_v2g_mobile','capped_v2g_mobile')
DECLARED_CANDIDATES=('clade_01','clade_02','clade_12','star')


def _read_trial(path):
    """One raw-file read supplies both JSON and its exact compressed hash."""
    before=path.stat();data=path.read_bytes()
    trial=json.loads(gzip.decompress(data))
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('raw trial changed while reading: '+path.name)
    return trial,dict(name=path.name,sha256=hashlib.sha256(data).hexdigest(),
        compressed_bytes=len(data),mtime_ns=after.st_mtime_ns)


def _gate_projection(trial):
    """Gate needs booleans, method names and inheritance flags, not fit arrays."""
    selection=trial['selection']
    return dict(seed=trial['seed'],phase=trial['phase'],policy_id=trial.get('policy_id'),
        historical_failures=bool(trial.get('historical_failures')),
        rows=[dict(method=r['method']) for r in trial['rows']],
        selection=dict(random_used_by_primary=selection.get('random_used_by_primary'),
            uses_truth=selection.get('uses_truth'),uses_future_noise=selection.get('uses_future_noise'),
            methods={name:dict(inheritance_protected=r.get('inheritance_protected'))
                for name,r in selection.get('methods',{}).items()}))


def checked_trials(directory,freeze,phase,*,projector=None,manifest=None):
    """Return small projections, releasing each raw object before the next read.

    Completed histories remain phase/seed checked. A projector must detach
    small fields; it must not return the raw trial or large fit-state arrays.
    """
    if freeze.get('phase')!=phase:raise ValueError('unexpected evidence phase')
    project=_gate_projection if projector is None else projector
    answer=[];seen=set();planned=set(freeze['seeds'])
    for path in sorted(Path(directory).glob('trial_*.json.gz')):
        trial,entry=_read_trial(path)
        try:
            seed=trial['seed']
            if seed in seen or seed not in planned:raise ValueError('duplicate or unplanned independent history')
            if trial.get('phase')!=phase:raise ValueError('mixed trial phases')
            seen.add(seed)
            small=project(trial)
            if small is trial:raise ValueError('projector retained the full raw trial')
            answer.append(small)
            after=path.stat()
            if (after.st_size,after.st_mtime_ns)!=(entry['compressed_bytes'],entry['mtime_ns']):
                raise ValueError('raw trial changed during projection: '+path.name)
            if manifest is not None:manifest.append(dict(entry,seed=seed,phase=phase))
        finally:
            del trial
    return answer


def pilot_gate(directory,review,qp_review=None):
    directory=Path(directory);freeze=io.read_json(directory/'freeze.json')
    manifest=[];rows=checked_trials(directory,freeze,'development',manifest=manifest)
    reasons=[]
    if len(rows)!=len(freeze['seeds']):reasons.append('development histories incomplete')
    if not review.get('success'):reasons.append('independent replay failed or missing')
    if review.get('policy_id')!=freeze['policy_id']:reasons.append('review policy identity')
    hashes={r['name']:r['sha256'] for r in manifest}
    if review.get('trial_sha256')!=hashes:reasons.append('review input bytes differ')
    if sorted(review.get('seeds',[]))!=sorted(r['seed'] for r in rows):reasons.append('review seed coverage')
    if qp_review is not None:
        if (not qp_review.get('success') or qp_review.get('policy_id')!=freeze['policy_id']
                or qp_review.get('trial_sha256')!=hashes
                or sorted(qp_review.get('seeds',[]))!=sorted(r['seed'] for r in rows)):
            reasons.append('QP matrix replay failed or input identity differs')
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
        random_used_by_primary=False,independent_review=review,QP_matrix_review=qp_review,
        raw_manifest=manifest,raw_collection='one file read/project/release at a time')


def risk_report(calibration_directory,evaluation_directory,*,alpha=.1):
    cd,ed=Path(calibration_directory),Path(evaluation_directory)
    cf,ef=io.read_json(cd/'freeze.json'),io.read_json(ed/'freeze.json')
    if cal.freeze_identity(cf)!=cal.freeze_identity(ef):raise ValueError('policy versions differ')
    spec=cal.spec_from_freeze(cf,reference_id='declared_lv_asset_reference_v1',
        candidates=DECLARED_CANDIDATES,methods=METHODS,alpha=alpha,familywise=False)
    manifests={phase:[] for phase in ('calibration','evaluation')}
    def adapt(trial,freeze):
        small=cal.adapt_saved_trial(trial,freeze,spec)
        # Preserve the old per-spec candidate-miss rule when taking PRIMARY.
        small['base_candidate_miss']=bool(trial.get('candidate_miss',False)
            or small['true_name'] not in small['candidate_names'])
        rows={r['method']:r for r in trial['rows']}
        small['method_candidate_miss']={m:bool(rows.get(m,{}).get('candidate_miss',False)) for m in METHODS}
        return small
    cr=checked_trials(cd,cf,'calibration',projector=lambda t:adapt(t,cf),manifest=manifests['calibration'])
    er=checked_trials(ed,ef,'evaluation',projector=lambda t:adapt(t,ef),manifest=manifests['evaluation'])
    bundle=dict(schema='ev_compact_standard_records_v1',spec=spec.summary(),
        calibration=cr,evaluation=er,
        planned_calibration_ids=[str(x) for x in cf['seeds']],
        planned_evaluation_ids=[str(x) for x in ef['seeds']],
        raw_manifest=manifests,
        raw_directories=dict(calibration=str(cd.resolve()),evaluation=str(ed.resolve())),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='small saved-policy records; no raw states, fitting or action selection')
    bundle['bundle_id']=cal.object_hash(bundle)
    return risk_report_from_standard_records(bundle)


def _method_subset(records,methods):
    answer=[]
    for record in records:
        small=dict(record,methods={m:record['methods'][m] for m in methods})
        small['candidate_miss']=bool(record['base_candidate_miss']
            or any(record['method_candidate_miss'][m] for m in methods))
        answer.append(small)
    return answer


def risk_report_from_standard_records(bundle):
    """Offline statistical replay consumes only the embedded small bundle."""
    if bundle.get('schema')!='ev_compact_standard_records_v1':raise ValueError('unknown standard-record schema')
    if cal.object_hash({k:v for k,v in bundle.items() if k!='bundle_id'})!=bundle.get('bundle_id'):
        raise ValueError('standard-record content/hash mismatch')
    spec=cal.CalibrationSpec(**bundle['spec']);cr,er=bundle['calibration'],bundle['evaluation']
    if spec.methods!=METHODS or spec.familywise:raise ValueError('standard records require all individual policies')
    for phase,rows in (('calibration',cr),('evaluation',er)):
        planned=set(bundle['planned_'+phase+'_ids'])
        if any(r['history_id'] not in planned for r in rows):raise ValueError('unplanned standard history')
    answer=dict(reference='declared synthetic cable/length, load/source/noise distribution',
        planned_calibration=len(bundle['planned_calibration_ids']),planned_evaluation=len(bundle['planned_evaluation_ids']),
        actual_calibration=len(cr),actual_evaluation=len(er),alpha=spec.alpha,
        simulations_are_not_field_coverage=True,topology_probability=False,
        standard_records=bundle,raw_manifest=bundle['raw_manifest'],
        raw_collection='each raw read and adapted once; PRIMARY projected from small records')
    for label,methods,familywise in (('individual_policies',METHODS,False),
                                     ('primary_family',PRIMARY,True)):
        local_spec=replace(spec,methods=methods,familywise=familywise)
        cdata=cr if methods==METHODS else _method_subset(cr,methods)
        edata=er if methods==METHODS else _method_subset(er,methods)
        calibration=cal.calibrate(cdata,local_spec,expected_history_ids=[r['history_id'] for r in cr],
            planned_history_ids=bundle['planned_calibration_ids'])
        evaluation=cal.evaluate(edata,calibration,expected_history_ids=[r['history_id'] for r in er])
        answer[label]=dict(calibration=calibration,evaluation=evaluation)
    return answer


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    g=sub.add_parser('gate');g.add_argument('--directory',type=Path,required=True)
    g.add_argument('--review',type=Path,required=True);g.add_argument('--output',type=Path,required=True)
    g.add_argument('--qp-review',type=Path,required=True)
    r=sub.add_parser('risk');r.add_argument('--calibration',type=Path,required=True)
    r.add_argument('--evaluation',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=(pilot_gate(args.directory,io.read_json(args.review),io.read_json(args.qp_review)) if args.command=='gate'
            else risk_report(args.calibration,args.evaluation))
    io.write_json(args.output,result,compressed=args.output.suffix=='.gz')
    print('passed='+str(result.get('passed')) if args.command=='gate' else
        'calibration='+str(result['actual_calibration'])+' evaluation='+str(result['actual_evaluation']))


if __name__=='__main__':main()
