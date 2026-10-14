"""Additional read-only replay for voltage-correlation diagnostic evidence.

Every planned ID survives input/replay exceptions in the audit denominator.
The original nominal replay checks quality and joint sigmas independently;
extra checks cover calibrated root and absent channels for stress acquisitions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import averaging_stress as stress
from . import information_replay as nominal_replay


def audit_one(nominal,record,freeze,calibration):
    errors=[]
    nominal_audit=nominal_replay.audit_trial(nominal,calibration=calibration,expected=freeze,require_drawn_quality=True)
    errors.extend('nominal:'+e for e in nominal_audit['errors'])
    alternate=stress.replay_trial(record,nominal)
    errors.extend('stress:'+e for e in alternate['errors'])
    if not record.get('nominal_reconstruction',{}).get('success',False): errors.append('nominal_reconstruction_failed')
    rows={r['method']:r for r in nominal['records']}
    count=0
    for row in record['records']:
        if 'observations' not in row: continue
        obs=row['observations'];original=rows[row['method']]['observations']
        if set(obs)!=set(original): errors.append(row['method']+':observation_schema')
        root=np.asarray(obs.get('root_v'))
        if root.shape!=(4,) or not np.array_equal(root,np.ones(4)): errors.append(row['method']+':calibrated_root')
        if obs.get('root_v')!=original.get('root_v'): errors.append(row['method']+':root_changed')
        for name in ('source_p','source_q'):
            if obs.get(name) is not None or original.get(name) is not None: errors.append(row['method']+':unexpected_source_meter')
        count+=1
    if record['seed'] not in stress.IDS or str(record['seed'])!=record['history_id']: errors.append('fixed_history_identity')
    if record['selector_rerun'] or not record['same_saved_actions'] or record['extra_independent_history_count']!=0 or record['nominal_calibration_risk_transferred']: errors.append('scope_flags')
    return dict(history_id=record['history_id'],success=not errors,errors=errors,nominal_audit=nominal_audit,stress_audit=alternate,stress_root_absent_channel_checks=count)


def run(nominal_root=stress.NOMINAL_ROOT,output=stress.OUTPUT):
    nominal_root,output=Path(nominal_root),Path(output)
    freeze=stress.old.read_json(nominal_root/'freeze.json');cal=stress.old.read_json(nominal_root/'calibration/thresholds.json')
    audits=[]
    for seed in stress.IDS:
        try:
            n=stress.old.read_json(nominal_root/'evaluation'/f'trial_{seed}.json.gz')
            r=stress.old.read_json(output/'raw'/f'trial_{seed}.json.gz')
            a=audit_one(n,r,freeze,cal)
        except Exception as error:
            # Audit exceptions represent retained failed evidence, not sample removal.
            a=dict(history_id=str(seed),success=False,errors=['audit_exception:'+type(error).__name__+':'+str(error)],nominal_audit=None,stress_audit=None,stress_root_absent_channel_checks=0)
        audits.append(a);stress.old.write_json(output/'replay_extra'/f'trial_{seed}.json.gz',a,compressed=True)
    source=nominal_replay.audit_source(freeze['source_sha256'],snapshot=nominal_root/'science_source_snapshot.json.gz')
    summary=dict(schema='correlated_voltage_averaging_extra_replay_v1',success=all(a['success'] for a in audits) and source['success'],planned_paired_history_count=16,retained_paired_history_count=len(audits),extra_independent_history_count=0,failed_replay_count=sum(not a['success'] for a in audits),root_absent_channel_checks=sum(a['stress_root_absent_channel_checks'] for a in audits),full_nominal_quality_and_joint_sigma_audit=True,nominal_paired_outcomes_recomputed=True,true_generation_model_replayed=True,assumed_joint_likelihood_replayed=True,science_source_audit=source,results=[{k:a[k] for k in ('history_id','success','errors','stress_root_absent_channel_checks')} for a in audits])
    stress.old.write_json(output/'replay_extra_summary.json',summary)
    names=('averaging_stress.py','averaging_stress_protocol.md','averaging_stress_checks.py','information_replay.py','gradient_price_evidence.py')
    stress.old.write_json(output/'replay_extra_manifest.json',dict(source_sha256={name:stress.sha(stress.HERE/name) for name in names},raw_extra_replay_kept_local=True,summary_sha256=stress.sha(output/'replay_extra_summary.json'),replay_sha256={str(seed):stress.sha(output/'replay_extra'/f'trial_{seed}.json.gz') for seed in stress.IDS}))
    print(summary['success'],summary['retained_paired_history_count'],summary['failed_replay_count'],summary['root_absent_channel_checks'])
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--nominal-root',type=Path,default=stress.NOMINAL_ROOT);p.add_argument('--output',type=Path,default=stress.OUTPUT);a=p.parse_args();run(a.nominal_root,a.output)


if __name__=='__main__': main()
