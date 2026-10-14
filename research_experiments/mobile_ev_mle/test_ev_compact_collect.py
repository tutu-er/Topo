"""Saved-record integration only; no fitting or action selection."""
from copy import deepcopy
import hashlib,json
import pytest
from . import ev_compact_collect as collect
from . import ev_policy_calibration as cal


def setup_dirs(tmp_path):
    config=dict(development_seeds=list(range(989101,989105)),
        evaluation_seeds=list(range(989201,989213)),calibration_seeds=list(range(989301,989337)))
    payload=dict(config=config,prediction='ensemble',master_form='shared_control_square',
        domain='asset',source_sha256={'selector.py':'1'*64})
    policy_id=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    directories={}
    for phase in ('development','calibration','evaluation'):
        directory=tmp_path/phase;directory.mkdir();directories[phase]=directory
        freeze=dict(**payload,policy_config=payload,policy_id=policy_id,phase=phase,
            seeds=config[phase+'_seeds'],frozen_before_trials=True)
        collect.io.write_json(directory/'freeze.json',freeze)
    return directories,payload,policy_id


def save_trial(directory,payload,policy_id,seed,truth='star',gap=0.):
    names=collect.DECLARED_CANDIDATES
    fits=lambda losses:[dict(name=a,success=True,loss=losses[a]) for a in names]
    losses={a:0. for a in names};losses[truth]=gap
    row=dict(**payload,policy_config=payload,policy_id=policy_id,phase=directory.name,
        seed=seed,true_name=truth,candidate_names=list(names),historical_failures={},
        historical_fits=fits({a:0. for a in names}),proposal={'bank':[dict(name=a) for a in names]},
        rows=[dict(method=m,status='complete',candidate_miss=False,losses=losses,fits=fits(losses))
              for m in collect.METHODS],selection=dict(random_used_by_primary=False,
            uses_truth=False,uses_future_noise=False,methods={m:dict(inheritance_protected=True)
                for m in collect.METHODS}))
    path=directory/f'trial_{seed}.json.gz';collect.io.write_json(path,row,compressed=True)
    return path,row


def test_zero_calibration_records_produce_infinity_and_keep_every_candidate(tmp_path):
    d,p,i=setup_dirs(tmp_path);save_trial(d['evaluation'],p,i,989201,gap=1000.)
    out=collect.risk_report(d['calibration'],d['evaluation'])
    assert out['actual_calibration']==0 and out['actual_evaluation']==1
    for group in ('individual_policies','primary_family'):
        c=out[group]['calibration'];e=out[group]['evaluation']
        assert c['histories_total']==0 and c['histories_pending']==36
        for candidate in c['thresholds']['declared_asset_reference'].values():
            assert all(q['infinite'] for q in candidate.values())
        assert all(m['retained']==list(collect.DECLARED_CANDIDATES)
                   for m in e['records'][0]['methods'].values())


def test_thirty_six_candidate_specific_calibration_and_primary_family(tmp_path):
    d,p,i=setup_dirs(tmp_path)
    for index,seed in enumerate(range(989301,989337)):
        save_trial(d['calibration'],p,i,seed,collect.DECLARED_CANDIDATES[index%4],index//4)
    save_trial(d['evaluation'],p,i,989201,'star',9.)
    out=collect.risk_report(d['calibration'],d['evaluation'])
    assert out['actual_calibration']==36
    c=out['primary_family']['calibration']
    assert c['spec']['familywise']
    for candidate in c['thresholds']['declared_asset_reference'].values():
        assert candidate['__family__']['n']==9
        assert candidate['__family__']['rank']==9
        assert candidate['__family__']['threshold']==8
    assert out['primary_family']['evaluation']['totals']['any_method_false_rejections']==1


@pytest.mark.parametrize('change',[
    lambda r:r.update(phase='development'),lambda r:r.update(seed=0),
    lambda r:r.update(policy_id='bad')])
def test_bad_saved_phase_seed_or_policy_rejected(tmp_path,change):
    d,p,i=setup_dirs(tmp_path);path,row=save_trial(d['evaluation'],p,i,989201)
    change(row);collect.io.write_json(path,row,compressed=True)
    with pytest.raises(ValueError):collect.risk_report(d['calibration'],d['evaluation'])


def test_independent_gate_binds_exact_reviewed_inputs(tmp_path):
    d,p,i=setup_dirs(tmp_path)
    for seed in range(989101,989105):save_trial(d['development'],p,i,seed)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d['development'].glob('trial_*.json.gz')}
    reviewed=dict(success=True,policy_id=i,seeds=list(range(989101,989105)),trial_sha256=hashes)
    assert collect.pilot_gate(d['development'],reviewed)['passed']
    stale=deepcopy(reviewed);stale['policy_id']='other'
    assert not collect.pilot_gate(d['development'],stale)['passed']
    stale=deepcopy(reviewed);stale['trial_sha256'][next(iter(hashes))]='0'*64
    assert not collect.pilot_gate(d['development'],stale)['passed']
    stale=deepcopy(reviewed);stale['seeds']=stale['seeds'][:3]
    assert not collect.pilot_gate(d['development'],stale)['passed']
