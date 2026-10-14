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
    assert collect.pilot_gate(d['development'],reviewed,deepcopy(reviewed))['passed']
    stale_qp=deepcopy(reviewed);stale_qp['policy_id']='other'
    assert not collect.pilot_gate(d['development'],reviewed,stale_qp)['passed']


def test_each_raw_is_read_and_adapted_once_then_offline_replay(tmp_path,monkeypatch):
    from collections import Counter
    from pathlib import Path
    d,p,i=setup_dirs(tmp_path)
    for seed in (989301,989302,989303):save_trial(d['calibration'],p,i,seed)
    for seed in (989201,989202):save_trial(d['evaluation'],p,i,seed)
    reads=Counter();adapts=Counter();read_original=Path.read_bytes;adapt_original=cal.adapt_saved_trial
    def counted_read(path):
        if path.name.startswith('trial_'):reads[str(path)]+=1
        return read_original(path)
    def counted_adapt(trial,*args,**kwargs):
        adapts[trial['seed']]+=1
        return adapt_original(trial,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',counted_read)
    monkeypatch.setattr(cal,'adapt_saved_trial',counted_adapt)
    out=collect.risk_report(d['calibration'],d['evaluation'])
    assert len(reads)==len(adapts)==5 and set(reads.values())==set(adapts.values())=={1}
    assert len(out['standard_records']['calibration'])==3
    assert len(out['raw_manifest']['evaluation'])==2
    assert all('fits' not in method for r in out['standard_records']['calibration'] for method in r['methods'].values())
    bundle=json.loads(json.dumps(collect.io.serial(out['standard_records'])))
    def forbidden(*a,**kwargs):raise AssertionError('raw reread during offline statistical replay')
    monkeypatch.setattr(collect,'_read_trial',forbidden)
    replay=collect.risk_report_from_standard_records(bundle)
    for group in ('individual_policies','primary_family'):
        assert replay[group]==out[group]
    bundle['calibration'][0]['methods'][collect.PRIMARY[0]]['status']='tampered'
    with pytest.raises(ValueError,match='hash mismatch'):collect.risk_report_from_standard_records(bundle)


def test_previous_raw_root_is_released_before_next_load(tmp_path,monkeypatch):
    import weakref
    d,p,i=setup_dirs(tmp_path)
    for seed in (989301,989302):save_trial(d['calibration'],p,i,seed)
    save_trial(d['evaluation'],p,i,989201)
    original=collect._read_trial;last=[];observations=[]
    class RawRecord(dict):pass
    def watched(path):
        if last:assert last[-1]() is None,'previous complete raw still retained'
        value,manifest=original(path)
        value=RawRecord(value);last.append(weakref.ref(value));observations.append(path.name)
        return value,manifest
    monkeypatch.setattr(collect,'_read_trial',watched)
    out=collect.risk_report(d['calibration'],d['evaluation'])
    assert len(observations)==3 and all(r() is None for r in last)
    assert out['actual_calibration']==2


def test_gate_projection_releases_raw_and_excludes_large_arrays(tmp_path,monkeypatch):
    import weakref
    d,p,i=setup_dirs(tmp_path)
    for seed in (989101,989102):save_trial(d['development'],p,i,seed)
    freeze=collect.io.read_json(d['development']/'freeze.json')
    original=collect._read_trial;last=[]
    class RawRecord(dict):pass
    def watched(path):
        if last:assert last[-1]() is None
        value,manifest=original(path);value=RawRecord(value)
        value['giant_unused_state']=[0.]*10000
        last.append(weakref.ref(value));return value,manifest
    monkeypatch.setattr(collect,'_read_trial',watched)
    manifest=[];out=collect.checked_trials(d['development'],freeze,'development',manifest=manifest)
    assert len(out)==len(manifest)==2 and all(r() is None for r in last)
    assert all('giant_unused_state' not in r and 'historical_fits' not in r for r in out)


def test_streaming_statistics_equal_previous_two_adaptation_definition(tmp_path):
    d,p,i=setup_dirs(tmp_path);craw=[];eraw=[]
    for index,seed in enumerate(range(989301,989337)):
        path,row=save_trial(d['calibration'],p,i,seed,collect.DECLARED_CANDIDATES[index%4],index//4)
        if index==0:
            # Preserve even inconsistent per-method miss metadata exactly as
            # the previous separately adapted PRIMARY would interpret it.
            next(r for r in row['rows'] if r['method']=='quality_mobile')['candidate_miss']=True
        if index==1:
            method=next(r for r in row['rows'] if r['method']==collect.PRIMARY[0])
            method.update(status='fit_failure',fits=[],losses={})
        collect.io.write_json(path,row,compressed=True);craw.append(row)
    for index,seed in enumerate((989201,989202)):
        path,row=save_trial(d['evaluation'],p,i,seed,'star',9.)
        if index==0:
            next(r for r in row['rows'] if r['method']==collect.PRIMARY[1]).update(status='operation_failure',fits=[],losses={})
        collect.io.write_json(path,row,compressed=True);eraw.append(row)
    out=collect.risk_report(d['calibration'],d['evaluation'])
    cf=collect.io.read_json(d['calibration']/'freeze.json');ef=collect.io.read_json(d['evaluation']/'freeze.json')
    for label,methods,familywise in (('individual_policies',collect.METHODS,False),('primary_family',collect.PRIMARY,True)):
        spec=cal.spec_from_freeze(cf,reference_id='declared_lv_asset_reference_v1',
            candidates=collect.DECLARED_CANDIDATES,methods=methods,alpha=.1,familywise=familywise)
        cdata=[cal.adapt_saved_trial(r,cf,spec) for r in craw]
        edata=[cal.adapt_saved_trial(r,ef,spec) for r in eraw]
        threshold=cal.calibrate(cdata,spec,expected_history_ids=[str(r['seed']) for r in craw],
            planned_history_ids=[str(s) for s in cf['seeds']])
        evaluated=cal.evaluate(edata,threshold,expected_history_ids=[str(r['seed']) for r in eraw])
        assert out[label]==dict(calibration=threshold,evaluation=evaluated)


def test_duplicate_planned_seed_and_raw_change_rejected(tmp_path):
    d,p,i=setup_dirs(tmp_path);path,row=save_trial(d['calibration'],p,i,989301)
    collect.io.write_json(d['calibration']/'trial_duplicate.json.gz',row,compressed=True)
    with pytest.raises(ValueError,match='duplicate'):collect.risk_report(d['calibration'],d['evaluation'])


def test_raw_change_during_projection_rejected(tmp_path):
    d,p,i=setup_dirs(tmp_path);path,row=save_trial(d['development'],p,i,989101)
    freeze=collect.io.read_json(d['development']/'freeze.json')
    def mutate(trial):
        path.write_bytes(path.read_bytes()+b'\0')
        return {'seed':trial['seed']}
    with pytest.raises(ValueError,match='changed during projection'):
        collect.checked_trials(d['development'],freeze,'development',projector=mutate)
