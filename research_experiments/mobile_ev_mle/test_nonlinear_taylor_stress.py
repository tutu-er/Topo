"""Lightweight integrity/selection tests; no nonlinear solves or oracle data."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from research_experiments.mobile_ev_mle import nonlinear_taylor_stress as stress


PAIRS=[
    'star_centre_0__clade_01_centre_0__mean__local_refit',
    'star_centre_0__clade_02_centre_0__mean__local_refit',
    'star_centre_0__clade_12_centre_0__mean__local_refit',
    'clade_01_centre_0__clade_02_centre_0__mean__local_refit',
    'clade_01_centre_0__clade_12_centre_0__mean__local_refit',
    'clade_02_centre_0__clade_12_centre_0__mean__local_refit',
    'star_centre_0__clade_02_centre_1__mean__local_refit',
    'star_centre_1__clade_02_centre_1__mean__local_refit',
]


def identity(file):
    return dict(history_hash=file.stem,actions_hash='all_actions_'+file.stem,
        method_control_hashes={method:method+'_control' for method in stress.METHODS})


def result(file):
    return dict(input_trial_filename=file.name,history_hash=file.stem,status='diagnostic_complete_with_failures',
        seconds=9.,diagnostic_pair_names=PAIRS,omitted_pair_names=[],records=[
            dict(method=method,pair=pair,domain=domain,control_hash=method+'_control',
                restarts=[dict(success=False,status=0),dict(success=False,status=0)],
                success=False,best=dict(cost_upper=1.),ac_predict_calls=4)
            for method in stress.METHODS for pair in PAIRS for domain in stress.DOMAINS])


class FakeNLP:
    calls=[]
    @staticmethod
    def source_hashes():return {'nonlinear_pair_audit.py':'identical_source'}
    @classmethod
    def audit_history(cls,history,actions,*,starts,max_nfev,extra_pairs):
        assert (starts,max_nfev,extra_pairs)==(2,300,2)
        cls.calls.append(history)
        if history.stem.endswith('3'):raise RuntimeError('preserved operator failure')
        return result(history)


@pytest.fixture
def six_histories(tmp_path,monkeypatch):
    folder=tmp_path/'input';folder.mkdir()
    for index,error in enumerate((35.,11.,60.,3.,4.,12.)):
        value=dict(selection=dict(methods=dict(refit_milp=dict(actual_rollback=[
            dict(pair_name=PAIRS[1],joint_taylor_error_norm=error,physically_feasible=True)]))),
            truth_topology='poison_must_not_select',final_likelihood=1e100)
        if index==1:value['selection']['methods']['fixed_mle_milp']=dict(actual_rollback=[
            dict(physically_feasible=False,error='unavailable norm, keep denominator')])
        (folder/f'trial_{index+1}.json').write_text(json.dumps(value),encoding='utf-8')
    monkeypatch.setattr(stress,'_nlp',lambda:FakeNLP)
    monkeypatch.setattr(stress,'_inputs',lambda file,nlp:(file,{},identity(file)))
    FakeNLP.calls=[]
    manifest=tmp_path/'manifest.json'
    stress.prepare_manifest(folder,manifest)
    return folder,manifest


def reuse_fixture(tmp_path,file):
    folder=tmp_path/'reused';folder.mkdir()
    cfg=dict(source_sha256=FakeNLP.source_hashes(),starts=2,max_nfev=300,extra_pairs=2,
        methods=stress.METHODS,domains=stress.DOMAINS)
    stress._dump(folder/'config.json',cfg)
    stress._dump(folder/f'audit_{file.stem}.json',result(file))
    return folder


def test_selection_does_not_access_evaluator_fields_and_preserves_missing_norm():
    class PoisonTrial(dict):
        def __getitem__(self,key):
            if key!='selection':raise AssertionError('oracle field accessed')
            return super().__getitem__(key)
    trial=PoisonTrial(selection=dict(methods=dict(refit_milp=dict(actual_rollback=[
        dict(pair_name=PAIRS[3],joint_taylor_error_norm=9.,physically_feasible=True),
        dict(physically_feasible=False,error='saved AC failure')]))))
    view=stress.selection_view(trial)
    assert view['largest_computable']['joint_taylor_error_norm']==9
    assert view['required_rollback_count']==2 and view['computable_rollback_count']==1
    assert len(view['failure_ledger'])==1 and not view['all_required_Taylor_norms_available']


def test_six_history_selection_manifest_is_immutable(six_histories):
    folder,file=six_histories;manifest=stress._load(file)
    assert manifest['selected_file_names']==['trial_3.json','trial_1.json']
    assert len(manifest['all_history_ranking_by_available_norm'])==6
    assert manifest['incomplete_history_error_ordering']
    assert not manifest['statistical_sample']
    with pytest.raises(ValueError,match='never overwrite'):
        stress.prepare_manifest(folder,file)


def test_pair_ordering_preserved_but_cross_centre_missing_is_not_ignored():
    selected=dict(largest_computable=dict(pair='clade_02_centre_0__star_centre_0__mean__local_refit'))
    assert stress.check_worst_pair_scope(dict(diagnostic_pair_names=PAIRS),selected)['centre_zero_base_pair']
    selected['largest_computable']['pair']='star_centre_1__clade_12_centre_1__mean__local_refit'
    with pytest.raises(ValueError,match='missing'):
        stress.check_worst_pair_scope(dict(diagnostic_pair_names=PAIRS),selected)


def test_reuse_rejects_changed_sources_configuration_history_and_controls(tmp_path):
    file=tmp_path/'trial_1.json';folder=reuse_fixture(tmp_path,file)
    cfg=dict(starts=2,max_nfev=300,extra_pairs=2)
    reused,provenance=stress.validate_reuse(folder,file,None,None,identity(file),FakeNLP,cfg)
    assert len(reused['records'])==48 and provenance['status']=='reused_not_new_run'
    with pytest.raises(ValueError,match='configuration'):
        stress.validate_reuse(folder,file,None,None,identity(file),FakeNLP,dict(cfg,max_nfev=301))
    wrong=dict(identity(file),history_hash='different')
    with pytest.raises(ValueError,match='historical'):
        stress.validate_reuse(folder,file,None,None,wrong,FakeNLP,cfg)
    wrong=dict(identity(file),method_control_hashes={method:'different' for method in stress.METHODS})
    with pytest.raises(ValueError,match='control hash'):
        stress.validate_reuse(folder,file,None,None,wrong,FakeNLP,cfg)
    saved=stress._load(folder/'config.json');saved['source_sha256']={'wrong':'hash'}
    stress._dump(folder/'config.json',saved)
    with pytest.raises(ValueError,match='source hashes'):
        stress.validate_reuse(folder,file,None,None,identity(file),FakeNLP,cfg)


def test_one_failed_history_continues_to_hash_bound_reused_history(six_histories,tmp_path):
    source,file=six_histories;folder=reuse_fixture(tmp_path,source/'trial_1.json')
    out=tmp_path/'new';summary=stress.run_stress(file,out,reuse_folder=folder)
    assert len(FakeNLP.calls)==1 and FakeNLP.calls[0].name=='trial_3.json'
    assert summary['history_count']==2 and summary['required_history_failure_count']==1
    assert summary['reused_history_count']==1
    assert stress._load(out/'audit_trial_3.json')['status']=='required_stress_history_failure'
    reused=stress._load(out/'audit_trial_1.json')
    assert reused['reused_not_new_run'] and len(reused['records'])==48
    assert not any(row['success'] for row in reused['records'])


def test_post_manifest_input_change_never_runs_optimizer(six_histories,tmp_path):
    source,file=six_histories;changed=source/'trial_3.json'
    changed.write_text(changed.read_text(encoding='utf-8')+' ',encoding='utf-8')
    folder=reuse_fixture(tmp_path,source/'trial_1.json')
    summary=stress.run_stress(file,tmp_path/'new',reuse_folder=folder)
    assert summary['required_history_failure_count']==1 and not FakeNLP.calls
    assert summary['reused_history_count']==1


def test_real_nested_numpy_types_serialize_and_manifest_identity_stays_stable(tmp_path):
    value=dict(explanations=[dict(beta=np.array([.02,.01]),
        physical=dict(residual=np.float64(3e-15),feasible=np.bool_(True),count=np.int64(4)))],
        step=(np.array([[1.,2.]]),np.float64(1.5)))
    file=tmp_path/'nested.json';stress._dump(file,value)
    restored=stress._load(file)
    assert restored['explanations'][0]['beta']==[.02,.01]
    assert restored['explanations'][0]['physical']==dict(residual=3e-15,feasible=True,count=4)
    assert restored['step']==[[[1.,2.]],1.5]
    native=dict(rule='predeclared',methods=stress.METHODS,cost=1.25)
    expected=json.dumps(native,ensure_ascii=False,indent=2)+'\n'
    stress._dump(file,native)
    assert file.read_text(encoding='utf-8')==expected


def test_successful_new_audit_with_numpy_latents_is_saved_before_reuse(six_histories,tmp_path,monkeypatch):
    source,file=six_histories;folder=reuse_fixture(tmp_path,source/'trial_1.json')
    def numpy_audit(history,actions,**kwargs):
        value=result(history)
        value['records'][0]['best']['explanations']=[dict(beta=np.array([.01,.02]))]
        value['history_context']=dict(centres=np.array([np.float64(1.)]),calls=np.int64(4))
        return value
    monkeypatch.setattr(FakeNLP,'audit_history',numpy_audit)
    out=tmp_path/'numpy_run';summary=stress.run_stress(file,out,reuse_folder=folder)
    assert summary['required_history_failure_count']==0 and summary['history_count']==2
    saved=stress._load(out/'audit_trial_3.json')
    assert saved['records'][0]['best']['explanations'][0]['beta']==[.01,.02]
    assert saved['history_context']==dict(centres=[1.],calls=4)
    assert not saved['reused_not_new_run'] and summary['reused_history_count']==1
