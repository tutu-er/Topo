"""Execution boundaries and checkpoint behavior, independent of optimizer score."""
import json
import numpy as np
import pytest
from research_experiments.mobile_ev_mle import adaptive_milp_study as study,likelihood,symmetric_ac_local

def test_selector_restores_physical_call_hooks_when_context_raises(monkeypatch):
    old=likelihood.ac_predict;old_symmetric=symmetric_ac_local.ac_predict
    def failed(*a,**k):raise RuntimeError('context defect')
    monkeypatch.setattr(study,'_select_plans_impl',failed)
    with pytest.raises(RuntimeError,match='context defect'):study.select_plans(None,None,None,None)
    assert likelihood.ac_predict is old and symmetric_ac_local.ac_predict is old_symmetric

def test_joint_counter_includes_failed_candidate_work_and_restores(monkeypatch):
    original=likelihood.ac_predict
    def physical(*a,**k):raise RuntimeError('failed AC')
    monkeypatch.setattr(likelihood,'ac_predict',physical)
    def fake_bank(*a,**k):
        for _ in range(3):
            try:likelihood.ac_predict()
            except RuntimeError:pass
        return {},{'candidate':'failed AC'}
    monkeypatch.setattr(study,'fit_bank',fake_bank)
    fits,failures,calls=study.counted_joint_fit_bank()
    assert not fits and failures and calls==3
    assert likelihood.ac_predict is physical
    monkeypatch.setattr(likelihood,'ac_predict',original)

def test_expired_budget_writes_resumable_empty_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'source_hashes',lambda:{'fake.py':'scope'})
    def should_not_run(*a,**k):raise AssertionError('deadline must stop before trial')
    monkeypatch.setattr(study,'one_trial',should_not_run)
    _,rows=study.run_batch(tmp_path,phase='pilot',count=3,start_seed=1,deadline_utc='2000-01-01T00:00:00+00:00')
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert rows==[] and summary['history_count']==0
    assert summary['stop_reason']=='wall_clock_budget_checkpoint'
    assert summary['config']['count']==3

def test_resumption_rejects_source_change_before_sampling(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'source_hashes',lambda:{'fake.py':'old'})
    study.run_batch(tmp_path,phase='pilot',count=0,start_seed=1)
    monkeypatch.setattr(study,'source_hashes',lambda:{'fake.py':'changed'})
    with pytest.raises(ValueError,match='mix different'):
        study.run_batch(tmp_path,phase='pilot',count=0,start_seed=1)
