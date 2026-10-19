"""Thin-review adaptation tests, with no fitting or action selection."""
from copy import deepcopy
import pytest
from . import ev_compact_location_review as r
from . import ev_compact_location_audit as a
from . import test_ev_compact_location_audit as inputs


def parent():
    t=inputs.fixture()
    t['config']['physical']['device'].update(dt_hours=.25,setup_slots=1,travel_slots=1,drive_kwh=.2,
        max_moves=1,travel_cost=15.,setup_cost=15.,driver_budget=30.,cost_units='worker_minutes')
    return t


def test_adapter_keeps_parent_bytes_and_uses_derated_signed_mobile_operations():
    t=parent();inputs.stationary(t);saved=a.process_history(t)
    t['operations_config']['device']['power_kw']=5.
    s,p=r.replay_views(saved,t)
    assert p['config']['device']['power_kw']==5. and t['config']['physical']['device']['power_kw']==6.
    assert set(s['counts'])=={'candidate_fit_calls','fit_AC_calls','forward_AC_calls'}
    assert 'actual_returned_fit_calls' in saved['counts']
    assert s['conditions'][0]['resource_scope']['remap_has_no_operation_feasibility_or_equal_service_claim']
    assert 'remap_has_no_operation_feasibility_or_equal_service_claim' not in saved['conditions'][0]['resource_scope']
    for method in r.METHODS:
        config=r.signed_mobile_config(p['config'],method)
        assert config.mobile and config.allow_discharge and config.p_charge_max_kw==5.


def test_reused_review_receives_compact_method_names_and_preserves_zero_location_delta(monkeypatch):
    t=parent();inputs.stationary(t);saved=a.process_history(t)
    saved['parent_trial_sha256']='bytes';capture={}
    def replay(s,p,sha,**kwargs):
        capture.update(kwargs)
        assert sha=='bytes' and kwargs['methods']==r.METHODS
        return dict(counts=s['counts'],saved_fit_states_replayed=36,physical_states_replayed=4,
            conditions=[dict(new_remap=False) for _ in r.METHODS]),[
                dict(method=m,true_candidate='star') for m in r.METHODS]
    monkeypatch.setattr(r.base,'replay_history',replay)
    audit,candidates=r.replay_history(saved,t,'bytes')
    assert audit['new_fit_states_replayed']==0
    assert len(audit['solver_state_metadata'])==32
    assert all(x['voltage_max_abs_difference_pu']==0. for x in audit['physical_responses'])
    assert capture['service_config_builder'] is r.signed_mobile_config


def test_reviewer_rejects_calibration_claim_on_the_new_location_process(monkeypatch):
    t=parent();inputs.stationary(t);saved=a.process_history(t);saved['statistical_threshold_applied']=True
    monkeypatch.setattr(r.base,'replay_history',lambda *args,**kwargs:({},[]))
    with pytest.raises(ValueError,match='calibration'):
        r.replay_history(saved,t,'bytes')
