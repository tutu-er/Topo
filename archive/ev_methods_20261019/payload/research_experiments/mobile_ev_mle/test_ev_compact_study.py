"""Incumbent protection, task meaning and selector information boundaries."""
import inspect
import numpy as np
import pytest

from . import ev_compact_study as study


def prediction(values,weights=None,success=True):
    weights=np.full(len(values),1/len(values)) if weights is None else weights
    return dict(success=success,pairs=[dict(added=v,weight=w) for v,w in zip(values,weights)])


def test_capping_retains_negative_evidence_and_limits_easy_task_influence():
    p=prediction([-3.,2.,101.],[.2,.3,.5])
    assert study.task_value(p)==pytest.approx(50.5)
    assert study.task_value(p,20.)==pytest.approx(10.)
    assert study.task_value(p,0.)==pytest.approx(-.6)
    assert study.task_value(prediction([1.],success=False)) is None
    with pytest.raises(ValueError):study.task_value(prediction([1.]),-1.)


def test_incumbent_selection_ignores_incomplete_and_infeasible_predictions():
    rows=[dict(origin='small_domain',operation=dict(feasible=True),prediction=prediction([3.,4.])),
          dict(origin='new_raw',operation=dict(feasible=True),prediction=prediction([1.,2.])),
          dict(origin='bad_physics',operation=dict(feasible=False),prediction=prediction([1e5,1e5])),
          dict(origin='bad_fit',operation=dict(feasible=True),prediction=prediction([1e6],success=False))]
    assert study.choose_checked(rows)['origin']=='small_domain'
    assert study.choose_checked(rows,2.)['origin']=='small_domain'
    assert study.choose_checked(rows[2:]) is None


def test_nested_domain_inheritance_excludes_random_even_if_passed_in_dictionary():
    selected={name:np.ones((6,3))*i for i,name in enumerate(study.DOMAIN_FLAGS)}
    selected['random_mobile']=np.full((6,3),999.)
    ids=[k for k,v in study.inherited_candidates('v2g_mobile',selected)]
    assert ids==['charge_fixed','v2g_fixed','charge_mobile']
    assert 'random_mobile' not in ids


def test_selector_interface_contains_only_history_and_public_inputs():
    names=tuple(inspect.signature(study.select_controls).parameters)
    assert names==('history','bank','hfits','hm','fm','jm','seed','domain','prediction','master_form')
    assert study.CONFIG['selector_uses_truth'] is False
    assert study.CONFIG['selector_uses_future_noise'] is False


def test_all_declared_history_sets_are_disjoint():
    sets=[set(study.CONFIG[k]) for k in
          ('development_seeds','evaluation_seeds','calibration_seeds')]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert all(len(s)%4==0 for s in sets)


def test_tracking_power_headroom_respects_nameplate_and_keeps_constant_service():
    maximum_gain=1+study.CONFIG['ensemble']['tracking_fraction']
    for mobile,discharge in study.DOMAIN_FLAGS.values():
        d=study.service_domain(mobile=mobile,discharge=discharge)
        cfg=d.config
        assert cfg.p_charge_max_kw*maximum_gain<=6.+1e-12
        assert cfg.p_discharge_max_kw*maximum_gain<=6.+1e-12
        u=np.zeros((6,3));u[:,0]=2.
        assert study.base.realize_control(d,u)['success']
    assert study.operations_config()['device']['power_kw']==pytest.approx(6/1.05)
    assert study.CONFIG['physical']['device']['power_kw']==6.


def test_control_key_is_stable_under_solver_roundoff_but_separates_real_actions():
    a=np.array([[1.,-2.,0.]])
    assert study.control_key(a)==study.control_key(a+1e-12)
    assert study.control_key(a)!=study.control_key(a+.001)


def test_full_guard_preserves_a_good_parent_against_a_worse_MILP(monkeypatch):
    from types import SimpleNamespace
    baseline=np.zeros((6,3));baseline[:,0]=2.
    parent=np.zeros((6,3));parent[:2,0]=6.
    raw=np.zeros((6,3));raw[-2:,0]=6.
    domain=study.base.service_domain(mobile=True,discharge=True)
    model=dict(name='a__to__b')
    monkeypatch.setattr(study,'exchange_original_domain',lambda *a,**k:dict(
        u=raw.ravel(),evaluations=[dict(qp_results=[dict(physical_parameter=np.array([1.]))])],
        best_evaluation_index=0))
    monkeypatch.setattr(study,'make_ac_context',lambda *a,**k:None)
    monkeypatch.setattr(study,'evaluate_ac_witness',lambda *a,**k:dict(
        complete_AC_voltage_feasible=True,physical_box_feasible=True,max_channel_RMS=0.))
    def score(power):return prediction([float(power[0,0])])
    chosen,record=study._optimize_domain('v2g_mobile',domain,
        [('constant',baseline),('charge_fixed',parent)], [model],[object()],[0.],[1.],
        SimpleNamespace(n=3),[],{},None,None,score,master_form='shared_control_square')
    assert np.allclose(chosen,parent)
    assert record['selected_origin']=='charge_fixed'
    assert record['inheritance_protected']
    assert record['selected_score']>=record['initial_best_score']
    assert not record['guard']['accepted']
