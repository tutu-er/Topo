"""Integration checks for the new service-fair study before its AC batch."""
import inspect
from types import SimpleNamespace
import numpy as np
from . import ev_flexibility_study as study
from .ev_service_milp import constant_site_plan,check_service_plan
from .simulation import generate,observe,model,BASE_KW
from .information_context import quality_models
from .likelihood import Observations


def test_same_terminal_service_and_mobile_energy_are_explicit():
    c=study.device();fixed=constant_site_plan(0,4,3,c);mobile=constant_site_plan(1,4,3,c)
    assert fixed['success'] and mobile['success']
    a,b=(check_service_plan(x['plan'],c) for x in (fixed,mobile))
    assert np.isclose(fixed['plan'].soc[-1],mobile['plan'].soc[-1])
    assert np.isclose(a['grid_net_pu_hours']*BASE_KW,3.)
    assert b['grid_net_pu_hours']>a['grid_net_pu_hours']
    assert np.isclose(b['grid_net_pu_hours']-a['grid_net_pu_hours'],.01/.95)


def test_worker_window_restores_connection_not_extra_battery():
    c=study.device();expanded=study.device(staff=True)
    assert not constant_site_plan(2,4,3,c)['success']
    ready=constant_site_plan(2,4,3,expanded)
    assert ready['success'] and check_service_plan(ready['plan'],expanded)['feasible']
    assert np.isclose(ready['plan'].soc[-1]*3,1.3425)


def test_complete_ac_future_meters_include_ev_exactly_once():
    path=generate(812445,history_count=12,future_count=4,truth_family=((0,1),))
    h,_=observe(path,slice(0,12));f,_,_=quality_models(h,model(3))
    plan=constant_site_plan(0,4,3,study.device())['plan']
    white={k:np.zeros((4,3) if k in ('p','q','v') else 4) for k in ('p','q','v','master_p','master_q')}
    observed,physical=study.frozen.future_observe(path,plan,f,white)
    assert np.allclose(observed.p,path.p[12:]+plan.p+f.bias_p)
    assert np.allclose(observed.q,path.q[12:]+plan.q+f.bias_q)
    assert np.allclose(observed.v,physical['v']+f.bias_v)
    assert np.allclose(observed.master_p,physical['master_p'])
    assert np.allclose(observed.master_q,physical['master_q'])


def test_planning_interface_excludes_future_truth_and_noise():
    parameters=set(inspect.signature(study.select).parameters)
    assert not parameters.intersection({'path','truth','white','future_physical','true_name'})
    assert {'history','fits','quality'}.issubset(parameters)


def test_history_gate_skips_information_and_keeps_service(monkeypatch):
    h=Observations(p=np.full((12,3),.12),q=np.full((12,3),.04),v=np.ones((12,3)),
        root_v=np.ones(12),master_p=np.full(12,.46),master_q=np.full(12,.15))
    candidates=[SimpleNamespace(name='a'),SimpleNamespace(name='b')]
    fits={'a':SimpleNamespace(success=True,loss=0.),'b':SimpleNamespace(success=True,loss=100.)}
    def forbidden(*args,**kwargs): raise AssertionError('inactive H must not build information')
    monkeypatch.setattr(study,'build_centres',forbidden)
    plans,devices,record=study.select(h,candidates,fits,model(3),model(3),
        {'v_repeat_counts':np.full((4,3),5).tolist()},seed=444)
    assert not record['activated'] and record['inactive_reason']=='H_already_separated'
    assert record['selector_ac_calls']==0
    assert all(check_service_plan(plans[k],devices[k])['feasible'] for k in study.METHODS if k!='none')
    assert plans['quality_charge'].site==0
