import numpy as np
from . import ev_mobile_v3_design as design
from . import ev_mobile_v3_cases as cases
from .ev_gradient_flow_study import public_curves


class Engine:
    def __init__(self,score):self.score=score;self.calls=[]
    def counters(self):return dict(profile_evaluations=len(self.calls),joint_fit_calls=16*len(self.calls))
    def evaluate(self,u,**kwargs):
        self.calls.append(np.asarray(u).copy())
        return dict(success=True,score=self.score(np.asarray(u)),references=[],pairs=[])


def test_fixed_parent_first_and_preserved_when_no_iterations():
    domain=cases.service_domain(12,3,mobile=True)
    curves=public_curves(12,3);parent=curves['constant'];other=curves['analytic_square']
    engine=Engine(lambda u:0. if np.allclose(u,parent) else -1.)
    u,record=design.projected_policy({},domain,{'square':other,'fixed_parent':parent},
        engine=engine,max_steps=0,total_seconds=10.)
    np.testing.assert_array_equal(engine.calls[0],parent)
    np.testing.assert_array_equal(u,parent)
    assert record['parent_preserved'] and record['operation']['feasible']


def test_milp_proxy_improvement_requires_full_ac_acceptance(monkeypatch):
    from .ev_systematic_design import _realize
    domain=cases.service_domain(12,3,mobile=False);curves=public_curves(12,3)
    parent,new=curves['constant'],curves['analytic_square']
    operation=_realize(domain,new)
    monkeypatch.setattr(design.ac,'projected_context',lambda *a:([],{}))
    monkeypatch.setattr(design.proxy,'solve_projected_milp',lambda *a,**k:
        dict(success=True,power_kw=new,operation_x=operation['x'],exact_score=1000.,best_bound=1000.))
    engine=Engine(lambda u:10. if np.allclose(u,parent) else 9.)
    u,record=design.projected_policy({},domain,{'parent':parent},engine=engine,
        total_seconds=10.,max_steps=1)
    np.testing.assert_array_equal(u,parent)
    assert not record['steps'][0]['accepted']
    assert record['steps'][0]['AC_score']==9.


def test_failed_full_ac_proposal_is_recorded_and_parent_retained(monkeypatch):
    from .ev_systematic_design import _realize
    domain=cases.service_domain(12,3,mobile=False);curves=public_curves(12,3)
    parent,new=curves['constant'],curves['analytic_square'];operation=_realize(domain,new)
    monkeypatch.setattr(design.ac,'projected_context',lambda *a:([],{}))
    monkeypatch.setattr(design.proxy,'solve_projected_milp',lambda *a,**k:
        dict(power_kw=new,operation_x=operation['x']))
    engine=Engine(lambda u:1.)
    original=engine.evaluate
    def evaluate(u,**kwargs):
        return original(u,**kwargs) if np.allclose(u,parent) else dict(success=False,score=None,failure='rival failed')
    engine.evaluate=evaluate
    u,record=design.projected_policy({},domain,{'parent':parent},engine=engine,
        total_seconds=10.,max_steps=1)
    np.testing.assert_array_equal(u,parent)
    assert not record['steps'][0]['AC_profile']['success']
