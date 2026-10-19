"""Policy isolation and prospective sample/operation contracts."""
import inspect
import numpy as np
from . import study
from .config import CONFIG
from ..ev_mobile_service_domain import MobileServiceConfig
from ..ev_mobile_v4_domain import build_nested_domains
from .coefficients import QuadraticTask


def test_selection_signature_has_no_truth_or_future_simulator():
    assert set(inspect.signature(study.select).parameters)=={
        'observed','fit_options','config','seed','progress'}
    for phase in ('development','evaluation'):
        specifications=study.specifications(phase)
        assert len({s['seed'] for s in specifications})==len(specifications)
    assert set(CONFIG['development_seeds']).isdisjoint(CONFIG['evaluation_seeds'])
    evaluation=study.specifications('evaluation')
    assert {(s['bank'],s['family_index']) for s in evaluation}=={
        (bank,family) for bank in ('n3','n4') for family in range(4)}


def test_families_keep_original_parent_operation_without_extra_completion(monkeypatch):
    domains=build_nested_domains(MobileServiceConfig(n_slots=2,n_sites=2,dt_hours=1.,
        battery_capacity_kwh=10.,initial_soc=.5,terminal_energy_kwh=7.,
        soc_min=0.,soc_max=1.,eta_charge=1.,eta_discharge=1.,p_charge_max_kw=2.,
        p_discharge_max_kw=2.,setup_slots=0))
    # Complete one legal public curve once. The family must only lift it.
    domain=domains['fixed_charge'];model=domain.model.clone()
    for i,v in zip(domain.indices.u.ravel(),[1.,0.,1.,0.]):model.add_row({int(i):1},v,v)
    solved=model.solve(time_limit=2.)
    assert solved.x is not None
    base=np.asarray(solved.x);seen=[]
    def wrapper(d,tasks,parents,**kwargs):
        assert all(d.check(x)['feasible'] for x in parents.values())
        assert kwargs['total_seconds']==CONFIG['total_seconds_per_domain']
        seen.append(list(parents))
        x=next(iter(parents.values()))
        return dict(operation_x=x,score=tasks[0].value(d.decode(x)['u_kw']))
    monkeypatch.setattr(study,'solve_tasks',wrapper)
    monkeypatch.setattr(study.operation,'complete_curve',lambda *a,**k:(_ for _ in ()).throw(
        AssertionError('family must not re-complete parent powers')))
    task=QuadraticTask('audit',0.,np.zeros((2,2)),np.ones((2,2)))
    plans,logs,_=study.plan_family(domains,[task],{'constant':base},CONFIG)
    assert len(plans)==len(logs)==len(seen)==4
    assert 'parent_fixed_bidir' in seen[-1] and 'parent_mobile_charge' in seen[-1]
