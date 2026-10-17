"""Independent endpoint, repair and original-operation regression tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from research_experiments.mobile_ev_mle.ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from research_experiments.mobile_ev_mle.g_precision.coefficients import QuadraticTask
from research_experiments.mobile_ev_mle.full_pipeline.solver import solve_tasks, repair_operation, operation_diagnostics
from research_experiments.mobile_ev_mle.full_pipeline import solver as implementation


def domain(mobile=False):
    return build_mobile_service_domain(MobileServiceConfig(n_slots=2,n_sites=2,
        dt_hours=1.,p_charge_max_kw=2.,p_discharge_max_kw=2.,allow_discharge=False,
        battery_capacity_kwh=10.,initial_soc=.5,soc_min=0.,soc_max=1.,
        eta_charge=1.,eta_discharge=1.,terminal_energy_kwh=7.,mobile=mobile,
        travel_slots=1,setup_slots=0,travel_energy_kwh=0.,max_moves=1))


def complete(d,power):
    model=d.model.clone()
    for i,v in zip(d.indices.u.ravel(),np.asarray(power).ravel()):
        model.add_row({int(i):1.},float(v),float(v))
    result=model.solve(time_limit=2.)
    assert result.x is not None
    x=np.asarray(result.x[:d.model.n_variables])
    assert d.check(x)['feasible']
    return x


def solve(d,tasks,parents,**kwargs):
    opts=dict(iterations=3,segments=4,max_segments=16,solver_seconds=2.,total_seconds=8.,
        backend='scipy',relative_gap=1e-6,absolute_gap=1e-7)
    opts.update(kwargs)
    return solve_tasks(d,tasks,parents,**opts)


def test_affine_quadratic_endpoint_and_valid_bounds():
    d=domain();x=complete(d,[[1,0],[1,0]])
    task=QuadraticTask('ab',.4,np.array([[.8,0],[.2,0]]),np.array([[.25,0],[.5,0]]))
    out=solve(d,[task],{'constant':x})
    assert out['score']==pytest.approx(3.,abs=1e-7)
    assert out['objective_lower']<=3.<=out['objective_upper']
    assert out['completed'] and out['inheritance_protected']
    assert out['operation_check']['feasible']
    assert not out['full_route_or_curve_enumeration']


def test_zero_g_and_optional_motion():
    d=domain(True);x=complete(d,[[1,0],[1,0]])
    task=QuadraticTask('remote',0,np.zeros((2,2)),np.array([[0,0],[0,1]]))
    out=solve(d,[task],{'constant':x})
    assert out['score']==pytest.approx(4.,abs=1e-7)
    assert out['operation_check']['moves']==1
    assert np.asarray(out['power_kw'])[1,1]==pytest.approx(2.)
    assert out['objective_lower']<=4.<=out['objective_upper']


def test_all_competitors_and_nonendpoint_maxmin():
    d=domain();x=complete(d,[[1,0],[1,0]])
    tasks=[QuadraticTask('first',0,np.zeros((2,2)),np.array([[1,0],[0,0]])),
        QuadraticTask('second',0,np.zeros((2,2)),np.array([[0,0],[1,0]]))]
    out=solve(d,tasks,{'parent':x})
    assert out['score']==pytest.approx(1.)
    assert out['objective_lower']<=1.<=out['objective_upper']
    assert len(out['task_values'])==2


def test_diagnostics_locate_negative_continuous_power():
    d=domain();x=complete(d,[[2,0],[0,0]])
    x[d.indices.charge[1,0]]=-6.85e-7
    audit=operation_diagnostics(d,x)
    assert not audit['feasible']
    assert audit['box_violation']==pytest.approx(6.85e-7)
    assert any(v['kind']=='charge' and v['coordinates']==[1,0] for v in audit['variable_violations'])
    assert audit['linear_row_violations']


def test_fixed_integer_repair_uses_original_domain_and_replays_energy():
    d=domain();x=complete(d,[[2,0],[0,0]])
    original=x.copy();x[d.indices.charge[1,0]]=-6.85e-7
    x[d.indices.u[1,0]]=-6.85e-7
    fixed,receipt=repair_operation(d,x,time_limit=2.)
    assert receipt['success']
    assert not receipt['before']['feasible'] and receipt['after']['feasible']
    assert receipt['all_original_constraints_included']
    integers=np.asarray(d.model.integer,bool)
    np.testing.assert_array_equal(fixed[integers],original[integers])
    assert receipt['max_change']<1e-5
    assert d.check(fixed)['feasible']


def test_invalid_integer_pattern_is_not_silently_changed_to_another_route():
    d=domain();x=complete(d,[[1,0],[1,0]])
    x[d.indices.arc]=0
    fixed,receipt=repair_operation(d,x,time_limit=2.)
    assert fixed is None and not receipt['success']


def test_repair_keeps_raw_failure_and_scores_original_polynomial(monkeypatch):
    d=domain();parent=complete(d,[[1,0],[1,0]])
    raw=complete(d,[[2,0],[0,0]])
    raw[d.indices.charge[1,0]]=-6.85e-7
    raw[d.indices.u[1,0]]=-6.85e-7
    def fake(*args,**kwargs):
        return raw,dict(objective_upper=4.000001,status=0,message='injected numerical incumbent')
    monkeypatch.setattr(implementation,'_master',fake)
    task=QuadraticTask('square',0,np.zeros((2,2)),np.ones((2,2)))
    out=solve(d,[task],{'constant':parent},iterations=1)
    assert out['score']==pytest.approx(4.)
    assert out['selected_origin']=='repaired:0'
    assert out['operation_check']['feasible']
    assert out['failures'][0]['kind']=='raw_master_operation_rejected'
    assert out['iterations'][0]['operation_repair']['success']
    assert out['iterations'][0]['raw_operation_x']==raw.tolist()
    assert out['objective_lower']<=4.<=out['objective_upper']


def test_parent_is_preserved_under_total_budget_and_zero_tasks():
    d=domain();x=complete(d,[[1,0],[1,0]])
    zero=QuadraticTask('zero',0,np.zeros((2,2)),np.zeros((2,2)))
    out=solve(d,[zero],{'constant':x},total_seconds=1e-10)
    assert out['score']==0 and out['inheritance_protected']
    assert out['operation_check']['feasible']
    assert out['objective_lower']<=0<=out['objective_upper']


def test_strict_gurobi_backend_and_replay():
    pytest.importorskip('gurobipy')
    d=domain(True);x=complete(d,[[1,0],[1,0]])
    task=QuadraticTask('precision',0,np.zeros((2,2)),np.ones((2,2)))
    out=solve(d,[task],{'constant':x},backend='gurobi')
    assert out['success'] and out['operation_check']['feasible']
    detail=out['iterations'][0]['backend_details']
    assert detail['FeasibilityTol']==detail['IntFeasTol']==1e-9
    assert detail['threads']==1
    assert out['score']==pytest.approx(4.)
