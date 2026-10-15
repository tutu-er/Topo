"""Source-free quadratic baseline: exact tiny optima and operation invariants."""
import builtins
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_rx_precision as precision
from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain
from .ev_original_domain_milp import evaluate_original_pair


OPTIONS=dict(backend='scipy',max_iterations=4,control_segments=2,residual_segments=2,
    max_segments=8,solver_seconds=2.,total_seconds=10.,absolute_gap=1e-7,relative_gap=0.)


def tiny(*,discharge=False):
    return build_mobile_service_domain(MobileServiceConfig(n_slots=2,n_sites=1,dt_hours=1.,
        p_charge_max_kw=1.,p_discharge_max_kw=1.,allow_discharge=discharge,
        battery_capacity_kwh=10.,initial_soc=.5,soc_min=0.,soc_max=1.,
        terminal_energy_kwh=5. if discharge else 6.,eta_charge=1.,eta_discharge=1.,
        mobile=False,max_moves=0,max_throughput_kwh=2. if discharge else 1.))


def test_source_free_pair_is_exact_sum_with_no_half_and_QP_handles_empty_source():
    weights=np.array([[.25],[1.]])
    pair=precision.squared_precision_pair(weights)
    assert pair.A0.shape==(2,0) and pair.At.shape==(2,2,0)
    assert pair.shift.shape==(0,2) and pair.lower.shape==(0,)
    u=np.array([-1.,1.]);r=pair.residual(u,np.zeros(0))
    assert r@r==pytest.approx(1.25)
    qp=evaluate_original_pair(pair,u)
    assert qp['success'] and qp['primal_upper']==pytest.approx(1.25)
    assert qp['tangent_lower_numeric']<=1.25<=qp['primal_upper']+1e-12


def test_uniform_purecharge_linear_tie_has_strict_square_preference_and_valid_gap():
    d=tiny();flat=np.array([[.5],[.5]])
    out=precision.solve_squared_precision(d,SimpleNamespace(v_sigma=.001),
        initial_powers={'constant':flat},exchange_options=OPTIONS)
    assert out['success'] and out['operation']['audit']['feasible'],out
    # Sum u is fixed at one: linear prices tie. Exact corner Q=1 exceeds .5.
    assert out['power_kw'].sum()==pytest.approx(flat.sum(),abs=1e-8)
    assert np.sum(flat**2)==pytest.approx(.5)
    assert out['Q']==pytest.approx(1.,abs=1e-7),out
    assert out['Q']>np.sum(flat**2)
    assert out['completed'],out
    assert out['objective_lower']<=1.+1e-9<=out['objective_upper']+1e-9
    assert out['gap']<1e-6
    assert out['exchange']['score_scale']==1.
    assert out['inheritance_protected']


def test_zero_gradient_start_finds_tiny_bidirectional_exact_corners():
    d=tiny(discharge=True)
    out=precision.solve_squared_precision(d,SimpleNamespace(v_sigma=np.array([[2.],[1.]])),
        initial_powers=[np.zeros((2,1))],exchange_options=OPTIONS)
    assert out['success'] and out['completed'],out
    # Feasibility imposes u0+u1=0, |u|<=1. Both extreme points have Q=1.25.
    assert out['Q']==pytest.approx(1.25,abs=1e-7)
    assert out['objective_lower']<=1.25+1e-9<=out['objective_upper']+1e-9
    assert np.sort(out['power_kw'].ravel())==pytest.approx([-1.,1.],abs=1e-7)
    assert out['source_coordinates']==0 and not out['full_curve_enumeration']


def test_mobile_joint_operation_uses_precise_site_and_preserves_domain():
    d=build_mobile_service_domain(MobileServiceConfig(n_slots=4,n_sites=2,dt_hours=1.,
        p_charge_max_kw=2.,allow_discharge=False,battery_capacity_kwh=20.,initial_soc=.25,
        soc_min=0.,soc_max=1.,terminal_energy_kwh=7.,eta_charge=1.,eta_discharge=1.,
        mobile=True,travel_slots=1,setup_slots=1,travel_energy_kwh=.1,
        travel_cost=1.,setup_cost=1.,driver_budget=2.,max_moves=1,
        max_throughput_kwh=3.,min_arrival_throughput_kwh=.1))
    rows=len(d.model.rows)
    out=precision.solve_squared_precision(d,SimpleNamespace(v_sigma=np.tile([.002,.001],(4,1))),
        exchange_options={**OPTIONS,'absolute_gap':.01,'relative_gap':.01})
    assert out['success'],out
    decoded=d.decode(out['operation']['x'])
    assert d.check(out['operation']['x'])['feasible']
    assert np.max(out['power_kw'][:,1])>1.9
    assert any(a.kind=='travel' for a in decoded['selected_arcs'])
    assert decoded['energy_kwh'][-1]==pytest.approx(7.,abs=1e-7)
    assert out['Q']>4. and out['objective_lower']<=out['Q']+1e-9
    assert out['objective_upper'] is None or out['objective_upper']>=out['Q']-1e-7
    assert len(d.model.rows)==rows


def test_baseline_never_imports_AC_training_forecasts_or_reads_other_meters(monkeypatch):
    original=builtins.__import__
    forbidden=('likelihood','ev_fixed_train_fit','ev_fixed_train_design','ev_dynamic_study','ev_dynamic_loads')
    def guarded(name,*args,**kwargs):
        if any(part in name for part in forbidden):raise AssertionError('forbidden science import: '+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    importlib.reload(precision)
    class Meter:
        v_sigma=.001
        def __getattr__(self,name):raise AssertionError('only V precision is public: '+name)
    out=precision.solve_squared_precision(tiny(),Meter(),exchange_options=OPTIONS)
    assert out['success'] and out['uses_AC'] is False and out['uses_line_parameters'] is False
    assert not out['uses_forecast'] and not out['uses_truth'] and not out['uses_future_noise']


def test_invalid_sigma_owned_scaling_and_infeasible_initial_are_explicit():
    d=tiny()
    with pytest.raises(ValueError,match='positive and finite'):
        precision.solve_squared_precision(d,SimpleNamespace(v_sigma=0.))
    with pytest.raises(ValueError,match='owns exchange'):
        precision.solve_squared_precision(d,SimpleNamespace(v_sigma=.001),exchange_options={'score_scale':.5})
    out=precision.solve_squared_precision(d,SimpleNamespace(v_sigma=.001),
        initial_powers={'bad':np.full((2,1),2.),'constant':np.full((2,1),.5),'unavailable':None},
        exchange_options=OPTIONS)
    assert out['success'] and out['inheritance_protected']
    assert next(r for r in out['initials'] if r['origin']=='bad')['success'] is False
    assert next(r for r in out['initials'] if r['origin']=='unavailable')['reason']=='unavailable_parent'


def test_exchange_failure_retains_own_scored_feasible_initial(monkeypatch):
    def failure(*a,**k):raise RuntimeError('retained failure fixture')
    monkeypatch.setattr(precision,'exchange_original_domain',failure)
    out=precision.solve_squared_precision(tiny(),SimpleNamespace(v_sigma=.001),
        initial_powers=[np.full((2,1),.5)],exchange_options=OPTIONS)
    assert out['success'] and out['Q']==pytest.approx(.5)
    assert out['unfinished'] and not out['completed'] and out['objective_upper'] is None
    assert any(r['kind']=='exchange_exception' for r in out['failures'])
