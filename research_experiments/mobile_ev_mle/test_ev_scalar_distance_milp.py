"""Analytic source profiles, valid chords, operation MILP and bound direction."""
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_scalar_distance_milp as scalar
from .ev_original_domain_milp import OriginalDomainPair
from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain


def fixture_context(T=2,cvalue=-.2,*,source=False):
    n=1;m=T*3;c=np.zeros(m);B=np.zeros((m,T));G=np.zeros((m,2*T))
    for t in range(T):
        c[t]=cvalue;B[t,t]=1.
        if source:G[t,2*t]=1.
    pairs=[]
    for name,sign in [('a_to_b',1.),('b_to_a',-1.)]:
        pairs.append(OriginalDomainPair(name,sign*c,sign*B,sign*G,np.zeros((T,m,2*T)),
            np.full(2*T,-.25),np.full(2*T,.25),np.zeros(2*T),np.zeros((2*T,T))))
    return dict(success=True,bank=[SimpleNamespace(name='a'),SimpleNamespace(name='b')],
        control_shape=(T,n),pairs=pairs)


def fixture_domain(T=2,*,connected=True):
    return build_mobile_service_domain(MobileServiceConfig(n_slots=T,n_sites=1,
        dt_hours=1.,p_charge_max_kw=1.,p_discharge_max_kw=1.,battery_capacity_kwh=10.,
        initial_soc=.5,terminal_energy_kwh=5.,eta_charge=1.,eta_discharge=1.,
        mobile=False,initial_connected=connected,setup_slots=T,max_throughput_kwh=2.))


@pytest.mark.parametrize('p',[-1.,-.25,0.,.25,1.])
def test_box_active_rank_deficient_scalar_profile_has_correct_dual_and_curvature(p):
    out=scalar.scalar_profile(np.zeros(1),np.ones(1),np.array([[1.,0.]]),
        np.array([-.25,0.]),np.array([.25,0.]),np.zeros(2),p,power_range=(-1,1))
    exact=.5*max(abs(p)-.25,0.)**2
    assert out['primal_upper']==pytest.approx(exact,abs=1e-11)
    assert out['dual_lower']<=exact+1e-11
    assert out['primal_upper']-out['dual_lower']<1e-9
    assert out['curvature_upper']==1.
    for q in np.linspace(-1.,1.,11):
        lower=out['dual_intercept']+out['dual_slope']*q
        assert lower<=.5*max(abs(q)-.25,0.)**2+1e-11


def test_frozen_scalar_decomposition_and_finer_grid_error_bound():
    context=fixture_context(source=True);domain=fixture_domain()
    table=scalar.build_scalar_tables(context,domain,grid_points=5)
    for time in table['tables']:
        row=time[0];grid=row['grid'];values=row['primal_upper']
        for p in np.linspace(-1.,1.,101):
            upper=np.interp(p,grid,values);exact=.5*max(abs(p-.2)-.25,0.)**2
            assert upper>=exact-1e-11
            assert upper-exact<=row['total_error_upper']+1e-10
    u=np.array([[.4],[-.4]])
    bound=scalar.action_profile_bounds(context,u)
    exact=sum(.5*max(abs(p-.2)-.25,0.)**2 for p in u[:,0])
    assert bound['dual_lower']<=exact+1e-10<=bound['primal_upper']+1e-10


@pytest.mark.parametrize('points',[3,5,9,17])
def test_two_slot_MILP_brackets_independently_enumerated_profile_optimum(points):
    context=fixture_context();domain=fixture_domain()
    solved=scalar.solve_scalar_distance_milp(context,domain,grid_points=points,
        solver_seconds=5.,backend='scipy',initial_powers=[np.zeros((2,1))])
    # Independent audit: zero task, eta=1 implies p2=-p1, |p1|<=1.
    grid=np.linspace(-1.,1.,2001)
    exact=max(.5*((p-.2)**2+(-p-.2)**2) for p in grid)
    assert solved['success'] and solved['operation']['feasible']
    assert solved['frozen_objective_lower']<=exact+1e-8
    assert solved['frozen_objective_upper']>=exact-1e-8
    assert solved['frozen_absolute_gap']<1e-7
    assert solved['bound_consistent']
    assert solved['full_AC_global_gap'] is None
    if points==3:
        assert solved['added_binary_count']==solved['added_continuous_count']==0
    else:assert solved['added_binary_count']>0


def test_disconnected_EV_still_counts_the_background_exactly_once():
    context=fixture_context();domain=fixture_domain(connected=False)
    for points in (3,5):
        solved=scalar.solve_scalar_distance_milp(context,domain,grid_points=points,backend='scipy')
        assert solved['success'] and solved['power_kw']==pytest.approx(np.zeros((2,1)))
        assert solved['solver_proxy_incumbent']==pytest.approx(.04)
        assert solved['frozen_objective_lower']==pytest.approx(.04,abs=1e-10)


def test_scalar_rejects_any_cross_time_or_bilinear_term():
    context=fixture_context();pair=context['pairs'][0]
    B=pair.B.copy();B[0,1]=1e-15
    bad=OriginalDomainPair(pair.name,pair.c,B,pair.A0,pair.At,pair.lower,pair.upper,pair.xi0,pair.shift)
    context['pairs'][0]=bad
    with pytest.raises(ValueError,match='cross-time'):scalar.build_scalar_tables(context,fixture_domain())


def test_nested_tables_reuse_precomputed_nodes_and_reduce_curvature_error():
    context=fixture_context(source=True);domain=fixture_domain();cache={}
    tables=[scalar.build_scalar_tables(context,domain,grid_points=k,cache=cache) for k in (3,5,9,17)]
    assert tables[1]['cache_hits']>0 and tables[-1]['cache_hits']>0
    errors=[row['frozen_profile_error_upper'] for row in tables]
    assert all(b<=a+1e-10 for a,b in zip(errors,errors[1:]))


@pytest.mark.parametrize('change',['c','source_box','power_domain'])
def test_external_tables_reject_changed_context_or_operation(change):
    context=fixture_context();domain=fixture_domain()
    table=scalar.build_scalar_tables(context,domain,grid_points=3)
    if change=='power_domain':
        domain=build_mobile_service_domain(MobileServiceConfig(n_slots=2,n_sites=1,
            p_charge_max_kw=2.,p_discharge_max_kw=1.,battery_capacity_kwh=10.,
            initial_soc=.5,terminal_energy_kwh=5.,mobile=False))
    else:
        pair=context['pairs'][0]
        c=pair.c.copy();lo=pair.lower.copy()
        if change=='c':c[0]+=.1
        else:lo[0]-=.1
        context['pairs'][0]=OriginalDomainPair(pair.name,c,pair.B,pair.A0,pair.At,lo,pair.upper,pair.xi0,pair.shift)
    with pytest.raises(ValueError,match='different frozen'):
        scalar.solve_scalar_distance_milp(context,domain,grid_points=3,tables=table,backend='scipy')


def test_probe_planning_whitelist_excludes_truth_validation_and_future_noise():
    from copy import deepcopy
    from .ev_scalar_distance_probe import planning_view
    fit=dict(candidate='a',r=[.01],x=[.002],source_p=[.1],source_q=[.03],success=True,loss=1.,future_truth=9)
    record=dict(training={'p':[[1.]]},training_meter={},future_meter={},candidates=[],forecast={},
        training_result=dict(fits=[fit]),config=dict(profile={},source={},horizon=6),
        truth={'r':[99]},validation={'v':[999]},methods=[{'M':333}],background={'p_kw':[999]})
    changed=deepcopy(record)
    for key in ('truth','validation','methods','background'):changed[key]='changed heldout value'
    assert planning_view(record)==planning_view(changed)
    assert 'future_truth' not in planning_view(record)['training_fits'][0]


def test_real_AC_context_scalar_and_full_QP_profiles_agree():
    from .test_ev_fixed_train_design import context as fixture
    from . import ev_fixed_train_design as fixed
    context,*_=fixture.__wrapped__()
    domain=fixed._service(3,3,mobile=False,discharge=True)
    blocks=scalar._time_parts(context);u=np.zeros((3,3));u[:,0]=[3.,4.,5.]
    direct=scalar.action_profile_bounds(context,u)
    values=[]
    for row in blocks:
        total=0.
        for t,part in enumerate(row):
            out=scalar.scalar_profile(part['c'],part['B'][:,0],part['G'],part['lower'],part['upper'],
                part['center'],u[t,0],power_range=(-6.,6.))
            total+=out['primal_upper']
        values.append(total)
    assert np.mean(values)==pytest.approx(direct['primal_upper'],rel=1e-9,abs=1e-8)
    solved=scalar.solve_scalar_distance_milp(context,domain,grid_points=3,backend='scipy',solver_seconds=3.)
    assert solved['success'] and solved['bound_consistent']
    assert solved['added_binary_count']==0
