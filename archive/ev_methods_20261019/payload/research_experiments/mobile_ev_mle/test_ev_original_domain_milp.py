"""Independent algebra, original-domain, product-envelope and operation tests."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.optimize import LinearConstraint,Bounds,milp
from scipy.sparse import coo_matrix
from . import ev_original_domain_milp as od


def direct(name,c,B):
    c=np.asarray(c,float);B=np.asarray(B,float);m,t=B.shape
    return od.OriginalDomainPair(name,c,B,np.zeros((m,0)),np.zeros((t,m,0)),[],[],[],np.zeros((0,t)))


def cross_pair():
    return od.OriginalDomainPair('cross',[.2],[[.5,-.4]],[[0.]],[[[1.]],[[3.]]],
        [1.],[1.],[0.],[[1.,2.]])


def test_fixed_absolute_expansion_keeps_diagonal_and_genuine_cross_coefficients():
    p=cross_pair();poly=od.fixed_absolute_witness(p,[1.])
    assert poly['a']==pytest.approx([.2])
    assert poly['C']==pytest.approx(np.array([[1.5,2.6]]))
    assert poly['products'][(0,0)]==pytest.approx([-1.])
    assert poly['products'][(0,1)]==pytest.approx([-5.])
    assert poly['products'][(1,1)]==pytest.approx([-6.])
    for u in ([-.2,.4],[.1,.2],[.4,-.3]):
        assert od.polynomial_residual(poly,u)==pytest.approx(p.residual(u,[1.]),abs=1e-13)


def test_zero_width_original_box_qp_and_moving_boundary_value():
    p=cross_pair();u=[.1,.1]
    fit=od.evaluate_original_pair(p,u)
    assert fit['success'] and fit['fixed_indices'].tolist()==[0]
    assert fit['physical_parameter']==pytest.approx([1.])
    assert fit['primal_upper']==pytest.approx(float(p.residual(u,[1.])@p.residual(u,[1.])))
    boundary=od.OriginalDomainPair('boundary',[0.],[[0.]],[[1.]],[[[0.]]],[1.],[2.],[0.],[[1.]])
    for u in (0.,.3,.8):
        fit=od.evaluate_original_pair(boundary,[u])
        assert fit['primal_upper']==pytest.approx((1-u)**2)
        assert fit['tangent_lower_numeric']<=fit['primal_upper']
        assert fit['z']==pytest.approx([1-u])
        assert fit['active_lower'].tolist()==[0]


def test_rank_deficient_qp_preserves_unexplained_constant_and_negative_offset():
    p=od.OriginalDomainPair('rank',[0.,1.],[[0.],[1.]],[[1.,1.],[0.,0.]],
        np.zeros((1,2,2)),[-1.,-1.],[1.,1.],[0.,0.],np.zeros((2,1)))
    fit=od.evaluate_original_pair(p,[.4])
    assert fit['success'] and fit['primal_upper']==pytest.approx(1.4**2)
    out=od.exchange_original_domain([p],[0.],[1.],offsets={'rank':10.},initial_u=[.5],
        absolute_gap=1e-7,relative_gap=0.,max_iterations=3,solver_seconds=2.)
    assert out['objective_lower']<=-3.+1e-8<=out['objective_upper']+1e-8
    assert out['objective_lower']==pytest.approx(-3.,abs=1e-7)
    assert out['objective_upper']<0.


def test_full_domain_affine_witness_validation_and_neutral_comoving_cancellation():
    p=od.OriginalDomainPair('track',[0.],[[0.]],[[ -1.]],[[[0.]]],[-1.],[2.],[0.],[[1.]])
    out=od.exchange_original_domain([p],[0.],[1.],initial_u=[.5],score_scale=1.,absolute_gap=1e-7,relative_gap=0.)
    assert out['completed'] and out['status']=='gap_reached_initial_witness_bound'
    assert out['objective_upper']<1e-9 and out['objective_lower']==pytest.approx(0.,abs=1e-10)
    assert out['witness_counts']==[2]
    affine=next(w for w in out['witnesses'][0] if np.any(w['physical_slope']))
    poly=od.physical_witness_polynomial(p,affine,[0.],[1.])
    assert poly['products']=={} and np.linalg.norm(poly['C'])<1e-14
    for u in ([0.],[.3],[1.]):
        xi=affine['physical_intercept']+affine['physical_slope']@u
        assert od.polynomial_residual(poly,u)==pytest.approx(p.residual(u,xi))
    with pytest.raises(ValueError,match='not certified'):
        od.physical_witness_polynomial(p,dict(physical_intercept=[0.],physical_slope=[[3.]]),[0.],[1.])


def test_affine_witness_operation_lp_fallback_is_global_not_sample_validation():
    p=od.OriginalDomainPair('affine',[0.],[[0.,0.]],[[1.]],np.zeros((2,1,1)),
        [0.],[.7],[0.],np.zeros((1,2)))
    op=od._operation(2,np.zeros(2),np.ones(2),LinearConstraint([[1.,1.]],.5,.5),None)
    witness=dict(physical_intercept=[0.],physical_slope=[[1.,1.]])
    poly=od.physical_witness_polynomial(p,witness,np.zeros(2),np.ones(2),operation=op)
    validation=poly['witness_validation']
    assert validation['valid'] and 'operation_LP' in validation['method']
    assert validation['physical_enclosure_upper'][0]<.7
    assert len(validation['LP_audits'])==1


@pytest.mark.parametrize('point',[[-.1,.3],[.1,.15],[.3,-.1]])
def test_shared_product_outer_envelope_keeps_true_graph_with_signed_controls(point):
    low=np.array([-.2,-.3]);high=np.array([.4,.5]);point=np.asarray(point)
    op=od._operation(2,low,high,LinearConstraint(np.eye(2),point,point),None)
    keys=[(0,0),(0,1),(1,1)];lift=od._lift(op,low,high,keys,set(),4)
    lo=np.array(lift['lower']);hi=np.array(lift['upper'])
    for index,(t,s) in zip(lift['vid'],keys):lo[index]=hi[index]=point[t]*point[s]
    rr=[];cc=[];vv=[]
    for i,row in enumerate(lift['rows']):
        for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
    A=coo_matrix((vv,(rr,cc)),shape=(len(lift['rows']),len(lo))).tocsc()
    fit=milp(np.zeros(len(lo)),integrality=np.array(lift['integrality']),bounds=Bounds(lo,hi),
        constraints=LinearConstraint(A,lift['los'],lift['his']),options={'time_limit':2.})
    assert fit.success,fit.message


def test_cross_term_two_layer_master_bound_covers_known_quartic_maximum():
    p=cross_pair();con=LinearConstraint([[1.,1.]],.2,.2)
    out=od.exchange_original_domain([p],[-.2,-.3],[.4,.5],con,initial_u=[.1,.1],
        score_scale=1.,max_iterations=2,control_segments=8,residual_segments=8,max_segments=16,
        absolute_gap=1e-6,relative_gap=0.,solver_seconds=2.,add_comoving_witness=False)
    # u2=.2-u1 makes residual .48+.3*u1-2*u1²; max at .075.
    truth=(.48+.3*.075-2*.075**2)**2
    assert out['objective_lower']<=truth+1e-8<=out['objective_upper']+1e-8
    assert all((0,1) in it['master']['product_keys'] for it in out['iterations'])
    assert all(not it['master']['incumbent_is_true_score_lower'] for it in out['iterations'])
    assert all(it['master']['objective_lower'] is None for it in out['iterations'])
    for it in out['iterations']:
        for comp in it['master']['compressions']:
            expected=comp['product_cost_error_upper']+comp['residual_square_error_upper']+2*comp['qr_upward_cost_guard']
            assert comp['pwl_score_error_upper']==pytest.approx(expected)


def test_mean_and_worst_targets_differ_without_dropping_weak_model_pair():
    eps=.1;con=LinearConstraint([[1.,1.]],1.,1.)
    ps=[direct('12',[0.],[[eps,0.]]),direct('13',[0.],[[0.,1.]]),direct('23',[0.],[[eps,-1.]])]
    common=dict(initial_u=[.5,.5],score_scale=1.,max_iterations=3,solver_seconds=2.,absolute_gap=1e-4,relative_gap=0.)
    mean=od.exchange_original_domain(ps,[0.,0.],[1.,1.],con,aggregation='mean',**common)
    worst=od.exchange_original_domain(ps,[0.,0.],[1.,1.],con,aggregation='worst',**common)
    assert mean['u']==pytest.approx([0.,1.],abs=1e-7)
    assert mean['objective_lower']==pytest.approx(2/3,abs=1e-7)
    assert mean['worst_score_interval'][1]<1e-10
    truth=eps**2/(1+2*eps)**2
    assert worst['objective_lower']<=truth+1e-8<=worst['objective_upper']+1e-8
    assert len(mean['required_pairs'])==len(worst['required_pairs'])==3


def exclusive_operation():
    # x=(u0,u1,b), b chooses one active location, signed power allowed.
    A=np.array([[1.,0.,-1.],[-1.,0.,-1.],[0.,1.,1.],[0.,-1.,1.],[1.,1.,0.]])
    return dict(lower=[-1.,-1.,0.],upper=[1.,1.,1.],integrality=[0,0,1],control_indices=[0,1],
        constraints=LinearConstraint(A,[-np.inf]*4+[1.],[0.,0.,1.,1.,1.]),
        zero_product_pairs=[(0,1)],zero_product_proof='binary one-site linked signed power')


def test_extended_binary_operation_and_structural_zero_product_are_preserved():
    # r=u1-u0*u1. Zero product holds only because complete operation b is retained.
    p=od.OriginalDomainPair('exclusive',[0.],[[0.,1.]],[[0.]],[[[1.]],[[0.]]],
        [0.],[0.],[0.],[[0.,1.]])
    out=od.exchange_original_domain([p],[-1.,-1.],[1.,1.],operation=exclusive_operation(),
        initial_controls=[[1.,0.],[0.,1.]],score_scale=1.,absolute_gap=1e-7,relative_gap=0.,solver_seconds=2.)
    assert out['completed']
    assert out['u']==pytest.approx([0.,1.],abs=1e-7)
    assert out['operation_x'][2]==pytest.approx(0.)
    assert out['objective_lower']==pytest.approx(1.,abs=1e-7)
    assert out['zero_product_pairs']==[(0,1)]
    assert out['operation_completion_calls']==2
    assert all(len(e['operation_x'])==3 for e in out['evaluations'])


def test_genuine_nonzero_product_across_two_controls_not_silently_removed():
    p=od.OriginalDomainPair('cross_only',[0.],[[0.,0.]],[[0.]],[[[1.]],[[0.]]],
        [0.],[0.],[0.],[[0.,-1.]])
    con=LinearConstraint([[1.,1.]],1.,1.)
    out=od.exchange_original_domain([p],[0.,0.],[1.,1.],con,initial_u=[.5,.5],score_scale=1.,
        absolute_gap=1e-7,relative_gap=0.,max_iterations=2,solver_seconds=2.)
    assert out['objective_lower']==pytest.approx(1/16,abs=1e-8)
    assert out['objective_upper']>=1/16-1e-8
    assert all(it['master']['product_keys']==[(0,1)] for it in out['iterations'])


def test_conflicting_master_bound_keeps_previous_upper_and_unfinished(monkeypatch):
    p=direct('u2',[0.],[[1.]])
    def broken(*args,**kwargs):
        return dict(u=np.array([1.]),operation_x=np.array([1.]),objective_upper=.1,
            bound_consistent=False,message='intentional invalid bound')
    monkeypatch.setattr(od,'solve_original_master',broken)
    out=od.exchange_original_domain([p],[0.],[1.],initial_u=[.5],score_scale=1.,absolute_gap=1e-8,relative_gap=0.)
    assert out['status']=='numerical_master_upper_rejected' and out['unfinished']
    assert out['objective_upper']==out['initial_fixed_witness_upper']
    assert out['objective_upper']>=out['objective_lower']


def test_inner_exception_kept_without_dropping_required_pair(monkeypatch):
    ps=[direct('a',[1.],[[0.]]),direct('b',[2.],[[0.]])]
    original=od.evaluate_original_pair
    def fail_one(pair,u):
        if pair.name=='b':raise RuntimeError('declared inner failure')
        return original(pair,u)
    monkeypatch.setattr(od,'evaluate_original_pair',fail_one)
    out=od.exchange_original_domain(ps,[0.],[1.],initial_u=[.5],score_scale=1.,max_iterations=1,solver_seconds=2.)
    assert len(out['required_pairs'])==2
    assert all(len(e['qp_results'])==2 for e in out['evaluations'])
    assert out['inner_qp_failures']>0
    assert not out['completed']
    assert any(f['kind']=='inner_qp_exception' for f in out['failures'])


def test_twelve_required_pairs_including_zero_task_weight_are_all_evaluated():
    ps=[direct(str(i),[float(i%3)],[[0.]]) for i in range(12)]
    weights=np.r_[0.,np.full(11,1/11)]
    out=od.exchange_original_domain(ps,[0.],[1.],weights=weights,initial_controls=[[0.],[1.]],
        absolute_gap=1e-8,relative_gap=0.)
    assert out['completed'] and len(out['required_pairs'])==12
    assert out['qp_calls']==12*len(out['evaluations'])
    assert all(len(e['qp_results'])==12 for e in out['evaluations'])


@pytest.mark.parametrize('backend',['scipy','gurobi'])
def test_actual_mobile_service_domain_full_x_works_without_route_library(backend):
    if backend=='gurobi':pytest.importorskip('gurobipy')
    from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain
    cfg=MobileServiceConfig(n_slots=3,n_sites=2,dt_hours=1.,p_charge_max_kw=1.,allow_discharge=False,
        battery_capacity_kwh=10.,initial_soc=.5,soc_min=0.,soc_max=1.,eta_charge=1.,eta_discharge=1.,
        terminal_energy_kwh=6.,setup_slots=0,travel_slots=1,travel_energy_kwh=0.,max_moves=1)
    domain=build_mobile_service_domain(cfg);op=domain.as_dict()
    B=np.zeros((1,6));B[0,3]=1.;B[0,5]=1. # reward second site's later power
    p=direct('mobile',[0.],B)
    out=od.exchange_original_domain([p],op['control_lower'],op['control_upper'],operation=op,
        score_scale=1.,max_iterations=2,absolute_gap=1e-7,relative_gap=0.,solver_seconds=3.,backend=backend)
    assert out['u'] is not None
    assert domain.check(out['operation_x'])['feasible']
    assert len(out['operation_x'])==len(op['lower'])
    assert out['objective_lower']==pytest.approx(1.,abs=1e-6)
    assert out['objective_upper']>=out['objective_lower']
    assert out['backend']==backend


def test_adapter_and_domain_validation_reject_silent_model_changes():
    p=cross_pair()
    model=dict(name=p.name,c=p.c,B=p.B,A0=p.A0,At=p.At,domain=dict(
        physical_lower=p.lower,physical_upper=p.upper,base_center=p.base_center,shift=p.shift))
    copy=od.from_moving_model(model)
    assert copy.residual([.1,.2],[1.])==pytest.approx(p.residual([.1,.2],[1.]))
    with pytest.raises(ValueError,match='not silently ignored'):
        od.from_moving_model({**model,'control_curvature_difference':np.zeros((1,2))})
    with pytest.raises(ValueError,match='every required pair'):
        od.exchange_original_domain([p],[-1.,-1.],[1.,1.],offsets={})
    with pytest.raises(ValueError,match='nonnegative task'):
        od.exchange_original_domain([p],[-1.,-1.],[1.,1.],weights=[-1.])


def test_affine_preferred_keeps_legal_co_moving_witness_and_valid_nonconstant_bound():
    p=od.OriginalDomainPair('prefer',[0.,1.],[[0.],[1.]],[[-1.],[0.]],
        np.zeros((1,2,1)),[-1.],[2.],[0.],[[1.]])
    out=od.exchange_original_domain([p],[0.],[1.],offsets={'prefer':10.},initial_u=[.5],
        witness_policy='affine_preferred',absolute_gap=1e-7,relative_gap=0.,solver_seconds=2.)
    assert out['completed'],out['status']
    assert out['witness_policy']=='affine_preferred'
    assert out['objective_lower']==pytest.approx(-3.,abs=1e-7)
    assert out['objective_upper']>=-3.-1e-8
    assert out['witness_counts']==[1]
    assert all(np.any(w['physical_slope']) for w in out['witnesses'][0])
    assert all(not it['master']['product_keys'] for it in out['iterations'])


def test_affine_preferred_falls_back_to_fixed_for_nonzero_product_physical_box():
    p=cross_pair();con=LinearConstraint([[1.,1.]],.2,.2)
    out=od.exchange_original_domain([p],[-.2,-.3],[.4,.5],con,initial_u=[.1,.1],
        witness_policy='affine_preferred',score_scale=1.,max_iterations=1,
        absolute_gap=1e-6,relative_gap=0.,solver_seconds=2.)
    assert out['witness_policy']=='affine_preferred'
    assert all(not np.any(w['physical_slope']) for w in out['witnesses'][0])
    assert any(f['kind']=='affine_witness_rejected' for f in out['failures'])
    truth=(.48+.3*.075-2*.075**2)**2
    assert out['objective_lower']<=truth+1e-8<=out['objective_upper']+1e-8
    assert all((0,1) in it['master']['product_keys'] for it in out['iterations'])


def test_master_partial_start_retains_full_operation_true_products_and_unknown_segments(monkeypatch):
    p=od.OriginalDomainPair('exclusive',[0.],[[0.,1.]],[[0.]],[[[1.]],[[0.]]],
        [0.],[0.],[0.],[[0.,1.]])
    saved=[];solver=od._solve_mip
    def capture(*args,**kwargs):
        saved.append(kwargs['start'].copy())
        return solver(*args,**kwargs)
    monkeypatch.setattr(od,'_solve_mip',capture)
    result=od.solve_original_master([p],[[np.array([0.])]],[-1.,-1.],[1.,1.],
        operation=exclusive_operation(),warm_start_operation=[1.,0.,1.],score_scale=1.)
    assert result['backend']=='scipy'
    assert result['warm_start']['operation_validation']['feasible']
    assert not result['warm_start']['provided_to_solver'] # HiGHS scipy API has no MIPstart.
    seed=saved[0]
    assert seed[:3]==pytest.approx([1.,0.,1.])
    assert seed[3:5]==pytest.approx([1.,0.])
    assert seed[5]==pytest.approx(0.) # the structurally-zero u0*u1.
    assert np.isnan(seed[6:]).any()
    assert result['warm_start']['filled_variables']==6


def test_exchange_passes_best_operation_to_every_master_and_rejects_unknown_backend(monkeypatch):
    p=direct('u2',[0.],[[1.]])
    original=od.solve_original_master;starts=[]
    def capture(*args,**kwargs):
        starts.append(np.asarray(kwargs['warm_start_operation']).copy())
        return original(*args,**kwargs)
    monkeypatch.setattr(od,'solve_original_master',capture)
    result=od.exchange_original_domain([p],[0.],[1.],initial_u=[.5],absolute_gap=1e-7,
        relative_gap=0.,backend='scipy')
    assert result['completed'] and starts
    assert starts[0]==pytest.approx([.5])
    with pytest.raises(ValueError,match='backend'):
        od.exchange_original_domain([p],[0.],[1.],backend='invalid')


@pytest.mark.parametrize('aggregation',['mean','worst'])
def test_gurobi_and_scipy_agree_with_negative_offsets_and_binary_operation(aggregation):
    pytest.importorskip('gurobipy')
    ps=[direct('a',[1.],[[1.,0.]]),direct('b',[0.],[[0.,1.]])]
    shared=dict(operation=exclusive_operation(),initial_operation_points=[[0.,1.,0.]],
        offsets={'a':10.,'b':3.},aggregation=aggregation,score_scale=.5,
        absolute_gap=1e-6,relative_gap=0.,solver_seconds=3.,max_iterations=3)
    scipy=od.exchange_original_domain(ps,[-1.,-1.],[1.,1.],backend='scipy',**shared)
    gurobi=od.exchange_original_domain(ps,[-1.,-1.],[1.,1.],backend='gurobi',**shared)
    assert scipy['objective_lower']==pytest.approx(gurobi['objective_lower'],abs=1e-6)
    assert scipy['objective_upper']==pytest.approx(gurobi['objective_upper'],abs=1e-6)
    assert gurobi['objective_upper']<0.
    assert gurobi['completed'] and gurobi['iterations']
    assert gurobi['backend']=='gurobi'
    assert all(it['master']['warm_start']['provided_to_solver'] for it in gurobi['iterations'])
    assert all(e['operation_validation']['feasible'] for e in gurobi['evaluations'])
