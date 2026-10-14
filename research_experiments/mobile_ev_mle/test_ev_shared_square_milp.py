"""Independent analytic checks for the compact shared-control-square master."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.optimize import LinearConstraint
from . import ev_original_domain_milp as od
from . import ev_shared_square_milp as ss


def direct(name,c,C):
    c=np.asarray(c,float);C=np.asarray(C,float);m,K=C.shape
    return od.OriginalDomainPair(name,c,C,np.zeros((m,0)),np.zeros((K,m,0)),
        [],[],[],np.zeros((0,K)))


def witnesses(pairs):return [[np.empty(0)] for _ in pairs]


def exclusive_operation():
    # Signed u_j are both bounded by their own y_j; y0+y1<=1.
    return dict(lower=[-1.,-1.,0.,0.],upper=[1.,1.,1.,1.],integrality=[0,0,1,1],
        control_indices=[0,1],constraints=LinearConstraint(
            [[1,0,-1,0],[-1,0,-1,0],[0,1,0,-1],[0,-1,0,-1],[0,0,1,1]],
            [-np.inf]*5,[0,0,0,0,1]),zero_product_pairs=[(0,1)],
        zero_product_proof='This label is deliberately not used as the proof.')


def test_one_shared_signed_square_chord_has_exact_endpoint_maximum():
    p=direct('orthogonal',[.2,-.3],[[2.,0.],[0.,3.]])
    out=od.solve_original_master([p],witnesses([p]),[-1.,-.5],[.5,1.],
        master_form='shared_control_square',score_scale=1.,control_segments=4)
    truth=max((.2+2*u)**2+(-.3+3*v)**2 for u in [-1.,.5] for v in [-.5,1.])
    assert out['master_form']=='shared_control_square'
    assert out['objective_upper']>=truth-1e-8
    assert out['surrogate_incumbent']==pytest.approx(truth,abs=1e-8)
    assert out['n_shared_square_binary']==8
    assert out['n_product_binary']==out['n_residual_binary']==0
    assert od.evaluate_original_pair(p,out['u'])['primal_upper']==pytest.approx(truth,abs=1e-8)


def test_signed_energy_neutral_curve_is_found_by_milp_not_curve_library():
    p=direct('neutral',[0.,0.],np.eye(2))
    out=od.exchange_original_domain([p],[-1.,-1.],[1.,1.],LinearConstraint([[1.,1.]],0.,0.),
        initial_u=[0.,0.],master_form='shared_control_square',score_scale=1.,
        control_segments=4,max_iterations=2,absolute_gap=1e-8,relative_gap=0.,solver_seconds=2.)
    assert out['completed'],out['status']
    assert sorted(out['u'])==pytest.approx([-1.,1.])
    assert out['objective_lower']==pytest.approx(2.,abs=1e-8)
    assert out['objective_upper']>=2.-1e-8
    assert out['qp_calls']==len(out['evaluations'])
    assert all(not i['master']['incumbent_is_true_score_lower'] for i in out['iterations'])


def test_optimistic_master_incumbent_is_not_used_as_qp_lower():
    p=direct('fixed_midpoint',[0.],[[1.]])
    out=od.exchange_original_domain([p],[0.],[1.],LinearConstraint([[1.]],.5,.5),
        initial_u=[.5],master_form='shared_control_square',score_scale=1.,
        control_segments=1,residual_segments=1,max_segments=1,max_iterations=1,
        absolute_gap=1e-9,relative_gap=0.)
    master=out['iterations'][0]['master']
    assert master['surrogate_incumbent']==pytest.approx(.5,abs=1e-8)
    assert out['objective_lower']==pytest.approx(.25,abs=1e-8)
    assert out['objective_upper']>=.5-1e-8
    assert master['objective_lower'] is None and not master['incumbent_is_true_score_lower']
    assert out['unfinished']


def test_actual_operation_proves_signed_one_site_and_removes_large_cross_dot():
    p=direct('same_slot',[.3],[[2.,-3.]])
    op=exclusive_operation()
    out=od.solve_original_master([p],witnesses([p]),[-1.,-1.],[1.,1.],operation=op,
        master_form='shared_control_square',score_scale=1.,control_segments=4)
    truth=max((.3+2*u-3*v)**2 for u,v in [(1,0),(-1,0),(0,1),(0,-1)])
    assert out['master_form']=='shared_control_square',out.get('shared_square_fallback')
    assert out['objective_upper']>=truth-1e-8
    assert out['surrogate_incumbent']==pytest.approx(truth,abs=1e-8)
    assert abs(out['u'][0]*out['u'][1])<1e-9
    proof=out['shared_square_structure']
    assert proof['valid'] and not proof['labels_used_as_proof']
    assert proof['pairs'][0]['occupancy_sum_lp_upper']<2.
    normalized=od._operation(2,np.array([-1.,-1.]),np.array([1.,1.]),None,op)
    repeated=ss.certify_operation_zero_products(normalized,{(0,1)})
    assert repeated['cache_hit'] and repeated['operation_sha256']==proof['operation_sha256']


def test_fake_zero_label_is_rejected_and_generic_fallback_keeps_cross_term():
    p=direct('same_slot',[0.],[[1.,1.]])
    # No occupancy/power links exist: labeling two controls zero does not prove it.
    fake=dict(lower=[0.,0.],upper=[1.,1.],integrality=[0,0],control_indices=[0,1],
        zero_product_pairs=[(0,1)],zero_product_proof='single_EV_one_site_per_slot')
    out=od.solve_original_master([p],witnesses([p]),[0.,0.],[1.,1.],operation=fake,
        master_form='shared_control_square',score_scale=1.)
    assert out['master_form']=='residual_chord'
    assert 'unproved zero product' in out['shared_square_fallback']
    # Generic residual model keeps C0*u0+C1*u1 exactly, rather than dropping its square cross.
    assert out['objective_upper']>=4.-1e-7


def test_real_cross_time_gram_term_safely_falls_back_without_time_labels():
    p=direct('correlated_rows',[0.],[[1.,2.]])
    out=od.solve_original_master([p],witnesses([p]),[-1.,-1.],[1.,1.],
        master_form='shared_control_square',score_scale=1.)
    assert out['master_form']=='residual_chord'
    assert 'cross-control Gram' in out['shared_square_fallback']
    assert out['objective_upper']>=9.-1e-8


def test_unproved_zero_is_also_disabled_in_initial_bound_and_nonlinear_fallback():
    p=od.OriginalDomainPair('cross',[0.],[[0.,0.]],[[0.]],[[[1.]],[[0.]]],
        [0.],[0.],[0.],[[0.,-1.]])
    fake=dict(lower=[0.,0.],upper=[1.,1.],integrality=[0,0],control_indices=[0,1],
        zero_product_pairs=[(0,1)],zero_product_proof='unverified claim')
    out=od.exchange_original_domain([p],[0.,0.],[1.,1.],operation=fake,initial_u=[.5,.5],
        master_form='shared_control_square',score_scale=1.,max_iterations=1,solver_seconds=2.,
        absolute_gap=1e-8,relative_gap=0.)
    assert not out['shared_square_initial_structure']['valid']
    assert out['zero_product_pairs']==[]
    assert out['initial_fixed_witness_upper']>=1.-1e-8
    assert out['objective_upper']>=1.-1e-8
    assert out['objective_lower']==pytest.approx(1.,abs=1e-7)
    assert out['iterations'][0]['master']['product_keys']==[(0,1)]
    assert out['iterations'][0]['master']['fallback_keeps_all_products']


def test_nonaffine_witness_safe_fallback_retains_genuine_product():
    p=od.OriginalDomainPair('quadratic',[0.],[[0.,0.]],[[0.]],[[[1.]],[[0.]]],
        [0.],[0.],[0.],[[0.,-1.]])
    out=od.solve_original_master([p],[[np.array([0.])]],[-1.,-1.],[1.,1.],
        master_form='shared_control_square',score_scale=1.)
    assert out['master_form']=='residual_chord'
    assert 'co-moving' in out['shared_square_fallback']
    assert out['product_keys']==[(0,1)]
    assert out['objective_upper']>=1.-1e-8


def test_same_witness_compact_upper_dominates_original_qp_with_active_boundary():
    p=od.OriginalDomainPair('boundary',[.2,1.],[[.7],[.4]],[[1.],[0.]],np.zeros((1,2,1)),
        [0.],[2.],[0.],[[1.]])
    # xi(u)=1+u is globally legal; it is not necessarily the QP minimizer.
    witness=dict(physical_intercept=np.array([1.]),physical_slope=np.array([[1.]]))
    poly=od.physical_witness_polynomial(p,witness,[0.],[1.])
    rec=ss.compile_shared_square_witness(p,poly,np.array([0.]),np.array([1.]),set())
    for value in np.linspace(0.,1.,41):
        u=np.array([value]);qp=od.evaluate_original_pair(p,u)
        true_witness=p.residual(u,[1+value]);cost=float(true_witness@true_witness)
        compiled=rec['kappa']+rec['ell']@u+rec['diagonal']@(u*u)+rec['upward_cost_guard']
        assert compiled>=cost-1e-12>=qp['primal_upper']-1e-12
    assert od.evaluate_original_pair(p,[0.])['active_lower'].tolist()==[0]


def test_roundoff_guard_encloses_cancellation_of_large_coefficients():
    rng=np.random.default_rng(815)
    # Each column has disjoint row support, while a contains cancellation at 1e8 scale.
    K=4;m=24;n=3
    c=rng.normal(size=m)*1e8;A=rng.normal(size=(m,n))*1e8;delta=np.array([.1,.2,.3])
    B=np.zeros((m,K));At=np.zeros((K,m,n))
    for k in range(K):
        B[6*k:6*k+6,k]=rng.normal(size=6)*1e7
        At[k,6*k:6*k+6,:]=rng.normal(size=(6,n))*1e7
    p=od.OriginalDomainPair('scale',c,B,A,At,[-2.]*n,[2.]*n,[0.]*n,np.zeros((n,K)))
    poly=od.physical_witness_polynomial(p,dict(physical_intercept=delta,physical_slope=np.zeros((n,K))),[-1.]*K,[1.]*K)
    rec=ss.compile_shared_square_witness(p,poly,-np.ones(K),np.ones(K),set())
    ld=np.longdouble
    for u in rng.uniform(-1,1,(80,K)):
        true_r=p.c.astype(ld)+p.B.astype(ld)@u.astype(ld)
        true_r+=(p.A0.astype(ld)+np.einsum('k,kmn->mn',u.astype(ld),p.At.astype(ld)))@delta.astype(ld)
        true_cost=true_r@true_r
        upper=ld(rec['kappa'])+rec['ell'].astype(ld)@u.astype(ld)
        upper+=rec['diagonal'].astype(ld)@(u.astype(ld)**2)+ld(rec['upward_cost_guard'])
        assert upper>=true_cost
    assert rec['formation_guard']>0 and rec['coefficient_guard']>0


@pytest.mark.parametrize('form',['residual_chord','shared_control_square'])
def test_caps_are_applied_after_history_offset_and_leave_raw_scores(form):
    ps=[direct('positive',[2.],[[0.]]),direct('negative',[1.],[[0.]])]
    out=od.exchange_original_domain(ps,[0.],[1.],offsets={'positive':-2.,'negative':4.},
        score_caps={'positive':1.,'negative':None},initial_u=[.5],score_scale=1.,master_form=form,
        absolute_gap=1e-8,relative_gap=0.)
    entry=out['evaluations'][0]
    assert entry['pair_score_upper']==pytest.approx([6.,-3.])
    assert entry['capped_pair_score_upper']==pytest.approx([1.,-3.])
    assert out['mean_score_interval'][1]==pytest.approx(1.5)
    assert out['capped_mean_score_interval'][1]==pytest.approx(-1.)
    assert out['objective_lower']<=-1.+1e-9<=out['objective_upper']+1e-9
    assert out['worst_score_interval'][1]==pytest.approx(-3.)


def test_capped_mean_changes_curve_and_keeps_all_tasks():
    ps=[direct('large',[0.],[[10.,0.]]),direct('small',[0.],[[0.,1.]])]
    con=LinearConstraint([[1.,1.]],1.,1.)
    common=dict(initial_u=[.5,.5],score_scale=1.,master_form='shared_control_square',
        max_iterations=2,control_segments=64,residual_segments=4,max_segments=64,
        solver_seconds=3.,absolute_gap=1e-8,relative_gap=0.)
    mean=od.exchange_original_domain(ps,[0.,0.],[1.,1.],con,**common)
    capped=od.exchange_original_domain(ps,[0.,0.],[1.,1.],con,score_caps=[.1,None],**common)
    assert mean['u']==pytest.approx([1.,0.],abs=1e-7)
    optimum=(.1+(1-np.sqrt(.001))**2)/2
    assert capped['objective_lower']<=optimum+1e-8<=capped['objective_upper']+1e-8
    assert capped['u'][0]==pytest.approx(np.sqrt(.001),abs=1e-3)
    assert len(capped['required_pairs'])==2 and capped['qp_calls']==2*len(capped['evaluations'])


def test_worst_caps_and_negative_offsets_preserve_valid_lower_bounds():
    ps=[direct('one',[0.],[[1.,0.]]),direct('two',[0.],[[0.,2.]])]
    out=od.exchange_original_domain(ps,[-1.,-1.],[1.,1.],offsets=[-10.,2.],score_caps=[1.,.3],
        initial_u=[0.,0.],aggregation='worst',score_scale=1.,master_form='shared_control_square',
        max_iterations=2,absolute_gap=1e-7,relative_gap=0.,solver_seconds=2.)
    assert out['objective_lower']==pytest.approx(.3,abs=1e-7)
    assert out['objective_upper']>=.3-1e-8
    assert out['worst_score_interval'][1]>.3


def test_twelve_pairs_many_witnesses_share_only_eighteen_square_grids():
    K=18;ps=[direct(str(k),np.full(K,.01*k),np.eye(K)*(1+k/12)) for k in range(12)]
    # Four duplicate legal witnesses are intentional: grids must not multiply by witness count.
    ws=[[np.empty(0) for _ in range(4)] for p in ps]
    con=LinearConstraint(np.ones((1,K)),3.,3.)
    out=od.solve_original_master(ps,ws,-np.ones(K),np.ones(K),con,
        master_form='shared_control_square',score_scale=.5,control_segments=4,time_limit=2.)
    assert out['master_form']=='shared_control_square'
    assert out['n_shared_square_binary']==72
    assert out['n_residual_binary']==out['n_product_binary']==0
    assert len(out['shared_square_witnesses'])==48
    assert out['n_variables']<250
    assert out['objective_upper']>=0.


@pytest.mark.parametrize('backend',['scipy','gurobi'])
def test_true_mobile_flow_domain_proof_and_complete_operation_preserved(backend):
    if backend=='gurobi':pytest.importorskip('gurobipy')
    from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain
    cfg=MobileServiceConfig(n_slots=3,n_sites=2,dt_hours=1.,p_charge_max_kw=1.,p_discharge_max_kw=1.,
        allow_discharge=True,battery_capacity_kwh=10.,initial_soc=.5,soc_min=0.,soc_max=1.,
        eta_charge=1.,eta_discharge=1.,terminal_energy_kwh=6.,setup_slots=0,
        travel_slots=1,travel_energy_kwh=0.,max_moves=1)
    domain=build_mobile_service_domain(cfg);op=domain.as_dict()
    # Rows are time-disjoint but two locations in each row have a nonzero cross dot.
    C=np.zeros((3,6))
    for t in range(3):C[t,2*t:2*t+2]=[1.,2.]
    p=direct('mobile',np.zeros(3),C)
    out=od.exchange_original_domain([p],op['control_lower'],op['control_upper'],operation=op,
        master_form='shared_control_square',score_scale=1.,max_iterations=2,
        absolute_gap=1e-7,relative_gap=0.,solver_seconds=3.,backend=backend)
    assert out['u'] is not None
    assert domain.check(out['operation_x'])['feasible']
    assert len(out['operation_x'])==len(op['lower'])
    assert out['objective_upper']>=out['objective_lower']
    assert out['iterations']
    assert all(i['master']['master_form']=='shared_control_square' for i in out['iterations'])
    assert all(i['master']['shared_square_structure']['valid'] for i in out['iterations'])
    assert all(i['master']['n_operation_binary']>0 for i in out['iterations'])


def test_invalid_caps_and_option_are_rejected():
    p=direct('a',[0.],[[1.]])
    with pytest.raises(ValueError,match='nonnegative'):
        od.exchange_original_domain([p],[0.],[1.],score_caps=[-1.])
    with pytest.raises(ValueError,match='every required pair'):
        od.solve_original_master([p],witnesses([p]),[0.],[1.],score_caps={})
    with pytest.raises(ValueError,match='master_form'):
        od.exchange_original_domain([p],[0.],[1.],master_form='unimplemented')


def test_explicit_old_master_form_preserves_default_numeric_behavior():
    p=direct('old',[.2,-.3],[[1.,0.],[0.,1.]])
    kwargs=dict(initial_u=[.5,.5],score_scale=1.,max_iterations=2,
        absolute_gap=1e-8,relative_gap=0.,solver_seconds=2.)
    default=od.exchange_original_domain([p],[0.,0.],[1.,1.],**kwargs)
    explicit=od.exchange_original_domain([p],[0.,0.],[1.,1.],master_form='residual_chord',**kwargs)
    assert default['u']==pytest.approx(explicit['u'])
    assert default['objective_lower']==explicit['objective_lower']
    assert default['objective_upper']==explicit['objective_upper']
    assert default['status']==explicit['status']
