"""All-candidate aggregation, shared control MILPs and history-offset checks."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_systematic_design as design
from . import ev_fixed_train_design as fixed
from . import likelihood as lk
from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain
from .ev_original_domain_milp import OriginalDomainPair
from .simulation import model


def context(K=4,T=2,n=1,*,source=False,zero=False):
    bank=[SimpleNamespace(name='a'+str(i)) for i in range(K)];pairs=[];models=[]
    for a in range(K):
        for b in range(K):
            if a==b:continue
            name=f'a{a}__to__a{b}';m=T*(n+2)
            c=np.zeros(m);B=np.zeros((m,T*n));G=np.zeros((m,2*T))
            for t in range(T):
                for j in range(n):
                    c[t*n+j]=0. if zero else .1*(1+(a-b)**2)*(j+1)
                    B[t*n+j,t*n+j]=0. if zero else (.4+.1*a+.07*b)*(j+1)
                    if source:G[t*n+j,2*t]=.1*(j+1)
                if source:G[T*n+t,2*t]=1.;G[T*n+T+t,2*t+1]=1.
            pairs.append(OriginalDomainPair(name,c,B,G,np.zeros((T*n,m,2*T)),
                np.full(2*T,-.25),np.full(2*T,.25),np.zeros(2*T),np.zeros((2*T,T*n))))
            models.append(dict(name=name,a='a'+str(a),b='a'+str(b)))
    return dict(success=True,bank=bank,pairs=pairs,models=models,control_shape=(T,n))


def domain(T=2,n=1,*,mobile=False,terminal_site=None):
    return build_mobile_service_domain(MobileServiceConfig(n_slots=T,n_sites=n,dt_hours=1.,
        p_charge_max_kw=1.,p_discharge_max_kw=1.,battery_capacity_kwh=10.,initial_soc=.5,
        terminal_energy_kwh=5.,eta_charge=1.,eta_discharge=1.,mobile=mobile,
        travel_slots=1,setup_slots=1,travel_energy_kwh=0.,max_moves=1,
        terminal_site=terminal_site,max_throughput_kwh=2.))


def test_two_candidate_known_history_offsets_cancel_in_equal_reference_mean():
    ctx=context(2);values=[2.,8.]
    plain=design.aggregate_values(ctx,values)
    shifted=design.aggregate_values(ctx,values,dict(a0=30.,a1=2.))
    assert plain['score']==shifted['score']==5.
    assert shifted['nearest'][0]['score']<0.
    assert shifted['pair_offsets']==pytest.approx([-28.,28.])


def test_four_candidates_have_distinct_hard_rivals_and_history_changes_them():
    ctx=context(4)
    matrix=np.array([[0,1,4,6],[1,0,3,8],[2,1,0,5],[6,5,1,0]],float)
    values=[matrix[int(m['a'][1:]),int(m['b'][1:])] for m in ctx['models']]
    plain=design.aggregate_values(ctx,values)
    shifted=design.aggregate_values(ctx,values,dict(a0=0.,a1=10.,a2=0.,a3=0.))
    assert [r['active_competitors'] for r in plain['nearest']]==[['a1'],['a0'],['a1'],['a2']]
    assert shifted['nearest'][0]['active_competitors']==['a2']
    assert shifted['nearest'][2]['active_competitors']==['a0']
    assert shifted['score']!=plain['score']


@pytest.mark.parametrize('K',[2,4])
@pytest.mark.parametrize('grid',[3,9])
def test_shared_grid_bounds_all_candidate_nearest_objective(K,grid):
    ctx=context(K);op=domain();table=design.build_systematic_tables(ctx,op,grid_points=grid)
    h={f'a{i}':float(i) for i in range(K)}
    solved=design.solve_systematic_milp(ctx,op,tables=table,history_losses=h,backend='scipy',solver_seconds=5.)
    # Equal per-time affine constants make every directed profile depend on
    # p1²+p2². The net-zero task has its global maximum at (1,-1).
    values=[]
    for pair in ctx['pairs']:
        residual=pair.c+pair.B@np.array([1.,-1.]);values.append(.5*float(residual@residual))
    exact=design.aggregate_values(ctx,values,h)['score']
    assert solved['success'] and solved['operation']['feasible']
    assert solved['frozen_objective_lower']<=exact+1e-7
    assert solved['frozen_objective_upper']>=exact-1e-7
    assert solved['frozen_absolute_gap']<1e-7
    assert solved['reference_eta_count']==K and solved['source_pair_count']==K*(K-1)
    if grid==3:assert solved['added_binary_count']==0 and solved['added_continuous_count']==K
    else:
        # Controls/segments are shared; K(K-1) never multiplies binary count.
        assert solved['added_binary_count']==2*8
    assert solved['full_AC_global_gap'] is None


def test_rank_deficient_and_active_source_boxes_retain_valid_bounds():
    ctx=context(4,source=True);op=domain();cache={}
    table=design.build_systematic_tables(ctx,op,grid_points=3,cache=cache)
    for pair_times in table['tables']:
        for sites in pair_times:
            for row in sites:
                assert np.all(row['dual_lower']<=row['primal_upper']+1e-10)
    solved=design.solve_systematic_milp(ctx,op,tables=table,backend='scipy')
    assert solved['success'] and solved['bound_consistent']
    fine=design.build_systematic_tables(ctx,op,grid_points=9,cache=cache)
    assert fine['cache_hits']>0
    assert fine['nearest_error_upper']<=table['nearest_error_upper']+1e-8


def test_known_history_offset_can_make_score_negative_without_clipping():
    ctx=context(3,zero=True);op=domain()
    h=dict(a0=0.,a1=10.,a2=20.)
    table=design.build_systematic_tables(ctx,op,grid_points=3)
    solved=design.solve_systematic_milp(ctx,op,tables=table,history_losses=h,backend='scipy')
    assert solved['success']
    assert solved['frozen_objective_lower']==pytest.approx(-20./3,abs=1e-8)
    assert solved['frozen_objective_upper']==pytest.approx(-20./3,abs=1e-7)


def test_old_tables_rejected_before_solving_a_different_context_or_route_domain():
    ctx=context();op=domain();table=design.build_systematic_tables(ctx,op,grid_points=3)
    changed=context();pair=changed['pairs'][0];c=pair.c.copy();c[0]+=.1
    changed['pairs'][0]=OriginalDomainPair(pair.name,c,pair.B,pair.A0,pair.At,pair.lower,pair.upper,pair.xi0,pair.shift)
    with pytest.raises(ValueError,match='tables differ'):
        design.solve_systematic_milp(changed,op,tables=table,backend='scipy')
    route=domain(T=3)
    with pytest.raises(ValueError,match='tables differ'):
        design.solve_systematic_milp(ctx,route,tables=table,backend='scipy')


def test_mobile_operation_keeps_setup_background_and_all_candidate_pairs():
    ctx=context(4,T=4,n=2);op=domain(T=4,n=2,mobile=True,terminal_site=1)
    for grid in (3,9):
        table=design.build_systematic_tables(ctx,op,grid_points=grid)
        solved=design.solve_systematic_milp(ctx,op,tables=table,backend='scipy',solver_seconds=5.)
        assert solved['success'] and solved['operation']['feasible']
        assert solved['source_pair_count']==12 and solved['bound_consistent']
        assert len(solved['action_QP']['pairs'])==12
        decoded=op.decode(solved['operation_x'])
        assert any(arc.kind=='travel' for arc in decoded['selected_arcs'])
        assert solved['frozen_objective_lower']<=solved['frozen_objective_upper']+1e-7


def test_missing_competitor_or_bilinear_source_terms_are_rejected():
    ctx=context();ctx['pairs'].pop();ctx['models'].pop()
    with pytest.raises(ValueError,match='all directed'):design.build_systematic_tables(ctx,domain())
    ctx=context();pair=ctx['pairs'][0];At=pair.At.copy();At[0,0,0]=1e-15
    ctx['pairs'][0]=OriginalDomainPair(pair.name,pair.c,pair.B,pair.A0,At,pair.lower,pair.upper,pair.xi0,pair.shift)
    with pytest.raises(ValueError,match='At=shift=0'):design.build_systematic_tables(ctx,domain())


def test_table_budget_failure_preserves_required_pair_denominator():
    from time import perf_counter
    ctx=context();table=design.build_systematic_tables(ctx,domain(),deadline=perf_counter()-1.)
    assert not table['success'] and table['expected_directed_pairs']==12
    assert table['QP_calls']==0 and table['complete_directed_pair_tables']==0


def _real_context(n=5):
    T,H=4,12
    bank=[lk.RadialCandidate.from_supports('star',n,()),
        lk.RadialCandidate.from_supports('cherry01',n,((0,1),)),
        lk.RadialCandidate.from_supports('cherry23',n,((2,3),))]
    tm=model(n);fm=replace(tm,v_sigma=np.full((T,n),.00012),master_p_sigma=np.full(T,.001),master_q_sigma=np.full(T,.001))
    p=.09+.005*np.cos(np.arange(H)[:,None]+np.arange(n)[None,:]);q=.25*p
    sp=np.full(H,.08);sq=np.full(H,.03)
    fits={c.name:SimpleNamespace(r=np.full(c.m,.003),x=np.full(c.m,.001),source_p=sp.copy(),
        source_q=sq.copy(),p=p.copy(),q=q.copy(),success=True,loss=float(i+1)) for i,c in enumerate(bank)}
    phy=lk.ac_predict(bank[0],fits['star'].r,fits['star'].x,p,q,sp,sq,np.ones(H))
    obs=lk.Observations(p=p+tm.bias_p,q=q+tm.bias_q,v=phy['v']+tm.bias_v,
        root_v=np.ones(H),master_p=phy['master_p'],master_q=phy['master_q'])
    ref=np.zeros((T,n));ref[:,0]=3.
    ctx=fixed.build_fixed_contexts(obs,bank,fits,tm,fm,-fixed.CONFIG['command_power_kw'],
        fixed.CONFIG['command_power_kw'],ref)
    assert ctx['success'],ctx['diagnostics']
    return ctx,ref


@pytest.mark.parametrize('history',[False,True])
def test_real_n5_three_candidate_complete_AC_acceptance_and_parameter_freeze(history):
    ctx,u=_real_context();op=fixed._service(4,5,mobile=False,discharge=True)
    h={name:.5*fit.loss for name,fit in ctx['train_fits'].items()} if history else None
    power,record=design.scalar_policy(ctx,op,dict(constant=u),history_losses=h,grids=(3,),
        fit_options=dict(multistart=1,max_nfev=100),total_seconds=15.,solver_seconds=.5,backend='scipy',seed=73)
    assert record['success'] and record['selected_operation']['feasible']
    assert record['directed_pair_count']==6 and len(record['selected_profile']['pairs'])==6
    assert record['inheritance_protected'] and record['selected_score']>=record['initial_best_score']-1e-10
    assert not record['uses_truth'] and not record['uses_validation'] and not record['uses_future_noise']
    assert all(row['line_parameters_fixed'] for row in record['selected_profile']['pairs'])
    assert power.shape==(4,5)
    assert record['grids'][0]['raw']['reference_eta_count']==3


def test_policy_expired_budget_keeps_computation_failure_instead_of_dropping_candidates():
    ctx,u=_real_context();op=fixed._service(4,5,mobile=False,discharge=True)
    power,record=design.scalar_policy(ctx,op,dict(constant=u),total_seconds=1e-9,backend='scipy')
    assert power is None and not record['success']
    assert record['directed_pair_count']==6 and record['all_candidates_retained']


def test_failed_single_competitor_rejects_entire_proposal_and_keeps_initial_plan(monkeypatch):
    ctx=context(4);op=domain();initial=np.zeros((2,1))
    class Engine:
        def __init__(self,*args,**kwargs):self.count=0
        def evaluate(self,power,method='ac',**kwargs):
            self.count+=1;is_initial=np.max(abs(power))<1e-8
            rows=[dict(a=row['a'],b=row['b'],name=row['name'],success=is_initial or i>0,
                half_SSE=1.+float(np.sum(np.asarray(power)**2))) for i,row in enumerate(ctx['models'])]
            return dict(success=is_initial,pairs=rows)
        def counters(self):return dict(profile_evaluations=self.count)
    monkeypatch.setattr(design.gradient,'ProfileEngine',Engine)
    selected,record=design.scalar_policy(ctx,op,dict(constant=initial),grids=(3,),backend='scipy')
    assert record['success'] and selected==pytest.approx(initial)
    assert not record['grids'][0]['accepted']
    assert len(record['grids'][0]['raw_AC_profile']['pairs'])==12
    assert record['grids'][0]['raw_AC_score'] is None
    assert record['selected_score']==1.
    assert record['total_frozen_QP_calls']==record['table_QP_calls']+12


def test_pair_metadata_order_must_match_frozen_residual_matrices():
    ctx=context();ctx['models'][0],ctx['models'][1]=ctx['models'][1],ctx['models'][0]
    with pytest.raises(ValueError,match='metadata order'):design.build_systematic_tables(ctx,domain())


def test_compact_nodes_retain_site_specific_zero_dual_and_recomputable_curve():
    ctx=context(3,T=2,n=2,source=True);op=domain(T=2,n=2,mobile=True)
    table=design.build_systematic_tables(ctx,op,grid_points=3)
    for pair in table['tables']:
        for sites in pair:
            for curve in sites:
                params=curve['curve_parameters'];zero=curve['nodes'][list(curve['grid']).index(0.)]
                assert zero['dual_slope']==pytest.approx(float(params['b']@zero['residual']))
                assert 'physical_parameter' not in zero['source_QP']
                assert set(zero['source_QP'])>={'success','optimality','active_lower','active_upper','seconds'}
                for p in (-1.,0.,1.):
                    out=design.scalar.scalar_profile(params['c'],params['b'],params['G'],params['source_lower'],
                        params['source_upper'],params['source_center'],p)
                    assert zero['dual_intercept']+zero['dual_slope']*p<=out['primal_upper']+1e-10


@pytest.mark.parametrize('certificate',['missing','failed'])
def test_feasible_real_MILP_with_missing_or_failed_action_QP_retains_baseline(monkeypatch,certificate):
    ctx,u=_real_context();op=fixed._service(4,5,mobile=False,discharge=True)
    required=len(ctx['pairs'])
    def no_certificate(*args,**kwargs):
        if certificate=='missing':return None
        return dict(success=False,dual_lower=None,primal_upper=None,
            pairs=[dict(success=False,failure='declared certificate fit failure') for _ in range(required)])
    monkeypatch.setattr(design,'action_bounds',no_certificate)
    selected,record=design.scalar_policy(ctx,op,dict(constant=u),grids=(3,),total_seconds=15.,
        fit_options=dict(multistart=1,max_nfev=100),backend='scipy',seed=733)
    assert selected==pytest.approx(u) and record['success']
    raw=record['grids'][0]['raw']
    assert raw['operation']['feasible'] and raw['status']==0
    assert not raw['success'] and not record['grids'][0]['accepted']
    assert record['grid_failure_reasons']==['no complete feasible raw MILP proposal']
    assert record['primary_MILP_calls']==1 and record['primary_MILP_feasible']==0
    assert record['action_QP_calls']==(0 if certificate=='missing' else required)
    assert record['inheritance_protected'] and not record['all_requested_grids_processed']
