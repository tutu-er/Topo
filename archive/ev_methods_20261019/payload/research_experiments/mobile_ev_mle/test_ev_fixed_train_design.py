"""Fixed-training line invariants, exact derivatives and proxy bound checks."""
from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_fixed_train_design as design
from . import ev_original_domain_milp as od
from . import ev_shared_square_milp as squares
from . import likelihood as lk
from .candidates import RadialCandidate
from .simulation import model


@pytest.fixture(scope='module')
def context():
    bank=[RadialCandidate.from_supports('star',3,()),
          RadialCandidate.from_supports('clade_01',3,((0,1),))]
    tm=model(3);T=3
    fm=replace(tm,v_sigma=np.tile(np.array([.00012,.0001,.00014]),(T,1)),
        master_p_sigma=np.full(T,.001),master_q_sigma=np.full(T,.001))
    p=np.array([[.11,.12,.13],[.12,.11,.13],[.13,.12,.11],[.115,.125,.12]])
    q=np.full_like(p,.035);sp=np.array([.08,.1,.09,.11]);sq=np.array([.03,.035,.025,.03])
    fits={}
    for c in bank:
        r=np.full(c.m,.004);x=np.full(c.m,.0025)
        fits[c.name]=SimpleNamespace(r=r,x=x,p=p.copy(),q=q.copy(),source_p=sp.copy(),
            source_q=sq.copy(),success=True,loss=1.)
    phy=lk.ac_predict(bank[0],fits['star'].r,fits['star'].x,p,q,sp,sq,np.ones(4))
    obs=lk.Observations(p=p+tm.bias_p,q=q+tm.bias_q,v=phy['v']+tm.bias_v,
        root_v=np.ones(4),master_p=phy['master_p'],master_q=phy['master_q'])
    reference=np.zeros((T,3));reference[:,0]=2.
    ctx=design.build_fixed_contexts(obs,bank,fits,tm,fm,-6.,6.,reference)
    assert ctx['success'],ctx['diagnostics']
    return ctx,obs,bank,fits,tm,fm


def _direct_residual(ctx, m, u, source):
    """Independent full AC vector; no local Jacobian is used here."""
    bank={c.name:c for c in ctx['bank']};totalp=ctx['background']['p']+u/ctx['base_kw']
    q=ctx['background']['q'];T,n=u.shape
    values=[]
    for name,spq in ((m['b'],source),(m['a'],m['reference_source'])):
        sp,sq=np.asarray(spq).reshape(T,2).T;line=ctx['lines'][name]
        phy=lk.ac_predict(bank[name],line['r'],line['x'],totalp,q,sp,sq,ctx['root_v'])
        values.append(np.r_[((phy['v']+np.broadcast_to(ctx['future_meter'].bias_v,(T,n)))/
            np.broadcast_to(ctx['future_meter'].v_sigma,(T,n))).ravel(),
            phy['master_p']/np.broadcast_to(ctx['future_meter'].master_p_sigma,(T,)),
            phy['master_q']/np.broadcast_to(ctx['future_meter'].master_q_sigma,(T,))])
    return values[0]-values[1]


def test_context_has_source_only_coordinates_and_training_only_forecast(context):
    ctx,obs,bank,fits,tm,fm=context
    assert len(ctx['pairs'])==len(bank)*(len(bank)-1)==2
    assert ctx['diagnostics']['residual_rows']==15
    assert ctx['diagnostics']['uses_validation'] is False
    for m,pair in zip(ctx['models'],ctx['pairs']):
        assert pair.c.shape==(15,) and pair.B.shape==(15,9)
        assert pair.A0.shape==(15,6)
        assert np.count_nonzero(pair.At)==np.count_nonzero(pair.shift)==0
        assert len(pair.lower)==6 and pair.lower==pytest.approx(np.full(6,-.5))
        assert pair.residual(ctx['reference_kw'].ravel(),pair.xi0)==pytest.approx(m['reference_residual'],abs=1e-11)
        assert ctx['forecasts'][m['a']]['p']==pytest.approx(
            np.tile((obs.p-tm.bias_p).mean(axis=0),(3,1))+ctx['reference_kw']/20.)
        assert ctx['lines'][m['b']]['r']==pytest.approx(fits[m['b']].r)
        assert not ctx['lines'][m['b']]['r'].flags.writeable
        assert not ctx['lines'][m['b']]['x'].flags.writeable


def test_complete_AC_control_and_source_derivatives_match_finite_differences(context):
    ctx,*_=context;m=ctx['models'][0];pair=ctx['pairs'][0];ref=ctx['reference_kw'];xi=pair.xi0
    for k in (0,4,8):
        eps=.001;plus=ref.copy();minus=ref.copy();plus.ravel()[k]+=eps;minus.ravel()[k]-=eps
        derivative=(_direct_residual(ctx,m,plus,xi)-_direct_residual(ctx,m,minus,xi))/(2*eps)
        assert pair.B[:,k]==pytest.approx(derivative,rel=2e-6,abs=2e-7)
    for j in (0,3,5):
        eps=1e-6;plus=xi.copy();minus=xi.copy();plus[j]+=eps;minus[j]-=eps
        derivative=(_direct_residual(ctx,m,ref,plus)-_direct_residual(ctx,m,ref,minus))/(2*eps)
        assert pair.A0[:,j]==pytest.approx(derivative,rel=2e-6,abs=5e-6)


def test_source_QP_brackets_its_affine_cost_and_respects_original_box(context):
    ctx,*_=context;u=ctx['reference_kw'].copy();u[1,0]=4.
    for pair in ctx['pairs']:
        qp=od.evaluate_original_pair(pair,u.ravel())
        assert qp['success']
        assert qp['tangent_lower_numeric']<=qp['primal_upper']+1e-8
        assert qp['primal_upper']-qp['tangent_lower_numeric']<1e-5
        assert np.all(qp['physical_parameter']>=pair.lower) and np.all(qp['physical_parameter']<=pair.upper)
        r=pair.residual(u.ravel(),qp['physical_parameter'])
        assert qp['primal_upper']==pytest.approx(float(r@r),abs=1e-8)


def test_full_AC_profiles_keep_lines_and_PQ_exact_and_EV_once(context):
    ctx,*_=context;u=ctx['reference_kw'].copy();u[1,0]=4.;u[2]=[0.,-1.,0.]
    before={k:(v['r'].copy(),v['x'].copy()) for k,v in ctx['lines'].items()}
    result=design.evaluate_fixed_action(ctx,u,fit_options=dict(multistart=1,max_nfev=120),seed=82)
    assert result['success'],result
    assert result['line_parameters_unchanged']
    assert len(result['pairs'])==len(ctx['pairs'])
    assert result['cap20_mean']<=result['mean']
    for entry in result['pairs']:
        fit=result['fit_results'][entry['name']];b=entry['b']
        assert np.array_equal(fit.r,before[b][0]) and np.array_equal(fit.x,before[b][1])
        assert fit.p==pytest.approx(ctx['background']['p']+u/20.,abs=1e-14)
        assert fit.q==pytest.approx(ctx['background']['q'],abs=1e-14)
        assert entry['added']==pytest.approx(.5*fit.loss)
        assert entry['source_interface']==1
    for name,line in ctx['lines'].items():
        assert np.array_equal(line['r'],before[name][0]) and np.array_equal(line['x'],before[name][1])


def test_AC_witness_replay_has_exact_reference_and_explicit_channel_gate(context):
    ctx,*_=context
    sources={p.name:p.xi0 for p in ctx['pairs']}
    check=design.check_fixed_witness(ctx,ctx['reference_kw'],source_by_pair=sources)
    assert check['success'] and check['gate_passed']
    for r in check['pairs']:
        assert set(r['channel_error'])=={'v','master_p','master_q'}
        assert r['max_channel_RMS']<1e-8
        assert r['complete_AC_residual']==pytest.approx(r['proxy_residual'],abs=1e-8)
    moved=ctx['reference_kw'].copy();moved[1]=[0.,5.7,0.]
    changed=design.check_fixed_witness(ctx,moved,gate=0.)
    assert changed['success'] and not changed['gate_passed']
    assert len(changed['pairs'])==len(ctx['pairs'])
    assert any(r['max_channel_RMS']>1e-8 for r in changed['pairs'])


def test_shared_square_structure_is_proved_from_single_EV_domain(context):
    ctx,*_=context;d=design._service(3,3,mobile=True,discharge=True);data=d.as_dict()
    op=od._operation(9,data['control_lower'],data['control_upper'],None,data)
    zeros=set(data['zero_product_pairs']);audit=squares.certify_operation_zero_products(op,zeros)
    assert audit['valid']
    for pair in ctx['pairs']:
        poly=od.physical_witness_polynomial(pair,dict(physical_intercept=pair.xi0,
            physical_slope=np.zeros_like(pair.shift)),data['control_lower'],data['control_upper'],operation=op)
        assert poly['products']=={}
        compiled=squares.compile_shared_square_witness(pair,poly,data['control_lower'],data['control_upper'],zeros)
        assert np.all(compiled['diagonal']>=0)
        assert compiled['cross_numeric_guard']>=0 and compiled['upward_cost_guard']>=0


def test_QP_lower_and_MILP_upper_enclose_known_profile_maximum_with_cap():
    pair=od.OriginalDomainPair('toy',[0.,0.],np.eye(2),[[1.],[0.]],
        np.zeros((2,2,1)),[-.1],[.1],[0.],np.zeros((1,2)))
    common=dict(master_form='shared_control_square',initial_u=[.5,.5],backend='scipy',
        max_iterations=3,control_segments=4,residual_segments=4,max_segments=8,
        solver_seconds=2.,total_seconds=10.,absolute_gap=1e-6,relative_gap=0.)
    out=od.exchange_original_domain([pair],[0.,0.],[1.,1.],**common)
    exact=.5*((1.-.1)**2+1.)
    assert out['objective_lower']<=exact+1e-8<=out['objective_upper']+1e-8
    assert out['objective_lower']==pytest.approx(exact,abs=1e-7)
    capped=od.exchange_original_domain([pair],[0.,0.],[1.,1.],score_caps=[.5],**common)
    assert capped['objective_lower']<=.5+1e-8<=capped['objective_upper']+1e-8
    assert capped['objective_lower']==pytest.approx(.5,abs=1e-7)
    assert all(not it['master']['incumbent_is_true_score_lower'] for it in out['iterations'])


def test_selector_siblings_share_initials_random_is_postposed_and_validation_absent(context,monkeypatch):
    ctx,obs,bank,fits,tm,fm=context;events=[]
    def predicted(context,power,**kwargs):
        score=float(np.sum(np.asarray(power)**2))
        return dict(success=True,pairs=[dict(added=score,weight=.5),dict(added=score/2,weight=.5)],
            cost=dict(fit_calls=0,ac_calls=0),line_parameters_unchanged=True)
    def exchange(pairs,lo,hi,**kwargs):
        events.append('cap' if kwargs['score_caps'] is not None else 'mean')
        x=kwargs['initial_operation_points'][0];u=kwargs['operation']['u_map']@x
        return dict(u=u,completed=False,unfinished=True,objective_lower=0.,objective_upper=1.,gap=1.)
    class Generator:
        def normal(self,*,size):
            events.append('random');return np.ones(size)
    monkeypatch.setattr(design,'evaluate_fixed_action',predicted)
    monkeypatch.setattr(design,'exchange_original_domain',exchange)
    monkeypatch.setattr(design,'check_fixed_witness',lambda *a,**k:dict(gate_passed=True,pairs=[]))
    monkeypatch.setattr(design.np.random,'default_rng',lambda seed:Generator())
    choices,out=design.select_fixed_controls(obs,bank,fits,tm,fm,seed=23,
        modes=('refined_v2g_mobile','capped_v2g_mobile','random_mobile'),exchange_options=dict(backend='scipy'))
    assert events[-1]=='random' and events.count('cap')==1 and events.count('mean')==5
    assert out['uses_validation'] is False and out['random_used_by_primary'] is False
    a=out['methods']['refined_v2g_mobile']['initial_candidates']
    b=out['methods']['capped_v2g_mobile']['initial_candidates']
    assert len(a)==len(b)
    for x,y in zip(a,b):
        assert x['origin']==y['origin'] and 'random' not in x['origin']
        assert np.array_equal(x['power_kw'],y['power_kw'])
    assert all(choices[k] is not None for k in choices)
    assert 'validation' not in signature(design.select_fixed_controls).parameters
