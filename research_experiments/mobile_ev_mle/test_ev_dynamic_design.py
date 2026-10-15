"""Dynamic forecast derivatives, common AC acceptance and matched EV baselines."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_fixed_train_design as design
from . import ev_original_domain_milp as od
from . import likelihood as lk
from .candidates import RadialCandidate
from .simulation import model
from .test_ev_fixed_train_design import _direct_residual


@pytest.fixture
def data():
    bank=[RadialCandidate.from_supports('star',3,()),
          RadialCandidate.from_supports('clade_01',3,((0,1),))]
    tm=model(3);T=3
    fm=replace(tm,v_sigma=np.full((T,3),.00012),
        master_p_sigma=np.full(T,.001),master_q_sigma=np.full(T,.001))
    p=np.array([[.11,.12,.13],[.12,.11,.13],[.13,.12,.11],[.115,.125,.12]])
    q=np.full_like(p,.035);sp=np.array([.08,.1,.09,.11]);sq=np.array([.03,.035,.025,.03])
    fits={c.name:SimpleNamespace(r=np.full(c.m,.004),x=np.full(c.m,.0025),
        p=p.copy(),q=q.copy(),source_p=sp.copy(),source_q=sq.copy(),success=True,loss=1.) for c in bank}
    phy=lk.ac_predict(bank[0],fits['star'].r,fits['star'].x,p,q,sp,sq,np.ones(4))
    obs=lk.Observations(p=p+tm.bias_p,q=q+tm.bias_q,v=phy['v']+tm.bias_v,
        root_v=np.ones(4),master_p=phy['master_p'],master_q=phy['master_q'])
    forecast=dict(p=np.array([[.10,.13,.11],[.15,.08,.14],[.09,.16,.10]]),
        q=np.array([[.03,.04,.02],[.05,.02,.04],[.02,.05,.03]]),root=np.array([1.,.999,1.001]),
        source_by_candidate={c.name:dict(source_p=np.array([.08,.11,.09])+.002*i,
            source_q=np.array([.025,.035,.03])) for i,c in enumerate(bank)},
        metadata=dict(training_only=True,uses_validation=False,uses_truth=False,
            uses_future_noise=False,source_interface=1,method='predeclared test training forecast'))
    reference=np.zeros((T,3));reference[:,0]=2.
    return obs,bank,fits,tm,fm,forecast,reference


def _context(data,**options):
    obs,bank,fits,tm,fm,_,ref=data
    result=design.build_fixed_contexts(obs,bank,fits,tm,fm,-6.,6.,ref,**options)
    assert result['success'],result['diagnostics']
    return result


def test_default_context_is_numerically_unchanged_and_defaults_stay_frozen(data):
    first=_context(data);second=_context(data,forecast=None)
    for a,b in zip(first['pairs'],second['pairs']):
        for key in ('c','B','A0','At','base_center','shift','physical_lower','physical_upper'):
            assert np.array_equal(getattr(a,key),getattr(b,key))
    assert len(design.METHODS)==11
    assert 'quality_v2g_fixed' not in design.METHODS
    assert 'quality_v2g_fixed' in design.KNOWN_METHODS
    assert np.array_equal(first['background']['p'],np.tile(
        (data[0].p-data[3].bias_p).mean(axis=0),(3,1)))


def test_explicit_dynamic_forecast_derivatives_match_independent_AC(data):
    ctx=_context(data,forecast=data[5]);ref=data[6]
    assert ctx['diagnostics']['forecast_metadata']['training_only'] is True
    assert ctx['background']['p']==pytest.approx(data[5]['p'])
    assert ctx['background']['q']==pytest.approx(data[5]['q'])
    for m,pair in zip(ctx['models'],ctx['pairs']):
        source=data[5]['source_by_candidate'][m['b']]
        assert pair.xi0==pytest.approx(np.column_stack((source['source_p'],source['source_q'])).ravel())
        for k in (0,4,8):
            eps=.001;plus=ref.copy();minus=ref.copy();plus.ravel()[k]+=eps;minus.ravel()[k]-=eps
            fd=(_direct_residual(ctx,m,plus,pair.xi0)-_direct_residual(ctx,m,minus,pair.xi0))/(2*eps)
            assert pair.B[:,k]==pytest.approx(fd,rel=3e-6,abs=2e-7)
        for k in (0,3,5):
            eps=1e-6;plus=pair.xi0.copy();minus=pair.xi0.copy();plus[k]+=eps;minus[k]-=eps
            fd=(_direct_residual(ctx,m,ref,plus)-_direct_residual(ctx,m,ref,minus))/(2*eps)
            assert pair.A0[:,k]==pytest.approx(fd,rel=3e-6,abs=5e-6)
    # The dynamic path genuinely changes same-site coefficients between slots.
    block=ctx['pairs'][0].B
    assert not np.allclose(block[:3,0],block[3:6,3],rtol=1e-6,atol=1e-10)


def test_root_alias_and_rebuild_keep_unique_training_lines_and_forecasts(data):
    forecast=deepcopy(data[5]);forecast['root_v']=forecast.pop('root')
    ctx=_context(data,forecast=forecast);changed=data[6].copy();changed[1,1]=1.5
    rebuilt=design.rebuild_fixed_context(ctx,changed)
    assert rebuilt['success'],rebuilt['diagnostics']
    assert rebuilt['diagnostics']['old_proxy_bounds_carried'] is False
    assert rebuilt['diagnostics']['forecast_sha256']==ctx['diagnostics']['forecast_sha256']
    for name,line in ctx['lines'].items():
        assert rebuilt['lines'][name]['sha256']==line['sha256']
        for channel in ('source_p','source_q'):
            assert np.array_equal(rebuilt['forecasts'][name][channel],ctx['forecasts'][name][channel])
        assert rebuilt['forecasts'][name]['p']==pytest.approx(forecast['p']+changed/20.)


@pytest.mark.parametrize('fault',('shape','nonfinite','source_missing','source_interface',
    'validation','truth','noise','root_conflict','source_box'))
def test_explicit_forecast_rejects_invalid_dimensions_or_provenance(data,fault):
    f=deepcopy(data[5])
    if fault=='shape':f['p']=f['p'][:1]
    elif fault=='nonfinite':f['q'][0,0]=np.nan
    elif fault=='source_missing':f['source_by_candidate'].pop('star')
    elif fault=='source_interface':f['metadata']['source_interface']=2
    elif fault=='validation':f['metadata']['uses_validation']=True
    elif fault=='truth':f['metadata']['uses_truth']=True
    elif fault=='noise':f['metadata']['uses_future_noise']=True
    elif fault=='root_conflict':f['root_v']=f['root']+.01
    elif fault=='source_box':f['source_by_candidate']['star']['source_p'][0]=.6
    obs,bank,fits,tm,fm,_,ref=data
    result=design.build_fixed_contexts(obs,bank,fits,tm,fm,-6.,6.,ref,forecast=f)
    assert not result['success'] and result['diagnostics']['failure']
    assert result['pairs']==[]


def test_dynamic_full_AC_score_keeps_PQ_lines_and_source_interface(data):
    ctx=_context(data,forecast=data[5]);u=data[6].copy();u[1,0]=4.;u[2,1]=-1.
    result=design.evaluate_fixed_action(ctx,u,fit_options=dict(multistart=1,max_nfev=120),seed=11)
    assert result['success'],result
    for entry in result['pairs']:
        fit=result['fit_results'][entry['name']];line=ctx['lines'][entry['b']]
        assert np.array_equal(fit.r,line['r']) and np.array_equal(fit.x,line['x'])
        assert fit.p==pytest.approx(data[5]['p']+u/20.,abs=1e-14)
        assert fit.q==pytest.approx(data[5]['q'],abs=1e-14)
        assert entry['source_interface']==1


def test_half_SSE_profile_gradient_has_correct_sign_and_per_kw_scale(data):
    ctx=_context(data,forecast=data[5]);u=data[6].copy();u[1,1]=1.2;u[2,0]=-.5
    answer=design.fixed_profile_gradient(ctx,u)
    assert answer['max_QP_numerical_gap']<1e-5
    def value(action):
        return sum(m['task_weight']*.5*od.evaluate_original_pair(pair,action.ravel())['primal_upper']
            for m,pair in zip(ctx['models'],ctx['pairs']))
    for k in range(u.size):
        eps=1e-4;plus=u.copy();minus=u.copy();plus.ravel()[k]+=eps;minus.ravel()[k]-=eps
        fd=(value(plus)-value(minus))/(2*eps)
        assert answer['gradient_per_kw'].ravel()[k]==pytest.approx(fd,rel=3e-6,abs=1e-6)
    assert answer['mean_half_SSE']==pytest.approx(value(u))
    qps={p.name:od.evaluate_original_pair(p,u.ravel()) for p in ctx['pairs']}
    assert design.fixed_profile_gradient(ctx,u,qps)['gradient_per_kw']==pytest.approx(answer['gradient_per_kw'])
    wrong=u.copy();wrong[0,0]+=.1
    with pytest.raises(ValueError,match='does not belong'):
        design.fixed_profile_gradient(ctx,wrong,qps)


def _mock_planning(monkeypatch,*,bad_raw=False,witness_exception=False):
    """Isolate selection logic; derivative/AC tests above use the real physics."""
    domains=[]
    class Domain:
        def __init__(self,T,n,mobile,discharge):
            self.T=T;self.n=n;self.mobile=mobile;self.discharge=discharge
            self.u=np.zeros((T,n));self.u[:,0]=4.
        def solve_prices(self,prices,**kwargs):return SimpleNamespace(x=self.u.ravel().copy(),status=0)
        def decode(self,result):return dict(u_kw=result.x.reshape(self.T,self.n))
        def check(self,result):return dict(feasible=True)
        def as_dict(self):return dict(control_lower=np.full(self.T*self.n,-6.),
            control_upper=np.full(self.T*self.n,6.),u_map=np.eye(self.T*self.n))
    def service(T,n,*,mobile,discharge):
        d=Domain(T,n,mobile,discharge);domains.append(d);return d
    monkeypatch.setattr(design,'_service',service)
    monkeypatch.setattr(design,'_realize',lambda d,p:dict(success=True,x=np.asarray(p).ravel(),audit=dict(feasible=True)))
    raw=np.zeros((3,3));raw[:,0]=[5.,3.,4.]
    monkeypatch.setattr(design,'exchange_original_domain',lambda *a,**k:dict(u=raw.ravel()))
    def prediction(context,power,**kwargs):
        is_raw=np.array_equal(power,raw);score=float(np.sum(np.asarray(power)**2))
        pairs=[dict(name=p.name,added=score,weight=m['task_weight'],success=not(bad_raw and is_raw))
            for m,p in zip(context['models'],context['pairs'])]
        return dict(success=True,pairs=pairs,cost=dict(fit_calls=0,ac_calls=0),line_parameters_unchanged=True)
    monkeypatch.setattr(design,'evaluate_fixed_action',prediction)
    def witness(*a,**k):
        if witness_exception:raise ValueError('diagnostic failure')
        return dict(success=True,gate_passed=False,pairs=[])
    monkeypatch.setattr(design,'check_fixed_witness',witness)
    return raw,domains


@pytest.mark.parametrize('diagnostic_exception',(False,True))
def test_full_AC_accepts_improving_complete_plan_despite_proxy_gate(data,monkeypatch,diagnostic_exception):
    raw,_=_mock_planning(monkeypatch,witness_exception=diagnostic_exception)
    obs,bank,fits,tm,fm,*_=data
    old,oldrec=design.select_fixed_controls(obs,bank,fits,tm,fm,seed=2,modes=('charge_fixed',))
    new,newrec=design.select_fixed_controls(obs,bank,fits,tm,fm,seed=2,modes=('charge_fixed',),acceptance='full_ac_score')
    assert not np.array_equal(old['charge_fixed'],raw)
    assert np.array_equal(new['charge_fixed'],raw)
    record=newrec['methods']['charge_fixed']
    assert not record['raw_proxy_gate']
    assert record['selected_score']>record['initial_best_score']
    assert record['guard']['all_AC_tasks_required'] is True
    assert oldrec['acceptance_policy']=='legacy_proxy_gate'


def test_full_AC_rejects_incomplete_tasks_and_protects_external_initials(data,monkeypatch):
    raw,_=_mock_planning(monkeypatch,bad_raw=True)
    obs,bank,fits,tm,fm,*_=data
    initials=dict(highlow=dict(power_kw=raw,metadata=dict(deterministic=True,
        uses_validation=False,uses_truth=False,uses_future_noise=False)))
    choices,record=design.select_fixed_controls(obs,bank,fits,tm,fm,seed=2,
        modes=('charge_fixed',),acceptance='full_ac_score',baseline_initials=initials)
    assert not np.array_equal(choices['charge_fixed'],raw)
    rows=record['methods']['charge_fixed']['initial_candidates']
    assert any(r['origin']=='highlow' and r['AC_score'] is None for r in rows)
    with pytest.raises(ValueError,match='non-whitelisted'):
        design.select_fixed_controls(obs,bank,fits,tm,fm,seed=2,modes=('constant',),
            baseline_initials={'random':initials['highlow']})


def test_complete_highlow_initial_is_protected_on_equal_MILP_score(data,monkeypatch):
    raw,_=_mock_planning(monkeypatch)
    obs,bank,fits,tm,fm,*_=data
    initials=dict(highlow=dict(power_kw=raw,metadata=dict(deterministic=True,
        uses_validation=False,uses_truth=False,uses_future_noise=False)))
    choices,record=design.select_fixed_controls(obs,bank,fits,tm,fm,seed=2,
        modes=('charge_fixed',),acceptance='full_ac_score',baseline_initials=initials)
    assert np.array_equal(choices['charge_fixed'],raw)
    assert record['methods']['charge_fixed']['selected_origin']=='highlow'
    assert record['methods']['charge_fixed']['inheritance_protected']
    assert record['deterministic_initial_metadata']['highlow']['deterministic'] is True


def test_all_four_quality_names_use_matching_operation_domains(data,monkeypatch):
    _,domains=_mock_planning(monkeypatch)
    obs,bank,fits,tm,fm,*_=data
    names=('quality_fixed','quality_v2g_fixed','quality_charge_mobile','quality_mobile')
    choices,record=design.select_fixed_controls(obs,bank,fits,tm,fm,seed=3,modes=names)
    assert len(domains)==4
    assert [(d.mobile,d.discharge) for d in domains]==list(design.DOMAIN_FLAGS.values())
    assert set(choices)==set(names)
    assert record['methods']['quality_v2g_fixed']['domain']=='v2g_fixed'
    assert record['methods']['quality_charge_mobile']['domain']=='charge_mobile'
    assert not any(name in record['methods'] for name in design.DOMAIN_FLAGS)


@pytest.mark.parametrize('domain_name',tuple(design.DOMAIN_FLAGS))
def test_early_stay_tie_rule_is_private_and_preserves_precision_objective(domain_name):
    mobile,discharge=design.DOMAIN_FLAGS[domain_name]
    domain=design._service(6,3,mobile=mobile,discharge=discharge)
    rows=deepcopy(domain.model.rows);lo=list(domain.model.row_lower);hi=list(domain.model.row_upper)
    result,record=design._quality_plan(domain,np.ones((6,3)),'early_stay')
    assert record['completed'],record
    assert len(record['stages'])==3 and all(s['ready_for_tie'] for s in record['stages'])
    assert domain.check(result)['feasible']
    assert domain.model.rows==rows and domain.model.row_lower==lo and domain.model.row_upper==hi
    power=domain.decode(result)['u_kw']
    if domain_name=='charge_fixed':
        assert abs(.25*power.sum()-3.)<1e-7
        assert power[0,0]==pytest.approx(design.CONFIG['command_power_kw'],abs=1e-6)
        assert power[1,0]==pytest.approx(design.CONFIG['command_power_kw'],abs=1e-6)
        assert np.max(abs(power[3:]))<1e-6
    # The tie constraint is a lower floor. An improved primary incumbent is
    # allowed; tiny upward differences can also occur within MILP tolerances.
    assert record['primary_score_after']>=record['primary_score_before']-1e-7
