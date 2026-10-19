"""Conditional DGP prefix, moments, pairing, and fixed-line integration checks."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_response_conditional_diagnostic as diagnostic
from . import ev_dynamic_loads as loads
from . import likelihood as lk
from .simulation import model


def _record(H=24):
    background=loads.generate_background(45871,H+10,n=3)
    return dict(background=background,training=dict(p=np.zeros((H,3))))


def test_saved_true_training_prefix_is_exact_and_does_not_require_future_values():
    record=_record();out=diagnostic.reconstruct_prefix(record)
    assert all(row['exact'] and row['max_abs']==0. for row in out['checks'].values())
    changed=deepcopy(record)
    H=out['H']
    for key in ('p_kw','q_kvar','pf','source_p_kw','source_q_kvar','root_v'):
        changed['background'][key][H:]+=123.
    again=diagnostic.reconstruct_prefix(changed)
    assert again['state_sha256']==out['state_sha256']
    assert again['physical_prefix_sha256']==out['physical_prefix_sha256']
    changed['background']['q_kvar'][0,0]+=.1
    with pytest.raises(ValueError,match='exact reconstruction'):diagnostic.reconstruct_prefix(changed)


def test_continuation_matches_closed_form_weighted_innovations_and_gap_index():
    prefix=diagnostic.reconstruct_prefix(_record())
    seeds={name:321+i for i,name in enumerate(diagnostic.stream_specs(prefix['config'],3))}
    output=diagnostic.continue_latent(prefix['state'],prefix['config'],3,10,seeds)
    for name,(mean,sd,phi) in diagnostic.stream_specs(prefix['config'],3).items():
        eps=np.random.default_rng(seeds[name]).standard_normal((10,len(mean)))
        for step in (1,5,10):
            innovation=sum(phi**(step-1-j)*eps[j] for j in range(step))*sd*np.sqrt(1-phi**2)
            expected=mean+phi**step*(prefix['state'][name]-mean)+innovation
            assert output[name][step-1]==pytest.approx(expected,rel=1e-13,abs=1e-14)
    future,meta=diagnostic.conditional_path(prefix,995901,0,32,4,6)
    assert meta['first_future_step']==5
    assert len(future['p_kw'])==6
    full=diagnostic.physical_from_latent(diagnostic.continue_latent(prefix['state'],prefix['config'],3,10,
        meta['stream_seeds']),prefix['config'])
    assert future['p_kw']==pytest.approx(full['p_kw'][4:])


def test_analytic_conditional_moments_match_gaussian_AR_simulation():
    mu,sd,phi,x0=2.,.15,.9,2.4
    h=np.array([1,5,10]);mean,variance=diagnostic.conditional_moments([x0],[mu],sd,phi,h)
    assert mean[:,0]==pytest.approx(mu+phi**h*(x0-mu))
    assert variance==pytest.approx(sd**2*(1-phi**(2*h)))
    rng=np.random.default_rng(77819);current=np.full(50000,x0);saved=[]
    for step in range(1,11):
        current=mu+phi*(current-mu)+sd*np.sqrt(1-phi**2)*rng.standard_normal(len(current))
        if step in h:saved.append(current.copy())
    for index,sample in enumerate(saved):
        assert sample.mean()==pytest.approx(mean[index,0],abs=4*np.sqrt(variance[index]/len(sample)))
        assert sample.var(ddof=1)==pytest.approx(variance[index],rel=.025)


def test_conditional_paths_deterministic_and_independent_of_saved_future():
    prefix=diagnostic.reconstruct_prefix(_record())
    a,ma=diagnostic.conditional_path(prefix,995901,7,32,4,6)
    b,mb=diagnostic.conditional_path(prefix,995901,7,32,4,6)
    c,mc=diagnostic.conditional_path(prefix,995901,8,32,4,6)
    assert ma==mb and ma['stream_seeds']!=mc['stream_seeds']
    assert a['p_kw']==pytest.approx(b['p_kw'])
    assert not np.array_equal(a['p_kw'],c['p_kw'])
    pf=a['p_kw']/np.hypot(a['p_kw'],a['q_kvar'])
    assert pf.min()>=.94-1e-12 and pf.max()<=.99+1e-12


def test_mc_standard_error_uses_paths_only_for_integration_and_keeps_failures():
    summary=diagnostic.mc_stats([1.,2.,3.,None],4)
    assert summary['mean']==2. and summary['complete']==3 and summary['failed']==1
    assert summary['MC_standard_error']==pytest.approx(1/np.sqrt(3))
    assert summary['independent_history_count']==1
    assert not summary['complete_path_integral_available']
    assert 'successful paths only' in summary['mean_scope']


def test_action_deduplication_keeps_all_six_method_aliases():
    u=np.zeros((6,3));u[:,0]=2.
    methods=[dict(name=name,power_kw=u.copy(),clean_future_M=2.,M=1.) for name in diagnostic.CONFIG['methods']]
    record=dict(methods=methods,planning_diagnostics=dict(methods={name:dict(future_profile_distance=3.)
        for name in diagnostic.CONFIG['methods']}))
    actions,aliases,original=diagnostic.gather_actions(record)
    assert len(actions)==1 and len(original)==6
    assert next(iter(aliases.values()))==diagnostic.CONFIG['methods']


def test_real_noiseless_AC_source_fits_keep_each_training_line_exact():
    bank=[lk.RadialCandidate.from_supports('star',3,()),lk.RadialCandidate.from_supports('clade_01',3,((0,1),))]
    fitted={c.name:SimpleNamespace(r=np.full(c.m,.004),x=np.full(c.m,.001),
        source_p=np.full(24,.1),source_q=np.full(24,.035)) for c in bank}
    future=dict(p_kw=np.full((6,3),2.4),q_kvar=np.full((6,3),.6),source_p_kw=np.full(6,2.),
        source_q_kvar=np.full(6,.7),root_v=np.ones(6))
    record=dict(true_name='star',truth=dict(r=fitted['star'].r,x=fitted['star'].x),
        config=dict(profile=dict(multistart=1,max_nfev=120,tol=1e-10),
            source=dict(source_p_bounds=[-.5,.5],source_q_bounds=[-.5,.5])))
    val=dict(star=3.,clade_01=4.)
    out=diagnostic.evaluate_action(future,np.zeros((6,3)),record,bank,fitted,val,model(3),seed=829)
    assert out['success'] and out['line_parameters_unchanged']
    assert out['M']==pytest.approx(out['loss_by_candidate']['clade_01']-out['loss_by_candidate']['star'])
    assert out['source_fits']['all_complete']
    assert out['observations']['p']==pytest.approx(np.full((6,3),.12)+model(3).bias_p)
