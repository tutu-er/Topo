"""Same-data/cone fits, block meter broadcasts, local witnesses and cache scope."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_systematic_refit as study
from . import likelihood as lk
from .ev_fixed_train_fit import subset_model,subset_observations
from .ev_rx_fit import fit_rx_inputs


OPTIONS=dict(r_bounds=(.002,.04),x_bounds=(.0002,.02),ratio_bounds=(.12,.5),
    source_p_bounds=(-.3,.3),source_q_bounds=(-.3,.3),multistart=1,max_nfev=180,tol=1e-10)


def case(seed=71,noisy=True):
    rng=np.random.default_rng(seed);T=21;n=3
    candidate=lk.RadialCandidate.from_supports('star',n,[])
    r=np.linspace(.008,.022,candidate.m);x=r*np.linspace(.2,.4,candidate.m)
    p=rng.uniform(.025,.17,(T,n));q=p*rng.uniform(.12,.45,(T,n))
    sp=rng.uniform(-.02,.07,T);sq=rng.uniform(-.01,.025,T)
    root=rng.uniform(.997,1.004,T)
    model=lk.MeasurementModel(.0007,.0004,
        .00012+np.arange(T*n).reshape(T,n)*1e-6,
        .0006+np.arange(T)*1e-5,.0008+np.arange(T)*2e-6,
        bias_p=np.array([.0008,-.0003,.0005]),
        bias_q=np.arange(T*n).reshape(T,n)*1e-6,bias_v=np.array([.0002,-.0001,.0001]))
    physical=lk.ac_predict(candidate,r,x,p,q,sp,sq,root)
    v=physical['v']+model.bias_v;mp=physical['master_p'].copy();mq=physical['master_q'].copy()
    if noisy:
        v+=rng.normal(size=(T,n))*model.v_sigma
        mp+=rng.normal(size=T)*model.master_p_sigma
        mq+=rng.normal(size=T)*model.master_q_sigma
    observations=lk.Observations(p+model.bias_p,q+model.bias_q,v,root,mp,mq)
    ids=(slice(0,12),slice(12,15),slice(15,None))
    blocks=tuple(subset_observations(observations,index) for index in ids)
    models=tuple(subset_model(model,observations,index) for index in ids)
    return candidate,blocks,models,dict(r=r,x=x,source_p=sp,source_q=sq),observations,model


def test_concat_models_broadcasts_each_original_block_before_joining():
    _,blocks,models,_,obs,model=case()
    merged=study.concat_models(blocks,models)
    for name in ('p','q','v'):
        np.testing.assert_array_equal(getattr(merged,name+'_sigma'),
            lk._array(getattr(model,name+'_sigma'),(obs.t,obs.n),name))
        np.testing.assert_array_equal(getattr(merged,'bias_'+name),
            lk._array(getattr(model,'bias_'+name),(obs.t,obs.n),name))
    for name in ('master_p','master_q'):
        np.testing.assert_array_equal(getattr(merged,name+'_sigma'),getattr(model,name+'_sigma'))
    with pytest.raises(ValueError,match='known calibrated'):
        study.concat_models((blocks[0],),(replace(models[0],fit_bias=True),))


def test_paired_totals_same_observations_cone_normalizer_and_block_losses():
    candidate,blocks,models,_,_,_=case()
    h,v,e=blocks;hm,vm,em=models;counter={}
    out=study.compare_fixed_and_joint(candidate,h,v,e,hm,vm,em,fit_options=OPTIONS,seed=22,call_counter=counter)
    assert out['success'],out
    assert out['checks']['same_observations_and_normalizer']
    assert out['checks']['joint_feasible_attainment_no_worse_than_fixed']
    assert out['joint']['joint_total']<=out['fixed']['fixed_total']+1e-7
    for route in ('fixed','joint'):
        post=out[route]['post'];state=post['state']
        assert post['loss']==pytest.approx(sum(row['loss'] for row in post['blocks'].values()),abs=1e-10)
        residual=np.concatenate([np.asarray(post['blocks'][name]['residual']) for name in ('training','validation','future')])
        assert post['loss']==pytest.approx(float(np.sum(residual*residual)),abs=1e-10)
        assert post['loglik']==pytest.approx(-.5*(post['loss']+post['normalizer']),abs=1e-10)
        ratio=np.asarray(state['x'])/state['r']
        assert ratio.min()>=OPTIONS['ratio_bounds'][0]-1e-12
        assert ratio.max()<=OPTIONS['ratio_bounds'][1]+1e-12
    fixed=out['fixed']
    assert fixed['fixed_holdout']==pytest.approx(sum(fixed['post']['blocks'][k]['loss'] for k in ('validation','future')))
    assert fixed['future_increment']==pytest.approx(fixed['post']['blocks']['future']['loss'])
    assert out['joint']['profile_increment']==pytest.approx(out['joint']['joint_total']-out['joint']['pre']['loss'])
    assert counter['fit_calls']==5  # train, fixed val, joint H+V, fixed E, joint H+V+E
    assert all(not out[route]['post'].get('global_optimality_certified',False) for route in ('fixed','joint'))


def test_cache_reuses_only_history_and_fixed_theta_does_not_change_with_future():
    candidate,blocks,models,_,_,_=case(noisy=False)
    h,v,e=blocks;hm,vm,em=models;counter={}
    history=study.prepare_refit_history(candidate,h,v,hm,vm,fit_options=OPTIONS,seed=24,call_counter=counter)
    count=counter['fit_calls']
    first=study.compare_fixed_and_joint(candidate,h,v,e,hm,vm,em,history_cache=history,
        fit_options=OPTIONS,seed=24,call_counter=counter)
    changed=replace(e,v=e.v+np.array([.00008,-.00005,.00003]))
    second=study.compare_fixed_and_joint(candidate,h,v,changed,hm,vm,em,history_cache=history,
        fit_options=OPTIONS,seed=24,call_counter=counter)
    assert counter['fit_calls']==count+4
    assert first['checks']['history_cache_reused'] and second['checks']['history_cache_reused']
    assert not second['cost']['history_cost_included_this_call']
    for name in ('r','x'):
        np.testing.assert_array_equal(first['fixed']['post']['state'][name],getattr(history.training_fit,name))
        np.testing.assert_array_equal(first['fixed']['post']['state'][name],second['fixed']['post']['state'][name])
    assert first['joint']['pre']==second['joint']['pre']
    assert first['fixed']['fixed_total']<1e-10
    with pytest.raises(ValueError,match='history cache differs'):
        study.compare_fixed_and_joint(candidate,h,replace(v,v=v.v+.00001),e,hm,vm,em,
            history_cache=history,fit_options=OPTIONS)


def test_supplied_training_fit_must_be_conditional_cone_and_match_meters():
    candidate,blocks,models,_,_,_=case(noisy=False)
    h,v,_=blocks;hm,vm,_=models
    trained=fit_rx_inputs(candidate,h,hm,**OPTIONS)
    state=study._state(trained)
    fake=SimpleNamespace(**state,success=True,candidate=candidate.name,loss=trained.loss,diagnostics={})
    with pytest.raises(ValueError,match='documented direct P/Q and X/R cone'):
        study.prepare_refit_history(candidate,h,v,hm,vm,training_fit=fake,fit_options=OPTIONS)
    with pytest.raises(ValueError,match='training loss does not match'):
        study.prepare_refit_history(candidate,h,v,replace(hm,bias_v=hm.bias_v+.0002),vm,
            training_fit=trained,fit_options=OPTIONS)


def test_local_optimizer_regression_retains_legal_witness_and_reports_failure(monkeypatch):
    candidate,blocks,models,truth,_,_=case(noisy=False)
    obs=lk.concat_observations(*blocks);model=study.concat_models(blocks,models)
    good=truth
    bad={**truth,'r':truth['r']*1.4,'x':truth['x']*1.4}
    fake=SimpleNamespace(**bad,success=True,candidate=candidate.name,loss=1234.,loglik=-1234.,
        status=2,message='synthetic local regression',nfev=5,optimality=0.,diagnostics={})
    monkeypatch.setattr(study,'fit_rx_inputs',lambda *args,**kwargs:fake)
    result,state=study._stage(candidate,obs,model,study._options(OPTIONS),[('known_feasible',good)],
        tuple(o.t for o in blocks),('training','validation','future'),11,None)
    assert not result['success'] and result['optimization_failure']
    assert result['selected_origin']=='feasible_witness:known_feasible'
    assert result['loss']<1e-20
    assert result['optimizer_recomputed_score']['loss']>1.
    assert result['optimizer']['message']=='synthetic local regression'
    np.testing.assert_array_equal(state['r'],truth['r'])
    assert not result['global_optimality_certified']


def test_common_box_and_unsupported_options_are_explicit():
    with pytest.raises(ValueError,match='common scalar lower/upper'):
        study._options(dict(source_p_bounds=(np.zeros(3),np.ones(3))))
    with pytest.raises(ValueError,match='unsupported fit options'):
        study._options(dict(truth_r=np.ones(3)))
