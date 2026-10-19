"""H-only guard checks with analytic mocks and a small actual-AC interface."""
from copy import deepcopy
from types import SimpleNamespace
import inspect
import numpy as np
import pytest
from . import ev_ac_design_guard as guard
from . import likelihood as lk
from .candidates import RadialCandidate
from .simulation import generate,observe,model


def mock_context(monkeypatch, *, candidate_failure=False, worse=False):
    cs=[SimpleNamespace(name=n,n=3) for n in ('a','b','c','d')]
    h=lk.Observations(p=np.full((3,3),.11),q=np.full((3,3),.06),v=np.ones((3,3)),
        root_v=[.99,1.,1.01],master_p=np.ones(3),master_q=np.ones(3))
    hm=lk.MeasurementModel(.01,.01,.01,master_p_sigma=.01,master_q_sigma=.01,
        bias_p=.01,bias_q=.02,bias_v=.003)
    fs={c.name:SimpleNamespace(success=True,loss=float(i+2),r=np.array([.01*(i+1)]),
        x=np.array([.005]),source_p=np.array([i*.01,i*.01,i*.01]),source_q=np.zeros(3))
        for i,c in enumerate(cs)}
    forwards=[];fit_inputs=[]
    def fake_predict(c,r,x,p,q,sp,sq,root):
        forwards.append(dict(name=c.name,p=p.copy(),q=q.copy(),sp=sp.copy(),root=root.copy()))
        t=len(p)
        return dict(p=p,q=q,v=np.full_like(p,1-r[0]),master_p=p.sum(axis=1)+sp,
            master_q=q.sum(axis=1)+sq,source_p=sp,source_q=sq)
    def fake_fits(candidates,obs,meter,*,initial,seed,fit_options):
        fit_inputs.append(dict(obs=obs,seed=seed,fit_options=fit_options))
        source=round((1-(obs.v[-1,0]-.003))/.01)-1
        variation=float(np.sum((obs.p[-2:,0]-obs.p[-2:,0].mean())**2))
        gain=-variation if worse else variation
        fitted={}
        failed={}
        for i,c in enumerate(candidates):
            # Own profile stays at H cost. Cross profiles add an action
            # contrast term; the most difficult candidate remains present.
            increment=0. if i==source else 1.+gain
            f=SimpleNamespace(success=not(candidate_failure and i==3 and variation>0),
                loss=initial[c.name].loss+increment)
            fitted[c.name]=f
            if not f.success:failed[c.name]='fit_nonconverged'
        return fitted,failed,dict(fit_calls=len(candidates),ac_calls=11,seconds=.01)
    monkeypatch.setattr(guard.lk,'ac_predict',fake_predict)
    monkeypatch.setattr(guard.io,'fit_all_candidates',fake_fits)
    monkeypatch.setattr(guard.io,'fit_record',lambda c,f:dict(name=c.name,loss=f.loss,success=f.success))
    return h,cs,fs,hm,forwards,fit_inputs


def test_all_sixteen_predicted_fit_states_and_ev_once(monkeypatch):
    h,cs,fs,m,forward,fit_inputs=mock_context(monkeypatch)
    out=guard.evaluate_predicted_action(h,cs,fs,m,m,m,[0.,6.],site=0,base_kw=20.,seed=19)
    assert out['success'] and len(out['records'])==4 and len(out['pairs'])==12
    assert out['required_fit_states']==out['returned_fit_states']==out['successful_fit_states']==16
    assert out['fit_calls']==16 and out['fit_ac_calls']==44 and out['forecast_calls']==4
    assert out['ac_calls']==48
    assert len(forward)==4 and len(fit_inputs)==4
    for r,i in zip(out['records'],range(4)):
        assert r['forecast_state']['p'][:,0]==pytest.approx([.1,.4])
        assert r['forecast_state']['p'][:,1:]==pytest.approx(np.full((2,2),.1))
        assert r['forecast_state']['q']==pytest.approx(np.full((2,3),.04))
        assert r['forecast_state']['source_p']==pytest.approx([i*.01]*2)
        assert r['predicted_observations']['p'][:,0]==pytest.approx([.11,.41])
        assert len(r['fits'])==4 and r['success']
    assert not out['uses_truth'] and not out['uses_future_noise']
    for pair in out['pairs']:
        assert pair['joint']-pair['added']==pytest.approx(.5*(pair['historical_b']-pair['historical_a']))


def test_guard_uses_full_ac_predicted_scores_to_accept_improved_contrast(monkeypatch):
    h,cs,fs,m,_,fit_inputs=mock_context(monkeypatch)
    out=guard.guard_predicted_actions(h,cs,fs,m,m,m,[3.,3.],[0.,6.],seed=19)
    assert out['accepted'] and out['selected_u']==pytest.approx([0.,6.])
    assert out['score_gain']>0
    assert out['evaluated_actions']==2 and not out['cache_hit']
    assert out['fit_calls']==32 and out['forecast_calls']==8
    assert len(fit_inputs)==8
    assert [r['seed'] for r in out['baseline']['records']]==[r['seed'] for r in out['candidate']['records']]


def test_guard_retains_baseline_on_actual_predicted_score_loss(monkeypatch):
    h,cs,fs,m,_,_=mock_context(monkeypatch,worse=True)
    out=guard.guard_predicted_actions(h,cs,fs,m,m,m,[3.,3.],[0.,6.])
    assert not out['accepted'] and out['selected_u']==pytest.approx([3.,3.])
    assert out['candidate_score']<out['baseline_score']
    assert out['reason']=='attained_local_predicted_score_not_improved'


def test_guard_caches_same_action_without_extra_fits(monkeypatch):
    h,cs,fs,m,_,fits=mock_context(monkeypatch)
    out=guard.guard_predicted_actions(h,cs,fs,m,m,m,[3.,3.],[3.,3.])
    assert out['candidate'] is out['baseline']
    assert out['cache_hit'] and out['evaluated_actions']==1
    assert out['fit_calls']==16 and len(fits)==4
    assert not out['accepted'] and out['reason']=='identical_action_cached'


def test_required_fit_failure_retained_in_all_pairs_and_baseline_selected(monkeypatch):
    h,cs,fs,m,_,_=mock_context(monkeypatch,candidate_failure=True)
    out=guard.guard_predicted_actions(h,cs,fs,m,m,m,[3.,3.],[0.,6.])
    bad=out['candidate']
    assert not out['accepted'] and not bad['success'] and bad['score'] is None
    assert out['selected_u']==pytest.approx([3.,3.])
    assert len(bad['records'])==4 and len(bad['pairs'])==12
    assert any(not pair['success'] for pair in bad['pairs'])
    assert all('d' in record['fit_failures'] for record in bad['records'])
    assert bad['returned_fit_states']==16 and bad['successful_fit_states']==12


def test_history_failure_and_forecast_failure_are_explicit(monkeypatch):
    h,cs,fs,m,_,_=mock_context(monkeypatch)
    invalid=deepcopy(fs);invalid['d'].success=False
    out=guard.evaluate_predicted_action(h,cs,invalid,m,m,m,[3.,3.])
    assert not out['success'] and out['forecast_calls']==0 and out['fit_calls']==0
    assert out['failures'][0]['candidate']=='d'
    original=guard.lk.ac_predict
    def one_failure(c,*args):
        if c.name=='d':raise RuntimeError('declared forecast failure')
        return original(c,*args)
    monkeypatch.setattr(guard.lk,'ac_predict',one_failure)
    failed=guard.evaluate_predicted_action(h,cs,fs,m,m,m,[3.,3.])
    assert not failed['success'] and len(failed['records'])==4 and len(failed['pairs'])==12
    assert failed['forecast_calls']==4 and failed['fit_calls']==12
    assert any(row['kind']=='reference_forecast_failure' for row in failed['failures'])


def test_api_has_no_simulated_truth_or_future_noise_input():
    for method in (guard.evaluate_predicted_action,guard.guard_predicted_actions):
        names=inspect.signature(method).parameters
        assert 'truth' not in names and 'noise' not in names and 'path' not in names


def test_small_actual_ac_joint_fit_interface_and_replay_records():
    # Two names for the same physical star isolate interface correctness
    # from the separate question of topology discrimination power.
    path=generate(155021,history_count=4,future_count=2,truth_family=())
    history,_=observe(path,slice(0,4));m=model(3)
    first=RadialCandidate.from_supports('star_a',3,())
    second=RadialCandidate.from_supports('star_b',3,())
    options=dict(multistart=1,max_nfev=150,tol=1e-8,linear_solver='exact')
    fitted=lk.fit_candidate(first,history,m,seed=15411,**options)
    assert fitted.success
    twin=deepcopy(fitted);twin.candidate=second.name
    out=guard.evaluate_predicted_action(history,[first,second],
        {first.name:fitted,second.name:twin},m,m,m,[1.,5.],fit_options=options,seed=15412)
    assert out['success'],out['failures']
    assert out['fit_calls']==4 and out['forecast_calls']==2 and out['ac_calls']>2
    assert len(out['records'])==2 and all(len(r['fits'])==2 for r in out['records'])
    expected=history.p.mean(axis=0)
    for record in out['records']:
        predicted=record['predicted_observations']
        assert predicted['p'][:,0]==pytest.approx(expected[0]+np.array([1.,5.])/20.)
        assert predicted['p'][:,1:]==pytest.approx(np.tile(expected[1:],(2,1)))
        joint=lk.concat_observations(history,lk.Observations(**predicted))
        assert joint.t==6
        for state in record['fits']:
            assert len(state['p'])==6 and len(state['source_p'])==6
            assert state['global_MLE_certified'] is False
