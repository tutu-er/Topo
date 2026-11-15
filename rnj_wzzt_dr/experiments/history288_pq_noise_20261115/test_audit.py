"""Arithmetic-only checks; no optimizer, fitting or AC execution."""
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
from .audit import standard_noise,noisy_observation,validate_timeline,strict_winner,bank_values


def test_partition_contiguous_no_double_scoring():
    t=[(datetime(2024,1,4,7,tzinfo=timezone.utc)+timedelta(minutes=15*i)).isoformat() for i in range(384)]
    v=validate_timeline(t)
    assert v['train_indices']==list(range(192))
    assert set(v['train_indices']).isdisjoint(v['validation_indices'])
    assert set(v['evaluation_indices']).isdisjoint(v['train_indices']+v['validation_indices'])
    with pytest.raises(ValueError):validate_timeline(t[:10]+t[11:])


def test_fixed_voltage_and_nested_PQ_scale():
    z=standard_noise(2026111521,18,2,0,0,3)
    clean={k:np.full((288,) if k=='root' else (288,3),value) for k,value in [('P',.02),('Q',-.003),('V',1.01),('root',1.02)]}
    zero=noisy_observation(clean,slice(None),z,0.)
    a=noisy_observation(clean,slice(None),z,.005);b=noisy_observation(clean,slice(None),z,.01)
    assert np.array_equal(zero['P'],clean['P']) and np.array_equal(zero['Q'],clean['Q'])
    assert np.array_equal(zero['V'],b['V']) and np.array_equal(a['root'],b['root'])
    # Two products and two subtractions accumulate at most a few input ulps.
    # The exact observation/noise reconstruction is checked separately.
    roundoff=4*np.finfo(float).eps*np.max(abs(clean['Q']))
    assert np.allclose(b['Q']-clean['Q'],2*(a['Q']-clean['Q']),atol=roundoff,rtol=0)
    assert np.array_equal(b['Y'],b['root'][:,None]**2-b['V']**2)


def test_noise_repeat_and_future_stream_separated():
    a=standard_noise(2026111521,18,0,1,0,2);b=standard_noise(2026111521,18,1,1,0,2)
    c=standard_noise(2026111521,18,0,1,1,2);h=standard_noise(2026111521,18,0,0,0,2)
    assert not np.array_equal(a['P'],b['P']) and not np.array_equal(a['P'],c['P'])
    assert not np.array_equal(a['P'],h['P'][:96])
    assert np.array_equal(a['P'],standard_noise(2026111521,18,0,1,0,2)['P'])


def test_native_tie_uses_complexity_then_loss():
    assert strict_winner([1.,1.+1e-13],[3,1],10,1.,1e-12)==1
    assert strict_winner([1.,1.01],[3,1],10,1.,1e-12)==0


def test_reconstruct_support_channel_and_train_exclusion():
    native=dict(config={'zero_tol':1e-9},frozen_candidates=[dict(point=dict(supports=[[0],[0,1]],r=[2.,0.],x=[0.,1.]))])
    observation=dict(P=np.ones((2,2)),Q=np.ones((2,2)),Y=np.array([[4.,2.],[4.,2.]]))
    H,active,rows=bank_values(native,[101,102],observation,observation,{(0,),(0,1)})
    assert np.array_equal(H[0],[[2,0,1,1],[0,0,1,1]])
    assert rows[0]['validation_sse']==0 and rows[0]['full_support_exact']


def test_report_retains_failed_fit_denominator():
    from .report import historical_summary
    model=dict(history_correct=True,history_full_support_exact=False,truth_in_native_bank=True,
        full_support_truth_in_native_bank=False,bank_size=3,actual_fit_seconds=12.,stop_reason='limit',total_cap_exceeded=False)
    cells=[{'fit':{'models':{'original':model}}},{'fit':None}]
    answer=historical_summary(cells,'original')
    assert answer['n']==2 and answer['completed']==1 and answer['failed']==1
    assert answer['correct']==1 and answer['covered']==1 and answer['full_support_covered']==0


def test_voltage_joint_error_includes_shared_root_component():
    clean={k:np.full((288,) if k=='root' else (288,2),v) for k,v in [('P',1.),('Q',2.),('V',1.),('root',1.01)]}
    z={k:np.zeros_like(v) for k,v in clean.items()};z['root'][:]=1
    a=noisy_observation(clean,slice(None),z,0.)
    expected=(1.01*(1+.0005))**2-1
    assert np.array_equal(a['Y'][:,0],a['Y'][:,1])
    assert np.allclose(a['Y'],expected,atol=1e-15,rtol=0)


def test_offline_response_error_does_not_confuse_R_and_X():
    from .report import native_matrix
    point=dict(supports=[[0],[0,1]],r=[.2,.1],x=[.03,.04])
    H=native_matrix(point,2)
    assert np.allclose(H[:,:2],[[.3,.1],[.1,.1]])
    assert np.allclose(H[:,2:],[[.07,.04],[.04,.04]])
    assert not np.array_equal(H[:,:2],H[:,2:])
