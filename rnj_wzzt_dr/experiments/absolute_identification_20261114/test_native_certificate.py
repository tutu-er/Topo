"""Analytic checks only; no conditional physical-world experiments."""
import numpy as np
from scipy.stats import norm
from .native_certificate import failures,audit_target,undominated_exact_H


def main():
    H=np.zeros((1,1));P=np.ones((1,1))
    v=failures(H,0.,H,1.,[np.array([[4.]])],[0.],P,P,1.,0.)
    assert abs(v['failure'][0]-norm.cdf(-2.))<1e-14
    bank=[dict(H=H,history_sse=0.,topology_id='a'),
        dict(H=np.array([[4.]]),history_sse=0.,topology_id='b'),
        dict(H=np.array([[.2]]),history_sse=.01,topology_id='b'),
        dict(H=np.array([[.2]]),history_sse=.02,topology_id='b')]
    score=dict(sigma2=1.,parameters=[dict(H=H,history_sse=0.),dict(H=bank[1]['H'],history_sse=0.)],
        models=[dict(index=0,topology_id='a'),dict(index=1,topology_id='b')])
    r=audit_target(score,bank,P,P,H,1.,0,0.,False)
    assert r['representative_pass90'] and not r['native_raw_pass90'] and not r['native_distinct_H_pass90']
    assert r['omitted_worst_pair_failure']>.4 and r['native_raw_failure_sum']>r['native_distinct_H_failure_sum']
    assert undominated_exact_H(bank,[2,3])==[2]
    # Direct score expansion and analytic variance with a nonzero mean bias.
    rng=np.random.default_rng(731)
    target=rng.normal(size=(2,3));other=rng.normal(size=(2,3));mean=target+rng.normal(size=(2,3))*.2
    power=rng.normal(size=(4,3));old=rng.normal(size=(4,3));noise=rng.normal(size=(4,2))
    sigma2=.7;variance=.4;ht=.2;hj=.3
    result=failures(target,ht,mean,variance,[other],[hj],power,old,sigma2,0.)
    Y=power@mean.T+noise
    direct=(hj-ht+np.sum((Y-power@other.T)**2)-np.sum((Y-power@target.T)**2))/(2*sigma2)
    delta=power@(target-other).T
    assert abs(direct-(result['mean'][0]+np.sum(noise*delta)/sigma2))<1e-12
    assert abs(result['variance'][0]-variance*np.sum(delta**2)/sigma2**2)<1e-12
    cap=failures(target,ht,mean,variance,[other],[hj],power,old,sigma2,0.,True)
    assert cap['failure'][0]>=cap['actual_failure'][0]-1e-14
    zero=failures(H,0.,H,1.,[H],[0.],P,P,1.,0.)
    assert zero['failure'][0]==1.
    print('Native-bank analytic certificate checks passed: omitted-point counterexample, nested duplicates, biased mean/variance identity, cap, and zero-distance tie.')


if __name__=='__main__':main()
