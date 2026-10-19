"""Independent fixed-linear Gaussian power link for projected information.

No AC policy, action or threshold is chosen. Poisson mixtures provide explicit
numeric lower/upper bounds; finite positive-direction tasks demonstrate the
small-signal expansion and its finite-signal limitation.
"""
from pathlib import Path
import json
import numpy as np
from scipy.stats import chi2,ncx2,poisson


def power_interval(rank,alpha,noncentrality,terms=256):
    if int(rank)!=rank or rank<1 or not 0<alpha<1 or not np.isfinite(noncentrality) or noncentrality<0:
        raise ValueError('positive rank, alpha in (0,1), nonnegative finite signal required')
    if int(terms)!=terms or terms<0:raise ValueError('nonnegative truncation index required')
    critical=float(chi2.ppf(1-alpha,rank));k=np.arange(terms+1)
    lower=float(np.dot(poisson.pmf(k,noncentrality/2),chi2.sf(critical,rank+2*k)))
    missing=float(poisson.sf(terms,noncentrality/2));upper=min(1.,lower+missing)
    return dict(lower=lower,upper=upper,omitted_poisson_mass=missing,critical=critical,
        independent_scipy_tail=float(ncx2.sf(critical,rank,noncentrality)),rank=int(rank),alpha=alpha,noncentrality=noncentrality,
        bound_scope='analytic truncation of exact Poisson mixture; floating roundoff not certified')


def small_signal_coefficient(rank,alpha):
    critical=float(chi2.ppf(1-alpha,rank))
    return .5*(float(chi2.sf(critical,rank+2))-alpha)


def check_power_link():
    comparisons=[power_interval(r,.1,lam) for r in (1,2,3) for lam in (0.,.0001,1.,10.,100.)]
    errors=[abs(x['independent_scipy_tail']-.5*(x['lower']+x['upper'])) for x in comparisons]
    rank=2;alpha=.1;axis_signals=np.array([4.,1.]);average=float(axis_signals.mean())
    coefficient=small_signal_coefficient(rank,alpha);scales=[]
    for gamma in (1.,.5,.25,.125,.0625):
        lambdas=gamma**2*axis_signals
        actual=float(np.mean([ncx2.sf(chi2.ppf(1-alpha,rank),rank,x) for x in lambdas]))
        linear=alpha+coefficient*gamma**2*average
        bound=float(np.mean(lambdas**2)/8)
        scales.append(dict(gamma=gamma,actual_average_power=actual,small_signal_prediction=linear,
            power_increment_ratio=(actual-alpha)/(linear-alpha),absolute_remainder=abs(actual-linear),
            rigorous_remainder_upper_bound=bound,within_remainder_bound=abs(actual-linear)<=bound+1e-14))
    critical=chi2.ppf(1-alpha,rank)
    concentrated=float(ncx2.sf(critical,rank,5.))
    mixed=.5*(alpha+float(ncx2.sf(critical,rank,10.)))
    return dict(success=max(errors)<5e-13 and all(x['within_remainder_bound'] for x in scales),
        scope='fixed A/B/u, free linear Gaussian target, same positive rank; no AC adaptive calibration',
        projection_signal_noncentrality='zeta.T I zeta',engineering_average_half_distance=average/2,
        coefficient=coefficient,poisson_comparisons=comparisons,max_numeric_difference=max(errors),
        scales=scales,same_average_signal_counterexample=dict(mean_noncentrality=5.,concentrated_power=concentrated,
            zero_ten_mixture_power=mixed,difference=concentrated-mixed),
        remainder_proof='second forward differences of increasing p_df in [0,1] have absolute value at most one; |power second derivative|<=1/4, Taylor remainder<=lambda^2/8')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check_power_link()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(success=r['success'],max_numeric_difference=r['max_numeric_difference'],counterexample=r['same_average_signal_counterexample'])))