"""Independent bounded-line generative calibration; no AC or policy tuning.

The kernel matches the archived test helper used for the first 2048-history
numerical audit. Default count/seed reproduce that audit; raw histories remain
independent even though each has one coupled T/V/future realization.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm,truncnorm
from scipy.special import logsumexp
from scipy.optimize import brentq
from . import ev_bayes_response_ac as ac

def run_generated_linear_calibration(count=2048,seed=2026111701):
    """Independent declared-prior histories, not a saved-posterior self-check.

    One bounded standardized line theta in [0,1], and three correlated source
    states. T/V are one meter each; a fixed future action changes each model's
    line response. Exact scalar truncation is independent of the multidimensional
    GHK implementation. All models and seeds are fixed before drawing histories.
    """
    from time import perf_counter
    began=perf_counter();rng=np.random.default_rng(seed);sigma=.12
    prior_mean=np.array([.3,0.,0.,0.]);prior_cov=np.zeros((4,4));prior_cov[0,0]=.35**2
    tt=np.arange(3);prior_cov[1:,1:]=.25**2*.6**abs(tt[:,None]-tt[None,:])
    line_coefficients=np.array([[1.,.8,1.7],[.85,1.1,.6]])
    Hs=[]
    for coef in line_coefficients:
        A=np.zeros((3,4));A[:,0]=coef;A[np.arange(3),1+np.arange(3)]=1
        Hs.append(A/sigma)
    lower,upper=0.,1.;a0=(lower-prior_mean[0])/.35;b0=(upper-prior_mean[0])/.35
    logZ0=float(np.log(norm.cdf(b0)-norm.cdf(a0)))
    def bounded(gauss):
        sd=np.sqrt(gauss['cov'][0,0]);a=(lower-gauss['mean'][0])/sd;b=(upper-gauss['mean'][0])/sd
        logZ=float(ac.truncation._log_interval(a,b))
        lm,lv=truncnorm.stats(a,b,loc=gauss['mean'][0],scale=sd,moments='mv')
        joint=ac.restore_joint_moments(gauss['mean'],gauss['cov'],1,[lm],[[lv]])
        return a,b,sd,logZ,joint
    labels=rng.integers(0,2,count);theta=truncnorm.rvs(a0,b0,loc=.3,scale=.35,size=count,random_state=rng)
    source=rng.multivariate_normal(np.zeros(3),prior_cov[1:,1:],count)
    errors=rng.standard_normal((count,3))*sigma
    observations=line_coefficients[labels]*theta[:,None]+source+errors
    nodes,weights=np.polynomial.legendre.leggauss(48);nodes=(nodes+1)/2;weights=weights/2
    nodes96,weights96=np.polynomial.legendre.leggauss(96);nodes96=(nodes96+1)/2;weights96=weights96/2
    rows=[];max_sequence_error=0.;max_log_evidence_error=0.;max_quad_cdf_error=0.
    for i,(label,y) in enumerate(zip(labels,observations)):
        gaussians=[];bounds=[];logs=[];logsF=[];predictions=[]
        for k,A in enumerate(Hs):
            T=ac.gaussian_linear_update(prior_mean,prior_cov,y[:1]/sigma,[0.],A[:1])
            TV=ac.gaussian_linear_update(T['mean'],T['cov'],y[1:2]/sigma,[0.],A[1:2])
            direct=ac.gaussian_linear_update(prior_mean,prior_cov,y[:2]/sigma,np.zeros(2),A[:2])
            max_sequence_error=max(max_sequence_error,float(np.max(abs(TV['mean']-direct['mean']))),
                float(np.max(abs(TV['cov']-direct['cov']))))
            aa,bb,sd,logZ,mom=bounded(TV);_,_,_,logZT,_=bounded(T)
            logH=T['log_density']+logZT-logZ0
            logV=TV['log_density']+logZ-logZT
            log_direct=direct['log_density']+logZ-logZ0
            max_log_evidence_error=max(max_log_evidence_error,abs(logH+logV-log_direct))
            logs.append(logH+logV);gaussians.append(TV);bounds.append((aa,bb,sd))
            TVF=ac.gaussian_linear_update(TV['mean'],TV['cov'],y[2:]/sigma,[0.],A[2:])
            _,_,_,logZF,_=bounded(TVF)
            logsF.append(logH+logV+TVF['log_density']+logZF-logZ)
            compensation=TV['cov'][3,0]/TV['cov'][0,0]
            intercept=TV['mean'][3]-compensation*TV['mean'][0]
            slope=line_coefficients[k,2]+compensation
            conditional_sd=np.sqrt(TV['cov'][3,3]-TV['cov'][3,0]**2/TV['cov'][0,0]+sigma**2)
            theta_nodes=truncnorm.ppf(nodes,aa,bb,loc=TV['mean'][0],scale=sd)
            theta_nodes96=truncnorm.ppf(nodes96,aa,bb,loc=TV['mean'][0],scale=sd)
            predictions.append(dict(intercept=intercept,slope=slope,conditional_sd=conditional_sd,
                means=intercept+slope*theta_nodes,means96=intercept+slope*theta_nodes96,
                gaussian_mean=float((Hs[k][2]*sigma)@mom['mean']),
                gaussian_sd=float(np.sqrt((Hs[k][2]*sigma)@mom['cov']@(Hs[k][2]*sigma)+sigma**2))))
        pi=np.exp(np.asarray(logs)-logsumexp(logs));piF=np.exp(np.asarray(logsF)-logsumexp(logsF))
        def cdf(value,kind='exact'):
            if kind=='gaussian': return sum(pi[k]*norm.cdf((value-p['gaussian_mean'])/p['gaussian_sd']) for k,p in enumerate(predictions))
            points,ww=('means96',weights96) if kind=='exact96' else ('means',weights)
            return sum(pi[k]*np.dot(ww,norm.cdf((value-p[points])/p['conditional_sd'])) for k,p in enumerate(predictions))
        exact=[brentq(lambda value:cdf(value)-q,-4.,5.) for q in (.05,.95)]
        approx=[brentq(lambda value:cdf(value,'gaussian')-q,-4.,5.) for q in (.05,.95)]
        max_quad_cdf_error=max(max_quad_cdf_error,*[abs(cdf(value,'exact96')-q) for value,q in zip(exact,(.05,.95))])
        aa,bb,sd=bounds[label];g=gaussians[label]
        line_interval=truncnorm.ppf([.05,.95],aa,bb,loc=g['mean'][0],scale=sd)
        order=np.argsort(-piF);keep=order[:np.searchsorted(np.cumsum(piF[order]),.9)+1]
        rows.append(dict(history=i,true_model=int(label),posterior0=float(piF[0]),
            confidence=float(piF.max()),correct=int(piF.argmax()==label),predicted_error=float(1-piF.max()),
            theta=float(theta[i]),line_interval_coverage=bool(line_interval[0]<=theta[i]<=line_interval[1]),
            predictive_exact90_coverage=bool(exact[0]<=y[2]<=exact[1]),
            predictive_Gaussian_moment90_coverage=bool(approx[0]<=y[2]<=approx[1]),
            credible90_excludes_truth=bool(label not in keep),
            credible90_excluded_posterior_mass=float(1-piF[keep].sum())))
    def summary(key):
        values=np.array([r[key] for r in rows],float)
        return dict(mean=float(values.mean()),standard_error=float(values.std(ddof=1)/np.sqrt(count)),count=count)
    bins=[]
    for lo,hi in zip(np.arange(.5,1,.1),np.arange(.6,1.1,.1)):
        subset=[r for r in rows if lo<=r['confidence']<(hi+1e-12 if hi>.999 else hi)]
        bins.append(dict(lower=float(lo),upper=float(min(1,hi)),count=len(subset),
            average_confidence=float(np.mean([r['confidence'] for r in subset])) if subset else None,
            accuracy=float(np.mean([r['correct'] for r in subset])) if subset else None))
    return dict(status='complete',seed=seed,independent_histories=count,future_draws_per_history=1,
        model=dict(theta_prior='N(.3,.35^2) restricted to [0,1]',source_covariance=prior_cov[1:,1:].tolist(),
            meter_sigma=sigma,line_coefficients=line_coefficients.tolist(),topology_prior=[.5,.5],
            hyperparameters_known=True,exact_linear_model=True,AC_or_empirical_hyper_calibration=False),
        max_sequential_mean_covariance_error=max_sequence_error,max_sequential_log_evidence_error=max_log_evidence_error,
        max_exact_predictive_48_vs96_quadrature_CDF_error=max_quad_cdf_error,
        summaries={key:summary(key) for key in ('posterior0','correct','predicted_error','line_interval_coverage',
            'predictive_exact90_coverage','predictive_Gaussian_moment90_coverage',
            'credible90_excludes_truth','credible90_excluded_posterior_mass')},
        probability_bins=bins,rows=rows,seconds=float(perf_counter()-began))


def main():
    import argparse,hashlib,inspect,platform
    from pathlib import Path
    import scipy
    from . import gradient_price_study as io
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=2048)
    parser.add_argument('--seed',type=int,default=2026111701)
    args=parser.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('preserve existing calibration evidence')
    if args.count<2:raise ValueError('at least two independent histories required')
    result=run_generated_linear_calibration(args.count,args.seed)
    paired=np.array([r['predicted_error']-(1-r['correct']) for r in result['rows']])
    result['risk_calibration_paired_difference']=dict(predicted_minus_observed_error=float(paired.mean()),standard_error=float(paired.std(ddof=1)/len(paired)**.5))
    result['source_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('ev_bayes_response_linear_calibration.py','ev_bayes_response_ac.py','ev_bayes_response_truncation.py')}
    result['kernel_sha256']=hashlib.sha256(inspect.getsource(run_generated_linear_calibration).encode()).hexdigest()
    result['failed_histories']=0
    result['scope']='Known proper bounded-line linear Gaussian generative model; not nonlinear AC or empirical hyperparameter calibration.'
    result['environment']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    io.write_json(out,result,compressed=out.suffix=='.gz')
    print({key:value for key,value in result.items() if key not in ('rows','source_sha256','model','probability_bins')},flush=True)


if __name__=='__main__':main()
