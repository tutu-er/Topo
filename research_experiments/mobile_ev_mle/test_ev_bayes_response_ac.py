"""Compensation, sequential evidence and observed-only full-AC checks."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
from scipy.stats import truncnorm
from scipy.optimize import brentq

from . import ev_bayes_response_ac as ac
from . import ev_bayes_response_cases as cases
from . import ev_original_study as assets
from . import likelihood as lk
from .ev_fixed_train_fit import subset_model,subset_observations
from .simulation import model
from .ev_mobile_v4_cases import harmonic_features


def test_compensation_cross_covariance_changes_prediction_variance():
    # A terminal voltage measures theta+source: the two explanations compensate.
    updated=ac.gaussian_linear_update(np.zeros(2),np.eye(2),[1.], [0.], [[1.,1.]])
    np.testing.assert_allclose(updated['cov'],[[2/3,-1/3],[-1/3,2/3]],atol=1e-14)
    F=np.ones((1,2));full=float((F@updated['cov']@F.T)[0,0])
    ignoring_cross=float((F@np.diag(np.diag(updated['cov']))@F.T)[0,0])
    assert full==pytest.approx(2/3)
    assert ignoring_cross==pytest.approx(4/3)


def test_truncated_line_restores_source_mean_and_cross_covariance():
    # Half-normal theta with Gaussian source=-.8*theta+independent noise.
    mean=np.zeros(2);cov=np.array([[1.,-.8],[-.8,1.]])
    lm=np.sqrt(2/np.pi);lv=1-2/np.pi
    result=ac.restore_joint_moments(mean,cov,1,[lm],[[lv]])
    np.testing.assert_allclose(result['mean'],[lm,-.8*lm])
    np.testing.assert_allclose(result['cov'],[[lv,-.8*lv],[-.8*lv,.36+.64*lv]])
    assert result['cov'][0,1]<0


def test_sequential_untruncated_update_and_mass_ratio_equal_joint_evidence():
    m=np.zeros(2);C=np.array([[1.,.2],[.2,.7]])
    A=np.array([[1.,1.],[2.,-.5]])
    y=np.array([.7,-.3]);off=np.array([.1,-.2])
    T=ac.gaussian_linear_update(m,C,y[:1],off[:1],A[:1])
    TV=ac.gaussian_linear_update(T['mean'],T['cov'],y[1:],off[1:],A[1:])
    joint=ac.gaussian_linear_update(m,C,y,off,A)
    np.testing.assert_allclose(TV['mean'],joint['mean'],atol=2e-15)
    np.testing.assert_allclose(TV['cov'],joint['cov'],atol=2e-15)
    logZ=lambda r: norm.logcdf(r['mean'][0]/np.sqrt(r['cov'][0,0]))
    Hlog=T['log_density']+logZ(T)-np.log(.5)
    Vlog=TV['log_density']+logZ(TV)-logZ(T)
    assert Hlog+Vlog==pytest.approx(joint['log_density']+logZ(joint)-np.log(.5),abs=2e-15)
    # Moment matching then treating that Gaussian as a new prior applies the
    # boundary through the wrong density; this is a substantive counterexample.
    a=-T['mean'][0]/np.sqrt(T['cov'][0,0]);lam=norm.pdf(a)/norm.sf(a)
    line_mean=T['mean'][0]+np.sqrt(T['cov'][0,0])*lam
    line_var=T['cov'][0,0]*(1+a*lam-lam**2)
    bounded=ac.restore_joint_moments(T['mean'],T['cov'],1,[line_mean],[[line_var]])
    wrong=ac.gaussian_linear_update(bounded['mean'],bounded['cov'],y[1:],off[1:],A[1:])
    assert abs(wrong['log_density']-(Vlog))>1e-3


def test_covariance_ablations_keep_marginal_and_conditional_distinct():
    row=dict(line_dimension=1,joint_covariance=np.array([[1.,-.8],[-.8,1.]]))
    np.testing.assert_allclose(ac._mode_covariance(row,'no_cross'),np.eye(2))
    np.testing.assert_allclose(ac._mode_covariance(row,'line_known'),[[0,0],[0,.36]])
    with pytest.raises(ValueError): ac._mode_covariance(row,'unknown')


def test_eta_chain_rule_matches_physical_stacked_finite_difference():
    eta=np.array([-5.,-1.5,-6.,-1.2]);r,x=ac._rx(eta);h=1e-6
    finite=np.column_stack([(np.r_[*ac._rx(eta+h*np.eye(4)[j])]-
                            np.r_[*ac._rx(eta-h*np.eye(4)[j])])/(2*h) for j in range(4)])
    np.testing.assert_allclose(ac._eta_jacobian(r,x),finite,rtol=1e-9,atol=1e-12)


@pytest.fixture(scope='module')
def observed_case():
    return cases.make_case(cases.specifications('development')[0])


def test_proxy_noise_has_negative_pq_source_covariance(observed_case):
    case=observed_case;n=case['train'].n;H=case['train'].t
    values,R,record=ac.proxy_training_process(case['train'],case['tm'])
    expected=case['tm'].p_sigma**2
    np.testing.assert_allclose(R[:,0,-2],-np.broadcast_to(expected,(H,n))[:,0])
    np.testing.assert_allclose(R[:,-2,-2],np.broadcast_to(case['tm'].master_p_sigma**2,(H,))+
        np.broadcast_to(expected,(H,n)).sum(1))
    assert values.shape==(H,2*n+2)
    assert not record['proxy_used_as_measurement']
    assert np.linalg.eigvalsh(R).min()>0


def test_projected_hyper_noise_subtraction_matches_expected_trace(observed_case):
    case=observed_case;H=case['train'].t
    hyper=ac.estimate_process_hyperparameters(case['train'],case['tm'])
    X=harmonic_features(np.arange(H));M=np.eye(H)-X@np.linalg.pinv(X)
    phi=hyper['phi'];D=np.eye(H)[1:]-phi*np.eye(H)[:-1];W=D@M
    t=np.arange(H);K=phi**abs(t[:,None]-t[None,:])/(1-phi**2)
    cproc=np.trace(W@K@W.T)/(H-1);cmeter=np.trace(W@W.T)/(H-1)
    assert hyper['innovation_process_projection_factor']==pytest.approx(cproc)
    assert hyper['innovation_meter_projection_factor']==pytest.approx(cmeter)
    # Known-noise subtraction must account for fitted harmonics at small H.
    assert abs(cmeter-(1+phi**2))>.01
    _,noise,_=ac.proxy_training_process(case['train'],case['tm'])
    expected=sum(np.sum(W[:,t]**2)*noise[t] for t in range(H))/(H-1)
    np.testing.assert_allclose(hyper['subtracted_innovation_meter_covariance'],expected,atol=1e-20)
    # For arbitrary innovation Q, the declared correction is unbiased at fixed phi.
    Q=np.diag(np.linspace(.01,.03,2*case['train'].n+2));R=noise.mean(0)
    expected_cov=cproc*Q+cmeter*R
    np.testing.assert_allclose((expected_cov-cmeter*R)/cproc,Q,atol=1e-16)
    assert hyper['coefficient_prior_mean']==0
    assert not hyper['hyperparameter_uncertainty_integrated']


def test_process_kernel_retains_cross_channel_and_time_uncertainty(observed_case):
    hyper=ac.estimate_process_hyperparameters(observed_case['train'],observed_case['tm'])
    mean,K=ac.process_kernel(hyper,36)
    w=hyper['channel_count'];assert np.linalg.eigvalsh(K).min()>0
    np.testing.assert_array_equal(mean,np.zeros_like(mean))
    assert np.linalg.norm(K[:w,w:2*w])>0
    assert np.linalg.norm(hyper['innovation_covariance']-np.diag(np.diag(hyper['innovation_covariance'])))>0


@pytest.fixture(scope='module')
def context(observed_case):
    # A controlled interior AC fixture tests the numerical interface, separately
    # from the heterogeneous development/evaluation case-generating experiment.
    case=observed_case;H=32;V=20;F=8;n=3;tt=np.arange(H+V+F)
    star=case['bank'][0];bank=(star,replace(star,name='same_structure_second_center'))
    r=np.full(star.m,.003);x=np.full(star.m,.0007)
    p=.12+.012*np.cos(tt[:,None]/6+np.arange(n));q=.04+.005*np.sin(tt[:,None]/7+np.arange(n))
    sp=.12+.008*np.sin(tt/5);sq=.04+.003*np.cos(tt/6)
    physical=lk.ac_predict(star,r,x,p,q,sp,sq,np.ones(len(tt)))
    meter=model(n);rng=np.random.default_rng(41)
    white={name:rng.standard_normal(np.shape(physical[name])) for name in assets.CHANNELS}
    observed=assets.observation(physical,meter,white)
    train=subset_observations(observed,slice(0,H));evidence=subset_observations(observed,slice(H,H+V))
    fits={c.name:SimpleNamespace(success=True,r=r.copy(),x=x.copy(),
        source_p=sp[:H].copy(),source_q=sq[:H].copy()) for c in bank}
    ctx,record=ac.build_context(train,evidence,meter,meter,bank,fits,
        case['fm'],horizon=8,base_kw=case['base_kw'],options=dict(**assets.ASSET_BOUNDS,
            seed=112,truncation=dict(power=10,max_power=16,replicates=4,min_ess_fraction=0)))
    return ctx,record


def test_complete_ac_context_scores_evidence_then_updates_underlying_gaussian(context):
    ctx,record=context;logs=[]
    for row in record['candidate_updates'].values():
        before=row['T_untruncated_EF_mean'];C=row['T_untruncated_EF_covariance']
        ev=ac.gaussian_linear_update(before,C,row['evidence_observed_whitened'],
            row['evidence_affine_offset'],row['evidence_affine_jacobian'])
        assert ev['log_density']==pytest.approx(row['evidence_Gaussian_log_predictive'],abs=1e-7)
        expected=row['evidence_Gaussian_log_predictive']+row['logZTV']-row['logZT']
        assert row['evidence_log_predictive']==pytest.approx(expected)
        logs.append(row['training_log_predictive']+row['evidence_log_predictive'])
        assert row['T_line_integration']['status']=='ok'
        assert row['TV_line_integration']['status']=='ok'
    np.testing.assert_allclose(ctx['log_prior'],np.asarray(logs)-logsumexp(logs))
    assert record['training_raw_likelihood_count']==1
    assert record['fitted_source_pseudo_likelihood_count']==0
    assert record['options']['truncation']['seed']==112


def test_ac_covariance_decomposition_and_no_action_specific_line_refit(context):
    ctx,_=context;u=np.zeros((8,3));u[:,0]=1.5
    forecast=ac.forecast_bank(ctx,u,with_jacobian=True)
    assert forecast['bank']['means'].shape==(1,2,88)
    assert forecast['jacobian'].shape==(1,2,88,24)
    for i,p in enumerate(forecast['physical']):
        parts=p['covariance_parts'];S=forecast['bank']['covariances'][0,i]
        np.testing.assert_allclose(S,np.eye(88)+parts['line']+parts['state']+parts['cross'],atol=1e-10)
        assert np.linalg.eigvalsh(S).min()>.99
        assert p['ac_equation_residual']<2e-10
        np.testing.assert_allclose(p['latent_inputs_with_EV'][:,:3]-ctx['rows'][i]['latent_mean'][:,:3],u/ctx['base_kw'])
        assert p['line_center_sha256']==ctx['rows'][i]['line_sha256']
        assert np.linalg.norm(parts['cross'])>0


def test_mean_gradient_and_same_random_response_covariance(context):
    ctx,_=context;u=np.zeros((8,3));u[:,0]=1
    center=ac.forecast_bank(ctx,u,with_jacobian=True)
    direction=np.zeros_like(u);direction[:,1]=np.linspace(-.5,.5,8);h=1e-3
    plus=ac.forecast_bank(ctx,u+h*direction);minus=ac.forecast_bank(ctx,u-h*direction)
    finite=(plus['bank']['means']-minus['bank']['means'])/(2*h)
    predicted=np.einsum('lkdp,p->lkd',center['jacobian'],direction.ravel())
    np.testing.assert_allclose(finite,predicted,atol=3e-6,rtol=1e-6)
    same=ac.response_covariance(ctx,u,u)
    for row in same['rows']: np.testing.assert_array_equal(row['covariance'],np.zeros((88,88)))
    diff=ac.response_covariance(ctx,u)
    for row in diff['rows']:
        assert np.linalg.eigvalsh(row['covariance']).min()>-1e-10
        np.testing.assert_allclose(row['covariance'][:48],0,atol=1e-15)


def test_probability_ablations_change_only_declared_components(context):
    ctx,_=context;u=np.zeros((8,3))
    full=ac.forecast_bank(ctx,u)
    evidence=ac.forecast_bank(ctx,u,weights_mode='evidence_only')
    np.testing.assert_array_equal(full['bank']['means'],evidence['bank']['means'])
    np.testing.assert_array_equal(full['bank']['covariances'],evidence['bank']['covariances'])
    for mode in ('no_cross','line_known'):
        other=ac.forecast_bank(ctx,u,covariance_mode=mode)
        np.testing.assert_array_equal(full['bank']['means'],other['bank']['means'])
        np.testing.assert_array_equal(full['bank']['log_prior'],other['bank']['log_prior'])
        assert np.linalg.norm(full['bank']['covariances']-other['bank']['covariances'])>0


def test_saved_jacobian_ablations_equal_direct_propagation_without_ac(context,monkeypatch):
    ctx,_=context;u=np.zeros((8,3));u[:,0]=1
    full=ac.forecast_bank(ctx,u,with_jacobian=True)
    direct={mode:ac.forecast_bank(ctx,u,covariance_mode=mode,weights_mode='evidence_only')
        for mode in ('full','no_cross','line_known')}
    original=full['bank']['covariances'].copy()
    monkeypatch.setattr(ac,'_physical',lambda *a,**k: (_ for _ in ()).throw(AssertionError('extra AC')))
    for mode in direct:
        derived=ac.ablate_forecast(ctx,full,covariance_mode=mode,weights_mode='evidence_only')
        for key in ('means','covariances','log_prior'):
            np.testing.assert_array_equal(derived['bank'][key],direct[mode]['bank'][key])
        assert derived['AC_states']==0
    np.testing.assert_array_equal(full['bank']['covariances'],original)


def test_imprecise_bounded_integral_is_not_accepted(monkeypatch):
    monkeypatch.setattr(ac.truncation,'bounded_line_moments',lambda *a,**k:
        dict(status='precision_not_met',mean=np.zeros(2),cov=np.eye(2),log_mass=-.5,diagnostics={}))
    with pytest.raises(RuntimeError,match='bounded line integration failed'):
        ac._truncated(np.zeros(3),np.eye(3),2,{'bounds':{}},ac._options({}),0)


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
