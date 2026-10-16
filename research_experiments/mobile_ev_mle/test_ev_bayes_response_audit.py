"""Pure replay tests; no inference fit, optimization or generated history."""
from pathlib import Path

import numpy as np
import pytest

from . import ev_bayes_response_audit as audit
from . import ev_bayes_small_audit as independent
from .likelihood import RadialCandidate


def meter(T,n):
    return {**{k+'_sigma':.01 for k in ('p','q','v','master_p','master_q')},
            **{'bias_'+k:0. for k in ('p','q','v')}}


def test_independent_logline_jacobian_matches_radial_finite_differences():
    c=RadialCandidate.from_supports('two_level',3,[(0,1)])
    r=np.linspace(.008,.018,c.m);x=r*.7
    latent=np.array([[.04,.03,.02,.015,.01,.008,.01,.004],
                     [.025,.04,.03,.01,.012,.011,.012,.003]])
    physical=independent._physics(c,r,x,latent)
    M=meter(2,3);J=audit._line_jacobian(c,r,x,physical,M)
    eta=np.c_[np.log(r),np.log(x/r)].ravel()
    finite=[]
    for j in range(len(eta)):
        e=np.eye(len(eta))[j]*1e-5
        rp,xp=audit._eta_lines(eta+e);rm,xm=audit._eta_lines(eta-e)
        fp=independent._whiten(independent._physics(c,rp,xp,latent),M)
        fm=independent._whiten(independent._physics(c,rm,xm,latent),M)
        finite.append((fp-fm)/2e-5)
    assert J==pytest.approx(np.column_stack(finite),abs=2e-8,rel=1e-6)
    assert J.shape==(2*(3*3+2),2*c.m)


def test_restore_joint_preserves_conditional_covariance_and_line_state_coupling():
    mean=np.array([.2,-.1,1.,2.])
    C=np.array([[2.,.4,.3,.2],[.4,1.,-.2,.5],[.3,-.2,2.,.1],[.2,.5,.1,3.]])
    lm=np.array([.4,.1]);LC=np.array([[.4,.05],[.05,.3]])
    mu,cov=audit.restore_joint(mean,C,lm,LC)
    G=C[2:,:2]@np.linalg.inv(C[:2,:2]);conditional=C[2:,2:]-G@C[:2,2:]
    assert mu==pytest.approx(np.r_[lm,mean[2:]+G@(lm-mean[:2])])
    assert cov[:2,2:]==pytest.approx(LC@G.T)
    assert cov[2:,2:]-G@LC@G.T==pytest.approx(conditional)
    assert np.linalg.eigvalsh(cov).min()>0


def small_saved_prediction():
    candidates={k:RadialCandidate.from_supports(k,3,s) for k,s in
                (('star',[]),('branch',[(0,1)]))}
    updates={};fits=[]
    T,n=2,3
    latent=np.array([[.04,.03,.02,.015,.01,.008,.01,.004],
                     [.025,.04,.03,.01,.012,.011,.012,.003]])
    for label,c in candidates.items():
        r=np.full(c.m,.012);x=.7*r;eta=np.c_[np.log(r),np.log(x/r)].ravel();d=len(eta)
        # Nonzero line-state compensation, with a manifestly PSD covariance.
        factor=np.eye(d+T*(2*n+2))*.015
        factor[d,0]=.008;factor[d+2,-1]=-.006
        C=factor@factor.T
        updates[label]=dict(line_prior=dict(mean=eta),TV_posterior_future=dict(mean=np.r_[eta,latent.ravel()],cov=C))
        fits.append(dict(candidate=label,r=r,x=x,supports=c.supports))
    record=dict(prediction=dict(joint=dict(candidate_updates=updates,base_kw=20.,
        log_prior=np.log([.4,.6]),log_prior_evidence_only=np.log([.7,.3]))),
        methods=dict(none=dict(power_kw=np.zeros((T,n)))),future_meter=meter(T,n),training_result=dict(fits=fits))
    return record,candidates


def test_saved_joint_forecast_components_and_same_action_response_cancel():
    record,candidates=small_saved_prediction();power=np.array([[1.,0.,0.],[0.,.5,0.]])
    answer=audit.reconstruct_forecast(record,candidates,power,'joint')
    same=audit.reconstruct_forecast(record,candidates,power,'joint')
    zero=audit.reconstruct_forecast(record,candidates,np.zeros_like(power),'joint')
    for index,(part,F,G) in enumerate(zip(answer['parts'],answer['joint_jacobian'],zero['joint_jacobian'])):
        C=record['prediction']['joint']['candidate_updates'][list(candidates)[index]]['TV_posterior_future']['cov']
        assert answer['bank']['covariances'][0,index]==pytest.approx(np.eye(len(F))+F@C@F.T,abs=1e-12)
        assert np.max(abs(part['cross']))>0
        assert (F-same['joint_jacobian'][index])@C@(F-same['joint_jacobian'][index]).T==pytest.approx(np.zeros((len(F),len(F))))
        response=(F-G)@C@(F-G).T
        assert np.diag(response).max()>0
        assert np.linalg.norm(response)<np.linalg.norm(F@C@F.T+G@C@G.T)


def test_covariance_ablation_keeps_mean_and_only_changes_declared_blocks():
    record,candidates=small_saved_prediction();power=np.zeros((2,3))
    full=audit.reconstruct_forecast(record,candidates,power,'joint')
    no_cross=audit.reconstruct_forecast(record,candidates,power,'no_cross')
    known=audit.reconstruct_forecast(record,candidates,power,'line_known')
    weights=audit.reconstruct_forecast(record,candidates,power,'evidence_only')
    for other in (no_cross,known,weights):assert other['bank']['means']==pytest.approx(full['bank']['means'])
    for i,part in enumerate(full['parts']):
        assert full['bank']['covariances'][0,i]-no_cross['bank']['covariances'][0,i]==pytest.approx(part['cross'],abs=1e-12)
        assert np.linalg.eigvalsh(full['bank']['covariances'][0,i]-known['bank']['covariances'][0,i]).min()>-1e-10
    assert weights['bank']['covariances']==pytest.approx(full['bank']['covariances'])
    assert weights['bank']['log_prior']==pytest.approx(np.log([.7,.3]))


def test_actual_posterior_ties_and_credible_set_boundary_are_deterministic():
    bank=dict(means=np.zeros((1,2,5)),covariances=np.broadcast_to(np.eye(5),(1,2,5,5)),log_prior=np.log([.5,.5]))
    white=np.zeros((3,5));clean=np.zeros(5)
    result=audit.actual_statistics(bank,clean,white,1,1,1,credible_mass=.49)
    assert result['samples']['posterior']==pytest.approx(np.full((3,2),.5))
    assert np.array_equal(result['samples']['correct'],[0,0,0])
    assert np.array_equal(result['samples']['true_rejected'],[1,1,1])
    assert np.array_equal(result['samples']['credible_size'],[1,1,1])
    result=audit.actual_statistics(bank,clean,white,1,1,1,credible_mass=.9)
    assert np.array_equal(result['samples']['credible_size'],[2,2,2])
    assert result['summary']['meter_interval90_coverage']==1.


def test_voltage_coverage_uses_voltage_channel_positions_and_prediction_variance():
    means=np.zeros((1,2,5));means[0,1]=5.
    cov=np.broadcast_to(np.eye(5),(1,2,5,5)).copy();cov[0,0,2,2]=4.
    bank=dict(means=means,covariances=cov,log_prior=np.log([.7,.3]))
    white=np.array([[0.,0.,3.,0.,0.],[0.,0.,3.5,0.,0.]])
    result=audit.actual_statistics(bank,np.zeros(5),white,0,1,1)
    assert np.array_equal(result['samples']['voltage_interval90_coverage'],[1.,0.])
    assert result['summary']['meter_interval90_coverage']==pytest.approx(.9)


def test_explicit_failed_record_is_retained_without_fabricated_complete_audit():
    good=audit.audit_record(dict(spec=dict(index=0),status='execution_failure',failure='predeclared numerical failure'))
    assert good['success'] and not good['scientific_complete']
    assert good['counts']['independent_histories']==1
    assert good['unverified_due_failure']
    bad=audit.audit_record(dict(spec=dict(index=1),status='execution_failure'))
    assert not bad['success']
    missing=audit.audit_record(dict(status='complete'))
    assert not missing['success'] and missing['errors'][0]['check']=='schema'


def test_saved_selector_interfaces_are_observed_only_and_truth_parameter_is_rejected():
    root=Path(audit.__file__).parent
    source={name:(root/name).read_text(encoding='utf-8-sig') for name in
            ('ev_bayes_response_ac.py','ev_bayes_response_study.py')}
    result=audit.selector_interface_audit(source)
    assert 'truth' not in result['ev_bayes_response_ac.py/build_context']
    bad=dict(source);bad['ev_bayes_response_ac.py']=bad['ev_bayes_response_ac.py'].replace('def forecast_bank(context,power,','def forecast_bank(context,power,truth,')
    with pytest.raises(ValueError,match='observed-input'):
        audit.selector_interface_audit(bad)


def test_physical_domain_check_catches_ratio_and_sum_violations():
    bounds=dict(log_R=[[-2.,0.]],log_ratio=[[-1.,1.]],log_X=[[-2.,.3]])
    audit._line_domain([-.5,.1],bounds,'valid')
    with pytest.raises(ValueError,match='log_X'):
        audit._line_domain([-.2,.8],bounds,'invalid sum')
    with pytest.raises(ValueError,match='log_ratio'):
        audit._line_domain([-.5,1.1],bounds,'invalid ratio')
