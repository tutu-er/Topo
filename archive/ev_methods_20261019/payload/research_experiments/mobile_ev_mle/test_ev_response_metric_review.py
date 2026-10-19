"""Small saved-evidence/schema checks; no fit, selector or optimizer calls."""
from copy import deepcopy

import numpy as np
import pytest

from . import ev_response_metric_review as review


def strict_trial():
    fits=[]
    for name in review.NAMES:
        m=len(review.SUPPORTS[name])
        fits.append(dict(candidate=name,r=[.004]*m,x=[.001]*m))
    ids={r["candidate"]:review.line_id(r["r"],r["x"]) for r in fits}
    return dict(candidates=[dict(name=name,supports=review.SUPPORTS[name]) for name in review.NAMES],
        config=dict(declared_candidates=list(review.NAMES),wzzt_used=False,source_interface=1,PQ_inputs_fixed=True),
        training_result=dict(fits=fits),true_name="star",candidate_miss=False,source_sha256={},policy_id="fixture",
        selection=dict(declared_candidates=list(review.NAMES),WZZT_used=False,
            methods=dict(contrast_joint=dict(model=dict(names=list(review.NAMES),line_sha256=ids)))),
        methods=[dict(name=name,status="complete") for name in review.METHODS])


def test_strict_pair_replaces_four_candidate_schema_without_renaming():
    trial=strict_trial(); result=review.strict_pair_schema(trial)
    assert result["candidate_count"]==2 and result["method_denominator"]==9
    assert result["line_snapshots_checked"]==1


@pytest.mark.parametrize("kind",["supports","theta","candidate","duplicate_fit","method"])
def test_candidate_theta_and_failure_denominator_tampering_detected(kind):
    trial=strict_trial()
    if kind=="supports":trial["candidates"][1]["supports"]=review.SUPPORTS["star"]
    elif kind=="theta":trial["training_result"]["fits"][0]["r"][0]+=.001
    elif kind=="candidate":trial["candidates"][0]["name"]="other"
    elif kind=="duplicate_fit":trial["training_result"]["fits"].append(deepcopy(trial["training_result"]["fits"][0]))
    else:trial["methods"].pop()
    with pytest.raises(ValueError):review.strict_pair_schema(trial)


def test_schema_adapter_preserves_non_schema_errors_and_raw_trial(monkeypatch):
    trial=strict_trial(); before=deepcopy(trial)
    freeze=dict(source_sha256={},config=trial["config"],policy_id="fixture")
    monkeypatch.setattr(review.dynamic,"_source_audit",lambda *_:dict(source_count=101,current_source_matches=True))
    monkeypatch.setattr(review.dynamic,"review_trial",lambda *_a,**_k:dict(checks=[dict(name="physics",status="passed")],success=True))
    def profile(_):
        return dict(checks=[dict(name="planning_schema",status="failed",message="four-model bank")],
            errors=[dict(name="planning_schema",message="this experiment must retain the four-model bank"),
                dict(name="pair/physics",message="tampered physics")],counts=dict(unique_profiles=1),success=False)
    monkeypatch.setattr(review.profile_review,"review_profile_trial",profile)
    view,report=review.adapted_reviews(trial,freeze,{})
    assert not report["success"]
    assert report["profile_review"]["errors"]==[dict(name="pair/physics",message="tampered physics")]
    assert report["profile_review"]["legacy_superseded_checks"]
    assert view["selection"]["configuration"]==review.fixed.CONFIG
    assert trial==before and "configuration" not in trial["selection"]


class AlgebraReplay:
    """Explicit linear physical fixture; units follow per-kW beta conversion."""
    T,n=2,1
    sigma=np.ones((2,3));step=1e-3
    sources={name:np.zeros((2,2)) for name in review.NAMES}
    bounds=np.array([[-.5,.5],[-.5,.5]])
    lines={name:dict(r=[.004],x=[.001]) for name in review.NAMES}
    trial=dict(methods=[dict(name="square",power_kw=[[0.],[1.]])])
    def forward(self,name,p,q,u):
        values=np.zeros((2,3))
        if name==review.NAMES[0]:values[:,0]=2+2*np.asarray(u)[:,0]
        return dict(values=values,margin=.1)
    def derivative(self,name,p,q,u,*,kind):
        out=np.zeros((2,3,1 if kind=="P" else 2))
        if kind=="P" and name==review.NAMES[0]:out[:,0,0]=40.
        return out


def algebra_model():
    records=[]
    for low,high in ((-.1,.3),(-.2,.4)):
        # Model audit uses zero source coefficients, hence exact zero supports;
        # evaluation tests below substitute a nonzero declared interval.
        records.append(dict(source_singular_values=[0.,0.,0.],source_rank=0,rank_cutoff=0.,
            source_leakage_a=[0.,0.],source_leakage_b=[0.,0.],source_support_low=0.,source_support_high=0.))
    return dict(names=list(review.NAMES),anchor_kw=[[0.],[1.]],weights=[[1.,0.,0.]]*2,
        sigma=np.ones((2,3)),alpha=[2.,2.],beta=[[2.,0.],[0.,2.]],source_low=[0.,0.],source_high=[0.,0.],
        line_sha256={name:review.line_id([.004],[.001]) for name in review.NAMES},
        direction_records=records,global_AC_source_elimination_certified=False)


def algebra_evaluation(model):
    d=np.array([2.,4.]);g=review.gap_values(d,model["source_low"],model["source_high"]); score=float(g.sum())
    return dict(scalar_AC_contrast=d,scalar_affine_contrast=d,scalar_affine_error=[0.,0.],
        source_adjusted_gaps=g,affine_source_adjusted_gaps=g,score=score,
        nominal_projected_half_SSE=10.,affine_source_distance_lower_bound_from_L1=score**2/4,
        source_adjusted_AC_contrast_proxy_squared=score**2/4,full_AC_source_distance_certificate=False,
        terminal_and_internal_voltage_margin=.1,success=True,
        source_derivative_at_action={name:np.zeros((2,2)) for name in review.NAMES})


def test_model_beta_source_support_unit_and_rank_algebra():
    fake=AlgebraReplay(); model=algebra_model()
    out=review.verify_model(fake,model,np.zeros((2,1)),np.zeros((2,1)),projection="joint",rcond=1e-10)
    assert out["unit_weights"] and out["beta_max_absolute_FD_error"]==0
    altered=deepcopy(model);altered["beta"][0][0]=3.
    with pytest.raises(ValueError,match="beta"):
        review.verify_model(fake,altered,np.zeros((2,1)),np.zeros((2,1)),projection="joint",rcond=1e-10)


def test_tampered_fixed_w_AC_score_is_detected():
    fake=AlgebraReplay(); model=algebra_model()
    model.update(source_low=[-.1,-.2],source_high=[.3,.4])
    saved=algebra_evaluation(model);u=np.array([[0.],[1.]])
    out=review.verify_evaluation(fake,model,np.zeros_like(u),np.zeros_like(u),u,saved)
    assert out["score"]==pytest.approx(5.7)
    saved["score"]+=.25
    with pytest.raises(ValueError,match="fixed-w AC score"):
        review.verify_evaluation(fake,model,np.zeros_like(u),np.zeros_like(u),u,saved)


def test_affine_lower_bound_and_AC_proxy_labels_are_distinct():
    fake=AlgebraReplay();model=algebra_model();saved=algebra_evaluation(model);u=np.array([[0.],[1.]])
    saved["full_AC_source_distance_certificate"]=True
    with pytest.raises(ValueError,match="proxy"):
        review.verify_evaluation(fake,model,np.zeros_like(u),np.zeros_like(u),u,saved)
