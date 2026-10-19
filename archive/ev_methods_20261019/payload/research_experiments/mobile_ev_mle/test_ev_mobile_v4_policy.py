"""Counterexamples for the aligned AC policy, without a simulated truth oracle.

Physical path lifting and original parameter domains have separate tests.  The
small doubles below isolate reference sacrifice, common-score inheritance and
the mathematics of a bounded local correction; they never perform an AC fit.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_v4_bounded as bounded
from . import ev_mobile_v4_domain as nested
from . import ev_mobile_v4_policy as policy
from . import likelihood as lk
from .simulation import model


REFERENCES = ("a", "b")


def profile(values=(0., 0.), *, success=True):
    values = dict(zip(REFERENCES, values))
    nearest = [dict(a=a, half_gap=float(values[a])) for a in REFERENCES]
    return dict(success=success, score=float(np.mean(list(values.values()))),
        nearest=nearest, aggregation=dict(nearest=deepcopy(nearest)),
        references=[dict(a=a) for a in REFERENCES],
        pairs=[dict(a=a,b=b,success=success,half_gap=float(values[a]),
                    gradient_per_kw=np.zeros((1,1)))
               for a in REFERENCES for b in REFERENCES if a != b])


def test_mean_improvement_does_not_imply_each_reference_improves():
    parent, candidate = profile((4., 0.)), profile((1., 4.))
    assert candidate["score"] > parent["score"]
    result = policy.reference_floor(parent, candidate, expected_references=REFERENCES)
    assert result["complete"]
    assert not result["success"]
    assert result["violated"] == ["a"]
    assert result["slacks"] == {"a": -3., "b": 4.}


@pytest.mark.parametrize("corruption", ["both_missing", "duplicate", "nan", "inf", "extra"])
def test_floor_rejects_false_completeness_and_nonfinite_scores(corruption):
    parent, candidate = profile((1., 2.)), profile((2., 3.))
    if corruption == "both_missing":
        parent["nearest"] = parent["nearest"][:1]
        candidate["nearest"] = candidate["nearest"][:1]
    elif corruption == "duplicate":
        candidate["nearest"].append(dict(a="a", half_gap=100.))
    elif corruption in ("nan", "inf"):
        candidate["nearest"][0]["half_gap"] = float(corruption)
    else:
        candidate["nearest"].append(dict(a="c", half_gap=100.))
    result = policy.reference_floor(parent, candidate, expected_references=REFERENCES)
    assert not result["complete"]
    assert not result["success"]


@pytest.mark.parametrize("tolerance", [-1., np.nan, np.inf])
def test_floor_does_not_hide_a_reference_loss_with_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        policy.reference_floor(profile(), profile(), tolerance=tolerance,
                               expected_references=REFERENCES)


def context_inputs():
    count, n, T = 100, 2, 2
    p = np.full((count,n), .1)
    p[-4:] += .037  # Those four already observed rows must actually enter D.
    observed = lk.Observations(p=p, q=.3*p, v=np.ones_like(p),
                              master_p=np.zeros(count), master_q=np.zeros(count))
    meter, future_meter = model(n), model(n)
    bank = tuple(SimpleNamespace(name=name,m=1,n=n,supports=(tuple(range(n)),)) for name in REFERENCES)
    fits = {a:SimpleNamespace(success=True,r=np.array([.02]),x=np.array([.01]),
                 source_p=np.full(count,.01), source_q=np.full(count,.003))
            for a in REFERENCES}
    forecast = dict(p=np.full((T,n),.12),q=np.full((T,n),.04),root_v=np.ones(T),
        source_by_candidate={a:dict(source_p=np.full(T,.01),source_q=np.full(T,.003))
                             for a in REFERENCES},
        metadata=dict(uses_truth=False,uses_future_noise=False,
                      observed_history_only=True,observed_count=count))
    return observed,meter,bank,fits,future_meter,forecast


def make_context(inputs):
    return policy.build_context(*inputs,base_kw=20.,source_p_bounds=(-.5,.5),
                                source_q_bounds=(-.5,.5))


def test_context_passes_actual_observed_D_to_legacy_engine_and_identity(monkeypatch):
    inputs=context_inputs(); observed,meter,bank,fits,future_meter,forecast=inputs
    context=make_context(inputs)
    assert context["_rebuild_inputs"]["training"] is observed
    assert context["_rebuild_inputs"]["training"].t == 100
    np.testing.assert_array_equal(context["_rebuild_inputs"]["training"].p[-4:], observed.p[-4:])
    for a in REFERENCES:
        assert not context["lines"][a]["r"].flags.writeable
        assert not np.shares_memory(context["lines"][a]["r"],fits[a].r)

    class LegacyEngine:
        def __init__(self, ctx, *, fit_options, seed):
            self.training=ctx["_rebuild_inputs"]["training"]
            self.tm=ctx["_rebuild_inputs"]["train_meter"]
            assert self.training.t == 100
        def evaluate(self, power):
            result=profile()
            result.update(uses_validation=False,uses_training_history=True)
            return result
    monkeypatch.setattr(policy.complete,"JointPredictiveEngine",LegacyEngine)
    objective=policy.CommonACObjective(context,{},seed=7)
    record=objective.evaluate(np.zeros((2,2)))
    assert record["uses_observed_D"] and record["uses_validation"]
    assert objective.engine.training is observed
    assert record["history_scope"] == "same complete D as actual evaluation"
    changed=deepcopy(inputs)
    changed[0].p[-1,0] += .001
    other=policy.CommonACObjective(make_context(changed),{},seed=7)
    assert objective.objective_id != other.objective_id


@pytest.mark.parametrize("bad_input", ["H_only_metadata", "short_source_P", "short_source_Q", "nan_source_Q"])
def test_observed_D_claim_requires_actual_complete_historical_inputs(bad_input):
    inputs=list(context_inputs())
    if bad_input == "H_only_metadata":
        inputs[-1]["metadata"]["observed_count"]=96
    elif bad_input == "short_source_P":
        inputs[3]["a"].source_p=inputs[3]["a"].source_p[:96]
    elif bad_input == "short_source_Q":
        inputs[3]["a"].source_q=inputs[3]["a"].source_q[:96]
    else:
        inputs[3]["a"].source_q[-1]=np.nan
    with pytest.raises(ValueError):
        make_context(inputs)


def analytic_alignment_inputs(monkeypatch):
    """q(u)=min_{-1<=z<=1} .5*(2+3*u+z)^2 near u=0."""
    center=np.zeros((1,1))
    domain=bounded.make_domain([-1.],[1.],anchor=[-1.])
    bank=tuple(SimpleNamespace(name=a,m=1) for a in REFERENCES)
    fits={a:SimpleNamespace(r=np.array([.02]),x=np.array([.01]),
                 source_p=np.zeros(1),source_q=np.zeros(1)) for a in REFERENCES}
    context=dict(bank=bank,train_fits=fits,forecasts={a:dict(source_p=np.zeros(1),
         source_q=np.zeros(1)) for a in REFERENCES},information_scope="same actual D")
    rows=[dict(a=a,b=b,residual=np.array([2.]),B=np.array([[3.]]),J=np.array([[1.]]))
          for a in REFERENCES for b in REFERENCES if a != b]
    data=dict(pairs=rows,diagnostics={})
    monkeypatch.setattr(policy.geometry,"geometry_inputs",lambda context,power:deepcopy(data))
    monkeypatch.setattr(policy.bounded,"physical_delta_domain",lambda *args,**kwargs:domain)
    target=profile((5.,7.))
    for row in target["pairs"]:
        row["gradient_per_kw"]=np.array([[-2. if row["a"]=="a" else 4.]])
    options=dict(r_bounds=(1e-6,.1),x_bounds=(1e-6,.1),
                 source_p_bounds=(-.5,.5),source_q_bounds=(-.5,.5),ratio_bounds=(.1,2.))
    return context,center,target,options,data


def analytic_raw_value(u):
    residual=2.+3.*float(u)
    z=np.clip(-residual,-1.,1.)
    return .5*(residual+z)**2


def test_parent_correction_matches_full_signed_value_and_price_by_finite_difference(monkeypatch):
    context,p,target,options,data=analytic_alignment_inputs(monkeypatch)
    real_solve=bounded.solve_pair
    def scipy_solve(pair,u,**kwargs):
        return real_solve(pair,u,backend="scipy",seconds=1.)
    monkeypatch.setattr(policy.bounded,"solve_pair",scipy_solve)
    pairs,evidence=policy.aligned_pairs(context,p,target,options)
    assert len(pairs)==2
    for pair,row in zip(pairs,evidence["pairs"]):
        wanted=next(r for r in target["pairs"] if r["a"]==pair.a_label)
        def corrected(u):
            return analytic_raw_value(u)+pair.affine_offset+float(pair.affine_control[0])*u
        assert corrected(0.) == pytest.approx(wanted["half_gap"],abs=1e-10)
        h=1e-5
        derivative=(corrected(h)-corrected(-h))/(2*h)
        assert derivative == pytest.approx(float(wanted["gradient_per_kw"][0,0]),abs=1e-8)
        assert row["parent_QP"]["explanation_audit"]["feasible"]
    assert evidence["historical_scope"] == "same actual D"
    assert not evidence["exact_AC_bound"]


def test_finite_inner_gap_limits_alignment_claim_instead_of_exact_matching(monkeypatch):
    context,p,target,options,data=analytic_alignment_inputs(monkeypatch)
    # z=-.8 is feasible but not optimal: residual1.2, primal.72; q(0)=.5.
    approximate=dict(success=True,inner_gap=.24,primal=.72,primal_upper=.73,
        dual_lower=.49,raw_price=np.array([3.6]),price_error_norm_bound=3.*np.sqrt(.48))
    monkeypatch.setattr(policy.bounded,"solve_pair",lambda *args,**kwargs:deepcopy(approximate))
    pairs,evidence=policy.aligned_pairs(context,p,target,options,parent_relative_gap=1.)
    for pair,row in zip(pairs,evidence["pairs"]):
        wanted=row["full_AC_half_gap"]
        value=analytic_raw_value(0.)+pair.affine_offset
        assert wanted-.24-1e-12 <= value <= wanted+1e-12
        assert value < wanted-.1  # A positive gap must not be called exact.
        derivative=3.+float(pair.affine_control[0])
        assert abs(derivative-float(np.asarray(row["full_AC_price"]).ravel()[0])) <= \
               row["price_alignment_error_bound"]+1e-12
        assert pair.affine_offset == pytest.approx(wanted-approximate["primal_upper"])


@pytest.mark.parametrize("invalid_gap", [np.nan, -1e-3, 1.])
def test_uncontrolled_parent_QP_cannot_be_used_for_alignment(monkeypatch,invalid_gap):
    context,p,target,options,data=analytic_alignment_inputs(monkeypatch)
    monkeypatch.setattr(policy.bounded,"solve_pair",lambda *args,**kwargs:
        dict(inner_gap=invalid_gap,primal_upper=.5))
    with pytest.raises(ValueError,match="gap"):
        policy.aligned_pairs(context,p,target,options)


@pytest.mark.parametrize("corruption", ["missing_target", "duplicate_target", "duplicate_raw"])
def test_alignment_keeps_all_directed_competitors_exactly_once(monkeypatch,corruption):
    context,p,target,options,data=analytic_alignment_inputs(monkeypatch)
    if corruption == "missing_target":
        target["pairs"].pop()
    elif corruption == "duplicate_target":
        target["pairs"][1]=deepcopy(target["pairs"][0])
    else:
        data["pairs"][1]=deepcopy(data["pairs"][0])
    with pytest.raises(ValueError,match="directed pair"):
        policy.aligned_pairs(context,p,target,options)


class PowerOnlyDomain:
    """Policy double; physical nesting is independently checked elsewhere."""
    def __init__(self,name):
        self.name=name
    def decode(self,x):
        return dict(u_kw=np.asarray(x,float).reshape(1,1))
    def check(self,x,**kwargs):
        return dict(feasible=np.asarray(x).shape==(1,1),issues=[])


def install_policy_problem(monkeypatch,*,mobile_bidirectional_values=(1.,4.),failure=None):
    table={0:(0.,0.),1:(4.,0.),2:(0.,3.),3:mobile_bidirectional_values}
    class FakeObjective:
        def __init__(self,context,fit_options,*,seed):
            self.objective_id="one-D-one-forecast-one-theta-one-seed"
            def evaluate(power):
                key=int(np.asarray(power).ravel()[0])
                if failure=="AC" and key==3:
                    return dict(success=False,failure="retained joint-fit failure")
                return profile(table[key])
            self.cache=nested.ACScoreCache(self.objective_id,evaluate)
            self.engine=SimpleNamespace(counters=lambda:dict(evaluations=self.cache.evaluations))
    monkeypatch.setattr(policy,"CommonACObjective",FakeObjective)
    monkeypatch.setattr(policy,"aligned_pairs",lambda *args,**kwargs:(["frozen"],{}))
    monkeypatch.setattr(policy,"tighten_operation_bounds",lambda domain:dict(name=domain.name))
    monkeypatch.setattr(policy,"factor_bound_provider",lambda *args:None)
    # Preserve actual cached score and full operation through real inherit_plan.
    monkeypatch.setattr(policy.nested,"lift_operation",lambda parent,child,x,**kwargs:np.array(x,copy=True))
    action={"fixed_charge":0,"fixed_bidir":1,"mobile_charge":2,"mobile_bidir":3}
    def master(pairs,operation,**kwargs):
        name=operation["name"]
        if failure=="MILP" and name=="mobile_bidir":
            raise RuntimeError("retained finite-budget MILP failure")
        return dict(proposals=[dict(operation_x=np.array([[float(action[name])]]),
                                   score_lower=100.,score_upper=101.)])
    monkeypatch.setattr(policy.master,"solve_bounded_milp",master)
    domains={name:PowerOnlyDomain(name) for name in nested.DOMAIN_FLAGS}
    squares={name:np.zeros((1,1)) for name in domains}
    context=dict(bank=tuple(SimpleNamespace(name=a) for a in REFERENCES))
    return context,domains,squares,{name:0. for name in domains}


def run_tiny_family(problem,*,guard):
    return policy.run_family(*problem,{},guard=guard,seed=7,total_seconds=1000.)


def test_same_mean_goal_inherits_best_parent_and_guard_exposes_reference_sacrifice(monkeypatch):
    problem=install_policy_problem(monkeypatch)
    aligned=run_tiny_family(problem,guard=False)
    guarded=run_tiny_family(problem,guard=True)
    for result in (aligned,guarded):
        plans=result["plans"]
        assert plans["fixed_bidir"].score>=plans["fixed_charge"].score
        assert plans["mobile_charge"].score>=plans["fixed_charge"].score
        assert plans["mobile_bidir"].score>=max(plans[p].score for p in nested.PARENTS["mobile_bidir"])
        assert {p.objective_id for p in plans.values()}=={result["objective_id"]}
    assert aligned["plans"]["mobile_bidir"].score==2.5
    assert guarded["plans"]["mobile_bidir"].score==2.
    step=guarded["logs"]["mobile_bidir"]["steps"][0]
    assert step["mean_improves"] and not step["accepted"]
    assert step["reference_floor"]["violated"]==["a"]
    assert step["rejection_reason"]=="reference_sacrifice"
    # Same power, different domain: parent is reused, not independently refitted.
    assert len(guarded["cache"])==4


def test_guard_uses_best_actual_parent_not_impossible_componentwise_parent_floor(monkeypatch):
    problem=install_policy_problem(monkeypatch,mobile_bidirectional_values=(4.2,.2))
    result=run_tiny_family(problem,guard=True)
    log=result["logs"]["mobile_bidir"]
    assert log["steps"][0]["accepted"]
    assert log["steps"][0]["reference_floor"]["floor"]=={"a":4.,"b":0.}
    assert result["plans"]["mobile_bidir"].score==pytest.approx(2.2)
    # The other parent's b=3 is intentionally not imposed on the new plan.
    assert log["steps"][0]["reference_floor"]["slacks"]["b"]==pytest.approx(.2)


@pytest.mark.parametrize("failure",["AC","MILP"])
def test_proposal_failure_keeps_scored_parent_and_explicit_failure_denominator(monkeypatch,failure):
    problem=install_policy_problem(monkeypatch,failure=failure)
    result=run_tiny_family(problem,guard=True)
    best=result["plans"]["mobile_bidir"]
    assert best.score==2.
    assert best.action_key==result["plans"]["fixed_bidir"].action_key
    assert result["logs"]["mobile_bidir"]["planning_failure"]
    if failure=="AC":
        assert any(not row["success"] for row in result["cache"].values())


def test_explicit_gradient_audit_uses_pair_prices_and_reports_step_convergence():
    from .ev_mobile_v4_gradient_check import audit_directed_derivatives
    parent_power=np.array([[.2,.1],[.3,.1]])
    weights=np.array([[1.,0.],[2.,3.]])
    calls=[]
    def value(power):
        x=float(np.sum(power*weights)); f=x**3+x**2+2*x
        result=profile((f,2*f))
        for row in result['pairs']:
            coefficient=1. if row['a']=='a' else 2.
            row['gradient_per_kw']=coefficient*(3*x*x+2*x+2)*weights
        return result
    def evaluate(power):
        calls.append(np.array(power,copy=True))
        return value(power)
    result=audit_directed_derivatives(evaluate,parent_power,value(parent_power),labels=REFERENCES)
    assert result['status']=='complete'
    assert result['derivative_slots']==8 and result['valid_derivative_slots']==8
    assert len(calls)==8 and not result['parent_refitted']
    assert not result['uses_truth'] and not result['changes_policy']
    for direction in result['directions']:
        for a in REFERENCES:
            rows=[r for r in result['derivatives'] if r['direction']==direction and r['a']==a]
            coarse=next(r['absolute_error'] for r in rows if r['h_kw']==.01)
            fine=next(r['absolute_error'] for r in rows if r['h_kw']==.005)
            # A cubic has an exact O(h^2) central-difference error.
            assert coarse==pytest.approx(4*fine,rel=1e-5,abs=1e-10)


def test_gradient_audit_keeps_failed_perturbations_and_all_derivative_slots():
    from .ev_mobile_v4_gradient_check import audit_directed_derivatives
    u=np.zeros((2,2)); base=profile()
    for row in base['pairs']:
        row['gradient_per_kw']=np.zeros_like(u)
    def evaluate(power):
        if power[0,0]>.007:
            raise RuntimeError('one perturbed local optimizer failed')
        return deepcopy(base)
    result=audit_directed_derivatives(evaluate,u,base,labels=REFERENCES)
    assert result['status']=='partial_failure'
    assert result['derivative_slots']==8
    assert result['failed_derivative_slots']==2
    assert result['valid_derivative_slots']==6
    assert len(result['perturbations'])==8
    assert sum(not row['success'] for row in result['perturbations'])==1
