"""Actual context calls, failed entry accounting, and binding restoration."""
import pytest
from . import information_context as context
from . import likelihood
from .information_context_counter import build_centres, ContextBuildError
from .test_information_context import fixture


def originals():
    return context.ac_predict, likelihood.ac_predict, likelihood.fit_candidate


def test_actual_history_jacobian_and_two_real_refits_match_reported_counts():
    history,meter,candidates,fits=fixture()
    before=originals()
    centers,record=build_centres(history,candidates,fits,meter,
        retain_perturbed_points=True,fit_options=dict(max_nfev=200,multistart=1))
    assert originals()==before
    assert len(centers)>=3 and len(record["attempts"])==2
    assert record["fit_calls"]==record["instrumented_fit_calls"]==2
    verification=record["call_counter_verification"]
    assert verification["alias_ac_calls"]==1
    assert verification["module_ac_calls"]>2
    assert record["ac_calls"]==1+verification["module_ac_calls"]
    assert record["call_counter_verification_passed"]
    assert verification["reported_ac_calls"]==record["instrumented_ac_calls"]
    assert verification["reported_fit_calls"]==2
    assert record["call_count_available"] and not record["call_count_unavailable"]


def test_exception_after_one_physical_ac_keeps_denominator_and_restores(monkeypatch):
    history,meter,candidates,fits=fixture()
    fit=fits["star"]
    def exceptional_builder(*args,**kwargs):
        context.ac_predict(candidates[0],fit.r,fit.x,fit.p,fit.q,
                           fit.source_p,fit.source_q,history.root_v)
        raise RuntimeError("history geometry rejected after AC")
    monkeypatch.setattr(context,"build_centres",exceptional_builder)
    before=originals()
    with pytest.raises(ContextBuildError) as caught:
        build_centres(history,candidates,fits,meter)
    assert originals()==before
    error=caught.value
    assert error.ac_calls==1 and error.fit_calls==0
    assert error.alias_ac_calls==1 and error.module_ac_calls==0
    assert error.cause_kind=="RuntimeError" and error.seconds>0
    assert error.call_count_available
    record=error.record()
    assert record["ac_calls"]==1 and not record["call_count_unavailable"]


def test_inner_likelihood_wrapper_delegates_once_and_failure_restores(monkeypatch):
    history,meter,candidates,fits=fixture()
    fit=fits["star"]
    def exceptional_builder(*args,**kwargs):
        outer=likelihood.ac_predict
        def inner_counted(*a,**k):return outer(*a,**k)
        likelihood.ac_predict=inner_counted
        try:
            likelihood.ac_predict(candidates[0],fit.r,fit.x,fit.p,fit.q,
                                  fit.source_p,fit.source_q,history.root_v)
            raise ValueError("inner context failure")
        finally:likelihood.ac_predict=outer
    monkeypatch.setattr(context,"build_centres",exceptional_builder)
    before=originals()
    with pytest.raises(ContextBuildError) as caught:
        build_centres(history,candidates,fits,meter)
    assert originals()==before
    assert caught.value.ac_calls==1 and caught.value.module_ac_calls==1
    assert caught.value.alias_ac_calls==0


def test_failed_fit_attempt_and_failed_ac_are_counted_before_exceptions(monkeypatch):
    history,meter,candidates,fits=fixture()
    def failed_fit(*args,**kwargs):
        likelihood.ac_predict(*args,**kwargs)
        raise RuntimeError("unused")
    def failed_ac(*args,**kwargs):raise RuntimeError("actual AC failed")
    def exceptional_builder(*args,**kwargs):likelihood.fit_candidate()
    monkeypatch.setattr(likelihood,"fit_candidate",failed_fit)
    monkeypatch.setattr(likelihood,"ac_predict",failed_ac)
    monkeypatch.setattr(context,"build_centres",exceptional_builder)
    before=originals()
    with pytest.raises(ContextBuildError) as caught:
        build_centres(history,candidates,fits,meter)
    assert originals()==before
    assert caught.value.fit_calls==1 and caught.value.ac_calls==1
    assert caught.value.cause_kind=="RuntimeError"


def test_successful_report_mismatch_is_exposed_and_observed_counts_preserved(monkeypatch):
    history,meter,candidates,fits=fixture();fit=fits["star"]
    def wrong_reporting(*args,**kwargs):
        context.ac_predict(candidates[0],fit.r,fit.x,fit.p,fit.q,
                           fit.source_p,fit.source_q,history.root_v)
        return [],dict(ac_calls=0,fit_calls=0)
    monkeypatch.setattr(context,"build_centres",wrong_reporting)
    before=originals();_,record=build_centres(history,candidates,fits,meter)
    assert originals()==before
    assert record["ac_calls"]==record["instrumented_ac_calls"]==1
    assert not record["call_counter_verification_passed"]
    assert record["call_counter_verification"]["reported_ac_calls"]==0