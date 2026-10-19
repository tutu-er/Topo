"""Information-boundary and local-solver-gap reporting tests; no fitting."""
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_v4_interface_check as check
from . import gradient_price_study as io
from .information_context import meter_to_record
from .likelihood import Observations
from .simulation import model


class NoActualTruth(dict):
    def __getitem__(self, key):
        if key in ("actual", "truth"):
            raise AssertionError("interface check must not read simulator truth or actual future")
        return super().__getitem__(key)


def fixture(monkeypatch):
    bank = {name:SimpleNamespace(name=name, n=1, m=1) for name in ("a","b")}
    D = Observations(p=np.full((2,1), .1), q=np.full((2,1), .03), v=np.ones((2,1)),
        root_v=np.ones(2), master_p=np.full(2,.2), master_q=np.full(2,.06))
    meter = meter_to_record(model(1))
    trained = [dict(name=name, r=[.002], x=[.0005], source_p=[.1,.11], source_q=[.03,.04])
               for name in bank]
    fits = {name:dict(loss=100. + 10.*index, state=dict(r=[.002], x=[.0005],
        source_p=[.1,.11,.12], source_q=[.03,.04,.05]),
        blocks={label:dict(residual=np.zeros((T,3)).tolist()) for label,T in (("training",2),("future",1))})
            for index,name in enumerate(bank)}
    record = NoActualTruth(spec=dict(seed=10,bank="n3"), true_name="a", config=dict(
        base_kw=20., training={}, source={}, ratio_bounds=[.1,.4]), observed=io.observations_record(D),
        observed_meter=meter, future_meter=meter, training_result=dict(fits=trained),
        forecast=dict(p=[[.1]], q=[[.03]], root_v=[1.], source_by_candidate={
            name:dict(source_p=[.12],source_q=[.05]) for name in bank}),
        methods={"guarded__fixed_charge":dict(action_key="action",power_kw=[[2.]])},
        reporting_cache={"action":dict(success=True,references=[dict(a="a",fits=fits)])},
        actual=object(), truth=object())
    monkeypatch.setattr(check.audit, "_bank", lambda record:bank)
    def predict(candidate,r,x,p,q,sp,sq,root):
        return dict(v=np.ones_like(p),master_p=np.asarray(p).sum(axis=1)+sp,
                    master_q=np.asarray(q).sum(axis=1)+sq)
    def replay(candidate,state,blocks,cfg):
        return dict(loss=fits[candidate.name]["loss"],
            blocks={label:dict(residual=np.zeros((len(obs["p"]),3))) for label,obs,meter in blocks})
    return record,predict,replay


def test_same_nominal_target_without_future_truth_and_local_loss_gap_reported(monkeypatch):
    record,predict,replay = fixture(monkeypatch)
    captured = []
    def evaluate(bank,D,dm,future,fm,fits,options,seed):
        captured.append((future,options,seed))
        return dict(complete=True,comparisons={"a":dict(joint=dict(loss=101.)),"b":dict(joint=dict(loss=110.))})
    result = check.check_record(record,dict(multistart_predictive=1,max_nfev=400),
        evaluator=evaluate,predictor=predict,replayer=replay)
    assert result["success"]
    assert not result["numerical_agreement"]
    assert result["max_absolute_candidate_loss_difference"] == 1.
    assert result["max_absolute_half_gap_difference"] == .5
    assert result["signed_half_gap_comparison"]["b"]["difference"] == -.5
    assert result["read_actual_future"] is False and result["read_generating_parameters"] is False
    future,options,seed = captured[0]
    np.testing.assert_allclose(future.p, np.full((1,1),.2) + np.asarray(model(1).bias_p))
    assert options["voltage_bounds"] == (.85,1.15)
    assert seed == 4010


def test_equal_sse_with_different_residual_target_is_not_accepted_as_same_target(monkeypatch):
    record,predict,replay = fixture(monkeypatch)
    def changed(candidate,state,blocks,cfg):
        result = replay(candidate,state,blocks,cfg)
        result["blocks"]["future"]["residual"][0,0] = .01
        return result
    def evaluate(*args):
        return dict(complete=True,comparisons={"a":dict(joint=dict(loss=100.)),"b":dict(joint=dict(loss=110.))})
    result = check.check_record(record,dict(multistart_predictive=1,max_nfev=400),
        evaluator=evaluate,predictor=predict,replayer=changed)
    assert not result["success"]
    assert result["numerical_agreement"]
    assert all(not row["same_nominal_target_verified"] for row in result["same_nominal_target_replays"].values())


def test_missing_candidate_optimizer_keeps_incomplete_interface_result(monkeypatch):
    record,predict,replay = fixture(monkeypatch)
    def evaluate(*args):
        return dict(complete=False,comparisons={"a":dict(joint=dict(loss=100.))},failures={"b":"local fit failed"})
    result = check.check_record(record,dict(multistart_predictive=1,max_nfev=400),
        evaluator=evaluate,predictor=predict,replayer=replay)
    assert not result["success"] and not result["numerical_agreement"]
    assert result["actual_entry_nearest_competitor"] is None
    assert result["rerun_evaluation"]["failures"]["b"] == "local fit failed"
