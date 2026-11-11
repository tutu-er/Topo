"""Small exact-graph and known-optimum checks; no private grid data."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pytest

from research_experiments.wzzt_dr_selection.dr_pricing.problem import PricingProblem
from research_experiments.wzzt_dr_selection.dr_pricing.bayes_qcqp.solver import solve, build_device_graph, _values


@dataclass
class Evidence:
    Q: np.ndarray
    linear: np.ndarray
    constant: float = 0.
    kind: str = "synthetic_expected_log_odds"

    def value(self,u):
        return self.constant+np.einsum("tr,trs,ts->",u.T,self.Q,u.T)+np.sum(self.linear*u.T)


def problem(resources=1,slots=1,decision_slots=1,energy=False):
    reset=np.zeros((resources,slots),bool);reset[:,0]=True
    terminal=np.zeros_like(reset)
    if energy:
        terminal[:,-1]=True
        if slots==8 and resources>1:
            reset[1,4]=True;terminal[1,3]=True
    config=dict(dt_hours=1. if slots==1 else .25,decision_slots=decision_slots,
        residual_unit="synthetic",information=dict(parameters_frozen=True,actual_future_used=False,truth_used=False),
        selection=dict(history_count=slots,new_count=slots,new_day_weight=1.,historical_loss_coarse=1.,
            historical_loss_fine=1.,relative_band=.03,absolute_tolerance=1e-6,
            candidate_ids=["C","F"],support_counts=[1,2],historically_eligible=[True,True],candidate_count=2),
        probability=dict(target=.9,independent_cells=True,shared_candidate_noise=True,scope="conditional_voltage"))
    arrays=dict(baseline_coarse=np.zeros((slots,1)),baseline_fine=np.zeros((slots,1)),
        effect_coarse=np.ones((1,resources)),effect_fine=np.zeros((1,resources)),sigma=np.ones((slots,1)),
        resource_lower_kw=-np.ones((resources,slots)),resource_upper_kw=np.ones((resources,slots)),
        resource_cost=np.ones((resources,slots)),resource_efficiency=np.ones(resources),
        resource_energy_lower_kwh=np.full(resources,-.2 if energy else -np.inf),
        resource_energy_upper_kwh=np.full(resources,.2 if energy else np.inf),
        resource_reset=reset,resource_terminal=terminal)
    if energy and resources>1:
        arrays["resource_efficiency"][1]=.8
        arrays["resource_lower_kw"][1]=-.7
    return PricingProblem(config,arrays)


def gurobi():
    gp=pytest.importorskip("gurobipy")
    return gp


@pytest.mark.parametrize("requests",[
    [[1.,.7,-1.,.4],[-1.,.8,.5,-.7]],
    [[-.9,1.,1.,-1.],[.9,-.5,-.8,1.]],
    [[0.,0.,0.,0.],[0.,0.,0.,0.]]])
def test_exact_hourly_projection_with_efficiency_reset_and_deadline(requests):
    gp=gurobi();p=problem(2,8,2,True);requests=np.asarray(requests)
    expected=p.device.nominal(requests)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag",0);env.start()
        with gp.Model(env=env) as m:
            m.Params.OutputFlag=0;m.Params.TimeLimit=10;m.Params.FeasibilityTol=1e-9
            x,u,cost=build_device_graph(m,p,requests)
            for index in np.ndindex(requests.shape): m.addConstr(x[index].expr==requests[index])
            m.setObjective(cost);m.optimize()
            assert m.SolCount>0
            np.testing.assert_allclose(_values(u),expected,atol=1e-8,rtol=0)
            assert abs(m.ObjVal-p.device.fees(expected))<1e-8


def test_zero_fee_is_analytically_certified_without_solver(tmp_path):
    p=problem();e=Evidence(np.ones((1,1,1)),np.zeros((1,1)),2.)
    result=solve(p,e,1.,tmp_path/"zero")
    assert result["qualified"] and not result["solver_performed"]
    assert result["objective"]==result["objbound"]==result["mipgap"]==0
    assert not (tmp_path/"zero/model.mps.gz").exists()


def test_rank_one_reverse_quadratic_known_global_optimum(tmp_path):
    gurobi();p=problem();e=Evidence(np.ones((1,1,1)),np.zeros((1,1)))
    result=solve(p,e,.25,tmp_path/"rank1",seconds=15,mip_gap=1e-7)
    assert result["qualified"] and result["solver_performed"]
    assert abs(result["objective"]-.25)<1e-6
    assert abs(result["objbound"]-.25)<1e-6
    assert result["replay_delivery_error"]<1e-8


def test_linear_evidence_and_affine_constant_known_optimum(tmp_path):
    gurobi();p=problem();e=Evidence(np.zeros((1,1,1)),np.ones((1,1)),.1)
    result=solve(p,e,.4,tmp_path/"linear",seconds=15,mip_gap=1e-7)
    assert result["qualified"]
    assert abs(result["objective"]-.09)<1e-6
    assert abs(result["evidence"]-.4)<1e-6


def test_infeasibility_preserved_not_relabelled_success(tmp_path):
    gurobi();p=problem();e=Evidence(np.ones((1,1,1)),np.zeros((1,1)))
    result=solve(p,e,2.,tmp_path/"infeasible",seconds=15)
    assert not result["has_solution"] and not result["qualified"]
    assert result["solver_status"]==3


def test_dense_cross_time_evidence_keeps_original_energy_graph(tmp_path):
    from research_experiments.wzzt_dr_selection.dr_pricing.bayes_qcqp.fixed_f import DenseQuadraticEvidence
    gurobi();p=problem(slots=2,energy=True)
    evidence=DenseQuadraticEvidence(np.array([[1.,-1.],[-1.,1.]]),np.zeros(2),0.,1,2)
    result=solve(p,evidence,.64,tmp_path/"dense_cross_time",seconds=15,mip_gap=1e-7)
    assert result["qualified"] and result["device_audit"]["passed"]
    assert abs(result["objective"]-.08)<1e-6
    assert abs(result["objbound"]-.08)<1e-6
    assert result["replay_delivery_error"]<1e-8
