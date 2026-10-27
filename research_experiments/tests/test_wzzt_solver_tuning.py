"""Counterexamples and independent electrical checks for the research pilot."""
import numpy as np
import pandas as pd
import pytest

from research_experiments.wzzt_solver_tuning.policy import extension_decision
from research_experiments.wzzt_solver_tuning.profile_gaps import primal_audit
from research_experiments.wzzt_solver_tuning.ac_screen import atom_network, ac_score


def test_worse_incumbent_does_not_prove_search_has_no_gain():
    result = extension_decision(1., 1.01, .5, feasible=True)
    assert result["decision"] == "refine_or_report_unresolved"
    assert result["gain_guaranteed"] < 0 < result["gain_possible"]


def test_acceptance_guarantees_two_thirds_of_possible_gain():
    rng = np.random.default_rng(42)
    for _ in range(500):
        current = 10**rng.uniform(-5, 1)
        lower, upper = sorted(rng.uniform(0, 1.2*current, 2))
        result = extension_decision(current, upper, lower, feasible=True,
                                    uncertainty_fraction=.5)
        if result["decision"] == "accept_bounded_improvement":
            # Even the optimum at the lowest allowed objective cannot improve
            # the current model by over 1.5 times this candidate's improvement.
            assert (current-upper)/(current-lower) >= 2/3-1e-12


def test_missing_primal_or_bound_remains_unresolved():
    assert extension_decision(1., .8, None, feasible=True)["decision"] == "refine_or_report_unresolved"
    assert extension_decision(1., .8, .79, feasible=False)["decision"] == "refine_or_report_unresolved"
    # A valid lower bound can establish stopping even without an incumbent.
    assert extension_decision(1., None, .9995, feasible=False)["decision"] == "stop_bound_proves_no_material_gain"


def test_raw_audit_rejects_fractional_and_infeasible_points():
    raw = dict(constraint_row_index=np.array([0]), constraint_col_index=np.array([0]),
        constraint_data=np.array([1.]), constraint_lower=np.array([.5]), constraint_upper=np.array([1.]),
        lower_bounds=np.array([0.]), upper_bounds=np.array([1.]), integrality=np.array([1]),
        objective_coefficients=np.array([2.]))
    assert primal_audit(raw,np.array([1.]))["feasible"]
    assert not primal_audit(raw,np.array([.75]))["feasible"]
    assert not primal_audit(raw,np.array([0.]))["feasible"]


def test_ac_factor_two_matches_exact_one_line_equation():
    from rnj_wzzt.models.ac_powerflow import solve_ac_power_flow_timeseries
    net = atom_network([10],[(0,)],[.02],[.04])
    p=pd.DataFrame({10:[.05,.08]});q=pd.DataFrame({10:[.02,.03]});root=pd.Series([1.,1.01])
    term=root.to_numpy()**2-2*(.01*p[10].to_numpy()+.02*q[10].to_numpy())
    v=np.sqrt((term+np.sqrt(term**2-4*(.01**2+.02**2)*(p[10].to_numpy()**2+q[10].to_numpy()**2)))/2)
    result=solve_ac_power_flow_timeseries(net,p,q,v_root=root)
    np.testing.assert_allclose(result["V_bus_mag"][10],v,atol=1e-10,rtol=0)
    score=ac_score([10],[(0,)],[.02],[.04],(p,q,pd.DataFrame({10:v}),root))
    assert score["valid"] and score["relative_voltage_rmse"]<1e-10
    assert score["max_active_balance_error"]<1e-10
    assert np.isfinite(score["block_mse_se"])


def test_ac_rejects_crossing_supports():
    with pytest.raises(ValueError,match="nonlaminar"):
        atom_network([10,11,12],[(0,),(1,),(2,),(0,1),(1,2)],[.01]*5,[.01]*5)
