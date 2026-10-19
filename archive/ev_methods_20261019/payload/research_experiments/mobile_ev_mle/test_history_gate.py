import numpy as np
from .history_gate import historical_candidate_pvalues,compatible_history


def test_identical_historical_models_do_not_get_screened():
    result=historical_candidate_pvalues([0.],[[0.],[0.]],[0,1],B=99,seed=1)
    assert result["pvalues"]==[1.,1.]
    assert compatible_history(result)["status"]=="active_discrimination_needed"


def test_separated_history_requires_no_further_finite_family_design():
    result=historical_candidate_pvalues([0.],[[0.],[20.]],[0,1],B=99,seed=2)
    assert compatible_history(result)["retained"]==[0]
    assert compatible_history(result)["status"]=="already_resolved_in_finite_family"


def test_unpruned_composite_null_keeps_a_matching_atom():
    result=historical_candidate_pvalues([10.],[[0.],[10.],[30.]],[0,0,1],B=99,seed=3)
    assert result["pvalues"][0]==1.
    assert compatible_history(result)["retained"]==[0]
