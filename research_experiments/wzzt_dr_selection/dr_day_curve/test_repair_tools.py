import numpy as np
import pytest
from .test_day_model import small_problem,small_scenarios
from .repair_tools import polish_curve


def test_small_zero_energy_error_is_fixed_and_rechecked():
    p=small_problem();a=np.array([[-.2,.2+3.e-9]])
    r=polish_curve(p,small_scenarios(),dict(status='target_feasible_cost_incumbent',mean_u=a.tolist()))
    assert abs(np.sum(r['mean_u']))<1.e-15
    assert r['numerical_polish']['maximum_change_kw']<=3.1e-9
    assert r['training']['passed']==1


def test_large_energy_violation_is_not_repaired():
    with pytest.raises(ValueError,match='too large'):
        polish_curve(small_problem(),small_scenarios(),dict(status='sample_target_unresolved',mean_u=[[-.2,.21]]))
