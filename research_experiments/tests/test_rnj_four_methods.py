"""Behavioral checks for the ordinary baseline used in the four-way comparison."""
import numpy as np
import pytest
pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.local_rebuild.model import canonical
from research_experiments.rnj_four_methods.run import ordinary_search, LIMITS


def example(initial_clade=None):
    p = np.eye(3)
    z = np.array([0., 1., 1.])
    y = p + np.outer(z, z)
    data = core._prepare_scenarios([dict(P_terminal=p, Q_terminal=0*p, drop_target=y)])
    family = canonical([*((i,) for i in range(3)), (0, 1, 2), *([initial_clade] if initial_clade else [])])
    bank = FloorBank(data, canonical([*family, (1, 2)]), dict(a=1., b=1., median_root_depth=1.), seconds=30.)
    return bank, family


def test_ordinary_keeps_wrong_rnj_merge_without_repair_even_below_five_blocks():
    bank, family = example((0, 1))
    try:
        result = ordinary_search(bank, family, {**LIMITS, 'total': 5., 'per_mip': 2.})
        assert not result['pre_joint'] and not result['tail_joint']
        assert all(a['kind'] == 'wzzt_extension' for a in result['attempts'])
        assert (0, 1) in canonical(result['point']['supports'])
        assert result['point']['objective'] == pytest.approx(2/9, abs=1e-8)
        assert (1, 2) not in canonical(result['point']['supports'])
    finally:
        bank.close()


def test_ordinary_still_extends_a_small_unresolved_region_instead_of_switching_to_joint():
    bank, family = example()
    try:
        result = ordinary_search(bank, family, {**LIMITS, 'total': 5., 'per_mip': 2.})
        assert result['point']['objective'] < 1e-8
        assert (1, 2) in canonical(result['point']['supports'])
        assert any(a.get('accepted') for a in result['attempts'])
        assert all(a['kind'] == 'wzzt_extension' for a in result['attempts'])
    finally:
        bank.close()
