from itertools import product
import numpy as np
import pytest

pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.joint_five.model import build_joint, open_small, remainder, joint_fit
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.local_rebuild.matched import all_hierarchies


def bank_for(p, q, y, family):
    data = core._prepare_scenarios([dict(P_terminal=p, Q_terminal=q, drop_target=y)])
    return FloorBank(data, family, dict(a=1., b=1., median_root_depth=1.))


def test_five_block_candidates_and_conflicts():
    p = np.eye(5); leaves = [(i,) for i in range(5)]
    bank = bank_for(p, p*0, p, leaves)
    try:
        _, meta = build_joint(bank.scaled, leaves)
        assert len(meta['candidates']) == 25 and meta['conflicts'] == 195
        assert meta['maximum_remainder'] == 5 and not meta['skipped_large_regions']
    finally:
        bank.close()


def test_single_joint_solve_adds_multiple_lines_and_matches_complete_family_lp():
    rng = np.random.default_rng(812); p = rng.uniform(.1, 1., (14, 6)); q = rng.uniform(.1, .4, p.shape)
    coarse = [(i,) for i in range(6)]+[(0, 1, 2), (3, 4, 5), tuple(range(6))]
    R = .2*np.eye(6); X = .1*np.eye(6)
    for s, r, x in [((0, 1, 2), .05, .03), ((3, 4, 5), .05, .03), ((0, 1), .2, .1), ((4, 5), .15, .06)]:
        z = np.array([i in s for i in range(6)], dtype=float); R += r*np.outer(z, z); X += x*np.outer(z, z)
    bank = bank_for(p, q, p@R.T+q@X.T, coarse)
    try:
        current = bank.fit(coarse, preserve=True)
        point, event = joint_fit(bank, current, coarse, seconds=5.)
        assert event['solve_started'] and event['accepted'] and len(event['selected_new']) >= 2
        assert event['candidate_count'] == 6 and event['audit']['valid']
        exact = []
        for h1, h2 in product(all_hierarchies(3), repeat=2):
            extra = [tuple(i for i in s) for s in h1]+[tuple(i+3 for i in s) for s in h2]
            exact.append(bank.fit([*coarse, *extra])['objective'])
        assert point['objective'] == pytest.approx(min(exact), abs=1e-8)
        assert event['dual_bound'] <= min(exact)+1e-8
    finally:
        bank.close()


def test_crossing_zero_loss_lp_is_rejected_by_joint_tree_selection():
    p = np.eye(3); z1 = np.array([1., 1., 0.]); z2 = np.array([0., 1., 1.])
    family = [(i,) for i in range(3)]+[(0, 1, 2)]
    bank = bank_for(p, p*0, p+np.outer(z1, z1)+np.outer(z2, z2), family)
    try:
        current = bank.fit(family, preserve=True)
        point, event = joint_fit(bank, current, family, seconds=5.)
        assert core.is_laminar_family(point['supports'])
        assert point['objective'] == pytest.approx(2/9, abs=1e-8)
    finally:
        bank.close()


def test_preselection_releases_wrong_small_merge_but_preserves_large_boundary():
    p = np.eye(4); z = np.array([0., 1., 1., 0.]); family = [(i,) for i in range(4)]+[(0, 1), tuple(range(4))]
    bank = bank_for(p, p*0, .3*p+.2*np.outer(z, z), family)
    try:
        keep, released = open_small(family, 4)
        assert (0, 1) in released
        point, event = joint_fit(bank, bank.fit(family, preserve=True), keep, seconds=5.)
        assert point['objective'] == pytest.approx(0., abs=1e-8)
        assert (1, 2) in map(tuple, point['supports']) and event['accepted']
    finally:
        bank.close()
    family = [(0, 1), (2, 3), (4, 5), tuple(range(6))]
    keep, released = open_small(family, 8)
    assert tuple(range(6)) in keep and remainder(keep, 8) <= 5


def test_large_polytomy_is_reported_not_silently_treated_as_five():
    p = np.eye(7); family = [(i,) for i in range(7)]
    bank = bank_for(p, p*0, p, family)
    try:
        _, meta = build_joint(bank.scaled, family)
        assert not meta['candidates'] and len(meta['skipped_large_regions']) == 1
        assert meta['maximum_remainder'] == 7
    finally:
        bank.close()


def test_stage_interventions_are_once_each_and_can_repair_rnj_small_error():
    from research_experiments.joint_five.search import search
    p = np.eye(4); z = np.array([0., 1., 1., 0.])
    family = [(i,) for i in range(4)]+[(0, 1), tuple(range(4))]
    results = []
    for pre, tail in ((False, False), (True, True)):
        bank = bank_for(p, p*0, .3*p+.2*np.outer(z, z), family)
        try:
            result = search(bank, family, pre_joint=pre, tail_joint=tail, seconds=6.)
            assert sum(e['kind'] == 'rnj_pre_joint' for e in result['attempts']) == int(pre)
            assert sum(e['kind'] == 'wzzt_tail_joint' for e in result['attempts']) == int(tail)
            results.append(result['point']['objective'])
        finally:
            bank.close()
    assert results[1] < 1e-8 and results[0] > 1e-3


def test_error_audit_separates_size_origin_and_zero_weight_structural_blocks():
    from research_experiments.joint_five.analyze import size_errors, blocking_rows, independent_lp
    true = {(0, 1), tuple(range(6))}; wrong_large = (0, 1, 2, 3, 6, 7)
    initial = dict(supports=[wrong_large], r=[0.], x=[0.])
    point = dict(supports=[wrong_large, (0, 2)], r=[0., .1], x=[0., 0.])
    audit = {tuple(q['support']): q for q in blocking_rows(point, initial, true, 8)}
    assert audit[(0, 1)]['wzzt_small'] and not audit[(0, 1)]['rnj_large']
    assert audit[tuple(range(6))]['rnj_large'] and not audit[tuple(range(6))]['wzzt_small']
    active = {q['size']:q for q in size_errors(point, true, 8)}
    structural = {q['size']:q for q in size_errors(point, true, 8, structural=True)}
    assert active['large']['FP'] == 0 and structural['large']['FP'] == 1
    p = np.eye(3); family = [(i,) for i in range(3)]
    bank = bank_for(p, p*0, .2*p, family)
    try:
        assert independent_lp(bank.data, bank.fit(family))['absolute_difference'] < 1e-9
    finally:
        bank.close()
