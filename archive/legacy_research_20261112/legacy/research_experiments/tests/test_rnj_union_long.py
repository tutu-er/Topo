from copy import deepcopy
from time import perf_counter
import numpy as np
import pytest
pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical
from research_experiments.rnj_union_long.model import DeadlineBank, training_candidates, joint_rebuild, pure_search


def make_bank(n, true):
    p = np.eye(n); y = np.eye(n)
    for C in true:
        z = np.array([i in C for i in range(n)], float); y += np.outer(z, z)
    data = core._prepare_scenarios([dict(P_terminal=p, Q_terminal=p*0, drop_target=y)])
    return data, [*((i,) for i in range(n)), tuple(range(n))]


def test_joint_reconstruction_mixes_parent_blocks_and_removes_false_ancestors():
    data, basis = make_bank(4, [(0, 1), (2, 3)])
    pool = canonical([*basis, (0, 1), (0, 1, 2), (2, 3), (1, 2, 3)])
    bank = DeadlineBank(data, pool, dict(a=1., b=1., median_root_depth=1.), deadline=perf_counter()+30)
    try:
        parents = [bank.fit([*basis, (0, 1), (0, 1, 2)], preserve=True), bank.fit([*basis, (2, 3), (1, 2, 3)], preserve=True)]
        assert min(p['objective'] for p in parents) > 1e-3
        rebuilt = joint_rebuild(bank, pool, parents, penalty=.001, deadline=perf_counter()+5)
        assert rebuilt['point']['objective'] < 1e-8
        assert {tuple(C) for C in rebuilt['point']['supports'] if 1 < len(C) < 4} == {(0, 1), (2, 3)}
        assert rebuilt['event']['audit']['valid']
        assert rebuilt['event']['bound'] <= rebuilt['score']+1e-8
    finally: bank.close()


def test_pool_does_not_use_validation_choices():
    point = dict(supports=[(0,), (1,), (2,), (0, 1, 2), (0, 1)], r=[1.]*5, x=[0.]*5, objective=.1)
    record = dict(trajectory=dict(path=[point]), exit=dict(point=point, raw_point=point), choices={'injected': '(0,2)'})
    common = dict(bank=[point], initial=point['supports'])
    a, _ = training_candidates(common, record, record, [(1, 2)])
    changed = deepcopy(record); changed['choices'] = {'point': {'supports': [(0, 2)]}}
    b, _ = training_candidates(common, changed, changed, [(1, 2)])
    assert a == b and (0, 2) not in a and (1, 2) in a


def test_pure_starts_with_only_leaf_full_basis_and_finds_data_supported_set():
    data, basis = make_bank(3, [(1, 2)])
    bank = DeadlineBank(data, basis, dict(a=1., b=1., median_root_depth=1.), deadline=perf_counter()+30)
    try:
        result = pure_search(bank, structural_deadline=perf_counter()+10, single_mip=5)
        assert result['initial_supports'] == canonical(basis)
        assert result['no_RNJ_NJ_supports']
        assert result['point']['objective'] < 1e-8
        assert (1, 2) in canonical(result['point']['supports'])
        assert all(a['kind'] == 'pure_wzzt_extension' for a in result['attempts'])
        assert len(result['path']) == 1+sum(a['accepted'] for a in result['attempts'])
        assert all(max(r,x)>1e-9 for point in result['path']
                   for C,r,x in zip(point['supports'],point['r'],point['x']) if 1<len(C)<3)
    finally: bank.close()
