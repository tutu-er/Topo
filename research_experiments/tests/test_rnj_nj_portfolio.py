import numpy as np
from research_experiments.rnj_nj_portfolio.run import nj_family, select_whole_tree
from research_experiments.local_rebuild.model import canonical
from rnj_wzzt.estimation import laminar_l1_milp as core


def test_known_additive_geometry_nj_recovers_pairs_and_is_scale_invariant():
    kernel = .2*np.ones((4, 4))+.5*np.eye(4)
    for C, w in [((0, 1), .4), ((2, 3), .6)]:
        z = np.array([i in C for i in range(4)], float)
        kernel += w*np.outer(z, z)
    a, _ = nj_family(kernel, [2, 3, 4, 5]); b, _ = nj_family(17*kernel, [2, 3, 4, 5])
    assert a == b
    assert {s for s in a if 1 < len(s) < 4} == {(0, 1), (2, 3)}


def row(source, blocks, internal, support):
    return dict(source=source, blocks=blocks, validation=float(np.mean(blocks)), internal=internal,
                point=dict(supports=canonical([*((i,) for i in range(3)), (0, 1, 2), support])))


def test_existing_paired_rule_is_a_sensitivity_not_the_strict_loss_rule():
    a = row('full', [.1]*24, 3, (0, 1))
    b = row('sparse', [.08, .1201]*12, 2, (1, 2))
    assert select_whole_tree([a, b], 'strict')['source'] == 'full'
    assert select_whole_tree([a, b], 'paired')['source'] == 'sparse'


def test_portfolio_selects_one_valid_tree_without_union_of_crossing_supports():
    a = row('full', [.2]*24, 1, (0, 1)); b = row('sparse', [.1]*24, 1, (1, 2))
    q = select_whole_tree([a, b], 'strict')
    assert q['point'] == b['point']
    assert core.is_laminar_family(q['point']['supports'])
    assert not core.is_laminar_family(canonical([*a['point']['supports'], *b['point']['supports']]))
