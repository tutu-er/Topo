"""Independent geometric/analytic checks of the symmetric design core."""
from itertools import product

import numpy as np
import pytest

try:
    from .symmetric import solve_soft_history_qp, symmetric_scores
except ImportError:
    from symmetric import solve_soft_history_qp, symmetric_scores


def test_history_supported_closest_pair_has_hand_calculated_distance():
    # Cutoff=4 retains atoms 0/1 and 2/3, but excludes atom 4 (loss=9).
    actual = symmetric_scores([0.], [[0.], [2.], [0.], [1.], [3.]],
                              {"probe": [[0., 0.], [3., 0.], [8., 0.], [5., 0.], [3., 0.]]},
                              [0, 0, 1, 1, 1], cutoff=4.)
    assert actual["supports"] == {"0": [0, 1], "1": [2, 3]}
    assert actual["actions"]["probe"]["distance_squared"] == pytest.approx(4.)
    assert actual["actions"]["probe"]["closest_atom_pair"] == [1, 3]
    assert actual["actions"]["probe"]["evaluated_atom_pairs"] == 4


def test_overlap_is_zero_with_nonzero_historical_loss():
    actual = symmetric_scores([7.], [[5.], [9.]], {"probe": [[4., 3.], [4., 3.]]}, [0, 1])
    assert actual["historical_losses"] == [4., 4.]
    assert actual["actions"]["probe"]["score"] == 0.


def test_history_set_expansion_cannot_increase_exact_distance():
    # Newly admitted atom shares the competing mean, so positive separation vanishes.
    inputs = ([0.], [[0.], [2.], [0.]], {"probe": [[0.], [10.], [10.]]}, [0, 0, 1])
    scores = [symmetric_scores(*inputs, cutoff=cutoff)["actions"]["probe"]["score"]
              for cutoff in (0., 1., 4., 36.)]
    assert scores == [100., 100., 0., 0.]


def test_label_swap_and_atom_row_permutation_preserve_geometry():
    history = np.array([[0.], [1.], [0.], [2.]])
    future = np.array([[0., 1.], [2., 3.], [7., 1.], [3., 4.]])
    labels = np.array([0, 0, 1, 1])
    base = symmetric_scores([0.], history, {"a": future}, labels, cutoff=4.)
    permutation = np.array([3, 1, 2, 0])
    swapped = symmetric_scores([0.], history[permutation], {"a": future[permutation]},
                               1 - labels[permutation], cutoff=4.)
    assert swapped["actions"]["a"]["score"] == base["actions"]["a"]["score"]
    chosen = swapped["actions"]["a"]["closest_atom_pair"]
    assert set(permutation[chosen]) == set(base["actions"]["a"]["closest_atom_pair"])


def test_three_topologies_use_closest_topology_pair():
    result = symmetric_scores([0.], [[0.], [0.], [0.]],
                              {"a": [[0.], [4.], [5.]]}, [0, 1, 2], cutoff=0.)
    assert result["actions"]["a"]["score"] == 1.
    assert result["actions"]["a"]["closest_topology_pair"] == [1, 2]
    assert len(result["actions"]["a"]["pairwise"]) == 3


def test_moving_an_unsupported_future_atom_cannot_change_score():
    history = np.array([[0.], [5.], [0.]])
    before = np.array([[0.], [1.], [3.]])
    after = np.array([[0.], [-1e6], [3.]])
    scores = symmetric_scores([0.], history, {"before": before, "after": after},
                              [0, 0, 1], cutoff=0.)["actions"]
    assert scores["before"]["score"] == scores["after"]["score"] == 9.


def test_finite_minimum_matches_independent_scalar_loop():
    # Independent enumeration validates multi-dimensional distance and support membership.
    history = np.array([[0., 1.], [2., -1.], [1., 0.], [3., 1.]])
    y = np.array([.5, -.25])
    means = np.array([[0., 3., -1.], [2., 1., 4.], [-1., 1., 3.], [2., 2., 5.]])
    losses = [sum((float(history[i, k]) - float(y[k])) ** 2 for k in range(2)) for i in range(4)]
    allowed = [[i for i in group if losses[i] <= min(losses[j] for j in group) + 4.]
               for group in ([0, 1], [2, 3])]
    expected = min(sum((float(means[i, k]) - float(means[j, k])) ** 2 for k in range(3))
                   for i, j in product(*allowed))
    result = symmetric_scores(y, history, {"a": means}, [0, 0, 1, 1], cutoff=4.)
    assert result["actions"]["a"]["score"] == pytest.approx(expected)


@pytest.mark.parametrize("channel", ["history", "future", "future_unsupported"])
def test_invalid_mean_must_not_remove_an_atom(channel):
    history = np.array([[0.], [9.], [0.]])
    future = np.array([[0.], [4.], [2.]])
    if channel == "history":
        history[1, 0] = np.nan
    else:
        future[1 if channel == "future_unsupported" else 0, 0] = np.inf
    with pytest.raises(ValueError, match="invalid mean"):
        symmetric_scores([0.], history, {"a": future}, [0, 0, 1], cutoff=0.)


def test_missing_atom_or_changed_future_dimension_is_invalid():
    with pytest.raises(ValueError, match="every atom"):
        symmetric_scores([0.], [[0.], [0.]], {"a": [[0.]]}, [0, 1])
    with pytest.raises(ValueError, match="same future"):
        symmetric_scores([0.], [[0.], [0.]], {"a": [[0.], [0.]], "b": [[0., 0.], [0., 0.]]}, [0, 1])


def test_soft_history_qp_matches_closed_form_shared_scalar_nuisances():
    # Two models: mu_0=a+u, mu_1=b-u; historical penalties (a^2+b^2)/sigmaH^2.
    sigma_h, sigma_f, u = 2., 3., 4.
    result = solve_soft_history_qp([2*u/sigma_f], [[1/sigma_f, -1/sigma_f]],
                                  np.eye(2)/sigma_h**2)
    denominator = sigma_f**2 + 2*sigma_h**2
    expected_a = -2*u*sigma_h**2/denominator
    assert result["objective"] == pytest.approx(4*u*u/denominator)
    assert result["nuisance_step"] == pytest.approx([expected_a, -expected_a])
    assert result["projected_objective"] == pytest.approx(result["objective"])
    assert result["normal_equation_residual"] < 1e-12
    # Effective metric is fractional: calling it an orthogonal projector would be wrong.
    metric = result["effective_metric"]
    assert not np.allclose(metric @ metric, metric)


def test_stronger_history_penalty_cannot_lower_soft_separation():
    result = [solve_soft_history_qp([2.], [[1., -1.]], strength*np.eye(2))["objective"]
              for strength in (0., .5, 1., 8.)]
    assert result == pytest.approx([0., .8, 4/3, 3.2])


def test_zero_gradient_still_has_quadratic_separation():
    def objective(u):
        return solve_soft_history_qp([2*u], [[1., -1.]], np.eye(2))["objective"]
    delta = 1e-3
    derivative = (objective(delta) - objective(-delta))/(2*delta)
    curvature = (objective(delta) - 2*objective(0.) + objective(-delta))/delta**2
    assert derivative == pytest.approx(0., abs=1e-14)
    assert curvature == pytest.approx(8/3)


def test_hard_history_scalar_distance_has_closed_form_zero_plateau():
    # |a|,|b|<=.5; this grid includes the exact continuous closest pairs below.
    coordinates = np.array([-.5, 0., .5])
    history = np.r_[coordinates, coordinates][:, None]
    future = {str(u): np.r_[coordinates+u, coordinates-u][:, None]
              for u in (0., .25, .5, 1.5)}
    result = symmetric_scores([0.], history, future, [0, 0, 0, 1, 1, 1], cutoff=.25)
    for u in (0., .25, .5, 1.5):
        assert result["actions"][str(u)]["score"] == pytest.approx(4*max(abs(u)-.5, 0.)**2)


def test_singular_unpenalized_nuisance_exactly_absorbs_future_difference():
    result = solve_soft_history_qp([2., -3.], np.eye(2), np.zeros((2, 2)))
    assert result["objective"] < 1e-25
    assert result["nuisance_step"] == pytest.approx([-2., 3.])
    assert result["effective_metric"] == pytest.approx(np.zeros((2, 2)), abs=1e-14)


def test_independently_free_future_total_power_can_erase_every_action():
    # In this scalar physical surrogate both topologies include total power zero.
    # Free baseline=-u would cancel the command. Histories can still be nonzero.
    for u in (0., .1, .3):
        baseline = -u
        total = baseline + u
        means = [[total, 1. - total, total], [total, 1. - 2.*total, total]]
        result = symmetric_scores([3.], [[2.], [4.]], {"a": means}, [0, 1])
        assert result["actions"]["a"]["score"] == 0.


def test_invalid_soft_history_precision_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_soft_history_qp([1.], [[1.]], [[-1.]])
    with pytest.raises(ValueError, match="symmetric"):
        solve_soft_history_qp([1.], [[1., 1.]], [[1., 1.], [0., 1.]])
