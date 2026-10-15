"""Independent derivative checks for the source-profile diagnostic."""
import numpy as np
import pytest

from .ev_fixed_gradient import nearest_summary, profile_pair


class AffineResponse:
    def __init__(self, control, source, bias):
        self.control = np.asarray(control, float)
        self.source = np.asarray(source, float)
        self.bias = np.asarray(bias, float)

    def evaluate(self, p, q, sp, sq, root, *, derivatives=True):
        T, n = p.shape
        assert n == 1
        jac = np.zeros((T, 3, 4))
        jac[:, :, 0] = self.control
        jac[:, :, -2:] = self.source
        values = self.bias + p*self.control + np.column_stack((sp, sq))@self.source.T
        return dict(values=values, jac=jac if derivatives else None)


@pytest.mark.parametrize('active', [False, True])
def test_profile_gradient_reoptimizes_source_at_interior_and_box_boundary(active):
    G = np.array([[.01, .02], [1., 0.], [0., 1.]])
    a = AffineResponse([-.01, 1., 0.], G, [1., 0., 0.])
    b = AffineResponse([-.025, 1., .002], G,
                       [1.001, -1. if active else .01, 0.])
    p = np.array([[.2], [.3]])
    q = np.zeros_like(p)
    s = np.zeros((2, 2))
    u = np.array([[2.], [-1.]])
    sigma = np.broadcast_to([.001, .002, .002], (2, 3))

    def run(power):
        return profile_pair(a, b, p, q, s, s, np.ones(2), power, sigma,
                            multistart=2, tol=1e-13)

    out = run(u)
    assert out['success']
    assert (out['active_mask'] != 0).any() == active
    assert out['source_kkt_violation'] < 2e-6
    eps = .0002
    for t in range(2):
        plus = u.copy(); minus = u.copy()
        plus[t, 0] += eps; minus[t, 0] -= eps
        high, low = run(plus), run(minus)
        assert high['success'] and low['success']
        fd = (high['half_SSE']-low['half_SSE'])/(2*eps)
        assert out['gradient_per_kw'][t, 0] == pytest.approx(fd, rel=2e-5, abs=2e-7)


def test_required_pair_failure_kept_and_nearest_tie_retained():
    pairs = [dict(a='a', b='b', success=True, half_SSE=-2.),
             dict(a='a', b='c', success=True, half_SSE=-2.),
             dict(a='b', b='a', success=True, half_SSE=3.)]
    out = nearest_summary(pairs)
    assert out['mean_nearest_half_SSE'] == .5
    assert out['nearest'][0]['active'] == ['b', 'c']
    pairs[0]['success'] = False
    assert not nearest_summary(pairs)['success']


def test_invalid_input_rejected_before_fitting():
    obj = AffineResponse([0., 1., 0.], np.zeros((3, 2)), [1., 0., 0.])
    with pytest.raises(ValueError, match='outside'):
        profile_pair(obj, obj, np.zeros((1, 1)), np.zeros((1, 1)),
            np.ones((1, 2)), np.zeros((1, 2)), np.ones(1), np.zeros((1, 1)),
            np.ones((1, 3)))


def test_probe_training_view_excludes_future_truth_and_validation():
    from copy import deepcopy
    from .ev_fixed_gradient_probe import training_view
    record = dict(training={'p':[[.1]]}, training_meter={'known':1},
        future_meter={'public_sigma':.01}, candidates=[{'name':'a'}],
        training_result={'fits':[{'candidate':'a'}], 'offline_extra':'do not pass'},
        config=dict(horizon=6, forecast_gap=4, profile={'multistart':2},
                    source={'source_p_bounds':[-.5, .5]}, offline_truth='do not pass'),
        truth={'r':[.02]}, validation={'v':[1.]}, background={'p':[.5]},
        methods=[{'M':123}], forecast_error={'RMSE':.1}, family_index=2)
    altered = deepcopy(record)
    for key in ('truth', 'validation', 'background', 'methods', 'forecast_error', 'family_index'):
        altered[key] = 'changed heldout information'
    assert training_view(record) == training_view(altered)
    assert set(training_view(record)['training_result']) == {'fits'}
    assert 'offline_truth' not in training_view(record)['config']


def test_reference_response_failure_is_retained():
    class Broken:
        def evaluate(self, *args, **kwargs):
            raise ValueError('declared nonpositive approximate voltage')
    out = profile_pair(Broken(), Broken(), np.zeros((1, 1)), np.zeros((1, 1)),
        np.zeros((1, 2)), np.zeros((1, 2)), np.ones(1), np.zeros((1, 1)), np.ones((1, 3)))
    assert not out['success']
    assert out['attempts'] == []
    assert 'reference_response' in out['failure']


def test_probe_failure_keeps_expected_denominator():
    from .ev_fixed_gradient_probe import summarize
    out = summarize([dict(status='driver_failure', task_id=7, failure='fixture')])
    assert out['histories'] == 1 and out['histories_complete'] == 0
    for method in out['methods'].values():
        assert method['profile_expected'] == 48
        assert method['profile_total'] == method['profile_success'] == 0
        assert method['kernel_median_seconds'] is None
