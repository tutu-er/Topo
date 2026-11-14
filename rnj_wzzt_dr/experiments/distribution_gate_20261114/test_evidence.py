"""Independent algebra/integration tests; no AC simulation, fitting or solver."""
import unittest
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from .evidence import DistributionEvidence, gaussian_power_integral
from ...io import plain


class ToyCase:
    H, repeat, N, R = 2, 2, 2, 3
    shape = (H, R)
    labels = (1, 2)
    lower = np.array([[-3., -4., -2.], [-2., -3., -1.]])
    upper = np.array([[4., 3., 2.], [3., 4., 1.]])
    B = np.array([[1., 0., 0.], [0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def array(self, u):
        a = np.asarray(u, float)
        if a.shape != self.shape or not np.isfinite(a).all():
            raise ValueError('Toy action shape/value')
        return a

    def response_effect(self, D, base_kw):
        return D@self.B/base_kw

    def shift_pq_pu(self, u, base_kw):
        return np.repeat(self.array(u)@self.B.T/base_kw, self.repeat, axis=0)


def toy():
    rng = np.random.default_rng(2026111423)
    H = rng.normal(scale=.04, size=(3, 2, 4))
    history = np.array([2., 2.4, 2.8])*1e-6
    native = [dict(H=h.tolist(), history_sse=float(s), topology_id=str(i))
              for i, (h, s) in enumerate(zip(H, history))]
    native.append(dict(H=(H[0]+.01).tolist(), history_sse=2.1e-6, topology_id='0'))
    score = dict(parameters=[dict(index=i, H=h.tolist(), history_sse=float(s))
                             for i, (h, s) in enumerate(zip(H, history))],
        models=[dict(index=i, topology_id=str(i)) for i in range(3)],
        native_bank=native, sigma2=2e-6, cells=20, reference_mse=2e-6, validation_tie=1e-12)
    power = rng.normal(scale=.02, size=(10, 4))
    validation = SimpleNamespace(p=power[:, :2], q=power[:, 2:],
        y=power@H[0].T+rng.normal(scale=.0005, size=(10, 2)),
        voltage=np.full((10, 2), .975), root=np.full(10, 1.005), labels=(1, 2))
    forecast = rng.normal(scale=.03, size=(4, 4))
    return ToyCase(), score, validation, forecast, rng


def joint_log_mgf(B, D, sigma2, sigma_e, power_sd, mean, lam):
    """Independent joint (x,e) Gaussian integration, retaining all signs."""
    n, c = D.shape
    A = np.block([[B, D.T/(2*sigma2)], [D/(2*sigma2), np.zeros((n, n))]])
    F = np.zeros((c+n, c+n))
    F[:c, :c] = np.diag(power_sd)
    F[c:, c:] = np.linalg.cholesky(sigma_e)
    m = np.r_[mean, np.zeros(n)]
    G = np.eye(c+n)+2*lam*F.T@A@F
    sign, logdet = np.linalg.slogdet(G)
    assert sign > 0 and np.linalg.eigvalsh(G).min() > 0
    a = F.T@A@m
    return float(-.5*logdet-lam*m@A@m+2*lam*lam*a@np.linalg.solve(G, a))


class DistributionEvidenceTests(unittest.TestCase):
    def test_full_native_mapping_and_same_topology_exclusion(self):
        case, score, val, forecast, _ = toy()
        evidence = DistributionEvidence(case, score, val, forecast, 200., target_mass=1.)
        self.assertEqual(len(evidence.rows), 8)
        for row in evidence.rows:
            i, j = row['target_native'], row['competitor_native']
            self.assertNotEqual(score['native_bank'][i]['topology_id'], score['native_bank'][j]['topology_id'])
            count = sum(r['target_rep'] == row['target_rep'] for r in evidence.rows)
            self.assertEqual(row['alpha'], .1/count)
        self.assertEqual(sum(r['competitor_native'] == 3 for r in evidence.rows), 2)

    def test_full_SSE_difference_and_shared_root_variance(self):
        case, score, val, forecast, rng = toy()
        evidence = DistributionEvidence(case, score, val, forecast, 200., mode='normal_fixed', target_mass=1.)
        u = rng.uniform(case.lower, case.upper)
        power = forecast+case.shift_pq_pu(u, 200.)
        info = evidence.verify_expansion(u)
        e = rng.normal(scale=.001, size=(4, 2))
        for k, row in enumerate(evidence.rows):
            i, j = row['target_rep'], row['competitor_native']
            Hi, Hj, K = evidence.H[i], evidence.native_H[j], evidence.means[i]
            y = power@K.T+e
            observed = (evidence.native_history[j]-evidence.history[i]
                +np.square(y-power@Hj.T).sum()-np.square(y-power@Hi.T).sum())/(2*evidence.sigma2)
            d = power@(Hi-Hj).T
            expected = info['mu'][k]+row['tie']+np.sum(e*d)/evidence.sigma2
            self.assertAlmostEqual(observed, expected, places=12)
            variance = (evidence.root_variance*np.square(d.sum(axis=1)).sum()
                +evidence.individual_variance*np.square(d).sum())/evidence.sigma2**2
            self.assertAlmostEqual(info['variance'][k], variance, places=12)

    def test_all_modes_expansions_and_valid_box_bounds(self):
        case, score, val, forecast, rng = toy()
        for mode in ('mgf_forecast', 'mgf_fixed', 'normal_fixed'):
            evidence = DistributionEvidence(case, score, val, forecast, 200., mode=mode, target_mass=1.)
            for _ in range(12):
                u = rng.uniform(case.lower, case.upper)
                info = evidence.verify_expansion(u)
                self.assertTrue(np.all(info['variance'] <= evidence.variance_upper+1e-10))
                if mode != 'normal_fixed':
                    self.assertTrue(np.all(info['margins'] <= evidence.gate_upper+1e-10))

    def test_sequential_integral_equals_independent_joint_gaussian(self):
        case, score, val, forecast, rng = toy()
        evidence = DistributionEvidence(case, score, val, forecast, 200., target_mass=1.)
        power = forecast+case.shift_pq_pu(rng.uniform(case.lower, case.upper), 200.)
        info = evidence.direct((power[::case.repeat]-forecast[::case.repeat])[:, [0, 2, 3]]*200.)
        for k, row in enumerate(evidence.rows):
            i, j, lam = row['target_rep'], row['competitor_native'], row['lambda_selected']
            D = evidence.H[i]-evidence.native_H[j]
            expected = lam*(row['tie']-row['h'])
            for t in range(evidence.T):
                expected += joint_log_mgf(evidence.mean_matrices[k], D, evidence.sigma2,
                    evidence.sigma_e, evidence.power_sd[t], power[t], lam)
            self.assertAlmostEqual(info['log_pair_failure_bound'][k], expected, places=11)

    def test_no_power_uncertainty_recovers_exact_normal_mgf(self):
        case, score, val, forecast, rng = toy()
        evidence = DistributionEvidence(case, score, val, forecast, 200., mode='mgf_fixed', target_mass=1.)
        info = evidence.verify_expansion(rng.uniform(case.lower, case.upper))
        for k, row in enumerate(evidence.rows):
            lam = row['lambda_selected']
            self.assertEqual(lam, .5)
            expected = -lam*info['mu'][k]+lam**2*info['variance'][k]/2
            self.assertAlmostEqual(info['log_pair_failure_bound'][k], expected, places=12)

    def test_indefinite_integral_matches_quadrature_without_clipping(self):
        A = np.array([[.25, .3], [.3, -.2]])
        sd, m = np.array([[.2, .3]]), np.array([.7, -.4])
        Q, logdet, mineig = gaussian_power_integral(A, sd)
        self.assertLess(np.linalg.eigvalsh(Q[0]).min(), 0.)
        nodes, weights = hermgauss(30)
        a, b = np.meshgrid(nodes, nodes, indexing='ij')
        x = m+np.sqrt(2)*np.c_[a.ravel(), b.ravel()]*sd
        log_weights = np.log(np.outer(weights, weights).ravel()/np.pi)
        log_quad = logsumexp(log_weights-np.einsum('ti,ij,tj->t', x, A, x))
        self.assertAlmostEqual(log_quad, -m@Q[0]@m-logdet[0], places=12)
        self.assertGreater(mineig, 0.)

    def test_nonintegrable_gaussian_domain_is_rejected_and_lambda_halves(self):
        with self.assertRaises(np.linalg.LinAlgError):
            gaussian_power_integral(np.array([[-1.]]), np.array([[1.]]))
        case, score, val, forecast, _ = toy()
        evidence = DistributionEvidence(case, score, val, forecast, 200., lambda_start=1e5, target_mass=1.)
        self.assertTrue(any(row['lambda_halvings'] > 0 for row in evidence.rows))
        for row in evidence.rows:
            self.assertGreater(row['minimum_G_eigenvalue'], 0.)
            self.assertEqual(row['lambda_selected'], 1e5*2**(-row['lambda_halvings']))

    def test_identical_predictions_do_not_certify_a_tie(self):
        case, score, val, forecast, _ = toy()
        for row in score['parameters']:
            row['H'] = np.zeros((2, 4)).tolist(); row['history_sse'] = 0.
        for row in score['native_bank']:
            row['H'] = np.zeros((2, 4)).tolist(); row['history_sse'] = 0.
        for mode in ('mgf_forecast', 'mgf_fixed', 'normal_fixed'):
            evidence = DistributionEvidence(case, score, val, forecast, 200., mode=mode, target_mass=1.)
            info = evidence.direct(np.zeros(case.shape))
            self.assertFalse(info['feasible'])
            np.testing.assert_array_equal(info['pair_failure'], 1.)

    def test_missing_voltages_and_unaligned_native_bank_raise(self):
        case, score, val, forecast, _ = toy()
        bad = deepcopy(val); bad.voltage = None
        with self.assertRaises(ValueError):
            DistributionEvidence(case, score, bad, forecast, 200.)
        bad_score = deepcopy(score); bad_score['parameters'][0]['index'] = 3
        with self.assertRaises(ValueError):
            DistributionEvidence(case, bad_score, val, forecast, 200.)

    def test_extreme_and_deterministic_normal_tails_are_JSON_safe(self):
        case, score, val, forecast, _ = toy()
        for i in range(1, 3):
            score['parameters'][i]['history_sse'] = 1e4
            score['native_bank'][i]['history_sse'] = 1e4
        evidence = DistributionEvidence(case, score, val, forecast, 200., mode='normal_fixed')
        info = evidence.direct(np.zeros(case.shape))
        self.assertGreater(len(info['pair_failure_numerical_underflow']), 0)
        self.assertTrue(all(v is not None and np.isfinite(v) for v in info['log_pair_failure_bound']))
        json.dumps(info, default=plain, allow_nan=False)
        for row in score['parameters']+score['native_bank']:
            row['H'] = np.zeros((2, 4)).tolist()
        evidence = DistributionEvidence(case, score, val, forecast, 200., mode='normal_fixed')
        info = evidence.direct(np.zeros(case.shape))
        self.assertTrue(all(v is None for v in info['log_pair_failure_bound']))
        self.assertTrue(all(v['reason'] == 'zero_variance_deterministic_success'
                            for v in info['log_pair_failure_nonfinite']))
        json.dumps(info, default=plain, allow_nan=False)

    def test_mean_correction_coupling_preserves_zero_D_column_action(self):
        # A power channel with D-column zero still enters E'D and the mean.
        case, score, val, forecast, _ = toy()
        base = np.array([[.3, .1, .2, .4], [.1, .4, .2, .3]])
        alternate = base.copy(); alternate[:, 0] += [.2, -.1]
        for i, H in enumerate((base, alternate, base+.05)):
            score['parameters'][i]['H'] = H.tolist()
            score['native_bank'][i]['H'] = H.tolist()
        score['native_bank'][3]['H'] = (base+.01).tolist()
        val.y = 1.3*np.c_[val.p, val.q]@base.T
        evidence = DistributionEvidence(case, score, val, forecast, 200., mode='mgf_fixed', target_mass=1.)
        k = next(k for k, row in enumerate(evidence.rows) if row['target_rep'] == 0 and row['competitor_native'] == 1)
        self.assertTrue(np.all((base-alternate)[:, 2] == 0))
        self.assertGreater(np.max(np.abs(evidence.mu_q[k, :, 0, 1])), 1e-8)
        evidence.verify_expansion(np.array([[1., 2., -1.], [2., -1., 1.]]))


if __name__ == '__main__':
    unittest.main()
