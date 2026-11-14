"""Independent SSE identities and boundary cases; no production AC or fitting."""
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm

from .solver import MeanRecovery, buffered_direct_audit
from ..reactive_flex_20261114.domain import PQIncrementCase
from ..reactive_flex_20261114.test_domain import fixture


class ToyCase:
    H = N = R = repeat = 1
    shape = (1, 1)
    lower = np.array([[-4.]])
    upper = np.array([[4.]])
    A_eq = np.empty((0, 1)); b_eq = np.empty(0)
    A_ub = np.empty((0, 1)); b_ub = np.empty(0)

    def array(self, u):
        a = np.asarray(u, float)
        if a.shape != self.shape or not np.isfinite(a).all(): raise ValueError('invalid toy action')
        return a

    def shift_pq_pu(self, u, base):
        return np.c_[self.array(u)[:, 0]/base, np.zeros(1)]

    def response_effect(self, D, base):
        return D[:, :1]/base

    def audit(self, u):
        u = self.array(u)
        return dict(passed=bool(np.all(u >= self.lower-1e-7) and np.all(u <= self.upper+1e-7)))

    def fee(self, u):
        return float(np.abs(self.array(u)).sum())

    def attach(self, model):
        u = model.addMVar(self.shape, lb=self.lower, ub=self.upper, name='u')
        absolute = model.addMVar(self.shape, lb=0., name='absolute')
        model.addConstr(absolute >= u); model.addConstr(absolute >= -u)
        return u, absolute.sum()


def toy():
    H = np.array([[[1., 0.]], [[-1., 0.]]])
    score = dict(parameters=[dict(H=h.tolist(), history_sse=0.) for h in H],
        models=[dict(topology_id='a'), dict(topology_id='b')],
        sigma2=1., cells=1, reference_mse=1., validation_tie=1e-12)
    validation = SimpleNamespace(p=np.ones((1, 1)), q=np.zeros((1, 1)), y=np.zeros((1, 1)))
    return ToyCase(), score, validation, np.zeros((1, 2)), H


class MeanGateTests(unittest.TestCase):
    def random_evidence(self):
        context, labels, forecast, resources = fixture()
        case = PQIncrementCase(context, labels, forecast, 200., q_resources=resources)
        rng = np.random.default_rng(11417)
        H = rng.normal(size=(3, 3, 6)); means = H+rng.normal(scale=.3, size=H.shape)
        history = np.array([.2, .5, .8])
        score = dict(parameters=[dict(H=h.tolist(), history_sse=float(v)) for h,v in zip(H,history)],
            models=[dict(topology_id=str(i)) for i in range(3)], sigma2=.7,
            cells=288, reference_mse=.7, validation_tie=1e-12)
        validation = SimpleNamespace(p=forecast[:, :3], q=forecast[:, 3:], y=rng.normal(size=(96, 3)))
        return rng, MeanRecovery(case,score,validation,forecast,200.,means,target_mass=1.)

    def test_expansion_matches_full_SSE_for_arbitrary_means_and_PQ_actions(self):
        rng, evidence = self.random_evidence()
        for _ in range(24):
            u = rng.uniform(evidence.case.lower, evidence.case.upper)
            info = evidence.verify_expansion(u)
            P,Q = evidence.case.aggregate(evidence.case.delivery(u))
            power = evidence.forecast+np.c_[P,Q]/200.
            for k,row in enumerate(evidence.mean_rows):
                i,j=row['target'],row['competitor']
                y = power@evidence.means[i].T
                independent = ((evidence.history[j]+np.square(y-power@evidence.H[j].T).sum())
                    -(evidence.history[i]+np.square(y-power@evidence.H[i].T).sum()))/(2*evidence.sigma2)
                self.assertAlmostEqual(info['mean_mu'][k],independent,places=8)

    def test_random_residual_changes_score_by_exact_inner_product(self):
        rng,evidence=self.random_evidence()
        u=rng.uniform(evidence.case.lower,evidence.case.upper)
        power=evidence.forecast+evidence.case.shift_pq_pu(u,200.)
        residual=rng.normal(size=(96,3))
        info=evidence.direct(u)
        for k,row in enumerate(evidence.mean_rows):
            i,j=row['target'],row['competitor']
            y=power@evidence.means[i].T+residual
            observed=(evidence.history[j]-evidence.history[i]
                +np.square(y-power@evidence.H[j].T).sum()-np.square(y-power@evidence.H[i].T).sum())/(2*evidence.sigma2)
            d=power@(evidence.H[i]-evidence.H[j]).T
            self.assertAlmostEqual(observed-info['mean_mu'][k],float((residual*d).sum()/evidence.sigma2),places=8)
            variance=evidence.sigma2*np.square(d/evidence.sigma2).sum()
            self.assertAlmostEqual(variance,2*info['mean_pair_J'][k],places=8)

    def test_nominal_means_recover_original_conditional_formula(self):
        case,score,val,forecast,H=toy()
        evidence=MeanRecovery(case,score,val,forecast,1.,H)
        for x in (0.,.1,1.,2.,-3.):
            info=evidence.verify_expansion(np.array([[x]]))
            np.testing.assert_allclose(info['mean_mu'],[2*x*x,2*x*x],atol=1e-12)
            self.assertEqual(info['original_feasible'],info['mean_feasible'])

    def test_identical_calibrated_means_make_bidirectional_confirmation_impossible(self):
        case,score,val,forecast,H=toy()
        evidence=MeanRecovery(case,score,val,forecast,1.,np.zeros_like(H))
        info=evidence.verify_expansion(np.array([[3.]]))
        self.assertTrue(info['original_feasible'])
        self.assertFalse(info['mean_feasible'])
        np.testing.assert_allclose(info['mean_mu'],0.,atol=1e-12)
        self.assertTrue(np.all(info['mean_pair_failure']>.5))
        self.assertFalse(buffered_direct_audit(case,evidence,np.array([[3.]]))['passed_augmented_acceptance'])

    def test_zero_distance_tie_is_not_false_success(self):
        case,score,val,forecast,H=toy()
        score['parameters'][1]['H']=H[0].tolist()
        evidence=MeanRecovery(case,score,val,forecast,1.,H)
        info=evidence.direct(np.array([[3.]]))
        np.testing.assert_array_equal(info['mean_pair_failure'],1.)
        self.assertFalse(info['feasible'])

    def test_auxiliary_form_has_correct_minus_sign_and_allows_no_smaller_s(self):
        z=norm.isf(.1)
        for J in (.01,.5,3.,8.):
            boundary=1e-7+z*np.sqrt(2*J)
            # For mu just below the true boundary, any admissible s fails.
            for s in np.sqrt(J)+np.array([0.,.2,1.]):
                self.assertLess(boundary-1e-4-1e-7-z*np.sqrt(2)*s,0.)
            self.assertAlmostEqual(boundary-1e-7-z*np.sqrt(2)*np.sqrt(J),0.,places=12)

    def test_resource_box_bound_remains_valid(self):
        rng,evidence=self.random_evidence()
        for _ in range(32):
            u=rng.uniform(evidence.case.lower,evidence.case.upper)
            self.assertTrue(np.all(evidence.direct(u)['J']<=evidence.upper+1e-9))

    def test_rejects_unaligned_or_nonfinite_means(self):
        case,score,val,forecast,H=toy()
        for bad in (H[:1],np.full_like(H,np.nan)):
            with self.assertRaises(ValueError):MeanRecovery(case,score,val,forecast,1.,bad)


if __name__ == '__main__':
    unittest.main()
