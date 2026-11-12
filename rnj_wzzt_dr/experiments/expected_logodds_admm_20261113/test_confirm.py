"""Full posterior and paired expected-evidence calculations."""
import unittest
import numpy as np
from scipy.special import expit, logsumexp

from .confirm import log_odds_samples


class Tests(unittest.TestCase):
    def test_log_odds_equal_direct_frozen_likelihood(self):
        H=np.array([[[1.,0.]],[[2.,0.]]]);history=np.log([.6,.4]);power=np.array([[[1.,0.]],[[2.,0.]]])
        error=np.array([[[.2]],[[-.3]]]);y=power@H[0].T+error
        odds=log_odds_samples(H,history,0,power,error,.7)
        loss=np.array([np.sum((y-power@h.T)**2,axis=(1,2)) for h in H]).T
        actual=history[None]-loss/(2*.7);actual-=logsumexp(actual,axis=1)[:,None]
        np.testing.assert_allclose(-logsumexp(-odds,axis=1),actual[:,0],atol=1e-14)

    def test_paired_mean_increment_is_exact_for_balanced_error(self):
        H=np.array([[[1.,0.]],[[2.,0.]]]);history=np.log([.5,.5])
        power=np.array([[[1.,0.]],[[1.,0.]]]);error=np.array([[[-1.]],[[1.]]])
        old=log_odds_samples(H,history,0,power,error,1.)
        new=log_odds_samples(H,history,0,power+np.array([1.,0.]),error,1.)
        self.assertAlmostEqual(float(np.mean(new[:,1]-old[:,1])),1.5,places=14)


if __name__=="__main__":
    unittest.main()
