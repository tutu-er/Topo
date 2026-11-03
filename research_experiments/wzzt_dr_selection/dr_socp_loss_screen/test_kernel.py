import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,ndtri
from scipy.optimize import minimize
from kernel import variance_of_sum,residual_variances,ar_cov,expected_hinge,correction,optimal_correction_risks,phi


class ScreenTests(unittest.TestCase):
    def test_variance_dense_independent_assembly(self):
        nodes=np.array([0,0,1]);w=[.15,.25,.2,.4];T=4;ar=ar_cov(T,.8)
        c=np.arange(12).reshape(3,4)/7.-.6
        coords=[(r,t) for r in range(3) for t in range(T)]
        K=np.array([[w[0]+w[1]*(nodes[r]==nodes[k])+(w[2]+w[3]*(r==k))*.8**abs(t-s)
                     for k,s in coords] for r,t in coords])
        self.assertAlmostEqual(variance_of_sum(c,nodes,w,ar),float(c.ravel()@K@c.ravel()),places=12)

    def test_marginal_variance_dense(self):
        nodes=np.array([0,0,1]);w=[.15,.25,.2,.4]
        e=np.array([[1.,-2.,.4],[2.,1.,-3.]])
        u=np.array([[1.,-.4],[.2,.3],[.7,-1.]])
        bg=np.array([[.1,.3],[.3,.2],[.7,-.2],[-.4,.3]])
        out=residual_variances(e,u,bg,nodes,w)
        K=np.array([[w[0]+w[2]+w[1]*(a==b)+w[3]*(i==j) for j,b in enumerate(nodes)] for i,a in enumerate(nodes)])
        for t in range(2):
            for n in range(2):
                c=e[n]*u[:,t];truth=c@K@c+bg[2*t+n]@bg[2*t+n]
                self.assertAlmostEqual(out[t,n],truth,places=13)

    def test_hinge_integral(self):
        for m,s in [(-2.,.7),(.1,2.),(3.,.4),(0.,1.)]:
            expected=quad(lambda z:max(0.,-(m+s*z))*phi(z),-np.inf,-m/s,epsabs=1e-11)[0]
            self.assertAlmostEqual(float(expected_hinge(np.array([m]),np.array([s]))[0]),expected,places=10)

    def test_pointwise_lower_bound_changes_signs(self):
        for c in [-3.,-.1,0.,1.,7.]:
            for f in [-4.,-.01,0.,.1,2.]:
                for sc in [-1.,1.]:
                    for sf in [-1.,1.]:
                        D=abs(c)-1.03*abs(f);G=sc*c-1.03*sf*f
                        self.assertGreaterEqual(D+1e-12,G-2.06*max(0.,-sf*f))

    def test_risk_bound_is_exact_marginal(self):
        m=np.array([-1.,.4,4.]);s=np.array([.2,2.,.7]);risk=np.array([.001,.005,.004])
        v=correction(m,s,risk)
        self.assertTrue(np.all(ndtr(-(v+m)/s)<=risk+1e-13))

    def test_allocation_against_independent_slsqp(self):
        m=np.array([.5,2.,-.2]);s=np.array([1.,.7,.3]);budget=.01
        a,v=optimal_correction_risks(m,s,budget)
        # Scale risk shares to order one; unscaled finite-difference SLSQP
        # differed by 1.8e-7 despite returning success in the first preflight.
        def gradient(q):
            z=-ndtri(budget*q)
            return np.where(s*z-m>0.,-budget*s/phi(z),0.)
        fit=minimize(lambda q:correction(m,s,budget*q).sum(),np.full(3,1/3),
            jac=gradient,method='SLSQP',bounds=[(1e-8,1.)]*3,
            constraints=[dict(type='eq',fun=lambda q:q.sum()-1.,jac=lambda q:np.ones(3))],options=dict(ftol=1e-13,maxiter=300))
        self.assertTrue(fit.success)
        self.assertAlmostEqual(v.sum(),fit.fun,places=7)
        self.assertLessEqual(a.sum(),budget+1e-14)

    def test_identical_residual_conservatism(self):
        # Same common noise nearly cancels in actual D; one-sided bound drops
        # the positive C correction and is much more conservative.
        actual_mean=-.03*np.sqrt(2/np.pi)
        gauss_part=0.;hc=float(expected_hinge(np.array([0.]),np.array([1.]))[0])
        self.assertAlmostEqual(gauss_part+2*hc-2.06*hc,actual_mean,places=14)
        v=correction(np.array([0.]),np.array([1.]),np.array([.01]))[0]
        self.assertGreater(2.06*v,4.)

    def test_deterministic_and_zero_cases(self):
        a,v=optimal_correction_risks(np.array([1.,-2.]),np.zeros(2),.01)
        np.testing.assert_array_equal(a,[0.,0.]);np.testing.assert_array_equal(v,[0.,2.])
        np.testing.assert_array_equal(expected_hinge(np.array([1.,-2.]),np.zeros(2)),[0.,2.])


if __name__=='__main__':unittest.main()
