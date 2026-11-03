import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from kernel import *
class TestJoint(unittest.TestCase):
    def test_plan_hash_signed_zero(self):
        from run import plan_hash
        self.assertEqual(plan_hash('same',np.array([0.,-0.,1.])),plan_hash('same',np.array([-0.,0.,1.])))
        self.assertNotEqual(plan_hash('first',np.zeros(3)),plan_hash('second',np.zeros(3)))
        self.assertNotEqual(plan_hash('same',np.zeros(3)),plan_hash('same',np.array([0.,0.,1e-100])))
    def test_abs_mean(self):
        for mu,sd in [(0,1),(2,.4),(-.8,2),(4,0)]:
            val=abs(mu) if sd==0 else quad(lambda z:abs(mu+sd*z)*np.exp(-z*z/2)/np.sqrt(2*np.pi),-12,12,points=[-mu/sd])[0]
            self.assertAlmostEqual(float(abs_mean(mu,sd)),val,places=7)
    def test_hermite_zero_mean_exact(self):
        for rho in [-.8,0,.4,.99,1]:
            cov=np.array([[1.,rho],[rho,1.]])
            q=hermite_moments(np.zeros(2),cov,np.array([1.,-1.03]),(16,64))
            c=2/np.pi*(np.sqrt(1-rho*rho)+rho*np.arcsin(rho)-1)
            exact=(1+1.03**2)*(1-2/np.pi)-2*1.03*c
            for r in q['rows']:
                self.assertLessEqual(r['variance_lower'],exact+1e-12)
                self.assertGreaterEqual(r['variance_upper'],exact-1e-12)
    def test_hermite_noncentral(self):
        mu=np.array([.7,-.3]);rho=.6;w=np.array([1.,-.7])
        q=hermite_moments(mu,np.array([[1,rho],[rho,1]]),w,(32,128))
        prod=quad(lambda z:abs(mu[0]+z)*float(abs_mean(mu[1]+rho*z,np.sqrt(1-rho*rho)))*np.exp(-z*z/2)/np.sqrt(2*np.pi),-12,12,points=[-.7],epsabs=1e-10)[0]
        m=abs_mean(mu,np.ones(2));var=1+mu**2-m**2
        exact=float((w*w)@var+2*w.prod()*(prod-m.prod()))
        self.assertLessEqual(q['rows'][-1]['variance_lower'],exact+1e-9)
        self.assertGreaterEqual(q['rows'][-1]['variance_upper'],exact-1e-9)
    def test_covariance_scaling(self):
        a=hermite_moments(np.array([.1,.3]),np.array([[1,.2],[.2,2]]),np.array([.5,-.8]),(32,))
        b=hermite_moments(np.array([.1,.3])*10,np.array([[1,.2],[.2,2]])*100,np.array([.5,-.8]),(32,))
        self.assertAlmostEqual(b['rows'][0]['variance_lower'],a['rows'][0]['variance_lower']*100)
    def test_spherical_rank1(self):
        a=spherical_probability(np.array([.3]),np.array([[1.]]),np.array([1.]),.8,4)
        exact=ndtr(-.8-.3)+1-ndtr(.8-.3)
        self.assertAlmostEqual(a['probability_estimate'],exact,places=12)
    def test_ray_disjoint_intervals(self):
        mu=np.array([1.,-2.,.2]);b=np.array([-2.,1.,.5]);w=np.array([1.,-.4,.3]);h=.3
        p=ray_probability(mu,b,w,h,2)
        grid=np.linspace(0,12,240001);D=np.abs(mu[None]+grid[:,None]*b)@w
        oracle=np.trapezoid((D>h)*grid*np.exp(-grid*grid/2),grid)
        self.assertLess(abs(p-oracle),5e-5)
    def test_bvn_special(self):
        for rho in [-1,-.7,0,.8,1]:
            expected=.25+np.arcsin(rho)/(2*np.pi)
            self.assertAlmostEqual(float(bvn_cdf(0.,0.,rho)),expected,places=12)
    def test_joint_mgf_common_noise(self):
        for lam in [.01,.5,1]:
            q=paired_logmgf(np.zeros(2),np.ones((2,2)),np.array([0]),np.array([1]),lam)
            exact=np.log(2)+.5*(.03*lam)**2+np.log(ndtr(.03*lam))
            self.assertAlmostEqual(float(q[0]),exact,places=10)
    def test_joint_mgf_quadrature(self):
        mu=np.array([.2,-.4]);rho=.3;lam=.2
        value=np.exp(paired_logmgf(mu,np.array([[1,rho],[rho,1]]),np.array([0]),np.array([1]),lam)[0])
        def integrand(z):
            m=mu[1]+rho*z;s=np.sqrt(1-rho*rho);t=1.03*lam
            f=np.exp(.5*t*t*s*s)*(np.exp(t*m)*ndtr(m/s+t*s)+np.exp(-t*m)*ndtr(-m/s+t*s))
            return np.exp(-lam*abs(mu[0]+z))*f*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        exact=quad(integrand,-12,12,points=[-.2],epsabs=1e-12)[0]
        self.assertAlmostEqual(value,exact,places=9)
    def test_milp_and_enumeration(self):
        for h in [.1,.5,3.]:
            mu=np.array([-.3,1.,-.4]);delta=np.array([1.,-.4,1.5]);w=np.array([1.,-.7,.2])
            q=nominal_milp(mu,delta,w,h)
            if q['enum_alpha'] is None:self.assertIsNone(q['alpha'])
            else:self.assertAlmostEqual(q['alpha'],q['enum_alpha'],places=6)
    def test_flat_deterministic(self):
        self.assertEqual(ray_probability(np.array([1.]),np.zeros(1),np.ones(1),.5,2),1)
        self.assertEqual(ray_probability(np.array([1.]),np.zeros(1),np.ones(1),1.,2),0)
    def test_cantelli_retains_tau(self):
        q=hermite_moments(np.zeros(2),np.ones((2,2)),np.array([1.,-1.03]),(128,))
        exact=.03**2*(1-2/np.pi)
        self.assertLess(q['rows'][-1]['variance_lower'],exact+1e-12)
        self.assertLess(q['rows'][-1]['variance_lower'],q['marginal_independence_variance']/100)
if __name__=='__main__':unittest.main()
