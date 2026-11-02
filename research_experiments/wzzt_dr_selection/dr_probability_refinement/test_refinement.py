import sys,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).parent))
import refine
from tail import shortfall,TailStart,density
from conditional_integral import conditional_probability


class IntegralTests(unittest.TestCase):
    def test_integral_and_derivatives(self):
        rng=np.random.default_rng(2026110801)
        for n in [1,2,7,15]:
            for _ in range(6):
                a=rng.normal(size=n);b=rng.normal(size=n);w=rng.normal(size=n)/n;t=float(rng.normal())
                if n>1:b[0]=0
                out=shortfall(a,b,w,t)
                knots=list((-a[b!=0]/b[b!=0]))
                event=conditional_probability(a,b,-w,-t,coefficient_gradients=False)
                points=sorted(set([-12.,12.]+[float(x) for x in knots+list(event.threshold_roots) if -12<x<12]))
                expected=sum(quad(lambda z:max(0.,t-np.dot(w,np.abs(a+b*z)))*density(z),l,r,epsabs=1e-11)[0] for l,r in zip(points[:-1],points[1:]))
                self.assertAlmostEqual(out['value'],expected,places=8)
                direction=rng.normal(size=n);eps=1e-6
                fd=(shortfall(a+eps*direction,b,w,t)['value']-shortfall(a-eps*direction,b,w,t)['value'])/(2*eps)
                np.testing.assert_allclose(fd,np.dot(out['gradient'],direction),rtol=2e-5,atol=1e-7)
                dt=(shortfall(a,b,w,t+eps)['value']-shortfall(a,b,w,t-eps)['value'])/(2*eps)
                self.assertAlmostEqual(dt,out['dt'],places=6)

    def test_zero_atom_strict_probability(self):
        a=np.array([0.]);b=np.array([0.]);w=np.array([1.])
        self.assertEqual(shortfall(a,b,w,0)['value'],0)
        self.assertEqual(conditional_probability(a,b,w,0,coefficient_gradients=False).probability,0)
        self.assertEqual(shortfall(a,b,w,2)['value'],2)

    def test_tail_reference(self):
        # A constant signed-L1 difference, positive/negative weights retained.
        out=shortfall(np.array([3.,2.]),np.zeros(2),np.array([1.,-1.03]),2.)
        self.assertAlmostEqual(out['value'],1.06)

    def test_device_gradient_on_real_frozen_problem(self):
        root=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a/wzzt_heterogeneous_pilot_stage/run_v2')
        _,p,cfg,alloc=refine.config(root)
        c=p['cases'][0];d=refine.Devices(alloc[c['case']],cfg,refine.profiles)
        packet=refine.load(root/c['name']/'problem.pkl')
        prob=refine.Probability(packet,d,3,2026110802,2026110803)
        tail=TailStart(prob);rng=np.random.default_rng(2026110804)
        x=rng.uniform(-.3,.3,size=(d.R,24));x[d.kind==1]=-np.abs(x[d.kind==1])
        v=np.r_[x.ravel(),.2];direction=rng.normal(size=len(v));direction/=np.linalg.norm(direction)
        q=tail.evaluate(v);eps=1e-5
        fd=(tail.evaluate(v+eps*direction)['value']-tail.evaluate(v-eps*direction)['value'])/(2*eps)
        np.testing.assert_allclose(fd,np.dot(q['gradient'],direction),rtol=5e-4,atol=3e-6)


if __name__=='__main__':unittest.main()
