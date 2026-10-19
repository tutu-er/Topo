"""Analytic formula audits only; the next bounded-QP policy is unimplemented.

These finite arithmetic grids audit identities and inequality directions.
They do not enumerate, optimize, or select EV actions in a research policy.
"""
from itertools import product
import unittest


def scalar_q(u):
    z=max(-1.,min(1.,-u))
    return .5*(u+z)**2


def shared_q(c,limit=.5):
    z=max(-limit,min(limit,-sum(c)/len(c)))
    return .5*sum((v+z)**2 for v in c)


def half_square_chord(x,grid):
    for left,right in zip(grid,grid[1:]):
        if left<=x<=right:
            w=(x-left)/(right-left)
            return (1-w)*.5*left**2+w*.5*right**2
    raise ValueError('outside stated factor range')


class NextModelMathTests(unittest.TestCase):
    def test_bounded_scalar_support_lower_and_feasible_upper(self):
        # q(u)=min_{-1<=z<=1} (u+z)^2/2. Free LS instead has q=0.
        self.assertEqual(scalar_q(2.),.5)
        for u in (-3.,-1.25,-1.,0.,.25,1.,2.):
            q=scalar_q(u)
            for v in (-3.,-.7,0.,.4,2.):
                ell=v*u-.5*v*v-abs(v)  # exact min_{z in [-1,1]} v*z
                self.assertLessEqual(ell,q+1e-12)
            for z in (-1.,-.2,1.):self.assertLessEqual(q,.5*(u+z)**2+1e-12)
            z=max(-1.,min(1.,-u));v=u+z
            self.assertAlmostEqual(v*u-.5*v*v-abs(v),q)

    def test_two_times_one_shared_change_and_convexity(self):
        # z is one shared line-coordinate change, not independent per time.
        self.assertEqual(shared_q((1.,-1.)),1.)
        independent=.5*((1.-.5)**2+(-1.+.5)**2)
        self.assertEqual(independent,.25)
        points=list(product((-2.,0.,2.),repeat=2))
        for u,w in product(points,repeat=2):
            middle=tuple((a+b)/2 for a,b in zip(u,w))
            self.assertLessEqual(shared_q(middle),.5*(shared_q(u)+shared_q(w))+1e-12)

    def test_active_ratio_boundary_support_exact(self):
        # R=1, X/R in [.1,.4], X_hat=.25: z=delta_X in [-.15,.15].
        c=-.35;z=.15;v=c+z
        primal=.5*(c+z)**2
        support_min=min(v*(-.15),v*.15)
        self.assertAlmostEqual(primal,.02)
        self.assertAlmostEqual(v*c-.5*v*v+support_min,primal)

    def test_shared_quadratic_and_chord_directions(self):
        # Residual c+u+z*(1,1): all fixed z explanations share Q=||u||^2/2.
        grid=(-2.,0.,2.);c=(.2,-.1);maximum_chord_error=1.
        for u in product((-2.,-.5,0.,.75,2.),repeat=2):
            actual=[];upper=[]
            for z in (-.5,.5):
                cr=tuple(v+z for v in c)
                exact=.5*sum((v+w)**2 for v,w in zip(u,cr))
                shared=.5*sum(v*v for v in u)+sum(v*w for v,w in zip(u,cr))+.5*sum(v*v for v in cr)
                chord=sum(half_square_chord(v,grid) for v in u)+sum(v*w for v,w in zip(u,cr))+.5*sum(v*v for v in cr)
                self.assertAlmostEqual(shared,exact)
                self.assertLessEqual(exact,chord+1e-12)
                self.assertLessEqual(chord-exact,maximum_chord_error+1e-12)
                actual.append(exact);upper.append(chord)
            bounded=shared_q(tuple(v+w for v,w in zip(u,c)))
            self.assertLessEqual(bounded,min(actual)+1e-12)
            self.assertLessEqual(min(actual),min(upper)+1e-12)

    def test_own_interval_subtraction_and_signed_gap(self):
        lab,uab,laa,uaa=1.5,2.5,.8,1.2
        exact_ab,exact_aa=1.5,1.2
        gap=exact_ab-exact_aa
        self.assertLessEqual(lab-uaa,gap)
        self.assertGreaterEqual(uab-laa,gap)
        self.assertGreater(lab-laa,gap)  # subtracting the lower own bound is wrong for a lower gap.
        gap=.4-1.
        self.assertLessEqual(.3-1.2,gap)
        self.assertGreaterEqual(.5-.8,gap)
        self.assertLess(gap,0.)  # eta must permit negative signed values.


if __name__=='__main__':unittest.main()
