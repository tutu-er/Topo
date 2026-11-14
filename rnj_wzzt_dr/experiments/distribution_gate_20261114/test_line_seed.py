"""Analytic one-dimensional counterexamples for optional initial-point search."""
import unittest
import numpy as np

from .line_seed import search_line, search_lines, line_coefficients
from .solver import BUFFER


class Case:
    def __init__(self, resources=1):
        self.H, self.R, self.shape = 1, resources, (1, resources)
        self.lower, self.upper = -np.ones(self.shape), np.ones(self.shape)
        self.fee_weights = np.arange(1, resources+1)[None, :]

    def array(self, u):
        u=np.asarray(u, float)
        if u.shape != self.shape or not np.isfinite(u).all(): raise ValueError('action')
        return u

    def audit(self, u):
        u=self.array(u)
        return dict(passed=bool(np.all(u>=self.lower-1e-12) and np.all(u<=self.upper+1e-12)))

    def fee(self, u):
        return float(np.sum(self.fee_weights*np.abs(self.array(u))))


class PolynomialEvidence:
    def __init__(self, *, q=1., a=0., c=0., normal=False, variance=.04, resources=1):
        self.mode='normal_fixed' if normal else 'mgf_fixed'
        self.rows=[dict(alpha=.025)]; self.guards=[]; self.z=np.array([2.])
        self.gate_q=np.zeros((1,1,resources,resources)); self.gate_q[0,0,0,0]=q
        self.gate_a=np.zeros((1,1,resources)); self.gate_a[0,0,0]=a
        self.gate_c=np.array([c])
        self.mu_q=self.gate_q.copy();self.mu_a=self.gate_a.copy();self.mu_c=self.gate_c.copy()
        self.variance_q=np.zeros_like(self.gate_q);self.variance_a=np.zeros_like(self.gate_a)
        self.variance_c=np.array([variance])

    def verify_expansion(self, u):
        value=np.einsum('hi,khij,hj->k',u,self.gate_q,u)+2*np.einsum('khi,hi->k',self.gate_a,u)+self.gate_c
        if self.mode=='normal_fixed':
            value-=self.z*np.sqrt(self.variance_c)
        guards=[float(np.einsum('hi,ij,hj',u,q,u)+2*np.sum(a*u)) for _,_,_,q,a in self.guards]
        return dict(feasible=bool(np.all(value>=-1e-7) and all(g>=-1e-7 for g in guards)),
            margins=value,guard_gains=guards,minimum_guard_gain=min(guards) if guards else None)


class LineSeedTests(unittest.TestCase):
    def test_disconnected_exterior_has_minimum_at_inner_boundary(self):
        case=Case(); e=PolynomialEvidence(q=1.,c=-.25+BUFFER)
        receipt,u=search_line(case,e,np.array([[-1.]]),np.array([[1.]]))
        self.assertTrue(receipt['feasible'])
        self.assertEqual(len(receipt['feasible_open_intervals']),2)
        self.assertAlmostEqual(abs(float(u[0,0])),.5,places=7)
        self.assertIn(receipt['selected_kind'],('inward_from_left_root','inward_from_right_root'))
        self.assertFalse(receipt['global_qcqp_optimality_claimed'])

    def test_negative_mean_squared_branch_is_excluded(self):
        case=Case();e=PolynomialEvidence(q=0.,a=.5,c=BUFFER,normal=True,variance=.04)
        r,u=search_line(case,e,np.array([[-1.]]),np.array([[1.]]))
        self.assertTrue(r['feasible'])
        self.assertAlmostEqual(float(u[0,0]),.4,places=7)
        self.assertGreater(float(u[0,0]),0.)

    def test_quartic_central_false_branch_is_excluded(self):
        case=Case();e=PolynomialEvidence(q=1.,c=-.2+BUFFER,normal=True,variance=.0025)
        r,u=search_line(case,e,np.array([[-1.]]),np.array([[1.]]))
        self.assertTrue(r['feasible'])
        self.assertEqual(len(r['feasible_open_intervals']),2)
        self.assertAlmostEqual(abs(float(u[0,0])),np.sqrt(.3),places=7)

    def test_fee_zero_crossing_is_an_interior_candidate(self):
        case=Case(2);e=PolynomialEvidence(q=0.,c=1.,resources=2)
        r,u=search_line(case,e,np.array([[-1.,1.]]),np.array([[1.,0.]]))
        self.assertTrue(r['feasible'])
        self.assertAlmostEqual(case.fee(u),1.,places=8)
        self.assertGreaterEqual(r['fee_breakpoint_count'],1)

    def test_guard_blocks_otherwise_cheaper_side(self):
        case=Case();e=PolynomialEvidence(q=1.,c=-.25+BUFFER)
        e.guards=[(0,1,None,np.array([[0.]]),np.array([[.5]]))]
        r,u=search_line(case,e,np.array([[-1.]]),np.array([[1.]]))
        self.assertTrue(r['feasible'])
        self.assertGreater(float(u[0,0]),0.)
        self.assertAlmostEqual(float(u[0,0]),.5,places=7)

    def test_infeasible_line_is_not_a_global_infeasibility_certificate(self):
        case=Case();e=PolynomialEvidence(q=-1.,c=-1.)
        r,u=search_line(case,e,np.array([[-1.]]),np.array([[1.]]))
        self.assertIsNone(u);self.assertFalse(r['feasible'])
        self.assertFalse(r['infeasibility_proved'])

    def test_invalid_resource_endpoint_and_bad_seed_are_recorded(self):
        case=Case();e=PolynomialEvidence(q=0.,c=1.)
        r,u=search_lines(case,e,np.array([[0.]]),{'bad':np.array([[2.]]),'nan':np.array([[np.nan]])})
        self.assertIsNone(u)
        self.assertEqual(r['exceptions'],1)
        self.assertEqual(r['lines'][0]['status'],'invalid_resource_endpoint')

    def test_line_expansion_matches_arbitrary_multivariate_quadratic(self):
        rng=np.random.default_rng(11424)
        q=rng.normal(size=(3,4,4));q=(q+q.swapaxes(1,2))/2
        a=rng.normal(size=(3,4));x=rng.normal(size=(3,4));d=rng.normal(size=(3,4));c=.37
        coeff=line_coefficients(q,a,c,x,d)
        for t in np.linspace(0,1,17):
            u=x+t*d
            expected=np.einsum('hi,hij,hj',u,q,u)+2*np.sum(a*u)+c
            self.assertAlmostEqual(np.polynomial.polynomial.polyval(t,coeff),expected,places=12)

    def test_constant_normal_mean_does_not_break_quartic_padding(self):
        case=Case();e=PolynomialEvidence(q=0.,a=0.,c=1.,normal=True,variance=.01)
        r,u=search_line(case,e,np.array([[-1.]]),np.array([[1.]]))
        self.assertTrue(r['feasible']);self.assertAlmostEqual(case.fee(u),0.,places=10)

    def test_double_root_is_reported_as_numerical_boundary_not_interval(self):
        case=Case();e=PolynomialEvidence(q=-1.,a=.2,c=-.04+BUFFER)
        r,u=search_line(case,e,np.array([[0.]]),np.array([[1.]]))
        self.assertTrue(all(hi-lo < 1e-7 for lo,hi in r['feasible_open_intervals']))
        self.assertEqual(r['narrow_interval_count'],len(r['feasible_open_intervals']))
        if u is not None:
            self.assertAlmostEqual(float(u[0,0]),.2,places=7)
            self.assertIn(r['selected_kind'],('isolated_or_boundary_root','inward_from_left_root',
                'inward_from_right_root','interval_midpoint'))
        self.assertFalse(r['infeasibility_proved'])


if __name__=='__main__': unittest.main()
