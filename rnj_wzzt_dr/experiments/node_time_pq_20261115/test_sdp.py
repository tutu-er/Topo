"""Small algebra and backend tests; no project AC, fitting or formal cases."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import sdp


class ToyCase:
    H=R=1;shape=(1,1);dt=1.
    lower=np.zeros(shape);upper=np.ones(shape)*2
    fee_plus=np.ones(shape);fee_minus=np.ones(shape)
    A_eq=np.empty((0,1));b_eq=np.empty(0)
    A_ub=np.array([[1.]]);b_ub=np.array([1.8])
    def array(self,u):return np.asarray(u,float).reshape(self.shape)
    def audit(self,u):
        return sdp.linear_audit(dict(A_eq=self.A_eq,b_eq=self.b_eq,A_ub=self.A_ub,
            b_ub=self.b_ub,lower=self.lower,upper=self.upper),u)
    def fee(self,u):return float(abs(u).sum())
    def attach(self,model,name='toy'):
        u=model.addMVar(self.shape,lb=self.lower,ub=self.upper,name=name)
        model.addConstr(u[0,0]<=1.8)
        return u,u.sum()


class ToyEvidence:
    mode='mgf_fixed';rows=[dict(target_native=0,competitor_native=1)];guards=[]
    gate_q=np.ones((1,1,1,1));gate_a=np.zeros((1,1,1));gate_c=np.array([-1.])
    def verify_expansion(self,u):
        margin=float(np.square(u).sum()-1)
        return dict(feasible=margin>=-1e-7,margins=np.array([margin]),guard_gains=[],
                    minimum_pair_margin=margin,pair_failure=[min(1.,.1*np.exp(-margin))])


class Algebra(unittest.TestCase):
    def test_encoded_rows_used_original_rows_retained(self):
        case=ToyCase()
        case.encoded_A_ub=np.array([[2.]])
        case.encoded_b_ub=np.array([3.6-1e-7])
        case.encoded_A_eq=np.array([[0.]])
        case.encoded_b_eq=np.array([0.])
        data=sdp.problem_arrays(case,ToyEvidence())
        np.testing.assert_array_equal(data['A_ub'],case.encoded_A_ub)
        np.testing.assert_array_equal(data['b_ub'],case.encoded_b_ub)
        np.testing.assert_array_equal(data['A_eq'],case.encoded_A_eq)
        np.testing.assert_array_equal(data['original_A_ub'],case.A_ub)

    def test_hourly_rank_one_lift_is_exact_and_keeps_linear_band(self):
        data=sdp.problem_arrays(ToyCase(),ToyEvidence())
        x=np.array([[1.5]]);W=np.einsum('hi,hj->hij',x,x)
        value=sdp.lifted_values(data['gate_q'],data['gate_a'],data['gate_c'],x,W)
        np.testing.assert_allclose(value,[1.25])
        info=sdp.moment_audit(data,x,W)
        self.assertTrue(info['linear']['passed']);self.assertEqual(info['blocks'][0]['numerical_rank'],1)
        self.assertLessEqual(info['maximum_McCormick_violation'],1e-12)
        self.assertFalse(sdp.linear_audit(data,np.array([[1.9]]))['passed'])

    def test_hourly_quadratics_not_full_day_outer_product(self):
        rng=np.random.default_rng(9);u=rng.normal(size=(4,3));Q=rng.normal(size=(2,4,3,3));a=rng.normal(size=(2,4,3));c=np.array([1.,-2.])
        W=np.einsum('hi,hj->hij',u,u)
        expected=np.einsum('hi,khij,hj->k',u,Q,u)+2*np.einsum('khi,hi->k',a,u)+c
        np.testing.assert_allclose(sdp.lifted_values(Q,a,c,u,W),expected,atol=1e-12)

    def test_fixed_seed_rounding_and_indefinite_covariance_are_explicit(self):
        mean=np.array([[.1,.2],[.3,.4]]);W=np.einsum('hi,hj->hij',mean,mean)
        W[0]+=np.diag([1.,-1e-4]);W[1]+=np.eye(2)
        a,d=sdp.rounded_actions(mean,W,31,3);b,_=sdp.rounded_actions(mean,W,31,3)
        self.assertEqual(len(a),4)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertAlmostEqual(d[0]['negative_eigenvalue_mass_clipped_for_rounding'],1e-4)

    def test_normal_unsupported_and_unavailable_backend_are_not_feasible(self):
        for mode,status in [('normal_fixed','unsupported_mode'),('mgf_fixed','sdp_backend_unavailable')]:
            evidence=ToyEvidence();evidence.mode=mode
            with tempfile.TemporaryDirectory() as d,patch.object(sdp,'runtime',return_value=None):
                receipt,points=sdp.solve(ToyCase(),evidence,seconds=2.,seed=1,record_directory=d)
                self.assertEqual(receipt['status'],status);self.assertEqual(points,{})
                self.assertIsNone(receipt['effective_qcqp_lower_bound']);self.assertFalse(receipt['bound_validated'])


class InstalledBackend(unittest.TestCase):
    @unittest.skipIf(sdp.runtime() is None,'No already installed SDP runtime')
    def test_small_SDP_relaxation_rounding_and_physical_projection(self):
        # min u, u^2>=1, 0<=u<=2: hourly SDP+box envelope has value .5.
        # Linear band u<=1.8 is kept. Gaussian draws require projection.
        with tempfile.TemporaryDirectory() as d:
            receipt,points=sdp.solve(ToyCase(),ToyEvidence(),seconds=20.,seed=201,record_directory=d)
            self.assertEqual(receipt['status'],'seeds_generated',receipt)
            self.assertAlmostEqual(receipt['raw_sdp_objective_diagnostic'],.5,places=3)
            self.assertIsNone(receipt['effective_qcqp_lower_bound']);self.assertFalse(receipt['bound_validated'])
            self.assertIn('raw_dual_residual',receipt['relaxation'])
            self.assertGreater(receipt['returned_probability_feasible_count'],0)
            self.assertGreater(receipt['returned_resource_only_count'],0)
            for point in points.values():self.assertTrue(ToyCase().audit(point)['passed'])
            for row in receipt['candidates']:
                if row.get('admitted_to_pool'):
                    with np.load(row['action_ref']['path']) as arrays:
                        np.testing.assert_array_equal(points[row['pool_name']],arrays[row['array_key']])
            self.assertTrue(any('projection' in r for r in receipt['candidates']))
            disk=json.loads((Path(d)/'receipt.json').read_text())
            self.assertEqual(disk['returned_candidates'],list(points))


if __name__=='__main__':unittest.main()
