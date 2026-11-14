"""Analytic tiny optima and an independent amplitude-box bound check."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from . import solver


class ToyCase:
    H=R=1
    shape=(1,1)
    def __init__(self,upper=3.):
        self.lower=np.zeros(self.shape);self.upper=np.full(self.shape,upper)
    def array(self,u):return np.asarray(u,float).reshape(self.shape)
    def audit(self,u):return dict(passed=bool(np.all(u>=-1e-8) and np.all(u<=self.upper+1e-8)))
    def fee(self,u):return float(np.asarray(u).sum())
    def attach(self,model):
        u=model.addMVar(self.shape,lb=self.lower,ub=self.upper,name='u')
        return u,u.sum()


class ToyEvidence:
    targets=[0]
    guards=[]
    rows=[dict(target_rep=0,target_native=0,competitor_native=1,alpha=.1)]
    def __init__(self,case,score,validation,forecast,base_kw,*,mode,target_mass):
        self.mode=mode
        self.gate_q=np.ones((1,1,1,1));self.gate_a=np.zeros((1,1,1));self.gate_c=np.array([-4.])
        self.mu_q=np.zeros((1,1,1,1));self.mu_a=np.full((1,1,1),.5);self.mu_c=np.zeros(1)
        self.variance_q=np.zeros((1,1,1,1));self.variance_a=np.zeros((1,1,1));self.variance_c=np.ones(1)
    def verify_expansion(self,u):
        x=float(np.asarray(u).item())
        margin=x-norm.isf(.1) if self.mode=='normal_fixed' else x*x-4.
        failure=float(norm.cdf(-x)) if self.mode=='normal_fixed' else min(1.,.1*np.exp(-margin))
        return dict(margins=np.array([margin]),guard_gains=np.array([]),
                    pair_failure=np.array([failure]),feasible=bool(margin>=-1e-7 and failure<=.1+1e-9))
    direct=verify_expansion


class Bounds(unittest.TestCase):
    def test_negative_diagonal_interior_maximum(self):
        result=solver.quadratic_box_upper(np.array([[-2.]]),np.zeros((1,1)),3.,np.array([[-2.]]),np.array([[1.]]))
        self.assertAlmostEqual(result['upper'],3.,places=7)

    def test_indefinite_and_offdiagonal_box(self):
        rng=np.random.default_rng(2026111420)
        for _ in range(10):
            q=rng.normal(size=(2,3,3));q=(q+q.transpose(0,2,1))/2
            a=rng.normal(size=(2,3));c=float(rng.normal())
            lower=rng.uniform(-2.,0.,size=(2,3));upper=rng.uniform(0.,3.,size=(2,3))
            bound=solver.quadratic_box_upper(q,a,c,lower,upper)['upper']
            points=rng.uniform(lower,upper,size=(2048,2,3))
            values=np.einsum('bhi,hij,bhj->b',points,q,points)+2*np.einsum('bhi,hi->b',points,a)+c
            self.assertLessEqual(float(values.max()),bound)


class NativeOptima(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gurobipy as gp
        cls.env=gp.Env(empty=True);cls.env.setParam('OutputFlag',0);cls.env.start()
    @classmethod
    def tearDownClass(cls):cls.env.dispose()
    def solve(self,mode,case=None):
        with patch.object(solver,'DistributionEvidence',ToyEvidence):
            return solver.plan(case or ToyCase(),{},None,np.zeros((1,1)),1.,mode=mode,
                seconds=2.,env=self.env,seed_points={},record_directory=None)
    def test_quadratic_gate_without_feasible_seed(self):
        receipt,point=self.solve('mgf_fixed')
        self.assertTrue(receipt['feasible']);self.assertFalse(receipt['initial']['feasible'])
        self.assertEqual(receipt['source'],'native_found')
        self.assertAlmostEqual(float(point.item()),2.,places=6)
        self.assertLess(receipt['relative_gap'],1e-4)
    def test_normal_gate_without_feasible_seed(self):
        receipt,point=self.solve('normal_fixed')
        self.assertTrue(receipt['feasible']);self.assertFalse(receipt['initial']['feasible'])
        self.assertAlmostEqual(float(point.item()),norm.isf(.1),places=6)
        self.assertLess(receipt['relative_gap'],1e-4)
    def test_box_proof_does_not_claim_chance_infeasibility(self):
        receipt,point=self.solve('mgf_fixed',ToyCase(upper=1.))
        self.assertIsNone(point);self.assertEqual(receipt['status'],'infeasible_by_probability_box_bound')
        self.assertFalse(receipt['optimizer_invoked'])
        self.assertIn('not the original probability',receipt['infeasibility_scope'])


if __name__=='__main__':unittest.main()
