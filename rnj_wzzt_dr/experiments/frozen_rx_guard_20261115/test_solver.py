"""PSD geometry, full-model wiring and tiny synthetic solver checks only."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from . import solver


class ToyCase:
    def __init__(self, shape=(1,2)):
        self.H,self.R=shape
        self.shape=shape
        self.lower=np.zeros(shape)
        self.upper=np.full(shape,3.)
    def array(self,u):
        u=np.asarray(u,float)
        if u.shape!=self.shape or not np.isfinite(u).all(): raise ValueError('action')
        return u
    def audit(self,u,tolerance=1e-7):
        u=self.array(u)
        return dict(passed=bool(np.all(u>=self.lower-tolerance) and np.all(u<=self.upper+tolerance)))
    def fee(self,u): return float(np.sum(self.array(u)*np.arange(1,self.R+1)[None,:]))
    def attach(self,model,name='toy'):
        u=model.addMVar(self.shape,lb=self.lower,ub=self.upper,name=name)
        return u,(u*np.arange(1,self.R+1)[None,:]).sum()


class ToyEvidence:
    targets=[0]
    guards=[]
    rows=[dict(target_rep=0,target_native=0,competitor_native=1,alpha=.1)]
    def __init__(self,case,score,validation,forecast,base_kw,*,mode,target_mass,**kwargs):
        self.mode=mode
        self.gate_q=np.zeros((1,1,2,2));self.gate_a=np.full((1,1,2),.5);self.gate_c=np.array([-2.])
        self.mu_q=self.gate_q.copy();self.mu_a=self.gate_a.copy()
        self.mu_c=np.array([-2.+norm.isf(.1)*.1])
        self.variance_q=self.gate_q.copy();self.variance_a=np.zeros((1,1,2));self.variance_c=np.array([.01])
    def verify_expansion(self,u):
        margin=float(np.asarray(u).sum()-2.)
        return dict(margins=np.array([margin]),guard_gains=np.array([]),
            pair_failure=np.array([min(1.,.1*np.exp(-margin))]),feasible=bool(margin>=-1e-7))
    direct=verify_expansion


def guard(Q=None,limit=.25):
    return [dict(id='test',Q=np.diag([1.,0.]) if Q is None else Q,limit=limit,units='synthetic')]


class AlgebraTests(unittest.TestCase):
    def test_full_flattening_includes_cross_hour_entries(self):
        case=ToyCase((2,3));B=np.array([[1.,2.,0.,-1.,0.,.5],[0.,1.,1.,0.,2.,0.]])
        Q=B.T@B;u=np.arange(6,dtype=float).reshape(2,3)/10
        wrapped=solver.PSDGuardCase(case,guard(Q,10.))
        result=wrapped.guard_audit(u)
        self.assertAlmostEqual(result['rows'][0]['value'],float(np.square(B@u.ravel()).sum()),places=13)
        self.assertTrue(result['passed'])
        self.assertEqual(wrapped.fee(u),case.fee(u))
        np.testing.assert_array_equal(wrapped.lower,case.lower)
    def test_bad_dimensions_indefiniteness_and_asymmetry_rejected(self):
        case=ToyCase()
        for Q in (np.eye(3),np.diag([1.,-1.]),np.array([[1.,.1],[0.,1.]]),np.full((2,2),np.nan)):
            with self.subTest(Q=str(Q)),self.assertRaises(ValueError): solver.PSDGuardCase(case,guard(Q))
        with self.assertRaises(ValueError): solver.PSDGuardCase(case,guard(limit=-1.))
        with self.assertRaises(ValueError): solver.PSDGuardCase(case,[])
    def test_physical_tolerance_cannot_expand_guard(self):
        wrapped=solver.PSDGuardCase(ToyCase(),guard())
        self.assertFalse(wrapped.audit(np.array([[.50000001,0.]]),tolerance=1.)['passed'])
        good=wrapped.audit(np.array([[.5,0.]]))
        self.assertTrue(good['passed'])
        self.assertFalse(good['psd_guard']['encoded_budget_passed'])
        self.assertEqual(good['psd_guard']['rows'][0]['slack'],0.)
    def test_zero_limit_uses_no_relative_buffer(self):
        wrapped=solver.PSDGuardCase(ToyCase(),guard(limit=0.))
        self.assertTrue(wrapped.guard_audit(np.array([[0.,1.]]))['passed'])
        self.assertFalse(wrapped.guard_audit(np.array([[1e-8,1.]]))['passed'])
        self.assertEqual(wrapped.metadata()[0]['relative_model_buffer'],0.)
    def test_multiple_budgets_are_all_checked(self):
        wrapped=solver.PSDGuardCase(ToyCase(),guard()+[dict(id='second',Q=np.diag([0.,1.]),limit=.04)])
        self.assertTrue(wrapped.audit(np.array([[.3,.1]]))['passed'])
        self.assertFalse(wrapped.audit(np.array([[.3,.3]]))['passed'])
    def test_caller_matrix_cannot_mutate_frozen_guard(self):
        Q=np.diag([1.,0.]);wrapped=solver.PSDGuardCase(ToyCase(),guard(Q))
        Q[:]=0.
        self.assertFalse(wrapped.audit(np.array([[1.,0.]]))['passed'])


class ReceiptTests(unittest.TestCase):
    def call(self,fake,**kwargs):
        with patch.object(solver.engine,'plan',fake):
            return solver.plan(ToyCase(),{},None,None,200.,guards=guard(),env=None,seconds=2.,**kwargs)
    def test_same_portfolio_and_restricted_bound_conflict(self):
        seeds={'bad':np.array([[1.,2.]]),'good':np.array([[.4,1.7]])}
        def fake(case,*args,**kwargs):
            self.assertIs(kwargs['seed_points'],seeds)
            self.assertFalse(case.audit(seeds['bad'])['passed'])
            self.assertTrue(case.audit(seeds['good'])['passed'])
            return dict(feasible=True,status='feasible',cost_yuan=3.8,lower_bound_yuan=4.,
                global_optimality_verified=True,optimizer_invoked=True),seeds['good'].copy()
        receipt,u=self.call(fake,seed_points=seeds)
        self.assertEqual(receipt['lower_bound_yuan'],0.)
        self.assertTrue(receipt['lower_bound_withdrawn'])
        self.assertFalse(receipt['global_optimality_verified'])
        self.assertFalse(receipt['seed_portfolio_modified'])
        self.assertFalse(receipt['psd_guard_seed_audits'][0]['physical_and_budget_feasible'])
        self.assertIn('WITH',receipt['raw_bound_scope'])
        self.assertEqual(receipt['cost_yuan'],ToyCase().fee(u))
    def test_direct_invalid_return_is_rejected_without_clipping(self):
        def fake(*args,**kwargs):
            return dict(feasible=True,status='feasible',cost_yuan=4.,lower_bound_yuan=0.),np.array([[1.,2.]])
        receipt,u=self.call(fake)
        self.assertIsNone(u)
        self.assertFalse(receipt['feasible'])
        self.assertFalse(receipt['returned_point_was_clipped'])
        self.assertFalse(receipt['infeasibility_proved'])
    def test_boundary_seed_withdraws_buffered_problem_bound(self):
        def fake(*args,**kwargs):
            return dict(feasible=True,status='feasible',cost_yuan=3.7,lower_bound_yuan=3.7,
                global_optimality_verified=True),np.array([[.5,1.6]])
        receipt,u=self.call(fake)
        self.assertIsNotNone(u)
        self.assertEqual(receipt['lower_bound_yuan'],0.)
        self.assertFalse(receipt['global_optimality_verified'])
        self.assertIn('buffered',receipt['lower_bound_withdrawal_reason'])
    def test_exception_does_not_retain_disallowed_anchor(self):
        def fake(*args,**kwargs): raise RuntimeError('synthetic failure')
        receipt,u=self.call(fake,seed_points={'anchor':np.array([[1.,2.]])})
        self.assertIsNone(u)
        self.assertFalse(receipt['new_guard_and_probability_feasible'])
        self.assertEqual(receipt['status'],'engine_exception')


class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gurobipy as gp
        cls.gp=gp
        cls.env=gp.Env(empty=True);cls.env.setParam('OutputFlag',0);cls.env.start()
    @classmethod
    def tearDownClass(cls): cls.env.dispose()
    def test_guard_is_inside_model_not_postprocessing(self):
        wrapped=solver.PSDGuardCase(ToyCase(),guard())
        with self.gp.Model(env=self.env) as model:
            model.Params.OutputFlag=0
            u,fee=wrapped.attach(model)
            model.addConstr(u==np.array([[.6,2.]]));model.setObjective(fee);model.optimize()
            self.assertEqual(model.Status,self.gp.GRB.INFEASIBLE)
    def test_three_unchanged_gate_modes_with_guard(self):
        seeds={'invalid_budget':np.array([[2.5,0.]]),'valid':np.array([[.4,1.7]])}
        for mode in ('normal_fixed','mgf_fixed','mgf_forecast'):
            with self.subTest(mode=mode),patch.object(solver.engine,'DistributionEvidence',ToyEvidence):
                receipt,u=solver.plan(ToyCase(),{},None,None,200.,guards=guard(),mode=mode,
                    env=self.env,seconds=2.,seed_points=seeds)
                self.assertTrue(receipt['feasible'])
                self.assertTrue(receipt['final_guard_and_physical_audit']['passed'])
                self.assertLessEqual(float(u[0,0]),.5+1e-12)
                self.assertGreaterEqual(float(u.sum()),2.-1e-7)
                self.assertLessEqual(receipt['cost_yuan'],3.8+1e-9)
                self.assertFalse(receipt['initial']['candidates'][0]['qcqp_feasible'])
                self.assertFalse(receipt['original_evidence_constraints_changed'])


if __name__=='__main__': unittest.main()
