"""W removal must change only the declared search operation and E allowance."""
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ..history import freeze_history
from ..identification import algorithm, trees
from ..identification.optimization import QPFitter, criterion, response_matrices
from ..observations import Observation


def fixture(connection=(0, 1), loss='mse'):
    n=4; rng=np.random.default_rng(2026111709)
    p=rng.uniform(.1,.6,(24,n));q=rng.uniform(.01,.25,(24,n))
    family=trees.canonical([*trees.basis(n),connection])
    target=dict(supports=[list(c) for c in family],
        r=[.3 if c==connection else .06 for c in family],
        x=[.14 if c==connection else .03 for c in family])
    R,X=response_matrices(target,n)
    obs=Observation(p,q,p@R.T+q@X.T+rng.normal(0.,.001,p.shape),tuple(range(101,105)))
    reference=.02 if loss=='mse' else .1
    fitter=QPFitter(obs,loss=loss,reference_mse=.02,reference_loss=reference,qp_seconds=10.)
    reference_point=fitter.fit(family)
    pool=trees.canonical([*trees.basis(n),connection,(2,3)])
    common=dict(guidance=dict(pool=pool,reference_family=family,R=R,X=X),
        reference_point=reference_point,reference_mse=.02,reference_loss=reference,loss=loss,
        seconds=0.,n=n,labels=obs.labels,training_sha256=algorithm._training_fingerprint(obs))
    config=algorithm.SearchConfig(loss=loss,total_seconds=30.,per_mip_seconds=5.,qp_seconds=10.,
        output_reserve_seconds=3.,internal_floor=0.)
    return obs,fitter,common,config


class WModeTests(unittest.TestCase):
    def test_invalid_mode_and_E_limit_fail_before_preprocessing(self):
        obs,_,_,cfg=fixture()
        for changes in ({'w_mode':'unknown'},{'e_seconds':0.},{'e_seconds':-1.},
                        {'e_seconds':float('inf')},{'e_seconds':float('nan')}):
            with self.subTest(changes=changes),patch.object(algorithm,'prepare_common') as prepare:
                with self.assertRaises(ValueError):algorithm.fit(obs,obs,config=replace(cfg,**changes))
                prepare.assert_not_called()

    def test_pool_only_keeps_pretrial_and_never_calls_native_extension(self):
        for loss in ('mse','mae'):
            with self.subTest(loss=loss):
                obs,fitter,common,cfg=fixture(loss=loss)
                cfg=replace(cfg,w_mode='pool_only')
                current=fitter.fit(trees.basis(obs.n));solver=SimpleNamespace(extend=Mock())
                result,event=algorithm._forward(current,fitter,solver,cfg,common['guidance']['pool'],
                    perf_counter()+20.,reference_family=common['guidance']['reference_family'])
                solver.extend.assert_not_called()
                self.assertTrue(event['accepted']);self.assertIn([0,1],result['supports'])
                self.assertGreater(event['pretrial_QP_count'],0)
                self.assertLess(criterion(result,obs.n,cfg.alpha),criterion(current,obs.n,cfg.alpha))
                self.assertEqual(event['returned_point_source'],'pool_pretrial_QP')
                self.assertEqual(event['w_mode'],'pool_only')
                self.assertEqual(event['solver']['stop_reason'],'ablation_pool_only')
                self.assertFalse(event['solver']['solve_started']);self.assertIsNone(event['solver']['bound'])
                self.assertIsNone(event['solver']['gap']);self.assertFalse(event['has_native_solution'])
                self.assertFalse(event['unrestricted_support_domain'])
                self.assertEqual(event['initial_score_scope'],'pool_pretrial_point_without_native_start')

    def test_full_can_receive_a_better_support_outside_the_pool(self):
        obs,fitter,common,cfg=fixture(connection=(0,2))
        current=fitter.fit(trees.basis(obs.n));target=fitter.fit(common['guidance']['reference_family'])
        pool=trees.canonical([*trees.basis(obs.n),(0,1)])
        native=SimpleNamespace(extend=Mock(return_value=(deepcopy(target),dict(solve_started=True,status=2))))
        result,event=algorithm._forward(current,fitter,native,cfg,pool,perf_counter()+20.)
        native.extend.assert_called_once();self.assertIn([0,2],result['supports'])
        self.assertTrue(event['native_improved_initial']);self.assertTrue(event['unrestricted_support_domain'])
        disabled=SimpleNamespace(extend=Mock())
        pooled,_=algorithm._forward(current,fitter,disabled,replace(cfg,w_mode='pool_only'),pool,perf_counter()+20.)
        disabled.extend.assert_not_called();self.assertNotIn([0,2],pooled['supports'])

    def test_skip_W_never_prescans_and_still_uses_original_E_pool_and_reference(self):
        obs,_,common,cfg=fixture();calls=[]
        class ReturnWarm:
            def __init__(self,*args):self.records=[]
            def extend(self,*args,**kwargs):raise AssertionError('Skip W must never call extend')
            def select(self,pool,keep,warm,**kwargs):
                calls.append((pool,keep,warm,kwargs));return deepcopy(warm),dict(solve_started=True,status=2)
            def close(self):pass
        with patch.object(algorithm,'StructuralSolver',ReturnWarm), \
             patch.object(algorithm,'_forward',side_effect=AssertionError('Skip W must not prescan')):
            result=algorithm.fit(obs,obs,method='101',config=replace(cfg,w_mode='skip'),common=common)
        self.assertEqual(len(calls),1);self.assertEqual([e['stage'] for e in result['events']],['E'])
        self.assertEqual(result['events'][0]['initial_source'],'single_RNJ_reference')
        expected=trees.canonical([*common['guidance']['pool'],*(tuple((i,j)) for i in range(obs.n) for j in range(i+1,obs.n))])
        self.assertEqual(calls[0][0],expected)
        self.assertEqual(calls[0][1],trees.basis(obs.n))
        self.assertFalse(result['complete_RNJ_output_fallback']);freeze_history(result,obs)
        self.assertEqual(result['rnj_pool_guidance']['structural_use'],'used')
        self.assertEqual(result['rnj_pool_guidance']['pool'],common['guidance']['pool'])

    def test_explicit_E_cap_is_passed_and_total_structural_deadline_still_wins(self):
        obs,_,common,cfg=fixture();deadlines=[]
        class NoSolution:
            def __init__(self,*args):self.records=[]
            def select(self,*args,**kwargs):
                deadlines.append(kwargs['deadline']);return None,dict(solve_started=True,status=9)
            def close(self):pass
        for allowance,expected in ((None,127.),(2.,102.),(100.,127.)):
            with self.subTest(allowance=allowance),patch.object(algorithm,'StructuralSolver',NoSolution), \
                 patch.object(algorithm,'perf_counter',return_value=100.):
                result=algorithm.fit(obs,obs,method='101',
                    config=replace(cfg,w_mode='skip',e_seconds=allowance),common=common)
                self.assertEqual(deadlines[-1],expected)
                self.assertEqual(result['config']['e_seconds'],allowance)

    def test_default_and_explicit_full_have_identical_calls_points_and_scores(self):
        obs,_,common,cfg=fixture()
        calls=[]
        class NoSolution:
            def __init__(self,*args):self.records=[]
            def extend(self,current,candidate,warm,**kwargs):
                calls.append(('W',deepcopy(current),candidate,deepcopy(warm),kwargs['deadline']))
                return None,dict(solve_started=True,status=9)
            def select(self,pool,keep,warm,**kwargs):
                calls.append(('E',pool,keep,deepcopy(warm),kwargs['deadline']))
                return None,dict(solve_started=True,status=9)
            def close(self):pass
        results=[];records=[]
        for config in (cfg,replace(cfg,w_mode='full',e_seconds=None)):
            calls.clear()
            with patch.object(algorithm,'StructuralSolver',NoSolution),patch.object(algorithm,'perf_counter',return_value=100.):
                results.append(algorithm.fit(obs,obs,method='101',config=config,common=common))
            records.append(deepcopy(calls))
        self.assertEqual(records[0],records[1])
        self.assertEqual(results[0]['path'],results[1]['path'])
        for key in ('selected','frozen_candidates','reference_mse','reference_loss','stop_reason'):
            self.assertEqual(results[0][key],results[1][key])
        self.assertEqual(results[0]['config']['w_mode'],'full')

    def test_pure_W_marks_RNJ_pool_unused(self):
        obs,_,common,cfg=fixture()
        with patch.object(algorithm,'_forward',side_effect=AssertionError('Skip mode')):
            result=algorithm.fit(obs,obs,method='000',config=replace(cfg,w_mode='skip'),common=common)
        self.assertEqual(result['rnj_pool_guidance']['structural_use'],'not_used')
        self.assertEqual(result['rnj_pool_guidance']['pool'],())
        self.assertTrue(result['rnj_pool_guidance']['training_scalar_reference_used'])

    def test_pool_only_and_skip_reach_real_E_without_calling_native_W(self):
        try:import gurobipy  # noqa: F401
        except ImportError:self.skipTest('Gurobi is not installed')
        for loss in ('mse','mae'):
            for mode in ('pool_only','skip'):
                with self.subTest(loss=loss,mode=mode):
                    obs,_,common,cfg=fixture(loss=loss)
                    with patch.object(algorithm.StructuralSolver,'extend',side_effect=AssertionError('Native W disabled')):
                        result=algorithm.fit(obs,obs,method='101',
                            config=replace(cfg,w_mode=mode,e_seconds=5.),common=common)
                    self.assertEqual(len(result['MIQP']),1)
                    self.assertEqual(result['MIQP'][0]['scope'],'finite laminar support pool')
                    self.assertTrue(result['MIQP'][0]['objective_replay_passed'])
                    self.assertEqual(result['MIQP'][0]['status'],2)
                    self.assertLessEqual(result['MIQP'][0]['requested_solve_seconds'],5.)
                    self.assertTrue(all(not e['solver']['solve_started'] for e in result['events'] if e['stage']=='W'))
                    if mode=='skip':self.assertFalse(any(e['stage']=='W' for e in result['events']))
                    freeze_history(result,obs)


if __name__=='__main__':unittest.main()
