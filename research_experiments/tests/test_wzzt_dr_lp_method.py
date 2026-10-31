"""Small interface/numerical checks; never read study observations or truth."""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
import numpy as np
import pandas as pd
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP
from research_experiments.wzzt_dr_selection.dr_pool_v2 import adapter
DIAGNOSTIC=dict(tiny_LP_optimize_calls=0)


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[1.,0.],[0.,1.],[1.,1.]])
        self.q=np.zeros_like(self.p)
        self.r=np.array([[.5,.1],[.1,.4]])
        self.y=self.p@self.r.T
        self.family=[(0,),(1,),(0,1)]

    def test_exact_same_objective_all_three_methods(self):
        objectives=[]
        for method in (0,1,2):
            with self.subTest(method=method):
                lp=PoolLP(None,self.family,data=(self.p,self.q,self.y),lp_method=method)
                try:
                    self.assertEqual(int(lp.model.Params.Method),method)
                    self.assertEqual(lp.model.Params.TimeLimit,15.)
                    self.assertEqual(lp.model.Params.Threads,1)
                    self.assertEqual(lp.model.Params.FeasibilityTol,1e-9)
                    self.assertEqual(lp.model.Params.OptimalityTol,1e-9)
                    DIAGNOSTIC['tiny_LP_optimize_calls']+=1
                    point=lp.fit(self.family)
                    objectives.append(point['objective'])
                    self.assertLess(point['objective'],1e-8)
                    self.assertEqual(lp.solves[-1]['lp_method'],method)
                    self.assertEqual(lp.solves[-1]['status'],2)
                finally:lp.close()
        self.assertLess(max(objectives)-min(objectives),1e-8)

    def test_default_remains_one(self):
        lp=PoolLP(None,self.family,data=(self.p,self.q,self.y))
        try:self.assertEqual(int(lp.model.Params.Method),1)
        finally:lp.close()

    def test_add_support_preserves_method(self):
        lp=PoolLP(None,self.family[:2],data=(self.p,self.q,self.y),lp_method=2)
        try:
            lp.add_support((0,1))
            self.assertEqual(lp.lp_method,2)
            self.assertEqual(int(lp.model.Params.Method),2)
            DIAGNOSTIC['tiny_LP_optimize_calls']+=1
            self.assertLess(lp.fit(self.family)['objective'],1e-8)
        finally:lp.close()

    def test_invalid_values_fail_before_solver_or_observation_read(self):
        for value in (-1,3,1.,True,False,np.bool_(True),None,'2'):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):PoolLP(None,[],lp_method=value)
                with patch.object(adapter,'normalize',side_effect=AssertionError('observation read')):
                    with self.assertRaises(ValueError):adapter.fit_observed(None,None,None,lp_method=value)

    def test_method_is_identity_and_settings(self):
        frames=(pd.DataFrame(self.p,columns=[201,202]),pd.DataFrame(self.q,columns=[201,202]),
                pd.DataFrame(np.ones_like(self.p),columns=[201,202]),pd.Series(np.ones(len(self.p))))
        normalized=(frames,['d']*len(self.p),{'d':len(self.p)})
        saved=[]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            for name,kwargs in (('default',{}),('barrier',{'lp_method':np.int64(2)})):
                out=Path(root)/name
                with patch.object(adapter,'normalize',return_value=normalized),\
                     patch.object(adapter,'source_hashes',return_value={'fixture':'fixed'}),\
                     patch.object(adapter,'build_bank',side_effect=RuntimeError('unit test stop before QP/LP')):
                    with self.assertRaisesRegex(RuntimeError,'unit test stop'):
                        adapter.fit_observed([{}],[{}],out,**kwargs)
                saved.append(json.loads((out/'result.json').read_text(encoding='utf-8')))
        self.assertEqual([r['lp_method'] for r in saved],[1,2])
        self.assertEqual([r['settings']['lp_method'] for r in saved],[1,2])
        self.assertNotEqual(saved[0]['fingerprint'],saved[1]['fingerprint'])

    def test_adapter_dispatches_and_records_actual_method(self):
        frames=(pd.DataFrame(self.p,columns=[201,202]),pd.DataFrame(self.q,columns=[201,202]),
                pd.DataFrame(np.ones_like(self.p),columns=[201,202]),pd.Series(np.ones(len(self.p))))
        normalized=(frames,['d']*len(self.p),{'d':len(self.p)})
        fake=SimpleNamespace(model=SimpleNamespace(Params=SimpleNamespace(Method=2)),
            gp=SimpleNamespace(gurobi=SimpleNamespace(version=lambda:(0,0,0))),
            solves=[],cache={},close=lambda:None)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            out=Path(root)/'dispatch'
            with patch.object(adapter,'normalize',return_value=normalized),\
                 patch.object(adapter,'source_hashes',return_value={'fixture':'fixed'}),\
                 patch.object(adapter,'build_bank',return_value=([],self.family,{},{})),\
                 patch.object(adapter,'PoolLP',return_value=fake) as constructor:
                with self.assertRaisesRegex(RuntimeError,'expected fifteen'):
                    adapter.fit_observed([{}],[{}],out,lp_method=2)
            saved=json.loads((out/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(constructor.call_args.kwargs['lp_method'],2)
            self.assertEqual(saved['settings']['actual_lp_method'],2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MethodTests))
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),study_observations_read=0,
        full_identifier_calls=0,physical_simulations=0,small_analytical_fixed_family_LPs_planned=4,
        diagnostic=DIAGNOSTIC,executable=sys.executable,
        error_details=[dict(test=str(test),traceback=detail) for test,detail in result.errors],
        failure_details=[dict(test=str(test),traceback=detail) for test,detail in result.failures],
        source_sha256={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in
            (Path(__file__),REPO/'research_experiments/wzzt_dr_selection/dr_pool_v2/engine.py',
             REPO/'research_experiments/wzzt_dr_selection/dr_pool_v2/adapter.py')})
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
