"""Loss, objective, certificate and small structural-enumeration contracts."""
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import json
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..history import freeze_history
from ..observations import Observation
from ..identification import algorithm, optimization as opt, output_floor, trees


def data_fixture(n=4):
    rng = np.random.default_rng(20261117+n)
    p = rng.uniform(.1, .6, (16, n))
    q = rng.uniform(-.1, .3, (16, n))
    family = trees.canonical([*trees.basis(n), (0, 1)])
    point = dict(supports=[list(c) for c in family],
                 r=[.24 if c == (0,1) else .08 for c in family],
                 x=[.12 if c == (0,1) else .04 for c in family])
    R, X = opt.response_matrices(point, n)
    y = p@R.T+q@X.T+rng.normal(0., .002, p.shape)
    y[0,0] += .04
    return Observation(p,q,y,tuple(range(101,101+n))), family


def configuration(loss):
    return algorithm.SearchConfig(loss=loss, total_seconds=30., per_mip_seconds=10., qp_seconds=10.,
        output_reserve_seconds=3., internal_floor=0., mip_gap=1e-8, mip_abs_gap=1e-9)


def independent_design(data, family):
    rows=[]
    for t in range(data.points):
        for i in range(data.n):
            rows.append([sum(data.p[t,j] for j in c) if i in c else 0. for c in family]
                        +[sum(data.q[t,j] for j in c) if i in c else 0. for c in family])
    return np.asarray(rows)


class TrainingLossTests(unittest.TestCase):
    def test_mse_mean_and_mae_median_differ_with_independent_certificates(self):
        obs=Observation(np.ones((5,1)),np.zeros((5,1)),np.array([[.1],[.2],[.2],[.3],[1.7]]),(1,))
        fits={}
        for loss in ('mse','mae'):
            fitter=opt.QPFitter(obs,loss=loss,reference_mse=.04,reference_loss=.2,qp_seconds=10.)
            point=fitter.fit([(0,)])
            fits[loss]=point
            self.assertAlmostEqual(point['objective'],opt.errors(obs,point)[loss]/.2,places=12)
            self.assertAlmostEqual(point['mse'],opt.errors(obs,point)['mse'],places=12)
            self.assertAlmostEqual(point['mae'],opt.errors(obs,point)['mae'],places=12)
            self.assertIs(fitter.fit([(0,)]),point)
            self.assertEqual(len(fitter.records),1)
            if loss=='mae':
                rec=fitter.records[0]
                self.assertTrue(rec['certificate_passed'])
                self.assertLess(abs(rec['certificate_gap']),1e-10)
                self.assertLess(rec['certificate']['primal_violation'],1e-9)
                self.assertLess(rec['certificate']['dual_stationarity_max_abs'],1e-9)
                self.assertEqual(rec['forwarded_highs_options'],dict(threads=1))
                self.assertFalse(rec['tie_break_regularization'])
                F=independent_design(obs,[(0,)])/.2;y=obs.y.ravel()/.2
                theta=np.r_[point['r'],point['x']]
                s=np.sign(F@theta-y)/y.size
                for index,value in rec['certificate']['dual_witness_adjustments']:s[index]=value
                self.assertLessEqual(float(np.max(abs(s))),1./y.size)
                lower=-float(s@y)+float(np.minimum(0.,fitter.bound*(F.T@s)).sum())
                self.assertAlmostEqual(max(0.,lower),rec['lower_bound'],places=12)
        self.assertAlmostEqual(fits['mse']['r'][0],np.mean(obs.y),places=12)
        self.assertAlmostEqual(fits['mae']['r'][0],np.median(obs.y),places=12)

    def test_mae_upper_bound_is_part_of_the_dual_certificate(self):
        obs=Observation(np.ones((3,1)),np.zeros((3,1)),np.array([[2.5],[3.],[4.]]),(1,))
        fitter=opt.QPFitter(obs,loss='mae',reference_loss=.2,bound=2.)
        point=fitter.fit([(0,)])
        self.assertEqual(point['r'],[2.])
        expected=float(np.mean(obs.y-2.))/.2
        self.assertAlmostEqual(point['objective'],expected,places=12)
        self.assertAlmostEqual(fitter.records[-1]['lower_bound'],expected,places=12)

    def test_default_mse_is_identical_and_matches_independent_least_squares(self):
        obs,family=data_fixture()
        default=opt.QPFitter(obs,reference_mse=.03).fit(family)
        explicit=opt.QPFitter(obs,reference_mse=.03,loss='mse',reference_loss=.03).fit(family)
        for key in ('r','x','objective','mse'):
            self.assertEqual(default[key],explicit[key])
        solution=np.linalg.lstsq(independent_design(obs,family),obs.y.ravel(),rcond=None)[0]
        self.assertTrue(np.all((solution>0.) & (solution<2.)))
        np.testing.assert_allclose(np.r_[default['r'],default['x']],solution,rtol=1e-10,atol=1e-12)

    def test_common_keeps_same_mse_selector_scale_and_rejects_wrong_loss(self):
        obs,family=data_fixture(3)
        guidance=dict(reference_family=family,pool=family,R=np.eye(3),X=np.eye(3))
        with patch.object(trees,'rnj_guidance',return_value=guidance):
            mse=algorithm.prepare_common(obs,configuration('mse'))
            mae=algorithm.prepare_common(obs,configuration('mae'))
        self.assertEqual(mse['reference_mse'],mae['reference_mse'])
        self.assertEqual(mse['reference_loss'],mse['reference_mse'])
        self.assertEqual(mae['reference_loss'],mae['reference_point']['mae'])
        self.assertNotEqual(mae['reference_loss'],mae['reference_mse'])
        with self.assertRaisesRegex(ValueError,'different loss'):
            algorithm.fit(obs,obs,method='rnj',config=configuration('mae'),common=mse)
        legacy=deepcopy(mse);legacy.pop('loss');legacy.pop('reference_loss')
        for loss,common in (('mse',legacy),('mae',mae)):
            result=algorithm.fit(obs,obs,method='rnj',config=configuration(loss),common=common)
            self.assertEqual(result['selection_metric'],'validation_mse')
            self.assertEqual(result['reference_mse'],mse['reference_mse'])
            self.assertEqual(freeze_history(result,obs).current,result['selected']['index'])

    def test_mae_reference_and_empty_floor_use_the_training_loss(self):
        obs,family=data_fixture(3);cfg=configuration('mae')
        fitter=opt.QPFitter(obs,loss='mae',reference_loss=.1,reference_mse=.01)
        current=fitter.fit(trees.basis(3));reference=fitter.fit(family)
        ref=deepcopy(reference);ref['objective']=-123.
        with patch.object(fitter,'fit',side_effect=AssertionError('Reference is rescored, not refitted')):
            initial,source,info=algorithm._single_reference_initial(current,ref,fitter,cfg,family,trees.basis(3))
        self.assertEqual(source,'single_RNJ_reference')
        self.assertAlmostEqual(initial['objective'],opt.errors(obs,initial)['mae']/.1)
        self.assertEqual(info['reference_refit_count'],0)
        empty,record=output_floor._fit(fitter,(),None)
        self.assertAlmostEqual(empty['objective'],np.mean(abs(obs.y))/.1)
        self.assertEqual(empty['fit_loss'],'normalized_MAE')
        self.assertEqual(record['record']['certificate_gap'],0.)

    def test_positive_output_floor_refits_with_mae_and_keeps_mse_scoring(self):
        obs,family=data_fixture(3);cfg=replace(configuration('mae'),internal_floor=.08)
        guidance=dict(reference_family=family,pool=family,R=np.eye(3),X=np.eye(3))
        with patch.object(trees,'rnj_guidance',return_value=guidance):
            common=algorithm.prepare_common(obs,cfg)
        result=algorithm.fit(obs,obs,method='rnj',config=cfg,common=common)
        self.assertEqual(result['output_contraction']['status'],'complete')
        self.assertEqual(result['selection_metric'],'validation_mse')
        self.assertEqual(freeze_history(result,obs).current,result['selected']['index'])
        for row in result['frozen_candidates']:
            self.assertEqual(row['point']['loss_name'],'mae')
            self.assertAlmostEqual(row['point']['objective'],
                opt.errors(obs,row['point'])['mae']/common['reference_loss'])

    def test_algorithm_forwards_native_progress_without_new_constructor_arguments(self):
        obs,family=data_fixture(3);cfg=configuration('mse');updates=[]
        class Sink:
            every=1.
            def __call__(self,row):updates.append(row)
        intervals=[]
        common=dict(guidance=dict(R=np.eye(3),X=np.eye(3),pool=trees.basis(3),reference_family=family),
            reference_mse=.1,seconds=0.,n=3,labels=list(obs.labels),
            training_sha256=algorithm._training_fingerprint(obs))
        class FakeSolver:
            def __init__(self,fitter,config):self.records=[]
            def extend(self,current,candidate,warm,**kwargs):
                intervals.append(self.progress_interval)
                self.progress(dict(callback_event='MIP',model_name='unit',scope='one support',
                                   seconds=12.,objective=1.,bound=.5,gap=.5,nodes=2.,incumbent_count=1))
                return None,dict(status=9)
            def close(self):pass
        with patch.object(algorithm,'StructuralSolver',FakeSolver):
            algorithm.fit(obs,obs,method='100',config=cfg,common=common,heartbeat=Sink())
        native=[r for r in updates if r['stage']=='native_progress']
        self.assertEqual(len(native),1)
        self.assertEqual(native[0]['model_name'],'unit')
        self.assertEqual(native[0]['seconds'],12.)
        self.assertIn('elapsed_seconds',native[0])
        self.assertEqual(intervals,[1.])

    def test_invalid_loss_and_common_are_not_silently_treated_as_mse(self):
        obs,_=data_fixture(3)
        for loss in ('MAE','l2','',None):
            with self.subTest(loss=loss),self.assertRaises(ValueError):
                configuration(loss).validate()
            with self.assertRaises(ValueError):opt.QPFitter(obs,loss=loss)
        with self.assertRaisesRegex(ValueError,'different training losses'):
            opt.StructuralSolver(opt.QPFitter(obs),configuration('mae'))

    def test_lp_bad_dual_certificate_is_rejected_and_retained(self):
        obs,family=data_fixture(3)
        real=opt.linprog
        def wrong(*args,**kwargs):
            result=real(*args,**kwargs)
            result.ineqlin.marginals[:]=0.
            return result
        fitter=opt.QPFitter(obs,loss='mae',reference_loss=.1)
        with patch.object(opt,'linprog',side_effect=wrong),self.assertRaisesRegex(RuntimeError,'primal/dual'):
            fitter.fit(family)
        self.assertFalse(fitter.records[-1]['certificate_passed'])
        self.assertNotIn(family,fitter.cache)

    def test_progress_is_scalar_sanitized_throttled_and_nonfatal(self):
        obs,family=data_fixture(3);fitter=opt.QPFitter(obs);point=fitter.fit(family)
        cfg=configuration('mse');solver=opt.StructuralSolver(fitter,cfg)
        names=['MIP','MIPSOL','RUNTIME','MIP_OBJBST','MIP_OBJBND','MIP_SOLCNT','MIP_NODCNT',
               'MIPSOL_OBJBST','MIPSOL_OBJBND','MIPSOL_SOLCNT','MIPSOL_NODCNT','MIPSOL_OBJ']
        codes=SimpleNamespace(**{name:i+1 for i,name in enumerate(names)})
        value=opt.criterion(point,obs.n,cfg.alpha)
        class Model:
            Params=SimpleNamespace(TimeLimit=None)
            ModelName='synthetic_progress';SolCount=1;ObjVal=value;ObjBound=value
            MIPGap=0.;Status=2;Runtime=18.;NodeCount=3.;NumVars=10;NumBinVars=2;NumConstrs=8
            ConstrVio=0.;BoundVio=0.;IntVio=0.
            def optimize(self,callback):
                for event,seconds in [(codes.MIP,0.),(codes.MIP,5.),(codes.MIPSOL,6.),(codes.MIP,17.)]:
                    self.at=seconds
                    callback(self,event)
            def cbGet(self,key):
                if key==codes.RUNTIME:return self.at
                if key in (codes.MIP_OBJBST,codes.MIPSOL_OBJBST):return 1e100 if self.at==0. else value
                if key in (codes.MIP_OBJBND,codes.MIPSOL_OBJBND):return -np.inf if self.at==0. else value
                if key==codes.MIPSOL_OBJ:return value
                if key in (codes.MIP_SOLCNT,codes.MIPSOL_SOLCNT):return 0 if self.at==0. else 1
                return 3.
        def broken(_):raise OSError('mock progress sink unavailable')
        solver.progress=broken
        with patch.object(opt,'_gurobi',return_value=SimpleNamespace(GRB=SimpleNamespace(Callback=codes))):
            _,rec=solver._finish(Model(),perf_counter(),perf_counter()+3.,scope='unit',
                                point_factory=lambda:deepcopy(point),initial_objective=value)
        self.assertEqual([r['callback_event'] for r in rec['progress']],['MIP','MIPSOL','MIP','FINAL'])
        self.assertIsNone(rec['progress'][0]['objective']);self.assertIsNone(rec['progress'][0]['bound'])
        self.assertEqual(len(rec['progress_errors']),4)
        self.assertEqual(rec['initial_to_returned_gain'],0.)
        json.dumps(rec,allow_nan=False)


class SmallNativeLossTests(unittest.TestCase):
    def test_W_start_counts_old_support_even_if_canonical_seed_removed_it(self):
        try:import gurobipy  # noqa: F401
        except ImportError:self.skipTest('Gurobi is not installed')
        obs,family=data_fixture(4);old=(0,1,2);candidate=(0,1)
        for loss in ('mse','mae'):
            with self.subTest(loss=loss):
                fitter=opt.QPFitter(obs,loss=loss,reference_mse=.01,reference_loss=.1)
                current=fitter.fit(trees.canonical([*trees.basis(4),old]));warm=fitter.fit(family)
                self.assertNotIn(list(old),warm['supports'])
                cfg=configuration(loss);solver=opt.StructuralSolver(fitter,cfg)
                try:
                    _,record=solver.extend(current,candidate,warm,deadline=perf_counter()+5.)
                    expected=opt.errors(obs,warm)[loss]/fitter.reference_loss+cfg.alpha*2
                    self.assertAlmostEqual(record['initial_objective'],expected,delta=1e-12)
                    self.assertAlmostEqual(record['initial_objective']-opt.criterion(warm,4,cfg.alpha),cfg.alpha,delta=1e-12)
                finally:solver.close()

    def test_three_and_four_terminal_W_and_E_match_exhaustive_supports(self):
        try:import gurobipy  # noqa: F401
        except ImportError:self.skipTest('Gurobi is not installed')
        for n in (3,4):
            obs,_=data_fixture(n)
            candidates=[c for size in range(2,n) for c in combinations(range(n),size)]
            for loss in ('mse','mae'):
                with self.subTest(n=n,loss=loss):
                    cfg=configuration(loss)
                    fitter=opt.QPFitter(obs,loss=loss,reference_mse=.01,reference_loss=.01 if loss=='mse' else .08)
                    current=fitter.fit(trees.basis(n))
                    wpoints=[current]+[fitter.fit(trees.canonical([*trees.basis(n),c])) for c in candidates]
                    expected_w=min(opt.criterion(p,n,cfg.alpha) for p in wpoints)
                    pool=trees.canonical([*trees.basis(n),*candidates])
                    epoints=[]
                    for count in range(n-1):
                        for chosen in combinations(candidates,count):
                            fam=trees.canonical([*trees.basis(n),*chosen])
                            if trees.is_laminar(fam):epoints.append(fitter.fit(fam))
                    expected_e=min(opt.criterion(p,n,cfg.alpha) for p in epoints)
                    solver=opt.StructuralSolver(fitter,cfg);updates=[];solver.progress=updates.append
                    try:
                        for kind,expected in (('W',expected_w),('E',expected_e)):
                            if kind=='W':
                                raw,record=solver.extend(current,candidates[0],wpoints[1],deadline=perf_counter()+10.)
                            else:
                                raw,record=solver.select(pool,trees.basis(n),current,deadline=perf_counter()+10.)
                            self.assertIsNotNone(raw,record)
                            physical=opt.errors(obs,raw)[loss]/fitter.reference_loss+cfg.alpha*opt.internal_count(raw,n)
                            self.assertAlmostEqual(physical,record['objective'],delta=1e-7)
                            self.assertAlmostEqual(physical,expected,delta=2e-7)
                            self.assertEqual(record['solver'],'Gurobi convex MIQP' if loss=='mse' else 'Gurobi MILP')
                            self.assertEqual(record['progress'][-1]['callback_event'],'FINAL')
                            self.assertFalse(record['progress_errors'])
                            self.assertTrue(record['objective_replay_passed'])
                        self.assertTrue(updates)
                    finally:solver.close()


if __name__=='__main__':unittest.main()
