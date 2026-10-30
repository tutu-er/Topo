"""Contract tests only: no Gurobi environment, fit, calibration or simulation."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('reference_noise_driver',Path(__file__).with_name('reference_noise_driver.py'))
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)


def current():
    trees=[dict(family=[[0],[1],[2],[i,j]],config=f'locked_{k}')
        for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]*5)]
    cal=dict(a=1.,b=1.,median_root_depth=.5)
    return dict(terminal_labels=[101,102,103],native_bank=trees,
        pool=driver.base.canonical([s for t in trees for s in t['family']]),
        calibration=cal,original_calibrations=dict(raw=cal),fronts={})


SETTINGS=dict(max_n=3,coefficient_bound=2.,calibration_snapshot='current_observed',
    wall_seconds=100.,search_seconds=10.,max_LP_calls=1000,root_label=1)


class NoRead:
    def __iter__(self):raise AssertionError('truth read')
    def __getitem__(self,key):raise AssertionError('truth read')


class ReferenceNoisePureTests(unittest.TestCase):
    def test_native15_order_and_zero_optional_support(self):
        state=current();before=copy.deepcopy(state)
        families=driver.augmented_native_families(state)
        self.assertEqual(len(families),15)
        for tree,family in zip(state['native_bank'],families):
            self.assertIn((0,1,2),family)
            self.assertEqual(set(family)-{(0,1,2)},set(tuple(s) for s in tree['family']))
        self.assertEqual(state,before)
        self.assertEqual(len(set(families)),3)

    def test_original_PoolLP_domain_has_zero_lower_bounds(self):
        path=driver.base.REPO/'research_experiments/wzzt_dr_selection/dr_pool_v2/engine.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PoolLP')
        init=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        weights=next(n.value for n in ast.walk(init) if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Attribute) and t.attr=='weights' for t in n.targets))
        kws={k.arg:k.value for k in weights.keywords}
        self.assertEqual(ast.literal_eval(kws['lb']),0)
        self.assertEqual(ast.literal_eval(kws['ub']),0)
        fit=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
        attributes=[t.attr for n in ast.walk(fit) if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Attribute)]
        self.assertIn('UB',attributes);self.assertNotIn('LB',attributes)
        source=ast.unparse(fit)
        self.assertIn('upper[active] = upper[np.asarray(active) + k] = self.bound',source)
        runner=ast.parse(driver.BASE_DRIVER_PATH.read_text(encoding='utf-8'))
        method=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='_run_method')
        calls=[n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='PoolLP']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.literal_eval(next(k.value for k in calls[0].keywords if k.arg=='bound')),2.)

    def test_same_current_calibration_and_observations_no_truth(self):
        state=current();state['truth']=NoRead();train=object();valid=object();budget=object()
        calls=[]
        def fake(name,history,t,v,pool,settings,b,**kw):
            calls.append((name,history,t,v,pool,settings,b,kw))
            return dict(status='complete',LP_solves=[],points=[],selected={})
        with patch.object(driver,'_BASE_METHOD',fake):
            result=driver.run_augmented(state,train,valid,SETTINGS,budget,day_ids=['observed'])
        name,history,t,v,pool,settings,b,kw=calls[0]
        self.assertEqual(name,'current_root_augmented_rnj')
        self.assertIs(history,state);self.assertIs(t,train);self.assertIs(v,valid);self.assertIs(b,budget)
        self.assertIs(history['calibration'],state['calibration'])
        self.assertIs(kw['current_state'],state)
        self.assertEqual(kw['families'],driver.augmented_native_families(state))
        self.assertIn((0,1,2),pool);self.assertFalse(result['full_root_positive_weight_required'])
        self.assertEqual(result['coefficient_lower_bound'],0.)
        self.assertEqual(result['coefficient_upper_bound'],2.)
        self.assertFalse(result['shared_cross_method_LP_cache'])

    def test_reject_domain_and_calibration_changes_without_runner(self):
        with patch.object(driver,'_BASE_METHOD',side_effect=AssertionError('runner called')):
            for key,value in [('max_n',4),('coefficient_bound',3.),('calibration_snapshot','historical')]:
                altered=dict(SETTINGS,**{key:value})
                with self.assertRaises(ValueError):driver.run_augmented(current(),None,None,altered,None)

    def test_prepare_freezes_ninth_method_without_opening_packets(self):
        # Keep creation and automatic cleanup inside the explicitly writable stage.
        with tempfile.TemporaryDirectory(prefix='pure_freeze_',dir=Path(__file__).resolve().parent) as temp:
            directory=Path(temp);packet=directory/'observed.pkl';packet.write_bytes(b'not a pickle: prepare never reads observations')
            config=dict(status='root_frozen_before_solver',settings=SETTINGS,
                histories=[dict(id='h',path=str(packet),sha256=driver.base.sha(packet))],snapshots=[])
            config_path=directory/'config.json';config_path.write_text(json.dumps(config))
            source_hashes={str(driver.BASE_DRIVER_PATH):driver.base.sha(driver.BASE_DRIVER_PATH)}
            with patch.object(driver.base,'_fixed_source_hashes',return_value=source_hashes):
                frozen=driver.prepare(directory/'execution',config_path)
            self.assertEqual(len(frozen['methods']),9)
            self.assertEqual(frozen['methods'].count(driver.METHOD),1)
            for method in ('current_rebuilt_rnj','current_pool_wzzt','all_legal_family_reference'):
                self.assertIn(method,frozen['methods'])
            self.assertEqual(frozen['source_sha256'][str(Path(driver.__file__).resolve())],driver.base.sha(driver.__file__))
            self.assertEqual(driver.verify(directory/'execution')['root_augmented_rnj']['root_lower_bound'],0.)
            self.assertEqual(driver.base.sha(driver.BASE_DRIVER_PATH),driver.BASE_DRIVER_SHA256)


if __name__=='__main__':unittest.main()
