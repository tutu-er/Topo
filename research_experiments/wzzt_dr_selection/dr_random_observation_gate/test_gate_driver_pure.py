"""Combinatorial/input contract tests only. No engine model is instantiated."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

SPEC=importlib.util.spec_from_file_location('small_gate',Path(__file__).with_name('gate_driver.py'))
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


class GatePureTests(unittest.TestCase):
    def test_complete_n3_and_gated_n4_family_counts(self):
        self.assertEqual(len(gate.legal_families(3)),8)
        self.assertEqual(len(gate.legal_families(4)),52)

    def test_leaf_supports_always_in_every_family(self):
        for n in (3,4):
            leaves={(j,) for j in range(n)}
            for family in gate.legal_families(n):
                self.assertTrue(leaves<=set(family))
                self.assertEqual(gate.validate_supports(family,n,family=True),family)

    def test_crossing_candidates_allowed_but_direct_tree_not_silently_repaired(self):
        c=[(0,1),(1,2)]
        self.assertEqual(len(gate.validate_supports(c,3)),2)
        with self.assertRaises(ValueError):gate.validate_supports(c,3,family=True)

    def test_small_same_pool_reference_is_restricted_to_declared_supports(self):
        pool=[(0,),(1,),(2,),(0,1),(0,1,2)]
        families=gate.legal_families(3,pool)
        self.assertEqual(len(families),4)
        self.assertTrue(all(set(family)<=set(pool) for family in families))
        with self.assertRaises(ValueError):gate.validate_supports([(0,3)],3)

    def test_discovery_arrays_use_real_observed_contract_and_ignore_validation(self):
        import numpy as np
        from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import scene
        p=np.full((96,3),.02);q=np.full((96,3),.007);v=np.full((96,3),.99);root=np.ones(96)
        old=scene(p,q,v,root,[101,102,103],'pure_old',0)
        future=scene(p*1.1,q*1.1,v,root,[101,102,103],'pure_future',1)
        class Poison:
            def __iter__(self):raise AssertionError('validation was read by candidate builder')
        observed=gate.observed_training_for_discovery(dict(training=[old],validation=Poison()),dict(training=[future],validation=Poison()))
        self.assertEqual(set(observed),{'history_train','future_train'})
        np.testing.assert_allclose(observed['history_train']['P'],p)
        np.testing.assert_allclose(observed['future_train']['Q'],q*1.1)
        np.testing.assert_allclose(observed['future_train']['Y'],root[:,None]**2-v**2)

    def test_current_snapshot_calibration_pool_and_direct_root_are_shared(self):
        import numpy as np
        from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import scene
        p=np.full((96,3),.02);q=np.full((96,3),.007);v=np.full((96,3),.99);root=np.ones(96)
        def one(name,offset):return scene(p,q,v,root,[101,102,103],name,offset)
        old_packet=dict(training=[one('old_train',0)],validation=[one('old_val',3)])
        future=dict(training=[one('new_train',6)],validation=[one('new_val',8)])
        leaves=[(0,),(1,),(2,)]
        old=dict(terminal_labels=[101,102,103],calibration={'marker':'old'},old_pool=leaves+[(0,1)],native_bank=[],starts=[])
        current=dict(status='complete',terminal_labels=[101,102,103],calibration={'marker':'current'},
                     pool=leaves+[(1,2)],old_pool=leaves+[(1,2)],native_bank=[],starts=[],seconds=0.,
                     fronts={},original_calibrations={})
        calls=[]
        def method(name,state,train,valid,pool,settings,budget,**kwargs):
            calls.append(dict(name=name,cal=state['calibration'],pool=set(pool),direct=kwargs['direct_family'],
                              current_state=kwargs['current_state']))
            return dict(status='complete',LP_solves=[],selected={})
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(gate,'prepare_observed_state',return_value=current) as prepared,patch.object(gate,'_run_method',side_effect=method):
                result=gate.identify_snapshot(old_packet,future,old,
                    dict(candidate_supports=[(0,2)],direct_family=[(0,2)]),
                    dict(max_n=3),gate.Budget(10,100),Path(temporary)/'result.json')
            self.assertEqual(prepared.call_count,1)
        self.assertEqual(result['calibration_snapshot'],'current_observed')
        self.assertEqual(result['new_supports'],((0,2),))
        byname={c['name']:c for c in calls}
        self.assertIn('current_pool_wzzt',byname)
        self.assertIn('current_rebuilt_rnj',byname)
        self.assertTrue(all(c['cal']=={'marker':'current'} for c in calls))
        self.assertTrue(all(c['current_state'] is current for c in calls))
        self.assertEqual(byname['old_pool_wzzt']['pool'],set(old['old_pool']))
        self.assertEqual(byname['current_pool_wzzt']['pool'],set(current['pool']))
        self.assertEqual(byname['expanded_pool_wzzt']['pool'],set(current['pool'])|{(0,2)})
        self.assertIn((0,1,2),byname['direct_control_tree_refit']['direct'])


if __name__=='__main__':unittest.main()
