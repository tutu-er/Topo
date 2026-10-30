import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from confirm import POLICIES,RULES,METHODS,build_actions,make_jobs,summarized,selector_replay

class Menus:
    @staticmethod
    def resolve(name):return dict(id=name)
    @staticmethod
    def baseline_menu(name,seed):return dict(id=f'{name}-{seed}')
class Budget:
    calls=0;started=0

class ConfirmationTests(unittest.TestCase):
    def test_prices_depend_on_predeclared_seeds_only(self):
        flows=[dict(id='r0',baseline_seed=11),dict(id='r1',baseline_seed=12)];before=copy.deepcopy(flows)
        actions=build_actions(flows,Menus);self.assertEqual(flows,before)
        self.assertEqual(actions[0]['menus']['zero']['id'],'zero')
        self.assertEqual(actions[0]['menus']['pair_rotation']['id'],'pair_rotation-14')
        self.assertEqual(actions,build_actions(flows,Menus))
    def test_seventy_predeclared_jobs_are_unique_without_truth_values(self):
        a=build_actions([dict(id='r0',baseline_seed=1),dict(id='r1',baseline_seed=2)],Menus)
        cases=[dict(id=str(i)) for i in range(7)]
        jobs=make_jobs(a,cases,934431);self.assertEqual(len(jobs),70)
        self.assertEqual(jobs,make_jobs(a,cases,934431));self.assertTrue(all(set(j)=={'flow','case','policy'} for j in jobs))
    def test_duplicate_jobs_rejected(self):
        a=build_actions([dict(id='r0',baseline_seed=1)],Menus)
        with self.assertRaises(ValueError):make_jobs(a,[dict(id='x'),dict(id='x')],1)
    def test_unstarted_results_never_become_zero_error_successes(self):
        rows=[dict(flow='r0',case='zero',policy='zero',execution_status='not_executed')]
        s=summarized(rows,{},Budget());self.assertEqual(s['status'],'bounded_finished_with_unresolved')
        self.assertTrue(all(e['complete']==0 and e['unresolved_or_unstarted']==14 for e in s['results']))
        self.assertTrue(all(p['strict_vs_original']=={'unresolved':1} for p in s['paired']))
    def test_counterfactual_cost_is_not_summed_seven_times(self):
        rows=[dict(flow='r0',case=str(i),policy='single2',execution_status='executed',actual_payment=1.2) for i in range(7)]
        s=summarized(rows,{},Budget());self.assertEqual(s['payments_per_four_event_deployment'],{'r0|single2':1.2})
        rows[-1]['actual_payment']=1.3
        with self.assertRaises(RuntimeError):summarized(rows,{},Budget())
    def test_selector_signature_does_not_accept_truth(self):
        import inspect
        self.assertEqual(list(inspect.signature(selector_replay).parameters),['method','observations','cal','rules'])
        self.assertEqual(selector_replay({'status':'not_started_budget'},None,None,None),{'status':'not_started_budget'})

if __name__=='__main__':unittest.main()
