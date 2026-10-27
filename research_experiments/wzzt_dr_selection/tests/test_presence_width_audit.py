import unittest
from unittest.mock import patch
import numpy as np
from research_experiments.wzzt_dr_selection import presence_width_audit as audit
from research_experiments.wzzt_dr_selection.core import enumerate_laminar_families, all_blocks, fit_l1


def record(P,Q,Y):
    fs=[]
    for b in enumerate_laminar_families(len(P)):
        f=fit_l1(P,Q,Y,b)
        fs.append({'blocks':b,'beta':f.beta.tolist(),'loss':f.loss,'dual_bound':f.dual_bound,'success':f.success})
    cs=[]
    for C in all_blocks(len(P),include_singletons=False):
        omitted=[f for f in fs if C not in f['blocks']]
        lower=min(f['dual_bound'] for f in omitted)
        cs.append({'C':C,'reference_lower':lower,'reference_upper':min(f['loss'] for f in omitted),
                   'reference_requires_block':lower>1e-7})
    return {'history':{'P':P.tolist(),'Q':Q.tolist(),'Y':Y.tolist()},'family_fits':fs,
            'error_budget':0.,'full_reference_complete':True,'candidates':cs,'greedy':{'selected':[(i,) for i in range(len(P))]}}


class PresenceWidthTests(unittest.TestCase):
    def test_maximal_support_compression_is_zero_extension(self):
        P=np.ones((3,1));r=record(P,0*P,2*P)
        eligible,kept=audit.maximal_compatible(r)
        self.assertEqual((len(eligible),len(kept)),(8,3))
        self.assertTrue(all(any(set(audit.family(f))<=set(audit.family(g)) for g in kept) for f in eligible))

    def test_zero_R_effect_does_not_exclude_unobserved_X(self):
        P=np.eye(2);r=record(P,0*P,2*P)
        s=audit.presence(r)['rows'][0]
        self.assertEqual(s['classification'],'no_P_effect_within_tolerance')
        self.assertEqual(s['channels']['X']['status'],'unbounded_verified')
        self.assertAlmostEqual(s['channels']['R']['upper_bound'],0.)

    def test_two_node_sync_and_alternation_widths(self):
        P=np.ones((2,1));r=record(P,0*P,2*P)
        b=((0,),(1,),(0,1));c=(0,1)
        sync=np.array([[1.,0.],[1.,0.]])
        for U,expected in [(sync,1.),(np.eye(2),0.)]:
            w=audit.pair_width(r,b,b,c,U,np.zeros_like(U))
            self.assertEqual(w['status'],'optimal')
            self.assertAlmostEqual(w['value'],expected)
            self.assertAlmostEqual(w['upper_bound'],expected)

    def test_two_node_history_bound_not_artificial(self):
        P=np.ones((2,1));r=record(P,0*P,2*P)
        s=audit.presence(r)['rows'][0]
        self.assertEqual(s['classification'],'P_effect_possible_unresolved')
        self.assertAlmostEqual(s['channels']['R']['value'],1.)
        self.assertEqual(s['channels']['sum']['status'],'unbounded_verified')

    def test_channel_coupling_when_P_equals_Q(self):
        P=np.eye(2);Y=np.array([[2.,.5],[.5,2.]])
        s=audit.presence(record(P,P,Y))['rows'][0]
        self.assertAlmostEqual(s['channels']['R']['value'],.5)
        self.assertAlmostEqual(s['channels']['X']['value'],.5)
        self.assertAlmostEqual(s['channels']['sum']['value'],.5)
        self.assertTrue(s['shared_block_necessary'])

    def test_price_width_matches_independent_row_sum_formula(self):
        P=np.ones((3,1));Y=np.array([[2.26],[2.26],[2.20]])
        r=record(P,0*P,Y);s=audit.presence(r)
        with patch.object(audit,'observed_path',side_effect=AssertionError('truth evaluation reached')):
            d=audit.width_design(r,s)
        expected={'no_reward':1.115,'uniform_early':1.1246,'node0_early':1.1054,
                  'node1_early':1.1054,'node2_early':1.1246,'uniform_late':1.1246}
        for m in d['menus']:self.assertAlmostEqual(m['remaining_width'],expected[m['menu']],places=6)
        self.assertEqual(d['choices']['remaining_width']['ties'],['node0_early','node1_early'])
        self.assertEqual(len(d['choices']['d_opt']['ties']),3)

    def test_failure_keeps_pair_diagnostics(self):
        P=np.ones((3,1));r=record(P,0*P,2*P);s=audit.presence(r)
        failed={'status':'solver_error','value':None,'upper_bound':None,'lp_calls':1}
        with patch.object(audit,'pair_width',return_value=failed):
            with self.assertRaises(audit.AuditFailure) as caught:audit.width_design(r,s)
        self.assertTrue(caught.exception.details['partial_pair_records'])

    def test_empty_target_return_is_complete(self):
        P=np.eye(2);r=record(P,0*P,2*P);screen=audit.presence(r)
        d=audit.width_design(r,screen)
        self.assertEqual(d['status'],'no_P_targets')
        self.assertTrue(d['no_new_data'])
        self.assertEqual(d['choices'],{})

    def test_no_compatible_set_is_not_necessity(self):
        P=np.eye(2);Y=np.array([[1.,2.],[0.,1.]])
        s=audit.presence(record(P,0*P,Y))
        self.assertEqual(s['rows'][0]['classification'],'model_budget_incompatible')
        self.assertFalse(s['rows'][0]['P_price_candidate'])
        d=audit.width_design(record(P,0*P,Y),s)
        self.assertEqual(d['status'],'model_budget_incompatible')
        self.assertIsNone(d['no_new_data'])
        self.assertTrue(all(c['status']=='model_budget_incompatible' and c['value'] is None for c in s['rows'][0]['channels'].values()))

if __name__=='__main__':unittest.main()
