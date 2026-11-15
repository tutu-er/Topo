"""Exact row filtering and historical-only strict selection tests."""
import unittest
from copy import deepcopy

import numpy as np

from ...dr.selection_mse import choose_mse
from ..distribution_gate_20261114.evidence import DistributionEvidence
from ..distribution_gate_20261114.test_evidence import toy
from .evidence import SingleTargetEvidence, historical_target


def measured_toy():
    case, score, val, forecast, rng = toy()
    power = np.column_stack((val.p, val.q))
    for i, p in enumerate(score['native_bank']):
        p['history_sse'] = float(np.square(val.y-power@np.asarray(p['H']).T).mean())*val.y.size
        p['internal'] = i+1
    for p in score['parameters']:
        p['history_sse'] = score['native_bank'][p['index']]['history_sse']
    score['native_index'] = choose_mse(
        [p['history_sse']/score['cells'] for p in score['native_bank']],
        [p['internal'] for p in score['native_bank']],
        K=1/score['reference_mse'], epsilon_scaled=score['validation_tie'])['selected']
    assert score['native_index'] == 0
    return case, score, val, forecast, rng


class SelectionTests(unittest.TestCase):
    def test_matches_original_rows_for_every_law_and_action(self):
        case, score, val, forecast, rng = measured_toy()
        for mode in ('normal_fixed','mgf_fixed','mgf_forecast'):
            full = DistributionEvidence(case,score,val,forecast,200.,mode=mode,target_mass=1.)
            one = SingleTargetEvidence(case,score,val,forecast,200.,mode=mode)
            rows = [k for k,r in enumerate(full.rows) if r['target_native']==score['native_index']]
            guards = [k for k,g in enumerate(full.guards) if g[0]==one.targets[0]]
            self.assertEqual(one.rows,[full.rows[k] for k in rows])
            self.assertEqual(one.metadata['pair_count'],len(rows))
            for _ in range(4):
                u=rng.uniform(case.lower,case.upper)
                base,got=full.verify_expansion(u),one.verify_expansion(u)
                for field in ('mu','variance','margins','pair_failure','mean_log_score_minus_tie_under_declared_law'):
                    np.testing.assert_allclose(got[field],base[field][rows],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(got['guard_gains'],np.asarray(base['guard_gains'])[guards],rtol=1e-12,atol=1e-12)
                self.assertEqual(set(got['target_failure_sums']),{str(one.targets[0])})
                self.assertAlmostEqual(got['target_failure_sums'][str(one.targets[0])],
                                       base['target_failure_sums'][str(one.targets[0])],places=13)
            self.assertEqual(one.selection['retained_forward_guards'],len(score['parameters'])-1)
            self.assertTrue(all(g[0]==one.targets[0] for g in one.guards))

    def test_every_foreign_native_point_kept_without_pruning(self):
        case,score,val,forecast,_=measured_toy()
        # Native3 is another frozen parameter point of target0's own family.
        # Switch the winner to representative1: both native0 and3 must survive.
        val.y=np.column_stack((val.p,val.q))@np.asarray(score['native_bank'][1]['H']).T
        for i,p in enumerate(score['native_bank']):
            p['history_sse']=float(np.square(val.y-np.column_stack((val.p,val.q))@np.asarray(p['H']).T).mean())*val.y.size
        for p in score['parameters']:p['history_sse']=score['native_bank'][p['index']]['history_sse']
        score['native_index']=1
        one=SingleTargetEvidence(case,score,val,forecast,200.)
        self.assertEqual([r['competitor_native'] for r in one.rows],[0,2,3])
        self.assertTrue(all(r['alpha']==.1/3 for r in one.rows))

    def test_tie_uses_internal_count_then_native_order(self):
        case,score,val,forecast,_=measured_toy()
        common=deepcopy(score['native_bank'][0]['H'])
        sse=score['native_bank'][0]['history_sse']
        for p in score['native_bank']:
            p.update(H=deepcopy(common),history_sse=sse,internal=3)
        for p in score['parameters']:p.update(H=deepcopy(common),history_sse=sse)
        score['native_bank'][1]['internal']=score['native_bank'][2]['internal']=1
        score['native_index']=1
        one=SingleTargetEvidence(case,score,val,forecast,200.)
        self.assertEqual(one.selection['selected_native'],1)
        self.assertEqual(one.selection['eligible_native_indices'],[0,1,2,3])
        self.assertEqual(one.targets,[1])

    def test_nonrepresentative_winner_fails_without_substitution(self):
        case,score,val,forecast,_=measured_toy()
        score['native_bank'][3].update(H=deepcopy(score['native_bank'][0]['H']),
            history_sse=score['native_bank'][0]['history_sse'],internal=0)
        score['native_index']=3
        with self.assertRaisesRegex(ValueError,'not a frozen representative'):
            SingleTargetEvidence(case,score,val,forecast,200.)

    def test_bad_declared_winner_or_changed_history_fails(self):
        case,score,val,forecast,_=measured_toy()
        wrong=deepcopy(score);wrong['native_index']=1
        with self.assertRaisesRegex(ValueError,'strict historical'):
            historical_target(wrong,val)
        changed=deepcopy(val);changed.y=changed.y+.01
        with self.assertRaisesRegex(ValueError,'validation SSE'):
            historical_target(score,changed)

    def test_input_score_and_observations_are_not_mutated(self):
        case,score,val,forecast,_=measured_toy()
        before=deepcopy(score); y=val.y.copy(); power=forecast.copy()
        SingleTargetEvidence(case,score,val,forecast,200.)
        self.assertEqual(before,score)
        np.testing.assert_array_equal(y,val.y)
        np.testing.assert_array_equal(power,forecast)


if __name__=='__main__':unittest.main()
