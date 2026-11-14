"""Historical-only catalogue and unchanged resource-domain checks; no solver/AC."""
import json
import unittest

import numpy as np

from .resources import build_resources, choose_focus, tier, CONTRACT_TIERS
from ..reactive_flex_20261114.domain import PQIncrementCase


def fixture():
    labels = [411, 412, 413, 414]
    context = dict(node_mapping=[dict(label=label, feeder_id=k+1, scale=1.) for k, label in enumerate(labels)],
        resource_metadata_at_history_start=[dict(feeder=k+1, date='2024-01-01',
            PV_systems_kW=pv, batteries_kW=0., EV_chargers_kW=ev, hot_water_tanks_kW=thermal)
            for k, (pv, ev, thermal) in enumerate(zip([30.,20.,10.,5.], [1.,4.,3.,2.], [2.,1.,5.,4.]))])
    train = dict(labels=np.asarray(labels), P=np.ones((192, 4))*.05,
                 Q=np.tile([.004, -.008, .012, -.022], (192, 1)))
    H0 = np.zeros((4, 8)); H1 = np.c_[np.zeros((4,4)), np.diag([1.,4.,2.,3.])]
    score = dict(labels=labels, native_bank=[dict(H=H0.tolist(), topology_id='a'), dict(H=H1.tolist(), topology_id='b')],
                 models=[dict(index=0, probability=.6), dict(index=1, probability=.4)])
    return dict(id='toy', start_date='2024-01-04'), score, context, train


class ResourceChecks(unittest.TestCase):
    def build(self, *args, base=100.):
        result = build_resources(*args, base)
        self.assertEqual(result['status'], 'ready', result['problems'])
        json.dumps(result, allow_nan=False)
        return result

    def test_units_and_quantile_use_supplied_base(self):
        entry, score, context, train = fixture()
        score['native_bank'] = score['native_bank'][:1]
        score['models'] = [dict(index=0, probability=1.)]
        train['Q'][:, 0] = -.015
        a = self.build(entry, score, context, train, base=100.)
        b = self.build(entry, score, context, train, base=200.)
        self.assertEqual(a['q_resources'][0]['capacity_kvar'], 2.)
        self.assertEqual(b['q_resources'][0]['capacity_kvar'], 3.)
        self.assertEqual(a['nodes'][0]['training_abs_Q95_kvar'], 1.5)
        self.assertEqual(a['placement']['node_labels'], [])
        self.assertEqual(a['placement']['reason'], 'no_foreign_topology')
        self.assertNotEqual(a['input_hashes']['base_kw'], b['input_hashes']['base_kw'])

    def test_frozen_placement_relabels_without_node_specific_floor(self):
        entry, score, context, train = fixture()
        result = self.build(entry, score, context, train)
        self.assertEqual(result['placement']['node_labels'], [412, 414])
        self.assertEqual([r['capacity_kvar'] for r in result['q_resources']], [1., 5., 2., 5.])
        self.assertTrue(all(r['capacity_kvar'] <= .8*r['assumed_equipment_rating_kvar'] for r in result['q_resources']))
        relabel = [29, 8, 102, 77]
        score['labels'] = relabel; train['labels'] = np.asarray(relabel)
        for row, label in zip(context['node_mapping'], relabel):
            row['label'] = label
        changed = self.build(entry, score, context, train)
        self.assertEqual(changed['placement']['node_labels'], [8, 77])

    def test_maximin_keeps_weakest_rival_and_deterministic_tie(self):
        energy = np.array([[10.,0.,0.], [0.,6.,6.]])
        selected = choose_focus(energy, [0,1,2], [301,302,303])
        self.assertEqual(selected['node_labels'], [301,302])
        self.assertEqual(selected['minimum_pair_sensitivity'], 6.)
        self.assertEqual(selected['combinations'], 3)

    def test_raw_same_family_parameter_points_are_all_compared(self):
        entry, score, context, train = fixture()
        score['native_bank'].append(dict(H=(np.asarray(score['native_bank'][1]['H'])*.1).tolist(), topology_id='b'))
        result = self.build(entry, score, context, train)
        self.assertEqual(result['placement']['native_pairs'], [[0,1],[0,2]])
        self.assertEqual(result['placement']['target_native_indices'], [0,1])

    def test_eligibility_and_less_than_two_focus_sites(self):
        entry, score, context, train = fixture()
        for row in context['resource_metadata_at_history_start'][1:]:
            row['PV_systems_kW'] = 0.
        context['resource_metadata_at_history_start'][1]['batteries_kW'] = .2
        result = self.build(entry, score, context, train)
        self.assertEqual(result['placement']['node_labels'], [411])
        self.assertEqual([r['node_label'] for r in result['q_resources']], [411,412])
        self.assertEqual([r['capacity_kvar'] for r in result['q_resources']], [5.,1.])

    def test_P_top_counts_scaling_and_common_price_schema(self):
        entry, score, context, train = fixture()
        context['node_mapping'][0]['scale'] = .1
        result = self.build(entry, score, context, train)
        rows = result['p_resources']
        self.assertEqual([r['node_label'] for r in rows if r['kind']=='pv'], [412,413,414])
        self.assertEqual([r['node_label'] for r in rows if r['kind']=='ev'], [412,413])
        self.assertEqual([r['node_label'] for r in rows if r['kind']=='thermal'], [413,414])
        self.assertTrue(all(r['contract_kw']==.1*r['scaled_registered_kw'] for r in rows))

    def test_ignored_future_truth_and_voltage_cannot_change_catalogue(self):
        entry, score, context, train = fixture()
        a = self.build(entry, score, context, train)
        entry.update(truth={'bad':'future'}, physical_cache={'path':'do-not-open'})
        score['true_index'] = 99; context['future_registration'] = {'PV_systems_kW':1e10}
        train['Y'] = np.full((192,4),np.nan); train['V'] = np.full((192,4),np.inf)
        b = self.build(entry, score, context, train)
        self.assertEqual(a['configuration_sha256'], b['configuration_sha256'])
        self.assertEqual(a['input_hashes'], b['input_hashes'])

    def test_overflow_and_future_registration_are_unresolved(self):
        entry, score, context, train = fixture()
        train['Q'][:,0] = .3
        result = build_resources(entry, score, context, train, 100.)
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['problems'][0]['node_label'], 411)
        self.assertIn('catalogue', result['problems'][0]['error'])
        context['resource_metadata_at_history_start'][0]['date'] = '2024-01-05'
        result = build_resources(entry, score, context, train, 100.)
        self.assertEqual(result['status'], 'unresolved')
        self.assertIn('history start', result['problems'][0]['error'])
        self.assertRaises(ValueError, tier, 20.01, CONTRACT_TIERS)

    def test_PQ_domain_units_time_grid_and_Q_only_delivery(self):
        entry, score, context, train = fixture()
        result = self.build(entry, score, context, train)
        entry.update(p_resources=result['p_resources'], q_resources=result['q_resources'])
        forecast = np.tile(np.r_[np.full(4,-.01),np.zeros(4)], (96,1))
        case = PQIncrementCase(context, train['labels'], forecast, 100., q_resources=result['q_resources'], fraction=.1)
        self.assertEqual(case.resources[:case.P_R], result['p_resources'])
        self.assertEqual(case.P_R, 7); self.assertEqual(case.Q_R, 4)
        self.assertEqual(case.shape, (24,11)); self.assertEqual(case.dt, 1.)
        u = np.zeros(case.shape); u[2,case.P_R] = -1.
        self.assertTrue(case.audit(u)['passed'])
        p, q = case.aggregate(case.delivery(u))
        self.assertTrue(np.array_equal(p, np.zeros((96,4))))
        self.assertTrue(np.array_equal(q[8:12,0], np.full(4,-1.)))
        self.assertEqual(float(q.sum()), -4.)  # four quarter-hour readings, one kvarh
        self.assertAlmostEqual(case.fee(u), .05)
        self.assertAlmostEqual(case.amounts(u)['reactive_abs_kvarh'], 1.)
        self.assertTrue(np.array_equal(case.baseline, np.zeros(case.shape)))


if __name__ == '__main__':
    unittest.main()
