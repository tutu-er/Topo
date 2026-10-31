"""Synthetic source-schema counterexamples, not new topology experiments."""
from copy import deepcopy
from itertools import combinations
import unittest

try:
    from . import history_reference as h
except ImportError:
    import history_reference as h


def point(supports, r, x=None, objective=0., **metadata):
    return dict(supports=deepcopy(supports), r=list(r), x=[0.]*len(r) if x is None else list(x),
                objective=objective, validation_mae=objective, **metadata)


def fixture():
    leaves = [[0], [1], [2]]
    family = [*leaves, [0, 1]]
    bank = point(leaves, [.1, .1, 0.], objective=1., native_family=family, config='raw_RX75_0.16', origin='tree_bank')
    start = point(leaves, [.1, .1, 0.], objective=1.)
    after = point([*leaves, [0, 2]], [.1, .1, 0., .2], objective=.9)
    search = dict(path=[start, after], reason='pool_local_optimum', decisions=[
        dict(candidates=[dict(candidate=[0, 2], removed=[], objective=.9),
                         dict(candidate=[1, 2], removed=[], objective=.9+5e-13)]),
        dict(candidates=[])])
    selections = [deepcopy(bank), dict(deepcopy(start), origin='tree_initialized_correction', path_index=0),
                  dict(deepcopy(after), origin='tree_initialized_correction', path_index=1)]
    return dict(status='complete', truth_input_allowed=False, final_strategy='multistart_original_rule',
                terminal_labels=[101, 102, 103], root_label=1, input_sha256=dict(training='train-hash', validation='val-hash'),
                source_sha256={'engine.py': 'source-hash'}, fingerprint='synthetic-only',
                settings=dict(coefficient_bound=2., final_validation_fraction=.03, resolution_factor=.16),
                training_rows=4, validation_rows=4, bank=[bank], searches={'primary': search},
                selection_candidates=selections)


class ReferenceTests(unittest.TestCase):
    def test_discarded_weak_and_zero_are_not_recovered_as_exact_zero(self):
        # Both original values 0 and 5e-10 lead to this same saved result.
        packet = h.extract_reference(fixture())
        unknown = packet['references'][0]['unavailable_coefficients']
        self.assertEqual(unknown[0]['support'], (0, 1))
        self.assertEqual(unknown[0]['r_interval'], [0., 1e-9])
        self.assertTrue(unknown[0]['r_interval'][0] <= 5e-10 <= unknown[0]['r_interval'][1])
        self.assertFalse(packet['references'][0]['exact_full_parameter_point_available'])
        self.assertFalse(packet['risk_ready'])

    def test_next_path_uses_pruned_current_not_previous_complete_container(self):
        packet = h.extract_reference(fixture())
        row = packet['references'][-1]
        self.assertIn((0, 2), row['complete_container'])
        self.assertNotIn((0, 1), row['complete_container'])
        self.assertEqual(row['unavailable_coefficients'], [])

    def test_cross_family_deletion_is_reconstructed(self):
        source = fixture()
        for p in (source['bank'][0], source['searches']['primary']['path'][0],
                  source['selection_candidates'][0], source['selection_candidates'][1]):
            p['supports'].append([0, 1]); p['r'].append(.3); p['x'].append(0.)
        source['searches']['primary']['decisions'][0]['candidates'][0]['removed'] = [[0, 1]]
        source['searches']['primary']['decisions'][0]['candidates'][1]['removed'] = [[0, 1]]
        recovered = h.recover_path_containers(source)
        self.assertNotIn((0, 1), recovered[('primary', 1)]['family'])
        self.assertIn((0, 2), recovered[('primary', 1)]['family'])

    def test_ordered_tie_rule_does_not_use_numerically_smaller_near_tie(self):
        source = fixture()
        source['searches']['primary']['decisions'][0]['candidates'][1]['objective'] = .9-5e-13
        family = h.recover_path_containers(source)[('primary', 1)]['family']
        self.assertIn((0, 2), family)
        self.assertNotIn((1, 2), family)

    def test_same_supports_different_weights_are_distinct_points(self):
        source = fixture()
        another = deepcopy(source['bank'][0]); another['config'] = 'raw_R_0'; another['r'][0] = .2
        source['bank'].append(another); source['selection_candidates'].append(deepcopy(another))
        packet = h.extract_reference(source)
        ids = [row['numerical_point_id'] for row in packet['references']]
        self.assertNotEqual(ids[0], ids[-1])
        self.assertEqual(ids[0], ids[1])

    def test_zero_singleton_is_preserved_but_not_positive(self):
        row = h.extract_reference(fixture())['references'][0]
        self.assertIn((2,), row['saved_supports'])
        self.assertIn((2,), row['retained_exact_zero_numerical_supports'])
        self.assertNotIn((2,), row['known_positive_numerical_supports'])

    def test_no_validation_or_contraction_filter_is_applied(self):
        source = fixture()
        source['selection_candidates'][-1]['validation_mae'] = 100.
        packet = h.extract_reference(source)
        self.assertEqual(len(packet['references']), len(source['selection_candidates']))
        self.assertEqual(packet['references'][-1]['historical_validation_mae'], 100.)
        self.assertFalse(packet['algorithm_bound_is_physical_prior'])
        self.assertFalse(packet['topology_posterior'])
        self.assertFalse(packet['continuous_parameter_coverage'])

    def test_selection_remeasured_and_path_solver_objectives_are_kept_separate(self):
        source = fixture()
        source['searches']['primary']['path'][1]['solver_objective'] = .9
        source['selection_candidates'][2]['objective'] = .90000000002
        source['selection_candidates'][2]['solver_objective'] = .9
        packet = h.extract_reference(source)
        row = packet['references'][2]
        self.assertEqual(row['selection_saved_point_objective'], .90000000002)
        self.assertEqual(row['source_path_objective'], .9)
        self.assertEqual(row['solver_objective'], .9)

    def test_fixed_prediction_loss_bounds_match_every_subset_of_unknowns(self):
        known = [(0,), (1,)]
        unknown = [(0, 1), (0, 1, 2)]
        predictions = [[], [(0,)], [(0,), (0, 1)], [(0,), (1,), (0, 1), (0, 1, 2)]]
        for predicted in predictions:
            errors = []
            for size in range(len(unknown)+1):
                for chosen in combinations(unknown, size):
                    errors.append(len(set(predicted) ^ (set(known) | set(chosen))))
            bounds = h.fixed_prediction_loss_bounds(predicted, known, unknown)
            self.assertEqual(bounds['lower'], min(errors))
            self.assertEqual(bounds['upper'], max(errors))
            self.assertFalse(bounds['bounds_future_input_or_algorithm_changes'])

    def test_full_fit_preserves_a_positive_weight_below_pruning_threshold(self):
        source = fixture()
        full = dict(supports=[[0], [1], [2], [0, 1]], r=[.1, .1, 0., 5e-10], x=[0.]*4,
                    weight_upper_bound=2., support_prune_tolerance=1e-9,
                    semantics='full clipped solver coefficients before numerical support pruning')
        source['bank'][0]['full_fit'] = deepcopy(full)
        source['selection_candidates'][0]['full_fit'] = deepcopy(full)
        packet = h.extract_reference(source)
        row = packet['references'][0]
        self.assertEqual(row['unavailable_coefficients'], [])
        self.assertIn((0, 1), row['known_positive_numerical_supports'])
        self.assertTrue(row['exact_full_parameter_point_available'])
        self.assertNotIn((0, 1), row['saved_supports'])
        self.assertFalse(packet['risk_ready'])

    def test_inconsistent_full_fit_does_not_silently_replace_saved_weights(self):
        source = fixture()
        source['selection_candidates'][0]['full_fit'] = dict(
            supports=[[0], [1], [2], [0, 1]], r=[.2, .1, 0., 0.], x=[0.]*4,
            weight_upper_bound=2., support_prune_tolerance=1e-9, semantics='invalid mismatch')
        with self.assertRaises(ValueError):
            h.extract_reference(source)

    def test_inconsistent_cross_record_and_bad_source_raise(self):
        source = fixture(); source['searches']['primary']['decisions'][0]['candidates'][0]['removed'] = [[0, 1]]
        with self.assertRaises(ValueError):
            h.extract_reference(source)
        source = fixture(); source['truth_input_allowed'] = True
        with self.assertRaises(ValueError):
            h.extract_reference(source)

    def test_duplicate_path_source_cannot_hide_a_missing_step_at_same_row_count(self):
        source = fixture()
        source['selection_candidates'][2] = deepcopy(source['selection_candidates'][1])
        self.assertEqual(len(source['selection_candidates']), 3)
        with self.assertRaisesRegex(ValueError, 'provenance coverage'):
            h.extract_reference(source)


if __name__ == '__main__':
    unittest.main()
