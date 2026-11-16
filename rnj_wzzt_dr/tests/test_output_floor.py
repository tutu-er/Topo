"""Native output-floor checks on small fixed-support examples; no MIQP or AC."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..observations import Observation
from ..identification import algorithm, output_floor as floor, trees
from ..identification.optimization import QPFitter, response_matrices


def point(supports, r, x=None):
    return dict(supports=deepcopy(supports), r=list(r), x=[0.] * len(r) if x is None else list(x))


def observations(p, n=None):
    n = n or max(max(C) for C in p['supports'])+1
    design = np.eye(2*n)
    R, X = response_matrices(p, n)
    P, Q = design[:, :n], design[:, n:]
    return Observation(P, Q, P@R.T+Q@X.T, tuple(range(101, 101+n)))


def calibration(n):
    return dict(a=1., b=1., m=1., median_root_depth=1., terminal_count=n)


def fit_for(p, n=None):
    return QPFitter(observations(p, n), reference_mse=.1, qp_seconds=10.)


class OutputFloorTests(unittest.TestCase):
    def test_default_and_invalid_floor(self):
        self.assertEqual(algorithm.SearchConfig().internal_floor, .08)
        for value in (-.01, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(algorithm.SearchConfig(), internal_floor=value).validate()
        replace(algorithm.SearchConfig(), internal_floor=0.).validate()

    def test_zero_is_exact_bypass_without_calibration_or_budget(self):
        p = point([[1], [0], [0, 1]], [.1, 0., .001])
        p['metadata'] = {'keep': [3, 1]}
        f = fit_for(p); bank = [p]
        with patch.object(f, 'fit', side_effect=AssertionError('Zero floor must not refit')):
            q, receipt = floor.contract_point(p, f, 0., None, deadline=-1.)
            b, bank_receipt = floor.contract_bank(bank, f, 0., None, deadline=-1.)
        self.assertIs(q, p); self.assertIs(b, bank)
        self.assertIsNone(receipt); self.assertIsNone(bank_receipt)
        self.assertEqual(f.records, [])

    def test_calibration_is_arrays_only_and_matches_formula(self):
        R = np.array([[.4, .1], [.1, .8]])
        X = np.array([[.3, .05], [.05, .5]])
        with patch.object(trees, 'fit_sensitivity', side_effect=AssertionError('Must reuse fitted arrays')):
            c = floor.calibrate(dict(R=R, X=X))
        self.assertAlmostEqual(c['mu_r'], 1.)
        self.assertAlmostEqual(c['mu_x'], .7)
        self.assertAlmostEqual(c['median_root_depth'], .75*.6+.25*.4/.7)
        with self.assertRaises(ValueError):
            floor.calibrate(dict(R=np.zeros((2, 2)), X=np.zeros((2, 2))))

    def test_support_coordinates_are_checked_before_integer_conversion(self):
        for bad in (1.5, True, np.bool_(True), '1'):
            p = point([[0], [bad]], [.1, .2])
            with self.subTest(value=bad), self.assertRaises(ValueError):
                floor._point(p, 2)
        for good in (1, np.int64(1), 1.):
            p = point([[0], [good]], [.1, .2])
            family, _, _ = floor._point(p, 2)
            self.assertEqual(family, [(0,), (1,)])

    def test_common_removed_but_zero_singletons_stay_and_cache_reused(self):
        p = point([[0], [1], [0, 1]], [0., 0., .01])
        f = fit_for(p)
        bank, receipt = floor.contract_bank([p, p], f, .1, calibration(2))
        self.assertEqual(bank[0]['supports'], [[0], [1]])
        self.assertIs(bank[0], bank[1])
        self.assertEqual(len(f.records), 1)
        self.assertTrue(receipt['candidates'][1]['steps'][0]['fit']['cache_hit'])
        self.assertEqual(receipt['candidates'][0]['steps'][0]['removed_support'], [0, 1])
        self.assertTrue(floor.validate_bank(bank, receipt, 2)['passed'])
        self.assertEqual(p['supports'], [[0], [1], [0, 1]])

    def test_refit_can_create_a_second_violation(self):
        p = point([[0], [1], [2], [0, 1], [0, 1, 2]], [.01, .01, .01, .1, .7])
        f = SimpleNamespace(data=observations(p), reference_mse=.1, bound=2.,
                            qp_seconds=10., cache={}, records=[])
        def refit(family):
            q = point([list(C) for C in family], [.05 if len(C)>1 else .01 for C in family])
            f.cache[family] = q
            f.records.append(dict(family=family, certificate_gap=0.))
            return q
        f.fit = refit
        q, receipt = floor.contract_point(p, f, .2, calibration(3))
        self.assertEqual([s['removed_support'] for s in receipt['steps']], [[0, 1], [0, 1, 2]])
        self.assertEqual(q['supports'], [[0], [1], [2]])
        self.assertTrue(all(not s['eligible'] for s in receipt['final_strengths']))

    def test_equal_strength_tie_and_relative_roundoff(self):
        p = point([[0], [1], [2], [3], [2, 3], [0, 1]], [.1]*4+[.01, .01])
        q, receipt = floor.contract_point(p, fit_for(p), .1, calibration(4))
        self.assertEqual(receipt['steps'][0]['removed_support'], [0, 1])
        f = .01
        near = point([[0], [1], [0, 1]], [.2, .2, f*(1-.5e-10)])
        kept, info = floor.contract_point(near, fit_for(near), f, calibration(2))
        self.assertIs(kept, near)
        self.assertEqual(info['deletions'], 0)

    def test_stale_append_and_mutation_are_rejected_without_refit(self):
        p = point([[0], [1], [0, 1]], [.1, .2, .01])
        bank, receipt = floor.contract_bank([p], fit_for(p), .08, calibration(2))
        with self.assertRaisesRegex(ValueError, 'count'):
            floor.validate_bank([*bank, bank[0]], receipt, 2)
        bad = deepcopy(bank); bad[0]['r'][0] += .001
        with self.assertRaisesRegex(ValueError, 'parameters changed'):
            floor.validate_bank(bad, receipt, 2)
        stale = deepcopy(receipt); stale['calibration']['a'] *= 2.
        with self.assertRaisesRegex(ValueError, 'strength evidence changed'):
            floor.validate_bank(bank, stale, 2)

    def test_expired_budget_rejects_before_any_qp(self):
        p = point([[0], [1], [0, 1]], [.2, .2, .01]); f = fit_for(p)
        with patch.object(f, 'fit', side_effect=AssertionError('Must not start')), self.assertRaises(floor.OutputFloorError) as caught:
            floor.contract_bank([p], f, .08, calibration(2), deadline=-1.)
        self.assertTrue(caught.exception.receipt['deadline_exceeded'])
        self.assertFalse(caught.exception.receipt['partial_bank_accepted'])
        self.assertEqual(caught.exception.qp_records, [])

    def test_soft_deadline_overrun_keeps_record_but_rejects_point(self):
        p = point([[0], [1], [0, 1]], [.2, .2, .01]); f = fit_for(p)
        now = [0.]
        def refit(family):
            q = point([list(C) for C in family], [.2]*len(family))
            f.cache[family] = q; f.records.append(dict(family=family, objective=.1, certificate_gap=0.))
            self.assertEqual(f.qp_seconds, 1.)
            now[0] = 2.
            return q
        with patch.object(floor, 'perf_counter', side_effect=lambda: now[0]), patch.object(f, 'fit', side_effect=refit):
            with self.assertRaises(floor.OutputFloorError) as caught:
                floor.contract_point(p, f, .08, calibration(2), deadline=1.)
        self.assertEqual(f.qp_seconds, 10.)
        self.assertEqual(len(caught.exception.qp_records), 1)
        self.assertEqual(caught.exception.receipt['steps'][0]['fit']['status'], 'failed')

    def test_one_failed_candidate_rejects_entire_bank(self):
        stable = point([[0], [1]], [.2, .2])
        weak = point([[0], [1], [0, 1]], [.2, .2, .01]); f = fit_for(weak)
        with patch.object(f, 'fit', side_effect=RuntimeError('fixture QP failure')):
            with self.assertRaises(floor.OutputFloorError) as caught:
                floor.contract_bank([stable, weak], f, .08, calibration(2))
        self.assertEqual(caught.exception.receipt['completed_candidates'], 1)
        self.assertEqual(caught.exception.receipt['candidate_count'], 2)
        self.assertEqual(len(caught.exception.receipt['candidates']), 2)
        self.assertFalse(caught.exception.receipt['partial_bank_accepted'])

    def test_training_fingerprint_is_checked(self):
        p = point([[0], [1], [0, 1]], [.1, .2, .01]); f = fit_for(p)
        c = floor.calibrate(dict(R=np.eye(2), X=np.eye(2)), f.data)
        other = Observation(f.data.p, f.data.q, f.data.y+.01, f.data.labels)
        with self.assertRaisesRegex(floor.OutputFloorError, 'different training'):
            floor.contract_point(p, QPFitter(other), .08, c)

    def test_fit_places_floor_after_search_and_before_validation(self):
        weak = point([[0], [1], [2], [0, 1], [0, 1, 2]], [.2, .2, .2, .01, .02], [.1]*3+[0., 0.])
        train = observations(weak)
        guidance = dict(R=np.eye(3), X=np.eye(3), pool=trees.canonical(weak['supports']),
                        warm_families=(), reference_family=trees.canonical(weak['supports']))
        common = dict(n=3, labels=train.labels, training_sha256=algorithm._training_fingerprint(train),
                      seconds=0., reference_mse=.1, guidance=guidance)
        class NoMIQP:
            def __init__(self, *args): self.records = []
            def close(self): pass
        def run_one(threshold, validation):
            steps = [(weak, dict(stage='W', accepted=True)), (weak, dict(stage='W', accepted=False))]
            cfg = algorithm.SearchConfig(total_seconds=30., per_mip_seconds=5., qp_seconds=10.,
                output_reserve_seconds=3., internal_floor=threshold)
            with patch.object(algorithm, 'StructuralSolver', NoMIQP), patch.object(algorithm, '_forward', side_effect=steps), \
                 patch.object(trees, 'fit_sensitivity', side_effect=AssertionError('Calibration must reuse guidance')):
                return algorithm.fit(train, validation, method='000', config=cfg, common=common)
        original = run_one(0., train)
        contracted = run_one(.08, train)
        self.assertIsNone(original['output_contraction'])
        self.assertEqual(original['selected']['index'], 1)
        self.assertEqual(contracted['path'], original['path'])
        self.assertEqual(contracted['events'], original['events'])
        self.assertEqual(contracted['MIQP'], original['MIQP'])
        self.assertEqual(contracted['uncontracted_output_path'], [r['point'] for r in original['frozen_candidates']])
        self.assertEqual(len(contracted['frozen_candidates']), len(original['frozen_candidates']))
        self.assertTrue(all(len(C)==1 for r in contracted['frozen_candidates'] for C in r['point']['supports']))
        self.assertEqual(contracted['selected']['index'], 0)
        altered = Observation(train.p, train.q, train.y+.05, train.labels)
        second = run_one(.08, altered)
        self.assertEqual([r['point'] for r in contracted['frozen_candidates']],
                         [r['point'] for r in second['frozen_candidates']])
        self.assertTrue(floor.validate_bank([r['point'] for r in contracted['frozen_candidates']],
                                           contracted['output_contraction'], 3)['passed'])


if __name__ == '__main__':
    unittest.main()
