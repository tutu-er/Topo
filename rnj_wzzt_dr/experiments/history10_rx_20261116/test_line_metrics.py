"""Small independent arithmetic checks; no fitting, optimization, or AC."""
import tempfile
import unittest
from pathlib import Path

from .line_metrics import paired_line_metrics, compute


class LineMetricsTests(unittest.TestCase):
    def test_support_pairing_and_ohm_conversion(self):
        # zbase=.8: coefficients .075/.025 mean .03/.01 ohm.
        # Reversed support order must not change which true branch is compared.
        truth = {(0,): {'r_ohm': .02, 'x_ohm': .04},
                 (0, 1): {'r_ohm': .04, 'x_ohm': .08}}
        point = {'supports': [[1, 0], [0]], 'r': [.075, .025], 'x': [.3, .1]}
        result = paired_line_metrics(point, truth, .8)
        self.assertAlmostEqual(result['r']['mape_percent'], 37.5)
        self.assertAlmostEqual(result['x']['mape_percent'], 25.)
        self.assertAlmostEqual(result['r']['mean_absolute_error_ohm'], .01)
        self.assertAlmostEqual(result['x']['mean_absolute_error_ohm'], .02)
        self.assertAlmostEqual(result['equal_weight_R_X_MAPE_percent'], 31.25)

    def test_true_zero_is_separate_from_mape(self):
        truth = {(0,): {'r_ohm': 0., 'x_ohm': 0.},
                 (1,): {'r_ohm': .04, 'x_ohm': 0.}}
        point = {'supports': [[0], [1]], 'r': [.05, .125], 'x': [.1, .0]}
        result = paired_line_metrics(point, truth, .8)
        self.assertEqual(result['r']['zero_truth_count'], 1)
        self.assertAlmostEqual(result['r']['zero_truth_mean_absolute_error_ohm'], .02)
        self.assertAlmostEqual(result['r']['mape_percent'], 25.)
        self.assertAlmostEqual(result['r']['mean_absolute_error_ohm'], .015)
        self.assertIsNone(result['x']['mape_percent'])
        self.assertIsNone(result['equal_weight_R_X_MAPE_percent'])

    def test_missing_or_duplicate_support_is_rejected(self):
        truth = {(0,): {'r_ohm': .02, 'x_ohm': .03}}
        for supports in ([[1]], [[0], [0]]):
            with self.subTest(supports=supports), self.assertRaises(AssertionError):
                paired_line_metrics({'supports': supports, 'r': [.1] * len(supports),
                                     'x': [.1] * len(supports)}, truth, .8)

    def test_existing_output_rejected_before_input_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                compute(Path(tmp) / 'no_run', Path(tmp) / 'no_audit', Path(tmp))


if __name__ == '__main__':
    unittest.main()
