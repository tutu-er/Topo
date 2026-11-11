"""MSE-primary selection, normalized ties, and the fitting information boundary."""
from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

import algorithm
import data
from trees import basis


def observations(n=2, scale=1.):
    return data.Observation(scale*np.eye(n), np.zeros((n, n)), np.zeros((n, n)), tuple(range(n)))


def point(n, singleton_r, *, extra=()):
    supports = [*basis(n), *extra]
    values = {tuple(C): 0. for C in supports}
    values.update({(i,): float(r) for i, r in enumerate(singleton_r)})
    return dict(supports=[list(C) for C in supports],
                r=[values[tuple(C)] for C in supports], x=[0.]*len(supports))


def reversed_ranking():
    # On P=I, Y=0: A has MAE=.25, MSE=.25; B has MAE=.30,
    # MSE=.18. Both have feasible response coefficients in [0,2].
    return [point(2, (0., 1.)), point(2, (.6, .6))]


def common_for(observed):
    """Avoid unrelated geometry fitting in these selector/interface unit tests."""
    return dict(n=observed.n, labels=list(observed.labels), reference_mse=1., seconds=0.,
                training_sha256=algorithm._training_fingerprint(observed),
                guidance=dict(reference_family=basis(observed.n)))


class MSESelectionTests(unittest.TestCase):
    def test_MAE_and_MSE_ordering_are_opposite_and_MSE_is_primary(self):
        observed = observations()
        selected, mae_view, candidates = algorithm._output(
            reversed_ranking(), observed, observed, 1., algorithm.SearchConfig())
        self.assertEqual(selected["index"], 1)
        self.assertEqual(mae_view["index"], 0)
        self.assertAlmostEqual(selected["validation_mse"], .18, places=12)
        self.assertAlmostEqual(selected["validation_mae"], .30, places=12)
        self.assertAlmostEqual(mae_view["validation_mse"], .25, places=12)
        self.assertAlmostEqual(mae_view["validation_mae"], .25, places=12)
        self.assertEqual(len(candidates), 2)

    def test_public_fit_keeps_MSE_selected_and_MAE_only_as_diagnostic(self):
        observed = observations()
        mse_view, mae_view, candidates = algorithm._output(
            reversed_ranking(), observed, observed, 1., algorithm.SearchConfig())
        # The baseline branch provides a cheap real fit, while the actual
        # hand-computed library isolates result-field wiring from tree search.
        with patch.object(algorithm, "_output", return_value=(mse_view, mae_view, candidates)):
            result = algorithm.fit(observed, observed, method="rnj", common=common_for(observed))
        self.assertEqual(result["selection_metric"], "validation_mse")
        self.assertEqual(result["selected"]["index"], 1)
        self.assertEqual(result["selected_mse_view"], result["selected"])
        self.assertEqual(result["selected_mae_view"]["index"], 0)
        self.assertEqual(result["selected"]["point"], reversed_ranking()[1])

    def test_physical_rescaling_with_reference_preserves_the_selected_model(self):
        for scale in (1e-6, 1., 1e6):
            with self.subTest(scale=scale):
                observed = observations(scale=scale)
                selected, _, _ = algorithm._output(
                    reversed_ranking(), observed, observed, scale**2, algorithm.SearchConfig())
                self.assertEqual(selected["index"], 1)
                self.assertAlmostEqual(selected["validation_scaled_mse"], .18, places=12)
                np.testing.assert_allclose(selected["validation_mse"], .18*scale**2, rtol=1e-12)
        # At scale=1e-6 the physical MSE difference is below 1e-12. The
        # selector must still distinguish it using reference-normalized MSE.

    def test_MSE_numerical_tie_prefers_fewer_internal_connections(self):
        observed = observations(3)
        complex_model = point(3, (0., 0., 0.), extra=((0, 1),))
        config = algorithm.SearchConfig()
        for delta, expected_index in ((.5e-12, 1), (2e-12, 0)):
            # P=I gives normalized MSE=r²/3 for equal singleton r values.
            simple_model = point(3, (np.sqrt(3*delta),)*3)
            with self.subTest(normalized_MSE_delta=delta):
                selected, _, _ = algorithm._output(
                    [complex_model, simple_model], observed, observed, 1., config)
                self.assertEqual(selected["index"], expected_index)
        self.assertEqual(config.validation_tie, 1e-12)

    def test_fit_accepts_no_test_or_truth_inputs_and_loads_neither(self):
        signature = inspect.signature(algorithm.fit)
        self.assertNotIn("test", signature.parameters)
        self.assertNotIn("truth", signature.parameters)
        self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()))
        observed = observations()
        with self.assertRaises(TypeError):
            algorithm.fit(observed, observed, method="rnj", truth=((0, 1),))
        with self.assertRaises(TypeError):
            algorithm.fit(observed, observed, method="rnj", test=observed)
        with patch.object(data, "case_truth", side_effect=AssertionError("truth access inside fit")) as true_loader, \
                patch.object(data, "load_observations", side_effect=AssertionError("test-file access inside fit")) as loader:
            result = algorithm.fit(observed, observed, method="rnj", common=common_for(observed))
        true_loader.assert_not_called()
        loader.assert_not_called()
        self.assertFalse(result["truth_used"])
        self.assertFalse(result["test_used"])
        self.assertEqual(result["selection_metric"], "validation_mse")


if __name__ == "__main__":
    unittest.main()
