"""GPU arithmetic regression, optional dependency; no fitting or selection."""
import importlib.util
import unittest
import numpy as np
from .. import likelihood as lk
from ..reference_benchmark.derivatives import fit_response_derivatives
from .gpu_benchmark import fixture, compare
from .gpu import GPUWorkspace, gpu_problem_context


class CPUFixtureTests(unittest.TestCase):
    def test_physical_fixture_and_exact_zero_edges(self):
        candidate, data = fixture(32, 192)
        self.assertEqual(np.count_nonzero(data['r'] == 0), 32)
        physical = lk.ac_predict(candidate, **data)
        self.assertLess(physical['ac_equation_residual'], 2e-10)
        self.assertLess(physical['power_balance_residual'], 2e-10)


@unittest.skipUnless(importlib.util.find_spec('cupy'), 'CuPy optional dependency unavailable')
class GPUArithmeticTests(unittest.TestCase):
    def test_fp64_forward_and_selected_derivatives(self):
        for n, t, zero in ((3, 12, False), (3, 12, True), (32, 12, True), (32, 192, True)):
            with self.subTest(n=n, t=t, exact_zero=zero):
                candidate, data = fixture(n, t, zero_edges=zero)
                cpu = lk.ac_predict(candidate, **data)
                work = GPUWorkspace(candidate); dev = work.inputs(**data)
                actual = work.predict(dev); host = work.host(actual)
                self.assertTrue(compare(cpu, host, ('voltage', 'master_p', 'master_q', 'branch_current'))['passed'])
                for lines in (False, True):
                    expected = fit_response_derivatives(candidate, cpu, include_lines=lines)
                    derived = work.host(work.derivatives(actual, include_lines=lines))
                    self.assertTrue(compare(expected, derived, ('v', 'master_p', 'master_q'))['passed'])

    def test_invalid_inputs_are_rejected(self):
        candidate, data = fixture(3, 12)
        work = GPUWorkspace(candidate)
        data['r'][0] = -1.
        with self.assertRaises(ValueError): work.inputs(**data)

    def test_hybrid_context_preserves_residual_blocks_and_restores(self):
        from ..ev_rx_fit import _RxInputProblem
        from .blocks import problem_blocks
        candidate, data = fixture(3, 12, zero_edges=False)
        phy = lk.ac_predict(candidate, **data)
        observed = lk.Observations(p=data['p'], q=data['q'], v=phy['v']+1e-5,
            root_v=data['root_v'], master_p=phy['master_p'], master_q=phy['master_q'])
        model = lk.MeasurementModel(p_sigma=.001, q_sigma=.001, v_sigma=.0005,
                                    master_p_sigma=.001, master_q_sigma=.001)
        initial = {key: data[key] for key in ('r', 'x', 'source_p', 'source_q')}
        problem = _RxInputProblem(candidate, observed, model, initial=initial,
            r_bounds=(0., .1), x_bounds=(0., .05), ratio_bounds=(.1, .5))
        vector = problem.start.copy(); residual = problem.evaluate(vector).copy()
        line, source = problem_blocks(problem, vector)
        original = problem.base.evaluate
        with gpu_problem_context(problem) as context:
            actual = problem.evaluate(vector)
            gpu_line, gpu_source = context.blocks(problem, vector)
            np.testing.assert_allclose(actual, residual, atol=2e-9, rtol=2e-10)
            np.testing.assert_allclose(gpu_line, line, atol=2e-9, rtol=2e-10)
            np.testing.assert_allclose(gpu_source, source, atol=2e-9, rtol=2e-10)
            self.assertIn('voltage', problem.base.cache['physical'])
        self.assertEqual(problem.base.evaluate, original)


if __name__ == '__main__':
    unittest.main()
