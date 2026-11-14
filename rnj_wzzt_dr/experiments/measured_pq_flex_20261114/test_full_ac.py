"""Independent two-bus voltage quadratic, export and conservation checks."""
import unittest
import numpy as np
from .full_ac import solve


class FullStateTests(unittest.TestCase):
    def test_two_bus_analytic_import_export_and_zero(self):
        r, x = .04, .02
        network = dict(root_bus=0, base_kw=1000., base_kv=1., base_mva=1.,
            buses=[dict(bus_id=0), dict(bus_id=1)],
            branches=[dict(from_bus=0, to_bus=1, status=True, r_ohm=r, x_ohm=x)])
        p, q = np.array([[.1], [-.1], [0.]]), np.array([[.03], [-.02], [0.]])
        state, receipt = solve(network, [1], p, q, np.ones(3))
        b = 1-2*(r*p[:, 0]+x*q[:, 0])
        expected = np.sqrt((b+np.sqrt(b*b-4*(r*r+x*x)*(p[:, 0]**2+q[:, 0]**2)))/2)
        np.testing.assert_allclose(state['terminal_voltage_pu'][:, 0], expected, atol=1e-11, rtol=0)
        self.assertLess(state['root_supply_kw'][1], 0)
        self.assertTrue((state['branch_loss_kw'][:2] > 0).all())
        np.testing.assert_allclose(state['root_supply_kw'], p[:, 0]*1000+state['branch_loss_kw'].sum(axis=1), atol=1e-6)
        np.testing.assert_array_equal(state['branch_loss_kw'][2], [0.])
        self.assertLess(receipt['max_KCL_residual_pu'], 1e-9)
        self.assertLess(receipt['max_branch_loss_identity_residual_kVA'], 1e-10)


if __name__ == '__main__':
    unittest.main()
