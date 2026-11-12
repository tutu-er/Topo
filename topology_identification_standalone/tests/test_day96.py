"""Single-day data boundary, paired randomness and physical-profile checks."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import data


def capture_day(**kwargs):
    captured = {}
    original = data._ac_voltages
    def capture(network, labels, p, q, voltage):
        captured.update(network=copy.deepcopy(network), labels=labels, p=p.copy(),
                        q=q.copy(), root=voltage.copy())
        return original(network, labels, p, q, voltage)
    with patch.object(data, "_ac_voltages", side_effect=capture) as ac:
        observed, metadata = data.generate_day96(**kwargs)
    if ac.call_count != 1:
        raise AssertionError("A single-day call invoked multiple AC trajectories")
    return observed, metadata, captured


class Day96Tests(unittest.TestCase):
    def test_only_one_day_and_no_truth_or_multiday_generator(self):
        original = data._ac_voltages
        with patch.object(data, "case_truth", side_effect=AssertionError("truth read")), \
             patch.object(data, "generate_synthetic", side_effect=AssertionError("extra days")), \
             patch.object(data, "_ac_voltages", wraps=original) as ac:
            observed, meta = data.generate_day96("paper15", 3, .0005)
        self.assertEqual(ac.call_count, 1)
        self.assertEqual(ac.call_args.args[2].shape, (96, observed.n))
        self.assertEqual(observed.points, 96)
        self.assertEqual(observed.voltage.shape, observed.p.shape)
        self.assertEqual(observed.root.shape, (96,))
        self.assertEqual(meta["generated_days"], 1)
        self.assertFalse(meta["extra_days_generated"])
        self.assertFalse(meta["truth_returned"])
        np.testing.assert_array_equal(observed.y, observed.root[:, None]**2-observed.voltage**2)

    def test_deterministic_and_case_specific_streams(self):
        entropy, noises = [], []
        for case in data.CASES:
            first, meta = data.generate_day96(case, 7, .0005)
            second, repeated = data.generate_day96(case, 7, .0005)
            for name in ("p", "q", "y", "voltage", "root"):
                np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
            self.assertEqual(meta, repeated)
            entropy.append(tuple(meta["seed_entropy"]))
            noises.append(meta["standardized_noise_sha256"])
        self.assertEqual(len(set(entropy)), 4)
        self.assertEqual(len(set(noises)), 4)

    def test_noise_levels_share_clean_day_and_standardized_noise(self):
        clean, m0 = data.generate_day96("paper15", 8, 0.)
        low, m1 = data.generate_day96("paper15", 8, .0002)
        high, m2 = data.generate_day96("paper15", 8, .001)
        self.assertEqual(m0["clean_observation_sha256"], m1["clean_observation_sha256"])
        self.assertEqual(m1["clean_observation_sha256"], m2["clean_observation_sha256"])
        self.assertEqual(m0["standardized_noise_sha256"], m2["standardized_noise_sha256"])
        np.testing.assert_array_equal(low.p, high.p)
        np.testing.assert_array_equal(low.q, high.q)
        np.testing.assert_allclose((low.voltage/clean.voltage-1.)/.0002,
                                   (high.voltage/clean.voltage-1.)/.001, atol=2e-12, rtol=0.)
        np.testing.assert_allclose((low.root/clean.root-1.)/.0002,
                                   (high.root/clean.root-1.)/.001, atol=2e-12, rtol=0.)

    def test_load_and_independent_profile_change_only_declared_drivers(self):
        base, bm = data.generate_day96("paper15", 9, .0005)
        low, lm = data.generate_day96("paper15", 9, .0005, profile="load_scale", level=.5)
        np.testing.assert_allclose(low.p, .5*base.p, atol=0., rtol=1e-15)
        np.testing.assert_allclose(low.q, .5*base.q, atol=0., rtol=1e-15)
        np.testing.assert_array_equal(low.root, base.root)
        self.assertEqual(bm["physical_branch_sha256"], lm["physical_branch_sha256"])
        self.assertEqual(bm["latent_drivers_sha256"], lm["latent_drivers_sha256"])
        original, _, drivers = capture_day(case="paper15", replicate=9, voltage_sigma=0., pq_sigma=0.)
        weak, _, reduced = capture_day(case="paper15", replicate=9, voltage_sigma=0., pq_sigma=0.,
                                      profile="independent_scale", level=.1)
        phase = 2*np.pi*np.arange(96)/96
        daily = .65+.20*np.sin(phase-.5*np.pi)+.10*np.sin(2*phase-.9*np.pi)
        nominal = np.array([b["pd_kw"] for b in drivers["network"]["buses"] if b["is_terminal"]])/drivers["network"]["base_kw"]
        original_factor, weak_factor = original.p/nominal, weak.p/nominal
        interior = (original_factor > .151) & (original_factor < 1.399)
        expected = daily[:, None]+.1*(original_factor-daily[:, None])
        np.testing.assert_allclose(weak_factor[interior], expected[interior], atol=1e-14, rtol=0.)
        np.testing.assert_array_equal(drivers["root"], reduced["root"])

    def test_rx_preserves_magnitude_and_weak_edge_is_predeclared(self):
        _, _, base = capture_day(case="paper15", replicate=4, voltage_sigma=.0005)
        _, _, changed = capture_day(case="paper15", replicate=4, voltage_sigma=.0005, profile="rx_ratio", level=2.)
        for a, b in zip(base["network"]["branches"], changed["network"]["branches"], strict=True):
            self.assertAlmostEqual(np.hypot(a["r_ohm"], a["x_ohm"]), np.hypot(b["r_ohm"], b["x_ohm"]), places=14)
            self.assertAlmostEqual(b["r_ohm"]/b["x_ohm"], 2*a["r_ohm"]/a["x_ohm"], places=13)
        _, wm, weak = capture_day(case="paper15", replicate=4, voltage_sigma=.0005, profile="weak_edge", level=.1)
        _, again = data.generate_day96("paper15", 22, .001, profile="weak_edge", level=.3)
        chosen = wm["weak_edge"]
        self.assertEqual(chosen["original_branch_id"], again["weak_edge"]["original_branch_id"])
        altered = [(a, b) for a, b in zip(base["network"]["branches"], weak["network"]["branches"], strict=True)
                   if a["r_ohm"] != b["r_ohm"] or a["x_ohm"] != b["x_ohm"]]
        self.assertEqual(len(altered), 1)
        a, b = altered[0]
        self.assertAlmostEqual(b["r_ohm"], .1*a["r_ohm"])
        self.assertAlmostEqual(b["x_ohm"], .1*a["x_ohm"])
        # Fresh loading proves the pressure call did not alter the physical case.
        self.assertEqual(data.load_case("paper15"), base["network"])

    def test_one_same_timepoint_matches_independent_ybus_power_equations(self):
        observed, _, saved = capture_day(case="paper15", replicate=6, voltage_sigma=0., pq_sigma=0.)
        net = saved["network"]; buses = [row["bus_id"] for row in net["buses"]]
        index = {bus: i for i, bus in enumerate(buses)}
        ybus = np.zeros((len(buses), len(buses)), complex)
        zbase = net["base_kv"]**2/net["base_mva"]
        for edge in net["branches"]:
            if not edge["status"]:
                continue
            a, b = index[edge["from_bus"]], index[edge["to_bus"]]
            admittance = zbase/complex(edge["r_ohm"], edge["x_ohm"])
            ybus[a, a] += admittance; ybus[b, b] += admittance
            ybus[a, b] -= admittance; ybus[b, a] -= admittance
        load = np.zeros(len(buses), complex)
        for k, bus in enumerate(saved["labels"]):
            load[index[bus]] = saved["p"][0, k]+1j*saved["q"][0, k]
        unknown = [i for i in range(len(buses)) if buses[i] != net["root_bus"]]
        vroot = saved["root"][0]
        def residual(values):
            voltage = np.full(len(buses), complex(vroot))
            voltage[unknown] = values[:len(unknown)]+1j*values[len(unknown):]
            error = voltage*np.conj(ybus@voltage)+load
            return np.r_[error[unknown].real, error[unknown].imag]
        solved = root(residual, np.r_[np.full(len(unknown), vroot), np.zeros(len(unknown))], tol=1e-10)
        self.assertLess(np.max(np.abs(residual(solved.x))), 1e-9)
        voltage = np.full(len(buses), complex(vroot))
        voltage[unknown] = solved.x[:len(unknown)]+1j*solved.x[len(unknown):]
        np.testing.assert_allclose(observed.voltage[0], np.abs(voltage[[index[b] for b in observed.labels]]), atol=1e-9, rtol=0.)

    def test_invalid_profiles_or_seed_values_fail_before_simulation(self):
        invalid = (dict(profile="extra_day"), dict(profile="baseline", level=.5), dict(level=0.),
                   dict(replicate=-1), dict(replicate=True), dict(study_seed=1.5),
                   dict(voltage_sigma=float("nan")), dict(pq_sigma=-.1))
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                data.generate_day96(**values)


if __name__ == "__main__":
    unittest.main()
