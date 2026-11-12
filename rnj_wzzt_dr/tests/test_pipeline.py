from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ..history import fit_observed, freeze_history
from ..observations import Observation, fingerprint
from ..identification import algorithm
from ..identification.optimization import predict, QPFitter
from ..identification.trees import basis
from ..dr.device_map import Devices
from ..dr.miqp_mse import MSEGate, solve_factorized
from ..dr.pipeline import load_resources, pair_gate, plan, prepare_dr, zero_screen
from ..dr.zero import joint, spectral_pair


RUN_SOLVER = False


def fixture():
    rng = np.random.default_rng(7112)
    p, q = rng.uniform(.1, .4, (20, 3)), rng.uniform(.01, .2, (20, 3))
    points = []
    for support, weight in (((0, 1), .12), ((1, 2), .1), ((0, 2), .15)):
        family = [*basis(3), support]
        points.append(dict(supports=[list(c) for c in family],
                           r=[.03]*4+[weight], x=[.02]*4+[weight*.3]))
    empty = Observation(p, q, np.zeros_like(p), (101, 102, 103))
    obs = Observation(p, q, predict(empty, points[0])+rng.normal(0, .001, p.shape), empty.labels)
    cfg = algorithm.SearchConfig()
    selected, mae, bank = algorithm._output(points, obs, obs, .2, cfg)
    result = dict(method="111", selected=selected, selected_mae_view=mae,
        frozen_candidates=bank, reference_mse=.2, config=vars(cfg), events=[])
    return obs, result


class PipelineTests(unittest.TestCase):
    def test_native_bank_and_current_are_preserved(self):
        obs, result = fixture()
        h = freeze_history(result, obs)
        self.assertEqual(len(h.models), 3)
        self.assertEqual(h.current, result["selected"]["index"])
        self.assertFalse(h.H.flags.writeable)
        self.assertAlmostEqual(h.K, 5.)
        self.assertEqual(h.epsilon_scaled, 1e-12)
        original = h.observed.y.copy()
        obs.y[:] = 42.
        np.testing.assert_array_equal(h.observed.y, original)
        self.assertFalse(h.observed.y.flags.writeable)

    def test_history_and_new_cells_use_original_mse_ties_without_refit(self):
        obs, result = fixture(); h = freeze_history(result, obs)
        new = Observation(obs.p[:7], obs.q[:7], obs.y[:7], obs.labels)
        before = h.parameter_sha256
        with patch.object(QPFitter, "fit", side_effect=AssertionError("must stay frozen")):
            selected = h.select(new)
        combined = Observation(np.concatenate((obs.p, new.p)), np.concatenate((obs.q, new.q)),
                               np.concatenate((obs.y, new.y)), obs.labels)
        expected, _, _ = algorithm._output([r["point"] for r in result["frozen_candidates"]],
                                          obs, combined, h.reference_mse, algorithm.SearchConfig())
        self.assertEqual(selected["selected"], expected["index"])
        self.assertEqual(selected["total_cells"], combined.y.size)
        self.assertFalse(selected["refitted"])
        self.assertEqual(before, h.parameter_sha256)

    def test_delta_uses_final_validation_tolerance_and_correct_sign(self):
        obs, result = fixture(); h = freeze_history(result, obs)
        new_cells = 21
        expected = (obs.y.size+new_cells)*.2*1e-12-h.sse[1]+h.sse[0]
        self.assertAlmostEqual(h.required_gap(0, 1, new_cells), expected, places=15)
        self.assertNotEqual(h.required_gap(0, 1, new_cells), h.required_gap(1, 0, new_cells))

    def test_stale_parameters_matrix_history_and_label_order_rejected(self):
        obs, result = fixture()
        for mutate in (lambda r: r["frozen_candidates"][0]["R"][0].__setitem__(0, 999),
                       lambda r: r["selected"].__setitem__("index", 2)):
            bad = copy.deepcopy(result); mutate(bad)
            with self.assertRaises(ValueError): freeze_history(bad, obs)
        result["handoff"] = dict(validation_sha256="wrong")
        with self.assertRaises(ValueError): freeze_history(result, obs)
        self.assertEqual(len(fingerprint(obs)), 64)

    def test_zero_screen_reads_full_bank_even_when_display_has_one(self):
        obs, result = fixture()
        result["competition"] = dict(output_count=1, dr_candidate_indices=[])
        h = freeze_history(result, obs)
        d = prepare_dr(h, np.column_stack((obs.p[:4], obs.q[:4])), voltage_sigma=.001)
        screen = zero_screen(d, target=0)
        self.assertEqual(len(screen["pairs"]), 2)
        self.assertEqual({r["competitor"] for r in screen["pairs"]}, {1, 2})
        self.assertIsNone(screen["minimum_fee"])  # No device contract was supplied.

    def test_spectral_integral_matches_dense_gate_in_both_target_directions(self):
        obs, result = fixture(); h = freeze_history(result, obs)
        forecast = np.column_stack((obs.p[:4], obs.q[:4]))
        d = prepare_dr(h, forecast, voltage_sigma=.001, power_relative_sd=.08, time_ar=.4)
        for target, competitor in ((0, 1), (1, 0)):
            spectral = next(r for r in zero_screen(d, target=target)["pairs"] if r["competitor"] == competitor)
            view = dict(d, p=SimpleNamespace(T=4), base_kw=100.,
                        original_device=SimpleNamespace(node=np.array([0, 1]), qratio=np.array([.2, .3])))
            gate = pair_gate(view, target=target, competitor=competitor, full=False)
            self.assertTrue(gate.zero_control_only)
            self.assertAlmostEqual(float(gate.offset@gate.offset)-gate.h, spectral["strict_margin"], places=10)
            self.assertAlmostEqual(gate.delta, h.required_gap(target, competitor, 12)/gate.scale**2, places=10)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "full control factor"):
                    solve_factorized(None, gate, Path(tmp)/"unused", None, {})

    def test_zero_auxiliary_support_does_not_create_new_topology(self):
        obs, result = fixture()
        # Use a distinct, legal zero internal support on a four-node family.
        p = np.eye(4); q = np.zeros_like(p); y = .1*p
        obs = Observation(p, q, y, (1, 2, 3, 4))
        star = dict(supports=[[i] for i in range(4)], r=[.1]*4, x=[0.]*4)
        other = dict(supports=[*star["supports"], [0, 1]], r=[.1]*4+[0.], x=[0.]*5)
        selected, _, bank = algorithm._output([star, other], obs, obs, 1., algorithm.SearchConfig())
        result = dict(method="111", selected=selected, frozen_candidates=bank,
                      reference_mse=1., config=vars(algorithm.SearchConfig()), events=[])
        h = freeze_history(result, obs)
        self.assertEqual(h.topology_ids[0], h.topology_ids[1])
        d = prepare_dr(h, np.column_stack((p, q)), voltage_sigma=.001)
        self.assertEqual(zero_screen(d, target=0)["joint"]["competitors"], 0)

    def test_full_control_factor_keeps_node_order_qratio_and_kw_base(self):
        obs, result = fixture(); h = freeze_history(result, obs)
        forecast = np.column_stack((obs.p[:4], obs.q[:4]))
        d = prepare_dr(h, forecast, voltage_sigma=.001)
        original = SimpleNamespace(node=np.array([2, 0]), qratio=np.array([.3, -.2]))
        d.update(p=SimpleNamespace(T=4), original_device=original, base_kw=100.)
        gate = pair_gate(d, target=0, competitor=1, full=True)
        u = np.array([[0., 1., 0., -.4], [.2, 0., -.3, 0.]])
        changed = forecast.copy()
        for r, node in enumerate(original.node):
            changed[:, node] += u[r]/100.
            changed[:, 3+node] += original.qratio[r]*u[r]/100.
        dh = h.H[0]-h.H[1]
        sd = np.maximum(np.sqrt(np.mean((obs.y-d["history_power"]@h.H[0].T)**2, axis=0)),
                        2*np.sqrt(2)*d["voltage_sigma"])
        a = .25*(1-.5*np.tile((sd/gate.scale)**2, 4))
        power_sd = .05*np.max(abs(forecast), axis=0)
        node_factor = dh[:, :3]*power_sd[:3]+dh[:, 3:]*power_sd[3:]
        temporal = .7**np.abs(np.arange(4)[:, None]-np.arange(4)[None, :])
        C = np.kron(temporal, node_factor@node_factor.T/gate.scale**2)
        G = np.eye(12)+2*np.sqrt(a)[:, None]*C*np.sqrt(a)[None, :]
        mean = (changed@dh.T/gate.scale).ravel()*np.sqrt(a)
        expected = float(mean@np.linalg.solve(G, mean))
        self.assertAlmostEqual(gate.value(u), expected, places=10)

    def test_nonzero_joint_optimizer_stays_explicitly_pending(self):
        from ..dr import pipeline
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"unimplemented"
            with patch.object(pipeline, "zero_screen", return_value=dict(minimum_fee=None, pairs=[{}, {}])):
                with self.assertRaisesRegex(NotImplementedError, "multi-competitor"):
                    plan(dict(p=object()), target=0, output=out)
            self.assertFalse(out.exists())

    def test_equal_response_tie_is_checked_for_arbitrary_target(self):
        obs, result = fixture(); h = freeze_history(result, obs)
        d = prepare_dr(h, np.column_stack((obs.p[:4], obs.q[:4])), voltage_sigma=.001)
        d["H"] = np.stack([h.H[0], h.H[0]])
        models = [dict(h.models[0], id=i, internal=0, topology_id=str(i)) for i in range(2)]
        d.update(models=models, history_sse=np.array([h.sse[0]]*2))
        yes = zero_screen(d, target=0)
        no = zero_screen(d, target=1)
        self.assertEqual(yes["joint"]["success_lower"], 1.)
        self.assertEqual(no["joint"]["success_lower"], 0.)
        self.assertFalse(no["joint"]["sufficient_90"])

    def test_union_bound_keeps_dependence_and_unproved_pairs(self):
        rows = [dict(log_failure_upper=np.log(.06), pair_90_sufficient=True)]*2
        cert = joint(rows)
        self.assertAlmostEqual(cert["success_lower"], .88)
        self.assertFalse(cert["sufficient_90"])

    def test_forecast_is_explicit_and_bad_horizons_or_laws_rejected(self):
        obs, result = fixture(); h = freeze_history(result, obs)
        for forecast, kwargs in ((np.ones((4, 3)), {}), (np.ones((4, 6)), dict(time_ar=1.)),
                                  (np.ones((4, 6)), dict(power_relative_sd=-1))):
            with self.assertRaises(ValueError): prepare_dr(h, forecast, voltage_sigma=.001, **kwargs)

    def test_original_resources_nominal_and_fee_match(self):
        original, adapter, _ = load_resources("paper15", labels=tuple(range(101, 116)))
        rng = np.random.default_rng(117)
        values = np.clip(rng.uniform(-.1, .1, adapter.zero.shape), adapter.lower, adapter.upper)
        requests = adapter.expand(values)
        a = original.nominal(requests)
        b = adapter.nominal(adapter.requests_to_values(requests))
        np.testing.assert_allclose(a, b["mean_u"], rtol=0., atol=1e-12)
        self.assertAlmostEqual(original.fees(a), b["fee"], places=10)
        self.assertTrue(adapter.audit_delivery(b["mean_u"])["passed"])

    def test_zero_fee_certificate_needs_unchanged_resources(self):
        # One saved topology makes zero sufficient within that finite bank.
        obs, result = fixture()
        original, adapter, base = load_resources("paper15", labels=tuple(range(101, 116)))
        n = 15; p = np.eye(n); q = np.zeros_like(p); y = .1*p
        obs = Observation(p, q, y, tuple(range(101, 116)))
        point = dict(supports=[[i] for i in range(n)], r=[.1]*n, x=[0.]*n)
        selected, _, bank = algorithm._output([point], obs, obs, 1., algorithm.SearchConfig())
        result = dict(method="111", selected=selected, frozen_candidates=bank, reference_mse=1.,
                      config=vars(algorithm.SearchConfig()), events=[])
        h = freeze_history(result, obs)
        d = prepare_dr(h, np.ones((96, 30))*.001, voltage_sigma=.001,
                       resources=(original, adapter, base))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"quote"
            with patch("sys.modules", {**sys.modules, "gurobipy": None}):
                answer = plan(d, target=0, output=out)
            self.assertEqual(answer["objective"], 0.)
            self.assertEqual(answer["lower_bound"], 0.)
            self.assertTrue((out/"complete.json").exists())
            with self.assertRaises(FileExistsError): plan(d, target=0, output=out)

    def test_renamed_copy_runs_without_parent_repository(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/"renamed"
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
            run = subprocess.run([sys.executable, "-I", "-B", str(dest/"run.py"), "doctor"],
                                 cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            self.assertTrue(json.loads(run.stdout)["sources"]["passed"])

    def test_no_parent_imports_in_active_modules(self):
        import ast
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            if "compat" in path.relative_to(root).parts or "outputs" in path.relative_to(root).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                self.assertFalse(any(n.startswith(("research_experiments", "rnj_wzzt.", "topology_identification_standalone"))
                                     for n in names), str(path))


class SolverTests(unittest.TestCase):
    def setUp(self):
        if not RUN_SOLVER:
            self.skipTest("Enable check --solver for the bounded solver smoke tests")

    def test_current_identifier_runs_small_W_T_E_paths(self):
        obs, _ = fixture()
        cfg = algorithm.SearchConfig(total_seconds=30., per_mip_seconds=5., qp_seconds=10., output_reserve_seconds=3.)
        for method in ("rnj", "nj", "100", "101", "111"):
            with self.subTest(method=method):
                result = fit_observed(obs, obs, method=method, config=cfg)
                frozen = freeze_history(result, obs)
                self.assertEqual(frozen.current, result["selected"]["index"])
                self.assertFalse(result["truth_used"])
                self.assertFalse(result["test_used"])
                self.assertFalse(result["total_cap_exceeded"])

    def test_retained_nonzero_pair_solver_matches_small_analytic_cost(self):
        arrays = dict(resource_lower_kw=-np.ones((1, 1)), resource_upper_kw=np.zeros((1, 1)),
            resource_cost=np.ones((1, 1)), resource_efficiency=np.ones(1),
            resource_energy_lower_kwh=np.array([-4.]), resource_energy_upper_kwh=np.zeros(1),
            resource_reset=np.ones((1, 1)), resource_terminal=np.zeros((1, 1)))
        device = Devices(arrays, .25, 1)

        class SmallProblem:
            active = np.arange(1); lower = -np.ones(1); upper = np.zeros(1)
            zero = np.zeros(1); warm = np.zeros(1)
            def expand(self, x): return np.asarray(x).reshape(1, 1)
            def requests_to_values(self, x): return np.asarray(x).ravel()
            def nominal(self, x):
                x = self.expand(x); u = self.device.nominal(x)
                return dict(x=x, mean_u=u, fee=self.device.fees(u))
            def audit_delivery(self, u): return self.device.audit(u)

        p = SmallProblem(); p.device = device
        gate = MSEGate(np.zeros(1), np.eye(1), np.array([0]), .2, 1., 0., .9, 1., 1., 0., (0., 0.), 1)
        gate.zero_control_only = False; gate.input_receipt = dict(scope="deterministic solver unit fixture")
        with tempfile.TemporaryDirectory() as tmp:
            # Independent feasible witness: min .25*u² s.t. u²>=.2, -1<=u<=0.
            # The original solver retains this input if a boundary incumbent
            # misses the exact replay check by floating-point roundoff.
            warm = np.array([[-np.sqrt(.2)*(1+1e-8)]])
            answer = solve_factorized(p, gate, Path(tmp)/"pair", warm,
                                       dict(seconds=5., mip_gap=.001))
            self.assertTrue(answer["qualified"], answer)
            self.assertAlmostEqual(answer["objective"], .05, delta=1e-5)
            self.assertTrue(answer["device"]["passed"])
