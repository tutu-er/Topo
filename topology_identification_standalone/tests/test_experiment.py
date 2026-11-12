"""CLI accounting and relocation checks; no large benchmark is run here."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import experiment


def args_for(output, **changes):
    values = dict(command="demo", dataset="toy", methods=["rnj", "nj"], output=output,
                  jobs=["pilot"], case="toy6", points=16, seed=7, time_limit=5., alpha=.003, block_limit=5)
    values.update(changes)
    return argparse.Namespace(**values)


class ExperimentTests(unittest.TestCase):
    def test_factorial_ids_and_aliases(self):
        self.assertEqual(set(experiment.FACTORIAL), {f"{i:03b}" for i in range(8)})
        self.assertEqual(experiment.methods_from_names(["pure_wzzt,rnj_wzzt", "rte", "111"]), ["000", "100", "111"])
        with self.assertRaises(ValueError):
            experiment.methods_from_names(["unknown"])

    def test_scores_exclude_bases_and_respect_response_orientation(self):
        truth = [(0,), (1,), (2,), (0, 1, 2), (0, 1)]
        point = dict(supports=truth, r=[1., 2., 3., .2, .4], x=[0.]*5)
        scores = experiment.topology_scores(point, truth, 3)
        self.assertEqual((scores["F1"], scores["exact"], scores["TP"]), (1., True, 1))
        R, X = experiment.response_matrices(point, 3)
        np.testing.assert_allclose(R, np.diag([1., 2., 3.])+.2*np.ones((3, 3))+.4*np.outer([1., 1., 0.], [1., 1., 0.]))
        np.testing.assert_array_equal(X, np.zeros((3, 3)))

    def test_demo_and_bundled_budget_defaults_are_distinct(self):
        self.assertEqual(experiment.experiment_config(args_for(None, time_limit=None)).total_seconds, 30.)
        # benchmark must use the full budget even though --dataset is only an
        # ablation option and retains its default value "toy".
        bundled = args_for(None, command="benchmark", time_limit=None)
        self.assertEqual(experiment.experiment_config(bundled).total_seconds, 14400.)

    def test_ambiguity_output_setting_does_not_change_search_gate(self):
        config = experiment.experiment_config(args_for(None, ambiguity_kappa=.03))
        self.assertEqual(config.ambiguity_kappa, .03)
        self.assertEqual(config.relative_gain, 0.)

    def test_factorial_effects_preserve_negative_T_and_all_orders(self):
        rows = []
        for method in experiment.FACTORIAL:
            R, T, E = map(int, method)
            rows.append(dict(job="toy", method=method, status="complete", exact=False,
                             F1=.8+.1*R-.04*T+.02*E-.02*R*T, total_seconds=1+R+T+E))
        effects, orders, shapley = experiment.ablation_effects(rows)
        self.assertEqual((len(effects), len(orders), len(shapley)), (12, 18, 3))
        self.assertEqual(len({row["order"] for row in orders}), 6)
        values = {row["module"]: row["F1"] for row in shapley}
        self.assertAlmostEqual(values["R"], .09)
        self.assertAlmostEqual(values["T"], -.05)
        self.assertAlmostEqual(values["E"], .02)
        rows[0]["status"] = "failed"
        effects, _, shapley = experiment.ablation_effects(rows)
        self.assertEqual(sum(row["status"] == "incomplete_pair" for row in effects), 3)
        self.assertTrue(all(row["F1"] is None and row["available_pairs"] == 3 for row in shapley))

    def test_pair_and_triple_interactions_preserve_missing_cells(self):
        rows = []
        for method in experiment.FACTORIAL:
            R, T, E = map(int, method)
            rows.append(dict(job="toy", method=method, status="complete", exact=False,
                             F1=.8+.1*R-.04*T+.02*E-.02*R*T+.01*R*T*E, total_seconds=1+R+T+E))
        interactions = experiment.ablation_interactions(rows)
        self.assertEqual(len(interactions), 7)
        values = {(row["interaction"], row["condition_value"]): row["delta_F1"] for row in interactions}
        self.assertAlmostEqual(values[("RT", 0)], -.02)
        self.assertAlmostEqual(values[("RT", 1)], -.01)
        self.assertAlmostEqual(values[("RTE", None)], .01)
        self.assertTrue(all(abs(row["delta_seconds"]) < 1e-12 for row in interactions))
        rows[0]["status"] = "budget_exceeded"
        incomplete = [row for row in experiment.ablation_interactions(rows) if row["status"] == "incomplete_cells"]
        self.assertEqual(len(incomplete), 4)
        self.assertTrue(all(row["delta_F1"] is None and row["available_cells"] == row["planned_cells"]-1 for row in incomplete))

    def test_over_budget_result_is_preserved_but_excluded_from_valid_scores(self):
        def fake_fit(train, validation, **kwargs):
            point = dict(supports=[(i,) for i in range(train.n)], r=[0.]*train.n, x=[0.]*train.n)
            return dict(selected=dict(point=point), path=[point], events=[], total_cap_exceeded=True)
        with tempfile.TemporaryDirectory(prefix="standalone-budget-", dir=ROOT.parent) as temporary:
            output = Path(temporary)/"results"
            with patch.object(experiment.algorithm, "prepare_common", return_value={}), patch.object(experiment.algorithm, "fit", fake_fit), redirect_stdout(io.StringIO()):
                summary = experiment.run_experiment(args_for(output, methods=["rnj"]))
            self.assertEqual((summary["planned"], summary["completed"], summary["budget_exceeded"], summary["failed"], summary["pending"]), (1, 0, 1, 0, 0))
            self.assertIsNone(summary["methods"][0]["F1_completed_mean"])
            saved = json.loads(next(output.glob("runs/*/*.json")).read_text())
            self.assertEqual(saved["status"], "budget_exceeded")
            self.assertIn("result", saved)
            self.assertNotIn("metrics", saved)

    def test_failure_retains_denominator_and_truth_stays_out_of_fit(self):
        received = []
        def fake_common(train, config):
            return dict(seconds=0., reference_mse=1.)
        def fake_fit(train, validation, *, method, config, common):
            received.append((train.points, validation.points, method))
            if method == "nj":
                raise RuntimeError("controlled unavailable solver")
            point = dict(supports=[(i,) for i in range(train.n)], r=[0.]*train.n, x=[0.]*train.n)
            return dict(selected=dict(point=point), path=[point], events=[])
        with tempfile.TemporaryDirectory(prefix="standalone-experiment-", dir=ROOT.parent) as temporary:
            output = Path(temporary)/"results"
            with patch.object(experiment.algorithm, "prepare_common", fake_common), patch.object(experiment.algorithm, "fit", fake_fit), redirect_stdout(io.StringIO()):
                summary = experiment.run_experiment(args_for(output))
            self.assertEqual((summary["planned"], summary["completed"], summary["failed"], summary["pending"]), (2, 1, 1, 0))
            self.assertEqual(received, [(16, 48, "rnj"), (16, 48, "nj")])
            failed = next(output.glob("runs/*/nj.json"))
            self.assertIn("controlled unavailable solver", json.loads(failed.read_text())["error"])
            # An identical completed plan is resumed without fitting again.
            with patch.object(experiment.algorithm, "prepare_common", side_effect=AssertionError("should not refit")), redirect_stdout(io.StringIO()):
                resumed = experiment.run_experiment(args_for(output))
            self.assertEqual(resumed["planned"], 2)
            with self.assertRaises(ValueError):
                experiment.run_experiment(args_for(output, seed=8))

    def test_relocated_folder_runs_with_parent_imports_blocked(self):
        """The copied directory, renamed and outside its original location,
        must run the two tree baselines with an isolated Python import path.
        """
        with tempfile.TemporaryDirectory(prefix="standalone-relocation-", dir=ROOT.parent) as temporary:
            temporary = Path(temporary)
            copied = temporary/"renamed_package"
            shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns(
                "results", "outputs", "tmp", "__pycache__", ".pytest_cache"))
            output = temporary/"results"
            code = r'''
import importlib.abc, pathlib, sys
copied = pathlib.Path(sys.argv[1]).resolve()
class NoParentProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'research_experiments', 'rnj_wzzt', 'terminal_case33', 'theft_wzzt'}:
            raise ImportError('Parent project import forbidden: '+fullname)
sys.meta_path.insert(0, NoParentProject())
sys.path.insert(0, str(copied))
import experiment
code = experiment.main(['demo', '--methods', 'rnj', 'nj', '--points', '16', '--time-limit', '10', '--output', sys.argv[2]])
if code:
    import json
    for file in pathlib.Path(sys.argv[2]).glob('runs/*/*.json'):
        record = json.loads(file.read_text())
        if record.get('error'):
            print(file.name, record['error'])
import data
assert data.verify_bundled_data()['passed']
catalog = data.load_catalog()
assert len(catalog) == 48
job, meta = next(iter(catalog.items()))
observed = data.load_observations(meta['train'])
assert observed.points == 96
assert data.case_truth(meta['case'], observed.labels)
for name in ('experiment', 'algorithm', 'optimization', 'trees', 'data'):
    module = sys.modules.get(name)
    if module is not None:
        assert pathlib.Path(module.__file__).resolve().is_relative_to(copied), (name, module.__file__)
assert not any(name.split('.')[0] in {'research_experiments', 'rnj_wzzt', 'terminal_case33', 'theft_wzzt'} for name in sys.modules)
raise SystemExit(code)
'''
            result = subprocess.run([sys.executable, "-I", "-c", code, str(copied), str(output)],
                                    cwd=temporary, text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary = json.loads((output/"summary.json").read_text())
            self.assertEqual((summary["planned"], summary["completed"], summary["failed"]), (2, 2, 0))


if __name__ == "__main__":
    unittest.main()
