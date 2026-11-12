"""Check experiment denominators and independently reproducible saved models."""
import numpy as np

from research_experiments.rnj.run_source_ac96_benchmark import (
    job_catalog, plain, save_model, summarize,
)
from time import perf_counter


def test_all_64_jobs_are_unique_and_retained_before_execution(tmp_path):
    jobs = job_catalog()
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 64
    rows = summarize(tmp_path)
    assert len(rows) == 64
    assert all(row["status"] == "not_run" for row in rows)
    assert all("clade_f1" not in row for row in rows)


def test_model_objective_keeps_source_terms_and_no_truth_dependencies(tmp_path):
    arrays = {"P": np.array([[1., 2.], [2., 1.]]), "Q": np.zeros((2, 2)),
              "Y": np.array([[2., 3.], [4., 3.]]),
              "P_source": np.array([1., 2.]), "Q_source": np.zeros(2)}
    model = {"R": np.eye(2), "X": np.zeros((2, 2)),
             "source_g_r": [1., 1.], "source_g_x": [0., 0.]}
    result = save_model(tmp_path, model, arrays, perf_counter())
    assert result["has_model"] and result["train_mae"] == 0
    with np.load(tmp_path / "model.npz") as saved:
        assert set(saved.files) == {"R", "X", *arrays}


def test_nonfinite_diagnostics_serialize_as_missing_not_zero():
    assert plain({"gap": float("nan"), "bound": np.float64(np.inf)}) == {"gap": None, "bound": None}
