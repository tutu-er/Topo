"""Input/freeze/sampling integration checks with a small analytic Gaussian case."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from .test_run import inputs, _json
from .fixed_f_run import load, plan, sample
from .fixed_f import conditional_mean_posterior


def fixed_inputs(tmp_path):
    case, settings = inputs(tmp_path, initial_prior=4., eta=.6, beta=.8)
    values = json.loads(settings.read_text())
    values.update(fixed_f_arrays_file="fixed_f_arrays.npz", mean_posterior_target=.7,
                  fixed_f_scope="synthetic background power error only")
    _json(settings, values)
    np.savez_compressed(case/"fixed_f_arrays.npz", covariance=np.zeros((1,1)))
    return case, settings


def test_fixed_f_plan_freezes_inputs_and_samples_shared_model(tmp_path):
    case, settings = fixed_inputs(tmp_path)
    output = tmp_path/"quote"
    planned = plan(case, settings, "mean", output)
    assert planned["qualified"] and planned["fee"] == 0.
    assert planned["initial_log_odds"] == pytest.approx(4.)
    measured = sample(case, settings, {"mean":output}, tmp_path/"sample", count=256, seed=41)
    assert measured["unique_random_paths"] == 256
    assert measured["metrics"]["mean"] == measured["metrics"]["natural_zero"]
    assert measured["metrics"]["mean"]["mean_posterior_F"] == pytest.approx(
        conditional_mean_posterior(0., 4.))
    # Tampering is rejected before sampling, rather than silently rescoring.
    np.savez_compressed(case/"fixed_f_arrays.npz", covariance=np.ones((1,1)))
    with pytest.raises(ValueError, match="Frozen input changed"):
        sample(case, settings, {"mean":output}, tmp_path/"rejected", count=8, seed=41)


def test_sampling_factor_must_match_full_covariance(tmp_path):
    case, settings = fixed_inputs(tmp_path)
    np.savez_compressed(case/"fixed_f_arrays.npz", covariance=np.ones((1,1)),
                        ar_spatial_factor=np.zeros((1,1,1)), time_ar=np.array(.8))
    with pytest.raises(ValueError, match="Sampling factor and frozen full covariance differ"):
        load(case, settings)


def test_fixed_f_probability_target_is_separate_and_analytic_zero(tmp_path):
    case, settings = fixed_inputs(tmp_path)
    result = plan(case, settings, "probability", tmp_path/"probability")
    assert result["qualified"] and not result["solver"]["solver_performed"]
    assert result["guarantees"]["posterior_event_lower_bound"] >= .8


def test_standalone_command_has_a_working_package_import():
    script = Path(__file__).with_name("fixed_f_run.py")
    result = subprocess.run([sys.executable,"-B",str(script),"--help"],
                            capture_output=True,text=True,check=False)
    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout
