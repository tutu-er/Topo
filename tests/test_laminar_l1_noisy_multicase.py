from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from experiments.run_laminar_l1_noisy_multicase import (
    _aggregate,
    build_benchmark_cases,
    contaminate_measurements,
    run_one,
)
from terminal_case33.data.clade_grid_case_bank import make_clade_grid_case


def _toy_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.RangeIndex(12, name="time_index")
    columns = pd.Index([101, 102], name="terminal")
    p = pd.DataFrame(
        np.linspace(0.01, 0.05, 24).reshape(12, 2),
        index=index,
        columns=columns,
    )
    q = pd.DataFrame(
        np.linspace(0.005, 0.02, 24).reshape(12, 2),
        index=index,
        columns=columns,
    )
    target = 0.4 * p + 0.2 * q
    return p, q, target


def test_benchmark_cases_cover_distinct_four_and_six_terminal_topologies() -> None:
    cases = build_benchmark_cases()
    assert set(cases) == {
        "case33_aggregate4",
        "grid4_k2",
        "small6_branched",
        "grid6_k2_deep",
        "grid6_k3_wide",
    }
    counts = {
        name: int(net.buses["bus_type"].eq("observed_terminal").sum())
        for name, net in cases.items()
    }
    assert counts == {
        "case33_aggregate4": 4,
        "grid4_k2": 4,
        "small6_branched": 6,
        "grid6_k2_deep": 6,
        "grid6_k3_wide": 6,
    }


def test_requested_paper_style_cases_have_expected_terminal_counts() -> None:
    requested = ["paper15", "soumalas11", "flynn16", "pengwah18"]
    cases = build_benchmark_cases(requested)
    assert list(cases) == requested
    counts = {
        name: int(net.buses["bus_type"].eq("observed_terminal").sum())
        for name, net in cases.items()
    }
    assert counts == {
        "paper15": 15,
        "soumalas11": 11,
        "flynn16": 16,
        "pengwah18": 18,
    }


def test_contamination_is_reproducible_and_separates_response_from_eiv() -> None:
    p, q, target = _toy_frames()
    response_only = contaminate_measurements(
        p,
        q,
        target,
        seed=17,
        predictor_noise_fraction=0.0,
    )
    response_only_again = contaminate_measurements(
        p,
        q,
        target,
        seed=17,
        predictor_noise_fraction=0.0,
    )
    p_observed, q_observed, target_observed, metadata, mask = response_only
    pd.testing.assert_frame_equal(p_observed, p)
    pd.testing.assert_frame_equal(q_observed, q)
    pd.testing.assert_frame_equal(target_observed, response_only_again[2])
    assert np.array_equal(mask, response_only_again[4])
    assert int(mask.sum()) == ceil(0.05 * target.size)
    assert metadata["p_noise_rmse"] == 0.0
    assert metadata["q_noise_rmse"] == 0.0
    assert not np.allclose(target_observed, target)

    eiv = contaminate_measurements(
        p,
        q,
        target,
        seed=17,
        predictor_noise_fraction=0.02,
    )
    assert not np.allclose(eiv[0], p)
    assert not np.allclose(eiv[1], q)
    # Identical seed keeps the response contamination identical; only P/Q change.
    pd.testing.assert_frame_equal(eiv[2], target_observed)
    assert np.array_equal(eiv[4], mask)


def test_noisy_grid4_smoke_run_is_auditable() -> None:
    net = make_clade_grid_case(
        4,
        2,
        impedance_scale=3.0,
        case_name="test_grid4_k2",
    )
    metrics, atoms = run_one(
        case_name="test_grid4_k2",
        net=net,
        regime="response_outliers",
        seed=20260903,
        sample_count=30,
        train_count=20,
        validation_count=5,
        time_limit=15.0,
        coefficient_bound=2.0,
        response_laplace_fraction=0.03,
        outlier_fraction=0.05,
        gross_outlier_fraction=1.0,
    )
    assert metrics["terminal_count"] == 4
    assert metrics["truth_used_for_candidate_generation"] is False
    assert metrics["selected_family_is_laminar"] is True
    assert metrics["all_attempts_certified_optimal"] is True
    assert 0.0 <= metrics["nontrivial_support_f1"] <= 1.0
    assert np.isfinite(metrics["test_mae_noisy_fixed_training_intercept"])
    assert np.isfinite(metrics["R_matrix_relative_frobenius_error"])
    assert np.isfinite(metrics["X_matrix_relative_frobenius_error"])
    assert set(atoms.columns) >= {"support_labels", "r_value", "x_value"}


def test_aggregate_reports_rates_and_solver_failures() -> None:
    frame = pd.DataFrame(
        [
            {
                "case": "c",
                "regime": "r",
                "terminal_count": 4,
                "nontrivial_support_exact": True,
                "nontrivial_support_f1": 1.0,
                "R_matrix_relative_frobenius_error": 0.1,
                "X_matrix_relative_frobenius_error": 0.2,
                "test_mae_noisy_fixed_training_intercept": 0.01,
                "test_mae_noisy_over_signal_mad": 0.1,
                "all_attempts_certified_optimal": True,
                "fit_wall_seconds": 1.0,
            },
            {
                "case": "c",
                "regime": "r",
                "terminal_count": 4,
                "nontrivial_support_exact": False,
                "nontrivial_support_f1": 0.5,
                "R_matrix_relative_frobenius_error": 0.3,
                "X_matrix_relative_frobenius_error": 0.4,
                "test_mae_noisy_fixed_training_intercept": 0.03,
                "test_mae_noisy_over_signal_mad": 0.3,
                "all_attempts_certified_optimal": False,
                "fit_wall_seconds": 3.0,
            },
        ]
    )
    summary = _aggregate(frame).iloc[0]
    assert summary["run_count"] == 2
    assert summary["topology_exact_rate"] == 0.5
    assert summary["nontrivial_support_f1_mean"] == 0.75
    assert summary["timeout_or_uncertified_run_count"] == 1
    assert summary["fit_wall_seconds_total"] == 4.0
