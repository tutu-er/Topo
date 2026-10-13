"""Adapter isolation and saved-evidence regression checks; no H optimization."""
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_information_milp as original
from . import adaptive_history, ev_operations_milp
from .shared_power_square_adapter import make_search_adapter
from .shared_power_square_audit import (
    DEFAULT_ARCHIVE, replay_qp_record, verify_archive_manifest,
    verify_honly_saved, verify_single_upper_saved, verify_snapshot_incumbent,
)
from .shared_power_square_upper import upper_master_shared


def test_adapter_copies_search_and_changes_only_upper_policy():
    globals_before = original.adaptive_search.__globals__.copy()
    clone, audit = make_search_adapter()
    assert clone.__code__ is original.adaptive_search.__code__
    assert clone.__defaults__ is original.adaptive_search.__defaults__
    assert clone.__kwdefaults__ == original.adaptive_search.__kwdefaults__
    assert clone.__globals__["upper_master"] is upper_master_shared
    assert set(audit["changed_globals"]) == {"upper_master"}
    assert all(original.adaptive_search.__globals__[key] is value
               for key, value in globals_before.items())


def test_common_qp_trace_passes_through_result_and_preserves_input(monkeypatch):
    nuisance = np.array([.3])
    expected = {"success": True, "status": 0, "message": "mock",
                "z": nuisance, "q_lower": 2., "q_upper": 3., "numeric_gap": 1.,
                "optimality": 1., "roundoff_guard": 1e-12, "acceleration": "mock"}
    def source_qp(pair, u, **kwargs):
        return expected
    monkeypatch.setitem(original.adaptive_search.__globals__, "solve_pair", source_qp)
    trace = []
    clone, audit = make_search_adapter(original.upper_master, qp_trace=trace)
    controls = np.array([[1., 0.]])
    returned = clone.__globals__["solve_pair"](SimpleNamespace(name="pair"), controls)
    assert returned is expected and set(audit["changed_globals"]) == {"solve_pair"}
    controls[0, 0] = 9.
    assert trace[0]["u_kw"][0, 0] == 1.


def test_qp_trace_preserves_exception(monkeypatch):
    def source_qp(pair, u):
        raise RuntimeError("preserved failure")
    monkeypatch.setitem(original.adaptive_search.__globals__, "solve_pair", source_qp)
    trace = []
    clone, _ = make_search_adapter(qp_trace=trace)
    with pytest.raises(RuntimeError, match="preserved failure"):
        clone.__globals__["solve_pair"](SimpleNamespace(name="pair"), [[0.]])
    assert trace[0]["exception"] == "RuntimeError" and trace[0]["success"] is False


def test_replay_rejects_wrong_primal_and_infeasible_nuisance():
    pair = {"c": np.array([2.]), "B": np.array([[1.]]), "A": np.array([[1.]]),
            "F": np.array([[0.]]), "lower": np.array([-1.]), "upper": np.array([1.])}
    record = {"u_kw": [[0.]], "z": [-1.], "q_upper": 1., "q_lower": 1.,
              "numeric_gap": 0., "roundoff_guard": 0.}
    assert replay_qp_record(pair, record) == (0., 0.)
    with pytest.raises(AssertionError):
        replay_qp_record(pair, {**record, "q_upper": 2., "numeric_gap": 1.})
    with pytest.raises(AssertionError):
        replay_qp_record(pair, {**record, "z": [2.]})


def test_saved_maximization_bound_and_integer_feasibility_are_checked():
    snapshot = {"lower": [0.], "upper": [2.], "integer": [1], "rows": [{"0": 1.}],
                "row_lower": [0.], "row_upper": [2.], "objective": {"0": 1.}}
    record = {"n_variables": 1, "n_rows": 1, "objective_value": 1., "objective_bound": 2.,
              "raw_minimization_fun": -1., "raw_minimization_mip_dual_bound": -2.}
    assert verify_snapshot_incumbent(snapshot, record, np.array([1.])) == (0., 0.)
    with pytest.raises(AssertionError):
        verify_snapshot_incumbent(snapshot, {**record, "objective_bound": .5}, np.array([1.]))
    with pytest.raises(AssertionError):
        verify_snapshot_incumbent(snapshot, record, np.array([1.5]))


def test_archived_honly_evidence_replays_without_optimizer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer or selector was called by saved-evidence audit")
    monkeypatch.setattr(original, "adaptive_search", forbidden)
    monkeypatch.setattr(original, "solve_pair", forbidden)
    monkeypatch.setattr(original, "lsq_linear", forbidden)
    monkeypatch.setattr(adaptive_history, "build_history_context", forbidden)
    monkeypatch.setattr(adaptive_history, "build_frozen_pairs", forbidden)
    monkeypatch.setattr(adaptive_history, "fit_candidate", forbidden)
    monkeypatch.setattr(ev_operations_milp.LinearMILP, "solve", forbidden)
    integrity = verify_archive_manifest()
    result = verify_honly_saved(DEFAULT_ARCHIVE / "originals" / "Honly_search_v1")
    assert integrity["manifest_files_verified"] > 100
    assert result["QP_records_replayed"] == 696
    assert result["MILP_snapshots_verified"] == 16
    assert result["full_AC_saved_witness_records_replayed"] == 96
    assert result["original_upper_missing_incumbent_and_bound"] == 3
    assert result["shared_upper_with_incumbent_and_bound"] == 3
    assert result["optimal_status_with_nonzero_mip_gap_preserved"] == 1


def test_same_pool_single_upper_comparison_replays():
    result = verify_single_upper_saved(DEFAULT_ARCHIVE / "originals" / "H_710009_compare")
    assert result["saved_QP_records_replayed"] == 72
    assert result["MILP_snapshots_verified"] == 2
    assert result["old_missing_incumbent_and_bound_preserved"]
