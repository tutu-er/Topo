"""Small analytic checks only; no study NPZ, household or solver calls."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research_experiments.wzzt_dr_selection.dr_day_modules import history_audit as audit


def test_positive_proportion_and_declared_noise_reference():
    y = np.array([1., 2., 4., 3.])
    x = 2*y
    value = audit.pair_channel(x, y)
    assert value["status"] == "positive_proportion_evaluated"
    assert value["alpha_unconstrained"] == value["alpha_nonnegative"] == 2.
    assert value["residual_rms"] == 0.
    assert value["rms_normalized_residual"] == 0.
    assert np.isclose(value["noise_reference_rms"], .005*np.sqrt(np.mean(x*x+4*y*y)))
    assert value["at_one_noise_reference"] and value["at_two_noise_reference"]


def test_zero_intercept_does_not_hide_constant_offset():
    y = np.array([1., 2., 3., 4.])
    x = y+1.
    value = audit.pair_channel(x, y)
    alpha = np.dot(x,y)/np.dot(y,y)
    assert np.isclose(value["alpha_nonnegative"], alpha)
    assert np.isclose(value["residual_rms"], np.sqrt(np.mean((x-alpha*y)**2)))
    assert value["residual_rms"] > .1


def test_power_unit_scaling_preserves_normalized_diagnostics():
    x = np.array([2., 3., 4., 6.])
    y = np.array([1., 2., 3., 4.])
    left, right = audit.pair_channel(x,y), audit.pair_channel(200*x,200*y)
    assert np.isclose(right["residual_rms"], 200*left["residual_rms"])
    assert np.isclose(right["noise_reference_rms"], 200*left["noise_reference_rms"])
    assert np.isclose(right["rms_normalized_residual"], left["rms_normalized_residual"])
    assert np.isclose(right["residual_over_noise_reference"], left["residual_over_noise_reference"])


def test_zero_and_constant_channels_stay_explicitly_unclassified():
    y = np.array([1.,2.,3.,4.])
    near_zero = audit.pair_channel(np.zeros(4),y)
    constant = audit.pair_channel(np.full(4,2.),np.ones(4))
    assert near_zero["status"] == "near_zero_power_unresolved"
    assert near_zero["alpha_nonnegative"] is None
    assert constant["status"] == "constant_channel_not_standardized"
    assert constant["residual_rms"] == 0.
    for value in (near_zero,constant):
        assert value["rms_normalized_residual"] is None
        assert value["at_one_noise_reference"] is None
        assert value["at_two_noise_reference"] is None


def test_negative_best_proportion_is_not_clipped_into_a_positive_match():
    y = np.array([1.,2.,3.,4.])
    value = audit.pair_channel(-2*y,y)
    assert value["alpha_unconstrained"] == -2.
    assert value["alpha_nonnegative"] == 0.
    assert value["status"] == "nonpositive_best_proportion"
    assert value["at_one_noise_reference"] is None
    assert value["target_has_negative"]


def test_invalid_values_do_not_get_dropped_or_interpolated():
    for x,y in (([1.,np.nan],[1.,2.]),([],[]),([1.,2.],[1.])):
        with pytest.raises(ValueError):
            audit.pair_channel(x,y)
    with pytest.raises(ValueError):
        audit.pair_channel([1.,2.],[1.,2.],sigma=0.)


def test_pair_checks_both_directions_and_keeps_PQ_unresolved():
    y = np.array([1.,2.,3.,4.])
    p = np.column_stack((2*y,y))
    q = np.zeros_like(p)
    value = audit.pair_record(p,q,0,1,[201,202])
    assert set(value["channels"]["P_kw"]["directions"]) == {"i_from_j","j_from_i"}
    assert value["channels"]["P_kw"]["descriptive_flags"]["1"] is True
    assert value["channels"]["Q_kvar"]["descriptive_flags"]["1"] is None
    assert value["both_PQ_flags"]["1"] is None
    counts = audit.empty_counts()
    audit.add_counts(counts,value)
    assert counts["pair_count"] == 1
    assert counts["both_PQ"]["1"]["unresolved"] == 1


def test_pair_classification_is_not_tied_to_label_order():
    y = np.array([1.,2.,3.,4.])
    p = np.column_stack((2*y,y))
    q = p*.35
    a = audit.pair_record(p,q,0,1,[201,202])
    b = audit.pair_record(p[:,::-1],q[:,::-1],0,1,[202,201])
    assert a["both_PQ_flags"] == b["both_PQ_flags"]
    assert a["channels"]["P_kw"]["directions"]["i_from_j"]["alpha_nonnegative"] == 2.
    assert b["channels"]["P_kw"]["directions"]["j_from_i"]["alpha_nonnegative"] == 2.


def test_fixed_four_hour_windows_cover_all_days_without_discarding_rows():
    for days in (3,6):
        windows = audit.window_masks(days)
        assert len(windows) == 7
        assert windows[0][1].sum() == days*96
        assert all(mask.sum() == days*16 for _,mask in windows[1:])
        np.testing.assert_array_equal(sum(mask.astype(int) for _,mask in windows[1:]),np.ones(days*96))


def test_metadata_projection_retains_complete_pool_and_never_reads_forbidden_fields():
    class Restricted(dict):
        def __getitem__(self,key):
            if key in {"test","existing_result","truth","final_errors"}:
                raise AssertionError("forbidden source field read")
            return super().__getitem__(key)
    rows = []
    for case in audit.CASES:
        for bundle in range(10):
            rows.append(Restricted(job_id=f"main__{case}__b{bundle}__reference__v0002__N288",
                case=case,bundle=bundle,N=288,condition="reference",voltage_sigma=.0002,
                pq_sigma=.005,terminal_count=16,declared_data_error=None,
                train=dict(path="train_observed.npz",catalog_sha256="train"),
                validation=dict(path="validation_observed.npz",catalog_sha256="validation"),
                test="do not read",existing_result="do not read"))
    selected = audit.select_metadata(dict(rows=rows))
    assert len(selected) == 40
    assert sum(r["retained_zero_gain_control"] for r in selected) == 2
    assert sum(r["previous_closed_b7_control"] for r in selected) == 1
    assert all(set(r["sources"]) == {"train","validation"} for r in selected)
    assert all("test" not in r and "existing_result" not in r for r in selected)


def test_observed_loader_indexes_only_labels_P_Q():
    class Archive:
        files = ["labels","P","Q","V","root"]
        def __init__(self):self.accessed=[]
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,key):
            self.accessed.append(key)
            if key=="labels":return np.array([201,202])
            if key=="P":return np.ones((3,96,2))*.01
            if key=="Q":return np.ones((3,96,2))*.005
            raise AssertionError("forbidden value channel indexed")
    archive=Archive()
    source=dict(path="never-opened.npz",catalog_hash_matches=True,sha256="locked")
    with patch.object(audit,"sha",return_value="locked"), patch.object(audit.np,"load",return_value=archive) as loader:
        labels,p,q=audit.load_pq(source,2)
    assert archive.accessed == ["labels","P","Q"]
    assert loader.call_args.kwargs["allow_pickle"] is False
    assert labels == [201,202]
    np.testing.assert_array_equal(p,np.full((3,96,2),2.))
    np.testing.assert_array_equal(q,np.full((3,96,2),1.))
