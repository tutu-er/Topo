import numpy as np
import pytest
try:
    from .power import (finite_rank_pvalue,joint_gaps,calibrate_rank_nulls,
                        candidate_pvalues,power_scores)
except ImportError:
    from power import (finite_rank_pvalue,joint_gaps,calibrate_rank_nulls,
                       candidate_pvalues,power_scores)


def test_all_ranks_and_ties_are_conservative():
    full=np.arange(100,dtype=float)
    ranks=np.array([finite_rank_pvalue(full[i],np.delete(full,i)) for i in range(100)])
    assert np.count_nonzero(ranks<=.1)==10
    assert finite_rank_pvalue(0.,np.zeros(99))==1


def test_atom_is_shared_across_history_and_future():
    # Separate H/N refits would yield 0 for topology0, joint refit yields100.
    gaps=joint_gaps([[10.]],[0.,100.,20.],[[0.],[10.],[10.]],[0,0,1])
    np.testing.assert_allclose(gaps,[[80.,0.]])


def test_glr_and_composite_pvalue_include_true_null():
    hm=np.zeros((3,1)); labels=np.array([0,0,1])
    fm={"a":np.array([[0.],[.1],[3.]]),"b":np.array([[0.],[.1],[-3.]])}
    cal=calibrate_rank_nulls([0.],hm,fm,labels,B=99,seed=100)
    values=candidate_pvalues([0.],hm,fm["a"],labels,[[2.]],cal)
    z=joint_gaps([[2.]],[0,0,0],fm["a"],labels)[0,0]
    assert values[0,0]>=finite_rank_pvalue(z,cal["values"][0])
    assert len(cal["values"])==len(labels)


def test_identical_distributions_remain_unresolved_and_strong_design_works():
    hm=np.zeros((2,1)); labels=np.array([0,1])
    fm={"same":np.zeros((2,1)),"separate":np.array([[-5.],[5.]])}
    scores=power_scores([0.],hm,fm,labels,training_count=64,calibration_count=99,seed=220)
    assert scores["same"]["score"]==0.
    assert scores["separate"]["score"]>.95


def test_null_simulation_false_rejection_with_action_adaptation():
    # Both controls get calibrated together; opposite-sign action responses.
    hm=np.zeros((2,1));labels=np.array([0,1])
    fm={"left":np.array([[0.],[-2.]]),"right":np.array([[0.],[2.]])}
    cal=calibrate_rank_nulls([0.],hm,fm,labels,B=399,seed=888)
    rng=np.random.default_rng(777)
    y=rng.normal(size=(3000,1))
    left=candidate_pvalues([0.],hm,fm["left"],labels,y,cal)[:,0]
    right=candidate_pvalues([0.],hm,fm["right"],labels,y,cal)[:,0]
    # Even choosing the action with the smaller p after seeing Y is covered
    # by the simulated all-action maximum for this finite simple null.
    assert np.mean(np.minimum(left,right)<=.1)<.13


def test_calibration_cannot_be_reused_for_changed_history_or_action():
    hm=np.zeros((2,1));labels=np.array([0,1]);fm={"a":np.array([[0.],[1.]])}
    cal=calibrate_rank_nulls([0.],hm,fm,labels,B=9,seed=14)
    with pytest.raises(ValueError,match="history"):
        candidate_pvalues([1.],hm,fm["a"],labels,[[0.]],cal)
    with pytest.raises(ValueError,match="action"):
        candidate_pvalues([0.],hm,np.array([[0.],[2.]]),labels,[[0.]],cal)
