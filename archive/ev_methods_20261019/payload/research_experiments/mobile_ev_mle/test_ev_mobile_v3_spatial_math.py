"""Analytic mechanism checks; no new scientific action selector."""
import numpy as np


def block(support,n,r=.01):
    a=np.zeros(n);a[list(support)]=1.
    return -r*np.outer(a,a)


def test_four_terminal_remote_ambiguity_needs_remote_excitation():
    difference=block((2,3),4)
    e0=np.eye(4)[0];e2=np.eye(4)[2]
    np.testing.assert_array_equal(difference@e0,np.zeros(4))
    assert np.linalg.norm(difference@e2)>0
    # Removing a common voltage shift does not remove a proper branch block.
    common=np.eye(4)-np.ones((4,4))/4
    assert np.linalg.norm(common@difference@e2)>0


def test_three_terminal_moving_outside_true_block_loses_direct_excitation():
    difference=block((0,1),3)
    assert np.linalg.norm(difference@np.eye(3)[0])>0
    np.testing.assert_array_equal(difference@np.eye(3)[2],np.zeros(3))


def test_constant_prices_can_still_have_positive_minimum_task_mobility_value():
    T=8;travel_and_setup=2;connected=T-travel_and_setup
    a1,a2=2.,3.
    fixed=min(a1*T,0.)
    # Tiny allocation audit only: no route/action library in actual selector.
    moving=max(min(a1*n1,a2*(connected-n1)) for n1 in range(connected+1))
    assert fixed==0 and moving==6.
