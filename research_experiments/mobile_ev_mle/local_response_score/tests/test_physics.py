import numpy as np
from ..physics import box_absolute_norm,box_norm,norm_record


def test_nonzero_box_center_is_not_zero_centered_radius():
    lo=np.array([[2.,3.]])
    hi=np.array([[4.,5.]])
    cov=np.eye(2)[None]
    assert abs(box_absolute_norm(lo,hi,cov)['total']-np.sqrt(41.))<1e-12
    assert abs(box_norm((hi-lo)/2,cov)['total']-np.sqrt(2.))<1e-12
    # Translated response comparisons must translate their box centres too.
    assert norm_record((lo+hi)/2,cov)['total']==5.
