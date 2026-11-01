"""Hand-computable nonunique-weight example for validation face bounds."""
import numpy as np
import pandas as pd
import pytest

from research_experiments.wzzt_dr_selection.dr_validation_excitation.profile_margin import refit_pair


def obs(p,y):
    index=pd.date_range('2001-01-01',periods=len(p),freq='15min')
    p=np.asarray(p,float);y=np.asarray(y,float)
    return (pd.DataFrame(p,index=index,columns=[201,202]),
            pd.DataFrame(np.zeros_like(p),index=index,columns=[201,202]),
            pd.DataFrame(np.sqrt(1-y),index=index,columns=[201,202]),
            pd.Series(np.ones(len(p)),index=index))


def test_reweighting_nonunique_fine_family_has_validation_range():
    train=obs([[.1,.1],[.2,.2]],[[.1,.1],[.2,.2]])
    valid=obs([[.1,0],[.2,0]],[[.06,.04],[.12,.08]])
    r=refit_pair(train,valid,[(0,),(1,)],[(0,),(1,),(0,1)],ranges=True,face_epsilon=1e-10,face_seconds=8.)
    assert r['coarse_training_loss'] < 1e-8 and r['fine_training_loss'] < 1e-8
    assert r['coarse_validation_loss'] == pytest.approx(.06,abs=1e-7)
    f=r['optimal_face_ranges']['fine']
    assert f['lower'] == pytest.approx(0.,abs=1e-7)
    assert f['upper'] == pytest.approx(.06,abs=1e-6)
    assert r['safe_validation_exit_margin'] < 0 < r['optimistic_validation_exit_margin']
    # Merely choosing validation-best fine weights would falsely imply a
    # tie-independent exclusion; all weights may still equal zero or be weak.
    assert r['optimal_face_ranges']['coarse']['parameter_domain']==[0.,2.]


def test_point_fit_uses_training_only():
    train=obs([[.1,.1],[.2,.2]],[[.1,.1],[.2,.2]])
    a=refit_pair(train,obs([[.1,0],[.2,0]],[[.06,.04],[.12,.08]]),[(0,),(1,)],[(0,),(1,),(0,1)])
    b=refit_pair(train,obs([[.1,0],[.2,0]],[[.09,.01],[.18,.02]]),[(0,),(1,)],[(0,),(1,),(0,1)])
    assert a['fine_weights']==b['fine_weights']
    assert a['fine_training_loss']==b['fine_training_loss']
