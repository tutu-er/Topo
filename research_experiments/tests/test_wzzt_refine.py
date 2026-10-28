from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"outputs/source_ac288_runtime_20261018"))
import numpy as np
import pandas as pd
from research_experiments.wzzt_retune.engine import PoolLP,compatible
from research_experiments.wzzt_refine.core import backward,two_step,pair_family,ridge_alpha


def observations(matrix):
    p=pd.DataFrame(np.eye(len(matrix)),columns=list(range(10,10+len(matrix))))
    q=p*0;v=pd.DataFrame(np.sqrt(1-p.to_numpy()@matrix),columns=p.columns)
    return p,q,v,pd.Series(np.ones(len(p)))


def atom(matrix,support,value):matrix[np.ix_(support,support)]+=value


def test_backward_can_remove_positive_false_atom_despite_training_loss_rise():
    matrix=np.eye(4)*.1;atom(matrix,[0,1],.01)
    train=observations(matrix);validation=observations(np.eye(4)*.1)
    leaves=[(i,) for i in range(4)];lp=PoolLP(train,[*leaves,(0,1)])
    initial=lp.fit([*leaves,(0,1)])
    assert initial['r'][-1]>.009
    result=backward(lp,initial['supports'],validation,seconds=2)
    assert result['decisions'][0]['accepted']
    simpler=result['points'][0]
    assert simpler['objective']>initial['objective']+1e-4
    assert all((i,) in [tuple(s) for s in simpler['supports']] for i in range(4))
    assert [0,1] not in simpler['supports']
    lp.close()


def test_two_true_additions_cross_a_false_prior_and_beat_both_single_replacements():
    matrix=np.eye(4)*.1;atom(matrix,[0,2],.015);atom(matrix,[0,1],.01);atom(matrix,[2,3],.01)
    obs=observations(matrix);leaves=[(i,) for i in range(4)]
    lp=PoolLP(obs,[*leaves,(0,2),(0,1),(2,3)])
    initial=lp.fit([*leaves,(0,2)])
    single=[lp.fit([*leaves,s])['objective'] for s in ((0,1),(2,3))]
    assert min(single)>initial['objective']+1e-5
    result=two_step(lp,initial['supports'],obs,seconds=2)
    assert result['points']
    best=result['points'][0]
    assert best['objective']<initial['objective']-1e-5
    assert [0,1] in best['supports'] and [2,3] in best['supports']
    assert [0,2] not in best['supports']
    assert abs(initial['objective']-.0025)<1e-8
    assert abs(best['objective']-.001875)<1e-8
    lp.close()


def test_pair_family_preserves_leaves_removes_crossings_and_rejects_crossing_new_pair():
    leaves=[(i,) for i in range(4)]
    family=pair_family([*leaves,(0,2)],(0,1),(2,3))
    assert set(leaves)<=set(family) and (0,2) not in family
    assert compatible(family,(0,1))
    assert pair_family(leaves,(0,1),(1,2)) is None


def test_normalized_ridge_penalty_scales_with_repeated_rows_and_channel_units():
    p=np.eye(3);q=p*.2
    alpha,_=ridge_alpha(p,q,.001)
    twice,_=ridge_alpha(np.vstack((p,p)),np.vstack((q,q)),.001)
    scaled,_=ridge_alpha(10*p,10*q,.001)
    assert np.isclose(twice,2*alpha)
    assert np.isclose(scaled,100*alpha)


def test_public_refinement_rejects_permuted_validation_terminal_order():
    import pytest
    from research_experiments.wzzt_refine.worker import refine_bank
    obs=observations(np.eye(4)*.1)
    wrong=tuple(f.iloc[:,::-1] for f in obs[:3])+(obs[3],)
    point=dict(supports=[[i] for i in range(4)],r=[.1]*4,x=[0]*4)
    with pytest.raises(ValueError,match='ordered terminal columns'):
        refine_bank(obs,wrong,[point],point['supports'],dict(a=1,b=0,median_root_depth=1))
