import numpy as np
import pytest

pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.local_rebuild.matched import all_hierarchies
from research_experiments.rnj_count.engine import joint_fit,criterion,count,select_output,search
from research_experiments.rnj_count.diagnose import error_type


def bank_for(p,y,family):
    data=core._prepare_scenarios([dict(P_terminal=p,Q_terminal=p*0,drop_target=y)])
    return FloorBank(data,family,dict(a=1.,b=1.,median_root_depth=1.))


def test_crossing_truth_and_compatible_overresolution_are_different_errors():
    true={(0,1,2)}
    assert error_type((0,1),true)==('compatible_overresolution',[])
    assert error_type((1,3),true)==('crosses_truth',[(0,1,2)])
    assert error_type((0,1,2),true)==('true',[])


def test_penalty_can_delete_a_line_even_when_raw_mae_increases():
    p=np.eye(3);z=np.array([1.,1.,0.]);keep=[(i,) for i in range(3)]+[(0,1,2)]
    bank=bank_for(p,.3*p+.002*np.outer(z,z),[*keep,(0,1)])
    try:
        current=bank.fit([*keep,(0,1)],preserve=True)
        point,event=joint_fit(bank,current,keep,penalty=.01,seconds=3.,stage='test')
        assert event['accepted'] and point['objective']>current['objective']+1e-5
        assert count(point,3)==0 and criterion(point,3,.01)<criterion(current,3,.01)
        assert event['regularized_bound']<=criterion(point,3,.01)+1e-8
    finally:
        bank.close()


def test_regularized_joint_model_matches_complete_small_family_enumeration():
    p=np.eye(4);R=.3*p
    for support,weight in [((0,1),.2),((0,1,2),.002)]:
        z=np.array([i in support for i in range(4)],float);R+=weight*np.outer(z,z)
    keep=[(i,) for i in range(4)]+[tuple(range(4))];bank=bank_for(p,R,keep)
    try:
        current=bank.fit(keep,preserve=True);penalty=.005
        point,event=joint_fit(bank,current,keep,penalty=penalty,seconds=3.,stage='test')
        exact=min(criterion(bank.fit([*keep,*h],preserve=True),4,penalty) for h in all_hierarchies(4))
        assert criterion(point,4,penalty)==pytest.approx(exact,abs=1e-8)
        assert event['regularized_bound']<=exact+1e-8
    finally:
        bank.close()


def test_output_penalty_has_nonincreasing_count_on_a_fixed_candidate_library():
    p=np.eye(3);z=np.array([1.,1.,0.]);keep=[(i,) for i in range(3)]+[(0,1,2)]
    bank=bank_for(p,.3*p+.02*np.outer(z,z),[*keep,(0,1)])
    try:
        points=[bank.fit(keep),bank.fit([*keep,(0,1)])]
        outputs=[select_output(bank,points,bank.data,penalty=v,resolution=0.) for v in (0.,.001,.01,.1)]
        counts=[count(q['point'],3) for q in outputs]
        assert counts==sorted(counts,reverse=True) and counts[0]==1 and counts[-1]==0
    finally:
        bank.close()


def test_zero_coefficient_preserves_unregularized_search_contract():
    p=np.eye(4);keep=[(i,) for i in range(4)]+[(0,1),tuple(range(4))]
    bank=bank_for(p,.3*p,keep)
    try:
        result=search(bank,keep,coefficient=0.,seconds=4.)
        assert result['penalty']==0. and result['coefficient']==0.
        assert all(result['path'][i+1]['objective']<=result['path'][i]['objective']+1e-9 for i in range(len(result['path'])-1))
    finally:
        bank.close()


def test_tuning_prefers_simplicity_only_inside_the_prespecified_predictive_band():
    from research_experiments.rnj_count.tune import choose
    rows=[dict(coefficient=0.,mean_log_ratio=0.,mean_internal=8.,group_log_ratios=[0.]*8),
          dict(coefficient=.003,mean_log_ratio=.0005,mean_internal=6.,group_log_ratios=[.0005]*8),
          dict(coefficient=.1,mean_log_ratio=.02,mean_internal=2.,group_log_ratios=[.02]*8)]
    selected,best=choose(rows)
    assert selected['coefficient']==.003 and best['coefficient']==0.
    assert not rows[-1]['eligible']


def test_protected_coarse_count_is_a_reported_constant_in_the_mip_bound():
    p=np.eye(6);keep=[(i,) for i in range(6)]+[(0,1,2),(3,4,5),tuple(range(6))]
    z=np.array([1.,1.,0.,0.,0.,0.]);bank=bank_for(p,.3*p+.1*np.outer(z,z),keep)
    try:
        current=bank.fit(keep,preserve=True)
        point,event=joint_fit(bank,current,keep,penalty=.001,seconds=3.,stage='test')
        assert event['protected_count_cost']==pytest.approx(.002)
        assert event['regularized_bound']<=criterion(point,6,.001)+1e-8
        assert criterion(point,6,.001)<=event['regularized_incumbent']+1e-8
    finally:
        bank.close()
