from time import perf_counter
from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical
from research_experiments.lp_rebuild.engine import FloorBank
from research_experiments.joint_five.search import search
from research_experiments.flexible_exit.model import MaskBank
from research_experiments.rnj_small_seed.engine import exit_repair
from research_experiments.iterative_exit.zero import release_zero_atoms
from research_experiments.iterative_exit.engine import fit_mainline,select_outputs,weakest_deletion


def data_for(C=(1,2)):
    p=np.eye(3);z=np.array([i in C for i in range(3)],float);y=.3*p+.2*np.outer(z,z)
    return core._prepare_scenarios([dict(P_terminal=p,Q_terminal=0*p,drop_target=y)])


def setup(data):
    family=canonical([(0,),(1,),(2,),(0,1),(0,1,2)])
    bank=FloorBank(data,family,dict(a=1.,b=1.,median_root_depth=1.))
    return bank,family


def test_ordinary_forward_releases_zero_initial_merge_before_searching():
    data=data_for();bank,family=setup(data)
    try:
        old=search(bank,family,seconds=10.,per_mip=5.,zero_release=False)
        fixed=search(bank,family,seconds=10.,per_mip=5.)
        assert old['point']['objective']>1e-3
        assert fixed['point']['objective']<1e-8 and (1,2) in fixed['point']['supports']
        assert (0,1) not in fixed['point']['supports']
        assert any((0,1) in e['removed'] and e['accepted'] and e['stage']=='initial' for e in fixed['zero_events'])
    finally:bank.close()


def test_zero_prior_is_protected_and_numerical_zero_can_be_rejected():
    bank,family=setup(data_for())
    try:
        fixed=search(bank,family,seconds=5.,hard=((0,1),))
        assert (0,1) in fixed['point']['supports'] and (1,2) not in fixed['point']['supports']
    finally:bank.close()
    point=dict(supports=family,r=[.3,.3,.3,.5e-9,0.],x=[0.]*5,objective=.1)
    class AdverseRefit:
        data=SimpleNamespace(n=3)
        def fit(self,keep,*,preserve):return dict(supports=keep,r=[.3]*len(keep),x=[0.]*len(keep),objective=.100001)
    retained,events=release_zero_atoms(AdverseRefit(),point)
    assert retained is point and not events[0]['accepted']


def test_mask_survives_exit_pool_and_warm_start_construction():
    data=data_for();bank=MaskBank(data,[(0,),(1,),(2,),(0,1,2)],dict(a=1.,b=1.,median_root_depth=1.),
        forbidden=((0,1),),deadline=perf_counter()+30.)
    old,initial=setup(data)
    try:
        warm=old.fit(initial,preserve=True);current=bank.fit([(0,),(1,),(2,),(0,1,2)],preserve=True)
        point,event=exit_repair(bank,initial,current,np.eye(3),reference=.01,prefix_seconds=0.,total_seconds=10.,
            warm_candidates=[warm],forbidden=((0,1),))
        assert (0,1) not in canonical(event['pool'])
        assert (0,1) not in canonical(point['point']['supports'])
        assert point['point']['objective']<1e-8
    finally:bank.close();old.close()


def test_stall_forks_once_and_never_masks_a_hard_prior():
    data=data_for((0,1));validation=data_for((1,2));bank,initial=setup(data)
    try:points=[bank.fit(initial,preserve=True)]
    finally:bank.close()
    common=dict(initial=initial,pool=initial,bank=points,kernel=np.eye(3).tolist(),calibration=dict(a=1.,b=1.,median_root_depth=1.))
    out=fit_mainline(data,validation,common,reference=.01,seconds=30.,prefix_seconds=8.,per_mip=3.,lp_seconds=3.,post_reserve=3.)
    assert out['recovery_attempts']==1 and set(out['branches'])=={'baseline','release','masked'}
    C=out['selected_block']
    assert all(C not in canonical(p['supports']) for p in out['branches']['masked']['trajectory']['path'])
    assert out['with_rnj_fallback']['validation'] <= out['choices']['baseline']['validation']+1e-12
    assert not out['total_cap_exceeded'] and not out['LP_failures']
    protected=fit_mainline(data,validation,common,reference=.01,seconds=20.,prefix_seconds=5.,per_mip=3.,lp_seconds=3.,
        post_reserve=2.,hard=((0,1),))
    assert protected['recovery_attempts']==0 and (0,1) in canonical(protected['with_rnj_fallback']['point']['supports'])
    assert all((0,1) not in canonical(e.get('released',())) for e in protected['prefix']['attempts'])


def test_validation_can_adopt_or_reject_masked_counterfactual_independent_of_train_loss():
    baseline=dict(point={'objective':0.},validation=.2,internal=3,index=0,source='baseline')
    free=dict(point={'objective':.1},validation=.15,internal=3,index=0,source='release')
    masked=dict(point={'objective':.3},validation=.1,internal=3,index=0,source='masked')
    proposal,own,final=select_outputs(dict(baseline=baseline,release=free,masked=masked))
    assert final['source']=='masked' and final['point']['objective']>baseline['point']['objective']
    proposal,own,final=select_outputs(dict(baseline=baseline,release=free,masked={**masked,'validation':.4}))
    assert final['source']=='release'
    assert masked['source']=='masked'  # no in-place changes or permanent exclusion.


def test_deletion_ranking_respects_hard_supports():
    data=data_for((0,1));bank,initial=setup(data)
    try:
        point=bank.fit(initial,preserve=True)
        best,record=weakest_deletion(bank,point,hard=((0,1),),deadline=perf_counter()+5.)
        assert best is None and record['complete']
    finally:bank.close()
