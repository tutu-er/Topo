"""Analytic and saved-matrix numerical regressions; no fresh AC fitting."""
from pathlib import Path
import gzip,json
import numpy as np
import pytest
from scipy.optimize import LinearConstraint
from .ev_bilinear_support_milp import BilinearPair,evaluate_qp
from .ev_profile_chord_milp import FrozenQuadratic
from . import ev_bilinear_score_exchange_v2 as v2


def direct(name,c,B):
    c=np.asarray(c,float);B=np.asarray(B,float)
    return BilinearPair(name,c,B,np.zeros((len(c),0)),np.zeros((B.shape[1],len(c),0)),[],[])


def test_lp_dual_interval_is_valid_and_tightens_control_box():
    low=np.zeros(2);high=np.ones(2)
    form=v2._lp_form(low,high,LinearConstraint([[1.,1.]],1.,1.))
    q=FrozenQuadratic('sum',[2.],[[1.,1.]])
    a,b,var,audits=v2.tight_residual_intervals(q,low,high,form)
    assert a[0]<=3.<=b[0] and b[0]-a[0]<1e-10
    assert audits[0]['lower_lp']['success']
    assert audits[0]['upper_lp']['success']
    for x in np.linspace(0,1,11):
        assert a[0]<=float((q.d+q.T@np.array([x,1-x]))[0])<=b[0]


def test_signed_lp_dual_box_correction_does_not_require_exact_stationarity():
    form=v2._lp_form(np.zeros(2),np.ones(2),LinearConstraint([[1.,1.]],.8,1.2))
    bound,audit=v2._linear_dual_lower([.4,-.7],np.zeros(2),np.ones(2),form)
    assert audit['success']
    # Minimum at (0,1), which satisfies the declared linear domain.
    assert bound<=-.7 and bound==pytest.approx(-.7,abs=1e-10)


@pytest.mark.parametrize('offset',[0.,3.])
def test_interior_max_min_and_negative_signed_objective(offset):
    pairs=[direct('a',[0.],[[1.]]),direct('b',[1.],[[-1.]])]
    out=v2.exchange_bilinear_scores_v2(pairs,[0.],[1.],offsets={'a':offset,'b':offset},
        score_scale=1.,initial_u=[0.],absolute_gap=1e-7,relative_gap=0.,max_iterations=4,segments=2)
    truth=.25-offset
    assert out['completed'],out['status']
    assert out['objective_lower']<=truth+1e-8<=out['objective_upper']+1e-8
    assert out['u'][0]==pytest.approx(.5,abs=1e-6)
    assert all(r['upper_update_accepted'] for r in out['iterations'])
    assert out['gap']>=0 and out['qp_calls']==2*len(out['evaluations'])


def test_conflicting_solver_upper_is_rejected_not_clipped_or_closed(monkeypatch):
    pairs=[direct('a',[0.],[[1.]]),direct('b',[1.],[[-1.]])]
    def broken(*args,**kwargs):
        return dict(u=np.array([.5]),objective_lower=.25,objective_upper=.1,bound_consistent=False,
            message='deliberately inconsistent solver bound')
    monkeypatch.setattr(v2,'solve_tight_chord_master',broken)
    out=v2.exchange_bilinear_scores_v2(pairs,[0.],[1.],score_scale=1.,initial_u=[0.],
        absolute_gap=1e-8,relative_gap=0.)
    assert out['unfinished'] and not out['completed']
    assert out['status']=='numerical_master_upper_rejected'
    assert out['objective_upper']==out['initial_fixed_witness_upper']
    assert out['objective_upper']>=.25
    assert out['objective_lower']==pytest.approx(.25)
    assert out['iterations'][0]['rejected_upper']==.1
    assert not out['iterations'][0]['upper_update_accepted']


def test_feasible_initial_qp_witnesses_remove_midpoint_scale_pollution():
    p=BilinearPair('cancel',[0.],[[1.]],[[1.]],[[[0.]]],[-1e5],[1e5])
    out=v2.exchange_bilinear_scores_v2([p],[0.],[1.],score_scale=1.,initial_u=[.5],
        absolute_gap=1e-7,relative_gap=0.,max_iterations=2)
    assert out['objective_lower']==pytest.approx(0.,abs=1e-9)
    assert out['objective_upper']>=0.
    # Starting feasible QP witness z=-.5 yields a global squared upper .25,
    # while later z controls close the cancellation geometry.
    assert out['initial_fixed_witness_upper']<.251
    assert all(len(e['qp_results'])==1 for e in out['evaluations'])


@pytest.fixture(scope='module')
def saved():
    path=Path(__file__).resolve().parent/'evidence'/'ev_cross_all'/'diagnostics'/'score_v2_first_saved_fixture.json.gz'
    fixture=json.loads(gzip.decompress(path.read_bytes()))
    pairs=[BilinearPair(**row) for row in fixture['pairs']]
    con=LinearConstraint(fixture['control_A'],fixture['control_lb'],fixture['control_ub'])
    return fixture,pairs,con


def test_saved_first_condition_conflict_is_explained_and_initial_bound_suffices(saved):
    f,pairs,con=saved
    assert f['original_v1']['status']=='numerical_bound_inconsistency'
    assert f['original_v1']['raw_gap']==pytest.approx(-8.66273522071026e-6)
    out=v2.exchange_bilinear_scores_v2(pairs,f['lower'],f['upper'],con,
        offsets=f['offsets'],score_scale=f['score_scale'],initial_u=f['initial_u'],
        absolute_gap=.01,relative_gap=.01)
    assert out['status']=='gap_reached_initial_witness_bound' and out['completed']
    assert out['objective_upper']>=out['objective_lower']
    assert out['objective_upper']<2e-5 and out['midpoint_fixed_witness_upper']>1e7
    assert len(out['required_pairs'])==12 and out['qp_calls']==12
    assert out['u']==pytest.approx(f['initial_u']) and not out['iterations']


def test_saved_first_condition_tight_budget_retains_all_pairs_and_valid_audited_upper(saved):
    f,pairs,con=saved
    out=v2.exchange_bilinear_scores_v2(pairs,f['lower'],f['upper'],con,
        offsets=f['offsets'],score_scale=f['score_scale'],initial_u=f['initial_u'],
        absolute_gap=1e-7,relative_gap=0.,max_iterations=1,segments=4,solver_seconds=2.,total_seconds=10.)
    assert len(out['required_pairs'])==12
    assert all(len(e['qp_results'])==12 for e in out['evaluations'])
    assert out['objective_upper']>=out['objective_lower']
    for u in ([3.]*4,[6.,6.,0.,0.],[6.,0.,6.,0.],[0.,6.,0.,6.]):
        value=min(.5*(evaluate_qp(p,u)['q_tangent_lower']-f['offsets'][p.name]) for p in pairs)
        assert out['objective_upper']>=value-1e-9
    if out['status']=='numerical_master_upper_rejected':
        assert out['unfinished'] and out['objective_upper']==out['initial_fixed_witness_upper']
    for it in out['iterations']:
        assert it['master']['scale']<1e3
        assert all(row['compressed_rows']<=5 for row in it['compressions'])
