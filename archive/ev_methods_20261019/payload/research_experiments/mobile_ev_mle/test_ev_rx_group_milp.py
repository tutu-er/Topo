"""Analytic grouped bounds and a tiny independent operation-state audit."""
import itertools
import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from . import ev_original_domain_milp as od
from . import ev_shared_square_milp as ss


def direct(name,c,B):
    c=np.asarray(c,float);B=np.asarray(B,float);m,n=B.shape
    return od.OriginalDomainPair(name,c,B,np.zeros((m,0)),np.zeros((n,m,0)),
                                [],[],[],np.zeros((0,n)))


def witnesses(pairs):
    return [[np.empty(0)] for _ in pairs]


@pytest.mark.parametrize('form',['residual_chord','shared_control_square'])
def test_equal_group_mass_with_unequal_sizes_and_zero_task(form):
    pairs=[direct('zero',[0.],[[0.]]),direct('easy',[10.],[[0.]]),
           direct('other',[2.],[[0.]])]
    mapping={'zero':'A','easy':'A','other':'B'}
    out=od.exchange_original_domain(pairs,[0.],[1.],aggregation='nearest',
        task_groups=mapping,initial_u=[.5],score_scale=1.,master_form=form,
        absolute_gap=1e-8,relative_gap=0.)
    assert out['completed']
    assert out['objective_lower']==pytest.approx(2.,abs=1e-9)
    assert out['objective_upper']>=2.-1e-9
    assert out['group_weights']=={'A':.5,'B':.5}
    assert out['group_score_intervals']['A']==pytest.approx([0.,0.])
    assert out['qp_calls']==3*len(out['evaluations'])
    master=od.solve_original_master(pairs,witnesses(pairs),[0.],[1.],
        aggregation='nearest',task_groups={'A':['zero','easy'],'B':['other']},
        score_scale=1.,master_form=form)
    mean=od.solve_original_master(pairs,witnesses(pairs),[0.],[1.],
        score_scale=1.,master_form=form)
    assert master['surrogate_incumbent']==pytest.approx(2.,abs=1e-8)
    assert master['n_group_targets']==2
    assert master['n_binary']==mean['n_binary']
    assert master['n_variables']==mean['n_variables']+2
    assert master['objective_lower'] is None


@pytest.mark.parametrize('form',['residual_chord','shared_control_square'])
def test_nearest_nonzero_while_global_worst_is_zero(form):
    pairs=[direct('left',[0.],[[1.]]),direct('right',[1.],[[-1.]]),
           direct('zero',[0.],[[0.]])]
    kwargs=dict(aggregation='nearest',task_groups={'left':'A','right':'A','zero':'B'},
                initial_u=[.5],score_scale=1.,master_form=form,
                absolute_gap=1e-7,relative_gap=0.,solver_seconds=2.,max_iterations=3)
    out=od.exchange_original_domain(pairs,[0.],[1.],**kwargs)
    # Independent scalar audit: .5*min(u²,(1-u)²) maximizes at u=.5.
    truth=max(.5*min(u*u,(1-u)**2) for u in np.linspace(0.,1.,1001))
    assert truth==pytest.approx(.125)
    assert out['objective_lower']<=truth+1e-8
    assert out['objective_upper']>=truth-1e-8
    assert out['objective_lower']==pytest.approx(truth,abs=1e-7)
    assert out['u']==pytest.approx([.5],abs=1e-7)
    assert out['worst_score_interval'][1]==0.
    assert out['completed']
    assert len(out['required_pairs'])==3


@pytest.mark.parametrize('form',['residual_chord','shared_control_square'])
def test_independent_three_state_audit_covers_grouped_bounds(form):
    pairs=[direct('a0',[0.,0.],[[1.,0.],[0.,2.]]),
           direct('a1',[1.,1.],[[-1.,0.],[0.,-1.]]),
           direct('b0',[0.,0.],[[2.,0.],[0.,.5]]),
           direct('b1',[2.,1.],[[-1.,0.],[0.,-.5]])]
    operation=dict(lower=[0.,0.],upper=[1.,1.],integrality=[1,1],control_indices=[0,1],
                   constraints=LinearConstraint([[1.,1.]],-np.inf,1.))
    allowed=[np.array(u,float) for u in itertools.product((0.,1.),repeat=2) if sum(u)<=1.]
    def score(u):
        costs=[float((p.c+p.B@u)@(p.c+p.B@u)) for p in pairs]
        return .5*(min(costs[:2])+min(costs[2:]))*.5
    truth=max(map(score,allowed))
    assert truth==pytest.approx(.75)
    out=od.exchange_original_domain(pairs,[0.,0.],[1.,1.],operation=operation,
        aggregation='nearest',task_groups={'A':['a0','a1'],'B':['b0','b1']},
        initial_u=[0.,0.],master_form=form,score_scale=.5,
        absolute_gap=1e-7,relative_gap=0.,solver_seconds=2.,max_iterations=4)
    assert out['objective_lower']<=truth+1e-8
    assert out['objective_upper']>=truth-1e-8
    assert out['objective_lower']==pytest.approx(truth,abs=1e-7)
    assert out['u']==pytest.approx([1.,0.],abs=1e-7)
    assert out['qp_calls']==4*len(out['evaluations'])
    for entry in out['evaluations']:
        costs=entry['pair_score_lower']
        assert entry['score_lower']==pytest.approx(.5*(min(costs[:2])+min(costs[2:])))
    assert all(it['master']['objective_lower'] is None for it in out['iterations'])


@pytest.mark.parametrize('form',['residual_chord','shared_control_square'])
def test_surrogate_incumbent_is_not_lower_and_scale_preserved(form):
    pairs=[direct('power',[0.],[[1.]]),direct('constant',[1.],[[0.]])]
    out=od.exchange_original_domain(pairs,[0.],[1.],LinearConstraint([[1.]],.5,.5),
        aggregation='nearest',task_groups={'power':'A','constant':'B'},
        score_scale=.5,initial_u=[.5],master_form=form,control_segments=1,residual_segments=1,
        max_segments=1,max_iterations=1,absolute_gap=1e-9,relative_gap=0.)
    master=out['iterations'][0]['master']
    assert master['surrogate_incumbent']==pytest.approx(.375,abs=1e-8)
    assert out['objective_lower']==pytest.approx(.3125,abs=1e-8)
    assert out['objective_upper']>=.375-1e-8
    assert master['objective_lower'] is None and not master['incumbent_is_true_score_lower']
    assert out['unfinished']


@pytest.mark.parametrize('form',['residual_chord','shared_control_square'])
def test_offsets_and_caps_aggregate_after_per_pair_transform(form):
    pairs=[direct('p',[2.],[[0.]]),direct('n',[1.],[[0.]]),direct('q',[2.],[[0.]])]
    out=od.exchange_original_domain(pairs,[0.],[1.],aggregation='nearest',
        task_groups={'A':['p','q'],'B':['n']},offsets={'p':-2.,'n':4.,'q':0.},
        score_caps={'p':1.,'n':None,'q':None},score_scale=1.,initial_u=[.5],
        master_form=form,absolute_gap=1e-8,relative_gap=0.)
    assert out['objective_lower']==pytest.approx(-1.,abs=1e-8)
    assert out['objective_upper']>=-1.-1e-8
    assert out['nearest_score_interval'][1]==pytest.approx(.5)
    assert out['capped_nearest_score_interval'][1]==pytest.approx(-1.)


@pytest.mark.parametrize('bad',[None,{}, {'A':['a']}, {'A':['a','b'],'B':['b']},
    {'A':['a','a'],'B':['b']}, {'A':['a','unknown'],'B':['b']}, {'A':[],'B':['a','b']},
    {'a':'A'}, {'a':'','b':'B'}])
def test_invalid_partitions_rejected_before_any_inner_fit(bad,monkeypatch):
    pairs=[direct('a',[0.],[[1.]]),direct('b',[0.],[[2.]])]
    monkeypatch.setattr(od,'evaluate_original_pair',lambda *args:pytest.fail('inner fit must not run'))
    with pytest.raises(ValueError,match='task_groups|partition|group'):
        od.exchange_original_domain(pairs,[0.],[1.],aggregation='nearest',task_groups=bad,initial_u=[0.])
    for form in ('residual_chord','shared_control_square'):
        with pytest.raises(ValueError,match='task_groups|partition|group'):
            od.solve_original_master(pairs,witnesses(pairs),[0.],[1.],
                aggregation='nearest',task_groups=bad,master_form=form)


def test_nonuniform_weights_rejected_and_legacy_modes_do_not_silently_use_groups():
    pairs=[direct('a',[0.],[[1.]]),direct('b',[0.],[[2.]])]
    for form in ('residual_chord','shared_control_square'):
        with pytest.raises(ValueError,match='uniform pair weights'):
            od.solve_original_master(pairs,witnesses(pairs),[0.],[1.],aggregation='nearest',
                task_groups={'a':'A','b':'B'},weights=[.2,.8],master_form=form)
    with pytest.raises(ValueError,match='requires nearest'):
        od.exchange_original_domain(pairs,[0.],[1.],task_groups={'a':'A','b':'B'})


def test_group_error_bound_uses_largest_pair_error_per_group():
    spec={'A':(0,1),'B':(2,)}
    assert od._aggregation_error([100.,0.,1.],np.ones(3)/3,'nearest',spec)==50.5
    lower=od._aggregate([1.,3.,5.],np.ones(3)/3,'nearest',spec)
    upper=od._aggregate([2.,9.,5.2],np.ones(3)/3,'nearest',spec)
    assert lower==3. and upper>=lower
    pairs=[direct('a',[0.],[[1.]]),direct('b',[0.],[[2.]])]
    out=ss.solve_shared_square_master(pairs,witnesses(pairs),[0.],[1.],
        aggregation='nearest',task_groups={'a':'A','b':'B'},score_scale=1.)
    assert out['task_groups']=={'A':['a'],'B':['b']}
    assert out['objective_upper']>=2.5-1e-8


def test_nearest_mapping_survives_compact_to_generic_fallback():
    pairs=[direct('cross',[0.],[[1.,1.]]),direct('constant',[2.],[[0.,0.]])]
    out=od.solve_original_master(pairs,witnesses(pairs),[0.,0.],[1.,1.],
        aggregation='nearest',task_groups={'cross':'A','constant':'B'},
        score_scale=1.,master_form='shared_control_square')
    assert out['master_form']=='residual_chord'
    assert 'cross-control Gram' in out['shared_square_fallback']
    assert out['task_groups']=={'A':['cross'],'B':['constant']}
    assert out['group_weights']=={'A':.5,'B':.5}
    assert out['objective_upper']>=4.-1e-8


def test_nearest_inner_failure_preserves_pair_and_marks_unfinished(monkeypatch):
    pairs=[direct('failed',[0.],[[0.]]),direct('other',[2.],[[0.]])]
    original=od.evaluate_original_pair
    def fail_one(pair,u):
        if pair.name=='failed':raise RuntimeError('declared failure')
        return original(pair,u)
    monkeypatch.setattr(od,'evaluate_original_pair',fail_one)
    out=od.exchange_original_domain(pairs,[0.],[1.],initial_u=[.5],
        aggregation='nearest',task_groups={'failed':'A','other':'B'},score_scale=1.,
        absolute_gap=1e-8,relative_gap=0.)
    assert out['objective_lower']==pytest.approx(2.,abs=1e-8)
    assert not out['completed'] and out['unfinished']
    assert len(out['required_pairs'])==2
    assert all(len(e['qp_results'])==2 for e in out['evaluations'])
