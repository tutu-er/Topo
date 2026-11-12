from time import perf_counter
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('gurobipy')
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.flexible_exit.model import MaskBank
from research_experiments.iterative_benchmark.run import common_select
from research_experiments.iterative_benchmark.analyze import paired,solver_records


def test_matched_selection_keeps_mask_local_and_labels_RNJ_protection():
    p=np.eye(3);z=np.array([0.,1.,1.]);y=.3*p+.5*np.outer(z,z)
    data=core._prepare_scenarios([dict(P_terminal=p,Q_terminal=0*p,drop_target=y)])
    validation=core._prepare_scenarios([dict(P_terminal=np.tile(p,(4,1)),Q_terminal=np.tile(0*p,(4,1)),drop_target=np.tile(y,(4,1)))])
    base=[(0,),(1,),(2,),(0,1,2)];pool=[*base,(0,1),(1,2)]
    cal=dict(a=1.,b=1.,median_root_depth=1.)
    bank=MaskBank(data,pool,cal,deadline=perf_counter()+30.)
    masked=MaskBank(data,[*base,(1,2)],cal,forbidden=((0,1),),deadline=perf_counter()+30.)
    try:
        wrong=bank.fit([*base,(0,1)],preserve=True);right=masked.fit([*base,(1,2)],preserve=True)
        paths=dict(baseline=(bank,[wrong]),masked=(masked,[right]))
        choices,views=common_select(bank,dict(bank=[wrong]),paths,validation)
        assert views['final']['source']=='masked' and views['final']['validation']<1e-8
        assert (0,1) not in [tuple(C) for C in choices['masked']['point']['supports']]
        _,protected=common_select(bank,dict(bank=[right]),paths,validation)
        assert protected['final']['source']=='complete_rnj'
        assert protected['final']['validation']<=views['final']['validation']+1e-12
    finally:bank.close();masked.close()


def test_paired_counts_retain_opposite_exact_changes():
    rows=[]
    for job,values in [('a',((.5,False),(1.,True))),('b',((1.,True),(.75,False)))]:
        for method,(f1,exact) in zip(('before','after'),values):
            rows.append(dict(job=job,panel='main',case='toy',method=method,view='final',clade_f1=f1,exact=exact,
                validation_mae=.1,train_mae=.1,test_mae=.1))
    result=paired(pd.DataFrame(rows),'before','final','after','final','matched')
    assert sum(r['new_exact'] for r in result)==1 and sum(r['lost_exact'] for r in result)==1
    assert sum(r['delta_F1'] for r in result)==pytest.approx(.25)


def test_solver_counts_include_unaccepted_steps_and_exit():
    branch=dict(trajectory=dict(attempts=[dict(kind='wzzt_extension',status=9,accepted=False),
        dict(kind='joint_tail',status=2,accepted=True)]),exit_event=dict(status=2))
    records=solver_records(branch)
    assert [r['status'] for r in records]==[9,2,2]
