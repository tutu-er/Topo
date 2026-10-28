"""Independent LP, lifted feasibility and replaceable-prior regressions."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"outputs/source_ac288_runtime_20261018"))
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from research_experiments.wzzt_retune.engine import PoolLP,extension_start,compatible
from research_experiments.wzzt_systematic.common import scenes_from_observed,raw_model
from research_experiments.wzzt_solver_tuning.profile_gaps import primal_audit


def example():
    p=pd.DataFrame(np.tile(np.eye(3),(2,1)),columns=[10,11,12]);q=p*0
    r=np.diag([.002,.003,.004]);r[np.ix_([0,1],[0,1])]+=.005
    v=pd.DataFrame(np.sqrt(1-p.to_numpy()@r),columns=p.columns)
    return p,q,v,pd.Series(np.ones(len(p)))


def test_reusable_lp_matches_independent_highs_and_closed_form():
    obs=example();family=[(0,),(1,),(2,),(0,1)]
    lp=PoolLP(obs,family)
    point=lp.fit(family)
    f=lp.features;m=f.shape[0];k=f.shape[1]
    reference=linprog(np.r_[np.zeros(k),np.ones(m)/m],A_ub=np.block([[f,-np.eye(m)],[-f,-np.eye(m)]]),
                     b_ub=np.r_[lp.y.ravel(),-lp.y.ravel()],bounds=[(0,2)]*k+[(0,None)]*m,method="highs")
    assert reference.success
    assert abs(point["objective"]-reference.fun)<1e-10
    assert point["objective"]<1e-12
    lp.close()


def test_lifted_start_satisfies_every_original_free_milp_row():
    from rnj_wzzt.estimation import laminar_l1_milp as core
    obs=example();leaf=((0,),(1,),(2,));lp=PoolLP(obs,[*leaf,(0,1)])
    point=lp.fit([*leaf,(0,1)])
    model=core.build_extension_model(core._prepare_scenarios(scenes_from_observed(obs)),leaf,r_upper_bound=2,x_upper_bound=2)
    raw=raw_model(model)
    primal=extension_start(model,leaf,(0,1),point,lp.p,lp.q,lp.y)
    assert primal_audit(raw,primal)["feasible"]
    primal[model.z.start]=.5
    assert not primal_audit(raw,primal)["feasible"]
    lp.close()


def test_free_discovery_can_expand_the_lp_library_without_losing_old_fits():
    lp=PoolLP(example(),[(0,),(1,),(2,)])
    old=lp.fit([(0,),(1,),(2,)])["objective"]
    lp.add_support((0,1))
    assert lp.fit([(0,),(1,),(2,)])["objective"]==old
    assert lp.fit([(0,),(1,),(2,),(0,1)])["objective"]<old
    lp.close()


def test_positive_crossing_prior_is_replaceable_but_leaves_are_preserved():
    family=[(0,),(1,),(2,),(3,),(0,2)]
    true_candidate=(0,1)
    assert not compatible(family,true_candidate)
    retained=[s for s in family if compatible([s],true_candidate)]
    assert retained==[(0,),(1,),(2,),(3,)]
    assert compatible(retained,true_candidate)


def test_warm_started_free_solve_retains_an_auditable_improved_model():
    from research_experiments.wzzt_retune.engine import free_search
    from research_experiments.wzzt_systematic.common import read
    root=Path(__file__).resolve().parents[2]/"tmp"
    with tempfile.TemporaryDirectory(dir=root) as directory:
        assert Path(directory).resolve().is_relative_to(root.resolve())
        lp=PoolLP(example(),[(0,),(1,),(2,),(0,1),(0,1,2)])
        result=free_search(lp,Path(directory)/"free",seconds=2,per_solve=.2)
        assert result["path"][-1]["objective"]<result["path"][0]["objective"]
        for record in (Path(directory)/"free").glob("*/result.json"):
            data=read(record)
            if data["accepted"]:assert data["selected_primal_audit"]["feasible"]
        lp.close()
