from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"outputs/source_ac288_runtime_20261018"))
import numpy as np
import pandas as pd
from research_experiments.wzzt_budget.worker import choose,contract,measured
from research_experiments.wzzt_retune.engine import PoolLP


def test_final_model_validation_overrides_stale_precontraction_score():
    p=pd.DataFrame(np.eye(2),columns=[10,11]);q=p*0
    v=pd.DataFrame(np.sqrt(1-p.to_numpy()@np.diag([.02,.03])),columns=p.columns)
    obs=p,q,v,pd.Series(np.ones(2))
    good=dict(supports=[[0],[1]],r=[.02,.03],x=[0,0],validation_mae=1.,objective=1.)
    bad=dict(supports=[[0],[1]],r=[0,0],x=[0,0],validation_mae=0.,objective=0.)
    selected=choose([measured(bad,obs,obs),measured(good,obs,obs)])
    assert np.allclose(selected['r'],[.02,.03])
    assert selected['validation_mae']<1e-12


def test_post_contraction_joint_refit_recovers_training_response():
    p=pd.DataFrame(np.eye(3),columns=[10,11,12]);q=p*0
    v=pd.DataFrame(np.sqrt(1-p.to_numpy()@np.diag([.02,.03,.04])),columns=p.columns)
    obs=p,q,v,pd.Series(np.ones(3))
    point=dict(supports=[[0],[1],[2],[0,1]],r=[.01,.02,.04,.001],x=[0,0,0,0])
    cut=contract(point,dict(median_root_depth=1,a=1,b=0))
    assert cut['supports']==[[0],[1],[2]]
    lp=PoolLP(obs,cut['supports'])
    fitted=measured(lp.fit(cut['supports']),obs,obs)
    assert fitted['objective']<1e-12
    assert fitted['objective']<measured(cut,obs,obs)['objective']
    lp.close()
