from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"outputs/source_ac288_runtime_20261018"))
import numpy as np
import pandas as pd
from research_experiments.wzzt_fine.common import sample_indices,SIZES,PoolView
from research_experiments.wzzt_retune.engine import PoolLP,search


def test_fine_sample_sets_are_nested_and_preserve_original_anchor_grids():
    sets=[set(sample_indices(N)) for N in SIZES]
    assert all(a<=b for a,b in zip(sets,sets[1:]))
    assert all(len(s)==N//3 for s,N in zip(sets,SIZES))
    for N,step in ((24,12),(48,6),(96,3)):assert np.array_equal(sample_indices(N),np.arange(0,96,step))


def test_pool_view_excludes_unavailable_support_even_with_shared_lp_columns():
    p=pd.DataFrame(np.eye(3),columns=[10,11,12]);q=p*0;r=np.eye(3)*.01;r[:2,:2]+=.02
    v=pd.DataFrame(np.sqrt(1-p.to_numpy()@r),columns=p.columns);obs=p,q,v,pd.Series(np.ones(3))
    leaves=[(i,) for i in range(3)];lp=PoolLP(obs,[*leaves,(0,1)])
    control=search(PoolView(lp,leaves),leaves,seconds=1,repair=True)
    assert len(control['path'])==1 and control['reason']=='pool_local_optimum'
    treatment=search(PoolView(lp,[*leaves,(0,1)]),leaves,seconds=1,repair=True)
    assert treatment['path'][-1]['objective']<control['path'][-1]['objective']
    lp.close()


def test_reader_uses_declared_nested_total_snapshots_without_changing_day_order():
    import tempfile
    from research_experiments.wzzt_fine.common import observed
    root=Path(__file__).resolve().parents[2]/'tmp'
    with tempfile.TemporaryDirectory(dir=root) as directory:
        assert Path(directory).resolve().is_relative_to(root.resolve())
        path=Path(directory)/'data.npz';values=np.broadcast_to(np.arange(96)[None,:,None],(3,96,2)).copy()
        values+=np.arange(3)[:,None,None]*1000
        np.savez_compressed(path,labels=[101,102],P=values,Q=values,V=values,root=values[:,:,0])
        for N in SIZES:
            p,q,v,voltage=observed(path,N)
            expected=values[:,sample_indices(N),0].ravel()
            assert len(p)==N and np.array_equal(p.iloc[:,0],expected)
            assert np.array_equal(voltage,expected)
        assert len(observed(path)[0])==288


def test_public_inference_rejects_misaligned_validation_labels_before_fitting():
    import pytest
    from research_experiments.wzzt_fine.pipeline import infer
    p=pd.DataFrame(np.ones((3,2)),columns=[101,102]);obs=p,p,p,pd.Series(np.ones(3))
    wrong=tuple(f.iloc[:,::-1] for f in obs[:3])+(obs[3],)
    with pytest.raises(ValueError,match='ordered terminal labels'):infer(obs,wrong)
