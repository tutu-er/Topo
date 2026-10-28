from pathlib import Path
import numpy as np
import pandas as pd

SIZES=(24,36,48,72,96)
SIGMAS=(.0002,.0005,.001)
MAIN=("NJ","RNJ_native","RNJ_L1_bank","baseline25","multistart","restart_control","ridge_1e3","ridge_1e2","ridge_union")
TARGET=("delete_all","delete_guard0","delete_guard1pct","delete_guard3pct","two_step_w3","two_step_w5")
AC_METHODS=("NJ","multistart","ridge_union")


def validate_observations(train,validation):
    labels=list(train[0].columns)
    if len(labels)<2 or len(set(labels))!=len(labels) or 1 in labels:raise ValueError("unique terminal labels excluding root 1 required")
    for obs in (train,validation):
        if len(obs[0])%3:raise ValueError("three equal physical-day blocks required")
        if any(list(f.columns)!=labels or not f.index.equals(obs[3].index) for f in obs[:3]):
            raise ValueError("P/Q/V/root indices and ordered terminal labels must align")
        if any(not np.isfinite(f.to_numpy()).all() for f in obs):raise ValueError("finite measurements required")
        if (obs[2].to_numpy()<=0).any() or (obs[3].to_numpy()<=0).any():raise ValueError("positive voltage magnitudes required")


def sample_indices(N):
    if N not in SIZES:raise ValueError("sample size outside declared fine grid")
    coarse=list(range(0,96,12));extra16=[i for i in range(0,96,6) if i not in coarse]
    extra32=[i for i in range(0,96,3) if i not in coarse+extra16]
    order=coarse+extra16[::2]+extra16[1::2]+extra32[::2]+extra32[1::2]
    return np.sort(order[:N//3])


def observed(path,N=None):
    with np.load(Path(path)) as d:
        pick=np.arange(96) if N is None else sample_indices(N)
        labels=d["labels"].astype(int).tolist()
        frames=[pd.DataFrame(d[name][:,pick].reshape(-1,len(labels)),columns=labels) for name in ("P","Q","V")]
        root=pd.Series(d["root"][:,pick].reshape(-1))
    return *frames,root


class PoolView:
    def __init__(self,lp,pool):self.lp,self.pool,self.n=lp,tuple(pool),lp.n
    def fit(self,family):return self.lp.fit(family)
