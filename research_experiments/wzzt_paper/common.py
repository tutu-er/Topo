from pathlib import Path
import numpy as np
import pandas as pd

CASES=('paper15','soumalas11','flynn16','pengwah18')
SIZES=(24,48,96,288)
SIGMAS=(0.,.0002,.0005,.001)
MAIN=('NJ','RNJ_native','RNJ_L1_bank','WZZT_leaf_path','WZZT_leaf_bank',
      'baseline25','multistart','restart_control','ridge_union','multistart_guard0','ridge_guard0')
STRESSES=('ideal_ac','low_xr','higher_load','light_load','weak_excitation',
          'high_pq_noise','root_noise','ar1_errors','outliers','tap_step')
AC_METHODS=('NJ','multistart','ridge_union')


def indices(N):
    if N not in SIZES:raise ValueError('outside frozen grid')
    return np.arange(0,96,288//N)


def observed(path,N=None):
    with np.load(Path(path)) as d:
        pick=np.arange(96) if N is None else indices(N);labels=d['labels'].astype(int).tolist()
        frames=[pd.DataFrame(d[k][:,pick].reshape(-1,len(labels)),columns=labels) for k in ('P','Q','V')]
        root=pd.Series(d['root'][:,pick].reshape(-1))
    return *frames,root


def jobs():
    out=[]
    def add(panel,case,bundle,condition,sigma,N):
        ident=f'{panel}__{case}__b{bundle}__{condition}__v{round(sigma*10000):04d}__N{N}'
        p=dict(job_id=ident,panel=panel,case=case,bundle=bundle,condition=condition,voltage_sigma=sigma,N=N,
               lambda_ablation=panel=='main' and bundle<3 and sigma>0 and N in (24,96),
               budget_ablation=panel=='main' and bundle<5 and sigma==.001 and N in (96,288),
               pair_ablation=panel=='main' and bundle==0 and sigma in (.0005,.001) and N in (48,288),
               ac_planned=(panel=='main' and sigma in (.0002,.001) and N in (48,288)) or (panel!='main' and bundle==0))
        methods=list(MAIN)
        if p['lambda_ablation']:methods+=['ridge_1e3','ridge_1e2']
        if p['budget_ablation']:methods+=['restart100','ridge100']
        if p['pair_ablation']:methods+=['ridge_two_step','ridge_two_step_guard0']
        p['methods']=methods;out.append(p)
    for c in CASES:
        for b in range(10):
            for s in SIGMAS:
                for N in SIZES:add('main',c,b,'reference',s,N)
    for c in CASES:
        for b in range(5):
            for condition in STRESSES:
                s=0. if condition=='ideal_ac' else .0005
                for N in (48,288):add('stress',c,b,condition,s,N)
    for n in (8,16,24,32):
        for t in range(3):
            for b in range(3):
                for s in (.0002,.001):
                    for N in (48,288):add('topology',f'random{n}_t{t}',b,'reference',s,N)
    for b in range(5):
        for s in (.0002,.001):
            for N in (48,288):add('case33','case33_terminal',b,'reference',s,N)
    return out


def noise_errors(z,rho):
    out=z.copy()
    for i in range(1,len(out)):out[i]=rho*out[i-1]+np.sqrt(1-rho*rho)*z[i]
    return out
