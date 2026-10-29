from time import perf_counter
from research_experiments.wzzt_fine.pipeline import infer as base_infer
from research_experiments.wzzt_fine.common import PoolView
from research_experiments.wzzt_retune.engine import PoolLP,canonical,search
from research_experiments.wzzt_refine.core import scored,select_pre,finalize,regularized_pool,backward,two_step


def guard(lp,points,train,validation,cal):
    chosen=select_pre(points);deletion=backward(lp,chosen['supports'],validation,seconds=15,max_deletions=1)
    accepted=[p for p in deletion['points'] if p['validation_mae']<=chosen['validation_mae']+1e-12]
    return finalize(points+accepted,train,validation,cal),deletion,accepted


def infer(train,validation,*,lambda_ablation=False,budget_ablation=False,pair_ablation=False,
          mode='all',checkpoint=None):
    t0=perf_counter();r=base_infer(train,validation,extensions=False,checkpoint=checkpoint)
    multi=r['base_points'];chosen=select_pre(multi);cal=r['calibration'];pool=r['pool']
    r.update(domains={},candidates={},experiment='frozen paper panel 20261029')
    if mode=='multistart':r['seconds']=perf_counter()-t0;return r
    if mode=='all':
        lp=PoolLP(train,pool,'raw')
        try:
            trajectory=search(lp,[(i,) for i in range(len(r['labels']))],seconds=25,repair=True)
            points=[scored(p,validation,'leaf25') for p in trajectory['path']]
            r['searches']['leaf25']=trajectory
            r['models']['WZZT_leaf_path']=finalize(points,train,validation,cal)
            r['models']['WZZT_leaf_bank']=finalize(r['rnj_bank']+points,train,validation,cal)
            r['models']['multistart_guard0'],d,g=guard(lp,multi,train,validation,cal)
            r['searches']['multistart_delete']=d;r['candidates']['multistart_guard0']=g;r['LP']['leaf_and_delete']=lp.solves
        finally:lp.close()
    aug=regularized_pool(train,pool);r['augmentation']=aug
    domains={'ridge_union':aug['pool']}
    if mode=='all':domains={'restart_control':pool,**domains}
    if lambda_ablation:
        for name,strength in (('ridge_1e3',.001),('ridge_1e2',.01)):
            members=list(pool)
            for front in aug['fronts']:
                if front['strength']==strength:
                    for tree in front['tree_families']:members.extend(tree['family'])
            domains[name]=canonical(members)
    if budget_ablation:domains.update(restart100=pool,ridge100=aug['pool'])
    r['domains']=domains
    for name,domain in domains.items():
        lp=PoolLP(train,aug['pool'],'raw')
        try:
            trajectory=search(PoolView(lp,domain),chosen['supports'],seconds=100 if name.endswith('100') else 25,repair=True)
            points=[scored(p,validation,name) for p in trajectory['path']]
            r['models'][name]=finalize(multi+points,train,validation,cal);r['candidates'][name]=points;r['searches'][name]=trajectory
            if name=='ridge_union' and mode=='all':
                bank=multi+points
                r['models']['ridge_guard0'],d,g=guard(lp,bank,train,validation,cal)
                r['searches']['ridge_delete']=d;r['candidates']['ridge_guard0']=g
                if pair_ablation:
                    pair=two_step(lp,select_pre(bank)['supports'],validation,seconds=25,width=5)
                    r['searches']['ridge_two_step']=pair;r['candidates']['ridge_two_step']=pair['points']
                    r['models']['ridge_two_step']=finalize(bank+pair['points'],train,validation,cal)
                    r['models']['ridge_two_step_guard0'],pd,pg=guard(lp,bank+pair['points'],train,validation,cal)
                    r['searches']['ridge_pair_delete']=pd;r['candidates']['ridge_two_step_guard0']=pg
            r['LP'][name]=lp.solves
        finally:lp.close()
        if checkpoint:checkpoint(r)
    r['seconds']=perf_counter()-t0
    return r
