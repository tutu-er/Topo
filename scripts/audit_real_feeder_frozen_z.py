"""Exact existing-z choice with historical weights frozen; diagnostic baseline.

Location is one-hot integer. Costs are evaluated directly for each existing z;
the ILP optimum is audited against exhaustive discrete costs. No theft LP or
continuous mixture of locations is used. This is not joint R/X estimation.
"""
import json
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
try:
    from scripts import real_feeder_phase_wzzt as ph
except ModuleNotFoundError:
    import real_feeder_phase_wzzt as ph


def main():
    root=ph.base.ROOT/'theft_wzzt/outputs/theft_real65037'
    feeder=ph.base.read_feeder();rows=[]
    for seed in range(2):
        for balanced in (True,False):
            label='balanced' if balanced else 'unbalanced'
            saved=json.loads((root/f'v3_tree_selection/seed{seed}_{label}_history.json').read_text())
            supports=tuple(frozenset(s) for s in saved['candidates'][0]['supports'])
            history=ph.phase_sample(feeder,74000+seed,128,balanced)
            raw=ph.phase_loss(history,supports,saved)
            scale=float(np.clip(raw@(history['p0']-history['p'].sum(axis=1))/(raw@raw),.2,5))
            for condition in ('persistent','intermittent'):
                current=ph.phase_sample(feeder,76000+seed,64,balanced,condition)
                amp=np.maximum(current['p0']-current['p'].sum(axis=1)-scale*ph.phase_loss(current,supports,saved),0)
                costs=np.array([np.mean(abs(ph.phase_predict(current,supports,saved['weights'],amp,h)-current['yp'])) for h in range(len(supports))])
                opt=milp(costs*1e4,integrality=np.ones(len(costs)),bounds=Bounds(0,1),constraints=LinearConstraint(np.ones((1,len(costs))),1,1))
                assert opt.status==0 and abs(opt.fun/1e4-costs.min())<1e-8
                chosen=int(np.argmax(opt.x));rank=np.argsort(costs)
                true_support=frozenset([2,3,4]) # EVALUATION ONLY: physical source 5.
                tied=np.flatnonzero(np.isclose(costs,costs.min(),rtol=0,atol=1e-9))
                rows.append(dict(seed=seed,balanced=balanced,condition=condition,chosen_support=sorted(supports[chosen]),
                                 source_region_correct=supports[chosen]==true_support,status='optimal',
                                 tied_minimum_supports=[sorted(supports[i]) for i in tied],
                                 truth_in_minimum_set=any(supports[i]==true_support for i in tied),
                                 uniquely_correct=len(tied)==1 and supports[chosen]==true_support,
                                 tie_cost_tolerance=1e-9,
                                 true_support_rank=int(np.flatnonzero(rank==supports.index(true_support))[0])+1 if true_support in supports else None,
                                 top_three=[dict(support=sorted(supports[i]),phase_mae_v_approx=float(costs[i]*2000/460)) for i in rank[:3]]))
    for row in rows:
        label='balanced' if row['balanced'] else 'unbalanced'
        seed=row['seed']
        hist=json.loads((root/f'v3_tree_selection/seed{seed}_{label}_history.json').read_text())
        scenario_path=root/f"v3_tree_selection/seed{seed}_{label}_{row['condition']}.json"
        if scenario_path.exists():
            fitted=json.loads(scenario_path.read_text())['fits']['joint']
            hsupports=hist['candidates'][0]['supports']
            hcost=(128*hist['train_mae']+64*row['top_three'][0]['phase_mae_v_approx']*460/2000)/192
            hcost+=.00015*sum(len(s)>1 for s in hsupports)+.001*float(np.array(hist['weights']).sum())
            jcost=fitted['train_mae']+.00015*sum(len(s)>1 for s in fitted['supports'])+.001*float(np.array(fitted['weights']).sum())
            row['historical_feasible_point_objective']=hcost
            row['joint_incumbent_objective']=jcost
            row['historical_point_improves_joint_incumbent']=hcost+1e-7<jcost
    output=dict(purpose='Fixed historical weights, exact selection of an existing z; separates search failure from model/history error',rows=rows)
    (root/'frozen_z_diagnostic.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
