"""Held-out windows for the simple existing-z integer-choice model.

Protocol: 2 historical fits x 2 physics x 2 source buses x 2 profiles x 20
independent test seeds. Each test window uses 64 synthetic AC snapshots.
No fitting/threshold tuning on these test windows. Top-3 is a ranked list,
not a statistically calibrated confidence region. No multi-source claim.
"""
from pathlib import Path
import hashlib
import json
import networkx as nx
import numpy as np
try:
    from scripts import real_feeder_phase_wzzt as ph
except ModuleNotFoundError:
    import real_feeder_phase_wzzt as ph


def main():
    root=ph.base.ROOT/'theft_wzzt/outputs/theft_real65037'
    out=root/'selected_z_heldout'
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(history_seeds=[0,1],physics=['balanced','unbalanced'],source_buses=[5,10],
                  conditions=['persistent','intermittent'],replicates=20,test_seed_start=90000,samples=64,
                  test_pairing='The same 20 load/noise seeds are paired across methods, locations and conditions; 160 distinct physics/source/profile windows, evaluated under 2 historical fits',
                  selection='exact exhaustive evaluation of the one-hot existing-z integer objective; historical parameters frozen',
                  amplitude='head-minus-meters, corrected by estimated phase/neutral losses calibrated only on history',
                  top3='ranked candidates, not calibrated confidence',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    feeder=ph.base.read_feeder();directed=nx.bfs_tree(feeder.graph,0);rows=[]
    for history_seed in protocol['history_seeds']:
        for balanced in (True,False):
            label='balanced' if balanced else 'unbalanced'
            saved=json.loads((root/f'v3_tree_selection/seed{history_seed}_{label}_history.json').read_text())
            supports=tuple(frozenset(s) for s in saved['candidates'][0]['supports'])
            history=ph.phase_sample(feeder,74000+history_seed,128,balanced)
            raw=ph.phase_loss(history,supports,saved)
            scale=float(np.clip(raw@(history['p0']-history['p'].sum(axis=1))/(raw@raw),.2,5))
            z,_=ph.features(history,supports);weights=np.asarray(saved['weights'])
            a=np.array([[h<=s for h in supports] for s in supports],float)
            response=z.T@((weights[:,0]+.4*weights[:,1])[:,None]*a)
            for bus in protocol['source_buses']:
                descendants=nx.descendants(directed,bus)|{bus}
                truth=frozenset(i for i,node in enumerate(feeder.meters) if node in descendants)
                for condition in protocol['conditions']:
                    cp=out/f'history{history_seed}_{label}_bus{bus}_{condition}.json'
                    if cp.exists():block=json.loads(cp.read_text())
                    else:
                        block=[]
                        for rep in range(protocol['replicates']):
                            current=ph.phase_sample(feeder,90000+rep,64,balanced,condition,bus)
                            amp=np.maximum(current['p0']-current['p'].sum(axis=1)-scale*ph.phase_loss(current,supports,saved),0)
                            normal=ph.phase_predict(current,supports,weights)
                            error=current['yp'][:,:,:,None]-normal[:,:,:,None]-amp[:,None,None,None]/3*response[None,:,None,:]
                            costs=np.mean(abs(error),axis=(0,1,2))
                            ordered=np.argsort(costs,kind='stable');tied=np.flatnonzero(np.isclose(costs,costs.min(),rtol=0,atol=1e-9))
                            block.append(dict(history_seed=history_seed,balanced=balanced,source_bus=bus,condition=condition,replicate=rep,
                                              truth_in_pool=truth in supports,unique_correct=len(tied)==1 and supports[tied[0]]==truth,
                                              truth_in_minimum_set=any(supports[i]==truth for i in tied),minimum_set_size=len(tied),
                                              truth_in_top3=any(supports[i]==truth for i in ordered[:3]),
                                              chosen_support=sorted(supports[ordered[0]]),tied_supports=[sorted(supports[i]) for i in tied],
                                              amplitude_mae_kw=float(np.mean(abs(amp-current['amplitude'])))))
                        cp.write_text(json.dumps(block,indent=2),encoding='utf-8')
                    rows.extend(block)
                    print(json.dumps(dict(history_seed=history_seed,balanced=balanced,bus=bus,condition=condition,n=len(block),
                                          unique_correct=sum(r['unique_correct'] for r in block),minset_coverage=sum(r['truth_in_minimum_set'] for r in block),top3=sum(r['truth_in_top3'] for r in block))),flush=True)
    groups=[]
    for balanced in (True,False):
        for bus in protocol['source_buses']:
            for condition in protocol['conditions']:
                selected=[r for r in rows if (r['balanced'],r['source_bus'],r['condition'])==(balanced,bus,condition)]
                groups.append(dict(balanced=balanced,bus=bus,condition=condition,evaluations=len(selected),
                                   truth_in_pool=sum(r['truth_in_pool'] for r in selected),unique_correct=sum(r['unique_correct'] for r in selected),
                                   minset_coverage=sum(r['truth_in_minimum_set'] for r in selected),top3_coverage=sum(r['truth_in_top3'] for r in selected),
                                   mean_minset_size=float(np.mean([r['minimum_set_size'] for r in selected])),
                                   amplitude_mae_kw=float(np.mean([r['amplitude_mae_kw'] for r in selected]))))
    (out/'summary.json').write_text(json.dumps(dict(protocol=protocol,groups=groups,total_evaluations=len(rows)),indent=2),encoding='utf-8')
    print(json.dumps(groups,indent=2),flush=True)


if __name__=='__main__':main()
