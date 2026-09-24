import json
from pathlib import Path
from time import perf_counter
import numpy as np
from paper_study.pilot import entry, ROOT, OUT
from paper_study.profile import fixed_profile
from paper_study.lp_profile import lp_profile
from theft_wzzt.theft.theft_simulation import TheftSpec, simulate_theft_scenarios
from theft_wzzt.theft.identified_tree import load_identified,to_theft_tree,region_label

def main():
    identified=load_identified(ROOT/'outputs/theft/identified_tree.json')
    weights=to_theft_tree(identified)[1]
    for rep in range(200,205):
        for bus in [None,2,108]:
            for amp in ([0] if bus is None else [2,4,8]):
                key=f'dev_{rep}_{bus}_{amp}'
                path=OUT/f'{key}.json'
                if path.exists():continue
                a=np.zeros(96);a[45:51]=amp
                start=perf_counter()
                net,s=simulate_theft_scenarios('paper15',None if bus is None else (TheftSpec(bus,tuple(a)),),replicate=rep)
                measured={k:s[k] for k in entry.MATRIX_CHANNELS+entry.SERIES_CHANNELS+('scenario_settings',)}
                truth=None if bus is None else region_label(bus,net,identified)
                rows=[]
                for window in [(44,52),(32,64)]:
                    tree,data,bounds,_,_=entry.prepare(measured,identified,window)
                    results=[fixed_profile(tree,data,weights,stable=True),lp_profile(tree,data,bounds)]
                    for result in results:
                        result.update(window=window,truth_region=truth,exact_region_match=result['best_location']==truth if truth is not None else None)
                        result['truth_profile_gap']=None if truth is None else next(row['loss'] for row in result['profile'] if row['location']==truth)-result['h1_loss']
                        rows.append(result)
                path.write_text(json.dumps(dict(replicate=rep,bus=bus,amplitude_kw=amp,results=rows),indent=2),encoding='utf-8')
                print(key, 'seconds',round(perf_counter()-start,2), [(r['method'],r['window'][1]-r['window'][0],r['exact_region_match'],round(r['gain'],1)) for r in rows],flush=True)
if __name__=='__main__':main()
