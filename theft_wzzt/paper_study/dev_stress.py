import json
import numpy as np
from paper_study.pilot import entry,ROOT,OUT
from paper_study.profile import fixed_profile
from paper_study.lp_profile import lp_profile
from theft_wzzt.theft.theft_simulation import TheftSpec,simulate_theft_scenarios
from theft_wzzt.theft.identified_tree import load_identified,to_theft_tree,region_label
identified=load_identified(ROOT/'outputs/theft/identified_tree.json');weights=to_theft_tree(identified)[1]
for rep in range(200,205):
    for bus in [None,2,108]:
        path=OUT/f'dev_drift_{rep}_{bus}.json'
        if path.exists():continue
        a=np.zeros(96);a[45:51]=8
        net,s=simulate_theft_scenarios('paper15',None if bus is None else (TheftSpec(bus,tuple(a)),),replicate=rep,impedance_scale=1.1)
        tree,data,bounds,_,_=entry.prepare(s,identified,(32,64))
        truth=None if bus is None else region_label(bus,net,identified)
        rows=[fixed_profile(tree,data,weights,stable=True),lp_profile(tree,data,bounds)]
        for row in rows:row.update(truth_region=truth,correct=None if bus is None else row['best_location']==truth)
        path.write_text(json.dumps(dict(replicate=rep,bus=bus,results=rows),indent=2),encoding='utf-8')
        print(rep,bus,[(r['method'],round(r['gain'],1),r['correct']) for r in rows],flush=True)
