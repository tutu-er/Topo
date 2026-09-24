import json
from pathlib import Path
import numpy as np
from paper_study.pilot import entry,ROOT
from paper_study_v2.profile import fit_profile
from paper_study.profile import fixed_profile
from paper_study.lp_profile import lp_profile
from theft_wzzt.theft.theft_simulation import TheftSpec,simulate_theft_scenarios
from theft_wzzt.theft.identified_tree import load_identified,to_theft_tree,region_label
OUT=ROOT/'outputs/theft_paper_v2';OUT.mkdir(exist_ok=True)
protocol=dict(stage='development',replicates=list(range(7000,7010)),conditions=['nominal','drift10','hetero10'],amps=[2,4,8],buses=[2,108],methods=['global_rx1','global_rx2','fixed_stable','stable_positive_lp'],reason='V1 held-out results disclosed low localization and threshold-transfer performance; V1 is retained. V2 uses entirely new development/calibration/test seeds.')
path=OUT/'development_protocol.json'
if not path.exists():path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
identified=load_identified(ROOT/'outputs/theft/identified_tree.json');weights=to_theft_tree(identified)[1]
for rep in range(7000,7010):
    for condition in protocol['conditions']:
        for bus in [2,108]:
            for amp in [2,4,8]:
                target=OUT/f'dev_{rep}_{condition}_{bus}_{amp}.json'
                if target.exists():continue
                a=np.zeros(96);a[45:51]=amp
                options={'impedance_scale':1.1} if condition=='drift10' else {}
                if condition=='hetero10':
                    def modify(net):
                        rng=np.random.default_rng(rep+500000)
                        factors=rng.uniform(.9,1.1,len(net.branches))
                        net.branches.loc[:,['r_ohm','x_ohm']]=net.branches[['r_ohm','x_ohm']].to_numpy()*factors[:,None]
                        return net
                    options['case_modifier']=modify
                net,s=simulate_theft_scenarios('paper15',(TheftSpec(bus,tuple(a)),),replicate=rep,**options)
                tree,data,bounds,_,_=entry.prepare(s,identified,(32,64))
                fits=[fit_profile(tree,data,weights,two_scales=b) for b in [False,True]]+[fixed_profile(tree,data,weights,stable=True),lp_profile(tree,data,bounds)]
                truth=region_label(bus,net,identified)
                for fit in fits:fit['correct']=fit['best_location']==truth
                target.write_text(json.dumps(dict(replicate=rep,condition=condition,bus=bus,amp=amp,truth=truth,results=fits),indent=1),encoding='utf-8')
        print(rep,condition,'done',flush=True)
