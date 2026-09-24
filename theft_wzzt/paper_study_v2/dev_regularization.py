import json
import numpy as np
from paper_study.pilot import entry,ROOT
from paper_study_v2.regularized import regularized_profile
from theft_wzzt.theft.theft_simulation import TheftSpec,simulate_theft_scenarios
from theft_wzzt.theft.identified_tree import load_identified,to_theft_tree,region_label
OUT=ROOT/'outputs/theft_paper_v2'
protocol=dict(stage='development',replicates=list(range(7000,7005)),conditions=['nominal','drift10','hetero10'],strengths=[3,10,30],amps=[2,4,8],buses=[2,108],selection='Select a single strength using these development rows only; report every strength and boundary condition')
p=OUT/'regularization_development_protocol.json'
if not p.exists():p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
identified=load_identified(ROOT/'outputs/theft/identified_tree.json');weights=to_theft_tree(identified)[1]
for rep in protocol['replicates']:
    for condition in protocol['conditions']:
        for bus in protocol['buses']:
            for amp in protocol['amps']:
                target=OUT/f'regdev_{rep}_{condition}_{bus}_{amp}.json'
                if target.exists():continue
                a=np.zeros(96);a[45:51]=amp
                options={'impedance_scale':1.1} if condition=='drift10' else {}
                if condition=='hetero10':
                    def modify(net):
                        rng=np.random.default_rng(rep+500000);f=rng.uniform(.9,1.1,len(net.branches))
                        net.branches.loc[:,['r_ohm','x_ohm']]=net.branches[['r_ohm','x_ohm']].to_numpy()*f[:,None]
                        return net
                    options['case_modifier']=modify
                net,s=simulate_theft_scenarios('paper15',(TheftSpec(bus,tuple(a)),),replicate=rep,**options)
                tree,data,bounds,_,_=entry.prepare(s,identified,(32,64));truth=region_label(bus,net,identified)
                fits=[regularized_profile(tree,data,weights,bounds,strength=k) for k in protocol['strengths']]
                for fit in fits:fit['correct']=fit['best_location']==truth
                target.write_text(json.dumps(dict(replicate=rep,condition=condition,bus=bus,amp=amp,truth=truth,results=fits),indent=1),encoding='utf-8')
        print(rep,condition,'done',flush=True)
