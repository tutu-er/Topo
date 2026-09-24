"""Prespecified duration/switch boundary checks, no method tuning."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'theft_wzzt';sys.path.insert(0,str(ROOT))
from paper_study.pilot import entry
from paper_study.lp_profile import lp_profile
from paper_study.profile import fixed_profile
from theft_wzzt.theft.identified_tree import load_identified,to_theft_tree,region_label
from theft_wzzt.theft.theft_simulation import TheftSpec,simulate_theft_scenarios
OUT=ROOT/'outputs/theft_paper';dest=OUT/'boundary';dest.mkdir(exist_ok=True)
protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),replicates=list(range(6000,6020)),durations=[1,6,16,32,96],amplitude_kw=8,buses=[2,108],switch='2 at45:48 then108 at48:51',window=[32,64],policy='Boundary characterization only; no retuning or pooling with matched calibration coverage claims',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path=OUT/'boundary_protocol.json'
if not path.exists():path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
identified=load_identified(ROOT/'outputs/theft/identified_tree.json');weights=to_theft_tree(identified)[1]
for duration in [1,6,16,32,96,'switch']:
    for bus in ([2,108] if duration!='switch' else [None]):
        for rep in range(6000,6020):
            target=dest/f'{duration}_{bus}_{rep}.json'
            if target.exists():continue
            if duration=='switch':
                a=np.zeros(96);b=np.zeros(96);a[45:48]=8;b[48:51]=8
                specs=(TheftSpec(2,tuple(a)),TheftSpec(108,tuple(b)))
                true_buses=[2,108]
            else:
                a=np.zeros(96);start=(96-duration)//2;a[start:start+duration]=8
                specs=(TheftSpec(bus,tuple(a)),);true_buses=[bus]
            net,s=simulate_theft_scenarios('paper15',specs,replicate=rep)
            tree,data,bounds,_,_=entry.prepare(s,identified,(32,64))
            fits=[lp_profile(tree,data,bounds),fixed_profile(tree,data,weights,stable=True),fixed_profile(tree,data,weights,stable=False)]
            truths=[region_label(b,net,identified) for b in true_buses]
            for fit in fits:
                fit['profile_candidate_min']=min(r['loss'] for r in fit['profile'])
            target.write_text(json.dumps(dict(duration=duration,bus=bus,replicate=rep,truth_regions=truths,results=fits),indent=1),encoding='utf-8')
        print(duration,bus,'done',flush=True)
