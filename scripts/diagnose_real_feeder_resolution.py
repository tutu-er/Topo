"""Truth-only diagnostic; never used by the estimator or candidate selection."""
import json
from pathlib import Path
import numpy as np
try:
    from scripts.real_feeder_joint_theft import read_feeder,ac_flow,ROOT
except ModuleNotFoundError:
    from real_feeder_joint_theft import read_feeder,ac_flow,ROOT


def main():
    feeder=read_feeder();base=feeder.snapshot[None]*.85
    columns=[]
    for bus in range(1,53):
        loads=base.copy();loads[:,bus]+=15*(1+.4j)/3
        voltage,_,_,audit=ac_flow(feeder,loads,np.array([230.]))
        columns.append(abs(voltage[0,feeder.meters,:3]-voltage[0,feeder.meters,3:]))
    target=columns[4]
    rows=[]
    for bus,voltage in enumerate(columns,1):
        difference=voltage-target
        rows.append(dict(bus=bus,max_phase_voltage_difference_v=float(np.max(abs(difference))),
                         rms_phase_voltage_difference_v=float(np.sqrt(np.mean(difference**2)))))
    rows.sort(key=lambda row:row['max_phase_voltage_difference_v'])
    result=dict(purpose='Evaluation-only voltage signature resolution, not inference or a confidence set',
                operating_point='0.85 times original unbalanced snapshot, root 230V known',
                fixed_extra_kw=15,q_over_p=.4,target_bus=5,
                experiment_phase_meter_noise_std_at_230v=.0002*230,ranked_positions=rows)
    out=ROOT/'theft_wzzt/outputs/theft_real65037/resolution_diagnostic.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(rows[:8],indent=2))


if __name__=='__main__':main()
