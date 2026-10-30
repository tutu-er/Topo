"""Read historical observations only, after freezing a uniform synthetic task.

This checks a numerical decomposition, not device availability inferred from AMI.
No old test/truth, future profiles, AC or optimization is used.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys
import numpy as np

REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
BASE=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
ROOT=BASE/'wzzt_dr_coded_qualification_v1'
BRIDGE=REPO/'research_experiments/wzzt_dr_selection/dr_competition_bridge'

def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ROOT.mkdir(exist_ok=True)
    protocol_path=ROOT/'protocol.json'
    if protocol_path.exists():raise RuntimeError('Existing qualification protocol retained; do not rerun.')
    inventory=REPO/'research_experiments/wzzt_dr_selection/evidence/scenario_review_v1/source_inventory.json'
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        case_rule='paper15 b8/b9, all4 original noise levels, N288 reference, chosen by candidate top-layer crossings',
        device=dict(baseline_kw=.2,baseline_kvar=.07,upper_kw=.4,
            start_hour=19,end_hour=21,initial_energy_kWh=0,final_energy_kWh=.4,
            efficiency=1.,non_V2G=True),
        interpretation='Declared simulated charge task; neither empirically calibrated EV nor device availability inferred from terminal load.',
        historical_check='Only numerical nonnegative decomposition of observed window; noisy AMI cannot prove a physical device exists.',
        input_sha256={str(inventory):sha(inventory)},
        source_sha256={str(Path(__file__)):sha(Path(__file__)),str(BRIDGE/'physical_bridge.py'):sha(BRIDGE/'physical_bridge.py')},
        no_new_profile=True,no_LP=True,no_AC=True,no_old_test_or_truth=True,
        status='frozen_before_reading_observed_values')
    write(protocol_path,protocol)
    rows=[r for r in json.loads(inventory.read_text(encoding='utf-8'))['rows']
          if r['case']=='paper15' and r['bundle'] in (8,9) and r['N']==288]
    assert len(rows)==8
    sys.path.insert(0,str(BRIDGE))
    from physical_bridge import load_history,forecast_mean
    records=[]
    for row in rows:
        h=load_history(row); forecast=forecast_mean(h)
        p=np.stack([d['P_terminal'].to_numpy()[76:84]*200 for d in h['training']+h['validation']])
        q=np.stack([d['Q_terminal'].to_numpy()[76:84]*200 for d in h['training']+h['validation']])
        fp=forecast['P_pu'][76:84]*200;fq=forecast['Q_pu'][76:84]*200
        label_records=[]
        for k,label in enumerate(h['labels']):
            label_records.append(dict(label=label,min_observed_P_kw=float(p[:,:,k].min()),
                min_observed_Q_kvar=float(q[:,:,k].min()),
                observed_negative_remainder_cells=int((p[:,:,k]<.2-1e-12).sum()+(q[:,:,k]<.07-1e-12).sum()),
                forecast_min_P_kw=float(fp[:,k].min()),forecast_min_Q_kvar=float(fq[:,k].min()),
                forecast_nonnegative_remainder=bool((fp[:,k]>=.2-1e-12).all() and (fq[:,k]>=.07-1e-12).all())))
        records.append(dict(job_id=row['job_id'],status='complete',history_receipt=h['receipt'],
            nodes=label_records,observed_days=6,old_training_rows=288,old_validation_rows=288))
    summary=dict(status='complete',records=records,case_count=len(records),device_case_checks=15*len(records),
        forecast_decomposition_pass=sum(n['forecast_nonnegative_remainder'] for r in records for n in r['nodes']),
        historical_all_window_pass=sum(n['observed_negative_remainder_cells']==0 for r in records for n in r['nodes']),
        device_existence_or_availability_proven=False,new_physical_or_solver_calls=0)
    write(ROOT/'qualification.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}))

if __name__=='__main__':main()
