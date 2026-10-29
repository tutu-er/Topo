"""Private observed-only historical interface; no simulation or identification."""
from pathlib import Path
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import make_observed_scenario

BASE=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
SOURCE=BASE/'wzzt_data_calibration/flow_runs/paper15_iflex_v1'
OUT=BASE/'dr_paper15_update_build'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

manifest=json.loads((SOURCE/'data_manifest.json').read_text());dates=manifest['forecast_history_dates']
past={};sources={}
for date in dates:
    assert date<manifest['action_date']
    path=SOURCE/'history'/(date+'.pkl');sources[str(path)]=sha(path)
    with path.open('rb') as stream:obs=pickle.load(stream)['observed']
    past[date]=make_observed_scenario(obs['P_terminal'],obs['Q_terminal'],obs['V_terminal'],obs['root_voltage'],obs['name'])

metadata=dict(base_kv=.4,base_mva=.2,base_kw=200.,node_capacity_kw=45.,
    baseline_P='measured aggregate of3controlhouseholds',baseline_Q='heterogeneous synthetic residentialPF',
    current_DR_increment=dict(delta_p_kw=.2,delta_q_kvar=.07,delta_q_over_delta_p=.35),
    current_price_budget_generic_currency=1.6,
    units=dict(P_terminal='pu',Q_terminal='pu',V_terminal='pu',V_root_observed='pu'),
    training_dates=dates[-3:-1],validation_dates=dates[-1:],forecast_history_dates=dates,
    source_role='datedpastobservationsonly; no truth/future channels')
packet=dict(training=[past[d] for d in dates[-3:-1]],validation=[past[dates[-1]]],
    metadata=metadata,terminal_labels=past[dates[-1]]['P_terminal'].columns.tolist())
assert set(packet['training'][0])=={'name','P_terminal','Q_terminal','V_terminal','V_root_observed','drop_target','units'}
with (OUT/'strictpast_observed.pkl').open('wb') as stream:pickle.dump(packet,stream,protocol=4)

p=np.mean(np.stack([past[d]['P_terminal'].to_numpy() for d in dates]),axis=0)*200.
q=np.mean(np.stack([past[d]['Q_terminal'].to_numpy() for d in dates]),axis=0)*200.
ix=pd.RangeIndex(24,name='hour_start');labels=packet['terminal_labels']
forecast=dict(P_kw=pd.DataFrame(p,index=ix,columns=labels),Q_kvar=pd.DataFrame(q,index=ix,columns=labels),metadata=metadata)
with (OUT/'historical_forecast.pkl').open('wb') as stream:pickle.dump(forecast,stream,protocol=4)
checks=dict(reduction_hour=19,rebound_hour=20,
    P_nonnegative=bool(np.min(p[19]-.2)>=0),Q_nonnegative=bool(np.min(q[19]-.07)>=0),
    P_capacity=bool(np.max(p[20]+.2)<=45.),
    feasible_labels=[int(label) for label,ok in zip(labels,(p[19]>=.2)&(q[19]>=.07)&(p[20]+.2<=45)) if ok],
    forecast_only=True,future_values_accessed=False)
receipt=dict(metadata=metadata,source_sha256=sources,script_sha256=sha(__file__),
    output_sha256={name:sha(OUT/name) for name in ['strictpast_observed.pkl','historical_forecast.pkl']},
    rows=dict(training=48,validation=24,forecast_source=240),forecast_checks=checks,
    export='private cache only; do not commit aggregated trajectories',
    full_phase_scope='no actual_days, heldoutresponsebanks, Source or truth fields accessed')
(OUT/'strictpast_observed_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(rows=receipt['rows'],checks=checks,outputs=receipt['output_sha256'])))
