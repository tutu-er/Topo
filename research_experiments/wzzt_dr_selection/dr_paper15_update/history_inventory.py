"""Read only dated past observed P/Q/V; never inspect truth/future fields.

No price optimization, identifier, AC simulation or solver is imported/called.
Pickle envelopes are existing trusted local artifacts; only observed channels
and declared scenario metadata are selected for analysis. No trajectories,
household IDs or node-level demand values are exported.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

BASE=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
SOURCE=BASE/'wzzt_data_calibration/flow_runs/paper15_iflex_v1'
REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
OUT=BASE/'dr_paper15_update_build'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def svd_diagnostics(matrix):
    matrix=np.asarray(matrix,dtype=float)
    rms=np.sqrt(np.mean(matrix**2,axis=0));scaled=matrix/np.maximum(rms,1e-30)
    singular=np.linalg.svd(scaled,compute_uv=False)
    rank=int(np.linalg.matrix_rank(scaled))
    return dict(rows=matrix.shape[0],columns=matrix.shape[1],rank=rank,
        full_column_rank=rank==matrix.shape[1],rms_scaled_singular_values=singular.tolist(),
        rms_scaled_condition=float(singular[0]/singular[-1]) if rank==matrix.shape[1] else None,
        smallest_to_largest=float(singular[-1]/singular[0]),
        rank_interpretation='numerical rank of noisy observed matrix; full rank alone is not statistical identifiability')


def diagnostics(p,q,y):
    qfit=p@np.linalg.lstsq(p,q,rcond=None)[0]
    common=float(np.sum(p*q)/np.sum(p*p))
    profile_diff=[];correlation=[]
    for i in range(p.shape[1]):
        for j in range(i):
            profile_diff.append(float(np.linalg.norm(p[:,i]-p[:,j])/np.sqrt(np.linalg.norm(p[:,i])**2+np.linalg.norm(p[:,j])**2)))
            correlation.append(float(np.corrcoef(p[:,i],p[:,j])[0,1]))
    variations=np.std(p,axis=0)/np.maximum(np.mean(p,axis=0),1e-30)
    return dict(P=svd_diagnostics(p),PQ=svd_diagnostics(np.column_stack([p,q])),
        day_demeaned_P=svd_diagnostics(p-np.mean(p,axis=0)),
        Q_residual_after_linear_P_projection_relative_fro=float(np.linalg.norm(q-qfit)/np.linalg.norm(q)),
        best_common_Q_over_P=common,
        common_Q_over_P_relative_residual=float(np.linalg.norm(q-common*p)/np.linalg.norm(q)),
        relative_P_profile_pair_difference_min_median_max=np.quantile(profile_diff,[0,.5,1]).tolist(),
        pair_P_correlation_min_median_max=np.quantile(correlation,[0,.5,1]).tolist(),
        terminal_P_CV_min_median_max=np.quantile(variations,[0,.5,1]).tolist(),
        Y_channel_count=y.shape[1],
        diagnostic_scope='measured power geometry only; no proxy price objective or topology target')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(SOURCE/'data_manifest.json')
    config=read(SOURCE/'run_state.json')['provenance']['config']
    summary=read(SOURCE/'summary.json')
    cutoff=pd.Timestamp(config['source']['calibration_end_exclusive'])
    action=pd.Timestamp(config['action_date'])
    dates=manifest['forecast_history_dates']; assert max(pd.to_datetime(dates))<cutoff<action
    assert dates==summary['history_dates'] and dates[-3:]==summary['wzzt_training_dates']
    channels=[];daily=[];hashes={str(SOURCE/n):sha(SOURCE/n) for n in ['data_manifest.json','run_state.json','summary.json']}
    schema=None;metadata=None
    for date in dates:
        path=SOURCE/'history'/(date+'.pkl');hashes[str(path)]=sha(path)
        with path.open('rb') as stream:envelope=pickle.load(stream)
        # Access only the observed branch; never index envelope['truth'].
        observed=envelope['observed']
        p=observed['P_terminal'];q=observed['Q_terminal'];v=observed['V_terminal'];root=observed['root_voltage']
        assert p.index.equals(q.index) and p.index.equals(v.index) and p.index.equals(root.index)
        assert p.columns.equals(q.columns) and p.columns.equals(v.columns)
        assert len(p)==24 and p.index[0].date().isoformat()==date and p.index[-1]<action
        for a in (p,q,v,root):assert np.isfinite(np.asarray(a)).all()
        y=root.to_numpy()[:,None]**2-v.to_numpy()**2
        channels.append((date,p.to_numpy(),q.to_numpy(),y))
        daily.append(dict(date=date,**diagnostics(p.to_numpy(),q.to_numpy(),y)))
        if schema is None:
            schema={k:type(observed[k]).__name__ for k in ['name','P_terminal','Q_terminal','V_terminal','root_voltage','drop_target','scenario_settings']}
            metadata={k:observed['scenario_settings'].get(k) for k in ['input_units','estimator_power_units','voltage_units','base_mva','impedance_scale','pq_noise_rel','v_noise_rel','root_meter_noise_rel','interval_hours']}
    def subset(items,rows=None):
        selected=[(p,q,y) if rows is None else (p[rows],q[rows],y[rows]) for _,p,q,y in items]
        return diagnostics(*(np.concatenate([s[k] for s in selected]) for k in range(3)))
    regimes=dict(forecast_history_10days_240rows=subset(channels),
        identification_history_3days_72rows=subset(channels[-3:]),
        identification_history_sparse_3days_18rows=subset(channels[-3:],np.arange(0,24,4)),
        strict_past_fit_2days_48rows=subset(channels[-3:-1]),
        strict_past_validation_1day_24rows=subset(channels[-1:]))
    base_kw=float(metadata['base_mva'])*1000
    forecast=np.mean(np.stack([p for _,p,_,_ in channels]),axis=0)*base_kw
    capacity=45.0
    feasible_by_hour=[]
    for hour in range(23):
        feasible=(forecast[hour]>=.2)&(forecast[hour+1]+.2<=capacity)
        feasible_by_hour.append(dict(reduction_hour=hour,rebound_hour=hour+1,
            feasible_nodes=int(feasible.sum()),infeasible_labels=[int(c) for c,ok in zip(p.columns,feasible) if not ok]))
    # Chronological candidate dates are inherited from an already-written plan,
    # selected by calendar only, not by future trajectories or recovery values.
    planned=[dict(action='2021-02-11',validation='2021-02-12'),dict(action='2021-02-16',validation='2021-02-17')]
    plan=REPO/'research_experiments/wzzt_dr_selection/dr_pool_v2/NEXT_PLAN_DRAFT.md'
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),source_path=str(SOURCE),
        scope='strictly past noisy observed aggregate P/Q/V only; no price/identifier/AC execution',
        source_sha256=hashes,script_sha256=sha(__file__),
        information_cutoff=str(cutoff.date()),original_action_date=str(action.date()),
        history_dates=dates,identifier_history_dates=dates[-3:],
        terminal_count=len(p.columns),terminal_labels=[int(c) for c in p.columns],observed_schema=schema,
        observed_units_and_noise=metadata,
        retained_DR_contract=dict(price_choices=[0,1,2],participation_probabilities=[0,.5,.75],
            response_kw=.2,total_generic_currency_budget=1.6,
            interpretation='retained synthetic Bernoulli whole-event model; do not reuse the old empirical1000NOK price model'),
        forecast_feasibility=dict(source='ten past observed Pmean, physical base conversion only',
            forecast_power_kw_min_max=[float(forecast.min()),float(forecast.max())],
            legacy_toy_1p5_kw_capacity_violated_nodes=int(np.any(forecast>1.5,axis=0).sum()),
            declared_paper15_capacity_kw=capacity,capacity_status='original declared45kWterminalcontract, not a thermal ampacity certificate',
            by_hour=feasible_by_hour,
            predeclared_candidate_window='six-hour shape starts18:00, reduction19:00, rebound20:00; choose clock before future values'),
        household_scope=dict(aggregate_only=True,households_per_terminal=manifest['households_per_terminal'],
            network_households=manifest['network_households'],mapping_seed=manifest['mapping_seed'],mapping_sha256=manifest['mapping_sha256']),
        no_future_channel_access=True,no_truth_channel_access=True,no_household_or_trajectory_export=True,
        regimes=regimes,daily=daily,
        next_calendar_pairs=dict(pairs=planned,basis='already locked NEXT_PLAN_DRAFT.md; no result screening',source_sha256=sha(plan),
            future_power_values_not_inspected=True,future_completeness_not_yet_verified=True),
        original_source_price_contract_NOT_new_protocol=dict(budget_nok_per_24h=config['planning']['budget_nok'],
            repeat_validation_budget_nok=config['planning']['budget_nok'],
            scope='paper15 empirical full-day menu model; distinct from five-node synthetic budget1.6'),
        caveats=['Observed P/Q include declared 0.5% measurement noise, which can lift numerical rank.',
            'P is real control-household aggregate; Q is synthetic residential power factor, V/root from AC and meter noise.',
            'Previous cached DR fits include an action day and are not strict pre-action candidate banks.',
            'Strict past new candidate competition would use observed history only; no existing DR cache should supply it.',
            'Geometry diagnostics do not establish topology uniqueness, reliability, or price efficacy.'])
    destination=OUT/'history_inventory.json';destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:{'P_rank':v['P']['rank'],'PQ_rank':v['PQ']['rank'],'P_condition':v['P']['rms_scaled_condition'],
        'PQ_condition':v['PQ']['rms_scaled_condition'],'Q_innovation':v['Q_residual_after_linear_P_projection_relative_fro']} for key,v in regimes.items()},indent=2))
    return result


if __name__=='__main__':main()
