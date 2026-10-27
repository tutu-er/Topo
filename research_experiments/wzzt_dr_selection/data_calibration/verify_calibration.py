"""Independent aggregate re-fit and saved-realization metric checks."""
import os, pathlib, json, hashlib
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
DATA=pathlib.Path(os.environ.get('WZZT_DR_DATA_DIR',str(ROOT)))
cfg=json.loads((ROOT/'protocol.json').read_text())
d=pd.read_pickle(DATA/'raw'/'selected_hourly.pkl')
d=d[d.Participation_Phase=='Phase_2'].copy()
d['Date']=pd.to_datetime(d.Date);d['Hour']=d.Hour.astype(int)
d['group']=d.Group_Phase2.astype(str)
d['band']=np.where(d.Hour<=16,'before',np.where(d.Hour<=18,'event','after'))
daily=d.groupby(['group','Date','band'],observed=True).Demand_kWh.mean()
ev=pd.read_csv(DATA/'raw'/'experiment_days.csv');ev=ev[ev.Participation_Phase=='Phase_2']
ev['Date']=pd.to_datetime(ev.Date)
menus=ev.set_index(['Group','Date']).Price_signal
holidays=set(pd.to_datetime(cfg['eligibility']['holidays_excluded_from_baseline']))
end=pd.Timestamp(cfg['split']['training_end_exclusive'])
ratios=[]
for (g,day,band),y in daily.items():
    if day>=end:continue
    all_days=daily.loc[g].index.get_level_values(0).unique()
    prior=sorted(x for x in all_days if x<=day-pd.Timedelta(days=2) and x.dayofweek<5 and x not in holidays and (g,x) not in menus.index)[-10:]
    if len(prior)<5:continue
    b=sum(float(daily[(g,x,band)]) for x in prior)/len(prior)
    ratios.append({'group':g,'date':day,'band':band,'menu':menus.get((g,day),''),'ratio':float(y)/b})
r=pd.DataFrame(ratios)
normal=r[(r.menu=='')&(r.date.dt.dayofweek<5)&~r.date.isin(holidays)]
bias=normal.groupby(['group','band']).ratio.mean()
events=r[r.menu.isin(cfg['dataset']['menus'])].copy()
events['effect']=[x.ratio/bias[(x.group,x.band)] for x in events.itertuples()]
effect=events.groupby(['menu','band']).effect.mean()
reported_bias=pd.read_csv(ROOT/'outputs'/'normal_day_factors.csv').set_index(['group','band']).factor
reported_effect=pd.read_csv(ROOT/'outputs'/'mean_factors.csv').set_index(['menu','band']).factor
bias_error=float((bias-reported_bias).abs().max());effect_error=float((effect-reported_effect).abs().max())
assert bias_error<1e-12 and effect_error<1e-12
bank=pd.read_pickle(DATA/'raw'/'whole_day_evaluation_bank.pkl')
results=pd.read_csv(ROOT/'outputs'/'group_day_results.csv')
max_error=0.;checked=0
for key,x in bank.items():
    group,date=key.rsplit('_',1)
    # Rebuild actual matrix directly from source rows, independently of saved predictions.
    raw=d[(d.group==group)&(d.Date==pd.Timestamp(date))].sort_values(['ID','Hour']).Demand_kWh.to_numpy().reshape(len(x['ids']),24)
    assert np.array_equal(raw,x['actual_kw'])
    for name,field in [('price_free','price_free_kw'),('price_mean','price_mean_kw')]:
        row=results[(results.group==group)&(results.date==date)&(results.model==name)].iloc[0]
        e=x[field]-raw
        computed={'hourly_mae_kw':np.abs(e).mean(),'hourly_mse_kw2':np.square(e).mean(),
                  'daily_energy_mae_kwh':np.abs(e.sum(axis=1)).mean(),'daily_energy_bias_kwh':e.sum(axis=1).mean(),
                  'event_mae_kw':np.abs(e[:,16:18]).mean(),'event_bias_kw':e[:,16:18].mean()}
        for metric,value in computed.items():max_error=max(max_error,abs(value-row[metric]))
        checked+=1
assert checked==118 and max_error<1e-12
split=pd.read_csv(ROOT/'outputs'/'split_manifest.csv')
assert ((pd.to_datetime(split.date)-pd.to_datetime(split.baseline_latest)).dt.days>=2).all()
out={'status':'pass','independent_group_aggregate_bias_checks':len(bias),'independent_menu_factor_checks':len(effect),
     'raw_actual_trajectory_checks':len(bank),'prediction_metric_records_checked':checked,
     'max_bias_difference':bias_error,'max_menu_difference':effect_error,'max_metric_difference':max_error,
     'chronological_history_checks':len(split),'topology_trials':0,
     'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
(ROOT/'outputs'/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
