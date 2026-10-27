"""Fixed two-mean-model comparison on independently held-out observed iFlex menus.

No topology labels, simulated responses, monotonic price constraint or test tuning.
Raw arrays and household predictions remain local; only group aggregates are exported.
"""
from __future__ import annotations
import datetime, hashlib, json, pathlib, sys, os
import numpy as np
import pandas as pd

ROOT=pathlib.Path(__file__).resolve().parent
DATA=pathlib.Path(os.environ.get('WZZT_DR_DATA_DIR',str(ROOT)))
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
CFG=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':digest(pathlib.Path(__file__)),
          'selected_hourly_sha256':digest(DATA/'raw'/'selected_hourly.pkl'),
          'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,'status':'started'}
dump('calibration_run.json',manifest)
df=pd.read_pickle(DATA/'raw'/'selected_hourly.pkl')
df=df[df.Participation_Phase.eq('Phase_2')].copy()
df['Date']=pd.to_datetime(df.Date); df['Hour']=df.Hour.astype(int)
df['Group']=df.Group_Phase2.astype(str); df['Price_signal']=df.Price_signal.astype(str)
ev=pd.read_csv(DATA/'raw'/'experiment_days.csv',keep_default_na=False)
ev=ev[(ev.Participation_Phase=='Phase_2') & ev.Group.isin(CFG['dataset']['groups'])].copy()
ev['Date']=pd.to_datetime(ev.Date)
assert not ev.duplicated(['Group','Date']).any()
prices=pd.read_csv(DATA/'raw'/'price_signals.csv',keep_default_na=False)
lookup=ev.set_index(['Group','Date']).Price_signal
df['menu_index']=pd.MultiIndex.from_frame(df[['Group','Date']]).map(lookup).fillna('')
expected_price=pd.MultiIndex.from_frame(df[['Price_signal','Hour']]).map(prices.set_index(['Price_signal','Hour']).Experiment_price_NOK_kWh)
event_mask=df.menu_index.ne('')
audit={'rows':len(df),'households':int(df.ID.nunique()),'dates':int(df.Date.nunique()),
       'date_min':str(df.Date.min().date()),'date_max':str(df.Date.max().date()),
       'duplicate_household_from':int(df.duplicated(['ID','From']).sum()),
       'duplicate_household_date_hour':int(df.duplicated(['ID','Date','Hour']).sum()),
       'missing_demand':int(df.Demand_kWh.isna().sum()),'negative_demand':int((df.Demand_kWh<0).sum()),
       'hour_from_mismatch':int((df.Hour!=(df.From.dt.hour+1)).sum()),
       'date_from_mismatch':int((df.Date!=df.From.dt.normalize()).sum()),
       'event_menu_mismatch':int((df.Price_signal!=df.menu_index).sum()),
       'event_price_mismatch':int((np.abs(df.loc[event_mask,'Experiment_price_NOK_kWh']-expected_price[event_mask])>1e-12).sum()),
       'time_axis':'Original Date/Hour retained. pyreadr decodes From at UTC default; local timezone not independently established; no clock relabeling.',
       'demand_quantiles_kw':{str(q):float(v) for q,v in df.Demand_kWh.quantile([0,.01,.5,.9,.99,1]).items()}}
sizes=df.groupby(['ID','Date'],observed=True).size()
audit['household_days']=len(sizes);audit['incomplete_household_days']=int((sizes!=24).sum())
audit['zero_energy_household_days']=int(df.groupby(['ID','Date'],observed=True).Demand_kWh.sum().eq(0).sum())
for k in ['duplicate_household_from','duplicate_household_date_hour','missing_demand','negative_demand','hour_from_mismatch','date_from_mismatch','event_menu_mismatch','event_price_mismatch','incomplete_household_days']:
    assert audit[k]==0, (k,audit[k])
bands={'before':np.arange(16),'event':np.arange(16,18),'after':np.arange(18,24)}
train_end=pd.Timestamp(CFG['split']['training_end_exclusive'])
holidays=pd.to_datetime(CFG['eligibility']['holidays_excluded_from_baseline'])
groups={}; rec=[]; excluded=[]
for group,sub in df.groupby('Group',sort=True):
    ids=sorted(sub.ID.unique()); dates=pd.DatetimeIndex(sorted(sub.Date.unique()))
    sub=sub.sort_values(['Date','ID','Hour'])
    assert len(sub)==len(dates)*len(ids)*24
    y=sub.Demand_kWh.to_numpy().reshape(len(dates),len(ids),24)
    menus=np.array([lookup.get((group,d),'') for d in dates],dtype=object)
    eligible=(menus=='') & (dates.dayofweek<5) & ~dates.isin(holidays)
    base=np.full_like(y,np.nan); latest={}
    for t,day in enumerate(dates):
        prior=np.flatnonzero(eligible & (dates<=day-pd.Timedelta(days=2)))[-10:]
        if len(prior)<5:
            if menus[t] in CFG['dataset']['menus']: excluded.append({'group':group,'date':str(day.date()),'menu':menus[t],'reason':'fewer_than_5_eligible_prior_days','prior_days':len(prior)})
            continue
        base[t]=y[prior].mean(axis=0); latest[t]=dates[prior[-1]]
        assert latest[t]<=day-pd.Timedelta(days=2)
        for band,h in bands.items():
            yy=float(y[t][:,h].mean()); bb=float(base[t][:,h].mean())
            assert bb>0
            rec.append({'group':group,'date':day,'menu':menus[t],'band':band,'actual':yy,'history':bb,'ratio':yy/bb,'households':len(ids),'baseline_latest':latest[t]})
    groups[group]={'dates':dates,'ids':ids,'menus':menus,'y':y,'base':base,'eligible':eligible,'latest':latest}
records=pd.DataFrame(rec)
# Calibrate group-specific ordinary-day forecast bias, then pooled menu multipliers.
normal=records[(records.date<train_end)&records.menu.eq('')&records.date.dt.dayofweek.lt(5)&~records.date.isin(holidays)]
bias=normal.groupby(['group','band']).ratio.mean()
records['baseline']=records.history*pd.MultiIndex.from_frame(records[['group','band']]).map(bias).to_numpy()
records['menu_ratio']=records.actual/records.baseline
training=records[(records.date<train_end)&records.menu.isin(CFG['dataset']['menus'])]
factors=training.groupby(['menu','band']).menu_ratio.mean()
assert len(factors)==12
coefs=[]
for (menu,band),v in factors.items():
    a=training[(training.menu==menu)&(training.band==band)]
    # Leave-one-training-date-out sensitivity. Refit the normal bias when deleting that date.
    values=[]
    for d in sorted(training.date.unique()):
        nb=normal[normal.date!=d].groupby(['group','band']).ratio.mean()
        q=a[a.date!=d]
        if len(q):
            b=q.history*pd.MultiIndex.from_frame(q[['group','band']]).map(nb).to_numpy()
            values.append(float((q.actual/b).mean()))
    coefs.append({'menu':menu,'band':band,'factor':float(v),'relative_change_percent':float((v-1)*100),
                  'training_group_days':len(a),'training_dates':int(a.date.nunique()),'leave_one_date_out_min':min(values),'leave_one_date_out_max':max(values)})
pd.DataFrame(coefs).to_csv(OUT/'mean_factors.csv',index=False)
bias.rename('factor').reset_index().to_csv(OUT/'normal_day_factors.csv',index=False)
evalrows=[]; residual_rows=[]; local_bank={}; counts=[]
for group,g in groups.items():
    pred0=g['base'].copy();pred1=g['base'].copy()
    for band,h in bands.items():
        pred0[:,:,h]*=bias[(group,band)]
        pred1[:,:,h]*=bias[(group,band)]
    for t,(day,menu) in enumerate(zip(g['dates'],g['menus'])):
        if menu not in CFG['dataset']['menus'] or not np.isfinite(g['base'][t]).all(): continue
        if day<train_end: split='train'
        elif pd.Timestamp(CFG['split']['primary_test_start'])<=day<=pd.Timestamp(CFG['split']['primary_test_end_inclusive']): split='primary_test'
        elif pd.Timestamp(CFG['split']['late_test_start'])<=day<=pd.Timestamp(CFG['split']['late_test_end_inclusive']): split='late_test'
        else: split='unused'
        for band,h in bands.items(): pred1[t][:,h]*=factors[(menu,band)]
        assert np.isfinite(pred1[t]).all() and pred1[t].min()>=0
        counts.append({'group':group,'date':str(day.date()),'menu':menu,'split':split,'households':len(g['ids']), 'baseline_latest':str(g['latest'][t].date())})
        for model,pred in [('price_free',pred0[t]),('price_mean',pred1[t])]:
            e=pred-g['y'][t]
            energy=e.sum(axis=1)
            row={'group':group,'date':str(day.date()),'menu':menu,'split':split,'model':model,'households':len(g['ids']),
                 'hourly_mae_kw':float(np.abs(e).mean()),'hourly_mse_kw2':float((e*e).mean()),
                 'group_mean_hourly_mae_kw_per_home':float(np.abs(e.mean(axis=0)).mean()),
                 'daily_energy_mae_kwh':float(np.abs(energy).mean()),'daily_energy_bias_kwh':float(energy.mean())}
            for band,h in bands.items():
                row[band+'_mae_kw']=float(np.abs(e[:,h]).mean());row[band+'_bias_kw']=float(e[:,h].mean())
                row[band+'_group_mean_mae_kw_per_home']=float(np.abs(e[:,h].mean(axis=0)).mean())
            evalrows.append(row)
        # Preserve synchronized whole-day arrays without mean-centering actual test responses.
        local_bank[group+'_'+str(day.date())]={'ids':g['ids'],'actual_kw':g['y'][t],'price_free_kw':pred0[t],'price_mean_kw':pred1[t],'split':split,'menu':menu}
        e=g['y'][t]-pred1[t]
        lag=float(np.corrcoef(e[:,:-1].ravel(),e[:,1:].ravel())[0,1])
        corr=np.corrcoef(e)
        off=corr[np.triu_indices(len(e),1)];off=off[np.isfinite(off)]
        residual_rows.append({'group':group,'date':str(day.date()),'split':split,'menu':menu,'mean_kw':float(e.mean()),'sd_kw':float(e.std()),'lag1_raw_residual_correlation':lag,'mean_pairwise_within_day_household_correlation':float(off.mean()),'actual_peak_kw':float(g['y'][t].max()),'actual_mean_daily_kwh':float(g['y'][t].sum(axis=1).mean())})
pd.to_pickle(local_bank,DATA/'raw'/'whole_day_evaluation_bank.pkl')
results=pd.DataFrame(evalrows); results.to_csv(OUT/'group_day_results.csv',index=False)
pd.DataFrame(counts).to_csv(OUT/'split_manifest.csv',index=False)
pd.DataFrame(residual_rows).to_csv(OUT/'whole_day_residual_diagnostics.csv',index=False)
metrics=[c for c in results if c.endswith(('_kw','_kw2','_home','_kwh'))]
summary=[]
for (split,model),s in results.groupby(['split','model']):
    row={'split':split,'model':model,'group_days':len(s),'dates':int(s.date.nunique()),'household_days':int(s.households.sum()),'household_hours':int(s.households.sum()*24)}
    row.update({k:float(s[k].mean()) for k in metrics})
    row['household_weighted_hourly_mae_kw']=float(np.average(s.hourly_mae_kw,weights=s.households))
    summary.append(row)
pd.DataFrame(summary).to_csv(OUT/'split_metrics.csv',index=False)
results.groupby(['split','menu','model'])[metrics].mean().reset_index().to_csv(OUT/'menu_metrics.csv',index=False)
cross=ev.assign(split=np.where(ev.Date<train_end,'train',np.where(ev.Date.between(CFG['split']['primary_test_start'],CFG['split']['primary_test_end_inclusive']),'primary_test',np.where(ev.Date.between(CFG['split']['late_test_start'],CFG['split']['late_test_end_inclusive']),'late_test','unused'))))
audit.update({'raw_event_group_days':len(ev),'raw_event_dates':int(ev.Date.nunique()),'excluded_events':excluded,
              'split_menu_group_days':cross.groupby(['split','Price_signal']).size().reset_index(name='group_days').to_dict('records'),
              'validated_forecast_group_days':len(counts),
              'baseline_lag_days_min':min((pd.Timestamp(x['date'])-pd.Timestamp(x['baseline_latest'])).days for x in counts),
              'future_temperature_or_current_day_controls_used':False,
              'calendar_used':'weekday and declared holidays only',
              'local_whole_day_bank_sha256':digest(DATA/'raw'/'whole_day_evaluation_bank.pkl')})
dump('data_quality.json',audit)
manifest.update({'status':'complete','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned_models':2,'completed_models':2,'topology_runs':0,'failed_models':0,'timed_out_models':0,'simulated_response_runs':0})
dump('calibration_run.json',manifest)
print(json.dumps({'audit':audit,'metrics':summary,'coefficients':coefs},indent=2,ensure_ascii=False),flush=True)
