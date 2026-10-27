"""Prepare empirical planning history and separate evaluator-only DR trajectories.

Published household demand is used at its original hourly scale. Counterfactual
DR is an explicitly semi-synthetic multiplication by held-out *aggregate*
treatment/control trajectories; it is not an observed individual price effect.
"""
from __future__ import annotations
import hashlib, json, pathlib, sys
import numpy as np
import pandas as pd

MENUS=('P_2','P_5','P_10','P_15')
GROUP_CONTROL={'H2':'Control','Os_2':'Os_Control','Trom_2':'Trom_Control'}
HOLIDAYS=pd.to_datetime(['2020-12-24','2020-12-25','2020-12-26','2020-12-31','2021-01-01'])
BANDS={'before':np.arange(16),'event':np.arange(16,18),'after':np.arange(18,24)}

def sha256(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def _phase2_controls(data_root, ids):
    """Only this preparation stage needs pyreadr; subsequent runs use the cache."""
    cache=data_root/'raw'/'controls_phase2_for_flow.pkl'
    receipt=cache.with_suffix('.json')
    source=data_root/'raw'/'iFlex dataset.RData'
    fingerprint={'rdata_sha256':sha256(source),'ids_sha256':hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()}
    if cache.exists() and receipt.exists():
        saved=json.loads(receipt.read_text())
        if all(saved.get(k)==v for k,v in fingerprint.items()) and saved.get('cache_sha256')==sha256(cache):
            return pd.read_pickle(cache)
    local_deps=data_root/f'deps_py{sys.version_info.major}{sys.version_info.minor}'
    if local_deps.exists():sys.path.insert(0,str(local_deps))
    elif sys.version_info[:2]==(3,13):sys.path.insert(0,str(data_root/'deps'))
    try: import pyreadr
    except ImportError as e: raise RuntimeError('Install pyreadr in this Python environment or its isolated data-cache deps directory before prepare') from e
    full=pyreadr.read_r(str(source),use_objects=['data_hourly'])['data_hourly']
    selected=full.loc[full.ID.isin(ids)&full.Participation_Phase.eq('Phase_2')].copy()
    selected.to_pickle(cache)
    receipt.write_text(json.dumps({**fingerprint,'cache_sha256':sha256(cache),'rows':len(selected)},indent=2),encoding='utf-8')
    return selected

def _daily_matrices(frame, ids):
    selected=frame[frame.ID.isin(ids)].copy()
    selected['Date']=pd.to_datetime(selected.Date);selected['Hour']=selected.Hour.astype(int)
    if selected.duplicated(['Date','ID','Hour']).any():raise ValueError('duplicate household/date/hour')
    dates=pd.DatetimeIndex(sorted(selected.Date.unique()))
    ix=pd.MultiIndex.from_product([dates,sorted(ids),range(1,25)],names=['Date','ID','Hour'])
    values=selected.set_index(['Date','ID','Hour']).Demand_kWh.reindex(ix).to_numpy().reshape(len(dates),len(ids),24)
    # Missing future values fail preparation, never cause a different mapping.
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('incomplete or invalid household trajectories')
    return dates,values

def _ratio_trajectory(t_actual,t_history,c_actual,c_history,band_norms):
    ta=t_actual.mean(axis=0);tp=t_history.mean(axis=(0,1))
    ca=c_actual.mean(axis=0);cp=c_history.mean(axis=(0,1))
    if min(tp.min(),ca.min(),cp.min())<=0:raise ValueError('undefined response normalization')
    out=(ta/tp)/(ca/cp)
    for band,h in BANDS.items():out[h]/=float(band_norms[band])
    if not np.isfinite(out).all() or (out<0).any():raise ValueError('invalid full-day response factor')
    return out

def split_response_banks(records):
    """Split complete source dates per menu; never split groups from the same date."""
    banks={'action':{},'validation':{}};metadata={'action':{},'validation':{}}
    for menu in MENUS:
        rows=[r for r in records if r['menu']==menu]
        dates=sorted({r['date'] for r in rows})
        if len(dates)<2:raise ValueError(f'{menu} needs >=2 held-out dates for separate action/validation banks')
        pivot=max(1,len(dates)//2)
        sets={'action':set(dates[:pivot]),'validation':set(dates[pivot:])}
        assert sets['action'].isdisjoint(sets['validation'])
        for role,ds in sets.items():
            chosen=[r for r in rows if r['date'] in ds]
            banks[role][menu]=np.stack([r['factors'] for r in chosen])[:,:,None]
            metadata[role][menu]=[{'group':r['group'],'date':r['date'],'menu':menu} for r in chosen]
    return banks,metadata

def prepare_iflex_bundle(data_root,calibration_dir,terminals,config,output_dir):
    data_root=pathlib.Path(data_root);calibration_dir=pathlib.Path(calibration_dir);output_dir=pathlib.Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    raw=data_root/'raw'; cfg=config['source']
    manifest=json.loads((calibration_dir/'outputs'/'raw_manifest.json').read_text())
    for f in manifest['files']:
        if sha256(raw/f['file'])!=f['sha256']:raise ValueError('raw file hash mismatch: '+f['file'])
    parts=pd.read_csv(raw/'participants.csv',keep_default_na=False)
    ok=parts.Participation_status_Phase2.astype(str).str.upper().eq('OK')
    treatments=parts[ok&parts.Group_Phase2.isin(GROUP_CONTROL)].copy()
    controls=parts[ok&parts.Group_Phase2.isin(GROUP_CONTROL.values())].copy()
    assert controls.Control_Price_Phase2.eq('Control group').all()
    cd=_phase2_controls(data_root,set(controls.ID))
    td=pd.read_pickle(raw/'selected_hourly.pkl');td=td[td.Participation_Phase.eq('Phase_2')].copy()
    ev=pd.read_csv(raw/'experiment_days.csv');ev['Date']=pd.to_datetime(ev.Date)
    ev=ev[(ev.Participation_Phase=='Phase_2')&ev.Group.isin(GROUP_CONTROL)]
    if ev.duplicated(['Group','Date']).any():raise ValueError('duplicate event assignment')
    event_lookup=ev.set_index(['Group','Date']).Price_signal
    normalization=pd.read_csv(calibration_dir/'outputs'/'control_training_normal_factors.csv').set_index(['stratum','band']).normal_factor
    calibration_receipt=json.loads((calibration_dir/'outputs'/'control_training_summary.json').read_text(encoding='utf-8'))
    cutoff=pd.Timestamp(cfg['calibration_end_exclusive'])
    if pd.Timestamp(calibration_receipt['training_end_exclusive'])>cutoff:raise ValueError('calibration after decision cutoff')
    action=pd.Timestamp(config['action_date']);validation=pd.Timestamp(config['validation_date'])
    if cutoff>action-pd.Timedelta(days=1) or validation<=action:raise ValueError('invalid chronological split')

    # Choose IDs solely from eligibility labels and a frozen RNG, never by future outcomes.
    eligible=sorted(controls.loc[(controls.Group_Phase2==cfg['group'])&(controls.Region==cfg['region']),'ID'])
    needed=len(terminals)*int(cfg['households_per_terminal'])
    if len(eligible)<needed:raise ValueError('not enough fixed eligible households')
    chosen=np.random.default_rng(cfg['mapping_seed']).permutation(eligible)[:needed].tolist()
    mapping={int(bus):chosen[j*cfg['households_per_terminal']:(j+1)*cfg['households_per_terminal']] for j,bus in enumerate(terminals)}
    aggregated={};dates=None
    for bus,ids in mapping.items():
        ds,arr=_daily_matrices(cd,ids)
        if dates is None:dates=ds
        if not ds.equals(dates):raise ValueError('household date mismatch')
        aggregated[bus]=arr.sum(axis=1)
    daily={str(day.date()):pd.DataFrame({bus:aggregated[bus][i] for bus in terminals},index=pd.date_range(day,periods=24,freq='h')) for i,day in enumerate(dates)}
    history_dates=[d for d in dates if d<=action-pd.Timedelta(days=2) and d.dayofweek<5 and d not in HOLIDAYS][-cfg['forecast_days']:]
    if len(history_dates)!=cfg['forecast_days']:raise ValueError('insufficient past forecast days')
    history_days={str(d.date()):daily[str(d.date())] for d in history_dates}
    actual_days={'action':daily[str(action.date())],'validation':daily[str(validation.date())]}

    # Evaluator-only bank. Neither these factors nor future loads are forecast inputs.
    event_records={}; stratum_counts=[]
    first=pd.Timestamp(cfg['response_bank_start']);last=pd.Timestamp(cfg['response_bank_end'])
    if first<cutoff:raise ValueError('response bank overlaps calibration period')
    for (group,region),membership in treatments.groupby(['Group_Phase2','Region']):
        ids=sorted(membership.ID);cids=sorted(controls.loc[(controls.Group_Phase2==GROUP_CONTROL[group])&(controls.Region==region),'ID'])
        ds,tvals=_daily_matrices(td,ids);cs,cvals=_daily_matrices(cd,cids)
        if not ds.equals(cs):raise ValueError('treatment/control dates differ')
        assigned=np.array([event_lookup.get((group,d),'') for d in ds],dtype=object)
        eligible_days=(assigned=='')&(ds.dayofweek<5)&~ds.isin(HOLIDAYS)
        norms={band:normalization[(group+'|'+region,band)] for band in BANDS}
        valid=0
        for j,d in enumerate(ds):
            menu=assigned[j]
            if not(first<=d<=last) or menu not in MENUS:continue
            prior=np.flatnonzero(eligible_days&(ds<=d-pd.Timedelta(days=2)))[-10:]
            if len(prior)<5:raise ValueError('held-out event lacks declared history')
            factor=_ratio_trajectory(tvals[j],tvals[prior],cvals[j],cvals[prior],norms)
            key=(group,str(d.date()),menu)
            event_records.setdefault(key,[]).append((len(ids),factor));valid+=1
        stratum_counts.append({'group':group,'region':region,'households':len(ids),'controls':len(cids),'heldout_events':valid})
    records=[{'group':g,'date':d,'menu':m,'factors':np.average(np.stack([x[1] for x in values]),axis=0,weights=[x[0] for x in values])} for (g,d,m),values in sorted(event_records.items())]
    banks,bank_metadata=split_response_banks(records)
    inputs={
        str(raw/f['file']):f['sha256'] for f in manifest['files']
    }
    for name in ['control_training_normal_factors.csv','control_training_menu_factors.csv','control_training_summary.json']:
        path=calibration_dir/'outputs'/name;inputs[str(path)]=sha256(path)
    data_manifest={'source_record':8248802,'source_hashes':inputs,'households_per_terminal':cfg['households_per_terminal'],
        'network_households':needed,'source_eligible_households':len(eligible),'mapping_seed':cfg['mapping_seed'],
        'mapping_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest(),
        'forecast_history_dates':[str(d.date()) for d in history_dates],
        'forecast_latest_date':str(history_dates[-1].date()),'action_date':str(action.date()),'validation_date':str(validation.date()),
        'response_bank_group_days':len(records),'response_bank_strata':stratum_counts,
        'response_bank_counts':{role:{m:len(a) for m,a in bank.items()} for role,bank in banks.items()},
        'response_mode':'heldout hourly aggregate treatment/control ratios, fixed training band normalization, no recentering',
        'response_spatial_dependence':'common full-day factor for terminals receiving same menu',
        'counterfactual_status':'semi-synthetic; not observed household-specific counterfactual power',
        'power_units':'kW from original 1h kWh, summed3households; no scale factor',
        'time_axis':'original dataset Date/Hour preserved, no unsupported local timezone translation',
        'simulation_Q':'separate synthetic power-factor model',
        'no_reward_source':'observed Control-group household consumption'}
    bundle={'history_days':history_days,'actual_days':actual_days,'response_banks':banks,'bank_metadata':bank_metadata,'data_manifest':data_manifest}
    pd.to_pickle(bundle,output_dir/'privatebundle.pkl')
    (output_dir/'private_household_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    (output_dir/'data_manifest.json').write_text(json.dumps(data_manifest,indent=2,ensure_ascii=True),encoding='utf-8')
    return bundle
