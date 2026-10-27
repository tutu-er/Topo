import pathlib, sys, json, hashlib, platform, os
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
DATA=pathlib.Path(os.environ.get('WZZT_DR_DATA_DIR',str(ROOT)))
sys.path.insert(0,str(DATA/'deps'))
import pyreadr
data=pyreadr.read_r(str(DATA/'raw'/'iFlex dataset.RData'),use_objects=['data_hourly','participants','price_signals','experiment_days'])
audit={'python':sys.version,'pandas':pd.__version__,'pyreadr':pyreadr.__version__,'platform':platform.platform(),'objects':{}}
for key,df in data.items():
    audit['objects'][key]={'rows':len(df),'columns':list(df.columns),'dtypes':{c:str(t) for c,t in df.dtypes.items()}}
    if key!='data_hourly':
        csv=pd.read_csv(DATA/'raw'/(key+'.csv'))
        a=df.copy().reset_index(drop=True); b=csv.copy().reset_index(drop=True)
        equal={}
        for c in a:
            if pd.api.types.is_numeric_dtype(a[c]):
                import numpy as np
                equal[c]=bool(np.allclose(a[c].astype(float),pd.to_numeric(b[c],errors='coerce').astype(float),equal_nan=True))
            else:
                equal[c]=bool(a[c].astype('string').fillna('<NA>').equals(b[c].astype('string').fillna('<NA>')))
        audit['objects'][key]['csv_cell_equivalence']=equal
        norm_a=a.astype('string').replace({'NA':pd.NA,'Ok':'OK'}).fillna('<NA>')
        norm_b=b.astype('string').replace({'NA':pd.NA,'Ok':'OK'}).fillna('<NA>')
        audit['objects'][key]['csv_equivalence_after_explicit_NA_and_Ok_normalization']={c:bool(norm_a[c].equals(norm_b[c])) for c in a}
parts=data['participants']
sel=parts.loc[parts.Group_Phase2.isin(['H2','Os_2','Trom_2']) & parts.Participation_status_Phase2.astype(str).str.upper().eq('OK')]
df=data['data_hourly']
audit['hourly_all']={'ids':int(df.ID.nunique()),'date_min':str(df.Date.min()),'date_max':str(df.Date.max()),'phase_counts':{str(k):int(v) for k,v in df.Participation_Phase.value_counts(dropna=False).items()},'demand_missing':int(df.Demand_kWh.isna().sum()),'demand_negative':int((df.Demand_kWh<0).sum())}
sub=df.loc[df.ID.isin(sel.ID)].copy()
sub=sub.merge(sel[['ID','Group_Phase2','Region']],on='ID',validate='many_to_one')
sub.to_pickle(DATA/'raw'/'selected_hourly.pkl')
audit['selected']={'index_households':len(sel),'rows':len(sub),'ids':int(sub.ID.nunique()),'phase_counts':{str(k):int(v) for k,v in sub.Participation_Phase.value_counts(dropna=False).items()},'date_min':str(sub.Date.min()),'date_max':str(sub.Date.max()),'price_counts':{str(k):int(v) for k,v in sub.Price_signal.value_counts(dropna=False).items()},'time_examples':sub[['From','Date','Hour']].head(26).astype(str).to_dict('records')}
(ROOT/'outputs'/'initial_data_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(audit,indent=2,ensure_ascii=False),flush=True)
