"""Versioned, cohort-aware records for the completed WZZT experiment (no solves)."""
from pathlib import Path
import hashlib
import json
import sqlite3
import zipfile
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import sha,read,write,matrices
from research_experiments.wzzt_fine.evaluate import signature

COHORTS=('original','free_domain','joint_extension','lp60_retry','postfilter_guard')
DIMENSIONS=('panel','case','condition','bundle','N','voltage_sigma','pq_sigma')


class Sources:
    def __init__(self,source,output):
        self.source=Path(source).resolve();self.output=Path(output).resolve();self.used={}
        self.delivery=read(self.source/'delivery_manifest.json');self.raw=read(self.source/'result_manifest.json');self.frozen=read(self.source/'manifest.json')
        self.consume(self.source/'delivery_manifest.json');self.consume(self.source/'result_manifest.json');self.consume(self.source/'manifest.json')

    def consume(self,path):
        path=Path(path).resolve();key=str(path)
        if key not in self.used:
            actual=sha(path);expected=None
            if path.parent==self.source:expected=self.delivery.get(path.name)
            elif path.is_relative_to(self.output):
                relative=path.relative_to(self.output).as_posix();expected=self.raw.get(relative,self.frozen.get(relative))
            if expected is not None and actual!=expected:raise ValueError('source bytes changed: '+key)
            self.used[key]=dict(sha256=actual,bound_to_original_manifest=expected is not None)
        return path

    def file(self,name):
        path=self.source/name
        if not path.exists():
            from .records import restore
            restore(self.source,names=[name])
        return self.consume(path)

    def frame(self,name):return pd.read_csv(self.file(name))
    def object(self,name):return read(self.file(name))


def _merge_dimensions(frame,inputs):
    metadata=inputs[['job_id',*DIMENSIONS]].set_index('job_id')
    frame=frame.copy()
    for column in DIMENSIONS:
        expected=frame.job_id.map(metadata[column])
        if column in frame:
            mask=frame[column].notna()
            if not (frame.loc[mask,column].to_numpy()==expected[mask].to_numpy()).all():
                raise ValueError('supplement dimensions disagree: '+column)
        frame[column]=expected
    return frame


def create_snapshot(sources,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    original=sources.frame('per_job.csv');inputs=pd.DataFrame(sources.object('datasets.json'))
    holdout=set(sources.frame('postfilter_guard_holdout.csv').job_id)
    frames=[]
    spec=[('original','per_job.csv'),('free_domain','free_domain_models.csv'),
          ('joint_extension','joint_extension_models.csv'),('lp60_retry','lp60_retry.csv'),
          ('postfilter_guard','postfilter_guard.csv')]
    for cohort,file in spec:
        d=original.copy() if cohort=='original' else sources.frame(file)
        if 'model_available' not in d:d['model_available']=d['returned']
        if 'variant' not in d:d['variant']='ridge_postfilter_guard'
        if 'linear_test_mae' in d:d=d.rename(columns={'linear_test_mae':'test_mae'})
        d=_merge_dimensions(d,inputs);d['cohort']=cohort
        d['record_id']=cohort+'::'+d.job_id+'::'+d.variant
        d['score_policy']=np.where(d.model_available,'available_model','unavailable_retained_as_zero')
        d['exact_clades']=d.model_available & d.clade_f1.ge(1-1e-12)
        d['fixed_LP_seconds']=60 if cohort=='lp60_retry' else 15
        d['evidence_role']='predeclared' if cohort=='original' else 'secondary' if cohort in ('free_domain','lp60_retry') else 'secondary_after_main' if cohort=='joint_extension' else 'exploratory'
        d['prospective_filter_holdout']=(cohort=='postfilter_guard') & d.job_id.isin(holdout)
        d['source_table']=file;d['source_table_sha256']=sources.used[str(sources.source/file)]['sha256']
        frames.append(d)
    results=pd.concat(frames,ignore_index=True,sort=False)
    results=results.drop(columns=[c for c in ('supports_json','r_json','x_json') if c in results])
    results=results.sort_values(['cohort','job_id','variant']).reset_index(drop=True)
    parameters=[];index=results.set_index('record_id');meta=inputs.set_index('job_id')

    def add(cohort,row):
        method=row.get('variant','ridge_postfilter_guard');key=cohort+'::'+row['job_id']+'::'+method
        labels=row.get('labels',json.loads(meta.loc[row['job_id'],'labels_json']));point=row['point']
        parameters.append(dict(record_id=key,labels_json=json.dumps(labels),point_json=json.dumps(point),
            point_signature=signature(point),source_result=row.get('source_result',meta.loc[row['job_id'],'result_path']),
            source_result_sha256=row.get('source_result_sha256',meta.loc[row['job_id'],'result_sha256'])))

    for cohort,name in [('original','selected_models.jsonl'),('joint_extension','joint_extension_models.jsonl'),
                        ('lp60_retry','lp60_models.jsonl'),('postfilter_guard','postfilter_guard_models.jsonl')]:
        with sources.file(name).open(encoding='utf-8') as stream:
            for line in stream:add(cohort,json.loads(line))
    for job in sorted(set(results[results.cohort=='free_domain'].job_id)):
        path=sources.consume(sources.output/'free_domain'/job/'result.json');r=read(path)
        for method,point in r['models'].items():
            add('free_domain',dict(job_id=job,variant=method,labels=r['labels'],point=point,source_result=str(path),source_result_sha256=sha(path)))
    params=pd.DataFrame(parameters);pindex=params.set_index('record_id')
    results['model_signature']=results.record_id.map(pindex.point_signature)
    results['ac_key']=None
    fits=[]
    for cohort,folder in [('original',sources.output/'ac'),('lp60_retry',sources.output/'lp60_retry/ac')]:
        for file in sorted(folder.glob('*.json')):
            a=read(sources.consume(file));fit=a.get('fitting',{});test=a.get('testing',{});physical=a.get('physical',{});info=fit.get('refit',{})
            job=a['key'].rsplit('__',1)[0];key=cohort+'::'+a['key']
            fits.append(dict(ac_key=key,cohort=cohort,job_id=job,model_signature=a.get('model_signature',a['key'].rsplit('__',1)[1]),
                fitting_valid=fit.get('valid',False),testing_valid=test.get('valid',False),physical_valid=physical.get('valid',False),
                locally_converged=info.get('success'),nfev=info.get('nfev'),fit_status=info.get('status'),
                train_rows=fit.get('ac_converged_points'),test_rows=test.get('ac_converged_points'),
                train_initial_rmse=fit.get('initial_relative_voltage_rmse'),train_rmse=fit.get('relative_voltage_rmse'),
                test_rmse=test.get('relative_voltage_rmse'),physical_test_rmse=physical.get('relative_voltage_rmse'),
                fitting_seconds=fit.get('seconds'),testing_seconds=test.get('seconds'),physical_seconds=physical.get('seconds'),
                r_refit_json=json.dumps(fit.get('r_refit')),x_refit_json=json.dumps(fit.get('x_refit')),
                source_path=str(file),source_sha256=sha(file),infrastructure_error=a.get('error')))
    ac=pd.DataFrame(fits);validkeys=set(ac.ac_key)
    for i,row in results[results.cohort.isin(['original','lp60_retry']) & results.model_available].iterrows():
        key=row.cohort+'::'+row.job_id+'__'+row.model_signature
        if row.get('ac_evaluated') is True or row.get('ac_evaluated')==1:
            if key not in validkeys:raise ValueError('AC link missing: '+key)
            results.at[i,'ac_key']=key
    # A single portable index points back to the existing observation/truth paths.
    config=[]
    for job in inputs.job_id:
        p=read(sources.consume(sources.output/'jobs'/job/'payload.json'))
        config.append(dict(job_id=job,configuration_json=json.dumps(p['config']),planned_methods_json=json.dumps(p['methods'])))
    inputs=inputs.merge(pd.DataFrame(config),on='job_id',validate='one_to_one')
    tables={'model_results':results,'parameters':params,'inputs':inputs,'ac_fits':ac,
        'daily_diagnostics':sources.frame('data_diagnostics.csv'),'physical_regimes':sources.frame('physical_regimes.csv'),
        'greedy_events':sources.frame('greedy_events.csv'),'mip_events':sources.frame('free_domain_MIP_events.csv'),
        'timing':sources.frame('standalone_timing.csv'),'joint_diagnostics':sources.frame('joint_selected_diagnostics.csv'),
        'filter_sensitivity':sources.frame('postfilter_sensitivity.csv')}
    db=destination/'records.sqlite'
    with sqlite3.connect(db) as conn:
        for name,frame in tables.items():frame.to_sql(name,conn,index=False,if_exists='replace')
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS record_key ON model_results(record_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS condition_key ON model_results(cohort,panel,"case",condition,N,voltage_sigma,bundle,variant)')
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS parameter_key ON parameters(record_id)')
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS input_key ON inputs(job_id)')
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS ac_key ON ac_fits(ac_key)')
        conn.execute('DROP VIEW IF EXISTS case33_recovered')
        conn.execute("""CREATE VIEW case33_recovered AS SELECT * FROM model_results r WHERE panel='case33' AND
            (cohort='lp60_retry' OR (cohort IN ('original','postfilter_guard') AND NOT EXISTS
             (SELECT 1 FROM model_results s WHERE s.job_id=r.job_id AND s.cohort='lp60_retry')))""")
    results.to_csv(destination/'model_records.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write(destination/'snapshot_summary.json',dict(model_records=len(results),available_models=int(results.model_available.sum()),
        unique_input_conditions=inputs.job_id.nunique(),cohorts=results.groupby('cohort').agg(records=('record_id','size'),available=('model_available','sum')).reset_index().to_dict('records'),
        parameters=len(params),AC_unique=len(ac),AC_links=int(results.ac_key.notna().sum()),daily_diagnostics=len(tables['daily_diagnostics'])))
    return tables


def restore_database(directory):
    directory=Path(directory);manifest=read(directory/'manifest.json');db=directory/'records.sqlite'
    if db.exists():
        if sha(db)!=manifest['database_sha256']:raise ValueError('local database bytes changed')
        return db
    archive=directory/'records.sqlite.zip'
    if sha(archive)!=manifest['files'][archive.name]:raise ValueError('database archive changed')
    with zipfile.ZipFile(archive) as z:
        if z.namelist()!=['records.sqlite']:raise ValueError('unexpected archive paths')
        data=z.read('records.sqlite')
    if hashlib.sha256(data).hexdigest()!=manifest['database_sha256']:raise ValueError('database member changed')
    db.write_bytes(data);return db


def query(directory,*,cohort='original',panel=None,case=None,condition=None,N=None,voltage_sigma=None,bundle=None,variant=None):
    """One row per cohort/input/method. Never pool cohorts by default."""
    if cohort is not None and cohort not in COHORTS:raise ValueError('unknown cohort')
    fields=dict(cohort=cohort,panel=panel,case=case,condition=condition,N=N,voltage_sigma=voltage_sigma,bundle=bundle,variant=variant)
    chosen={k:v for k,v in fields.items() if v is not None}
    sql='SELECT * FROM model_results'+(' WHERE '+' AND '.join('"'+k+'"=?' for k in chosen) if chosen else '')+' ORDER BY record_id'
    with sqlite3.connect(restore_database(directory)) as conn:return pd.read_sql_query(sql,conn,params=list(chosen.values()))


def load_record(directory,record_id,*,with_observations=False):
    with sqlite3.connect(restore_database(directory)) as conn:
        conn.row_factory=sqlite3.Row
        r=conn.execute('SELECT * FROM model_results WHERE record_id=?',(record_id,)).fetchone()
        if r is None:raise KeyError(record_id)
        if not r['model_available']:raise ValueError('model unavailable; inspect retained failure fields')
        p=conn.execute('SELECT * FROM parameters WHERE record_id=?',(record_id,)).fetchone()
        inp=dict(conn.execute('SELECT * FROM inputs WHERE job_id=?',(r['job_id'],)).fetchone())
        a=conn.execute('SELECT * FROM ac_fits WHERE ac_key=?',(r['ac_key'],)).fetchone() if r['ac_key'] else None
        has_settings=conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='method_settings'").fetchone()
        setting=conn.execute('SELECT * FROM method_settings WHERE cohort=? AND variant=?',(r['cohort'],r['variant'])).fetchone() if has_settings else None
    point=json.loads(p['point_json']);labels=json.loads(p['labels_json'])
    if signature(point)!=p['point_signature']:raise ValueError('model signature mismatch')
    R,X=matrices(len(labels),point['supports'],point['r'],point['x'])
    out=dict(record=dict(r),point=point,labels=labels,R=R,X=X,input=inp,configuration=json.loads(inp['configuration_json']),
        settings=dict(setting) if setting else None,ac_fit=dict(a) if a else None)
    if with_observations:
        from .common import observed
        for split in ('train','validation','test'):
            file=Path(inp[split+'_path'])
            if sha(file)!=inp[split+'_sha256']:raise ValueError('observation bytes changed')
            out[split]=observed(file,int(r['N']) if split=='train' else None)
    return out
