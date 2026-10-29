from pathlib import Path
import json
import sqlite3
import hashlib
import zipfile
from research_experiments.wzzt_systematic.common import matrices,sha
from .common import observed


def restore(directory,*,names=None):
    directory=Path(directory).resolve();archive=directory/'records.zip'
    manifest=json.loads((directory/'delivery_manifest.json').read_text(encoding='utf-8'))
    if sha(archive)!=manifest['records.zip']:raise ValueError('record archive bytes changed')
    with zipfile.ZipFile(archive) as z:
        selected=z.namelist() if names is None else names
        for name in selected:
            if Path(name).name!=name or (directory/name).resolve().parent!=directory:raise ValueError('archive path outside delivery folder')
            data=z.read(name)
            if hashlib.sha256(data).hexdigest()!=manifest[name]:raise ValueError('archived member hash mismatch')
            path=directory/name
            if path.exists():
                if sha(path)!=manifest[name]:raise ValueError('existing local record changed: '+name)
            else:path.write_bytes(data)


def find(catalog,*,case=None,panel=None,condition=None,voltage_sigma=None,bundle=None,N=None,variant=None):
    if not Path(catalog).is_file():restore(Path(catalog).parent,names=['catalog.sqlite'])
    conn=sqlite3.connect(catalog);conn.row_factory=sqlite3.Row;conditions=[];values=[]
    for key,value in dict(case=case,panel=panel,condition=condition,voltage_sigma=voltage_sigma,bundle=bundle,N=N,variant=variant).items():
        if value is not None:conditions.append('"'+key+'"=?');values.append(value)
    query='SELECT * FROM selected_models JOIN datasets USING(job_id)'
    if conditions:query+=' WHERE '+' AND '.join(conditions)
    try:return [dict(r) for r in conn.execute(query+' ORDER BY job_id,variant',values)]
    finally:conn.close()


def load(catalog,*,case,panel,condition,voltage_sigma,bundle,N,variant,with_observations=False):
    rows=find(catalog,case=case,panel=panel,condition=condition,voltage_sigma=voltage_sigma,bundle=bundle,N=N,variant=variant)
    if len(rows)!=1:raise ValueError(f'expected one row; found {len(rows)}')
    row=rows[0]
    if not row['model_available']:raise ValueError('unavailable model; inspect failure status')
    labels=json.loads(row['labels_json']);supports=json.loads(row['supports_json']);r=json.loads(row['r_json']);x=json.loads(row['x_json'])
    R,X=matrices(len(labels),supports,r,x);out=dict(labels=labels,supports=supports,r=r,x=x,R=R,X=X,record=row)
    if with_observations:
        for split in ('train','validation','test'):
            path=Path(row[split+'_path'])
            if sha(path)!=row[split+'_sha256']:raise ValueError('input bytes changed')
            out[split]=observed(path,N if split=='train' else None)
    return out


def load_trace(catalog,**filters):
    rows=find(catalog,**filters)
    if not rows or len({r['job_id'] for r in rows})!=1:raise ValueError('specify exactly one input condition')
    row=rows[0];path=Path(row['result_path'])
    if sha(path)!=row['result_sha256']:raise ValueError('raw result bytes changed')
    return json.loads(path.read_text(encoding='utf-8'))
