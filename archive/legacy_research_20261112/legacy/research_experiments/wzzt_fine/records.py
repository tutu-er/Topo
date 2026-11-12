from pathlib import Path
import json
import sqlite3
from research_experiments.wzzt_systematic.common import matrices,sha
from .common import observed


def find(catalog,*,case=None,voltage_sigma=None,bundle=None,N=None,variant=None):
    path=Path(catalog)
    if not path.is_file():raise FileNotFoundError(path)
    conditions=[];values=[]
    for key,value in (("case_name",case),("voltage_sigma",voltage_sigma),("bundle",bundle),("N",N),("variant",variant)):
        if value is not None:conditions.append(key+"=?");values.append(value)
    sql="SELECT * FROM selected_models JOIN datasets USING(job_id)"
    if conditions:sql+=" WHERE "+" AND ".join(conditions)
    sql+=" ORDER BY case_name,voltage_sigma,bundle,N,variant"
    connection=sqlite3.connect(path);connection.row_factory=sqlite3.Row
    try:return [dict(r) for r in connection.execute(sql,values)]
    finally:connection.close()


def load(catalog,*,case,voltage_sigma,bundle,N,variant,with_observations=False):
    rows=find(catalog,case=case,voltage_sigma=voltage_sigma,bundle=bundle,N=N,variant=variant)
    if len(rows)!=1:raise ValueError(f"expected one model, found {len(rows)}")
    r=rows[0]
    if not r['model_available']:raise ValueError('this method has no saved model; inspect failure status')
    labels=json.loads(r.pop('labels_json'));supports=json.loads(r.pop('supports_json'))
    rv,xv=json.loads(r.pop('r_json')),json.loads(r.pop('x_json'));R,X=matrices(len(labels),supports,rv,xv)
    out=dict(labels=labels,supports=supports,support_labels=[[labels[i] for i in s] for s in supports],r=rv,x=xv,R=R,X=X,record=r)
    if with_observations:
        for split in ('train','validation','test'):
            path=Path(r[split+'_path'])
            if sha(path)!=r[split+'_sha256']:raise ValueError('input hash changed: '+str(path))
            out[split]=observed(path,N if split=='train' else None)
    return out


def load_trace(catalog,*,case,voltage_sigma,bundle,N):
    rows=find(catalog,case=case,voltage_sigma=voltage_sigma,bundle=bundle,N=N)
    if not rows:raise ValueError('input not found')
    row=rows[0];path=Path(row['result_path'])
    if sha(path)!=row['result_sha256']:raise ValueError('trace hash changed')
    return json.loads(path.read_text(encoding='utf-8'))
