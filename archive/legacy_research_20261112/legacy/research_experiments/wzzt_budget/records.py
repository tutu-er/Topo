"""Read selected RNJ+WZZT models and their matched observations from a catalog."""
from pathlib import Path
import argparse
import json
import sqlite3
from research_experiments.wzzt_systematic.common import matrices,observed,sha


def find(catalog,*,case=None,condition=None,N=None,variant=None):
    catalog=Path(catalog).resolve()
    if not catalog.is_file():raise FileNotFoundError(catalog)
    clauses=[];values=[]
    for column,value in (("case_name",case),("condition_name",condition),("N",N),("variant",variant)):
        if value is not None:clauses.append(f"{column}=?");values.append(value)
    query="SELECT * FROM selected_models JOIN datasets USING(job_id)"
    if clauses:query+=" WHERE "+" AND ".join(clauses)
    query+=" ORDER BY case_name,condition_name,N,variant"
    with sqlite3.connect(catalog) as connection:
        connection.row_factory=sqlite3.Row
        rows=[dict(row) for row in connection.execute(query,values)]
    connection.close()
    return rows


def load(catalog,*,case,condition,N,variant,with_observations=False,verify=True):
    rows=find(catalog,case=case,condition=condition,N=N,variant=variant)
    if len(rows)!=1:raise ValueError(f"expected one selected model, found {len(rows)}")
    row=rows[0]
    labels=json.loads(row.pop("labels_json"))
    supports=json.loads(row.pop("supports_json"));r=json.loads(row.pop("r_json"));x=json.loads(row.pop("x_json"))
    R,X=matrices(len(labels),supports,r,x)
    result=dict(labels=labels,supports=supports,support_labels=[[labels[i] for i in s] for s in supports],r=r,x=x,R=R,X=X,record=row)
    if with_observations:
        for name in ("train","validation","test"):
            file=Path(row[name+"_path"])
            if verify and sha(file)!=row[name+"_sha256"]:raise ValueError("data hash changed: "+str(file))
            result[name]=observed(file,N if name=="train" else None)
    return result


def load_trace(catalog,*,case,condition,N,verify=True):
    rows=find(catalog,case=case,condition=condition,N=N)
    if not rows:raise ValueError("input group not found")
    row=rows[0];file=Path(row["result_path"])
    if verify and sha(file)!=row["result_sha256"]:raise ValueError("search trace hash changed: "+str(file))
    return json.loads(file.read_text(encoding="utf-8"))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("catalog");p.add_argument("--case");p.add_argument("--condition")
    p.add_argument("--N",type=int);p.add_argument("--variant");a=p.parse_args()
    rows=find(a.catalog,case=a.case,condition=a.condition,N=a.N,variant=a.variant)
    keys=("job_id","variant","clade_f1","validation_mae","linear_test_mae","selected_origin","search_stop")
    print(json.dumps([{k:r.get(k) for k in keys} for r in rows],ensure_ascii=False,indent=2))
