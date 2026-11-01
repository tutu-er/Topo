"""Persist optimization evidence before optional numerical/DTO conversion."""
from __future__ import annotations
import hashlib,json,traceback
from pathlib import Path
import numpy as np
from .adapter import build_plan
from .repair_tools import polish_curve


def _write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def persist_and_adapt(folder,result,problem,scenarios,resources,node_labels,case):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    _write(folder/'raw_solver.json',result)
    r=dict(result)
    if r.get('mean_u') is None:return r
    scientific_status=r['status']
    try:
        r=polish_curve(problem,scenarios,r)
        _write(folder/'polished_solver.json',r)
        adapted=build_plan(resources,np.asarray(r['mean_u']),np.asarray(r['prices']),
            node_labels=node_labels,budget=problem['budget'],policy='fixed24h_minfee',
            metadata=dict(scope='conditional model pilot; not deployed',case=case))
        path=folder/'plan.json';_write(path,adapted['plan'])
        r['plan']=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        r['generic_adapter_audit']=adapted['audit']
    except Exception as error:
        r.update(status='adapter_failed',optimization_status=scientific_status,
            adapter_error=dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),
            raw_solver_retained=True)
    return r
