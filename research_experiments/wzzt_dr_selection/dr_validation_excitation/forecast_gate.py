"""History-only mean forecast gate for a true refitted 3% exit condition.

First evaluate no purchase. A positive margin here is a free predicted exit,
not evidence of priced DR benefit. No old test observation is loaded here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP, canonical
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize, arrays, matrices
from .audit_existing import read, ref, bound, write, family
from .profile_margin import validation


def evaluate_gate(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    c=read(source/'main/protocol.json')
    protocol=dict(status='frozen',created_utc=datetime.now(timezone.utc).isoformat(),
        historical_inputs=dict(history=c['history_packet'],fit=c['historical_baseline_result']),
        old_protocol=ref(source/'main/protocol.json'),budget=1.6,
        forecast='same-time mean of all six historical measured P/Q/drop days',
        new_rows=[480,480],weight_bounds=[0,2],validation_band=.03,
        working_reference='historical finer complete joint family; not true topology',
        operation='no-purchase gate before searching prices',
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).parent.glob('*.py')},
        new_AC=0,new_RNJ_W=0,new_actual_future=0)
    write(output/'protocol.json',protocol)
    with bound(c['history_packet']).open('rb') as stream:h=pickle.load(stream)['history']
    old=read(bound(c['historical_baseline_result'])); comp=c['competition']
    coarse=family(old['selection_candidates'][comp['selected_index']])
    fine=family(old['selection_candidates'][comp['alternative_index']])
    daily=[]
    for day in h['training']+h['validation']:
        obs,_,_=normalize([day],'past')
        daily.append(arrays(obs))
    mean=tuple(np.mean([v[k] for v in daily],axis=0) for k in range(3))
    obs,_,_=normalize(h['training'],'past_train'); train=arrays(obs)
    obs,_,_=normalize(h['validation'],'past_val'); valid=arrays(obs)
    predicted_train=tuple(np.vstack((train[k],mean[k],mean[k])) for k in range(3))
    predicted_valid=tuple(np.vstack((valid[k],mean[k],mean[k])) for k in range(3))
    lp=PoolLP(None,canonical([*coarse,*fine]),data=predicted_train,lp_method=2)
    try:
        p0=lp.fit(coarse);p1=lp.fit(fine)
        v0,v1=validation(p0,predicted_valid),validation(p1,predicted_valid)
        result=dict(status='complete',coarse_validation=v0,fine_validation=v1,
            no_purchase_exit_margin=v0-1.03*v1-1e-12,
            acceptance_margin=p0['objective']-p1['objective']-1e-10-.001*p0['objective'],
            training_LP_calls=len(lp.solves),
            input_scope='history only; predicted mean future, no physical/test future',
            gate='free_predicted_exit' if v0-1.03*v1-1e-12>0 else 'paid_response_search_needed',
            new_prices_selected=False,new_actual_test=False)
        write(output/'summary.json',result)
        print(json.dumps(result),flush=True)
        return result
    finally:lp.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();evaluate_gate(args.source,args.output)


if __name__=='__main__':main()
