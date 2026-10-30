"""One saved-fit algebraic decomposition witness; imports no solver or simulator."""
from pathlib import Path
import copy
import hashlib
import json
import pickle
import numpy as np

BASE=Path(__file__).resolve().parent.parent
GEN=BASE/'wzzt_dr_reference_noise_generation_v1'
EXE=BASE/'wzzt_dr_reference_noise_execution_v1'
OUT=Path(__file__).resolve().parent
CASE='case_007';SID='s_602c8cc264d072fcc944';METHOD='current_root_augmented_rnj'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def matrices(point,n):
    R=np.zeros((n,n));X=R.copy()
    for s,r,x in zip(point['supports'],point['r'],point['x']):
        R[np.ix_(s,s)]+=r;X[np.ix_(s,s)]+=x
    return R,X
def arrays(scenes):
    result=[[],[],[]]
    for scene in scenes:
        p=scene['P_terminal'].to_numpy(dtype=float);q=scene['Q_terminal'].to_numpy(dtype=float)
        v=scene['V_terminal'].to_numpy(dtype=float);root=scene['V_root_observed'].to_numpy(dtype=float)
        for store,value in zip(result,(p,q,root[:,None]**2-v**2)):store.append(value)
    return tuple(np.concatenate(x) for x in result)
def contract(point,cal):
    keep=[i for i,s in enumerate(point['supports']) if len(s)==1 or
        cal['a']*point['r'][i]+cal['b']*point['x'][i]+1e-10>=.16*cal['median_root_depth']]
    return dict(supports=[point['supports'][i] for i in keep],
        r=[point['r'][i] for i in keep],x=[point['x'][i] for i in keep])
def effective(point):return set(tuple(s) for s,r,x in zip(point['supports'],point['r'],point['x']) if max(r,x)>1e-9)
def l1(point,data):
    p,q,y=data;R,X=matrices(point,p.shape[1]);return float(np.mean(abs(p@R.T+q@X.T-y)))

def main():
    truth_path=GEN/'evaluator_truth.json';truth=read(truth_path)
    case=next(c for c in truth['cases'] if c['id']==CASE)
    mapping=next(m for m in truth['evaluations'] if m['case_id']==CASE and m['snapshot_id']==SID)
    assert case['noise_id']=='voltage_only' and case['truth']['b']==.4 and case['truth']['c']==.6
    assert mapping['policy']=='pair_random'
    record_path=EXE/'records'/f'{SID}.json';record=read(record_path)
    selected=record['methods'][METHOD]['selected'];raw=copy.deepcopy(selected['selected_uncontracted'])
    cal=record['applied_calibration'];a=cal['a'];b=cal['b'];tau=.16*cal['median_root_depth']
    root=raw['supports'].index([0,1,2]);alpha=.35;bound=2.
    cfit=np.array(raw['r'])+alpha*np.array(raw['x'])
    xmin=max(0.,(cfit[root]-bound)/alpha);xmax=min(bound,cfit[root]/alpha)
    strength=lambda x:float(a*(cfit[root]-alpha*x)+b*x)
    interval=[strength(xmin),strength(xmax)]
    assert min(interval)<tau<max(interval),'This witness must straddle the original threshold.'
    alternative=copy.deepcopy(raw);alternative['x'][root]=float(xmax)
    # This is the exact nonnegative endpoint (0,cfit/alpha). Avoid computing
    # cfit-alpha*(cfit/alpha), whose cancellation rounded slightly negative in v1.
    if xmax==cfit[root]/alpha:
        alternative['r'][root]=0.0
    else:alternative['r'][root]=float(cfit[root]-alpha*xmax)
    for key in ('r','x'):assert all(0.<=v<=bound for v in alternative[key])
    R0,X0=matrices(raw,3);R1,X1=matrices(alternative,3)
    cfg=read(EXE/'protocol.json');spec=next(s for s in cfg['snapshots'] if s['id']==SID)
    hist=next(h for h in cfg['histories'] if h['id']==spec['history_id'])
    history=pickle.loads(Path(hist['path']).read_bytes());future=pickle.loads(Path(spec['path']).read_bytes())
    splits={name:arrays(history[name]+future[name]) for name in ('training','validation')}
    original_final=contract(raw,cal);new_final=contract(alternative,cal)
    target=set(tuple(s) for s in mapping['target_family'])
    assert effective(raw)==target and [0,1,2] not in original_final['supports']
    assert effective(alternative)==target and effective(new_final)==target
    assert effective(selected)==effective(original_final)
    checks={}
    for name,data in splits.items():
        p,q,y=data;pred0=p@R0.T+q@X0.T;pred1=p@R1.T+q@X1.T
        checks[name]=dict(rows=len(p),max_Q_minus_point35P=float(np.max(abs(q-alpha*p))),
            max_raw_prediction_difference=float(np.max(abs(pred0-pred1))),
            original_raw_L1=l1(raw,data),alternative_raw_L1=l1(alternative,data),
            raw_L1_difference=abs(l1(raw,data)-l1(alternative,data)),
            original_final_L1=l1(original_final,data),alternative_final_L1=l1(new_final,data))
        assert checks[name]['max_raw_prediction_difference']<1e-12
        assert checks[name]['raw_L1_difference']<1e-12
    np.savez(OUT/'algebra_arrays_v2.npz',supports=np.array(raw['supports'],dtype=object),
        original_r=raw['r'],original_x=raw['x'],alternative_r=alternative['r'],alternative_x=alternative['x'],
        R_original=R0,X_original=X0,R_alternative=R1,X_alternative=X1,
        K_original=R0+alpha*X0,K_alternative=R1+alpha*X1,
        root_x_interval=[xmin,xmax],root_strength_interval=interval,tau=tau)
    receipt=dict(status='verified_single_saved_fit_witness',version=2,case_id=CASE,snapshot_id=SID,method=METHOD,
        scope='post-selection algebra only; no new fit, rule change, price design or recovery claim',
        noise_id=case['noise_id'],true_b=case['truth']['b'],true_c=case['truth']['c'],true_pair=case['truth']['pair'],
        alpha=alpha,coefficient_domain=[0.,bound],calibration=cal,
        root_original=[raw['r'][root],raw['x'][root]],root_alternative=[alternative['r'][root],alternative['x'][root]],
        root_cfit=float(cfit[root]),root_feasible_x_interval=[xmin,xmax],root_strength_interval=interval,
        tau=tau,root_original_strength=strength(raw['x'][root]),root_alternative_strength=strength(xmax),
        crossing_x=float((tau-a*cfit[root])/(b-alpha*a)),strength_slope=float(b-alpha*a),
        original_raw_supports=raw['supports'],original_final_supports=original_final['supports'],
        alternative_final_supports=new_final['supports'],target_supports=mapping['target_family'],
        max_block_cfit_difference=float(np.max(abs(cfit-(np.array(alternative['r'])+alpha*np.array(alternative['x']))))),
        max_combined_K_difference=float(np.max(abs((R0+alpha*X0)-(R1+alpha*X1)))),checks=checks,
        source_sha256=sha(__file__),input_sha256={str(p):sha(p) for p in
            (truth_path,record_path,EXE/'protocol.json',Path(hist['path']),Path(spec['path']))},
        arrays_sha256=sha(OUT/'algebra_arrays_v2.npz'),new_LP=0,new_QP=0,new_AC=0,new_data=False,
        endpoint_rounding_correction='root r explicitly 0.0 at x=cfit/.35; strict nonnegative domain checked',
        preserved_v1_sha256={name:sha(OUT/name) for name in ('audit_v1.py','receipt_v1.json','algebra_arrays_v1.npz')},
        original_rules_unchanged=True,modified_record_or_packet=False)
    (OUT/'receipt_v2.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('status','root_cfit','root_feasible_x_interval',
        'root_strength_interval','tau','root_original_strength','root_alternative_strength',
        'crossing_x','max_combined_K_difference','checks','source_sha256')},indent=2))

if __name__=='__main__':main()
