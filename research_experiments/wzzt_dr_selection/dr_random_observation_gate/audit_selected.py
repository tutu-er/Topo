"""Independent saved-model L1/calibration/contract audit; never calls solvers."""
from pathlib import Path
import hashlib,json,pickle
from collections import Counter
import numpy as np

BASE=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
ROOT=BASE/'wzzt_dr_random_observation_execution_v1'
OUT=BASE/'dr_random_protocol_audit_stage'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(scenes):
    p=np.concatenate([s['P_terminal'].to_numpy() for s in scenes])
    q=np.concatenate([s['Q_terminal'].to_numpy() for s in scenes])
    y=np.concatenate([s['V_root_observed'].to_numpy()[:,None]**2-s['V_terminal'].to_numpy()**2 for s in scenes])
    return p,q,y
def matrix(point,n):
    a=np.zeros((n,n));b=np.zeros_like(a)
    for support,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        a[np.ix_(support,support)]+=r;b[np.ix_(support,support)]+=x
    return a,b

def main():
    cfg=read(ROOT/'protocol.json');rows=read(ROOT/'records.json')
    histories={r['id']:pickle.loads(Path(r['path']).read_bytes()) for r in cfg['histories']}
    snapshots={r['id']:r for r in cfg['snapshots']}
    checks=[];fail=[];status=Counter();diff_train=diff_val=0.;selected_count=0;candidate_cal_count=0
    def check(name,ok,**detail):
        item=dict(name=name,pass_=bool(ok),**detail);checks.append(item)
        if not ok:fail.append(item)
    for group in ('source_sha256','input_sha256'):
        for path,h in cfg[group].items():check(group,sha(path)==h,path=path)
    for rec in rows:
        result=read(rec['result_path']);snapshot=snapshots[rec['id']]
        old=histories[snapshot['history_id']];future=pickle.loads(Path(snapshot['path']).read_bytes())
        for packet in (old,future):
            check('metadata_observed_only',packet['metadata'].get('observed_only') is True,record=rec['id'])
            forbidden={'truth','true_family','R','X','K','R_true','X_true','true_supports','p0','q0','true_baseline'}
            check('metadata_no_truth',not(forbidden&set(packet['metadata'])),record=rec['id'])
        train=arrays(old['training']+future['training']);valid=arrays(old['validation']+future['validation'])
        check('complete_history_and_new_rows',len(train[0])==480 and len(valid[0])==480,record=rec['id'])
        state=read(result['current_state_path']);cal=result['applied_calibration']
        check('snapshot_current_calibration',cal==state['calibration'],record=rec['id']);candidate_cal_count+=1
        for method,data in result['methods'].items():
            status[data['status']]+=1
            if data['status']!='complete':continue
            point=data['selected'];selected_count+=1;n=len(result['terminal_labels'])
            r,x=matrix(point,n)
            train_mae=float(np.abs(train[0]@r.T+train[1]@x.T-train[2]).mean())
            valid_mae=float(np.abs(valid[0]@r.T+valid[1]@x.T-valid[2]).mean())
            dt=abs(train_mae-point['train_mae']);dv=abs(valid_mae-point['validation_mae'])
            diff_train=max(diff_train,dt);diff_val=max(diff_val,dv)
            check('selected_train_L1',dt<=1e-12,record=rec['id'],method=method,difference=dt)
            check('selected_validation_L1',dv<=1e-12,record=rec['id'],method=method,difference=dv)
            check('fitted_common_box',min(point['r']+point['x'])>=-1e-12 and max(point['r']+point['x'])<=2+1e-12,
                  record=rec['id'],method=method)
            points=data['points'];best=min(p['validation_mae'] for p in points)
            eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*best+1e-12]
            chosen=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
            check('stored_3pct_selection',chosen==point['selection_index'],record=rec['id'],method=method)
            raw=points[chosen];tau=.16*cal['median_root_depth']
            keep=[i for i,s in enumerate(raw['supports']) if len(s)==1 or cal['a']*raw['r'][i]+cal['b']*raw['x'][i]+1e-10>=tau]
            expected_support=[raw['supports'][i] for i in keep]
            expected_r=[raw['r'][i] for i in keep];expected_x=[raw['x'][i] for i in keep]
            check('shared_current_contract_support',point['supports']==expected_support,record=rec['id'],method=method)
            check('shared_current_contract_parameters',point['r']==expected_r and point['x']==expected_x,record=rec['id'],method=method)
    result=dict(status='passed' if not fail else 'failed',source_sha256=sha(__file__),
        snapshot_records=len(rows),selected_models=selected_count,current_calibration_records=candidate_cal_count,
        checks=len(checks),failed_checks=fail,method_statuses=dict(status),max_train_L1_difference=diff_train,
        max_validation_L1_difference=diff_val,new_LP_calls=0,new_QP_calls=0,new_AC_calls=0,
        imports_project_identifier=False,truth_evaluation_not_repeated=True,
        calibration_scope='same numerical current-training calibration within a snapshot; not numerically fixed across distinct policies',
        all_checks=checks)
    OUT.mkdir(exist_ok=True);(OUT/'selected_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='all_checks'}))

if __name__=='__main__':main()
