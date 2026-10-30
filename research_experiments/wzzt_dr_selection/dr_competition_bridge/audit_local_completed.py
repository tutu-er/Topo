"""Independent replay of completed conditional records; no parent imports/LP/AC."""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np

ROOT=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_dr_competition_bridge_v1')
REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(f):return tuple(sorted(set(tuple(sorted(s)) for s in f),key=lambda s:(len(s),s)))
def compatible(f,c):
 c=set(c)
 return all(not c&set(s) or c<=set(s) or set(s)<=c for s in f)
def matrices(p,n):
 r=np.zeros((n,n));x=r.copy()
 assert len(p['supports'])==len(p['r'])==len(p['x'])
 for s,rv,xv in zip(p['supports'],p['r'],p['x']):
  assert rv>=0 and xv>=0 and math.isfinite(rv) and math.isfinite(xv)
  r[np.ix_(s,s)]+=rv;x[np.ix_(s,s)]+=xv
 return r,x
def effective(p):return {tuple(sorted(s)) for s,r,x in zip(p['supports'],p['r'],p['x']) if r>0 or x>0}
def predict(p,values):
 r,x=matrices(p,values[0].shape[1]);return values[0]@r.T+values[1]@x.T
def close(a,b,tolerance=2e-10):
 assert abs(float(a)-float(b))<=tolerance,(a,b,tolerance)

def main():
 protocol=read(ROOT/'protocol.json')
 job=protocol['local_price_case']
 row=next(r for r in protocol['cases'] if r['job_id']==job)
 hist=[];input_hashes={}
 for role in ('train','validation'):
  p=Path(row[role]['path']);assert p.name==role+'_observed.npz'
  assert sha(p)==row[role]['catalog_sha256'];input_hashes[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as f:
   assert set(f.files)=={'P','Q','V','root','labels'}
   pv,qv,vv,rv=(np.asarray(f[k],float) for k in ('P','Q','V','root'))
   labels=f['labels'].astype(int).tolist()
  assert pv.shape==(3,96,15) and qv.shape==pv.shape and vv.shape==pv.shape and rv.shape==(3,96)
  hist.append((pv,qv,rv[:,:,None]**2-vv**2))
 old_path=Path(row['existing_result']['path']);assert sha(old_path)==row['existing_result']['catalog_sha256']
 old=read(old_path);input_hashes[str(old_path)]=sha(old_path)
 folder=ROOT/'planning'/job/'response_table'
 plan_protocol=read(folder/'plan_protocol.json')
 identity=plan_protocol['identity'];positions=[labels.index(x) for x in identity['active_labels']]
 assert len(old['rnj_bank'])==15
 forecast=(np.concatenate([hist[0][0],hist[1][0]]).mean(axis=0),
           np.concatenate([hist[0][1],hist[1][1]]).mean(axis=0))
 futures={}
 for key in range(4):
  p,q=(a.copy() for a in forecast)
  for j,pos in enumerate(positions):
   if (key>>j)&1:
    p[76:80,pos]-=.2/200;p[80:84,pos]+=.2/200
    q[76:80,pos]-=.07/200;q[80:84,pos]+=.07/200
  futures[key]=(p,q)
 train=tuple(a.reshape(288,15) for a in hist[0])
 valid=tuple(a.reshape(288,15) for a in hist[1])
 cal=old['calibration']
 checked=[];max_l1_error=0.;max_validation_error=0.;all_moves=[]
 for path in sorted((folder/'training_masks').glob('*.json')):
  record=read(path)
  if record['status']!='complete':continue
  h=old['base_points'][record['raw_index']]
  keys=record['training_mask_key']
  hypothetical={key:predict(h,futures[key]) for key in futures}
  p=np.concatenate([train[0],futures[keys[0]][0],futures[keys[1]][0]])
  q=np.concatenate([train[1],futures[keys[0]][1],futures[keys[1]][1]])
  y=np.concatenate([train[2],hypothetical[keys[0]],hypothetical[keys[1]]])
  assert p.shape==(480,15) and record['training_rows']==480 and record['validation_rows']==480
  point_count=0
  for point in record['candidate_points']:
   f=canonical(point['supports'])
   assert all(compatible(f[:i],c) for i,c in enumerate(f))
   loss=float(np.mean(abs(predict(point,(p,q))-y)))
   max_l1_error=max(max_l1_error,abs(loss-point['objective']))
   close(loss,point['objective']);point_count+=1
  assert record['bank_points']==record['candidate_points'][:15]
  assert all(p.get('origin')=='tree_bank' for p in record['bank_points'])
  for source,bank_point in zip(old['rnj_bank'],record['bank_points']):
   assert source['config']==bank_point['config']
   assert canonical(source['native_family'])==canonical(bank_point['native_family'])
   assert set(canonical(bank_point['supports']))<=set(canonical(source['native_family']))
  receipts={canonical(row['family']):row for row in record['LP_solves']}
  assert record['LP_count']==len(record['LP_solves'])
  assert all(r['status']==2 and r['max_constraint_violation']<=1e-7 and r['max_bound_violation']<=1e-7 for r in record['LP_solves'])
  path_moves=[]
  for name,trajectory in record['paths'].items():
   assert trajectory['reason']=='pool_local_optimum'
   states=trajectory['path'];state=0
   for decision in trajectory['decisions']:
    current=states[state];family=canonical(current['supports'])
    close(current['objective'],decision['current'])
    expected_candidates=[c for c in canonical(plan_protocol['pool']) if len(c)>=2 and c not in family]
    assert [tuple(c['candidate']) for c in decision['candidates']]==expected_candidates
    best=None
    for item in decision['candidates']:
     c=tuple(item['candidate'])
     retained=[s for s in family if compatible([s],c)]
     removed=[s for s in family if s not in retained]
     assert canonical(item['removed'])==canonical(removed)
     proposed=canonical(retained+[c])
     assert proposed in receipts
     close(item['objective'],receipts[proposed]['objective'])
     if best is None or item['objective']<best['objective']-1e-12:best=item
    gain=0. if best is None else current['objective']-best['objective']
    close(gain,decision['gain'])
    if gain>1e-10+.001*current['objective']:
     state+=1;assert state<len(states)
     close(states[state]['objective'],best['objective'])
     path_moves.append(dict(name=name,removed=best['removed'],gain=gain))
    assert decision['full_sweep']
   assert state==len(states)-1
  all_moves.extend(path_moves)
  points=record['candidate_points']
  past_errors=[];future_errors=[]
  for point in points:
   past_errors.append(float(abs(predict(point,valid)-valid[2]).sum()))
   future_errors.append({k:float(abs(predict(point,futures[k])-hypothetical[k]).sum()) for k in futures})
  target=effective(h)
  assert target=={tuple(s) for s in record['target_raw_positive_family']}
  assert len(record['validation_results'])==16
  for vr in record['validation_results']:
   a,b=vr['validation_key']
   scores=[(past_errors[i]+future_errors[i][a]+future_errors[i][b])/(480*15) for i in range(len(points))]
   for offset,which in ((None,'wzzt'),(15,'RNJ')):
    values=scores if offset is None else scores[:offset]
    best=min(values)
    eligible=[i for i,loss in enumerate(values) if loss<=1.03*best+1e-12]
    index=min(eligible,key=lambda i:(len(points[i]['supports']),values[i],i))
    assert index==vr[which+'_selected_index']
    chosen=points[index]
    keep=[i for i,(s,r,x) in enumerate(zip(chosen['supports'],chosen['r'],chosen['x'])) if len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=.16*cal['median_root_depth']]
    expected=dict(supports=[chosen['supports'][i] for i in keep],r=[chosen['r'][i] for i in keep],x=[chosen['x'][i] for i in keep])
    selected=vr['selected_wzzt' if which=='wzzt' else 'selected_RNJ']
    assert canonical(expected['supports'])==canonical(selected['supports'])
    close(values[index],vr[which+'_raw_selection_mae'])
    post=(float(abs(predict(expected,valid)-valid[2]).sum())+
          float(abs(predict(expected,futures[a])-hypothetical[a]).sum())+
          float(abs(predict(expected,futures[b])-hypothetical[b]).sum()))/(480*15)
    max_validation_error=max(max_validation_error,abs(post-vr[which+'_postcontract_mae']))
    close(post,vr[which+'_postcontract_mae'])
    assert len(effective(expected)^target)==vr['wzzt_loss' if which=='wzzt' else 'RNJ_loss']
  checked.append(dict(file=path.name,sha256=sha(path),candidates=point_count,LP_receipts=len(record['LP_solves']),validation_masks=16,accepted_moves=len(path_moves)))
 out=ROOT/'audit_partial_planning.json'
 assert not out.exists(),'retain existing independent audit'
 report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='passed',scope='Completed conditional training records only; independent raw objective and selection replay, no parent imports',
  source_sha256={str(Path(__file__).resolve()):sha(__file__)},inputs_sha256=input_hashes,
  records=checked,completed_training_masks=len(checked),candidate_points=sum(r['candidates'] for r in checked),
  validation_selections=sum(2*r['validation_masks'] for r in checked),LP_receipts_checked=sum(r['LP_receipts'] for r in checked),
  max_objective_difference=max_l1_error,max_postcontract_validation_difference=max_validation_error,
  accepted_moves=all_moves,solver_calls=0,AC_calls=0,NPZ_files_read=['old train observed','old validation observed'],
  new_future_values_read=False,truth_files_read=False,full_LP_dual_proof=False,
  limitations=['Rejected fits have receipts without coefficients, so their raw residuals are not independently reconstructed','Exact global LP optimality is not proved by this receipt/numerical replay','Only already-complete records; interrupted record retained and not restarted'])
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','inputs_sha256','records','accepted_moves','limitations')}))

if __name__=='__main__':main()
