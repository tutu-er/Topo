"""Independent direct concatenated-score check of three already-scored states."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np

BASE=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
OUT=BASE/'wzzt_dr_competition_validation_noise_v4'
PARENT=BASE/'wzzt_dr_competition_bridge_v2'
def read(p):return json.loads(Path(p).read_text())
def matrix(p):
 r=np.zeros((15,15));x=r.copy()
 for s,a,b in zip(p['supports'],p['r'],p['x']):r[np.ix_(s,s)]+=a;x[np.ix_(s,s)]+=b
 return r,x
def main():
 table=json.loads(gzip.decompress((OUT/'tables.json.gz').read_bytes()));cfg=read(PARENT/'protocol.json')
 row=next(r for r in cfg['cases'] if r['voltage_sigma']==.001);folder=PARENT/'planning'/row['job_id']/'response_table'
 old=read(row['existing_result']['path']);cal=old['calibration'];days=[]
 for role in ('train','validation'):
  with np.load(row[role]['path'],allow_pickle=False) as f:days.append({k:f[k].copy() for k in ('P','Q','V','root')})
 fp=np.concatenate([d['P'] for d in days]).mean(axis=0);fq=np.concatenate([d['Q'] for d in days]).mean(axis=0)
 vp=days[1]['P'].reshape(288,15);vq=days[1]['Q'].reshape(288,15);vy=(days[1]['root'][:,:,None]**2-days[1]['V']**2).reshape(288,15)
 keys=[tuple(k) for k in table['keys']];indices={k:i for i,k in enumerate(keys)};positions=[old['labels'].index(v) for v in table['active_labels']]
 cases=[(1,0,(0,0),(0,0)),(2,7,(3,3),(1,2)),(0,3,(2,1),(3,2))];results=[]
 for m,seed_index,training,validation in cases:
  record=read(folder/f'training_masks/m{m}_u{training[0]}_{training[1]}.json');h=old['base_points'][record['raw_index']];hr,hx=matrix(h)
  pp=[vp];qq=[vq];yy=[vy];seed=620010+seed_index
  for event,mask in enumerate(validation):
   p,q=fp.copy(),fq.copy()
   for j,pos in enumerate(positions):
    if mask>>j&1:p[76:80,pos]-=.001;p[80:84,pos]+=.001;q[76:80,pos]-=.00035;q[80:84,pos]+=.00035
   v=np.sqrt(1.-(p@hr.T+q@hx.T))
   zz=[np.random.default_rng(np.random.SeedSequence([seed,event,s])).normal(size=(96,n)) for s,n in enumerate((15,15,15,1))]
   pp.append(p+.005*np.maximum(abs(p),1e-12)*zz[0]);qq.append(q+.005*np.maximum(abs(q),1e-12)*zz[1])
   observed_v=v+.001*v*zz[2];root=1.+.001*zz[3];yy.append(root**2-observed_v**2)
  p,q,y=(np.concatenate(v) for v in (pp,qq,yy));scores=[]
  for point in record['candidate_points']:
   r,x=matrix(point);scores.append(float(abs(p@r.T+q@x.T-y).mean()))
  for alg,limit in enumerate((len(scores),15)):
   best=min(scores[:limit]);eligible=[i for i,v in enumerate(scores[:limit]) if v<=1.03*best+1e-12]
   chosen=min(eligible,key=lambda i:(len(record['candidate_points'][i]['supports']),scores[i],i));point=record['candidate_points'][chosen]
   target={tuple(s) for s,r,x in zip(h['supports'],h['r'],h['x']) if r>0 or x>0}
   family={tuple(s) for s,r,x in zip(point['supports'],point['r'],point['x']) if (r>0 or x>0) and (len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=cal['tau'])}
   expected=table['losses'][seed_index+1][m][indices[training]][indices[validation]][alg]
   expected_score=table['selected_raw_validation_mae'][seed_index+1][m][indices[training]][indices[validation]][alg]
   assert len(family^target)==expected and abs(scores[chosen]-expected_score)<1e-11
   results.append(dict(model=m,seed=seed,training=training,validation=validation,algorithm=alg,error=expected,selected_index=chosen,score_difference=abs(scores[chosen]-expected_score)))
 manifest=read(OUT/'manifest.json')
 for name,entry in manifest.items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==entry['sha256']
 result=dict(status='passed',independent_direct_states=3,algorithm_scores=6,results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  protocol_and_outputs_verified=len(manifest),solver_calls=0,AC_calls=0,new_actual_future_read=False)
 path=OUT/'audit_three_states.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='results'}))
if __name__=='__main__':main()
