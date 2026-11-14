"""Offline universal-support audit; no identifier, AC, or candidate mutation.

Input arrays and fitted values remain under an explicitly supplied output path.
Use --summary-only to inspect a completed receipt without fitting anything.
"""
from pathlib import Path
import sys, json, hashlib, collections, platform, argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from topology_identification_standalone import data as physical
from rnj_wzzt_dr.observations import Observation
from rnj_wzzt_dr.identification.optimization import QPFitter

OUT=ROOT/'outputs/measured_pq_flex_20261114_audit'
INDEX=ROOT/'outputs/measured_pq_sources_20261114/qualified_inputs_v2.json'
SPLITS={'train':(0,192),'validation':(192,288),'new_day':(288,384)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def matrices(pt,n):
 z=np.array([[i in c for i in range(n)] for c in pt['supports']],float)
 return (z.T*np.array(pt['r']))@z,(z.T*np.array(pt['x']))@z
def common(pt,n):
 k=next((k for k,c in enumerate(pt['supports']) if len(c)==n),None)
 return (0.,0.) if k is None else (pt['r'][k],pt['x'][k])
def mse(e):return float(np.mean(np.square(e)))
def rms(e):return float(np.sqrt(mse(e)))
def ratio(a,b):return float(a/b) if b else None
def select(ms,counts,native):
 v=np.array(ms)/native['reference_mse']; ix=np.flatnonzero(v<=v.min()+native['config']['validation_tie'])
 return int(min(ix,key=lambda i:(counts[i],v[i],i)))
def prediction(R,X,o):return o.p@R.T+o.q@X.T
def truth_and_R(net,labels):
 children=collections.defaultdict(list)
 for b in net['branches']:
  if b['status']:children[b['from_bus']].append(b['to_bus'])
 lookup={x:i for i,x in enumerate(labels)}
 def desc(v):return ({lookup[v]} if v in lookup else set()).union(*(desc(c) for c in children[v]))
 f=[];R=np.zeros((len(labels),len(labels)));X=R.copy();zbase=net['base_kv']**2/net['base_mva']
 for b in net['branches']:
  if not b['status']:continue
  c=tuple(sorted(desc(b['to_bus'])));f.append(c);z=np.array([i in c for i in range(len(labels))],float)
  R+=2*b['r_ohm']/zbase*np.outer(z,z);X+=2*b['x_ohm']/zbase*np.outer(z,z)
 return sorted(set(f),key=lambda x:(len(x),x)),R,X,[sorted(desc(c)) for c in children[net['root_bus']]]
def topo(pt,truth,n,tol,remove=False):
 a={tuple(c) for c,r,x in zip(pt['supports'],pt['r'],pt['x']) if max(r,x)>tol and not(remove and len(c)==n)}
 t=set(truth); ai={c for c in a if 1<len(c)<n};ti={c for c in t if 1<len(c)<n}
 return dict(full_exact=a==t,internal_exact=ai==ti,missing=len(t-a),extra=len(a-t))
def run():
 if OUT.exists():raise FileExistsError('Preserve the completed audit; choose a new --out directory')
 OUT.mkdir(parents=True)
 index=read(INDEX); protocol=dict(index_sha256=digest(INDEX),source_sha256=digest(__file__),python=platform.python_version(),
  purpose='diagnose universal-support coefficient; do not replace candidates or tune outcomes',
  input_count=len(index['inputs']),controlled_fits='true family, true family+universal; each fitted on observed, observed without root meter error, clean physical training only',
  controlled_bound=2.,evaluation='all 192 train,96 validation,96 new-day rows saved previously; no AC/identifier rerun',
  first_order_response='2*physical impedance/base impedance, a physical linear reference, not new AC Jacobian')
 write(OUT/'protocol.json',protocol);rows=[]
 for entry in index['inputs']:
  native_path=Path(entry['files']['101_native.json']['path']);d=native_path.parent
  if digest(native_path)!=entry['files']['101_native.json']['sha256']:raise ValueError('Native candidate source changed')
  native=read(native_path);n=len(native['frozen_candidates'][0]['labels']);labels=tuple(native['frozen_candidates'][0]['labels'])
  z=np.load(d/'private_physical.npz');net=physical.load_case(entry['case']);truth,Rtrue,Xtrue,roots=truth_and_R(net,labels);obs={};clean={};noroot={}
  for s,(lo,hi) in SPLITS.items():
   f=np.load(d/(s+'.npz'));obs[s]=Observation(f['P'],f['Q'],f['Y'],labels)
   clean[s]=Observation(z['P'][lo:hi],z['Q'][lo:hi],z['root'][lo:hi,None]**2-z['V'][lo:hi]**2,labels)
   noroot[s]=Observation(f['P'],f['Q'],z['root'][lo:hi,None]**2-f['V']**2,labels)
  hist=native['selected']['index'];bank=[];counts=[]
  for i,can in enumerate(native['frozen_candidates']):
   pt=can['point'];R,X=matrices(pt,n);rc,xc=common(pt,n);R0,X0=R-rc,X-xc;counts.append(sum(1<len(c)<n for c in pt['supports'])); losses={}
   for s,o in obs.items():
    co=prediction(np.full((n,n),rc),np.full((n,n),xc),o);e=o.y-prediction(R,X,o);ec=clean[s].y-prediction(R,X,clean[s]);without=o.y-prediction(R0,X0,o);clean_without=clean[s].y-prediction(R0,X0,clean[s])
    lo,hi=SPLITS[s];f=np.load(d/(s+'.npz'));rooterr=f['root'][:,None]**2-z['root'][lo:hi,None]**2;terminalerr=z['V'][lo:hi]**2-f['V']**2;pqerr=-prediction(R,X,Observation(o.p-clean[s].p,o.q-clean[s].q,np.zeros_like(o.y),labels))
    assert np.allclose(e,ec+rooterr+terminalerr+pqerr,rtol=0,atol=1e-14)
    losses[s]=dict(observed_mse=mse(e),deleted_observed_mse=mse(without),clean_mse=mse(ec),deleted_clean_mse=mse(clean_without),
     common_response_rms=rms(co),common_response_over_observed_residual_rms=rms(co)/rms(e),common_response_over_clean_signal_rms=rms(co)/rms(clean[s].y),
     root_meter_error_rms=rms(rooterr),terminal_meter_error_rms=rms(terminalerr),pq_meter_error_response_rms=rms(pqerr),
     delete_delta_decomposition=dict(square_common=mse(co),clean_bias=2*float(np.mean(ec*co)),root_meter=2*float(np.mean(rooterr*co)),terminal_meter=2*float(np.mean(terminalerr*co)),pq_meter=2*float(np.mean(pqerr*co))))
   bank.append(dict(index=i,r_common=rc,x_common=xc,common_active=max(rc,xc)>native['config']['zero_tol'],
    r_over_median_self_r=ratio(rc,float(np.median(np.diag(R)))),x_over_median_self_x=ratio(xc,float(np.median(np.diag(X)))),
    r_operator_fraction=ratio(n*rc,float(np.linalg.norm(R,2))),x_operator_fraction=ratio(n*xc,float(np.linalg.norm(X,2))),
    single_node_10kW_common_delta_y=rc*10/net['base_kw'],single_node_10kvar_common_delta_y=xc*10/net['base_kw'],
    topology=topo(pt,truth,n,native['config']['zero_tol']),deleted_topology=topo(pt,truth,n,native['config']['zero_tol'],True),losses=losses))
  decisions={}
  for split in ['validation','cumulative']:
   for remove in [False,True]:
    k='deleted_observed_mse' if remove else 'observed_mse';sc=[b['losses']['validation'][k] if split=='validation' else (b['losses']['validation'][k]+b['losses']['new_day'][k])/2 for b in bank];ix=select(sc,counts,native);decisions[split+('_deleted' if remove else '')]=dict(index=ix,mse=sc[ix],topology=bank[ix]['deleted_topology' if remove else 'topology'])
  other=[b['index'] for b in bank if b['index']!=hist];runner=min(other,key=lambda i:bank[i]['losses']['validation']['observed_mse']) if other else None;pairs=[]
  Rt,Xt=matrices(native['frozen_candidates'][hist]['point'],n)
  for i in other:
   Rj,Xj=matrices(native['frozen_candidates'][i]['point'],n);dr=bank[hist]['r_common']-bank[i]['r_common'];dx=bank[hist]['x_common']-bank[i]['x_common'];o=obs['new_day'];delta=prediction(Rt-Rj,Xt-Xj,o);dc=prediction(np.full((n,n),dr),np.full((n,n),dx),o);d0=delta-dc
   pairs.append(dict(competitor=i,is_nearest_validation=i==runner,common_r_difference=dr,common_x_difference=dx,mean_squared_distance=mse(delta),common_only_distance=mse(dc),common_only_over_total=mse(dc)/mse(delta) if mse(delta)>0 else None,deleted_distance=mse(d0),deleted_over_full_distance=mse(d0)/mse(delta) if mse(delta)>0 else None))
  controls={}
  for model,data in [('observed',obs),('without_root_meter',noroot),('clean',clean)]:
   if model=='observed' and entry['case']=='pengwah18' and entry['start_date']=='2024-10-14':
    controls[model]=dict(status='delegated_to_parallel_pengwah_four_structure_audit');continue
   fits={}
   for add in [False,True]:
    family=truth+[tuple(range(n))] if add else truth;fitter=QPFitter(data['train'],reference_mse=1.,bound=2.,qp_seconds=30.);pt=fitter.fit(family);R,X=matrices(pt,n);rc,xc=common(pt,n);loss={}
    for s in SPLITS:
     loss[s]=dict(observed_mse=mse(obs[s].y-prediction(R,X,obs[s])),clean_mse=mse(clean[s].y-prediction(R,X,clean[s])),fit_domain_mse=mse(data[s].y-prediction(R,X,data[s])))
    fits['with_common' if add else 'true_only']=dict(r_common=rc,x_common=xc,losses=loss,QP=fitter.records,point=pt)
   controls[model]=fits
  rows.append(dict(case=entry['case'],start_date=entry['start_date'],native_ref=entry['files']['101_native.json'],physical_sha256=digest(d/'private_physical.npz'),
    root_bus=net['root_bus'],root_degree=len(roots),root_terminal_groups=roots,physical_universal_support=tuple(range(n)) in truth,
    selected=hist,nearest_validation_competitor=runner,bank=bank,decisions=decisions,pairs=pairs,controlled_fits=controls))
  write(OUT/'progress.json',dict(completed=len(rows),total=len(index['inputs'])));print(entry['case'],entry['start_date'],'done',flush=True)
 qp_count=sum(len(c)*int('with_common' in c) for r in rows for c in r['controlled_fits'].values())
 delegated=sum(2 for r in rows for c in r['controlled_fits'].values() if 'status' in c)
 write(OUT/'RESULTS.json',dict(protocol=protocol,rows=rows));write(OUT/'complete.json',dict(inputs=len(rows),controlled_QPs=qp_count,delegated_QPs=delegated,result_sha256=digest(OUT/'RESULTS.json'),source_sha256=digest(__file__)))

def summarize():
 """Return aggregates only, with no fitted coefficient or observed array dump."""
 receipt=read(OUT/'complete.json')
 if receipt['result_sha256']!=digest(OUT/'RESULTS.json'):raise ValueError('Completed results changed')
 rows=read(OUT/'RESULTS.json')['rows'];selected=[r['bank'][r['selected']] for r in rows]
 summary=dict(inputs=len(rows),native_candidates=sum(len(r['bank']) for r in rows),
  selected_common_active=sum(b['common_active'] for b in selected),
  physical_common_supports=sum(r['physical_universal_support'] for r in rows),
  validation_selection_changes=sum(r['decisions']['validation']['index']!=r['decisions']['validation_deleted']['index'] for r in rows),
  cumulative_selection_changes=sum(r['decisions']['cumulative']['index']!=r['decisions']['cumulative_deleted']['index'] for r in rows),
  controlled_QPs=receipt['controlled_QPs'],delegated_QPs=receipt['delegated_QPs'])
 for domain in ['observed','clean']:
  gain=np.array([100*(b['losses']['new_day']['deleted_'+domain+'_mse']/b['losses']['new_day'][domain+'_mse']-1) for b in selected])
  summary['fixed_delete_new_day_'+domain]=dict(improved=int((gain < -1e-9).sum()),worsened=int((gain > 1e-9).sum()),unchanged=int((np.abs(gain)<=1e-9).sum()),minimum_percent=float(gain.min()),median_percent=float(np.median(gain)),maximum_percent=float(gain.max()))
 return summary

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--index',type=Path,default=INDEX)
 parser.add_argument('--out',type=Path,default=OUT)
 parser.add_argument('--summary-only',action='store_true')
 args=parser.parse_args();INDEX=args.index.resolve();OUT=args.out.resolve()
 if not args.summary_only:run()
 print(json.dumps(summarize(),ensure_ascii=False,indent=2))
