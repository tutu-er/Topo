"""Verify saved matrices, witnesses, primal/tangent bounds and snapshot rows. No solver."""
from pathlib import Path
import gzip,hashlib,json,math
import numpy as np

root=Path(__file__).resolve().parent
def js(name):return json.loads((root/name).read_text(encoding='utf-8'))
manifest=js('pair_manifest.json');summary=js('comparison_summary.json')
assert hashlib.sha256((root/'reconstructed_pairs_and_witnesses.npz').read_bytes()).hexdigest()==manifest['archive_sha256']
before=js('source_hashes_before.json');after=js('source_hashes_after.json')
assert before['frozen_core_sha256']==after['frozen_core_sha256'] and after['unchanged']
assert before['new_module_sha256']==after['new_module_sha256']
with np.load(root/'reconstructed_pairs_and_witnesses.npz',allow_pickle=False) as archive:
    pairs=[{key:archive[f'pair_{i}_{key}'].copy() for key in ['c','B','A','F','lower','upper','scale','witness_0']} for i in range(24)]
max_primal_difference=0.;max_tangent_difference=0.;point_lb=[]
for i,(name,p) in enumerate(zip(manifest['names'],pairs)):
    h=hashlib.sha256(name.encode())
    for key in ['c','B','A','F','lower','upper','scale']:
        a=p[key];h.update(str(a.shape).encode());h.update(np.ascontiguousarray(a).tobytes())
    assert h.hexdigest()==manifest['fingerprints'][name]
    for row in p['B']:
        assert len(np.unique(np.flatnonzero(row!=0.0)//3))<=1
    assert np.all(p['witness_0']>=p['lower']) and np.all(p['witness_0']<=p['upper'])
for filename in ['constant_qp.json','none_qp.json','shared_selected_point_qp.json']:
    x=js(filename);assert x['complete'] and len(x['pair_results'])==24
    u=np.asarray(x['u_kw']).ravel();lows=[];ups=[]
    for p,r in zip(pairs,x['pair_results']):
        z=np.asarray(r['z']);assert np.all(z>=p['lower']) and np.all(z<=p['upper'])
        C=np.vstack([p['A'],p['F']]);d=np.r_[p['c']+p['B']@u,np.zeros(len(p['F']))]
        residual=d+C@z;primal=float(residual@residual);gradient=2*C.T@residual
        correction=np.minimum(gradient*(p['lower']-z),gradient*(p['upper']-z))
        tangent=max(0,primal+float(np.sum(correction))-r['roundoff_guard'])
        max_primal_difference=max(max_primal_difference,abs(primal-r['q_upper']))
        max_tangent_difference=max(max_tangent_difference,abs(tangent-r['q_lower']))
        assert abs(primal-r['q_upper'])<=1e-8*max(1,abs(primal))
        assert abs(tangent-r['q_lower'])<=1e-8*max(1,abs(tangent))
        assert r['numeric_gap']==r['q_upper']-r['q_lower']
        lows.append(r['q_lower']);ups.append(r['q_upper'])
    assert min(lows)==x['q_lower'] and min(ups)==x['q_upper'];point_lb.append(x['q_lower'])
snapshots={}
for label in ['old','shared']:
    data=gzip.decompress((root/(label+'_model_snapshot.json.gz')).read_bytes())
    assert hashlib.sha256(data).hexdigest()==summary[label]['snapshot_sha256']
    snapshots[label]=json.loads(data)
    assert len(snapshots[label]['lower'])==summary[label]['model_variables']
    assert len(snapshots[label]['rows'])==summary[label]['model_rows']
assert summary['old']['global_best_bound'] is None and summary['old']['incumbent'] is None
best_bound=summary['shared']['global_best_bound'];assert best_bound>=max(point_lb)-1e-6
with np.load(root/'master_incumbents.npz',allow_pickle=False) as archive:x=archive['shared_master_x']
snapshot=snapshots['shared'];violations=[]
for v,lo,hi in zip(x,snapshot['lower'],snapshot['upper']):violations.extend([float(lo)-v,v-float(hi)])
for row,lo,hi in zip(snapshot['rows'],snapshot['row_lower'],snapshot['row_upper']):
    value=math.fsum(float(c)*x[int(i)] for i,c in row.items());violations.extend([float(lo)-value,value-float(hi)])
max_row_violation=max(0,max(violations));assert max_row_violation<=1e-6
score=math.fsum(float(c)*x[int(i)] for i,c in snapshot['objective'].items())*snapshot['metadata']['scale']
assert abs(score-summary['shared']['incumbent'])<=1e-8
knots=snapshot['metadata']['knots'];assert all(0 in k for k in knots)
result={'status':'PASS','optimizer_called':False,'actual_H_pairs':24,'frozen_hashes_unchanged':15,
        'matrix_and_snapshot_sha256_verified':True,'exact_time_block_support_verified':True,
        'saved_point_QP_records_recomputed':72,'max_primal_replay_difference':max_primal_difference,
        'max_tangent_replay_difference':max_tangent_difference,'max_shared_master_row_violation':max_row_violation,
        'global_proxy_best_bound':best_bound,'master_incumbent_recomputed':score,
        'old_missing_incumbent_and_bound_preserved':True}
print(json.dumps(result,ensure_ascii=False,indent=2))
