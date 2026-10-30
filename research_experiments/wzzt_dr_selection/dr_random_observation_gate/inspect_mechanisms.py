"""Read saved fits only: input rank, original thresholds, and parameter mixing."""
import json,pickle,os
from pathlib import Path
import numpy as np

BASE=Path(os.environ.get('WZZT_RANDOM_GATE_SCRATCH',r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a'))
GEN=BASE/'wzzt_dr_random_observation_generation_v2'
RUN=BASE/'wzzt_dr_random_observation_execution_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def packet(p):return pickle.loads(Path(p).read_bytes())
cfg=read(GEN/'config.json');truth=read(GEN/'evaluator_truth.json');cases={r['id']:r for r in truth['cases']};snap={r['id']:r for r in cfg['snapshots']}
hist={h['id']:packet(h['path']) for h in cfg['histories']};rows=[]
for ev in truth['evaluations']:
    s=snap[ev['snapshot_id']];d=read(RUN/'records'/(s['id']+'.json'));cur=read(d['current_state_path']);fut=packet(s['path'])
    p=np.concatenate([x['P_terminal'].to_numpy() for x in hist[s['history_id']]['training']+fut['training']])
    q=np.concatenate([x['Q_terminal'].to_numpy() for x in hist[s['history_id']]['training']+fut['training']])
    case=cases[ev['case_id']];t=case['truth'];K=np.asarray(t['K']);front=cur['fronts']['raw'];Kfit=np.asarray(front['R'])+.35*np.asarray(front['X'])
    m=d['methods']['current_pool_wzzt'];raw=m['selected']['selected_uncontracted'];final=m['selected']
    rows.append(dict(case_id=ev['case_id'],policy=ev['policy'],noise=case['noise_id'],b=t['b'],c=t['c'],pair=t['pair'],
        rank_P=int(np.linalg.matrix_rank(p,tol=1e-10)),rank_PQ=int(np.linalg.matrix_rank(np.c_[p,q],tol=1e-10)),
        P_singular_values=np.linalg.svd(p,compute_uv=False).tolist(),
        dense_composite_kernel_max_error=float(np.abs(Kfit-K).max()),
        raw_supports=raw['supports'],final_supports=final['supports'],calibration=d['applied_calibration'],
        W_origin=final.get('selected_origin'),
        scans=[dict(path=name,reason=path['reason'],steps=[dict(current=z['current'],gain=z['gain'],threshold=1e-10+.001*z['current'],
            accepted=z['gain']>1e-10+.001*z['current']) for z in path['decisions']]) for name,path in m['paths'].items()],
        full8_points=[dict(supports=x['supports'],validation_mae=x['validation_mae'],train_mae=x['objective']) for x in d['methods']['all_legal_family_reference']['points']],
        raw_train=raw['objective'],raw_validation=raw['validation_mae'],final_validation=final['validation_mae']))
out=dict(rows=rows,source='saved observations and fitted matrices only; no new solve or policy change',new_solver_calls=0)
(RUN/'mechanism_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
for r in rows:
    if r['c']==.6 and r['b']==.4:
        print(json.dumps({k:r[k] for k in ('case_id','policy','noise','pair','rank_P','rank_PQ','dense_composite_kernel_max_error','raw_supports','final_supports','W_origin')}))
