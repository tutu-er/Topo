"""Rebuild only the two fixed H contexts to serialize AC replay inputs.

No selector rerun. Require exact pair fingerprint match with saved pilot matrices.
"""
from pathlib import Path
from types import SimpleNamespace
import gzip,hashlib,json,sys,time
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import honly_search_pilot as pilot

def load(name):return json.loads(name.read_text(encoding='utf-8'))
def restore(snapshot):
    from research_experiments.mobile_ev_mle.likelihood import Observations,MeasurementModel
    context={'history':Observations(**snapshot['history']), 'model':MeasurementModel(**snapshot['measurement_model']),
             'config':snapshot['config'],'channels':tuple(snapshot['channels']),'scenarios':snapshot['scenarios'],'centres':{}}
    for scenario in context['scenarios']:
        for key in ['p0','q0','root_v']:scenario[key]=np.asarray(scenario[key])
    for name,rows in snapshot['centres'].items():
        candidate=pilot.RadialCandidate.from_supports(name,3,pilot.SUPPORTS[name])
        context['centres'][name]=[{'candidate':candidate,'beta':np.asarray(row['beta']),
                'history_state':np.asarray(row['history_state']),
                'fit':SimpleNamespace(**{k:np.asarray(v) for k,v in row['fit'].items()})} for row in rows]
    return context

results=[]
for ident,folder,seed in pilot.CASES:
    case=ROOT/f'H_{ident}'
    history,_=pilot.read_case(ident,folder)
    candidates=[pilot.RadialCandidate.from_supports(name,3,support) for name,support in pilot.SUPPORTS.items()]
    began=time.perf_counter()
    context=pilot.build_history_context(history,candidates,pilot.measurement_model(3),horizon=8,max_centers=2,max_attempts_per_candidate=3,
        multistart=2,max_nfev=300,line_trust_fraction=.30,source_delta_bound=.05,center_loss_slack=12.,seed=seed,
        forecast_support=[dict(name='mean',baseline='mean',source_baseline='mean')])
    constant=np.zeros((8,3));constant[:,0]=3
    raw=pilot.build_frozen_pairs(context,constant,source_mode='local_refit')
    pairs=[pilot.old.FrozenPair(**r) for r in raw]
    manifest=load(case/'pair_manifest.json')
    assert {p.name:p.fingerprint() for p in pairs}==manifest['fingerprints']
    snapshot={'history_id':ident,'seed':seed,'reconstructed_pair_fingerprints_exact_match':True,
        'history':{k:getattr(history,k) for k in pilot.FIELDS},'measurement_model':vars(context['model']),
        'config':context['config'],'channels':context['channels'],'scenarios':context['scenarios'],
        'centres':{name:[{'beta':r['beta'],'history_state':r['history_state'],
               'fit':{k:getattr(r['fit'],k) for k in ['r','x','p','q','source_p','source_q']}} for r in rows] for name,rows in context['centres'].items()},
        'note':'latent states here are fitted from H, not simulator truth; no future measurements'}
    pilot.packed(case/'AC_replay_context.json.gz',snapshot)
    restored=restore(json.loads(gzip.decompress((case/'AC_replay_context.json.gz').read_bytes())))
    compared=0;maxcost=0.;maxerror=0.;feasibility_disagreements=0
    for label in ['original','shared']:
        qp=load(case/label/'selected_point_qp.json');audit=load(case/label/'selected_audit.json')
        previous=load(case/label/'full_AC_rollback.json');u=np.asarray(audit['raw_controls']['u_kw'])
        for p,z,old_record in zip(raw,qp['records'],previous):
            replay=pilot.evaluate_frozen_pair_actual(restored,p,u,np.asarray(z['z']))
            maxcost=max(maxcost,abs(replay['actual_soft_cost']-old_record['actual_soft_cost']),abs(replay['affine_soft_cost']-old_record['affine_soft_cost']))
            maxerror=max(maxerror,abs(replay['joint_taylor_error_norm']-old_record['joint_taylor_error_norm']))
            feasibility_disagreements+=int(bool(replay['physically_feasible'])!=bool(old_record['physically_feasible']))
            compared+=1
    assert maxcost<1e-8 and maxerror<1e-10 and feasibility_disagreements==0
    result={'history_id':ident,'pair_fingerprint_exact_match':True,'context_reconstruction_seconds':time.perf_counter()-began,
            'selectors_rerun':False,'saved_context_offline_AC_record_comparisons':compared,
            'max_actual_or_affine_cost_difference':maxcost,'max_taylor_norm_difference':maxerror,
            'feasibility_disagreements':feasibility_disagreements,'context_archive_sha256':pilot.sha(case/'AC_replay_context.json.gz')}
    pilot.dump(case/'AC_replay_context_validation.json',result);results.append(result)
pilot.dump(ROOT/'AC_replay_context_validation.json',{'scope':'two existing fixed H contexts reconstituted for saved evidence replay; no selector rerun','histories':results})
print(json.dumps({'status':'PASS','records_replayed':sum(r['saved_context_offline_AC_record_comparisons'] for r in results),
                  'selector_reruns':False,'histories':results},ensure_ascii=False,indent=2))
