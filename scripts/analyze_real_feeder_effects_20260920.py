"""Replay existing cases for effect analysis; no new fitting or tuning.

Checks saved results, decomposes amplitude bias, and measures ranking margins.
Truth loss is used exclusively to evaluate error sources, never in inference.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

try:
    from scripts import real_feeder_phase_wzzt as ph
except ModuleNotFoundError:
    import real_feeder_phase_wzzt as ph


def main():
    folder=ph.base.ROOT/'theft_wzzt/outputs/theft_real65037'
    feeder=ph.base.read_feeder(); rows=[]; input_hashes={}
    history_info=[]
    for seed in (0,1):
        for balanced in (True,False):
            label='balanced' if balanced else 'unbalanced'
            hp=folder/f'v3_tree_selection/seed{seed}_{label}_history.json'
            fitted=json.loads(hp.read_text(encoding='utf-8'))
            input_hashes[hp.name]=hashlib.sha256(hp.read_bytes()).hexdigest()
            supports=tuple(frozenset(s) for s in fitted['candidates'][0]['supports'])
            weights=np.array(fitted['weights'])
            hist=ph.phase_sample(feeder,74000+seed,128,balanced)
            raw=ph.phase_loss(hist,supports,fitted)
            scale=float(np.clip(raw@(hist['p0']-hist['p'].sum(axis=1))/(raw@raw),.2,5))
            z,_=ph.features(hist,supports)
            a=np.array([[h<=s for h in supports] for s in supports],float)
            response=z.T@((weights[:,0]+.4*weights[:,1])[:,None]*a)
            history_info.append(dict(seed=seed,balanced=balanced,candidates=len(supports),
                                     meter39_singleton_weights=weights[supports.index(frozenset([2]))].tolist(),
                                     loss_calibration_scale=scale))
            for bus in (5,10):
                for condition in ('persistent','intermittent'):
                    cp=folder/f'selected_z_heldout/history{seed}_{label}_bus{bus}_{condition}.json'
                    saved=json.loads(cp.read_text(encoding='utf-8'))
                    input_hashes[cp.name]=hashlib.sha256(cp.read_bytes()).hexdigest()
                    for original in saved:
                        current=ph.phase_sample(feeder,90000+original['replicate'],64,balanced,condition,bus)
                        loss=scale*ph.phase_loss(current,supports,fitted)
                        raw_amp=current['p0']-current['p'].sum(axis=1)-loss
                        amp=np.maximum(raw_amp,0)
                        normal=ph.phase_predict(current,supports,weights)
                        residual=current['yp'][:,:,:,None]-normal[:,:,:,None]-amp[:,None,None,None]/3*response[None,:,None,:]
                        costs=np.mean(abs(residual),axis=(0,1,2));order=np.argsort(costs,kind='stable')
                        tied=np.flatnonzero(np.isclose(costs,costs.min(),rtol=0,atol=1e-9))
                        assert original['tied_supports']==[sorted(supports[i]) for i in tied]
                        mae=float(np.mean(abs(amp-current['amplitude'])))
                        assert abs(mae-original['amplitude_mae_kw'])<1e-10
                        error=amp-current['amplitude']; active=current['amplitude']>0; inactive=~active
                        balance_error=current['p0']-current['p'].sum(axis=1)-current['amplitude']-current['loss']
                        loss_underestimate=current['loss']-loss; clipping=amp-raw_amp
                        identity=np.max(abs(error-loss_underestimate-balance_error-clipping))
                        assert identity<1e-10
                        rows.append(dict(**original,mean_amp_error_kw=float(error.mean()),
                                         active_mae_kw=float(np.mean(abs(error[active]))),
                                         active_bias_kw=float(error[active].mean()),
                                         inactive_mean_kw=float(amp[inactive].mean()) if np.any(inactive) else None,
                                         mean_loss_underestimate_kw=float(loss_underestimate.mean()),
                                         mean_meter_balance_error_kw=float(balance_error.mean()),
                                         mean_clipping_bias_kw=float(clipping.mean()),
                                         rank1_rank2_gap_equivalent_v=float((costs[order[1]]-costs[order[0]])*2000/460),
                                         top3_supports=[sorted(supports[i]) for i in order[:3]],
                                         decomposition_error=float(identity)))
    groups=[]
    for balanced in (True,False):
        for bus in (5,10):
            for condition in ('persistent','intermittent'):
                group=[r for r in rows if (r['balanced'],r['source_bus'],r['condition'])==(balanced,bus,condition)]
                record=dict(balanced=balanced,bus=bus,condition=condition,n=len(group))
                for metric in ('amplitude_mae_kw','mean_amp_error_kw','active_mae_kw','active_bias_kw','inactive_mean_kw',
                               'mean_loss_underestimate_kw','mean_meter_balance_error_kw','mean_clipping_bias_kw'):
                    values=[r[metric] for r in group if r[metric] is not None]
                    record[metric]=float(np.mean(values)) if values else None
                for label,key in (('unique','unique_correct'),('minset','truth_in_minimum_set'),('top3','truth_in_top3')):
                    record[label]=sum(r[key] for r in group)
                unique_margins=[r['rank1_rank2_gap_equivalent_v'] for r in group if r['unique_correct']]
                record['unique_correct_median_rank_margin_v']=float(np.median(unique_margins)) if unique_margins else None
                groups.append(record)
    counts={}
    for path in (folder/'v3_tree_selection').glob('*.checkpoint.json'):
        input_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        item=json.loads(path.read_text())
        for kind in ('topology_only','theft'):
            key='H0_current' if kind=='topology_only' else ('H1_null' if '_clean_' in path.name else 'H1_theft')
            count=counts.setdefault(key,dict(total=0,optimal=0,incomplete=0))
            count['total']+=1;count[item[kind]['status']]+=1
    historical=list((folder/'v3_tree_selection').glob('*_history.json'))
    counts['history']=dict(total=len(historical),optimal=sum(json.loads(p.read_text())['status']=='optimal' for p in historical))
    assert len(rows)==320
    implementation_paths=[Path(__file__).resolve(),Path(ph.__file__).resolve(),Path(ph.base.__file__).resolve(),
                          ph.base.ROOT/'data_external/lvnetworkdataset_20260920/manifest.json']
    implementation_hashes={str(p.relative_to(ph.base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in implementation_paths}
    summary=dict(protocol='Post-hoc deterministic replay of the existing 320 evaluations; no model change, no additional independent samples',
                 implementation_hashes=implementation_hashes,
                 validation=dict(replayed_evaluations=len(rows),all_saved_minimum_sets_match=True,
                                 all_saved_amplitude_mae_match=True,
                                 truth_loss_usage='Evaluation only, never used for prediction or selection'),
                 source_hashes=input_hashes,history_models=history_info,groups=groups,solver_breakdown=counts,
                 max_error_identity_residual=max(r['decomposition_error'] for r in rows),rows=rows)
    target=folder/'effect_assessment_20260920.json'
    target.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(groups=groups,solver_breakdown=counts,history=history_info),indent=2))


if __name__=='__main__':main()
