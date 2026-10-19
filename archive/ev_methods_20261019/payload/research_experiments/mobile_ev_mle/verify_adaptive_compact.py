"""Offline, project-independent evidence verification; no AC/fit/selector calls.

Usage: python verify_adaptive_compact.py evidence/adaptive_d_compact
The archive contains exact source bytes (optionally gzip) and only hashes for
excluded raw NPZ observations. These hashes identify the omitted archives;
this offline checker does not claim to replay their AC objective.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, gzip, hashlib, io, json, math
import numpy as np
from scipy.stats import beta

def sha(b): return hashlib.sha256(b).hexdigest()
def event(values):
    n=len(values);k=sum(bool(x) for x in values)
    return dict(count=k,denominator=n,rate=k/n if n else None,
        CP95=[0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
              1. if k==n else float(beta.ppf(.975,k+1,n-k))])
def measure(values):
    a=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return dict(denominator=len(values),available=len(a),missing=len(values)-len(a),
        mean=float(np.mean(a)) if a else None,median=float(np.median(a)) if a else None,
        total=float(sum(a)) if a else None,descriptive_available_subset=True)
def valid(bank):
    return len(bank['losses'])==len(bank['success']) and all(bank['success']) and all(
        v is not None and math.isfinite(v) and v>=0 for v in bank['losses'])
def score(row):
    t=row.get('true_index'); banks=row['records']
    complete=t is not None and all(valid(b) for b in banks)
    rel=max(b['losses'][t]-min(b['losses']) for b in banks) if complete else math.inf
    absolute=max(b['losses'][t] for b in banks) if complete else math.inf
    saved=row['score_record'];assert complete==saved['complete']
    if complete:
        assert math.isclose(rel,saved['relative_gap'],abs_tol=1e-9)
        assert math.isclose(absolute,saved['absolute_loss'],abs_tol=1e-9)
    return complete,rel,absolute
def masks(bank,thresholds):
    if not valid(bank): return [False]*len(bank['losses']),[False]*len(bank['losses'])
    low=min(bank['losses'])
    return ([v-low>thresholds['relative'] for v in bank['losses']],
            [v>thresholds['absolute'] for v in bank['losses']])
def history(row,method,thresholds):
    banks=[b for b in row['records'] if b['method']==method]
    assert banks
    rounds=[masks(b,thresholds) for b in banks]
    combined=[[a or b for a,b in zip(rel,absolute)] for rel,absolute in rounds]
    retained=[not x for x in combined[-1]]
    relative_retained=[not x for x in rounds[-1][0]]
    t=row.get('true_index');covered=t is not None
    false=covered and any(x[t] for x in combined)
    complete=all(valid(b) for b in banks)
    physical=all(b.get('operational_voltage_ok',False) for b in banks)
    sf=any(b.get('selection_status')=='selection_failure_fallback' for b in banks)
    correct=covered and complete and physical and sum(retained)==1 and retained[t]
    wrong=[j for j in range(len(retained)) if not covered or j!=t]
    eliminated=sum(not retained[j] for j in wrong)
    e=dict(false_rejection=bool(false),truth_omission=not covered,
        truth_loss_event=bool(not covered or false),unique_correct_retention=bool(correct),
        algorithm_unique_correct_retention=bool(correct and not sf),
        relative_unique_correct_retention=bool(covered and complete and physical and sum(relative_retained)==1 and relative_retained[t]),
        wrong_unique_retention=bool(sum(retained)==1 and (not covered or not retained[t])),
        unresolved=sum(retained)>1,empty_retained=sum(retained)==0,
        selection_failure=sf,fit_failure=not complete,physical_failure=not physical,
        bank_adequacy_alarm=any(valid(b) and min(b['losses'])>thresholds['absolute'] for b in banks),
        any_wrong_candidate_eliminated=eliminated>0,
        all_wrong_candidates_eliminated=bool(wrong and eliminated==len(wrong)))
    last=banks[-1]
    return dict(events=e,retained=retained,round_masks=combined,
        relative_truth_rejected=covered and any(x[0][t] for x in rounds),
        absolute_truth_rejected=covered and any(x[1][t] for x in rounds),
        wrong_candidates_eliminated=eliminated,wrong_candidate_count=len(wrong),
        evaluation_seconds=sum(b['seconds'] for b in banks) if all(b.get('seconds') is not None for b in banks) else None,
        joint_fit_AC_calls=sum(b['ac_calls'] for b in banks) if all(b.get('ac_calls') is not None for b in banks) else None,
        impedance_relative_rmse=last.get('impedance_relative_rmse'))

def verify(folder):
    folder=Path(folder);manifest=json.loads((folder/'source_manifest.json').read_text())
    blobs={}
    for item in manifest['files']:
        if not item.get('stored'): continue
        data=(folder/item['stored']).read_bytes()
        assert sha(data)==item['stored_sha256'],item['stored']
        raw=gzip.decompress(data) if item['compression']=='gzip' else data
        if item.get('projection'):
            assert sha(raw)==item['projection_sha256'] and len(raw)==item['projection_bytes'],item['source']
        else:
            assert sha(raw)==item['source_sha256'] and len(raw)==item['source_bytes'],item['source']
        blobs[item['source']]=raw
    def read(p):return json.loads(blobs[p])
    def csvrows(p):return list(csv.DictReader(io.StringIO(blobs[p].decode('utf-8-sig'))))
    def batch(prefix):
        cfg=read(prefix+'/config.json')
        paths=sorted(p for p in blobs if p.startswith(prefix+'/trial_') and p.endswith('.json'))
        rows=[read(p) for p in paths]
        assert len(rows)==cfg['count']
        assert {r['seed'] for r in rows}==set(range(cfg['start_seed'],cfg['start_seed']+cfg['count']))
        assert len({r['history_id'] for r in rows})==len(rows)
        return cfg,rows
    spec=manifest['spec'];cal_cfg,cal=batch(spec['calibration'])
    scored=[score(r) for r in cal];rank=math.ceil((len(cal)+1)*.95)
    thresholds={k:sorted(z[i] for z in scored)[rank-1] for k,i in [('relative',1),('absolute',2)]}
    result=dict(version='independent_adaptive_compact_v1',
        byte_preserving_sources_verified=sum(bool(i.get('stored')) and not i.get('projection') for i in manifest['files']),
        loss_and_timing_projections_verified=sum(bool(i.get('projection')) for i in manifest['files']),
        omitted_NPZ_hashes=sum(i.get('excluded_reason')=='raw_NPZ_not_in_compact' for i in manifest['files']),
        project_modules_imported=False,AC_or_optimizers_run=False,
        calibration=dict(denominator=len(cal),complete=sum(z[0] for z in scored),
            failed_infinity_count=sum(not z[0] for z in scored),rank=rank,thresholds=thresholds,
            joint_marginal_rank_bound=2*(len(cal)+1-rank)/(len(cal)+1)),batches={})
    for prefix in spec['evaluation']:
        cfg,rows=batch(prefix);official=read(prefix+'_audit/audit_summary.json')
        assert official['evaluation_history_count']==len(rows)
        assert not {r['history_id'] for r in rows}&{r['history_id'] for r in cal}
        assert all(math.isclose(thresholds[k],official['calibration'][k]['threshold'],abs_tol=1e-10) for k in thresholds)
        methods={};by_method={}
        for method in cfg['methods']:
            hr=[history(r,method,thresholds) for r in rows];by_method[method]=hr
            events={key:event([h['events'][key] for h in hr]) for key in hr[0]['events']}
            for key,value in events.items():
                assert value['count']==official['methods'][method]['events'][key]['count'],(prefix,method,key)
            wrong=sum(h['wrong_candidates_eliminated'] for h in hr);wrong_n=sum(h['wrong_candidate_count'] for h in hr)
            assert wrong==official['methods'][method]['wrong_candidates_eliminated_total']
            assert wrong_n==official['methods'][method]['wrong_candidate_count_total']
            selections=[r['selection']['methods'].get(method,{}) for r in rows]
            methods[method]=dict(events=events,wrong_candidates_eliminated=wrong,wrong_candidate_denominator=wrong_n,
                candidate_units_are_not_independent=True,
                measurements={key:measure([h[key] for h in hr]) for key in ['evaluation_seconds','joint_fit_AC_calls','impedance_relative_rmse']},
                whole_selection_seconds=measure([s.get('total_selection_seconds') for s in selections]))
            if len(hr[0]['round_masks'])==2:
                methods[method]['round_truth_rejection']={str(k+1):event([h['round_masks'][k][r['true_index']] for h,r in zip(hr,rows)]) for k in [0,1]}
                methods[method]['relative_truth_rejection']=event([h['relative_truth_rejected'] for h in hr])
                methods[method]['absolute_truth_rejection']=event([h['absolute_truth_rejected'] for h in hr])
        fits=csvrows(prefix+'_audit/fits.csv')
        replay=dict(denominator=len(fits),status_counts=dict(Counter(z['ac_audit_status'] for z in fits)),
            max_abs_saved_loss_difference=max(abs(float(z['ac_audit_loss_difference'])) for z in fits if z['ac_audit_loss_difference']),
            this_verifier_reran_AC=False)
        impedance=csvrows(prefix+'_impedance/impedance_history_components.csv')
        groups=defaultdict(list)
        for z in impedance:groups[(z['method'],z['round'],z['edge_class'])].append(z)
        imp={}
        for key,group in groups.items():
            assert len({z['history_id'] for z in group})==len(rows)
            imp['__'.join(key)]=dict(denominator=len(group),applicable=sum(z['applicable']=='True' for z in group),
                status_counts=dict(Counter(z['estimate_status'] for z in group)),
                R_relative_rmse=measure([z['R_relative_rmse'] or None for z in group]),
                X_relative_rmse=measure([z['X_relative_rmse'] or None for z in group]))
        metrics=csvrows(prefix+'_empirical/history_method_metrics.csv')
        for method in cfg['methods']:
            selected=[z for z in metrics if z['method']==method]
            assert len(selected)==len(rows)
            methods[method]['empirical_saved_optimization_workload']={key:measure([z[key] or None for z in selected])
                for key in ['selection_ac_calls','selection_seconds','qp_count','milp_solve_count',
                            'milp_time_limit_count','upper_milp_missing_bound_count']}
            methods[method]['empirical_selection_scope']='last round only' if any(len(h['round_masks'])>1 for h in by_method[method]) else 'single round'
        result['batches'][prefix]=dict(denominator=len(rows),requested=cfg['count'],methods=methods,
            any_method_false_rejection=event([any(by_method[m][i]['events']['false_rejection'] for m in cfg['methods']) for i in range(len(rows))]),
            whole_history_seconds=measure([r.get('seconds') for r in rows]),
            whole_selector_seconds=measure([r['selection'].get('selector_total_seconds') for r in rows]),
            whole_selector_AC_calls=measure([r['selection'].get('selector_ac_predict_calls') for r in rows]),
            empirical_last_selection_seconds={m:measure([z['selection_seconds'] or None for z in metrics if z['method']==m]) for m in cfg['methods']},
            existing_saved_objective_replay=replay,impedance=imp,
            stress=cfg.get('stress'),marginal_gate_guarantee_for_stress=False if cfg.get('stress') else True)
    def sanitize(x):
        if isinstance(x,float) and not math.isfinite(x):return None
        if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
        if isinstance(x,list):return [sanitize(v) for v in x]
        return x
    return sanitize(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--write',type=Path)
    a=p.parse_args();result=verify(a.folder)
    if a.write:a.write.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(calibration=result['calibration'],batches={k:dict(denominator=v['denominator'],
        any_method_false_rejection=v['any_method_false_rejection']) for k,v in result['batches'].items()}),indent=2))
