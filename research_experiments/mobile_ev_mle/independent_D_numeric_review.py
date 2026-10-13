"""Read existing evidence; no project imports, AC calls, refits, or selectors.

The independent calculations below use JSON losses and scipy only for
Clopper-Pearson intervals.  Outputs go only to the prescribed review folder.
"""
import hashlib
import csv
import json
import math
from pathlib import Path
from collections import Counter
from scipy.stats import beta
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo')
BASE = ROOT / 'research_experiments/mobile_ev_mle'
D = BASE / 'outputs/adaptive_rolling'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def event(n, k):
    return dict(count=k, denominator=n, rate=k/n if n else None,
                CP_two_sided_95=[float(beta.ppf(.025, k, n-k+1)) if k else 0.,
                                 float(beta.ppf(.975, k+1, n-k)) if k<n else 1.])


def measurement(values):
    kept=[v for v in values if v is not None and math.isfinite(v)]
    return dict(history_count=len(values),available_count=len(kept),missing_count=len(values)-len(kept),
                mean=sum(kept)/len(kept) if kept else None,descriptive_available_subset=True)


def valid(rec):
    return (len(rec['losses']) == 4 and len(rec['success']) == 4
            and all(rec['success'])
            and all(v is not None and math.isfinite(v) and v >= 0 for v in rec['losses']))


def load(folder, expected):
    cfg = read(folder/'config.json')
    rows = [read(p) for p in sorted(folder.glob('trial_[0-9]*.json'))]
    ids = [str(r['history_id']) for r in rows]
    assert len(set(ids)) == len(ids)
    assert cfg['count'] == expected
    assert len(rows) == expected, 'incomplete batch; no inferential result written'
    assert set(ids) == {str(cfg['start_seed']+i) for i in range(expected)}
    for row in rows:
        assert row['independent_history_unit']
        assert not row['selection_uses_truth'] and not row['selection_uses_future_noise']
        assert row['candidate_names'] == ['star','clade_01','clade_02','clade_12']
        assert row['true_index']==row['truth_topology']
        assert [rec['round'] for rec in row['records']] == [1,2]
        for key in ('policy_id','population_id','source_sha256'):
            assert row[key] == cfg[key]
    return cfg, rows


def batch_content_manifest(folder):
    files=sorted(folder.glob('trial_[0-9]*.json'))+sorted(folder.glob('trial_[0-9]*_observations.npz'))
    entries=[(p.name,sha(p)) for p in files]
    return dict(file_count=len(files),json_history_count=sum(p.suffix=='.json' for p in files),
                archive_count=sum(p.suffix=='.npz' for p in files),
                sorted_filename_and_sha256_manifest_digest=hashlib.sha256(json.dumps(entries,separators=(',',':')).encode()).hexdigest())


def scores(rows):
    result = []
    for row in rows:
        complete = row.get('true_index') is not None and all(valid(rec) for rec in row['records'])
        truth = row.get('true_index')
        rel = max(rec['losses'][truth]-min(rec['losses']) for rec in row['records']) if complete else math.inf
        absolute = max(rec['losses'][truth] for rec in row['records']) if complete else math.inf
        saved = row['score_record']
        assert complete == saved['complete']
        if complete:
            assert abs(rel-saved['relative_gap']) < 1e-10
            assert abs(absolute-saved['absolute_loss']) < 1e-10
        else:
            assert saved['relative_gap'] is None and saved['absolute_loss'] is None
        result.append(dict(history_id=row['history_id'], complete=complete,
                           relative=rel, absolute=absolute))
    return result


def audit_saved_trajectory(folder, rows):
    count=0
    for row in rows:
        if row.get('execution_failure'):
            continue
        records=row['records']
        assert [r['observed_history_points'] for r in records]==[14,20]
        assert [r['executed_slots'] for r in records]==[2,6]
        assert [r['total_executed_slots'] for r in records]==[2,8]
        selections=row['selections']
        assert [s['ev_config']['n_slots'] for s in selections]==[8,6]
        first=records[0]['executed']; second=records[1]['executed']
        assert abs(first['grid_energy_kwh']+second['grid_energy_kwh']-6)<2e-6
        assert abs(selections[1]['ev_config']['net_grid_energy_kwh']-(6-first['grid_energy_kwh']))<2e-6
        assert abs(selections[1]['ev_config']['initial_soc']*60-first['energy_kwh'][-1])<2e-6
        assert selections[1]['ev_config']['initial_site']==first['next_site']
        for rec in records:
            assert np.allclose(np.asarray(rec['action']['p']),np.asarray(rec['executed']['u_kw'])/20,rtol=0,atol=1e-12)
        with np.load(folder/f"trial_{row['seed']}_observations.npz",allow_pickle=False) as archive:
            for channel in ['p','q','v','master_p','master_q','root_v']:
                H=archive['history_'+channel]
                Y1=archive['round_1_observed_'+channel]
                Y2=archive['round_2_observed_'+channel]
                H1=archive['round_1_joint_'+channel]
                H2=archive['round_2_joint_'+channel]
                assert np.array_equal(H1,np.concatenate([H,Y1]))
                assert np.array_equal(H2,np.concatenate([H1,Y2]))
            known=archive['known_actual_ev_kw']
            u=np.vstack([first['u_kw'],second['u_kw']])
            assert np.array_equal(known[:12],np.zeros((12,3)))
            assert np.array_equal(known[12:],u)
            assert np.array_equal(archive['realized_u_kw'],u)
            for rec in records:
                for fit in rec['fits']:
                    for channel in ['p','q','source_p','source_q']:
                        assert archive[f"round_{rec['round']}__{fit['name']}__fit_{channel}"].shape[0]==rec['observed_history_points']
        count+=1
    return dict(checked_complete_traces=count,all_checks_passed=True,
                tests=['12/14/20 meter lengths','2+6 executed slots','joint meter exact concatenation',
                       'known EV zero initially and exact executed prefixes','kW/20 pu mapping',
                       '6 kWh total and residual second task','SOC/site carry','saved fit state lengths'],
                AC_objective_not_recomputed=True)


def summarize_gate(rows, thresholds):
    events, round_events, histories = Counter(), {1:Counter(),2:Counter()}, []
    for row in rows:
        true=row['true_index']; rounds=[]
        physical=all(r.get('operational_voltage_ok',False) for r in row['records'])
        selection_failure=any(r.get('selection_status')=='selection_failure_fallback' for r in row['records'])
        for rec in row['records']:
            good=valid(rec)
            rel=[v-min(rec['losses']) > thresholds['relative'] for v in rec['losses']] if good else [False]*4
            absolute=[v > thresholds['absolute'] for v in rec['losses']] if good else [False]*4
            combined=[a or b for a,b in zip(rel,absolute)]
            rounds.append(dict(round=rec['round'], relative_reject=rel, absolute_reject=absolute,
                               combined_reject=combined, complete=good))
            for rule,mask in [('relative',rel),('absolute',absolute),('combined',combined)]:
                round_events[rec['round']]['false_rejection_'+rule]+=mask[true]
        any_rel=any(r['relative_reject'][true] for r in rounds)
        any_abs=any(r['absolute_reject'][true] for r in rounds)
        any_combined=any(r['combined_reject'][true] for r in rounds)
        final=[not v for v in rounds[-1]['combined_reject']]
        accumulated=[not any(r['combined_reject'][j] for r in rounds) for j in range(4)]
        complete=all(r['complete'] for r in rounds)
        final_unique=complete and physical and sum(final)==1 and final[true]
        cumulative_unique=complete and physical and sum(accumulated)==1 and accumulated[true]
        final_wrong_eliminated=sum(not final[j] for j in range(4) if j!=true)
        cumulative_wrong_eliminated=sum(not accumulated[j] for j in range(4) if j!=true)
        values=dict(false_rejection_relative=any_rel,false_rejection_absolute=any_abs,
                    false_rejection_combined=any_combined,
                    true_rejected_both_rounds=all(r['combined_reject'][true] for r in rounds),
                    true_rejected_only_round1=rounds[0]['combined_reject'][true] and not rounds[1]['combined_reject'][true],
                    true_rejected_only_round2=rounds[1]['combined_reject'][true] and not rounds[0]['combined_reject'][true],
                    final_unique_correct_retention=final_unique,
                    final_algorithm_unique_correct_retention=final_unique and not selection_failure,
                    final_unique_correct_but_earlier_truth_rejected=final_unique and rounds[0]['combined_reject'][true],
                    cumulative_unique_correct_retention=cumulative_unique,
                    final_unresolved=sum(final)>1,final_empty=sum(final)==0,
                    cumulative_unresolved=sum(accumulated)>1,cumulative_empty=sum(accumulated)==0,
                    final_wrong_unique=sum(final)==1 and not final[true],
                    fit_failure=not complete,selection_failure=selection_failure,physical_failure=not physical)
        events.update({k:int(v) for k,v in values.items()})
        events['final_wrong_candidates_eliminated']+=final_wrong_eliminated
        events['cumulative_wrong_candidates_eliminated']+=cumulative_wrong_eliminated
        histories.append(dict(history_id=row['history_id'], rounds=rounds, events=values,
                               final_retained=final,cumulative_retained=accumulated))
    n=len(rows)
    return dict(history_events={k:event(n,v) for k,v in events.items() if 'candidates_eliminated' not in k},
                round_events={str(r):{k:event(n,v) for k,v in e.items()} for r,e in round_events.items()},
                wrong_candidate_counts={k:v for k,v in events.items() if 'candidates_eliminated' in k},
                candidate_counts_are_not_independent=True, histories=histories)


cal_cfg,cal_rows=load(D/'calibration_v1',64)
frozen=read(D/'frozen_D_execution.json')
assert cal_cfg==frozen['calibration']
assert frozen['alpha_total']==.1 and frozen['relative_fraction']==.5
assert not frozen['conditional_alpha_spending_claim'] and not frozen['one_round_gates_reused']
cal_scores=scores(cal_rows)
max_attainment=Counter()
for row in cal_rows:
    t=row['true_index']
    if all(valid(r) for r in row['records']):
        for key,values in [('relative',[r['losses'][t]-min(r['losses']) for r in row['records']]),('absolute',[r['losses'][t] for r in row['records']])]:
            max_attainment[key+('_round1_strict_max' if values[0]>values[1] else '_round2_strict_max' if values[1]>values[0] else '_tie')]+=1
rank=math.ceil(65*.95)
thresholds={key:sorted(s[key] for s in cal_scores)[rank-1] for key in ['relative','absolute']}
hash_checks={}
for name,expected in cal_cfg['source_sha256'].items():
    path=BASE/(name.split('::',1)[1] if name.startswith('loaded::') else name)
    hash_checks[name]=dict(expected=expected,actual=sha(path),match=sha(path)==expected)

result=dict(review_version='independent_D_numeric_review_v1',
            review_did_not_import_project_code=True,review_did_not_run_AC_or_optimizers=True,
            frozen_manifest_sha256=sha(D/'frozen_D_execution.json'),
            calibration_config_sha256=sha(D/'calibration_v1/config.json'),
            calibration_count=64,calibration_complete=sum(s['complete'] for s in cal_scores),
            calibration_failure_infinity_count=sum(not s['complete'] for s in cal_scores),
            thresholds=thresholds,rank=rank,alpha_rule=.05,alpha_total=.1,
            single_rule_rank_tail_bound=3/65,combined_rank_tail_bound=6/65,
            rank_resolution=1/65,failures_tolerated_for_finite_threshold=2,
            population_tail_95th_percentile_over_calibration_samples_if_continuous=float(beta.ppf(.95,3,62)),
            policy_id=cal_cfg['policy_id'],population_id=cal_cfg['population_id'],
            source_hash_checks=hash_checks,conditional_alpha_spending_supported=False,
            guaranteed_scope='joint marginal over calibration and new complete shadow trajectory under exchangeability; not conditional on fixed calibration sample, history, topology or continuous parameters',
            calibration_scores=cal_scores)
result['calibration_max_score_attainment']=dict(max_attainment)
result['calibration_strata']=dict(Counter('star' if r['truth_topology']==0 else r['case']+'_cherry' for r in cal_rows))
result['calibration_threshold_rank_details']={key:dict(
    threshold_history_ids=[s['history_id'] for s in cal_scores if s[key]==thresholds[key]],
    strict_above_count=sum(s[key]>thresholds[key] for s in cal_scores),
    largest_five=[dict(history_id=s['history_id'],score=s[key]) for s in sorted(cal_scores,key=lambda s:s[key],reverse=True)[:5]]) for key in thresholds}
result['calibration_archive_and_carry_checks']=audit_saved_trajectory(D/'calibration_v1',cal_rows)
result['calibration_input_manifest']=batch_content_manifest(D/'calibration_v1')

C=read(BASE/'outputs/adaptive_milp/evaluation_v3_audit/audit_summary.json')
Cg=read(BASE/'outputs/adaptive_milp/evaluation_v3_audit/history_gates.json')
result['C_background']=dict(calibration_count=128,evaluation_count=C['evaluation_history_count'],
    thresholds={k:C['calibration'][k]['threshold'] for k in ['relative','absolute']},
    combined_rank_tail_bound=C['calibration']['combined_rank_tail_bound'],
    calibration_failure_count=C['calibration']['compute_failure_count'],
    any_method_false_rejection=event(len(Cg),sum(any(m['false_rejection'] for m in row['methods'].values()) for row in Cg)),
    methods={m:{k:v['count'] for k,v in x['events'].items() if k in ['false_rejection','algorithm_unique_correct_retention','unresolved','fit_failure','selection_failure']} for m,x in C['methods'].items()},
    direct_C_D_performance_comparison_valid=False,
    reason='different independent histories, policy, rounds, thresholds and method bank; no matched rolling baseline')

progress=read(D/'evaluation_v1/progress.json')
result['evaluation_progress_at_read']={k:progress.get(k) for k in ['completed','state','all_requested_histories_attempted']}
if progress.get('state')=='count_complete' and progress.get('completed')==96:
    ev_cfg,ev_rows=load(D/'evaluation_v1',96)
    assert ev_cfg==frozen['evaluation']
    assert not {r['history_id'] for r in cal_rows}&{r['history_id'] for r in ev_rows}
    for key in ['policy_id','population_id','source_sha256','methods']:
        assert cal_cfg[key]==ev_cfg[key]
    result['evaluation_config_sha256']=sha(D/'evaluation_v1/config.json')
    result['evaluation_input_manifest']=batch_content_manifest(D/'evaluation_v1')
    result['evaluation_scores']=scores(ev_rows)
    result['evaluation_archive_and_carry_checks']=audit_saved_trajectory(D/'evaluation_v1',ev_rows)
    result['evaluation']=summarize_gate(ev_rows,thresholds)
    result['evaluation']['strata']={label:summarize_gate([r for r in ev_rows if ('star' if r['truth_topology']==0 else r['case']+'_cherry')==label],thresholds) for label in ['star','strong_cherry','weak_cherry']}
    result['whole_trajectory_complexity']=dict(
        history_seconds=measurement([r.get('seconds') for r in ev_rows]),
        selection_seconds=measurement([r['selection'].get('selector_total_seconds') for r in ev_rows]),
        selection_AC_calls=measurement([r['selection'].get('selector_ac_predict_calls') for r in ev_rows]),
        joint_fit_AC_calls=measurement([sum(v['ac_calls'] for v in r['records']) if all(v.get('ac_calls') is not None for v in r['records']) else None for r in ev_rows]),
        whole_trial_AC_calls=measurement([r.get('ac_call_metrics',{}).get('whole_trial_ac_calls') for r in ev_rows]),
        per_round_selection_AC_calls={str(k):measurement([r['selection']['methods']['rolling_refit_milp'].get('rounds',{}).get(str(k),{}).get('selection_ac_predict_calls') for r in ev_rows]) for k in [1,2]})
    audit_path=D/'evaluation_v1_audit/audit_summary.json'
    if audit_path.exists():
        official=read(audit_path)
        assert official['evaluation_history_count']==96
        assert all(abs(official['calibration'][k]['threshold']-thresholds[k])<1e-10 for k in thresholds)
        mapping={'false_rejection':'false_rejection_combined','unique_correct_retention':'final_unique_correct_retention','algorithm_unique_correct_retention':'final_algorithm_unique_correct_retention','unresolved':'final_unresolved','fit_failure':'fit_failure','selection_failure':'selection_failure','physical_failure':'physical_failure'}
        result['official_count_matches']={a:official['methods']['rolling_refit_milp']['events'][a]['count']==result['evaluation']['history_events'][b]['count'] for a,b in mapping.items()}
        result['official_audit_sha256']=sha(audit_path)
        fits_path=D/'evaluation_v1_audit/fits.csv'
        if fits_path.exists():
            with fits_path.open(encoding='utf-8-sig') as f:
                saved_replay=list(csv.DictReader(f))
            result['existing_postprocess_AC_objective_replay_evidence']=dict(
                source_sha256=sha(fits_path),saved_fit_count=len(saved_replay),
                status_counts=dict(Counter(r['ac_audit_status'] for r in saved_replay)),
                maximum_absolute_loss_difference=max(abs(float(r['ac_audit_loss_difference'])) for r in saved_replay),
                independent_reviewer_reran_AC=False)

def sanitize(x):
    if isinstance(x,float) and not math.isfinite(x): return None
    if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
    if isinstance(x,list):return [sanitize(v) for v in x]
    return x

(HERE/'independent_D_numeric_review.json').write_text(json.dumps(sanitize(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:result.get(k) for k in ['calibration_count','thresholds','rank','combined_rank_tail_bound','evaluation_progress_at_read']},indent=2))
if 'evaluation' in result:
    print(json.dumps(result['evaluation']['history_events'],indent=2))
