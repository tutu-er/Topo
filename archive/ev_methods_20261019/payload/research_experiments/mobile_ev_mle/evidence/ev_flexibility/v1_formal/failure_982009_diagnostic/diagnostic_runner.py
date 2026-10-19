"""Independent continuation diagnostic for the preserved982009 max200 failure.

Only observed saved H+N data enter the six local fits. Two declared starts per
wrong candidate and max500 are diagnostic budgets; original records/threshold
and the max200 scientific policy remain immutable.
"""
from pathlib import Path
import json
from time import perf_counter
from research_experiments.mobile_ev_mle import likelihood
from research_experiments.mobile_ev_mle.information_fit_recovery import read_json,write_json,sha,exact_candidate,independent_replay,run_fit
from research_experiments.mobile_ev_mle.information_replay import _obs,_meter
from research_experiments.mobile_ev_mle.ev_flexibility_replay import source_audit
ROOT=Path.cwd();HERE=ROOT/'research_experiments/mobile_ev_mle';BASE=HERE/'evidence/ev_flexibility/v1_formal';OUT=BASE/'failure_982009_diagnostic'
if OUT.exists():raise ValueError('diagnostic already exists; preserve and stop')
paths=[BASE/'freeze.json',BASE/'science_source_snapshot.json.gz',BASE/'calibration_summary.json']+sorted((BASE/'calibration').glob('trial_*.json.gz'))
before={str(p.relative_to(BASE)):sha(p) for p in paths}
freeze=read_json(BASE/'freeze.json');sources=source_audit(BASE,freeze)
trial=read_json(BASE/'calibration/trial_982009.json.gz');record=next(r for r in trial['records'] if r['method']=='v2g_mobile')
failed=[r for r in record['fits'] if not r['success']]
assert record['status']=='joint_fit_failure' and {r['name'] for r in failed}=={'clade_01','clade_02','star'}
history=_obs(trial['history_observations']);future=_obs(record['observations']);joint=likelihood.concat_observations(history,future);meter=_meter(trial['measurement_models']['joint'])
protocol=dict(schema='v1_formal_max200_continuation_diagnostic_v1',history_id='982009',method='v2g_mobile',original_method_status=record['status'],original_calibration_failure_count=1,original_threshold_is_infinite=True,original_evaluation_H=0,requested_evaluation_H=48,diagnostic_only=True,original_records_or_threshold_reclassified=False,target_candidates=[r['name'] for r in failed],modes=['warm_history','continue_original_return'],options=dict(max_nfev=500,multistart=1,tol=1e-10,linear_solver='exact'),max_total_fit_calls=6,max_total_solver_starts=6,observed_data_only=True,new_selector_or_design_calls=0,immutable_before_sha256=before,source_before=sources,diagnostic_script_sha256=sha(Path(__file__)))
write_json(OUT/'protocol.json',protocol)
write_json(OUT/'inputs.json.gz',dict(history=trial['history_observations'],future=record['observations'],joint_meter=trial['measurement_models']['joint'],original_fits=record['fits'],historical_fits=trial['historical_fits'],action=record['action']))
hfits={r['name']:r for r in trial['historical_fits']};candidates=[r['name'] for r in trial['candidates']];originals=[];runs=[];started=perf_counter()
for original in failed:
 name=original['name'];candidate=exact_candidate(original,3);audit=independent_replay(candidate,original,joint,meter)
 assert audit['success'];originals.append(dict(name=name,declared_optimizer_success=False,declared_status=original['status'],declared_nfev=original['nfev'],loss=original['loss'],loglik=original['loglik'],independent_replay=audit))
 seed=int(trial['seeds']['policy'])+200+candidates.index(name)
 for mode,initial in [('warm_history',hfits[name]),('continue_original_return',original)]:
  result=run_fit(candidate,joint,meter,initial=initial,mode=mode,seed=seed,max_nfev=500,multistart=1)
  write_json(OUT/(name+'_'+mode+'.json.gz'),result);runs.append(result)
  best=result['returned_best_fit'];print(json.dumps(dict(candidate=name,mode=mode,success=best is not None and best['success'],loss=None if best is None else best['loss'],nfev=None if best is None else best['nfev'],AC_calls=result['attempted_AC_calls']['attempted'],seconds=result['solver_seconds']),sort_keys=True),flush=True)
write_json(OUT/'original_failed_states_replay.json',originals)
summary=[]
for original in failed:
 name=original['name'];rows=[r for r in runs if r['candidate']==name];valid=[r['returned_best_fit'] for r in rows if r['returned_best_fit'] is not None];success=[r for r in valid if r['success']]
 summary.append(dict(name=name,original_loss=original['loss'],original_loglik=original['loglik'],original_status=original['status'],original_success=False,diagnostic_successful_starts=len(success),diagnostic_start_count=len(rows),best_successful_loss=None if not success else min(r['loss'] for r in success),best_finite_returned_loss=None if not valid else min(r['loss'] for r in valid),all_saved_states_independent_replay_passed=all(r['all_replays_passed'] for r in rows),AC_calls=sum(r['attempted_AC_calls']['attempted'] for r in rows)))
after={str(p.relative_to(BASE)):sha(p) for p in paths};assert before==after and source_audit(BASE,freeze)==sources
result=dict(schema=protocol['schema'],diagnostic_only=True,original_records_and_infinite_threshold_unchanged=True,original_calibration_H=39,original_calibration_failures=1,original_evaluation_H=0,requested_evaluation_H=48,successful_candidate_diagnostics=sum(r['diagnostic_successful_starts']>0 for r in summary),candidate_denominator=3,candidates=summary,new_fit_calls=len(runs),new_solver_starts=sum(len(r['starts']) for r in runs),new_solver_nfev=sum(s['returned']['nfev'] for r in runs for s in r['starts'] if 'returned' in s),actual_AC_calls=sum(r['attempted_AC_calls']['attempted'] for r in runs),AC_failure_calls=sum(r['attempted_AC_calls']['failed'] for r in runs),new_saved_state_replays=sum(r['replay_count'] for r in runs),original_failed_state_replays=3,all_replays_passed=all(r['all_replays_passed'] for r in runs),solver_seconds=sum(r['solver_seconds'] for r in runs),whole_diagnostic_seconds=perf_counter()-started,immutable_sha256=after,scope='attained local continuation fits only; no global MLE, no new design, no scientific outcome or score/threshold change')
write_json(OUT/'summary.json',result);assert result['all_replays_passed'];print(json.dumps(result,sort_keys=True),flush=True)
