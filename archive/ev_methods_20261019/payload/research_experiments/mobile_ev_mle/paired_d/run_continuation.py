"""Execute the separately authorized continuation; preserve every checkpoint."""
from pathlib import Path
from datetime import datetime, timezone
import json, math
try:
    from . import paired_runner as r
except ImportError:
    import paired_runner as r

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs_v2'
OUT.mkdir(exist_ok=True)

def stamp(): return datetime.now(timezone.utc).isoformat()
def main():
    start=stamp()
    r.actual_worker_resources()
    manifest_path=OUT/'frozen.json'
    manifest=r.read(manifest_path) if manifest_path.exists() else r.freeze(manifest_path)
    r.dump(OUT/'continuation_execution.json',dict(started_at_utc=start,
        user_continuation_authorized_at_utc='2026-10-13T01:54:00+00:00',
        original_eight_hour_budget_expired=True,execution_deadline_utc='2026-10-13T03:00:00+00:00',
        source_and_statistics_frozen_before_new_histories=True))
    pilot=r.run_batch('pilot',OUT/'pilot',manifest_path)
    print('pilot '+json.dumps(pilot),flush=True)
    if not pilot['formal_schedule_eligible']: return
    calibration=r.run_batch('calibration',OUT/'calibration',manifest_path)
    print('calibration '+json.dumps(calibration),flush=True)
    if not calibration['all_requested_histories_attempted']: return
    rows=[r.read(OUT/'calibration'/f'trial_{s}.json') for s in range(940001,940033)]
    q={key:r.contracts.upper_quantile([
        z['score_record'][key] if z['score_record']['complete'] else math.inf for z in rows],.05)
        for key in ['relative','absolute']}
    gate=dict(frozen_at_utc=stamp(),calibration_count=32,quantiles=q,
        evaluation_started=False,policy_id=manifest['policy_id'],
        calibration_trial_sha256={f'trial_{s}.json':r.sha(OUT/'calibration'/f'trial_{s}.json') for s in range(940001,940033)})
    gate_path=OUT/'frozen_calibration_gate.json'
    if gate_path.exists():
        old=r.read(gate_path)
        assert old['quantiles']==r.finite(q) and old['calibration_trial_sha256']==gate['calibration_trial_sha256']
    else: r.dump(gate_path,gate)
    evaluation=r.run_batch('evaluation',OUT/'evaluation',manifest_path)
    print('evaluation '+json.dumps(evaluation),flush=True)
    if not evaluation['all_requested_histories_attempted']: return
    result=r.collect(OUT/'calibration',OUT/'evaluation',OUT/'audit')
    assert r.finite(result['calibration'])==r.read(gate_path)['quantiles']
    r.dump(OUT/'completion.json',dict(completed_at_utc=stamp(),
        calibration_count=32,evaluation_count=48,pilot_in_formal_denominator=False,
        frozen_gate_sha256=r.sha(gate_path),within_original_eight_hour_budget=False))
    print('complete '+json.dumps(result['methods']),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        r.dump(OUT/'execution_exception_checkpoint.json',dict(at_utc=stamp(),error=repr(exc)))
        raise
