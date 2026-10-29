"""Use all available past before pricing; keep raw aggregates private."""
from __future__ import annotations
import argparse,json,hashlib,pickle,sys,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import make_observed_scenario
from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import fit_observed,source_hashes
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,input_hash

SCRATCH=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
SOURCE=SCRATCH/'wzzt_data_calibration/flow_runs/paper15_iflex_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=root/'protocol_history.json'
    sources=source_hashes();sources[str(Path(__file__).resolve())]=sha(__file__)
    if target.exists():
        p=read(target)
        if p['source_sha256']!=sources:raise RuntimeError('frozen history sources changed')
        for f,h in p['input_sha256'].items():
            if sha(f)!=h:raise RuntimeError('frozen past input changed '+f)
        return p
    inventory=read(SCRATCH/'dr_paper15_update_build/history_inventory.json')
    dates=inventory['history_dates'];assert dates==sorted(set(dates)) and len(dates)==10
    scenes=[];hashes={}
    for date in dates:
        if date>=inventory['information_cutoff']:raise RuntimeError('future date not allowed')
        f=SOURCE/'history'/(date+'.pkl')
        if sha(f)!=inventory['source_sha256'][str(f)]:raise RuntimeError('past source hash drift')
        hashes[str(f)]=sha(f)
        # This loader extracts observed only; physical tables/truth never leave it.
        observed=pickle.loads(f.read_bytes())['observed']
        scene=make_observed_scenario(observed['P_terminal'],observed['Q_terminal'],observed['V_terminal'],observed['root_voltage'],observed['name'])
        if set(scene)!={'name','P_terminal','Q_terminal','V_terminal','V_root_observed','drop_target','units'}:
            raise RuntimeError('unexpected observation channel')
        if len(scene['P_terminal'])!=24:raise RuntimeError('incomplete past day')
        scenes.append(scene)
    packet=dict(training=scenes[:-1],validation=scenes[-1:],metadata=dict(training_dates=dates[:-1],validation_dates=dates[-1:],
        all_available_past_dates=dates,information_cutoff=inventory['information_cutoff'],base_kw=200.,base_kv=.4,base_mva=.2,
        no_unexplained_history_window=True,forecast_dates=dates,truth_used=False,new_future_accessed=False))
    private=root/'allpast_observed.pkl';private.write_bytes(pickle.dumps(packet,protocol=4));hashes[str(private)]=sha(private)
    obs,ids,counts=normalize(packet['training'],'training');val,vids,vcounts=normalize(packet['validation'],'validation')
    if len(obs[0])!=216 or len(val[0])!=24 or obs[0].index.intersection(val[0].index).size:
        raise RuntimeError('past split mismatch')
    p=dict(created_at=datetime.now(timezone.utc).isoformat(),stage='all available past no-reward baseline before DR decisions',
        source_sha256=sources,input_sha256=hashes,private_packet=str(private),
        training_dates=dates[:-1],validation_dates=dates[-1:],training_rows=216,validation_rows=24,
        normalized_input_sha256=dict(training=input_hash(obs,ids),validation=input_hash(val,vids)),
        settings=dict(search_seconds_per_path=25.,LP_seconds=15.,coefficient_bound=2.,
            actual_algorithm='existing observed finite-pool RNJ/WZZT best-strategy adapter; not free-support MILP',
            full_pool_scan=True,continuously_refit_from_zero=True,acceptance='gain>1e-10+.001*current',
            selection='original3% + support-count + original .16 contraction',DR_payment=0.,new_future_values_accessed=False),
        previous_short_window='48 training +24 validation was development only; do not claim DR reduces observations relative to it',
        frozen_after='allpast packet preparation; before baseline solver',
        stop='exactly one complete baseline call; existing partial/failed/completed result is retained, never implicit rerun')
    write(target,p);return p


def run(root):
    root=Path(root);p=prepare(root);done=root/'history_summary.json';out=root/'history_fit'
    if done.exists():return read(done)
    if (out/'result.json').exists() or (out/'checkpoint.json').exists():
        raise RuntimeError('existing run retained; inspect instead of implicit rerun')
    packet=pickle.loads(Path(p['private_packet']).read_bytes())
    result=fit_observed(packet['training'],packet['validation'],out,search_seconds=25.)
    # Post-return physical truth belongs to a separate evaluation only.
    from research_experiments.wzzt_dr_selection.dr_paper15_update.study import evaluate
    evaluation=evaluate(result['models'],result['terminal_labels'],packet['training'],packet['validation'])
    write(root/'history_evaluation.json',evaluation)
    summary=dict(status=result['status'],protocol_sha256=sha(root/'protocol_history.json'),fit_sha256=sha(out/'result.json'),
        private_packet_sha256=sha(p['private_packet']),input_sha256=result['input_sha256'],
        training_rows=result['training_rows'],validation_rows=result['validation_rows'],pool_size=result['pool_size'],
        actual_LPs=len(result['LP_solves']),all_local_scans_complete=result['all_local_scans_complete'],
        search_stop_statuses=result['search_stop_statuses'],evaluation=evaluation,
        no_future_values_accessed=True,DR_payment=0.,no_new_price_score=True)
    write(done,summary);print(json.dumps(summary,ensure_ascii=False),flush=True);return summary


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--prepare-only',action='store_true');x=a.parse_args()
    p=prepare(x.root) if x.prepare_only else run(x.root)
    if x.prepare_only:print(json.dumps({k:p[k] for k in ('stage','training_rows','validation_rows','settings')}),flush=True)
