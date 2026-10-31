"""One complete current joint historical baseline, reused by the actual trial."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib,json,pickle,sys,time,traceback
REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
sys.path.insert(0,str(REPO))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def main(root,case):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import SharedFitter
    folder=Path(root)/case;p=read(folder/'baseline_protocol.json')
    assert p['source_sha256']==source_hashes() and p['driver_sha256']==sha(__file__)
    assert p['history_sha256']==sha(folder/'history.pkl')
    if (folder/'baseline_ended.json').exists():raise FileExistsError('baseline terminal attempt retained')
    started=time.perf_counter();out=dict(case=case,status='started',protocol_sha256=sha(folder/'baseline_protocol.json'),truth_or_future_read=False)
    try:
        with (folder/'history.pkl').open('rb') as stream:h=pickle.load(stream)['history']
        fitter=SharedFitter(folder/'fit_cache',sources=p['source_sha256'],max_new_fits=1)
        fit=fitter(h['training'],h['validation'],folder/'baseline',root_label=h['root_label'],search_seconds=25.)
        assert fit['training_rows']==fit['validation_rows']==288
        out.update(status='complete',LP_calls=len(fit['LP_solves']),result_sha256=sha(folder/'baseline/result.json'),
                   search_stops=fit['search_stop_statuses'],seconds=time.perf_counter()-started)
    except Exception as exc:out.update(status='failed',exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
    write(folder/'baseline_ended.json',out);print(json.dumps(out,ensure_ascii=True),flush=True)
    return 0 if out['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True)
    a=p.parse_args();raise SystemExit(main(a.root,a.case))
