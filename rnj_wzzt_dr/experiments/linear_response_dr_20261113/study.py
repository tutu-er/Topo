"""History-only screen, bounded relaxed pricing, then fresh model confirmation."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
import argparse
from pathlib import Path
from time import perf_counter
import shutil,subprocess,traceback
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..expected_logodds_admm_20261113.run import REFERENCE,load_target
from .models import supporting_start,power_box_upper

MODES=('storage_relaxation','linear_response')


def prepare(out):
    require(not out.exists(),'Preserve existing attempts')
    old=io.read(REFERENCE/'planning/protocol.json');out.mkdir(parents=True)
    targets=[]
    for item in old['selected']:
        score_path=REFERENCE/'scores'/(item['id']+'.json');score=io.read(score_path)
        for target in item['targets']:
            path=REFERENCE/'development/targets'/(item['id']+f'__target{target}.json')
            qual=io.read(path);competitors=qual['gates']['0.9']['allocation']['active']
            if not competitors:
                competitors=[next(int(j) for j in np.argsort([-m['probability'] for m in score['models']]) if j!=target)]
            targets.append(dict(id=item['id'],case=item['case'],target=target,competitors=competitors,
                scores=io.ref(score_path),observed_qualification=io.ref(path)))
    sources={}
    for folder in (Path(__file__).parent,ROOT/'rnj_wzzt_dr/experiments/expected_logodds_admm_20261113',ROOT/'rnj_wzzt_dr/dr'):
        for path in sorted(folder.glob('*.py')):
            rel=path.relative_to(ROOT);dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest);sources[rel.as_posix()]=io.ref(path)
    protocol=dict(created_utc=utc(),sources=sources,targets=targets,gain_required=float(np.log(2)),
        commit_at_lock=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        modes=list(MODES),screen_seconds_per_target_mode=6.,ADMM_targets_max=4,ADMM_seconds_per_target_mode=30.,
        selection='Select two distinct inputs by the largest minimum achieved gain / required gain in the history-only screen; retain both pre-existing target directions. No physical future or truth used.',
        model_paths=2048,model_seed=2026111303,model_namespace='linear_response_dr_20261113_v1',
        screen_scope='One zero-anchor supporting-plane QP. Its infeasibility is not a certificate for the quadratic gain problem. The power-only box gives a valid upper bound.',
        storage_relaxation='hourly no-clipping domain; convex charge/discharge hull with original efficiency and energy/viability limits; simultaneous flows allowed; original net-delivery contract fee',
        linear_response='same hourly power/availability bounds and net-delivery fee, no intertemporal energy constraints; declared simplified response, not an original storage policy',
        comparison_scope='same frozen candidates, history, common sigma2, original power capacities and gamma; no identifier or parameter refits',
        synthetic='Separate analytic pair with equal row sums and a 1% shared branch; deliberately weak natural excitation. Does not modify real fits.',
        new_AC_runs=False,truth_role='joined after all planning for offline qualification and interpretation only')
    io.write(out/'protocol.json',protocol);io.write(out/'lock.json',dict(pid=os.getpid(),protocol_sha256=sha(out/'protocol.json')))
    return protocol


def screen(out):
    protocol=io.read(out/'protocol.json');rows=[]
    for item in protocol['targets']:
        try:
            data=load_target(item,protocol['gain_required']);e=data['evidence'];score=data['score']
            J=e.base_norm2*e.scales**2
            for mode in MODES:
                directory=out/'screen'/(item['id']+f"__target{item['target']}")/mode;directory.mkdir(parents=True)
                began=perf_counter();point=supporting_start(data,mode,protocol['screen_seconds_per_target_mode'])
                if point.get('u') is not None:
                    np.savez_compressed(directory/'quote.npz',u=point['u'],x=point['x'])
                    replay=data['original'].project(point['u'])
                    point.update(original_device_audit=data['original'].audit(point['u']),
                        projected_gains=e.gains(replay),projected_fee=data['original'].fees(replay))
                meta={k:v for k,v in point.items() if k not in ('u','x')}
                io.write(directory/'summary.json',meta)
                rows.append(dict(**item,mode=mode,status='complete',sigma2=score['sigma2'],
                    baseline_distance_standardized=J,baseline_raw_distance=2*score['sigma2']*J,
                    historical_log_odds=[score['models'][item['target']]['log_probability']-score['models'][j]['log_probability'] for j in item['competitors']],
                    power_box_gain_upper=power_box_upper(data),screen=meta,fee_scale=data['fee_scale']))
                print(item['id'],item['target'],mode,'QP',point['status'],'qualified',point['qualified'],'seconds',round(perf_counter()-began,2),flush=True)
                io.write(out/'screen_results.json',rows)
        except Exception as exc:
            rows.append(dict(**item,status='failed',error=repr(exc),traceback=traceback.format_exc()))
            io.write(out/'screen_results.json',rows);print('retained failure',item['id'],repr(exc),flush=True)
    io.write(out/'screen_complete.json',dict(status='complete',rows=len(rows),results_sha256=sha(out/'screen_results.json')))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['prepare','screen'],required=True)
    args=p.parse_args();out=args.output.resolve()
    if args.stage=='prepare':prepare(out)
    else:screen(out)


if __name__=='__main__':main()
