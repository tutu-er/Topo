"""Summarize numerical timing with failures and endpoint quality kept separate."""
from pathlib import Path
import argparse,json,statistics


def summarize(output):
    output=Path(output);rows=[]
    for path in sorted((output/'tasks').glob('*/result.json')):
        record=json.loads(path.read_bytes());task=record['task'];worker=record.get('worker',{})
        audit=worker.get('audit',{});fit=worker.get('fit',{})
        rows.append(dict(network=task['network'],candidate=task['candidate_name'],
            H=task['prefix'],start=task['start'],algorithm=task['algorithm'],
            inner_tol=task['inner_tol'],repeat=task['repeat'],status=record['status'],
            solver_success=worker.get('solver_success'),loss=audit.get('loss',fit.get('loss')),
            fit_seconds=worker.get('timings',{}).get('fit_seconds'),
            process_seconds=record.get('process_wall_seconds'),
            initialization_seconds=worker.get('timings',{}).get('initialization_seconds'),
            gradient=audit.get('reference_scaled_projected_gradient'),
            physical_pass=bool(audit.get('available') and audit.get('physical_voltage_feasible')
                and audit.get('line_domain_feasible') and audit.get('source_and_optimizer_box_feasible')
                and audit.get('ac_equation_residual',1)<2e-10
                and audit.get('power_balance_residual',1)<2e-10),
            replay_loss_difference=audit.get('reported_loss_absolute_difference'),
            initial_sha=worker.get('initial_state_sha256'),
            history_sha=task['history_content_sha256'],task_id=task['task_id']))
    index={(r['network'],r['candidate'],r['H'],r['start'],r['repeat'],r['algorithm']):r for r in rows}
    for row in rows:
        baseline=('fixed_ref' if row['algorithm'].startswith('fixed_') else
                  'ref_jac' if row['algorithm'] in ('block_jac','block_jac_gap','block_gpu') and
                    tuple(row[k] for k in ('network','candidate','H','start','repeat'))+('ref_jac',) in index
                  else 'ref_sparse')
        key=tuple(row[k] for k in ('network','candidate','H','start','repeat'))+(baseline,)
        other=index.get(key)
        row['baseline']=baseline;row['paired_baseline_available']=False
        if not other or row['loss'] is None or other['loss'] is None:continue
        same=(row['initial_sha']==other['initial_sha'] and row['history_sha']==other['history_sha'])
        row.update(paired_baseline_available=True,same_history_and_initial=same,
                   loss_difference=row['loss']-other['loss'])
        objective_ok=row['loss']<=other['loss']+1e-4+1e-6*abs(other['loss'])
        gradient_ok=(row['gradient'] is not None and other['gradient'] is not None
                     and row['gradient']<=max(1e-6,1.1*other['gradient']))
        row['objective_not_worse']=objective_ok;row['gradient_not_worse']=gradient_ok
        row['fit_speed_ratio']=(other['fit_seconds']/row['fit_seconds']
            if row['fit_seconds'] and other['fit_seconds'] else None)
        row['qualified_speed_comparison']=bool(same and row['solver_success'] and other['solver_success']
            and row['physical_pass'] and other['physical_pass'] and objective_ok and gradient_ok)
    groups={}
    for row in rows:
        key=(row['network'],row['candidate'],row['H'],row['start'],row['algorithm'],row['inner_tol'])
        groups.setdefault(key,[]).append(row)
    aggregate=[]
    for key,items in groups.items():
        values=[r['fit_seconds'] for r in items if r['fit_seconds'] is not None]
        qualified=[r['fit_speed_ratio'] for r in items if r.get('qualified_speed_comparison')
                   and r['algorithm']!=r['baseline']]
        aggregate.append(dict(network=key[0],candidate=key[1],H=key[2],start=key[3],algorithm=key[4],
            inner_tol=key[5],attempts=len(items),returned=sum(r['status']=='returned' for r in items),
            solver_successes=sum(bool(r['solver_success']) for r in items),
            physical_passes=sum(r['physical_pass'] for r in items),
            median_fit_seconds=statistics.median(values) if values else None,
            qualified_speed_pairs=len(qualified),
            median_qualified_speed_ratio=statistics.median(qualified) if qualified else None,
            losses=[r['loss'] for r in items],gradients=[r['gradient'] for r in items]))
    return dict(scope='numerical development on history; no EV effectiveness or statistical calibration',
        attempted_saved=len(rows),timeout_count=sum(r['status']=='timeout' for r in rows),
        exception_count=sum(r['status'] not in ('returned','timeout') for r in rows),
        qualification='same history/start, both successful and physical, loss <= ref+1e-4+1e-6*abs(ref), gradient <= max(1e-6,1.1*ref)',
        timing_scope='fit_seconds excludes cold worker import/initialization and independent final audit; process_seconds includes them',
        rows=rows,groups=aggregate)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=summarize(args.output)
    path=args.output/'summary.json';path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','groups')},ensure_ascii=False))


if __name__=='__main__':main()
