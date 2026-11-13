"""Development-only rescore with analytically fixed common-mode noise weight.

Exploratory candidate derived during the current analysis, not part of the
locked primary or secondary confirmation outcomes. No new fitting/controls.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import numpy as np
from ... import io
from .data import load_entry
from .relative_voltage_diagnostic import loss_vectors,choose


def all_losses(H,P,Y):
    raw,relative=loss_vectors(H,P,Y)
    weighted=relative+(raw-relative)/(Y.shape[1]+1)
    if np.any(weighted<relative-1e-12) or np.any(weighted>raw+1e-12):
        raise ArithmeticError('Fixed weighted loss must be between relative and raw loss')
    return dict(original=raw,relative=relative,weighted=weighted)


def aggregate(rows):
    result={}
    for name in sorted({r['strategy'] for r in rows}):
        own=[r for r in rows if r['strategy']==name]
        counts={kind:sum(r[kind+'_correct'] for r in own) for kind in ('original','relative','weighted')}
        result[name]=dict(n=len(own),correct=counts,
            weighted_vs_original=dict(rescues=sum(r['weighted_correct'] and not r['original_correct'] for r in own),
                losses=sum(r['original_correct'] and not r['weighted_correct'] for r in own)),
            weighted_vs_relative=dict(rescues=sum(r['weighted_correct'] and not r['relative_correct'] for r in own),
                losses=sum(r['relative_correct'] and not r['weighted_correct'] for r in own)))
    return result


def run(run_dir,out):
    if out.exists():
        raise ValueError('Preserve earlier exploratory results')
    protocol=io.read(run_dir/'protocol.json')
    if protocol['phase']!='development':
        raise ValueError('This newly derived criterion is restricted to development; no confirmation rescoring')
    if not (run_dir/'complete.json').exists():
        raise ValueError('Only completed old development data may be used')
    entries=io.read(run_dir/'inputs.json');rows=[];history=[];errors=[]
    for entry in entries:
        score,_,validation,truth=load_entry(entry)
        bank=score['native_bank'];H=np.asarray([v['H'] for v in bank]);counts=[v['internal'] for v in bank]
        past=all_losses(H,np.c_[validation.p,validation.q],validation.y)
        past_indices={kind:choose(loss,counts,score['cells'],score) for kind,loss in past.items()}
        history.append(dict(id=entry['id'],role=entry['role'],**{kind+'_correct':bank[k]['topology_id']==truth['truth_id'] for kind,k in past_indices.items()}))
        for day in range(protocol['days']):
            d=run_dir/'days'/entry['id']/f'day{day:03d}'
            for saved in io.read(d/'results.json'):
                if saved['physics']!='AC':
                    continue
                if saved['status']!='complete':
                    errors.append(dict(id=entry['id'],day=day,strategy=saved['strategy'],error='Existing execution incomplete'))
                    continue
                with np.load(d/(saved['strategy']+'_AC.npz')) as z:
                    P=np.c_[z['P'],z['Q']];Y=z['Y'].copy()
                future=all_losses(H,P,Y);cells=score['cells']+Y.size
                selected={kind:choose(past[kind]+loss,counts,cells,score) for kind,loss in future.items()}
                if selected['original']!=saved['native_selected_index']:
                    errors.append(dict(id=entry['id'],day=day,strategy=saved['strategy'],error='Original native selection mismatch'))
                rows.append(dict(id=entry['id'],day=day,role=entry['role'],case=entry['case'],strategy=saved['strategy'],
                    common_component_weight=1/(Y.shape[1]+1),selected={kind:int(k) for kind,k in selected.items()},
                    **{kind+'_correct':bank[k]['topology_id']==truth['truth_id'] for kind,k in selected.items()}))
    table={(r['id'],r['day'],r['strategy']):r for r in rows}
    versus_zero={}
    for strategy in sorted({r['strategy'] for r in rows}):
        if strategy=='no_dfs':continue
        own=[r for r in rows if r['strategy']==strategy]
        versus_zero[strategy]=dict(n=len(own),
            rescues=sum(r['weighted_correct'] and not table[r['id'],r['day'],'no_dfs']['weighted_correct'] for r in own),
            losses=sum(not r['weighted_correct'] and table[r['id'],r['day'],'no_dfs']['weighted_correct'] for r in own))
    result=dict(source=io.ref(__file__),protocol=io.ref(run_dir/'protocol.json'),
        classification='New exploratory criterion derived during analysis; development-only; not a locked confirmation metric',
        loss='sum ||residual-rowmean||² + N*rowmean²/(N+1), equivalently relative_SSE+(original_SSE-relative_SSE)/(N+1)',
        assumptions='First-order multiplicative voltage-meter Gaussian law at approximately unit root and terminal voltages, identical meter accuracy. Root meter error shared across terminals. Scalar likelihood factor omitted because common to all candidates. Power-meter error and frozen parameter bias remain.',
        changes=dict(new_parameter_fits=0,new_optimization=0,new_physical_simulations=0,new_future_paths=0,
            primary_or_locked_secondary_changes=0,candidate_parameters_changed=0,free_tuning_parameters=0),
        overall=aggregate(rows),by_role={role:aggregate([r for r in rows if r['role']==role]) for role in sorted({r['role'] for r in rows})},
        by_case={case:aggregate([r for r in rows if r['case']==case]) for case in sorted({r['case'] for r in rows})},
        weighted_strategy_vs_weighted_no_dfs=versus_zero,
        history=dict(n=len(history),correct={kind:sum(r[kind+'_correct'] for r in history) for kind in ('original','relative','weighted')},
            weighted_vs_original=dict(rescues=sum(r['weighted_correct'] and not r['original_correct'] for r in history),losses=sum(r['original_correct'] and not r['weighted_correct'] for r in history)),
            weighted_vs_relative=dict(rescues=sum(r['weighted_correct'] and not r['relative_correct'] for r in history),losses=sum(r['relative_correct'] and not r['weighted_correct'] for r in history))),
        errors=errors,passed=not errors)
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'source_snapshot.py')
    result['source_snapshot']=io.ref(out/'source_snapshot.py')
    io.write(out/'report.json',result);io.write(out/'rows.json',rows);io.write(out/'history.json',history)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.run.resolve(),args.out.resolve())
    import json
    print(json.dumps(dict(history=result['history'],overall={k:v for k,v in result['overall'].items() if not k.endswith('_rotation')},
        errors=result['errors'],passed=result['passed']),indent=2))
