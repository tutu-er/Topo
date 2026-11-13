"""Offline measured-direction decomposition; true matrices never enter planning."""
from pathlib import Path
import argparse
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,utc
from ..unified_posterior_dr_20261112.physical import open_runtime
from ..bidirectional_price_dr_20261113.diagnose import physical_H
from ..posterior_mse_20261112.deliver import write_csv
from .study import SCENARIOS
from ..unified_posterior_dr_20261112.history_scan import ROOT


def run(run,out):
    require(not out.exists(),'Preserve existing direction analysis');out.mkdir(parents=True)
    entries={e['request']['id']:e for e in io.read(run/'nominal30/inputs.json')}
    truth={r['id']:r for r in io.read(ROOT/'outputs/unified_posterior_dr_20261112_v1/evaluation.json')}
    cfg=io.read(run/'protocol.json');records=[];modules={};Hphysical={}
    io.write(out/'protocol.json',dict(created_utc=utc(),source=io.ref(__file__),scope='offline attribution on completed confirmation paths only',
        true_parameters_in_planner=False,new_independent_days=0,new_AC_calls=0))
    for scenario in SCENARIOS:
        root=run/scenario
        for identity in cfg['confirmation']['ids']:
            e=entries[identity];score=io.read(e['scores']['path']);H=np.array([p['H'] for p in score['parameters']]);sigma=score['sigma2']
            ti=next(i for i,m in enumerate(score['models']) if m['topology_id']==truth[identity]['truth_id'])
            j=e['target'] if e['target']!=ti else max((k for k in range(len(H)) if k!=ti),key=lambda k:score['models'][k]['probability'])
            D=H[ti]-H[j];rows=[];failed_quotes=0
            for directory in sorted((root/'confirmation/jobs'/identity).glob('day*')):
                meta=io.read(directory/'generation.json');cohort=e['request']['cohort']
                if cohort not in modules:modules[cohort]=open_runtime(root,cohort)[0]
                if identity not in Hphysical:Hphysical[identity]=physical_H(modules[cohort],meta,score['labels'])
                require(meta['physical_branch_sha256']==e['physical_branch_sha256'],'Physical direction version drift')
                failed_quotes+=not io.read(directory/'plans.json')['g4']['qualified']
                with np.load(directory/'physical.npz',allow_pickle=False) as p,np.load(directory/'observed.npz',allow_pickle=False) as o:
                    names=o['strategy'].tolist();values={}
                    for strategy in ('no_dr','optimized_g4'):
                        index=names.index(strategy);P=np.c_[p['P'][index],p['Q'][index]]
                        Y=p['true_root'][:,None]**2-p['V'][index]**2;difference=P@D.T
                        distance=float(np.sum(difference*difference)/(2*sigma))
                        parameter=float(np.sum((P@(Hphysical[identity]-H[ti]).T)*difference)/sigma)
                        ac=float(np.sum((Y-P@Hphysical[identity].T)*difference)/sigma)
                        odds=float(o['log_probability'][index,ti]-o['log_probability'][index,j])
                        values[strategy]=dict(distance=distance,parameter_bias=parameter,AC=ac,
                            meter_plus_fixed_history=odds-distance-parameter-ac,actual_log_odds=odds)
                    change={key:values['optimized_g4'][key]-values['no_dr'][key] for key in values['no_dr']}
                    require(abs(sum(change[k] for k in ('distance','parameter_bias','AC','meter_plus_fixed_history'))-change['actual_log_odds'])<1e-7,'Decomposition failed')
                    rows.append(change)
            records.append(dict(scenario=scenario,id=identity,paths=len(rows),failed_quotes=failed_quotes,
                competitor=j,truth_index=ti,**{key:float(np.mean([r[key] for r in rows])) for key in rows[0]}))
    write_csv(out/'summary.csv',records);io.write(out/'summary.json',records)
    io.write(out/'complete.json',dict(status='passed',groups=len(records),new_independent_days=0,new_AC_calls=0,source=io.ref(__file__)))
    for r in records:
        if r['id']=='main__input039__101':print(r,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run.resolve(),a.output.resolve())
