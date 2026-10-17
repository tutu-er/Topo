"""Compare saved A/B voltage responses without fitting or generating data."""
from pathlib import Path
import os,sys,json,gzip,argparse
from hashlib import sha256
from time import perf_counter
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
from research_experiments.mobile_ev_mle.ac_history_refresh import io
from research_experiments.mobile_ev_mle import ev_ac_repair_analysis as analysis
from research_experiments.mobile_ev_mle.evidence.ev_ac_repair_v1.scripts.oracle_mean_audit import layout


def voltage_errors(mean,base_mean,actual,base_actual,part):
    """RMS error of absolute voltage and of the increment over constant."""
    ix=part['indices'];unit=np.asarray(part['sigma'])*part['physical_scale']
    response=(np.asarray(mean)-np.asarray(base_mean))[ix]*unit
    signal=(np.asarray(actual)-np.asarray(base_actual))[ix]*unit
    rms=lambda v:float(np.sqrt(np.mean(v*v)))
    error=rms(response-signal);size=rms(signal)
    return dict(increment_error_V=error,increment_signal_V=size,
        relative_increment_error=error/size if size>1e-12 else None,
        absolute_voltage_error_V=rms((np.asarray(mean)-np.asarray(actual))[ix]*unit))


def run(A_root,B_root,output):
    if Path(output).exists():raise FileExistsError(output)
    A_root=Path(A_root);B_root=Path(B_root)
    assert io.load_record(B_root/'controller.json')['status']=='statistical_complete'
    began=perf_counter();rows=[];receipts=[]
    for index in range(32):
        paths={'A':A_root/'evaluation'/f'history_{index:03d}.json.gz',
               'B':B_root/'evaluation'/f'history_{index:03d}.json.gz'}
        records={name:io.load_record(path) for name,path in paths.items()}
        original=records['A'];parts,_,_=layout(original)
        names=original['candidate_names'];true=names.index(original['true_name'])
        for backend,record in records.items():
            receipts.append(dict(backend=backend,index=index,path=str(paths[backend]),
                sha256=sha256(paths[backend].read_bytes()).hexdigest()))
            base_key=record.get('plans',{}).get('constant',{}).get('action_key')
            base=record.get('actions',{}).get(base_key,{})
            base_banks=analysis.usable_banks(base)
            for plan in ('SQMD','BCMD'):
                key=record.get('plans',{}).get(plan,{}).get('action_key')
                action=record.get('actions',{}).get(key,{})
                banks=analysis.usable_banks(action)
                for model in ('G00','G11'):
                    row=dict(backend=backend,index=index,plan=plan,model=model,available=False)
                    rows.append(row)
                    if model not in banks or model not in base_banks:
                        row['failure']='saved action or accepted predictive bank unavailable';continue
                    means=analysis.scaled_bank(banks[model])[0]
                    base_means=analysis.scaled_bank(base_banks[model])[0]
                    if backend=='B':
                        assert key==original['plans'][plan]['action_key']
                        assert np.array_equal(action['power'],original['actions'][key]['power'])
                    values=[]
                    for regime in ('regular','shift'):
                        current=action.get('regimes',{}).get(regime,{})
                        reference=base.get('regimes',{}).get(regime,{})
                        if 'clean' not in current or 'clean' not in reference:break
                        if backend=='B':assert np.array_equal(current['clean'],original['actions'][key]['regimes'][regime]['clean'])
                        values.append(voltage_errors(means[true],base_means[true],
                            current['clean'],reference['clean'],parts['v']))
                    if len(values)!=2:
                        row['failure']='saved clean response unavailable';continue
                    row.update(available=True,**{field:float(np.mean([v[field] for v in values]))
                        if all(v[field] is not None for v in values) else None for field in values[0]})
        for receipt in receipts[-2:]:
            assert sha256(Path(receipt['path']).read_bytes()).hexdigest()==receipt['sha256']
    summaries={};paired={}
    for plan in ('SQMD','BCMD'):
        for model in ('G00','G11'):
            by_backend={}
            for backend in ('A','B'):
                selected=[r for r in rows if (r['backend'],r['plan'],r['model'])==(backend,plan,model) and r['available']]
                by_backend[backend]={r['index']:r for r in selected}
                summaries[f'{backend}__{plan}__{model}']=dict(histories=len(selected),total=32,
                    missing=32-len(selected))
                for field in ('increment_error_V','increment_signal_V','relative_increment_error','absolute_voltage_error_V'):
                    usable=[r[field] for r in selected if r[field] is not None]
                    summaries[f'{backend}__{plan}__{model}'][field]=float(np.mean(usable)) if usable else None
                    summaries[f'{backend}__{plan}__{model}'][field+'_histories']=len(usable)
            common=sorted(set(by_backend['A'])&set(by_backend['B']));result=dict(histories=len(common),total=32)
            for field in ('increment_error_V','relative_increment_error','absolute_voltage_error_V'):
                valid=[i for i in common if all(by_backend[b][i][field] is not None for b in ('A','B'))]
                diff=[by_backend['B'][i][field]-by_backend['A'][i][field] for i in valid]
                result[field]=dict(histories=len(valid),A=float(np.mean([by_backend['A'][i][field] for i in valid])) if valid else None,
                    B=float(np.mean([by_backend['B'][i][field] for i in valid])) if valid else None,
                    difference=float(np.mean(diff)) if diff else None,bootstrap95=analysis.interval(diff))
            paired[f'{plan}__{model}']=result
    result=dict(scope='saved fixed-plan prediction error; actual future realizations held fixed, not a conditional-mean bias estimator',
        histories=32,rows=rows,summaries=summaries,paired=paired,input_hashes=receipts,
        independent_unit='history; two existing regimes averaged',new_AC_calls=0,new_fits=0,new_actions=0,new_data=0,
        covariance_calibration_changes_means=False,seconds=perf_counter()-began)
    io.save_record(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--A-root',required=True)
    parser.add_argument('--B-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.A_root,args.B_root,args.output)
    print(json.dumps(result['paired']))
