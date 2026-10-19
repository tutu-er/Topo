"""Model self-risk versus observed error, without changing saved EV plans.

The new random numbers integrate a declared predictive density; they are not
new physical datasets, future meter noise, or new actions. No policy is tuned.
"""
from pathlib import Path
import sys,os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
WORK=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(WORK))
import argparse
from hashlib import sha256
from time import perf_counter
import numpy as np
from research_experiments.mobile_ev_mle.ac_history_refresh import io
from research_experiments.mobile_ev_mle import ev_ac_repair_analysis as analysis
from research_experiments.mobile_ev_mle import ev_bayes_small_probability as probability


def integrate(bank,multiplier,seed,draws=512,replicates=4):
    means,covs,prior=analysis.scaled_bank(bank,multiplier)
    model=dict(means=means[None],covariances=covs[None],log_prior=prior,weights=np.ones(1))
    bounds=probability.bhattacharyya_bank(model)
    risks=[];within=[];noise_hashes=[];began=perf_counter()
    for rep in range(replicates):
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),9120,rep]))
        noise=rng.normal(size=(1,len(means),draws,means.shape[-1]))
        noise_hashes.append(sha256(noise.astype('<f8').tobytes()).hexdigest())
        value=probability.evaluate_bank(model,noise)
        risks.append(float(value['risk']));within.append(float(value['risk_se']))
    return dict(model_risk=float(np.mean(risks)),
        independent_replicate_SE=float(np.std(risks,ddof=1)/np.sqrt(replicates)),
        mean_internal_stratified_SE=float(np.mean(within)),replicate_estimates=risks,
        Gaussian_Bhattacharyya_upper_bound=float(bounds['capped_bound']),
        numerical_integration_draws=draws*replicates*len(means),noise_sha256=noise_hashes,
        seconds=perf_counter()-began,parameter_and_probability_model_unchanged=True,
        zero_MC_SE_is_not_a_population_or_numerical_error_bound=True)


def run(A_root,B_root,output,phase='evaluation',expected=32,draws=512,replicates=4):
    if Path(output).exists():raise FileExistsError(output)
    began=perf_counter();rows=[];receipts=[]
    for backend,root in (('A',Path(A_root)),('B',Path(B_root))):
        calibration=io.load_record(root/'calibration.json')
        for index in range(expected):
            path=root/phase/f'history_{index:03d}.json.gz';record=io.load_record(path)
            receipts.append(dict(backend=backend,index=index,sha256=sha256(path.read_bytes()).hexdigest()))
            # Model integral seed uses only the public saved history identifier.
            seed=int(record['spec']['seed']);names=record.get('candidate_names',[])
            true=record.get('true_name');true_index=names.index(true) if true in names else None
            cache={}
            for label in ('constant','SQMD','BCMD'):
                key=record.get('plans',{}).get(label,{}).get('action_key')
                action=record.get('actions',{}).get(key,{})
                banks=analysis.usable_banks(action)
                for model in ('G00','G11','G11_cal'):
                    row=dict(backend=backend,index=index,plan=label,model=model,available=False)
                    rows.append(row);bank=banks.get('G11' if model=='G11_cal' else model)
                    a=calibration.get('G11_multiplier') if model=='G11_cal' else 1.
                    if bank is None or a is None or true_index is None:continue
                    cache_key=(key,'G11' if model=='G11_cal' else model,float(a))
                    try:
                        if cache_key not in cache:cache[cache_key]=integrate(bank,a,seed,draws,replicates)
                        row.update(cache[cache_key],available=True,multiplier=a)
                        error=[];clean_error=[]
                        for regime in ('regular','shift'):
                            actual=action.get('regimes',{}).get(regime,{})
                            if 'values' in actual:
                                error.append(analysis.scores(bank,actual['values'],true_index,a=a)['error'])
                                clean_error.append(analysis.scores(bank,actual['clean'],true_index,a=a)['error'])
                        row.update(observed_error=float(np.mean(error)) if len(error)==2 else None,
                                   observed_clean_error=float(np.mean(clean_error)) if len(clean_error)==2 else None)
                    except Exception as exc:row.update(available=False,failure=type(exc).__name__+': '+str(exc))
    summary={}
    for backend in ('A','B'):
        for label in ('constant','SQMD','BCMD'):
            for model in ('G00','G11','G11_cal'):
                rr=[r for r in rows if r['backend']==backend and r['plan']==label and r['model']==model
                    and r['available'] and r.get('observed_error') is not None]
                summary[f'{backend}__{label}__{model}']=dict(histories=len(rr),total_histories=expected,
                    missing_histories=expected-len(rr),
                    **{field:(float(np.mean([r[field] for r in rr])) if rr else None) for field in
                       ('model_risk','Gaussian_Bhattacharyya_upper_bound','observed_error','observed_clean_error')},
                    observed_error_bootstrap95=analysis.interval([r['observed_error'] for r in rr]))
    result=dict(scope='post-hoc fixed-plan model integration; no new physical observations or decisions',
        draws_per_candidate_per_replicate=draws,replicates=replicates,rows=rows,summaries=summary,
        input_hashes=receipts,seconds=perf_counter()-began,
        comparison_scope='model risk uses its own topology/process law; observed error uses the stated simulated truth law',
        observed_error_interval_unit='independent history; two saved regimes averaged',
        integration_SE_scope='within a fixed predictive model; excludes model error',
        method_selection_or_calibration_changed=False)
    io.save_record(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--A-root',required=True);parser.add_argument('--B-root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--expected',type=int,default=32)
    args=parser.parse_args();run(args.A_root,args.B_root,args.output,expected=args.expected)
