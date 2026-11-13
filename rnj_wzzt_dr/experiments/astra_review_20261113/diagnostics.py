"""Read-only decomposition of already opened AC paths; no new fits or AC solves.

Residual mean calibration is a planning diagnostic. Frozen candidate scores
are never changed. Raw derived paths remain in the ignored output directory.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, roots_hermitenorm
from scipy.stats import norm, binomtest, t as student_t

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'outputs/customer_flexibility_sweep_20261113_v1'
EXTRA = ROOT/'outputs/customer_flexibility_intensity_20261113_v1'
ID = 's0_c013'
GH, GW = roots_hermitenorm(384)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def reference(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def observation(path):
    with np.load(path) as z:
        return np.c_[z['P'],z['Q']], z['root'][:,None]**2-z['V']**2

def moments(mu, variance):
    p = float(GW@expit(mu+np.sqrt(max(0.,variance))*GH)/np.sqrt(2*np.pi))
    return p, float(norm.cdf(mu/np.sqrt(variance))) if variance else float(mu>=0)

def calibrations(X,y,H,sigma2):
    prediction=X@H.T
    r=y-prediction
    scale=np.sum(prediction*r,axis=0)/(np.sum(prediction**2,axis=0)+len(X)*sigma2)
    centered=prediction-prediction.mean(0)
    affine_scale=np.sum(centered*(r-r.mean(0)),axis=0)/(np.sum(centered**2,axis=0)+len(X)*sigma2)
    return {'constant':(r.mean(0),np.zeros_like(H)),
            'multiplicative':(np.zeros(y.shape[1]),scale[:,None]*H),
            'affine':(r.mean(0)-affine_scale*prediction.mean(0),affine_scale[:,None]*H)}

def main(out):
    if out.exists():
        raise ValueError('Preserve prior diagnostic output')
    out.mkdir(parents=True)
    hist=BASE/'histories'/ID
    score=read(hist/'score.json')
    H=np.asarray([p['H'] for p in score['parameters']])
    target=score['map_index']
    others=[j for j in range(len(H)) if j!=target]
    runner=max(others,key=lambda j:score['models'][j]['log_probability'])
    Ht,Hj=H[target],H[runner]
    D=Ht-Hj
    sigma2=score['sigma2']
    Lh=score['models'][target]['log_probability']-score['models'][runner]['log_probability']
    Xt,yt=observation(hist/'train.npz')
    Xv,yv=observation(hist/'validation.npz')
    Xall,yall=np.r_[Xt,Xv],np.r_[yt,yv]
    models={'nominal':(np.zeros(Ht.shape[0]),np.zeros_like(Ht))}
    for name,X,y in [('validation',Xv,yv),('whole_history',Xall,yall)]:
        models.update({name+'_'+k:v for k,v in calibrations(X,y,Ht,sigma2).items()})
    config=read(BASE/'resource_configs/higher.json')
    labels=config['expected_terminal_labels']
    nodes=[labels.index(r['node_label']) for r in config['resources']]
    ratios=[r.get('q_over_p',np.tan(np.arccos(r.get('power_factor',1.)))) for r in config['resources']]
    write(out/'protocol.json',dict(source=reference(__file__),score=reference(hist/'score.json'),
        histories=[reference(hist/(name+'.npz')) for name in ('train','validation')],
        new_AC_days=0,new_identifier_fits=0,new_optimized_plans=0,
        purpose='Decompose existing 32 paired paths; compare fixed simple historical residual corrections without changing candidate scoring.',
        split='48 half-hour samples starting08:00; train first32(08:00-23:30), validation last16(00:00-07:30). Full-history calibration uses both existing observed subsets.',
        calibration_choices=list(models),
        truth_use='True P/Q/V used only for offline error decomposition, never for calibration.',
        uncertainty='Gaussian moments are first-order meter-error diagnostics. Actual scoring includes exact original nonlinear meter noise.',
        conditional_scope='Same selected weak history; paths already used for prior confirmation and now development.'))
    rows=[]
    refs=[]
    policies=[('no_dfs',BASE,'no_dfs'),('small',BASE,'optimized_g0'),('two',EXTRA,'optimized_g0'),
              ('four',EXTRA,'optimized_g1'),('rotation_two',EXTRA,'rotation_g0')]
    history_stats={}
    for name,X,y in [('train',Xt,yt),('validation',Xv,yv),('whole_history',Xall,yall)]:
        d=X@D.T;r=y-X@Ht.T
        history_stats[name]=dict(points=len(X),mean_total_active_power_pu=float(X[:,:15].sum(1).mean()),
            target_MSE=float(np.mean(r*r)),pair_distance=float(np.sum(d*d)/(2*sigma2)),
            residual_projection=float(np.sum(r*d)/sigma2))
    for label,root,strategy in policies:
        results={r['day']:r for r in read(root/'confirmation'/ID/'results.json') if r['strategy']==strategy}
        for day in range(32):
            ddir=root/'confirmation'/ID/f'day{day:03d}'
            refs += [reference(ddir/(strategy+'.npz')),reference(ddir/'planner.npz'),reference(ddir/'physical_background.npz')]
            with np.load(ddir/'physical_background.npz') as z:
                rootv=np.array(z['root'])
            with np.load(ddir/'planner.npz') as z:
                forecast=np.array(z['forecast']);u=np.array(z[strategy])
            shift=np.zeros((24,30))
            for k,(node,ratio) in enumerate(zip(nodes,ratios)):
                shift[:,node]+=u[:,k]/200
                shift[:,15+node]+=ratio*u[:,k]/200
            nominal=forecast+np.repeat(shift,4,axis=0)
            with np.load(ddir/(strategy+'.npz')) as z:
                actual=np.c_[z['true_P'],z['true_Q']]
                true_v=np.array(z['true_V'])
                measured=np.c_[z['P'],z['Q']]
                ym=z['root'][:,None]**2-z['V']**2
                logodds=float(z['log_probability'][target]-z['log_probability'][runner])
                actual_p=float(np.exp(z['log_probability'][target]))
            ytrue=rootv[:,None]**2-true_v**2
            d=actual@D.T;b=ytrue-actual@Ht.T
            distance=float(np.sum(d*d)/(2*sigma2))
            bias=float(np.sum(b*d)/sigma2)
            expected_voltage_bias=float(.0005**2*np.sum(ytrue*d)/sigma2)
            variance_p=(.005*actual)**2
            gram_diagonal=np.sum(Hj*Hj-Ht*Ht,axis=0)
            expected_pq_bias=float(.5*np.sum(variance_p*gram_diagonal[None])/sigma2)
            power_gradient=b@D-d@Hj
            variance_pq=float(np.sum(variance_p*power_gradient**2)/sigma2**2)
            variance_v=float(4*.0005**2*(np.sum(true_v**4*d*d)+np.sum(rootv**4*d.sum(1)**2))/sigma2**2)
            oracle_mu=Lh+distance+bias+expected_voltage_bias+expected_pq_bias
            oracle_prob,oracle_correct=moments(oracle_mu,variance_pq+variance_v)
            by_time={}
            for name,ix in [('daytime',slice(0,64)),('night',slice(64,None))]:
                by_time[name]=dict(distance=float(np.sum(d[ix]**2)/(2*sigma2)),
                    frozen_bias=float(np.sum(b[ix]*d[ix])/sigma2),target_mse=float(np.mean(b[ix]**2)))
            predictions={}
            for cname,(offset,matrix) in models.items():
                for input_name,X in [('forecast',nominal),('true_power',actual)]:
                    delta=X@D.T
                    J=float(np.sum(delta*delta)/(2*sigma2))
                    bias_prediction=float(np.sum((offset[None]+X@matrix.T)*delta)/sigma2)
                    mu=Lh+J+bias_prediction
                    prob,correct=moments(mu,2*J)
                    predictions[cname+'_'+input_name]=dict(mu=mu,J=J,bias=bias_prediction,
                        pair_expected_weight=prob,pair_correct_probability=correct,
                        mean_error_vs_clean=mu-(Lh+distance+bias))
            row=dict(strategy=label,day=day,actual_weight=actual_p,actual_binary_weight=float(expit(logodds)),
                actual_correct=bool(results[day]['native_map_correct']),observed_logodds=logodds,
                true_power_distance=distance,frozen_bias=bias,clean_logodds=Lh+distance+bias,
                meter_logodds_error=logodds-(Lh+distance+bias),
                planned_fee=results[day]['planned_fee_yuan'],settled_fee=results[day]['settled_fee_yuan'],
                delivered_abs_kwh=results[day]['delivered_abs_kwh'],planning_feasible=results[day]['planning_feasible'],
                oracle_first_order=dict(mu=oracle_mu,variance=variance_pq+variance_v,voltage_variance=variance_v,
                    pq_variance=variance_pq,pq_mean_bias=expected_pq_bias,voltage_mean_bias=expected_voltage_bias,
                    expected_pair_weight=oracle_prob,correct_probability=oracle_correct),
                by_time=by_time,predictions=predictions)
            rows.append(row)
    summary=dict(history=history_stats,strategies={},calibration={},paired={})
    for label,_,_ in policies:
        own=[r for r in rows if r['strategy']==label]
        summary['strategies'][label]={k:float(np.mean([r[k] for r in own])) for k in ['actual_weight','actual_binary_weight','true_power_distance','frozen_bias','clean_logodds','observed_logodds','meter_logodds_error','planned_fee','settled_fee','delivered_abs_kwh']}
        summary['strategies'][label].update(correct=sum(r['actual_correct'] for r in own),days=len(own),
            planning_feasible=sum(r['planning_feasible'] is True for r in own),
            oracle_first_order={k:float(np.mean([r['oracle_first_order'][k] for r in own])) for k in own[0]['oracle_first_order']},
            time_decomposition={name:{k:float(np.mean([r['by_time'][name][k] for r in own])) for k in own[0]['by_time'][name]} for name in ('daytime','night')})
        summary['calibration'][label]={key:dict(
            mean_mu=float(np.mean([r['predictions'][key]['mu'] for r in own])),
            mean_pair_expected_weight=float(np.mean([r['predictions'][key]['pair_expected_weight'] for r in own])),
            mean_pair_correct_probability=float(np.mean([r['predictions'][key]['pair_correct_probability'] for r in own])),
            mean_error_vs_clean=float(np.mean([r['predictions'][key]['mean_error_vs_clean'] for r in own])),
            rmse_vs_clean=float(np.sqrt(np.mean([r['predictions'][key]['mean_error_vs_clean']**2 for r in own])))) for key in own[0]['predictions']}
        if label!='no_dfs':
            base=[r for r in rows if r['strategy']=='no_dfs']
            diff=np.array([r['actual_weight']-b['actual_weight'] for r,b in zip(own,base)])
            recovered=sum(r['actual_correct'] and not b['actual_correct'] for r,b in zip(own,base))
            lost=sum(not r['actual_correct'] and b['actual_correct'] for r,b in zip(own,base))
            se=np.std(diff,ddof=1)/np.sqrt(len(diff))
            summary['paired'][label]=dict(mean_probability_gain=float(diff.mean()),
                descriptive_t95_interval=[float(diff.mean()-student_t.ppf(.975,31)*se),float(diff.mean()+student_t.ppf(.975,31)*se)],
                recovered=recovered,lost=lost,unchanged=32-recovered-lost,
                exact_mcnemar_two_sided_p=float(binomtest(recovered,recovered+lost,.5).pvalue) if recovered+lost else 1.,
                declined=int(np.sum(diff< -1e-12)),improved=int(np.sum(diff>1e-12)),worst=float(diff.min()),
                scope='Descriptive fixed-history paired comparison; no multiple-policy correction and no population extrapolation.')
    write(out/'input_refs.json',refs)
    write(out/'private_rows.json',rows)
    write(out/'summary.json',summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=main(args.out.resolve())
    print(json.dumps({'history':result['history'],'paired':result['paired']},indent=2))
