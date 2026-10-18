"""Re-threshold saved observations and test predeclared resistance floors.

No new histories or true-topology calibration samples are generated. The
selector reads only the frozen public candidate domain and noisy meters.
The public all-asset table supplies a possible engineering R-floor; floors
above its minimum are explicitly stress tests with no coverage claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
from scipy.stats import chi2

from .bounds import TreeSpec, PhysicalDomain, Measurements, certify_candidate
from .experiment import read, save, digest
from ..case33_quality_fixed_rx import experiment as base


CONFIG=dict(version='alpha_resistance_sensitivity_v1',
    alphas=[.20,.10,.05,.01,.001],
    resistance_factors=[0.,.25,.50,1.,2.],
    solve_alpha=.05, workers=2,
    input_policy='reuse the exact eight frozen conditions; no new history or noise',
    floor_policy='factor times the minimum R of the full public closed-plus-tie asset table',
    stress_policy='factor>1 is outside the all-asset coverage assumption and is not recommended for deployment')


def residual_radius(result, meter, alpha):
    df=result['residual_chi_df']
    if not df:return 0.
    old_alpha=result['alpha']
    k0=result['residual_chi_linear_radius']/np.sqrt(chi2.isf(old_alpha/3,df))
    nv=np.asarray(meter['v']).size
    sigma=float(np.max(meter['sigma_v']))
    root=meter.get('sigma_root')
    if root is not None and np.any(np.asarray(root)>0):
        raise ValueError('this audit uses the declared fixed-root batch')
    quadratic=sigma*sigma*np.sqrt(2*np.log(2*nv/(alpha/3))*chi2.isf(alpha/3,nv))
    return float(k0*np.sqrt(chi2.isf(alpha/3,df))+quadratic)


def threshold_rows(result, meter, history, candidate, factor, floor_ohm):
    out=[]
    for alpha in CONFIG['alphas']:
        q=float(chi2.isf(alpha,result['raw_noise_dimension']))
        nr=residual_radius(result,meter,alpha)
        out.append(dict(history=history,candidate=candidate,samples=meter['samples'],
            source_mode=meter['source_mode'],factor=factor,r_min_ohm=floor_ohm,alpha=alpha,
            prior_covers_all_public_assets=factor<=1.,
            qp_lower=result['qp_support_lower'],qp_upper=result['qp_primal'],
            AC_sse_lower=result['ac_sse_lower'],raw_chi_square=q,J_noise_upper=.5*q,
            raw_reject=bool(result['ac_sse_lower']>q),
            sqrt_qp_lower=float(np.sqrt(result['qp_support_lower'])),
            AC_remainder=result['remainder_norm'],raw_noise_radius=result['lipschitz_k']*np.sqrt(q),
            residual_noise_radius=nr,residual_rejection_threshold=(result['remainder_norm']+nr)**2,
            residual_reject=bool(result['qp_support_lower']>(result['remainder_norm']+nr)**2)))
    return out


def solve_condition(bundle, history, factor, floor_ohm, destination):
    began=perf_counter();bundle=Path(bundle)
    meter_path=bundle/history['meter_file']
    meter=read(meter_path);assets=read(bundle/'public_assets.json')
    dom=deepcopy(assets['domain'])
    dom['power_caps']={int(k):v for k,v in dom['power_caps'].items()}
    dom['r_min']=floor_ohm/assets['zbase_ohm']
    if meter['source_mode']=='none':dom['source_power_cap']=0.
    obs=Measurements(**{k:np.asarray(meter[k],float) for k in
        ('p','q','v','root_v','sigma_p','sigma_q','sigma_v')},sigma_root=meter['sigma_root'])
    domain=PhysicalDomain(**dom)
    records=[]
    old=read(bundle/'results'/(history['id']+'.json'))
    by_id={s['candidate_id']:s for s in old['candidate_results']}
    for item in assets['candidates']:
        try:
            if factor==0:
                result=by_id[item['id']]['result']
            else:
                tree=TreeSpec(item['root'],item['edges'],item['observed_nodes'],
                    item['source_node'] if meter['source_mode']!='none' else None)
                result=certify_candidate(tree,obs,domain,alpha=CONFIG['solve_alpha'],
                    source_projection=bool(meter['source_projection'])).as_dict()
            records.append(dict(candidate_id=item['id'],success=True,result=result,
                alpha_rows=threshold_rows(result,meter,history['id'],item['id'],factor,floor_ohm)))
        except Exception as exc:
            records.append(dict(candidate_id=item['id'],success=False,failure=repr(exc)))
    out=dict(history_id=history['id'],factor=factor,r_min_ohm=floor_ohm,
        prior_covers_all_public_assets=factor<=1.,candidate_results=records,
        meter_sha256=digest(meter_path),public_assets_sha256=digest(bundle/'public_assets.json'),
        selector_read_paths=[str(meter_path),str(bundle/'public_assets.json'),
            str(bundle/'results'/(history['id']+'.json'))] if factor==0 else
            [str(meter_path),str(bundle/'public_assets.json')],
        source_hashes={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('bounds.py')]},
        seconds=perf_counter()-began)
    save(Path(destination)/'conditions'/f'factor_{factor:g}'/(history['id']+'.json'),out)
    return out


def run(source_archive, destination):
    began=perf_counter();source_archive=Path(source_archive);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    bundle=destination/'frozen_inputs'
    bundle.mkdir(exist_ok=True)
    with zipfile.ZipFile(source_archive) as z:
        # The archive is the existing repository evidence, not an external upload.
        z.extractall(bundle)
    manifest=read(bundle/'manifest.json');assets=read(bundle/'public_assets.json')
    net=base.make_context()['network']
    all_assets=net['branches']+net['metadata']['normally_open_ties']
    public_r_min=min(e['r_ohm'] for e in all_assets)
    public_z_min=min(np.hypot(e['r_ohm'],e['x_ohm']) for e in all_assets)
    old_z_min=assets['domain']['z_min']*assets['zbase_ohm']
    prior=dict(public_minimum_resistance_ohm=public_r_min,
        public_minimum_impedance_modulus_ohm=public_z_min,
        original_minimum_modulus_ohm=old_z_min,
        original_implied_physical_R_min_ohm=old_z_min/np.sqrt(1+assets['domain']['ratio_max']**2),
        original_QP_implied_R_min_ohm=old_z_min/(1+assets['domain']['ratio_max']),
        asset_table_source=net['provenance'],zbase_ohm=assets['zbase_ohm'])
    save(destination/'protocol.json',dict(config=CONFIG,prior=prior,
        source_archive_sha256=digest(source_archive),input_histories=len(manifest['histories']),
        truth_used_for_selection=False,source_hashes={p.name:digest(p) for p in
            [Path(__file__),Path(__file__).with_name('bounds.py')]}))
    settings=[(h,f,f*public_r_min) for f in CONFIG['resistance_factors'] for h in manifest['histories']]
    results=[]
    with ProcessPoolExecutor(max_workers=CONFIG['workers']) as pool:
        pending={pool.submit(solve_condition,bundle,h,f,r,destination):(h['id'],f) for h,f,r in settings}
        for future in as_completed(pending):
            value=future.result();results.append(value)
            save(destination/'progress.json',dict(completed=len(results),total=len(settings),
                failed_candidates=sum(not s['success'] for x in results for s in x['candidate_results'])))
    rows=[a for r in results for slot in r['candidate_results'] if slot['success'] for a in slot['alpha_rows']]
    rows.sort(key=lambda r:(r['factor'],r['alpha'],r['history'],r['candidate']))
    with (destination/'bounds_by_alpha.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for factor in CONFIG['resistance_factors']:
        for alpha in CONFIG['alphas']:
            block=[r for r in rows if r['factor']==factor and r['alpha']==alpha]
            summary.append(dict(factor=factor,r_min_ohm=factor*public_r_min,alpha=alpha,
                candidate_slots=len(block),prior_covers_all_public_assets=factor<=1.,
                raw_rejected=sum(r['raw_reject'] for r in block),
                residual_rejected=sum(r['residual_reject'] for r in block),
                max_AC_sse_lower=max(r['AC_sse_lower'] for r in block),
                qp_lower_range=[min(r['qp_lower'] for r in block),max(r['qp_lower'] for r in block)]))
    save(destination/'summary.json',dict(status='complete',protocol=CONFIG,prior=prior,summary=summary,
        candidate_failures=sum(not s['success'] for r in results for s in r['candidate_results']),
        elapsed_seconds=perf_counter()-began))
    print(json.dumps(dict(status='complete',conditions=len(results),candidate_solves=len(rows)//len(CONFIG['alphas']),
        failures=sum(not s['success'] for r in results for s in r['candidate_results']),
        max_AC_sse_lower=max(r['AC_sse_lower'] for r in rows),seconds=perf_counter()-began)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-archive',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.source_archive,args.output)
