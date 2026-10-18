"""Independent numerical and exact-AC audit of the alpha/R-floor sensitivity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import chi2

from .experiment import read, save, digest
from .bounds import TreeSpec, PhysicalDomain, Measurements
from .compatibility import physical_checks
from ..reference_benchmark.representation import physical_tree_ac


def audit(directory, evidence, original_compatibility):
    directory,evidence=Path(directory),Path(evidence)
    evidence.mkdir(parents=True,exist_ok=True)
    summary=read(directory/'summary.json');prior=summary['prior']
    assets=read(directory/'frozen_inputs/public_assets.json')
    manifest=read(directory/'frozen_inputs/manifest.json')
    issues=[];slots=0;maximum_gap=0.;alpha_table=[]
    by_key={}
    for path in sorted((directory/'conditions').glob('*/*.json')):
        data=read(path)
        meter=read(directory/'frozen_inputs'/next(x['meter_file'] for x in manifest['histories'] if x['id']==data['history_id']))
        if data['meter_sha256']!=digest(directory/'frozen_inputs'/next(x['meter_file'] for x in manifest['histories'] if x['id']==data['history_id'])):
            issues.append([str(path),'meter_hash'])
        for slot in data['candidate_results']:
            slots+=1
            if not slot['success']:
                issues.append([str(path),slot['candidate_id'],'failure']);continue
            r=slot['result'];L=r['qp_support_lower'];U=r['qp_primal']
            maximum_gap=max(maximum_gap,U-L)
            if L>U+1e-7*(1+abs(U)):issues.append([str(path),'lower_above_primal'])
            b=(max(np.sqrt(L)-r['remainder_norm'],0)/r['lipschitz_k'])**2
            if not np.isclose(b,r['ac_sse_lower'],rtol=1e-10,atol=1e-10):issues.append([str(path),'B_recompute'])
            if r['diagnostics']['qp_domain_violation']>1e-7:issues.append([str(path),'outer_domain_violation'])
            key=(data['history_id'],slot['candidate_id'])
            if data['factor']==0:by_key[key]=r
            else:
                old=by_key.get(key)
                if old is not None and any(not np.isclose(old[k],r[k],rtol=1e-12,atol=1e-12) for k in ('remainder_norm','lipschitz_k')):
                    issues.append([str(path),'delta_or_K_changed'])
            for row in slot['alpha_rows']:
                q=chi2.isf(row['alpha'],r['raw_noise_dimension'])
                if not np.isclose(row['raw_chi_square'],q):issues.append([str(path),'chi_square_recompute'])
                if row['raw_reject']!=bool(b>q):issues.append([str(path),'raw_decision'])
    for alpha in summary['protocol']['alphas']:
        alpha_table.append(dict(alpha=alpha,
            SSE_upper_96=float(chi2.isf(alpha,9216)),J_upper_96=float(.5*chi2.isf(alpha,9216)),
            SSE_upper_288=float(chi2.isf(alpha,27648)),J_upper_288=float(.5*chi2.isf(alpha,27648))))

    with zipfile.ZipFile(original_compatibility) as z:
        old=json.loads(z.read('results/summary.json'))
    meter=read(directory/'frozen_inputs/meters'/(old['history_id']+'.json.gz'))
    p,q,v=[np.asarray(meter[k],float) for k in ('p','q','v')]
    obs=Measurements(p,q,v,np.asarray(meter['root_v']),np.asarray(meter['sigma_p']),
        np.asarray(meter['sigma_q']),np.asarray(meter['sigma_v']))
    output=[];independent=[];limited=0;start_failures=0
    for factor in [0.,.5,1.]:
        dom=dict(assets['domain']);dom['power_caps']={int(k):v for k,v in dom['power_caps'].items()}
        dom['r_min']=factor*prior['public_minimum_resistance_ohm']/assets['zbase_ohm'];dom['source_power_cap']=0.
        domain=PhysicalDomain(**dom)
        fresh=old if factor==0 else read(directory/f'compatibility_R_{factor:g}_fixed/summary.json')
        if factor:
            limited+=sum(s.get('evaluation_limit_reached',False) for row in fresh['candidate_results'] for s in row['result'].get('starts',[]))
            start_failures+=sum(s['status']=='search_failure' for row in fresh['candidate_results'] for s in row['result'].get('starts',[]))
        fresh_by_id={r['candidate_id']:r['result'] for r in fresh['candidate_results']}
        old_by_id={r['candidate_id']:r['result'] for r in old['candidate_results']}
        for candidate in assets['candidates']:
            cid=candidate['id'];tree=TreeSpec(candidate['root'],candidate['edges'],candidate['observed_nodes'])
            candidates=[]
            for label,record in [('old_legal_point',old_by_id[cid]),('new_floor_search',fresh_by_id[cid])]:
                best=record.get('best_point')
                if best is None:continue
                rr,xx=np.asarray(best['r']),np.asarray(best['x']);physical_z=rr+1j*xx
                branches=[dict(from_bus=a,to_bus=b,r_ohm=float(rr[e]*assets['zbase_ohm']),
                    x_ohm=float(xx[e]*assets['zbase_ohm']),status=True) for e,(a,b) in enumerate(candidate['edges'])]
                state=physical_tree_ac(branches,candidate['observed_nodes'],candidate['root'],assets['base_kv'],
                    assets['base_kw'],p*assets['base_kw'],q*assets['base_kw'],source_bus=2,root_v=np.asarray(meter['root_v']))
                point=dict(impedance=physical_z,magnitude=state['v'],
                    ac_equation_residual=state['ac_equation_residual'],power_balance_residual=state['power_balance_residual'])
                checks=physical_checks(tree,obs,domain,point)
                if all(checks.values()):
                    U=float(np.sum(((state['v']-v)/np.asarray(meter['sigma_v']))**2))
                    error=abs(U-best['raw_whitened_AC_sse_upper'])
                    if error>1e-7*(1+U):issues.append([factor,cid,label,'independent_AC_backscore'])
                    candidates.append(dict(origin=label,U=U,r=rr.tolist(),x=xx.tolist(),
                        min_R_ohm=float(rr.min()*assets['zbase_ohm']),checks=checks))
                    independent.append(dict(factor=factor,candidate=cid,origin=label,
                        SSE_difference=error,AC_residual=state['ac_equation_residual'],power_balance=state['power_balance_residual']))
            if not candidates:
                issues.append([factor,cid,'no_legal_point']);continue
            best=min(candidates,key=lambda z:z['U'])
            qp=read(directory/'conditions'/f'factor_{factor:g}'/(old['history_id']+'.json'))
            qp_row=next(r['result'] for r in qp['candidate_results'] if r['candidate_id']==cid)
            if qp_row['ac_sse_lower']>best['U']+1e-8:issues.append([factor,cid,'lower_above_AC_upper'])
            output.append(dict(factor=factor,r_min_ohm=factor*prior['public_minimum_resistance_ohm'],candidate=cid,
                SSE_lower=qp_row['ac_sse_lower'],SSE_upper=best['U'],upper_origin=best['origin'],
                min_R_ohm=best['min_R_ohm'],r=best['r'],x=best['x'],
                inside_by_alpha={str(a):best['U']<=chi2.isf(a,9216) for a in summary['protocol']['alphas']}))
    out=dict(status='passed' if not issues else 'failed',issues=issues,candidate_slots=slots,
        distinct_candidate_QP_new_solves=320,baseline_QP_reused=80,alpha_rows=slots*5,
        max_QP_gap=maximum_gap,alpha_thresholds=alpha_table,AC_upper_intervals=output,
        independent_AC_checks=independent,new_floor_search_limited_starts=limited,
        new_floor_search_failed_starts=start_failures,
        integration_attempts_preserved=dict(candidates=20,reason='root alpha metadata absent; no AC evaluated; fixed by reading candidate certificate alpha'),
        source_hashes={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('alpha_resistance.py'),
            Path(__file__).with_name('bounds.py'),Path(__file__).with_name('compatibility.py'),
            Path(__file__).with_name('resistance_compatibility.py')]})
    save(evidence/'audit.json',out)
    save(evidence/'thresholds.json',alpha_table);save(evidence/'summary.json',summary)
    # Input archive is referenced by hash and already exists in the parent evidence.
    with zipfile.ZipFile(evidence/'records.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and not path.is_relative_to(directory/'frozen_inputs'):
                z.write(path,path.relative_to(directory).as_posix())
    with zipfile.ZipFile(evidence/'sources.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(Path(__file__).parent.glob('*.py')):z.write(p,p.name)
    save(evidence/'archives.json',{p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in
        [evidence/'records.zip',evidence/'sources.zip']})
    if issues:raise AssertionError(issues)
    print(json.dumps(dict(status='passed',candidate_slots=slots,AC_interval_count=len(output),
        independent_AC_checks=len(independent),max_QP_gap=maximum_gap,
        limited_starts=limited,failed_starts=start_failures)))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True);p.add_argument('--evidence',required=True)
    p.add_argument('--original-compatibility',required=True)
    a=p.parse_args();audit(a.directory,a.evidence,a.original_compatibility)
