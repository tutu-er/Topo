"""Read-only matching of saved fixed-parent curves against mobile score calls."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from . import gradient_price_study as io
from . import ev_systematic_cases as oldcases
from . import ev_mobile_cases_v2 as newcases


def curve_key(power):
    a=np.round(np.asarray(power,float),10);a[a==0.]=0.
    return a.astype('<f8').tobytes()


def inspect_parent(raw):
    methods={m['name']:m for m in raw['methods']};parent=methods['scalar_fixed']['power_kw']
    key=curve_key(parent);T,n=np.asarray(parent).shape
    selection=raw['selection']['methods']['scalar_mobile']['raw']
    initial=selection.get('initial_candidates',[]);matches=[]
    for index,item in enumerate(initial):
        profile=item.get('profile') or {}
        if profile.get('power_kw') is not None and curve_key(profile['power_kw'])==key:
            matches.append(dict(path=f'selection.methods.scalar_mobile.raw.initial_candidates[{index}].profile',
                stage='initial',origin=item.get('origin'),complete=bool(profile.get('success') and item.get('success')),
                pair_count=len(profile.get('pairs',[])),score=item.get('score'),
                max_absolute_difference_kw=float(np.max(abs(np.asarray(profile['power_kw'])-np.asarray(parent))))))
    for index,item in enumerate(selection.get('grids',[])):
        profile=item.get('raw_AC_profile') or {}
        if profile.get('power_kw') is not None and curve_key(profile['power_kw'])==key:
            matches.append(dict(path=f'selection.methods.scalar_mobile.raw.grids[{index}].raw_AC_profile',
                stage='grid',origin='scalar_MILP_'+str(item['grid_points']),complete=bool(profile.get('success') and item.get('raw_AC_score') is not None),
                pair_count=len(profile.get('pairs',[])),score=item.get('raw_AC_score'),
                max_absolute_difference_kw=float(np.max(abs(np.asarray(profile['power_kw'])-np.asarray(parent))))))
    complete=[m for m in matches if m['complete']]
    parent_rows=[dict(index=i,**{k:v for k,v in item.items() if k!='profile'}) for i,item in enumerate(initial) if item.get('origin')=='fixed_parent']
    skipped=any(item.get('attempted') is False and 'budget' in str(item.get('failure','')) for item in parent_rows)
    if any(m['complete'] and m['origin']=='fixed_parent' for m in matches):category='complete_fixed_parent_origin'
    elif any(m['complete'] and m['stage']=='initial' for m in matches):category='complete_other_initial_origin_same_curve'
    elif complete:category='complete_later_grid_same_curve'
    elif matches:category='failed_profiles_only'
    elif skipped:category='deadline_skipped_no_scored_duplicate'
    else:category='unknown'
    return dict(category=category,complete_parent_curve_scored=bool(complete),matching_profiles=matches,
        fixed_parent_initial_rows=parent_rows,explicit_parent_deadline_skip=skipped,
        parent_curve_sha256=sha256(key).hexdigest(),shape=[T,n],
        duplicate_rule='same rounded-to-10-decimal time-by-node power used by the frozen scalar selector; actual differences also recorded',
        selected_profile_not_counted_as_new_call=True)


def branch_compensation(raw):
    """First-order R/X combination on calibrated training PQ, not a new fit."""
    case=raw['case_id'];bank=newcases.candidate_bank('remote8' if case.startswith('remote8_H') else case) if case.startswith('remote8_H') or case=='twinblock12' else oldcases.candidate_bank(case)
    candidate=next(c for c in bank if c.name==raw['true_name'])
    fit=next(f for f in raw['training_result']['fits'] if f.get('candidate',f.get('name'))==raw['true_name'])
    tr,tm=raw['training'],raw['training_meter'];p=np.asarray(tr['p'])-np.asarray(tm['bias_p']);q=np.asarray(tr['q'])-np.asarray(tm['bias_q'])
    dr=np.asarray(fit['r'])-np.asarray(raw['truth']['r']);dx=np.asarray(fit['x'])-np.asarray(raw['truth']['x'])
    pf=p/np.hypot(p,q);rows=[]
    for e,support in enumerate(candidate.supports):
        P=p[:,list(support)].sum(axis=1);Q=q[:,list(support)].sum(axis=1)
        source_included=e==0
        if source_included:P=P+np.asarray(fit['source_p']);Q=Q+np.asarray(fit['source_q'])
        good=abs(P)>1e-8;kappa=Q[good]/P[good];k=float(np.median(kappa)) if kappa.size else None
        combo=dr[e]+k*dx[e] if k is not None else None;scale=abs(dr[e])+abs(k*dx[e]) if k is not None else None
        rows.append(dict(edge=e,support=list(support),R_true=raw['truth']['r'][e],X_true=raw['truth']['x'][e],
            R_training=fit['r'][e],X_training=fit['x'][e],delta_R=dr[e],delta_X=dx[e],
            kappa_median=k,kappa_range=[float(kappa.min()),float(kappa.max())] if kappa.size else None,
            delta_R_plus_kappa_delta_X=combo,cancellation_ratio=abs(combo)/scale if scale and scale>0 else None,
            opposite_sign=bool(dr[e]*dx[e]<0),source_included=source_included,
            near_zero_flow_rows=int((~good).sum())))
    return dict(training_points=len(p),terminal_measured_PF_ranges=[[float(pf[:,j].min()),float(pf[:,j].max())] for j in range(p.shape[1])],edges=rows,
        equation='For radial first-order voltage drop, delta(R*P+X*Q)=P*(delta_R+kappa*delta_X), kappa=Q/P.',
        input_scope='Calibrated observed training terminal P/Q; loss-free downstream sums; root-edge source uses the saved true-candidate training source fit. No latent training truth inserted.',
        interpretation='Small cancellation_ratio with opposite-signed R/X errors supports compensation along this load regime. It does not certify identical AC sensitivities or global identifiability.')


def run(root,audit_path,output):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    evidence=io.read_json(audit_path)
    moved={(bank,r['index']):r for bank,d in evidence['banks'].items() for r in d['rows'] if r['mobile']['moves']}
    selected={(bank,index) for bank in ('n3','n4') for index in (0,12,24,36)}|{('n3',9),('n3',20),('n4',14)}
    rows=[];comp=[]
    for bank,index in sorted(set(moved)|selected):
        choices=list((root/('mobile_v2_formal_eval_'+bank)).glob(f'case_{index:03d}_*.json.gz'))
        try:
            if len(choices)!=1:raise ValueError('exactly one raw history required')
            p=choices[0];content=p.read_bytes();raw=io.read_json(p)
            identity=dict(bank=bank,index=index,case_id=raw['case_id'],seed=raw['seed'],truth=raw['true_name'],
                raw_path=str(p.resolve()),raw_sha256=sha256(content).hexdigest())
            if (bank,index) in moved:rows.append(dict(**identity,**inspect_parent(raw)))
            if (bank,index) in selected:comp.append(dict(**identity,**branch_compensation(raw)))
        except Exception as exc:
            if (bank,index) in moved:rows.append(dict(bank=bank,index=index,category='unknown',failure=type(exc).__name__+': '+str(exc)))
            if (bank,index) in selected:comp.append(dict(bank=bank,index=index,failure=type(exc).__name__+': '+str(exc)))
        print(json.dumps(dict(bank=bank,index=index,processed=True)),flush=True)
    summary={bank:dict(denominator=sum(b==bank for b,i in moved),categories=dict(Counter(r['category'] for r in rows if r['bank']==bank)),
        complete_parent_curve_scored=sum(r.get('complete_parent_curve_scored',False) for r in rows if r['bank']==bank)) for bank in ('n3','n4')}
    result=dict(schema='mobile_v3_fixed_parent_score_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Only frozen v2 records with actual scalar mobility; all 18 n3 + 34 n4 retained. No solver/AC/fit.',
        denominator=52,summary=summary,rows=rows,training_compensation_diagnostic=comp,
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        upstream_evidence=dict(path=str(Path(audit_path).resolve()),sha256=sha256(Path(audit_path).read_bytes()).hexdigest()))
    io.write_json(output,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs'))
    parser.add_argument('--audit',type=Path,default=Path('outputs/mobile_v3_scalar_evidence_audit.json'))
    parser.add_argument('--output',type=Path,default=Path('outputs/mobile_v3_parent_initial_audit.json'))
    args=parser.parse_args();x=run(args.root,args.audit,args.output);print(json.dumps(x['summary']),flush=True)


if __name__=='__main__':main()
