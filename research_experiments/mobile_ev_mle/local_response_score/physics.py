"""Observation-only local AC remainder and convex-envelope audit after sealing."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import numpy as np

from ..local_rx_identifiability.cases import candidate_bank
from ..local_rx_identifiability.experiment import read, write
from ..local_rx_three_day.experiment import key
from .experiment import sha
from . import scoring as sc
from .remainder import exact_remainder, remainder_bound
from .rejection import alpha_split, fit_delta_qp, rejection_envelope


def norm_record(values, covariance):
    values=np.asarray(values,float)
    covariance=np.broadcast_to(np.asarray(covariance,float),(len(values),values.shape[1],values.shape[1]))
    weights=np.linalg.solve(covariance,values[...,None])[...,0]
    return dict(total=float(np.sqrt(np.sum(values*weights))),
                per_coordinate_RMS=float(np.sqrt(np.mean(np.sum(values*weights,axis=1))/values.shape[1])))


def box_norm(halfwidth, covariance):
    halfwidth=np.asarray(halfwidth,float)
    # Exact maximum of the weighted norm over a 2-dimensional centred box.
    signs=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    vertices=halfwidth[None,:,:]*signs[:,None,:]
    precision=np.linalg.inv(covariance)
    squared=np.einsum('cta,tab,ctb->ct',vertices,precision,vertices)
    maximum=squared.max(axis=0)
    return dict(total=float(np.sqrt(maximum.sum())),
                per_coordinate_RMS=float(np.sqrt(np.mean(maximum)/2)))


def process(directory):
    directory=Path(directory);config=read(directory/'configuration.json')
    output=directory/'physics.json'
    if output.exists():raise FileExistsError('no overwrite of physical audit')
    frozen={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('scoring.py'),
            Path(__file__).with_name('remainder.py'),Path(__file__).with_name('rejection.py'))}
    records=[];start=perf_counter()
    bank=candidate_bank()
    for spec in config['tasks']:
        name=key(spec);payload=read(directory/'inputs'/f'{name}.json.gz')
        inference=read(directory/'records'/f'{name}.json.gz')
        if inference['status']!='completed':
            records.append(dict(key=name,status='upstream_failure_kept'));continue
        obs=payload['train'];sigma=payload['sigma'];domain=payload['domain']
        data=sc.prepare(obs,sigma);cov=data['cov_y']
        fitted={c.name:inference['methods']['AC'][c.name]['training'] for c in bank}
        predictions={c.name:data['x']@sc.response_matrix(c,fitted[c.name]['r'],fitted[c.name]['x']).T for c in bank}
        item=dict(key=name,spec=spec,candidates={},pairs=[],truth_used=False)
        bounds={}
        for c in bank:
            fit=fitted[c.name]
            exact=exact_remainder(c,fit['r'],fit['x'],fit['p'],fit['q'],root_v=fit['root_v'])
            public=remainder_bound(c,domain)
            meter=remainder_bound(c,domain,obs,sigma,alpha_meter=.01)
            fixed=remainder_bound(c,domain,obs,sigma,alpha_meter=.01,fixed_r=fit['r'],fixed_x=fit['x'])
            meter_norm=box_norm(meter['contrast_halfwidth'],cov)
            public_norm=box_norm(np.broadcast_to(public['contrast_halfwidth'],(len(data['x']),2)),cov)
            bounds[c.name]=meter_norm
            candidate=dict(public=public,meter=meter,fixed_lines_only=fixed,
                feasible_AC_point_remainder=norm_record(exact['contrast_remainder'],cov),
                identity_max_abs=exact['identity_max_abs'],
                uniform_radius= meter_norm, public_uniform_radius=public_norm,
                public_to_meter_radius_ratio=public_norm['total']/meter_norm['total'],
                fixed_line_radius=box_norm(fixed['contrast_halfwidth'],cov),
                uniform_scope='all_public_lines_and_states_within_simultaneous_meter_rectangle',
                conditional_fixed_line_scope='only_saved_training_line_coefficients')
            candidate['envelopes']=[]
            for alpha in (.05,.01):
                split=alpha_split(alpha)
                envelope=(meter if np.isclose(split['alpha_pq'],.01) else
                          remainder_bound(c,domain,obs,sigma,alpha_meter=split['alpha_pq']))
                qp=fit_delta_qp(c,obs,sigma,domain,envelope['contrast_lower'],envelope['contrast_upper'])
                rule=rejection_envelope(qp,data,alpha,source='meter',gate_passed=False)
                candidate['envelopes'].append(dict(alpha=alpha,qp=qp,rule=rule))
            item['candidates'][c.name]=candidate
        for i,c in enumerate(bank):
            for other in bank[i+1:]:
                distance=norm_record(predictions[c.name]-predictions[other.name],cov)
                radius=bounds[c.name]['total']+bounds[other.name]['total']
                item['pairs'].append(dict(a=c.name,b=other.name,
                    fitted_linear_response_distance=distance,
                    sum_uniform_AC_radius=radius,
                    AC_radius_below_fitted_difference=radius < distance['total'],
                    radius_to_difference=radius/max(distance['total'],1e-30),
                    interpretation='scale_check_of_fitted_responses_not_uniform_candidate_separation'))
        # No proof of smaller remainders on all difficult candidates means the
        # planned strict-rejection phase is not enabled by this comparison.
        item['strict_rejection_activated']=False
        item['retained_candidates']=[c.name for c in bank]
        common=set(map(tuple,bank[0].supports))
        for c in bank[1:]:common.intersection_update(map(tuple,c.supports))
        item['common_supported_structure']=dict(
            supports=sorted(common,key=lambda x:(len(x),x)),
            unresolved_terminal_group=[0,1,2],
            alternatives=[list(s) for c in bank for s in c.supports if 1<len(s)<3],
            conditional_on_declared_candidate_class=True,
            interpretation='common full terminal group and singleton tips; no resolved pair among the three terminals; hidden edge count and R/X remain undecided')
        records.append(item)
    answer=dict(records=records,seconds=perf_counter()-start,source_manifest=frozen,
                truth_used=False,old_scientific_sources_modified=False,
                rejection_stage='audit_only_not_activated',
                gate='uniform_AC_remainder_must_resolve_difficult_candidate_differences_before_activation')
    write(output,answer)
    print(dict(tasks=len(records),seconds=answer['seconds'],path=str(output)),flush=True)
    return answer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    process(parser.parse_args().directory)


if __name__=='__main__':main()
