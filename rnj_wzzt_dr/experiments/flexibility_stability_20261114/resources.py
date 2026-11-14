"""Freeze registered P expansion and unchanged Q36 capability, without solves.

The added sites use registration and frozen historical nearest-pair P-column
strength only. The source forecasts are read solely for an analytic P-only
outer-bound diagnostic; they do not choose sites or capacities.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from ... import io


VARIANTS=('original_p10_q36','expanded_p10_q36')
FRACTION=.1
FIELDS=(('pv','PV_systems_kW',.3),('ev','EV_chargers_kW',.1),('thermal','hot_water_tanks_kW',.15))


def checked(reference):
    if io.ref(reference['path'])['sha256']!=reference['sha256']:
        raise ValueError('Changed sealed resource input: '+reference['path'])
    return Path(reference['path'])


def expanded_catalog(context,score,original):
    labels=score['labels'];H=np.asarray([row['H'] for row in score['parameters']])
    selected=sorted(range(len(H)),key=lambda i:(score['parameters'][i]['history_sse'],i))[:2]
    if len(selected)!=2:raise ValueError('Two frozen historical competitors required')
    D=H[selected[0]]-H[selected[1]]
    registered={int(row['feeder']):row for row in context['resource_metadata_at_history_start']}
    mapping={int(row['label']):row for row in context['node_mapping']}
    original_nodes={int(row['node_label']) for row in original}
    def capacity(node,field):
        row=mapping[node]
        return float(registered[int(row['feeder_id'])].get(field,0.) or 0.)*float(row['scale'])
    eligible=[node for node in labels if node not in original_nodes
              and any(capacity(node,field)>0. for _,field,_ in FIELDS)]
    ranking=sorted(eligible,key=lambda node:(-np.linalg.norm(D[:,labels.index(node)]),node))
    added_nodes=ranking[:3]
    if set(added_nodes)!={301,303,304}:
        raise ValueError('Frozen registration/P-response rule no longer yields locked sites301/303/304')
    extra=[]
    for node in added_nodes:
        info=mapping[node];registration=registered[int(info['feeder_id'])]
        for kind,field,price in FIELDS:
            scaled=capacity(node,field)
            if scaled<=0:continue
            extra.append(dict(kind=kind,node_label=int(node),feeder_id=int(info['feeder_id']),
                metadata_field=field,registration_date=registration['date'],
                registered_kw=float(registration[field]),source_to_network_scale=float(info['scale']),
                scaled_registered_kw=scaled,contract_kw=FRACTION*scaled,id=kind+'_'+str(node),q_over_p=0.,
                fee_positive_yuan_per_kwh=price,fee_negative_yuan_per_kwh=price))
    if len(extra)!=7:raise ValueError('Expected exactly seven registered additional P contracts')
    rows=deepcopy(original)+extra
    audit=dict(representative_pair=selected,
        pair_selection='Lowest two frozen historical SSE representatives, with index ties; no truth label used',
        added_nodes=added_nodes,
        location_rule='Among not-yet-enrolled nodes with positive PV/EV/hot-water registration, select three largest frozen P response-difference column norms',
        ranked_eligible_nodes=[dict(node=node,P_column_norm=float(np.linalg.norm(D[:,labels.index(node)]))) for node in ranking],
        added_resources=extra,original_resource_count=len(original),expanded_resource_count=len(rows),
        original_contract_kw=float(sum(row['contract_kw'] for row in original)),
        expanded_contract_kw=float(sum(row['contract_kw'] for row in rows)),
        original_nonPV_contract_kw=float(sum(row['contract_kw'] for row in original if row['kind']!='pv')),
        expanded_nonPV_contract_kw=float(sum(row['contract_kw'] for row in rows if row['kind']!='pv')),
        units='Registered kW are first multiplied by each source-to-network node scale; contract is10% of that scaled capacity',
        PV_rule='Original10--14 minimum-quarter-hour net-export proxy retained; no assumed extra PV generation',
        energy_rule='EV18--07, zero daily delta energy,2h buffer; hot water rho=.98,1h buffer,terminal restoration')
    return rows,audit


def p_box_upper(score,forecast,resources,fraction):
    labels=score['labels'];n=len(labels);H=np.asarray([row['H'] for row in score['parameters']])
    pair=sorted(range(len(H)),key=lambda i:(score['parameters'][i]['history_sse'],i))[:2]
    D=H[pair[0]]-H[pair[1]];sigma2=float(score['sigma2']);power=np.asarray(forecast)
    d0=power@D.T;clock=(8+np.arange(24)+.5)%24
    export=np.maximum(0.,-power[:,:n]*200.).reshape(24,4,n).min(1)
    lower=np.zeros((24,len(resources)));upper=lower.copy();effect=np.zeros((n,len(resources)))
    for j,row in enumerate(resources):
        node=labels.index(row['node_label']);cap=row['contract_kw']*fraction/FRACTION
        effect[:,j]=D[:,node]/200.
        if row['kind']=='pv':
            upper[:,j]=np.where((clock>=10)&(clock<14),np.minimum(cap,export[:,node]),0.)
        else:
            available=((clock>=18)|(clock<7)) if row['kind']=='ev' else np.ones(24,bool)
            lower[available,j]=-cap;upper[available,j]=cap
    low=d0+np.repeat(lower@np.maximum(effect,0).T+upper@np.minimum(effect,0).T,4,axis=0)
    high=d0+np.repeat(upper@np.maximum(effect,0).T+lower@np.minimum(effect,0).T,4,axis=0)
    J0=float(np.square(d0).sum()/(2*sigma2));bound=float(np.maximum(low*low,high*high).sum()/(2*sigma2))
    # The same buffered absolute gate, using both historical directions.
    from statistics import NormalDist
    z=NormalDist().inv_cdf(1-.1/(len(H)-1))
    h=-abs((score['parameters'][pair[1]]['history_sse']-score['parameters'][pair[0]]['history_sse'])/(2*sigma2))
    tie=max(1e-7,(score['cells']+96*n)*score['reference_mse']*score['validation_tie']/(2*sigma2))
    c=h-tie-1e-8;required=float(.5*(z+np.sqrt(z*z-2*c))**2)
    return dict(fraction=fraction,base_kw_kvar=200.,J0=J0,P_only_J_upper=bound,J_gain_upper=bound-J0,
        required_J=required,infeasible_even_in_outer_box=bound<required,
        scope='PureP and zeroQ. Interval outer bound ignores energy coupling and guards, so it can only overestimate attainable distance; no optimization or AC is run.')


def prepare(source_run,out):
    source_run=Path(source_run).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve previous resource specifications')
    if not (source_run/'complete.json').exists():raise ValueError('Completed source run required')
    protocol=io.read(source_run/'protocol.json')
    entries=[row for row in io.read(checked(protocol['inputs'])) if row['resource_variant']=='registered_q95_36']
    if len(entries)!=1:raise ValueError('Select the one fixed registeredQ36 input')
    configurations={};audits=[];bounds=[]
    for entry in entries:
        context=io.read(checked(entry['resource_context']));score=io.read(checked(entry['score']))
        identity=entry['original_id'];first=source_run/'fraction0.10'/entry['id']/'forecast00'
        config=io.read(first/'resource_case.json')
        original=[deepcopy(row) for row,unit in zip(config['resources'],config['action_units'],strict=True) if unit=='kW']
        q=deepcopy(entry['q_resources'])
        if len(original)!=7 or len(q)!=15 or abs(sum(row['capacity_kvar'] for row in q)-36.)>1e-12:
            raise ValueError('Expected original7P and unchanged15-node36kvar capability')
        expanded,audit=expanded_catalog(context,score,original)
        configurations[identity]={variant:dict(p_resources=deepcopy(rows),q_resources=deepcopy(q),
            physical_contract_fraction=FRACTION,original_P_count=7,
            Q_capability_role='Maximum offered capability; actual purchased daily reserve is a planning variable')
            for variant,rows in zip(VARIANTS,(original,expanded),strict=True)}
        audit.update(id=identity,score=entry['score'],resource_context=entry['resource_context'],
            original_resource_case=io.ref(first/'resource_case.json'),Q_resources_unchanged=True,
            original_P_prefix_unchanged=expanded[:7]==original,
            P_response_stream='773, original7 draws retained first, new7 appended',Q_response_stream='774, original15 draws/order unchanged',
            rotation_rule='P phases retain original prefix; Q phase indices always7+localQindex, including expanded P domain')
        audits.append(audit)
        for draw in range(protocol['forecasts']):
            folder=source_run/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            seal=io.read(folder/'seal.json');planner=checked(seal['planner'])
            with np.load(planner,allow_pickle=False) as z:forecast=z['forecast'].copy()
            for variant,rows in zip(VARIANTS,(original,expanded),strict=True):
                for fraction in (.1,.25):
                    bounds.append(dict(id=identity,forecast_index=draw,variant=variant,planner=seal['planner'],
                                       **p_box_upper(score,forecast,rows,fraction)))
    common=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=io.ref(__file__),
        source_run=io.ref(source_run/'complete.json'),source_protocol=io.ref(source_run/'protocol.json'),
        source_inputs=protocol['inputs'],physical_fraction=FRACTION,variants=list(VARIANTS),
        prices='Original activation prices unchanged. Availability price is set separately by the objective protocol.',
        baseline='No equipment baseline is added to measured netP/Q.',
        selection_inputs='Historical registration and frozen historical candidates only; no actual future voltage, true network or recovery label used',
        forecast_use='Only analytic reachability diagnostic after locations/capacities have been fixed',
        optimization_calls=0,AC_calls=0,identifier_refits=0)
    out.mkdir(parents=True)
    documents=dict(placements=dict(common,configurations_by_input=configurations),
        resource_audit=dict(common,rows=audits),p_only_bounds=dict(common,rows=bounds))
    refs={}
    for name,document in documents.items():
        path=out/(name+'.json');io.write(path,document);refs[name]=io.ref(path)
    completion=dict(status='complete',files=refs,source_unchanged=io.ref(__file__)['sha256']==common['source']['sha256'],
        checks=dict(unchanged_original_P_prefix=True,unchanged_Q36=True,added7_registered_resources=True,
                    no_new_optimization_or_AC=True))
    io.write(out/'complete.json',completion)
    return completion


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=Path('outputs/flexibility_resource_paired_20261114_v1'))
    parser.add_argument('--out',type=Path,default=Path('outputs/flexibility_stability_resources_20261114_v1'))
    args=parser.parse_args();print(prepare(args.source_run,args.out))
