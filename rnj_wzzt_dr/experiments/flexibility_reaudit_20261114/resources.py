"""Prepare broad pure-Q resource placements from historical registration only.

The existing two-site configuration is copied unchanged. New location sets use
all measured terminals with positive historical PV/battery registration, without
reading frozen H, true network parameters, forecasts, or future voltages.
Training Q is read only for a scale audit, never to alter service capacity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path, expected=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Sealed source changed: {path}')
    return dict(path=str(path), sha256=digest)


def checked(ref):
    return reference(ref['path'], ref['sha256'])


def q_contract(label, capacity):
    return dict(id=f'svg_{int(label)}', node_label=int(label), capacity_kvar=float(capacity),
        fee_positive_yuan_per_kvarh=.05, fee_negative_yuan_per_kvarh=.05,
        reservation_fee_yuan_per_kvar_day=0.,
        kind='new_independent_bidirectional_Q_contract',
        capacity_origin='Explicit synthetic reserved compensator capacity; registration kW establishes location only',
        price_origin='Illustrative common activation price, not a measured market price')


def audit_contracts(resources, labels):
    nodes=[int(r['node_label']) for r in resources]
    if len(set(nodes)) != len(nodes) or not set(nodes) <= set(labels):
        raise ValueError('Every Q device must have a unique measured terminal')
    capacities=np.asarray([r['capacity_kvar'] for r in resources],float)
    if not len(capacities) or not np.all(np.isfinite(capacities)) or np.any(capacities <= 0):
        raise ValueError('Expected positive finite Q capacities')
    if any(r['fee_positive_yuan_per_kvarh'] != .05 or r['fee_negative_yuan_per_kvarh'] != .05
           or r['reservation_fee_yuan_per_kvar_day'] != 0. for r in resources):
        raise ValueError('Unequal pricing would confound the placement comparison')
    return dict(node_count=len(nodes), node_labels=nodes, measured_node_count=len(labels),
        coverage_fraction=len(nodes)/len(labels), total_contracted_kvar=float(capacities.sum()),
        capacity_per_node_range_kvar=[float(capacities.min()),float(capacities.max())],
        full_activation_24h_fee_yuan=float(capacities.sum()*24*.05),
        fee_scope='Illustrative activation only; excludes hardware, installation, fixed maintenance, and capacity reservation')


def prepare_cohort(run_directory, previous_placement, cohort, base_kw=200.):
    input_ref=reference(Path(run_directory)/'inputs.json')
    legacy_ref=reference(previous_placement)
    entries=read(input_ref['path']); legacy=read(legacy_ref['path'])['configurations_by_input']
    configurations, standby, originals, audits={},{},{},[]
    for entry in entries:
        identifier=entry['id']
        context_ref=checked(entry['resource_context'])
        train_ref=checked(entry['files']['train.npz'])
        context=read(context_ref['path'])
        labels=[int(r['label']) for r in context['node_mapping']]
        with np.load(train_ref['path'],allow_pickle=False) as z:
            train_labels=list(map(int,z['labels']))
            q=z['Q'].copy()*base_kw  # no Y, V, future P/Q, or true network is read
        if train_labels != labels or q.shape != (192,len(labels)) or not np.isfinite(q).all():
            raise ValueError('Expected matching finite 192-point historical Q observations')
        registrations={int(r['feeder']):r for r in context['resource_metadata_at_history_start']}
        nodes=[]; eligible=[]
        for k,mapping in enumerate(context['node_mapping']):
            registered=registrations[int(mapping['feeder_id'])]
            pv=float(registered.get('PV_systems_kW',0.) or 0.)
            battery=float(registered.get('batteries_kW',0.) or 0.)
            scale=float(mapping['scale'])
            if min(pv,battery,scale) < 0 or not np.isfinite([pv,battery,scale]).all():
                raise ValueError('Invalid registration scale or capacity')
            qualifies=(pv+battery)*scale > 0.
            if qualifies:
                eligible.append(labels[k])
            nodes.append(dict(node_label=labels[k], feeder_id=int(mapping['feeder_id']),
                registration_date=registered['date'], source_to_network_scale=scale,
                scaled_PV_registered_kw=pv*scale,scaled_battery_registered_kw=battery*scale,
                registered_location_eligible=qualifies,
                historical_Q_mean_kvar=float(q[:,k].mean()),
                historical_Q_std_kvar=float(q[:,k].std(ddof=1)),
                historical_Q_RMS_kvar=float(np.sqrt(np.mean(q[:,k]**2))),
                historical_Q_min_kvar=float(q[:,k].min()),historical_Q_max_kvar=float(q[:,k].max()),
                historical_Q_max_abs_kvar=float(np.abs(q[:,k]).max())))
        if not eligible:
            raise ValueError('No registered PV/battery locations')
        preserved=deepcopy(legacy[identifier]['eligible_sites'])
        original_nodes=[int(r['node_label']) for r in preserved]
        if not set(original_nodes) <= set(eligible):
            raise ValueError('Original eligible sites are outside positive registrations')
        ordered_eligible=original_nodes+sorted(set(eligible)-set(original_nodes))
        ordered_all=original_nodes+sorted(set(labels)-set(original_nodes))
        old_by_node={int(r['node_label']):r for r in preserved}
        def resources_for(ordered_nodes,capacity):
            result=[]
            for label in ordered_nodes:
                resource=deepcopy(old_by_node[label]) if label in old_by_node else q_contract(label,capacity(label))
                resource['capacity_kvar']=float(capacity(label))
                result.append(resource)
            return result
        configurations[identifier]=dict(eligible_sites=preserved,
            registered_equal10=resources_for(ordered_eligible,lambda label:10./len(eligible)),
            registered_plus1=resources_for(ordered_eligible,lambda label:old_by_node[label]['capacity_kvar'] if label in old_by_node else 1.))
        standby[identifier]=dict(registered_2each=resources_for(ordered_eligible,lambda label:2.),
            all_measured_1each=resources_for(ordered_all,lambda label:1.))
        originals[identifier]=deepcopy(legacy[identifier])
        schemes={name:audit_contracts(resources,labels) for name,resources in configurations[identifier].items()}
        standby_audit={name:audit_contracts(resources,labels) for name,resources in standby[identifier].items()}
        if abs(schemes['eligible_sites']['total_contracted_kvar']-10.) > 1e-12 or abs(schemes['registered_equal10']['total_contracted_kvar']-10.) > 1e-12:
            raise ArithmeticError('Same-total capacity control failed')
        if configurations[identifier]['eligible_sites'] != legacy[identifier]['eligible_sites']:
            raise ArithmeticError('Existing eligible_sites configuration changed')
        if configurations[identifier]['registered_plus1'][:len(preserved)] != preserved:
            raise ArithmeticError('Expanded domain did not retain the original resource dictionaries')
        for resources in [*configurations[identifier].values(),*standby[identifier].values()]:
            if [r['id'] for r in resources[:len(preserved)]] != [r['id'] for r in preserved]:
                raise ArithmeticError('Resource order would alter paired original-device response streams')
        for node in nodes:
            k=labels.index(node['node_label'])
            scale_audit={}
            for name,resources in {**configurations[identifier],**standby[identifier]}.items():
                capacity=next((float(r['capacity_kvar']) for r in resources if r['node_label']==node['node_label']),0.)
                sd=node['historical_Q_std_kvar'];peak=node['historical_Q_max_abs_kvar']
                scale_audit[name]=dict(capacity_kvar=capacity,
                    capacity_over_historical_std=capacity/sd if sd > 1e-12 else None,
                    capacity_over_historical_max_abs=capacity/peak if peak > 1e-12 else None,
                    historical_sample_plus_contract_min_kvar=float(q[:,k].min()-capacity),
                    historical_sample_plus_contract_max_kvar=float(q[:,k].max()+capacity))
            node['contract_scale_audit']=scale_audit
        audits.append(dict(id=identifier,cohort=cohort,resource_context=context_ref,training_Q_source=train_ref,
            node_count=len(labels),eligible_node_labels=eligible,
            unregistered_node_labels=[label for label in labels if label not in eligible],
            schemes=schemes,standby=standby_audit,nodes=nodes,
            resource_array_order=ordered_eligible,original_Q_ids=[r['id'] for r in preserved],
            original_plan_embedding=dict(target_variant='registered_plus1',original_Q_columns=list(range(len(preserved))),
                appended_Q_columns_set_to_zero=True,original_resource_dictionaries_unchanged=True,
                original_feasible_domain_contained=True),
            training_only=True,training_Q_does_not_select_or_resize_resources=True,
            scale_scope='Historical extrema are observed sample extrema, not future bounds or converter ratings'))
    common=dict(cohort=cohort,created_utc=datetime.now(timezone.utc).isoformat(),source=reference(__file__),
        inputs=input_ref,previous_placements=legacy_ref,base_kw=base_kw,
        P_policy='Unchanged original P contracts, participation, clock windows, costs, and response streams',
        Q_policy='Independent continuous +/- capacity, every hour; no extra load/device baseline',
        Q_array_order='Original eligible_sites devices first, preserving IDs/order; remaining node labels ascending. Both broad variants share this order for response component 774.',
        capacity_origin='Synthetic new or explicitly procured independent Q reserves; kW registration gives locations only',
        price_yuan_per_kvarh=.05,reservation_price_yuan_per_kvar_day=0.,
        outcome_data_used=False,forecast_data_used=False,true_network_used=False)
    return dict(formal=dict(common,configurations_by_input=configurations,
                           variants=['eligible_sites','registered_equal10','registered_plus1']),
                standby=dict(common,configurations_by_input=standby,
                    status='Prepared alternative only; excluded from the three-variant comparison',
                    extra_installation='all_measured_1each includes nodes without PV/battery registration; registered_2each changes original 5-kvar limits to 2 kvar. Both are prepared alternatives only'),
                legacy=dict(common,configurations_by_input=originals,status='Reference only, unchanged original_sites and eligible_sites'),
                audit=dict(common,rows=audits))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run',type=Path,default=Path('outputs/measured_pq_dfs_main_20261114_v1'))
    parser.add_argument('--enriched-run',type=Path,default=Path('outputs/measured_pq_dfs_enriched_20261114_v1'))
    parser.add_argument('--main-placements',type=Path,default=Path('outputs/reactive_flex_placement_20261114_v2/main_configurations.json'))
    parser.add_argument('--enriched-placements',type=Path,default=Path('outputs/reactive_flex_placement_20261114_v2/enriched_configurations.json'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cohorts={name:prepare_cohort(run,old,name) for name,run,old in [
        ('main',args.main_run,args.main_placements),('enriched',args.enriched_run,args.enriched_placements)]}
    args.out.mkdir(parents=True,exist_ok=False)
    files={}
    for cohort,documents in cohorts.items():
        for kind,data in documents.items():
            destination=args.out/f'{cohort}_{kind}.json'
            with destination.open('x',encoding='utf-8') as stream:
                json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False)
                stream.write('\n')
            files[f'{cohort}_{kind}']=reference(destination)
    receipt=dict(status='complete',source=reference(__file__),files=files,
        checks=dict(original_eligible_sites_unchanged=True,all_sources_hashed=True,
            distinct_measured_locations=True,same_total_10kvar=True,
            no_new_day_data=True,no_Q_capacity_inferred_from_PV_kw=True),
        input_count=sum(len(c['audit']['rows']) for c in cohorts.values()))
    with (args.out/'complete.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(complete=reference(args.out/'complete.json'),
        enriched=cohorts['enriched']['audit']['rows'][0]['schemes'],
        standby=cohorts['enriched']['audit']['rows'][0]['standby'])))


if __name__=='__main__':
    main()
