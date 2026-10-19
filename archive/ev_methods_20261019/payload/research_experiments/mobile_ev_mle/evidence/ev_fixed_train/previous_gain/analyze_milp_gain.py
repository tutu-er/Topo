"""Read-only audit of saved frozen 12-history MILP evaluation; no fits/solves."""
from __future__ import annotations
import csv, gzip, hashlib, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path

OLD = Path(r'D:\0-github_workspace\Topo\outputs\ev_compact_stage_20261015')
BASE = OLD/'research_experiments/mobile_ev_mle'
E = BASE/'evidence/ev_compact'
OUT = Path(__file__).resolve().parent
MEAN, CAP = 'refined_v2g_mobile', 'capped_v2g_mobile'
METHODS = (MEAN, CAP)
inputs = {}

def raw(path):
    data = path.read_bytes()
    inputs[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return data

def rows(path):
    return list(csv.DictReader(raw(path).decode('utf-8-sig').splitlines()))

def num(x):
    return None if x == '' or x is None else float(x)

def yes(x):
    return x is True or x == 'True'

def mean(xs):
    return statistics.mean(xs)

def signs(xs):
    return dict(positive=sum(x>1e-7 for x in xs), zero=sum(abs(x)<=1e-7 for x in xs),
                negative=sum(x < -1e-7 for x in xs))

def write_csv(name, records):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)

methodrows=rows(E/'analysis_evaluation_v2/evaluation/method_rows.csv')
masters=rows(E/'analysis_evaluation_v2/evaluation/master_rows.csv')
tasks=rows(E/'analysis_evaluation_v2/evaluation/task_rows.csv')
locations=rows(E/'location_evaluation_v1/paired_locations.csv')
channelrows=rows(E/'location_evaluation_v1/candidate_channel_costs.csv')
channels={(int(r['seed']),r['method'],r['candidate']):r for r in channelrows
    if r['variant']=='original' and r['kind']=='noisy'}
lookup={(int(r['seed']),r['method']):r for r in methodrows}
location={(int(r['seed']),r['method']):r for r in locations}
tasklookup=defaultdict(list)
for r in tasks:
    if r['stage']=='selected' and r['method'] in METHODS:
        tasklookup[int(r['seed']),r['method']].append(r)
seeds=list(range(989201,989213))
assert len(methodrows)==12*11
assert len(locations)==12*2
assert all(len(tasklookup[s,m])==72 for s in seeds for m in METHODS)

# Every input is an existing export or a verified lossless compact trial.
# These reads inspect recorded curves only; there is no simulator/selector call.
actions={}
for s in seeds:
    trial=json.loads(gzip.decompress(raw(OLD/f'restored_evaluation_stage6_evaluation_slim/trial_{s}.slim.json.gz')))
    assert trial['seed']==s and trial['policy_id']==lookup[s,MEAN]['policy_id']
    bymethod={r['method']:r for r in trial['rows']}
    for m in METHODS:
        row=bymethod[m];op=trial['selection']['methods'][m]['operation']
        assert op['feasible'] and not trial['selection']['random_used_by_primary']
        powers=row['power_kw']
        assert powers==op['u_kw'] or max(abs(x-y) for a,b in zip(powers,op['u_kw']) for x,y in zip(a,b)) <1e-7
        moves=[a for a in op['route'] if a['kind']=='travel']
        for t,p in enumerate(powers):
            connected=[j for j,c in enumerate(op['connected'][t]) if c]
            activities=[a['kind'] for a in op['route'] if a['departure']<=t<a['arrival']]
            actions[s,m,t]=dict(seed=s,method=m,time_slot=t,slot_minutes=15,
                u_site0_kw=p[0],u_site1_kw=p[1],u_site2_kw=p[2],net_kw=sum(p),
                connected_site=connected[0] if connected else '',activity=activities[0] if activities else '',
                initial_energy_kwh=op['energy_kwh'][t],final_energy_kwh=op['energy_kwh'][t+1],
                destination=moves[0]['destination'] if moves else '',
                travel_departure_slot=moves[0]['departure'] if moves else '')

comparison=[]
tasksummary=[]
channelsummary=[]
for s in seeds:
    a,b=lookup[s,MEAN],lookup[s,CAP]
    la,lb=location[s,MEAN],location[s,CAP]
    dma=num(b['actual_M'])-num(a['actual_M'])
    ds=num(lb['delta_M'])-num(la['delta_M'])
    dc=num(lb['remap_noisy_M'])-num(la['remap_noisy_M'])
    assert abs(dma-ds-dc)<1e-7
    assert a['initial_actions_sha256']==b['initial_actions_sha256']
    assert a['configured_exchange_budget']==b['configured_exchange_budget']
    row=dict(seed=s,family=a['family'],mean_actual_M=num(a['actual_M']),cap_actual_M=num(b['actual_M']),
        cap_minus_mean_M=dma,mean_site_effect=num(la['delta_M']),cap_site_effect=num(lb['delta_M']),
        difference_of_site_effects=ds,cap_minus_mean_M_both_at_site0=dc,
        decomposition_error=dma-ds-dc,mean_predicted_mean=num(a['predicted_mean']),
        cap_predicted_mean=num(b['predicted_mean']),mean_predicted_cap20=num(a['predicted_cap20_mean']),
        cap_predicted_cap20=num(b['predicted_cap20_mean']),
        mean_predicted_worst=num(a['predicted_worst']),cap_predicted_worst=num(b['predicted_worst']),
        mean_selected_origin=a['selected_origin'],cap_selected_origin=b['selected_origin'],
        mean_is_new_MILP=yes(a['selected_is_new_MILP_proposal']),cap_is_new_MILP=yes(b['selected_is_new_MILP_proposal']),
        mean_proxy_completed=yes(a['proxy_completed']),cap_proxy_completed=yes(b['proxy_completed']),
        mean_proxy_gap=num(a['proxy_gap']),cap_proxy_gap=num(b['proxy_gap']),
        mean_grid_net_kwh=num(a['grid_net_kwh']),cap_grid_net_kwh=num(b['grid_net_kwh']),
        mean_grid_throughput_kwh=num(a['grid_throughput_kwh']),cap_grid_throughput_kwh=num(b['grid_throughput_kwh']),
        mean_moves=int(a['moves']),cap_moves=int(b['moves']),initials_and_budget_match=True)
    comparison.append(row)
    for m in METHODS:
        ts=tasklookup[s,m];w=[num(x['weight']) for x in ts];d=[num(x['added']) for x in ts]
        assert abs(sum(w)-1)<1e-10
        wm=sum(x*y for x,y in zip(w,d));wc=sum(x*min(y,20) for x,y in zip(w,d))
        r=lookup[s,m]
        assert abs(wm-num(r['predicted_mean']))<1e-7
        assert abs(wc-num(r['predicted_cap20_mean']))<1e-7
        tasksummary.append(dict(seed=s,method=m,weighted_mean=wm,weighted_cap20=wc,
            worst=min(d),maximum=max(d),tasks_nonpositive=sum(x<=0 for x in d),
            tasks_at_or_below_resolution=sum(x<=.01 for x in d),
            tasks_below20=sum(x<20 for x in d),tasks_above20=sum(x>20 for x in d),
            saturated_weight=sum(x for x,y in zip(w,d) if y>20),
            easy_task_excess_above20=sum(x*max(y-20,0) for x,y in zip(w,d)),
            positive_mass=sum(x*max(y,0) for x,y in zip(w,d))))
        a=channels[s,m,r['true_name']];b=channels[s,m,r['nearest_wrong']]
        component={k:num(b[k])-num(a[k]) for k in ('historical_SSE','future_p_SSE',
            'future_q_SSE','future_v_SSE','future_master_p_SSE','future_master_q_SSE')}
        assert abs(sum(component.values())-num(r['actual_M']))<1e-7
        channelsummary.append(dict(seed=s,method=m,true_candidate=r['true_name'],
            nearest_wrong=r['nearest_wrong'],actual_M=num(r['actual_M']),**component))

summary=dict(schema='saved_milp_gain_audit_v1',no_fit_no_solver=True,
    policy_id=lookup[seeds[0],MEAN]['policy_id'],histories=12,
    identical_initial_actions_and_budget=all(r['initials_and_budget_match'] for r in comparison),
    cap_minus_mean=dict(mean=mean(r['cap_minus_mean_M'] for r in comparison),
        median=statistics.median(r['cap_minus_mean_M'] for r in comparison),
        **signs([r['cap_minus_mean_M'] for r in comparison])),
    spatial_decomposition=dict(
        cap_mean_site_effect=mean(r['cap_site_effect'] for r in comparison),
        mean_mean_site_effect=mean(r['mean_site_effect'] for r in comparison),
        difference_of_site_effects=mean(r['difference_of_site_effects'] for r in comparison),
        remapped_curve_difference=mean(r['cap_minus_mean_M_both_at_site0'] for r in comparison),
        max_identity_error=max(abs(r['decomposition_error']) for r in comparison),
        scope='Algebraic attained-score decomposition with fixed-waveform counterfactual; not equal-service causal attribution'),
    methods={})
for m in METHODS:
    rr=[lookup[s,m] for s in seeds];tt=[r for r in tasksummary if r['method']==m]
    summary['methods'][m]=dict(
        mean_actual_M=mean(num(r['actual_M']) for r in rr),
        mean_predicted_mean=mean(num(r['predicted_mean']) for r in rr),
        mean_predicted_cap20=mean(num(r['predicted_cap20_mean']) for r in rr),
        mean_predicted_worst=mean(num(r['predicted_worst']) for r in rr),
        mean_nonpositive_task_count=mean(r['tasks_nonpositive'] for r in tt),
        mean_resolution_unresolved_task_count=mean(r['tasks_at_or_below_resolution'] for r in tt),
        mean_saturated_weight=mean(r['saturated_weight'] for r in tt),
        mean_easy_excess=mean(r['easy_task_excess_above20'] for r in tt),
        new_MILP_count=sum(yes(r['selected_is_new_MILP_proposal']) for r in rr),
        inherited_count=sum(yes(r['inherited_or_initial_retained']) for r in rr),
        MILP_lineage_count=sum(yes(r['source_lineage_from_MILP']) for r in rr),
        proxy_completed_count=sum(yes(r['proxy_completed']) for r in rr),
        proxy_gap_median=statistics.median(num(r['proxy_gap']) for r in rr),
        raw_AC_proxy_gate_pass_count=sum(yes(r['raw_proxy_gate']) for r in rr),
        recentered_count=sum(yes(r['recentered']) for r in rr),
        selected_origins=dict(Counter(r['selected_origin'] for r in rr)),
        mean_grid_net_kwh=mean(num(r['grid_net_kwh']) for r in rr),
        mean_grid_throughput_kwh=mean(num(r['grid_throughput_kwh']) for r in rr),
        discharge_histories=sum(any(actions[s,m,t]['net_kw'] < -1e-7 for t in range(6)) for s in seeds),
        moved_histories=sum(int(r['moves'])>0 for r in rr),
        slot_counts=dict(positive=sum(actions[s,m,t]['net_kw']>1e-7 for s in seeds for t in range(6)),
            negative=sum(actions[s,m,t]['net_kw']< -1e-7 for s in seeds for t in range(6)),
            zero=sum(abs(actions[s,m,t]['net_kw'])<=1e-7 for s in seeds for t in range(6)),
            absolute_hardware_command_bound=sum(abs(abs(actions[s,m,t]['net_kw'])-6/1.05)<1e-7 for s in seeds for t in range(6)),
            interior_nonzero=sum(1e-7<abs(actions[s,m,t]['net_kw'])<6/1.05-1e-7 for s in seeds for t in range(6))),
        mean_margin_components={k:mean(r[k] for r in channelsummary if r['method']==m) for k in
            ('historical_SSE','future_p_SSE','future_q_SSE','future_v_SSE','future_master_p_SSE','future_master_q_SSE')},
        destinations=dict(Counter(str(actions[s,m,0]['destination']) for s in seeds)),
        travel_departure_slots=dict(Counter(str(actions[s,m,0]['travel_departure_slot']) for s in seeds)))
summary['master_sizes']={}
for m in sorted(set(r['method'] for r in masters)):
    rr=[r for r in masters if r['method']==m]
    summary['master_sizes'][m]={k:sorted({int(r[k]) for r in rr}) for k in
        ('binary','operation_binary','shared_square_binary','residual_binary','product_binary')}

for name in ('ev_compact_study.py','ev_mobile_service_domain.py','ev_original_study.py',
             'ev_shared_square_milp.py','ev_original_shared_square_math.md','ev_compact_analysis.py'):
    raw(BASE/name)
raw(E/'evaluation_ensemble_v2/freeze.json')
write_csv('cap_mean_history_pairs.csv',comparison)
write_csv('task_distribution.csv',tasksummary)
write_csv('saved_primary_actions.csv',list(actions.values()))
write_csv('margin_channel_decomposition.csv',channelsummary)
(OUT/'milp_gain_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest=dict(analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    inputs=inputs,outputs={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    for p in OUT.iterdir() if p.name in ('cap_mean_history_pairs.csv','task_distribution.csv',
        'saved_primary_actions.csv','margin_channel_decomposition.csv','milp_gain_summary.json',
        'milp_gain_analysis.md')},no_policy_change=True,no_optimization=True)
(OUT/'milp_gain_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
