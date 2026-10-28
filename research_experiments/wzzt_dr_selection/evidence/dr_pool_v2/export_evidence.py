"""Export aggregate DR matrix evidence without household trajectories; no solves."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')


def family(f, model):
    m=f['models'][model]
    return sorted([s for s,r,x in zip(m['support_labels'],m['r_values'],m['x_values']) if max(r,x)>1e-9], key=lambda s:(len(s),s))


def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--physical',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    s=read(args.run/'summary.json');p=read(args.run/'protocol.json');physical=read(args.physical/'manifest.json')
    assert s['status']=='complete' and s['evaluated_cells']==48 and s['attempted_unique_fits']==42
    assert s['source_inputs_unchanged']
    fits={r['key']:read(args.run/'fit_cache'/r['key']/'result.json') for r in s['fit_cache_records']}
    rows=[]
    for r in s['records']:
        record={k:r[k] for k in ('scenario_id','noise','response_mode','response_weight','strategy','N','status','observation_key','evaluation','fit_cache')}
        record['periods']={phase:{'payment_nok':v['payment_nok'],'response_valid':v['response_audit']['valid'],'budget_valid':v['response_audit']['budget_valid'],'AC_valid':v['physical_checks']['usable_ac_solution'],'equations_pass':v['physical_checks']['electrical_equations_pass'],'voltage_pass':v['physical_checks']['voltage_limits_pass'],'thermal_limits_checked':v['physical_checks']['thermal_limits_checked']} for phase,v in r['periods'].items()}
        f=fits[r['observation_key']]
        record['active_supports']={m:family(f,m) for m in f['models']}
        record['selected_models']={m:{k:v[k] for k in ('support_labels','r_values','x_values','train_mae','validation_mae','selection_validation_mae','origin')} for m,v in f['models'].items()}
        rows.append(record)
    lookup={(r['scenario_id'],r['N'],r['strategy']):r for r in rows};pairs=[]
    for r in rows:
        if r['strategy']=='no_reward':continue
        for reference in ('no_reward','fixed_same_menu_first_node'):
            if reference==r['strategy'] or (reference=='fixed_same_menu_first_node' and r['strategy']!='planned_d_opt'):continue
            base=lookup[(r['scenario_id'],r['N'],reference)];f=fits[r['observation_key']];bf=fits[base['observation_key']]
            for m in f['models']:
                mm=r['evaluation']['models'][m]['final'];bb=base['evaluation']['models'][m]['final']
                pairs.append({'scenario_id':r['scenario_id'],'N':r['N'],'strategy':r['strategy'],'reference':reference,'model':m,'same_effective_family':r['active_supports'][m]==base['active_supports'][m],'internal_f1_delta':mm['internal_f1']-bb['internal_f1'],'root_matched_delta':mm['root_matched']-bb['root_matched'],'complete_effective_family_delta':int(mm['complete_effective_family'])-int(bb['complete_effective_family']),'same_pool':f['pool_support_labels']==bf['pool_support_labels'],'pool_size':f['pool_size'],'reference_pool_size':bf['pool_size'],'validation_mae_delta':r['evaluation']['models'][m]['validation_mae']-base['evaluation']['models'][m]['validation_mae']})
    fit_rows=[]
    for key,f in fits.items():
        fit_rows.append({'key':key,'result_sha256':sha(args.run/'fit_cache'/key/'result.json'),'elapsed_seconds':f['elapsed_seconds'],'pool_size':f['pool_size'],'LP_count':len(f['LP_solves']),'LP_statuses':dict(Counter(str(x['status']) for x in f['LP_solves'])),'search_stop_statuses':f['search_stop_statuses'],'search_accepted_steps':{k:len(v['path'])-1 for k,v in f['searches'].items()},'environment':f['environment']})
    aggregate={k:v for k,v in s.items() if k not in ('records','fit_cache_records')}
    aggregate.update(unique_fit_seconds=sum(r['elapsed_seconds'] for r in fit_rows),LP_count=sum(r['LP_count'] for r in fit_rows),physical_new_simulated_hours=physical['new_simulated_hours'],physical_conditions=16,paired_comparisons=len(pairs),all_paired_effective_families_identical=all(r['same_effective_family'] for r in pairs),all_paired_topology_metrics_equal=all(r['internal_f1_delta']==r['root_matched_delta']==r['complete_effective_family_delta']==0 for r in pairs),pair_pool_change_count=sum(not r['same_pool'] for r in pairs),pair_pool_change_note='same pool comparison repeated for three models; not independent count')
    write(out/'matrix_summary.json',aggregate);write(out/'cell_results.json',rows);write(out/'paired_results.json',pairs);write(out/'fit_summaries.json',fit_rows)
    write(out/'protocol.json',p);write(out/'physical_protocol.json',read(args.physical/'protocol.json'));write(out/'physical_manifest.json',physical)
    source_files=[args.run/'summary.json',args.run/'protocol.json',args.physical/'protocol.json',args.physical/'manifest.json',Path(__file__)]
    write(out/'evidence_manifest.json',{'source_sha256':{str(x.resolve()):sha(x) for x in source_files},'export_files_sha256':{x.name:sha(x) for x in out.glob('*.json') if x.name!='evidence_manifest.json'},'excluded':'raw household and aggregate trajectories, forecast arrays, private bundles, credentials','interpretation':'conditional finite empirical response support; no population probability or topology certificate'})
    print(json.dumps({k:aggregate[k] for k in ('evaluated_cells','completed_unique_fits','LP_count','unique_fit_seconds','all_paired_effective_families_identical','all_paired_topology_metrics_equal','pair_pool_change_count')},indent=2))


if __name__=='__main__':main()
