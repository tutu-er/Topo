"""Export aggregate evidence from an existing run; never fit or plan again."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def export(run, out):
    summary=read(run/'summary.json');state=read(run/'run_state.json');plan=read(run/'plan.json')
    if state['status']!='complete':raise ValueError('run incomplete; do not publish a complete denominator')
    out.mkdir(parents=True,exist_ok=True)
    checks=[read(p) for p in sorted((run/'history').glob('*.checks.json'))]
    audit=[];diagnostics=[]
    with (run/'privatebundle.pkl').open('rb') as f:bundle=pickle.load(f)
    forecast=np.asarray(plan['forecast_p_kw'])
    baseline_error={role:{'mae_kw_per_terminal_hour':float(np.abs(forecast-p.to_numpy()).mean()),
        'rmse_kw_per_terminal_hour':float(np.sqrt(np.mean((forecast-p.to_numpy())**2)))} for role,p in bundle['actual_days'].items()}
    for s in summary['strategies']:
        directory=run/'strategies'/s['strategy'];record=read(directory/'result.json')
        checks.extend(p['physical_checks'] for p in record['periods'].values())
        row={'strategy':s['strategy'],'assignment':s['assignment'],'periods':{}}
        for role,data in record['periods'].items():
            with (directory/(role+'.pkl')).open('rb') as f:period=pickle.load(f)
            actual=np.asarray(period['response']['p_kw']);baseline=bundle['actual_days'][role].to_numpy()
            row['periods'][role]={'payment_nok':data['total_payment_nok'],
                'total_peak_kw':float(actual.sum(axis=1).max()),'total_energy_kwh':float(actual.sum()),
                'total_change_kwh_from_realized_no_response':float((actual-baseline).sum()),
                'contract_valid':data['response_audit']['valid'],
                'physical_checks':data['physical_checks']}
        audit.append(row)
        fit=read(directory/'wzzt.json')
        diagnostics.append({'strategy':s['strategy'],'fit_status':fit['status'],
            'config':fit['config'],'core_sha256':fit['core_sha256'],'training_time_rows':fit['training_time_rows'],
            'validation_time_rows':fit['validation_time_rows'],'summary':fit.get('summary'),
            'attempted_extensions':fit.get('attempted_extensions',[]),'events':fit.get('events',[])})
    physical={'scenario_count':len(checks),'hourly_case_count':sum(c['sample_count'] for c in checks),
        'converged_hourly_case_count':sum(c['converged_samples'] for c in checks),
        'nonconverged_hourly_case_count':sum(c['nonconverged_samples'] for c in checks),
        'all_electrical_equations_pass':all(c['electrical_equations_pass'] for c in checks),
        'voltage_violation_cells':sum(c['voltage_violation_cells'] for c in checks),
        'voltage_min_pu':min(c['physical_voltage_min_pu'] for c in checks),
        'voltage_max_pu':max(c['physical_voltage_max_pu'] for c in checks),
        'thermal_limits_checked':False,
        'maximum_system_p_balance_residual_pu':max(c['max_system_p_balance_residual_pu'] for c in checks),
        'maximum_distflow_residual_pu2':max(c['max_distflow_identity_error_pu2'] for c in checks)}
    compact_plan={'selected_assignment':plan['selected_assignment'],'objective':plan['objective'],
        'candidate_count':len(plan['candidates']),'feasible_candidate_count':sum(a['feasible'] for a in plan['candidates']),
        'candidates':[{k:a[k] for k in ('name','menus','score','feasible','payment_at_mean_response_nok','conservative_payment_bound_nok')} for a in plan['candidates']]}
    files=[run/'prepared_receipt.json',run/'privatebundle.pkl',run/'plan.json',run/'response_draws.json',run/'summary.json',run/'run_state.json']
    files.extend(sorted((run/'strategies').glob('*/*.json')))
    provenance={'local_run_directory':str(run),'run_fingerprint':state['run_fingerprint'],
        'created_at':state['created_at'],'completed_at':state['completed_at'],
        'environment':state['environment'],'provenance':state['provenance'],
        'retained_local_file_sha256':{str(p.relative_to(run)):digest(p) for p in files},
        'exporter_sha256':digest(Path(__file__)),'raw_household_data_exported':False}
    payloads={'run_summary.json':summary,'execution_audit.json':{'physical':physical,'strategies':audit,
        'baseline_forecast_error':baseline_error,'planning':compact_plan,
        'scope':'One paired implementation-check realization, not an independent price-efficacy experiment'},
        'solver_diagnostics.json':diagnostics,'provenance.json':provenance}
    for name,payload in payloads.items():
        (out/name).write_text(json.dumps(payload,ensure_ascii=True,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    print(json.dumps(physical,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();export(a.run_dir.resolve(),a.output_dir.resolve())
