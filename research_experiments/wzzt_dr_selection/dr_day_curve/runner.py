"""One bounded historical-only daily-curve pilot, with resumable case records.

No true graph, prior test observation, simulator or fitter is opened. The
future voltage law is a declared conditional hypothesis checked separately.
"""
from __future__ import annotations

import argparse,hashlib,json,os,pickle,platform,sys,time,traceback,zipfile
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .day_model import build_problem,draw_scenarios,scenario_summary,resource_audit,canonical_prices,alternative_forecast_summary
from .day_solver import solve
from .result_bridge import persist_and_adapt

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
PRIVATE=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
PREVIOUS=PRIVATE/'wzzt_newday_weighted_v1/run_v2'


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(item):
    p=Path(item['path'])
    if sha(p)!=item['sha256']:raise ValueError('frozen input drift: '+str(p))
    return p
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n');tmp.replace(p)
def load_pickle(item):
    with checked(item).open('rb') as f:return pickle.load(f)


def curve_from_plan(plan,resources):
    u=np.zeros((len(resources),96));positions={x['resource_id']:i for i,x in enumerate(resources)}
    for module,mean in zip(plan['modules'],plan['amplitudes']):
        for resource,h in module['profiles'].items():u[positions[resource]]+=float(mean)*np.asarray(h)
    return u


def feasible_scale(problem,curve):
    if resource_audit(problem,curve)['feasible']:return curve
    lo=0.;hi=1.
    for _ in range(45):
        mid=(lo+hi)/2
        if resource_audit(problem,mid*curve)['feasible']:lo=mid
        else:hi=mid
    return (lo*(1-1.e-8))*curve


def warm_curves(problem,old_plans,resources):
    from research_experiments.wzzt_dr_selection.dr_frozen_distance.design import contrast_shape
    curves=[curve_from_plan(v,resources) for v in old_plans.values()]
    C=np.asarray(problem['contrast']);gradients=[]
    for d in problem['training']:
        gradients.append((np.sign(d['base_bad_micro'])@C).T)
    gradients.append(np.mean(gradients,axis=0))
    for g in gradients:
        for slots in (8,16,24,32,48):
            curve=.8*np.array([contrast_shape(x,slots) for x in g])
            curves.append(feasible_scale(problem,curve))
    return curves


def prepare(output,seconds,cost_seconds,unit_gate):
    if unit_gate is None:raise ValueError('provide the passing28-test JUnit receipt with --unit-gate')
    gate=Path(unit_gate).resolve();suites=list(ET.parse(gate).getroot().iter('testsuite'))
    count=sum(int(x.get('tests',0)) for x in suites)
    failures=sum(int(x.get('failures',0))+int(x.get('errors',0)) for x in suites)
    if count<28 or failures:raise ValueError('passing28-test unit gate required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    old=read(PREVIOUS/'protocol.json');prices=read(PREVIOUS/'price_freeze.json')
    sources=sorted(Path(__file__).parent.glob('*.py'))
    dependencies=[REPO/'research_experiments/wzzt_dr_selection'/s for s in [
        'dr_frozen_distance/frozen_pool.py','dr_frozen_distance/design.py',
        'dr_newday_weighted/weighted_selection.py','dr_day_modules/day_schedule.py',
        'dr_conic_pricing/resources.py','dr_conic_pricing/model.py']]
    cases=[]
    for i,c in enumerate(old['cases']):
        declared=read(checked(c['case_protocol']))
        cases.append(dict(name=c['name'],case=c['case'],pool=c['pool'],history=declared['history_packet'],
            old_plans=prices['cases'][c['name']]['plans'],
            planning_seed=2026111101+100*i,training_evaluation_seed=2026111102+100*i,
            validation_evaluation_seed=2026111103+100*i))
    protocol=dict(status='frozen',created_utc=now(),id='fixed_24h_no_payment_cap_minimum_fee_margin_v2',
        old_protocol=ref(PREVIOUS/'protocol.json'),old_price_freeze=ref(PREVIOUS/'price_freeze.json'),
        cases=cases,source_sha256={str(p.resolve()):sha(p) for p in sources+dependencies},
        planning_samples=24,required_samples=22,model_holdout_samples=256,
        slots=96,slot_hours=.25,history_rows=[288,288],budget=None,
        payment_cap_enforced=False,fees='objective and reported output only; physical device limits retained',
        planning_seconds=seconds,cost_seconds=cost_seconds,
        model='all96 mean kW variables per qualified node; scalar daily quotes; exact frozen-model validation margin; bilinear cost',
        reference_candidate='refined by historical candidate id, not selected with true labels',
        forecast='3 paired historical training P/Q/Y blocks; Ynew=Yblock+mu_refined(DR); fixed block residual',
        reference_sensitivity='same curves evaluated under coarse response hypothesis, no parameter refits',
        target='Mafter>=.2*max(-Mzero,0)+.0001 micro; 120 percent of baseline shortfall',
        probability_scope='empirical conditional working-law excitation probability only; no topology recovery or field guarantee',
        case_rule='all5 existing conditions, including zero-benefit controls; no truth filtering',
        economic_scope='constant elasticity synthetic contract; c/h scale redundancy canonicalized; not behavioral price estimation',
        no_new_AC=True,no_new_fitting=True,no_test_future=True,
        phase_gate='exact numerical toy and resource tests before historical solve; preserve no-incumbent/timeout/infeasible separately',
        unit_gate=ref(gate),
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                         gurobipy=version('gurobipy'),scipy=version('scipy'),pandas=version('pandas')))
    write(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sources+dependencies:
            arc='dr_day_curve/'+p.name if p.parent==Path(__file__).parent else p.relative_to(REPO).as_posix()
            archive.write(p,arc)
    write(output/'run_lock.json',dict(status='prepared',pid=os.getpid(),created_utc=now(),protocol=ref(output/'protocol.json')))
    print(dict(stage='prepared',cases=5,protocol=sha(output/'protocol.json')),flush=True)


def run(output):
    root=Path(output).resolve();protocol=read(root/'protocol.json')
    for p,digest in protocol['source_sha256'].items():
        if sha(p)!=digest:raise ValueError('source changed after protocol freeze: '+p)
    write(root/'run_lock.json',dict(status='running',pid=os.getpid(),started_utc=now(),protocol=ref(root/'protocol.json')))
    progress=[]
    for c in protocol['cases']:
        folder=root/'cases'/c['name'];folder.mkdir(parents=True,exist_ok=True);result_file=folder/'result.json'
        if result_file.exists():
            result=read(result_file);progress.append(dict(condition=c['name'],status=result['status'],result=ref(result_file)));continue
        try:
            pool=read(checked(c['pool']));packet=load_pickle(c['history']);history=packet['history'] if 'history' in packet else packet
            old_plans={k:read(checked(v)) for k,v in c['old_plans'].items()}
            resources=old_plans['optimized']['resources']
            problem=build_problem(pool,history,resources)
            scenarios=draw_scenarios(problem,protocol['planning_samples'],c['planning_seed'],balanced=True)
            with (folder/'problem.pkl').open('wb') as stream:pickle.dump(dict(problem=problem,scenarios=scenarios),stream,protocol=4)
            write(folder/'problem_frozen.json',dict(status='frozen',created_utc=now(),problem=ref(folder/'problem.pkl'),
                old_pool=c['pool'],history=c['history'],planning_seed=c['planning_seed']))
            starts=warm_curves(problem,old_plans,resources)
            result=solve(problem,scenarios,seconds=protocol['planning_seconds'],cost_seconds=protocol['cost_seconds'],
                         warm_curves=starts,log_path=folder/'solver.log')
            result.update(condition=c['name'],forecast_scope=problem['forecast_scope'],source_protocol=ref(root/'protocol.json'))
            result=persist_and_adapt(folder,result,problem,scenarios,resources,pool['labels'],c['case'])
            result['baselines']={}
            for k,v in old_plans.items():
                u=curve_from_plan(v,resources)
                result['baselines'][k]=dict(training=scenario_summary(problem,scenarios,u),resource_audit=resource_audit(problem,u))
            write(result_file,result)
        except Exception as error:
            result=dict(status='failed',condition=c['name'],error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            write(result_file,result)
        progress.append(dict(condition=c['name'],status=result['status'],result=ref(result_file)))
        write(root/'progress.json',dict(status='running',rows=progress,planned=5))
        print(dict(stage='solved',done=len(progress),planned=5,condition=c['name'],status=result['status'],
            training_passed=result.get('training',{}).get('passed'),slack=result.get('training',{}).get('feasibility_slack_micro'),
            payment=result.get('resource_audit',{}).get('expected_payment')),flush=True)
    write(root/'price_freeze.json',dict(status='frozen',created_utc=now(),results=progress,
        protocol=ref(root/'protocol.json'),no_holdout_or_test_future_read=True))
    write(root/'planning_ended.json',dict(status='complete',rows=progress,planned=5,price_freeze=ref(root/'price_freeze.json')))
    write(root/'run_lock.json',dict(status='planning_complete',ended_utc=now(),pid=os.getpid()))


def evaluate(output):
    root=Path(output).resolve();p=read(root/'protocol.json');freeze=ref(root/'price_freeze.json');cases=[]
    for c in p['cases']:
        folder=root/'cases'/c['name'];r=read(folder/'result.json')
        if r.get('mean_u') is None:
            cases.append(dict(condition=c['name'],status=r['status']));continue
        pool=read(checked(c['pool']));packet=load_pickle(c['history']);history=packet['history'] if 'history' in packet else packet
        plans={k:read(checked(v)) for k,v in c['old_plans'].items()};resources=plans['optimized']['resources']
        curves={k:curve_from_plan(v,resources) for k,v in plans.items()};curves['daily_optimized']=np.asarray(r['mean_u'])
        summaries={}
        for role,seed in [('training',c['training_evaluation_seed']),('validation',c['validation_evaluation_seed'])]:
            problem=build_problem(pool,history,resources,template_role=role)
            scenarios=draw_scenarios(problem,p['model_holdout_samples'],seed)
            summaries[role]={k:dict(reference_fine=scenario_summary(problem,scenarios,u),
                alternative_coarse=alternative_forecast_summary(problem,scenarios,u),resource_audit=resource_audit(problem,u)) for k,u in curves.items()}
        cases.append(dict(condition=c['name'],status='complete',planning_status=r['status'],summaries=summaries))
    write(root/'model_evaluation.json',dict(status='complete',cases=cases,price_freeze=freeze,
        denominator_per_condition=256,new_AC=0,new_fit_calls=0,new_observations=0,
        validation_scope='historical validation residual blocks withheld from curve optimization; aggregate historical losses already used in decision threshold',
        training_scope='new response factors on same training residual templates, not independent physical evidence'))
    write(root/'ended.json',dict(status='complete',created_utc=now(),model_evaluation=ref(root/'model_evaluation.json'),price_freeze=freeze))
    write(root/'run_lock.json',dict(status='complete',ended_utc=now(),pid=os.getpid()))
    print(dict(stage='model_evaluation_complete',conditions=len(cases),new_AC=0),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','evaluate'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seconds',type=float,default=60.);parser.add_argument('--cost-seconds',type=float,default=30.)
    parser.add_argument('--unit-gate',type=Path)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.output,a.seconds,a.cost_seconds,a.unit_gate)
    elif a.command=='run':run(a.output)
    else:evaluate(a.output)


if __name__=='__main__':main()
