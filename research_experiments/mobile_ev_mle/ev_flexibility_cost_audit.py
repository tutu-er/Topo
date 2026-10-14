"""Saved-timing-only cost diagnostic; never runs policy, optimizer or AC."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE/'evidence/ev_flexibility/v2_safe_formal'
if not any((ROOT/'calibration').glob('trial_*.json.gz')):
    ROOT=ROOT/'compact_full_diagnostics'

def load(path):return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())
def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),sum=float(a.sum()),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None,
        maximum=float(a.max()) if len(a) else None,minimum=float(a.min()) if len(a) else None)
def split_power(operation):return np.asarray(operation['split_charge_pu'])-np.asarray(operation['split_discharge_pu'])
def constant_power(device,site,t):
    assert device['setup_slots']==0
    access=np.asarray(device['availability'],bool)[:,site].copy()
    tau=int(device['travel_slots'][device['initial_site']][site]);access[:min(tau,t)]=False
    if device.get('departure_slot') is not None:access[int(device['departure_slot']):]=False
    expense=float(device['travel_energy'][device['initial_site']][site])
    delta=device['terminal_energy']-(device['initial_soc']*device['battery_capacity']-expense)
    assert delta>=0 and np.isscalar(device['p_charge_max'])
    possible=access & (device['p_charge_max']>0);u=np.zeros(t)
    u[possible]=delta/device['eta_charge']/(device['dt_hours']*int(possible.sum()))
    return u

def requests(search,device,base):
    """Reconstruct the saved deterministic cache requests, not new actions."""
    cache={};result=[];saved=search.get('evaluations',[])
    def register(site,u,origin):
        local=(site,np.asarray(u,dtype='<f8').tobytes())
        if local in cache:return
        i=len(result);assert i<len(saved),'too many reconstructed information calls'
        e=saved[i];cache[local]=e
        result.append(dict(key=(site,np.asarray(base*u,dtype='<f8').tobytes()),
            site=site,power_pu=np.asarray(u).tolist(),origin=origin,evaluation=e))
    for rec in search.get('site_records',[]):
        site=rec['site'];constant=rec['constant_operation']
        if constant['success']:
            initial=constant_power(device,site,4);register(site,initial,'constant')
        else:
            feasible=rec.get('feasible_initialization',{})
            if not feasible.get('success'):continue
            initial=split_power(feasible)
        starts=[initial];seen={np.asarray(initial,dtype='<f8').tobytes()}
        for seed in rec['starts']:
            if seed['kind']!='optimized_alternating_seed' or not seed['operation']['success']:continue
            u=split_power(seed['operation']);key=np.asarray(u,dtype='<f8').tobytes()
            if key not in seen:starts.append(u);seen.add(key)
        for entry in rec['starts']:
            if entry['kind']!='accepted_seed_input':continue
            index=entry['index'];register(site,starts[index],'start')
            for step in rec['steps']:
                if step['start']==index and 'trial_success' in step:
                    register(site,split_power(step['milp']),'price_step')
    assert len(result)==len(saved),'reconstructed count differs from saved information calls'
    assert sum(x['evaluation']['total_ac_calls'] for x in result)==search['total_ac_calls'],'search AC accounting mismatch'
    return result

def operation_times(selection):
    values=[]
    for x in selection.get('cheap_operations',[]):
        if x['operation'].get('solve_seconds') is not None:values.append(x['operation']['solve_seconds'])
    for search in selection.get('searches',{}).values():
        for site in search.get('site_records',[]):
            optional=site.get('feasible_initialization',{})
            if optional.get('solve_seconds') is not None:values.append(optional['solve_seconds'])
            for entry in site['starts']:
                if entry['kind']=='optimized_alternating_seed' and entry['operation'].get('solve_seconds') is not None:
                    values.append(entry['operation']['solve_seconds'])
            for step in site['steps']:
                if step['milp'].get('solve_seconds') is not None:values.append(step['milp']['solve_seconds'])
    return values

def action_key(row):
    return tuple(np.asarray(row['action'][k],dtype='<f8').tobytes() for k in ('p','q'))
def observation_key(row):
    given=row.get('observations')
    if not given:return None
    return tuple(np.asarray(given[k],dtype='<f8').tobytes() for k in ('p','q','v','root_v','master_p','master_q'))

def diagnose(records,phase):
    breakdown=[];calls=Counter();all_search=defaultdict(list);info_times=[];lp_times=[]
    info_requests=info_duplicates=duplicate_calls=0;duplicate_seconds=0.;scalar_reuse=scalar_calls=0;scalar_seconds=0.
    exact_joint_duplicates=duplicate_joint_ac=0;duplicate_joint_seconds=0.;joint_jobs=cache_hits=0
    same_charge_staff_sites=0;checked_charge_staff_sites=0;request_conflicts=[]
    for r in records:
        s=r['selection'];future=sum(x['seconds'] for x in r['records']);h=r['historical_cost']['seconds'];sel=s['seconds']
        assert r['total_ac_calls']==r['historical_cost']['ac_calls']+s['selector_ac_calls']+sum(x.get('ac_calls',0) for x in r['records'])
        item=dict(history_id=r['history_id'],case=r['case'],quality=r['quality'],whole_seconds=r['seconds'],H_fit_seconds=h,
            selection_seconds=sel,new_observation_and_joint_fit_seconds=future,other_unprofiled_seconds=r['seconds']-h-sel-future,
            selection_over15=sel>15,whole_over90=r['seconds']>90,history_fit_failed=bool(r['historical_failures']),
            joint_failed=any(x['status']!='complete' for x in r['records']),saved_gate_passed=r['gate_passed'],
            context_seconds=s.get('context',{}).get('instrumented_build_seconds',0.),search_seconds={k:v['seconds'] for k,v in s.get('searches',{}).items()},
            AC_H=r['historical_cost']['ac_calls'],AC_selection=s['selector_ac_calls'],AC_joint=sum(x.get('ac_calls',0) for x in r['records']))
        breakdown.append(item);calls.update(H=item['AC_H'],selection=item['AC_selection'],joint=item['AC_joint'])
        seen={};ops=operation_times(s);lp_times.extend(ops)
        for tag,search in s.get('searches',{}).items():
            all_search[tag].append(search['seconds'])
            reconstructed=requests(search,s['devices'][tag],r['config']['base_kva'])
            info_requests+=len(reconstructed)
            for point in reconstructed:
                e=point['evaluation'];info_times.append(e['seconds']);key=point['key']
                if key in seen:
                    old=seen[key]
                    if all(e[k]==old[k] for k in ('success','score','pairs','targets','failures')):
                        info_duplicates+=1;duplicate_calls+=e['total_ac_calls'];duplicate_seconds+=e['seconds']
                    else:request_conflicts.append(dict(history_id=r['history_id'],tag=tag,site=point['site']))
                else:seen[key]=e
        for check in s.get('baseline_information_checks',[]):
            if not check.get('evaluation',{}).get('success'):continue
            row=next(x for x in r['records'] if x['method']==check['method']);site=row['action']['site']
            key=(site,np.asarray(r['config']['base_kva']*np.asarray(row['action']['p'])[:,site],dtype='<f8').tobytes())
            if key in seen and seen[key]['success'] and seen[key]['score']==check['evaluation']['score']:
                scalar_reuse+=1;scalar_calls+=check['evaluation']['total_ac_calls'];scalar_seconds+=check['evaluation']['seconds']
        ch={x['site']:x for x in s.get('searches',{}).get('charge',{}).get('site_records',[])}
        st={x['site']:x for x in s.get('searches',{}).get('staff',{}).get('site_records',[])}
        for site in (0,1):
            if site in ch and site in st and 'selected_plan' in ch[site] and 'selected_plan' in st[site]:
                checked_charge_staff_sites+=1
                same_charge_staff_sites+=bool(ch[site]['selected_plan']['p']==st[site]['selected_plan']['p'] and ch[site]['selected_score']==st[site]['selected_score'])
        seen_data={}
        for row in r['records']:
            cache_hits+=bool(row.get('cache_hit',False));joint_jobs+=bool(row.get('fit_calls',0))
            key=observation_key(row)
            if key is not None and not row.get('cache_hit',False):
                if key in seen_data:
                    exact_joint_duplicates+=1;duplicate_joint_ac+=row.get('ac_calls',0);duplicate_joint_seconds+=row['seconds']
                else:seen_data[key]=row['method']
    return dict(phase=phase,n=len(records),planned_n=39 if phase=='calibration' else 48,
        completed_snapshot_only=len(records)!=(39 if phase=='calibration' else 48),
        timing={k:stats([r[k] for r in breakdown]) for k in ('whole_seconds','H_fit_seconds','selection_seconds','new_observation_and_joint_fit_seconds','other_unprofiled_seconds')},
        AC_calls=dict(calls),AC_total=sum(calls.values()),joint_fit_jobs=joint_jobs,joint_cache_hits=cache_hits,
        cost_gate=dict(selection_over15=sum(x['selection_over15'] for x in breakdown),whole_over90=sum(x['whole_over90'] for x in breakdown),
            saved_failed=sum(not x['saved_gate_passed'] for x in breakdown),overruns=[x for x in breakdown if not x['saved_gate_passed']]),
        selection_subcomponents=dict(search_seconds={k:stats(v) for k,v in all_search.items()},
            saved_information_evaluation_seconds=stats(info_times),saved_operational_solver_seconds=stats(lp_times)),
        exact_information_repeats=dict(full_gradient_requests=info_requests,duplicate_requests=info_duplicates,
            duplicate_AC_calls=duplicate_calls,duplicate_saved_seconds=duplicate_seconds,
            outputs_identical_conflicts=request_conflicts,scalar_checks_with_existing_full_result=scalar_reuse,
            scalar_repeat_AC_calls=scalar_calls,scalar_repeat_saved_seconds=scalar_seconds,
            same_charge_staff_selected_sites=same_charge_staff_sites,checked_charge_staff_selected_sites=checked_charge_staff_sites),
        exact_joint_data_repeats=dict(noncache_duplicate_jobs=exact_joint_duplicates,AC_calls=duplicate_joint_ac,
            enclosing_saved_seconds=duplicate_joint_seconds),per_history=breakdown)

def main():
    files={phase:sorted((ROOT/phase).glob('trial_*.json.gz')) for phase in ('calibration','evaluation')}
    manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for fs in files.values() for p in fs}
    data={phase:diagnose([load(p) for p in fs],phase) for phase,fs in files.items()}
    freeze=load(ROOT/'freeze.json');science={n:h for n,h in freeze['source_sha256'].items() if n not in ('ev_flexibility_formal_safe.py','ev_flexibility_formal_safe_protocol.md')}
    assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h for n,h in science.items())
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),scope='saved cost records only; no new optimizer, policy or AC calls',
        captured_file_count=len(manifest),input_sha256=manifest,scientific21_bytes_still_match_freeze=True,
        accounting_notes=['new_observation_and_joint_fit includes physical forward/concat/checks; pure joint-fit seconds not saved separately',
            'AC_total is stored fit/design call count; excludes history-observe and nine method truth-forward calls',
            'duplicate saved seconds are existing measured work, not a benchmark of future cache speedup',
            'no synchronous load, CPU-frequency or resource telemetry; no cause inferred for cross-run slowdown'],
        suggested_next_steps=['share per-H exact (site,power_kw,gradient_flag) information results across charge/V2G/staff; preserve failed result traces and call accounting',
            'reuse scalar score from successful exact full-gradient result instead of recomputing final constant/quality checks',
            'for identical observed joint data/model/initials/seed, cache candidate fits by fitting inputs, not only action bytes',
            'different central-difference neighbor points are necessary for current price; replacing them needs separate validated algorithm stage'],phases=data)
    target=HERE/'evidence/ev_flexibility/safe_cost_diagnostic.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    cal=data['calibration'];ev=data['evaluation'];c=cal['timing'];over=cal['cost_gate']['overruns'];repeat=cal['exact_information_repeats']
    lines=['# safe v2 保存成本诊断','',f"只读快照：校准 {cal['n']}/39，评价 {ev['n']}/48。科学21源码仍匹配冻结字节；无新优化或AC调用。",'',
        '| 校准组件 | 合计秒 | 每历史均值秒 | 中位数秒 |', '|---|---:|---:|---:|']
    for name,label in [('whole_seconds','整历史'),('H_fit_seconds','历史候选拟合'),('selection_seconds','EV选择'),('new_observation_and_joint_fit_seconds','新增计量与联合拟合'),('other_unprofiled_seconds','其余未单独计时')]:
        x=c[name];lines.append(f"| {label} | {x['sum']:.3f} | {x['mean']:.3f} | {x['median']:.3f} |")
    lines+=['',f"校准保存AC计数：历史 {cal['AC_calls']['H']}，选择 {cal['AC_calls']['selection']}，联合拟合 {cal['AC_calls']['joint']}，合计 {cal['AC_total']}。",'',
        f"成本门失败 {cal['cost_gate']['saved_failed']}/39：选择超过15秒 {cal['cost_gate']['selection_over15']}，整历史超过90秒 {cal['cost_gate']['whole_over90']}。以下保留全部超限历史：",'',
        '| H | 选择秒 | 整历史秒 | 历史拟合秒 | 新增/联合秒 |','|---|---:|---:|---:|---:|']
    for x in over:lines.append(f"| {x['history_id']} | {x['selection_seconds']:.3f} | {x['whole_seconds']:.3f} | {x['H_fit_seconds']:.3f} | {x['new_observation_and_joint_fit_seconds']:.3f} |")
    lines+=['',f"从保存功率和步骤复原全部信息请求，发现校准 {repeat['duplicate_requests']}/{repeat['full_gradient_requests']} 个跨域精确重复完整价格评估，重复调用 {repeat['duplicate_AC_calls']} 次，重复工作保存计时 {repeat['duplicate_saved_seconds']:.3f}秒；重复输出完全一致的冲突数 {len(repeat['outputs_identical_conflicts'])}。", 
        f"另有 {repeat['scalar_checks_with_existing_full_result']} 个末尾标量检查已有同点成功完整结果，对应 {repeat['scalar_repeat_AC_calls']} 次调用、{repeat['scalar_repeat_saved_seconds']:.3f}秒。这些数字是已有工作计时，尚未做缓存改造后的加速验证。",'',
        '优先考虑同历史、同展开点、同公开计量模型下共享精确输入的信息缓存。当前每次charge/V2G/staff搜索各建局部cache，charge与staff在节点0/1操作域相同；最终恒充/质量检查还会重算。新增AC拟合已经按精确动作缓存，可进一步检查完全相同联合观测是否重复拟合。不同中心差分邻点承担当前导数计算，不应简单删除。', '',
        '纯联合拟合秒数未单独保存，因此表中新增/联合时间包含物理forward、校验和数据拼接。保存AC计数不含历史observe及九方案真值forward。没有同期资源负载证据，不解释跨运行变慢原因。', f"评价完整48历史均值 {ev['timing']['whole_seconds']['mean']:.3f} 秒；选择超过15秒 {ev['cost_gate']['selection_over15']}，整历史超过90秒 {ev['cost_gate']['whole_over90']}，保存综合门失败 {ev['cost_gate']['saved_failed']}。综合门也检查设计有效性，不能全部解释成耗时超限。"]
    (HERE/'evidence/ev_flexibility/safe_cost_diagnostic.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({phase:{k:data[phase][k] for k in ('n','AC_total','cost_gate','exact_information_repeats','exact_joint_data_repeats')} for phase in data},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
