"""Post-search topology scoring, independent LP audit and reversible evidence."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import shutil
import numpy as np
import pandas as pd
from .run import ROOT, METHODS, read, write, sha
from .model import select_block
from research_experiments.local_rebuild.model import canonical, compatible
from research_experiments.rebuild_benchmark.engine import prepare, loss
from research_experiments.rebuild_benchmark.analyze import score
from research_experiments.joint_exit.analyze import independent_profile, markdown
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_paper.common import observed
from rnj_wzzt.estimation import laminar_l1_milp as core


def analyze(output):
    output = Path(output); p = read(output / 'protocol.json'); catalog = read(output / 'catalog.json')
    completion = read(output / 'completion.json'); checks = 0; preserved = 0
    for name, digest in p['source_hashes'].items(): assert sha(output / 'source' / name) == digest
    for path, digest in p.get('primary_pilot_references', {}).items(): assert sha(path) == digest
    for file, record in p['parent_manifests'].items():
        assert sha(file) == record['sha256']
        for name, item in record['files'].items(): assert sha(Path(file).parent / name) == item['sha256']; preserved += 1
    rows, events, diagnostics, profiles, costs = [], [], [], [], []
    for job in p['jobs']:
        meta = catalog[job]; start_file = output / 'parents' / job / 'start.json'
        assert sha(start_file) == p['start_hashes'][job]
        assert {s: sha(meta[s]) for s in ('train', 'validation', 'test')} == p['input_hashes'][job]
        for path, digest in p['reference_hashes'][job].items(): assert sha(path) == digest
        start = read(start_file); block = tuple(start['block']) if start['block'] else None; labels = start['labels']; n = len(labels)
        assert select_block(start['point'], n) == block
        data = {s: prepare(observed(meta[s], meta['N'] if s == 'train' else None)) for s in ('train', 'validation', 'test')}
        truth = physical_truth(Path(meta['truth_network']), labels); positions = {v: i for i, v in enumerate(labels)}
        true = {tuple(sorted(positions[v] for v in C)) for C in truth['clades']}
        cache = {}
        def profile(point):
            nonlocal checks
            family = canonical(point['supports']); weights = np.r_[point['r'], point['x']]
            assert core.is_laminar_family(family) and np.isfinite(weights).all()
            assert weights.min(initial=0.) >= -1e-8 and weights.max(initial=0.) <= 2.+1e-8
            if family not in cache:
                cache[family] = independent_profile(data['train'], family)
                profiles.append(dict(job=job, family=family, **cache[family]))
            assert abs(cache[family]['objective']-point['objective']) < 2e-7
            assert abs(loss(data['train'], point)-point['objective']) < 1e-8
            checks += 4
        header = {k: meta[k] for k in ('case', 'bundle', 'N', 'voltage_sigma')}
        header.update(job=job, eligible=block is not None, block=block, block_true=block in true if block else None)
        blocked = {C for C in true if block and not compatible(C, block)}
        historical = read(output / 'parents' / job / 'baseline_choices.json')['bank']
        control_file = output / 'jobs' / job / 'continue/result.json'
        control = read(control_file) if control_file.exists() else None
        base_points = dict(raw=start['point'], guarded=historical['point'])
        if control: base_points['proposal'] = control['historical_endpoint_choice']['point']
        for view in ('raw', 'proposal', 'guarded'):
            point = base_points.get(view)
            if point: profile(point)
            rows.append(dict(**header, method='historical', view=view, selected_source='historical',
                **score(point, data, truth, labels, canonical(start['pool']))))
        costs.append(dict(**header, method='historical', fresh_seconds=0., total_seconds=start['upstream_seconds'],
            timing='recorded components, not rerun'))
        for method in METHODS:
            file = output / 'jobs' / job / method / 'result.json'
            if not file.exists():
                for view in ('raw', 'proposal', 'guarded'):
                    rows.append(dict(**header, method=method, view=view, selected_source='failure', **score(None,data,truth,labels,())))
                continue
            r = read(file)
            assert r['protocol_sha256'] == sha(output / 'protocol.json') and r['start_sha256'] == p['start_hashes'][job]
            assert not r['truth_used'] and not r['test_opened'] and not r['LP_failures']
            assert r['input_hashes'] == {s: p['input_hashes'][job][s] for s in ('train', 'validation')}
            assert r['total_seconds'] <= p['limits']['total']+1. and not r['total_cap_exceeded']
            assert r['penalty'] == .003*start['reference'] and r['upstream_seconds'] == start['upstream_seconds']
            if method != 'continue' and block: assert r['drop_L'] >= r['before_L']-2e-7
            trajectory = r['trajectory']; forbidden = set(canonical(trajectory['forbidden']))
            for point in trajectory['path']:
                profile(point); assert not set(canonical(point['supports'])) & forbidden
            for trial in trajectory['events']:
                if trial.get('solve_started'):
                    assert trial['audit']['valid'] and trial['warm_start']['valid']
                    assert trial['objective'] <= trial['warm_start']['objective']+1e-8
                    if trial.get('bound') is not None:
                        assert trial['bound']+trial['constant'] <= trial['score']+1e-8
                    checks += 3
                if trial.get('accepted'): assert trial['before']-trial['score'] > trial['threshold']
                events.append(dict(job=job, method=method, kind=trial['kind'], status=trial.get('status'), accepted=trial['accepted'],
                    seconds=trial.get('seconds'), bound=trial.get('bound'), constant=trial.get('constant'),
                    requested_gap=trial.get('requested_gap'), relative_gap=trial.get('relative_gap')))
            assert r['guarded']['validation'] <= historical['validation']+1e-12
            points = dict(raw=trajectory['point'], proposal=r['proposal']['point'], guarded=r['guarded']['point'])
            for view, point in points.items():
                profile(point); metric = score(point, data, truth, labels, canonical(start['pool']))
                if view != 'raw': assert abs(metric['validation_mae']-r[view]['validation']) < 1e-10
                if method == 'mask_resume' and block and view != 'guarded': assert block not in canonical(point['supports'])
                rows.append(dict(**header, method=method, view=view,
                    selected_source=r['guarded']['source'] if view == 'guarded' else 'proposal', **metric))
            pred = {tuple(C) for C, rr, xx in zip(points['raw']['supports'], points['raw']['r'], points['raw']['x']) if 1 < len(C) < n and max(rr,xx) > 1e-9}
            diagnostics.append(dict(**header, method=method, applied=r['applied'], before_L=r['before_L'], drop_L=r['drop_L'], after_L=r['after_L'],
                before_J=r['before_J'], drop_J=r['drop_J'], after_J=r['after_J'],
                J_improved=r['before_J']-r['after_J'] > 1e-10+.001*r['before_L'],
                block_retained=r['block_retained'], blocked_true=len(blocked), recovered_blocked_true=len(pred & blocked),
                outside_old_pool=len(pred-set(canonical(start['pool']))),
                guarded_deploys_proposal=r['guarded']['source']=='proposal', stop=trajectory['stop']))
            costs.append(dict(**header, method=method, fresh_seconds=r['fresh_seconds'], total_seconds=r['total_seconds'],
                timing='recorded original mainline components plus measured new branch'))
            checks += 8
        print('audited', job, flush=True)
    dest = output / 'analysis'; dest.mkdir(exist_ok=True)
    f, d, e, c = map(pd.DataFrame, (rows, diagnostics, events, costs))
    for frame, name in ((f,'per_model.csv'),(d,'diagnostics.csv'),(e,'events.csv'),(c,'cost_per_input.csv')): frame.to_csv(dest/name,index=False)
    tables=[]
    for scope in ('all24','eligible'):
        g = f if scope == 'all24' else f[f.eligible]
        a = g.groupby(['view','method']).agg(inputs=('job','size'),available=('model_available','sum'),F1=('clade_f1','mean'),exact=('exact','sum'),
            train=('train_mae','mean'),validation=('validation_mae','mean'),test=('test_mae','mean')).reset_index()
        a['scope']=scope;tables.append(a)
    summary=pd.concat(tables,ignore_index=True);summary.to_csv(dest/'summary.csv',index=False)
    c.groupby('method').agg(inputs=('job','size'),fresh_median=('fresh_seconds','median'),fresh_sum=('fresh_seconds','sum'),
        total_median=('total_seconds','median'),total_sum=('total_seconds','sum')).to_csv(dest/'cost_summary.csv')
    d.groupby('method').agg(inputs=('job','size'),eligible=('eligible','sum'),J_improved=('J_improved','sum'),
        block_retained=('block_retained','sum'),blocked_true=('blocked_true','sum'),recovered_blocked_true=('recovered_blocked_true','sum'),
        outside_old_pool=('outside_old_pool','sum'),proposal_deployed=('guarded_deploys_proposal','sum')).to_csv(dest/'mechanism_summary.csv')
    pairs=[]
    for (view, method), group in f.groupby(['view','method']):
        if method=='historical':continue
        for ref in ('historical','continue'):
            if method==ref:continue
            x=group.set_index('job');y=f[(f.view==view)&(f.method==ref)].set_index('job')
            delta=x.clade_f1-y.clade_f1
            pairs.append(dict(view=view,method=method,reference=ref,inputs=len(delta),delta_F1=delta.mean(),
                wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta < -1e-10).sum()),
                new_exact=int((x.exact & ~y.exact).sum()),lost_exact=int((~x.exact & y.exact).sum())))
    pd.DataFrame(pairs).to_csv(dest/'paired.csv',index=False)
    f.groupby(['case','view','method']).agg(inputs=('job','size'),F1=('clade_f1','mean'),exact=('exact','sum')).to_csv(dest/'cases.csv')
    d.groupby(['block_true','method'],dropna=False).agg(inputs=('job','size'),J_improved=('J_improved','sum'),
        block_retained=('block_retained','sum'),recovered_blocked_true=('recovered_blocked_true','sum')).to_csv(dest/'block_truth.csv')
    write(dest/'independent_profiles.json',profiles)
    write(dest/'verification.json',dict(passed=completion['returned']==96 and not completion['failures'],returned=completion['returned'],
        planned=96,checks=checks,independent_LPs=len(profiles),parent_files_preserved=preserved,
        failures=completion['failures'],MIPs=int(e.status.notna().sum()) if len(e) else 0,
        gap_met=int((e.status==2).sum()) if len(e) else 0,timeouts=int((e.status==9).sum()) if len(e) else 0,
        analyzer_sha256=sha(Path(__file__))))
    shutil.copy2(Path(__file__),dest/'executed_analysis.py')
    delivery=Path(p.get('delivery_directory',str(ROOT/'docs/flexible_exit_20261101')))
    for file in dest.glob('*.csv'):shutil.copy2(file,delivery/file.name)
    for name in ('verification.json',):shutil.copy2(dest/name,delivery/name)
    for name in ('protocol.json','completion.json'):shutil.copy2(output/name,delivery/name)
    print(summary.query('view == "guarded"').to_string(index=False),flush=True)
    print(pd.read_csv(dest/'mechanism_summary.csv').to_string(index=False),flush=True)


def package(output):
    output=Path(output);p=read(output/'protocol.json');audit=read(output/'analysis/verification.json')
    assert audit['passed'] and audit['returned']==96
    delivery=Path(p.get('delivery_directory',str(ROOT/'docs/flexible_exit_20261101')));archive=delivery/'records.zip'
    for file,record in p['parent_manifests'].items():
        assert sha(file)==record['sha256']
        for name,item in record['files'].items():assert sha(Path(file).parent/name)==item['sha256']
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for folder in ('source','parents','jobs','analysis'):
            for file in (output/folder).rglob('*'):
                if file.is_file() and file.suffix in ('.py','.json','.csv','.log'):z.write(file,file.relative_to(output))
        for name in ('protocol.json','completion.json','catalog.json'):z.write(output/name,name)
        executed=output/'analysis/executed_analysis.py'
        assert sha(executed)==audit['analyzer_sha256']
        z.write(executed,'actual_analysis/analyze.py')
        for file in Path(__file__).parent.glob('*.py'):z.write(file,'actual_driver/'+file.name)
        for name in ('PLAN.md','CONCLUSIONS.md','RECOVERY.md','implementation_check.json'):
            if (delivery/name).exists():z.write(delivery/name,'reports/'+name)
        z.write(ROOT/'research_experiments/tests/test_flexible_exit.py','tests/test_flexible_exit.py')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in p['source_hashes'].items():assert hashlib.sha256(z.read('source/'+name)).hexdigest()==digest
        assert hashlib.sha256(z.read('actual_analysis/analyze.py')).hexdigest()==audit['analyzer_sha256']
    write(delivery/'delivery_audit.json',dict(passed=True,returned=96,archive_sha256=sha(archive),
        parent_files_preserved=audit['parent_files_preserved'],note='24 paired pilot inputs, no raw observations copied; immutable previous mainline retained.'))
    write(delivery/'manifest.json',dict(files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(delivery.iterdir()) if f.is_file() and f.name!='manifest.json'}))
    write(output/'state.json',dict(stage='complete',returned=96,verified=True))
    print('Flexible-exit pilot archived and verified',flush=True)


if __name__=='__main__':
    import sys
    {'analyze':analyze,'package':package}[sys.argv[1]](Path(sys.argv[2]).resolve())
