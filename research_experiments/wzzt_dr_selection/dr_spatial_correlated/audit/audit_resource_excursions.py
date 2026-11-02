"""Read saved deliveries only; no AC, fit, probability or main audit is run."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(reference):
    path = Path(reference['path'])
    if sha(path) != reference['sha256']:
        raise ValueError('saved physical packet hash mismatch')
    return path


def run(root):
    ledger_path = root/'observations_frozen.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    if ledger.get('status') != 'frozen' or not (root/'ended.json').is_file():
        raise ValueError('completed frozen physical batch required')
    groups = defaultdict(list)
    for record in ledger['rows']:
        item = dict(day=record['day'], status=record['status'])
        if record['status'] == 'observed' and 'packet' in record:
            with checked(record['packet']).open('rb') as stream:
                result = pickle.load(stream)
            private, summary = result['private'], result['summary']
            a = np.asarray(private['capacity_kw'], float)
            delivery = np.asarray(private['delivered_kw'], float)
            component = a[:, None] + delivery
            excess = np.maximum.reduce([np.zeros_like(component), -component,
                                        component-2*a[:, None]])
            item.update(excess=excess, capacities=a, nodes=summary['resource_labels'],
                        count_matches_original=int((excess>1.e-10).sum())==summary['resource_violation_cells'])
        groups[(record['law'],record['policy'])].append(item)
    tables=[]
    for (law,policy),items in sorted(groups.items()):
        valid=[r for r in items if 'excess' in r]
        by_node=defaultdict(list)
        for record in valid:
            for j,node in enumerate(record['nodes']):
                by_node[int(node)].append((record['excess'][j],float(record['capacities'][j])))
        nodes=[]
        for node,values in sorted(by_node.items()):
            curves=np.stack([x[0] for x in values]);capacities=[x[1] for x in values]
            if min(capacities)!=max(capacities):
                raise ValueError('capacity changed across frozen same-policy days')
            nodes.append(dict(node_label=node,capacity_kw=capacities[0],records=len(values),
                max_excess_kw=float(curves.max(initial=0.)),
                cells_over_1e_10_kw=int((curves>1.e-10).sum()),
                days_over_1e_10_kw=int(np.any(curves>1.e-10,axis=1).sum()),
                cells_over_1e_6_kw=int((curves>1.e-6).sum()),
                days_over_1e_6_kw=int(np.any(curves>1.e-6,axis=1).sum())))
        table=dict(law=law,policy=policy,planned_records=len(items),private_array_records=len(valid),
            unresolved_records=len(items)-len(valid),
            max_excess_kw=max((float(r['excess'].max(initial=0.)) for r in valid),default=None),
            cells_over_1e_10_kw=sum(int((r['excess']>1.e-10).sum()) for r in valid),
            days_over_1e_10_kw=sum(bool(np.any(r['excess']>1.e-10)) for r in valid),
            cells_over_1e_6_kw=sum(int((r['excess']>1.e-6).sum()) for r in valid),
            days_over_1e_6_kw=sum(bool(np.any(r['excess']>1.e-6)) for r in valid),
            original_violation_counts_reproduced=all(r['count_matches_original'] for r in valid),nodes=nodes)
        tables.append(table)
    report=dict(status='passed' if all(t['original_violation_counts_reproduced'] for t in tables) else 'failed',
        created_utc=datetime.now(timezone.utc).isoformat(),run_name=root.name,
        audit_source_sha256=sha(__file__),observations_ledger_sha256=sha(ledger_path),
        source_inputs='existing hash-checked physical result packets; no new observations',
        formula='component=a+delivered; excess=max(0,-component,component-2*a), in kW',
        threshold_interpretation='1e-10 reproduces original diagnostic; 1e-6 only reports magnitude, never filters or deletes observations',
        physical_records=len(ledger['rows']),independent_backgrounds=12,paired_response_laws=2,
        new_AC_calls=0,new_optimizer_calls=0,main_audit_rerun=False,tables=tables,
        public_scope='aggregate resource allocation and violation counts/magnitudes only; no load or delivery trajectories')
    out=HERE/'resource_excursion_summary.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],physical_records=report['physical_records'],
                         groups=len(tables),output=out.name,new_AC_calls=0),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='run_v2')
    args=parser.parse_args();path=Path(args.run)
    run(path if path.is_absolute() else HERE.parent/path)
