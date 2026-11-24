"""Cross-condition analysis: harm of wrong confirmations, b/J with a correct leader, planner reasons."""
import json, sys, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(r"C:\0-workspace\Topo\repo\outputs\hourly_probing_dev_20261120")
def load(c, tag="runs"):
    p = ROOT/c/f"{tag}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
for c in sys.argv[1:]:
    tag = "runs"
    if ":" in c: c, tag = c.split(":")
    recs = load(c, tag)
    if not recs: continue
    print(f"\n===== {c} ({tag}) n={len(recs)}")
    by = defaultdict(list)
    for r in recs: by[(r['stratum'], r['arm'])].append(r)
    for (st, a), rs in sorted(by.items()):
        n = len(rs); oc = Counter(r['outcome'] for r in rs)
        ew = [r for r in rs if r['ever_confirmed_wrong']]
        harm = [r['wrong_vdev'] for r in ew if r.get('wrong_vdev') is not None]
        eqv = sum(v <= 5e-4 for v in harm); near = sum(v <= 1e-3 for v in harm)
        types = Counter(t for r in ew for t in r['ever_wrong_types'])
        fv = [r['final_vdev'] for r in rs if r.get('final_vdev') is not None and r['final_leader_correct']]
        first = [r['first_correct_gate'] for r in rs if r['first_correct_gate'] is not None]
        print(f"{st} {a:17s} n={n:3d} correct={oc['correct']/n:4.0%} wrong_end={oc['wrong']/n:4.0%} "
              f"undec={oc['undecided']/n:4.0%} noprog={oc['unresolved_no_progress']/n:4.0%} beyond={oc['unresolved_beyond_reach']/n:4.0%} "
              f"ever_wrong={len(ew)/n:4.0%} {dict(types)} | wrong within 0.05%={eqv}/{len(harm)} within 0.1%={near}/{len(harm)} "
              f"max_dev={max(harm) if harm else 0:.4f} | med_h={np.median(first) if first else float('nan'):.0f} "
              f"probed={np.mean([r['probed_hours'] for r in rs]):.1f}h cost={np.mean([r['cost_cny'] for r in rs]):.2f} "
              f"correct-leader vdev med={np.median(fv) if fv else float('nan'):.5f}")
    # planner reasons and binding constraint for t_opt
    for st in ('S1', 'S2', 'S3'):
        rs = by.get((st, 't_opt_P'), [])
        if not rs: continue
        reasons = Counter(b['reason'] for r in rs for b in r['blocks'])
        ratio = [min(q/b['eta'][j] for j, q in b['q_planned'].items()) for r in rs for b in r['blocks']
                 if b['q_planned'] and all(b['eta'][j] > 0 for j in b['q_planned'])]
        print(f"  {st} t_opt reasons {dict(reasons)} q/eta med={np.median(ratio) if ratio else float('nan'):.3f}")
    # b/J only where the leader at the sealing gate was the truth
    bias = defaultdict(list)
    for r in recs:
        if r['arm'] != 't_opt_P': continue
        lc = {g['g']: g['leader_correct'] for g in r['gates']}
        for b in r['blocks']:
            if not lc.get(b['sealed_at']): continue
            for k, v in (b.get('bias') or {}).items():
                bias[(v['type'], k.split('_')[1])].append((v['whole'], v['probe']))
    for k, v in sorted(bias.items()):
        v = np.array(v)
        print(f"  b/J leader=truth {k}: n={len(v)} whole |>0.5|={np.mean(abs(v[:,0])>.5):.2f} med={np.median(v[:,0]):+.2f} "
              f"probe |>0.5|={np.mean(abs(v[:,1])>.5):.2f} med={np.median(v[:,1]):+.2f}")
    # S1 wrong confirmations: where was the truth?
    for r in recs:
        if r['stratum'] == 'S1' and r['ever_confirmed_wrong'] and r['arm'] in ('passive_bic',) and r['path'] == 0:
            gs = [g for g in r['gates'] if g.get('confirmed') and not g['leader_correct']][:1]
            for g in gs:
                print(f"  S1 wrong conf {r['input']} g={g['g']} truth_E={g.get('truth_E')} truth_eq={g.get('truth_eq')} vdev={g.get('vdev_true')}")
