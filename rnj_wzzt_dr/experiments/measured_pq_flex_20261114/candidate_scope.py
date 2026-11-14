"""Contrast constrained representatives with every frozen wrong-topology point."""
from pathlib import Path
import argparse
from collections import Counter
from ... import io


def check(run, analysis, out):
    run, analysis, out = Path(run), Path(analysis), Path(out)
    if out.exists():
        raise FileExistsError('Preserve scope diagnostic')
    entries = io.read(run/'inputs.json')
    scores = {}
    for e in entries:
        if io.ref(e['score']['path'])['sha256'] != e['score']['sha256']:
            raise ValueError('Score changed')
        scores[e['id']] = io.read(e['score']['path'])
    rows = io.read(analysis/'rows_with_bias_split.json')
    output = {}
    for strategy in sorted({r['strategy'] for r in rows}):
        own = [r for r in rows if r['strategy']==strategy and r['status']=='complete']
        scopes = {}
        for representative_only in (False, True):
            negative, positive, pair_count, negative_paths = 0, 0, 0, 0
            cases = Counter(); lowest = None
            for row in own:
                score = scores[row['id']]; bank = score['native_bank']
                rep_ids = {m['index'] for m in score['models']}
                target = row['historical_native_index']; bad = False
                for p in row['evidence'].get('target_pairs', []):
                    j = p['competitor']
                    if bank[j]['topology_id']==bank[target]['topology_id'] or (representative_only and j not in rep_ids):
                        continue
                    pair_count += 1; gain = p['actual_logodds_change']
                    positive += gain > 1e-8; negative += gain < -1e-8
                    if gain < -1e-8:
                        bad = True; cases[row['id']] += 1
                    if lowest is None or gain < lowest['gain']:
                        lowest = dict(id=row['id'],forecast=row['forecast_index'],repeat=row['repeat'],
                                      competitor=j,gain=gain,forecast_gain=p['forecast_J_gain'])
                negative_paths += bad
            scopes['representatives' if representative_only else 'all_native_parameters'] = dict(
                paths=len(own),pairs=pair_count,positive_pairs=positive,negative_pairs=negative,
                paths_with_any_negative=negative_paths,negative_pairs_per_history=dict(cases),minimum=lowest)
        output[strategy] = scopes
    result = dict(run=io.ref(run/'complete.json'),analysis=io.ref(analysis/'complete.json'),
        rows=io.ref(analysis/'rows_with_bias_split.json'),source=io.ref(__file__),
        definition='Fixed history target versus all parameter points of different internal topologies; representatives are those used by the existing planner.',strategies=output)
    io.write(out,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=check(a.run,a.analysis,a.out);print(r['strategies'].get('old_gamma2'))
