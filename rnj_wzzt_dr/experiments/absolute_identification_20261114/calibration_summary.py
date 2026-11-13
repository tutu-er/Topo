"""Post-experiment model-calibration diagnostics from independently audited rows."""
import argparse
from collections import defaultdict
from pathlib import Path
from scipy.stats import binom
from ... import io


def summarize(audit,inputs,out):
    if out.exists():raise ValueError('Preserve earlier diagnostic')
    verified=io.read(audit/'summary.json')
    if not verified['passed']:raise ValueError('Complete the numerical audit first')
    entries=io.read(inputs);rows=io.read(audit/'certificate_private.json');groups=defaultdict(list)
    modes=sorted({r['mode'] for r in rows});family_size=len(entries)*len(modes)
    for r in rows:
        if r['truth_in_selected_set'] and r['model_lower']>=.9-1e-6:
            groups[(r['id'],r['mode'],r['physics'])].append(r)
    results=[]
    for (identity,mode,physics),own in sorted(groups.items()):
        n=len(own);failed=sum(not r['native_correct'] for r in own)
        tail=float(binom.sf(failed-1,n,.1))
        results.append(dict(id=identity,mode=mode,physics=physics,qualified_days=n,native_failures=failed,
            delivered_model_below90=sum(r['delivered_action_model_lower']<.9-1e-6 for r in own),
            one_sided_tail_under_pointwise_failure_bound10percent=tail,
            bonferroni_tail_over_all_histories_and_modes=min(1.,family_size*tail)))
    io.write(out,dict(source=io.ref(__file__),audit=io.ref(audit/'summary.json'),inputs=io.ref(inputs),
        diagnostic='Checks transfer of each declared Gaussian lower bound to actual independent execution, not validity of the algebraic derivation. Under independent failures with every probability at most0.10, the binomial upper tail is conservative even for unequal probabilities. Results are post-experiment diagnostics; all planned histories/modes enter the family-size adjustment separately per physical model.',
        multiplicity_family_size_per_physics=family_size,groups=results))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=summarize(a.audit.resolve(),a.inputs.resolve(),a.out.resolve());print('histories/modes/physics groups',len(r),'minimum corrected tail',min(v['bonferroni_tail_over_all_histories_and_modes'] for v in r))
