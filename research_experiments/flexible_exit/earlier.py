"""Post-primary timing contrast, reusing the exact frozen numerical worker."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
from .run import ROOT, read, write, sha
from .model import select_block


def freeze(output):
    primary=ROOT/'outputs/flexible_exit_20261101';p=read(primary/'protocol.json')
    assert read(primary/'analysis/verification.json')['passed']
    if output.exists():raise FileExistsError('New timing contrast directory required.')
    output.mkdir(parents=True);shutil.copytree(primary/'source',output/'source');shutil.copy2(primary/'catalog.json',output/'catalog.json')
    starts={};eligible=0
    for job in p['jobs']:
        previous=read(next(path for path in p['reference_hashes'][job] if Path(path).parent.name=='small3_original_exit'))
        old=read(primary/'parents'/job/'start.json');point=previous['trajectory']['point'];block=select_block(point,len(old['labels']))
        eligible+=block is not None;file=output/'parents'/job/'start.json'
        # Charge the complete old mainline too, since validation protection can
        # return its final tree. No claim that this substitutes for old exit.
        write(file,{**old,'point':point,'block':block,
            'upstream_note':'full old mainline charged for unchanged validation protection; diagnostic branch, not standalone timing replacement'})
        starts[job]=sha(file)
        shutil.copy2(primary/'parents'/job/'baseline_choices.json',file.with_name('baseline_choices.json'))
    refs={str(primary/'protocol.json'):sha(primary/'protocol.json'),str(primary/'analysis/verification.json'):sha(primary/'analysis/verification.json')}
    refs.update({str(file):sha(file) for file in (primary/'jobs').glob('*/*/result.json')})
    new={**p,'created_utc':datetime.now(timezone.utc).isoformat(),'eligible':eligible,'start_hashes':starts,
        'stage':'saved prefix endpoint (terminal joint or forward stop), before the original pooled exit',
        'added_after_primary_results':True,'primary_pilot_references':refs,
        'purpose':'same24 inputs, same block rule and exact frozen worker; change only intervention timing',
        'delivery_directory':str(ROOT/'docs/flexible_exit_earlier_20261101'),
        'driver_sha256':sha(Path(__file__)),
        'plan_sha256':sha(ROOT/'docs/flexible_exit_earlier_20261101/PLAN.md')}
    write(output/'protocol.json',new);write(output/'state.json',dict(stage='frozen',planned=96,eligible=eligible))
    print('Frozen earlier timing contrast: same24 inputs, eligible',eligible,flush=True)


if __name__=='__main__':
    import sys
    freeze(Path(sys.argv[1]).resolve())
