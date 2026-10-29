from pathlib import Path
import sys
import traceback
from research_experiments.wzzt_systematic.common import read,write
from .common import observed
from .pipeline import infer


def run(file):
    file=Path(file);p=read(file)
    if p.get('data_error'):raise RuntimeError('physical data unavailable: '+p['data_error'])
    header={k:p[k] for k in ('job_id','panel','case','condition','bundle','N','voltage_sigma','pq_sigma','methods')}
    r=infer(observed(p['train'],p['N']),observed(p['validation']),lambda_ablation=p['lambda_ablation'],
        budget_ablation=p['budget_ablation'],pair_ablation=p['pair_ablation'],
        checkpoint=lambda r:write(file.parent/'checkpoint.json',dict(r,**header,status='checkpoint')))
    write(file.parent/'result.json',dict(r,**header,status='returned'))
    print(p['job_id'],'DONE',round(r['seconds'],2),flush=True)


if __name__=='__main__':
    try:run(sys.argv[1])
    except Exception as e:
        write(Path(sys.argv[1]).parent/'error.json',dict(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise
