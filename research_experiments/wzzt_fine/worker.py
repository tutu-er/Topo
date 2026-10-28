from pathlib import Path
import sys
import traceback
from research_experiments.wzzt_systematic.common import read,write
from .common import observed
from .pipeline import infer


def run(payload):
    payload=Path(payload);p=read(payload)
    header={k:p[k] for k in ("job_id","case","bundle","condition","N","voltage_sigma","pq_sigma","targeted")}
    result=infer(observed(p["train"],p["N"]),observed(p["validation"]),targeted=p["targeted"],
        checkpoint=lambda r:write(payload.parent/"checkpoint.json",dict(r,**header,status="checkpoint")))
    result.update({k:p[k] for k in ("job_id","case","bundle","condition","N","voltage_sigma","pq_sigma","targeted")})
    result["status"]="returned";write(payload.parent/"result.json",result)
    print(p["job_id"],"DONE",round(result["seconds"],2),flush=True)


if __name__=="__main__":
    try:run(sys.argv[1])
    except Exception as e:
        write(Path(sys.argv[1]).parent/"error.json",dict(error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc()));raise
