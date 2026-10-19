"""Retry transient Windows status-file locks without changing frozen science.

Original controller/worker source hashes, settings and result paths stay intact.
This launch wrapper changes only retries of the same atomic JSON write.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


def retry_write(writer, path, value, *, seconds=10., pause=time.sleep,
                clock=time.monotonic, notify=None):
    began=clock(); attempts=0
    while True:
        try:
            return writer(path,value)
        except PermissionError as error:
            attempts+=1
            if clock()-began >= seconds:
                raise
            if notify is not None:
                notify(dict(path=str(path),attempt=attempts,
                    elapsed_seconds=clock()-began,error=str(error)))
            pause(min(.05*2**min(attempts-1,4),.8))


def main():
    work=Path(__file__).resolve().parents[5]
    sys.path.insert(0,str(work))
    from research_experiments.mobile_ev_mle import ev_ac_repair_control as control
    original=control.atomic_json
    def notification(row):
        print('STATUS_WRITE_RETRY '+json.dumps(row),flush=True)
    def safe(path,value):
        return retry_write(original,path,value,notify=notification)
    control.atomic_json=safe
    control.main()


if __name__=='__main__':
    main()
