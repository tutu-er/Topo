"""Small scalar progress log; observations and coefficients stay out of stdout."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from time import perf_counter

from . import io


class Progress:
    def __init__(self, directory, *, method, loss, mode='text', every=10., stream=None):
        if mode not in ('text', 'json', 'off') or not math.isfinite(every) or every<=0:
            raise ValueError('Progress mode must be text/json/off and interval must be positive')
        self.directory=Path(directory);self.method=method;self.loss=loss
        self.mode=mode;self.every=every;self.stream=sys.stdout if stream is None else stream
        self.started=perf_counter();self.last_print=-float('inf');self.last_stage=None

    def __call__(self, event):
        now=perf_counter();stage=event['stage']
        row=dict(time_utc=datetime.now(timezone.utc).isoformat(),method=self.method,loss=self.loss,
                 stage=stage,wall_seconds=now-self.started,engine_seconds=event.get('elapsed_seconds'))
        for key in ('scope','model','model_name','runtime','in_solver_seconds','callback_event','objective','bound','gap','nodes','incumbent_count',
                    'new_incumbent','iteration','status','selected_index','candidate_count',
                    'native_calls','native_time_limits','native_improvements','validation_mse','error'):
            if key in event:
                value=event[key]
                row[key]=None if isinstance(value,float) and not math.isfinite(value) else value
        if event.get('path') is not None:row['path_points']=len(event['path'])
        detail=event.get('event',{})
        for key in ('accepted','gain','initial_source','current_score','initial_score','current_to_initial_gain',
                    'native_returned_score','native_improved_initial','closed_score','pretrial_QP_count'):
            if key in detail:row[key]=detail[key]
        if detail.get('solver'):
            row['solver_status']=detail['solver'].get('status')
            row['solver_gap']=detail['solver'].get('gap')
        io.write(self.directory/'progress.json',row)
        self.directory.mkdir(parents=True,exist_ok=True)
        with (self.directory/'events.jsonl').open('a',encoding='utf8') as target:
            target.write(json.dumps(row,default=io.plain,ensure_ascii=False,allow_nan=False)+'\n')
        if event.get('point') is not None and stage!='native_progress':
            io.write(self.directory/'checkpoint.json',dict(row,point=event['point'],path=event.get('path'),
                budget_reset_allowed=False,scope='Accepted-path checkpoint; not a saved branch-and-bound tree'))
        force=stage!='native_progress' or now-self.last_print>=self.every
        if self.mode!='off' and force:
            if self.mode=='json':line=json.dumps(row,default=io.plain,ensure_ascii=False,allow_nan=False)
            else:
                def fmt(value):return '—' if value is None else f'{value:.6g}' if isinstance(value,(float,int)) else str(value)
                details=' '.join(f'{key}={fmt(row[key])}' for key in
                    ('model_name','objective','bound','gap','nodes','path_points','accepted','initial_source','current_to_initial_gain',
                     'native_improved_initial','solver_status','solver_gap','selected_index','candidate_count','native_calls',
                     'native_time_limits','native_improvements','validation_mse') if key in row)
                line=f"[{self.method}/{self.loss}] {stage} elapsed={row['wall_seconds']:.1f}s"+((' '+details) if details else '')
            print(line,file=self.stream,flush=True);self.last_print=now
        self.last_stage=stage

    def finish(self, result):
        events=result.get('events',[])
        self(dict(stage='complete',selected_index=result['selected']['index'],
            candidate_count=len(result.get('frozen_candidates',[])),native_calls=len(result.get('MIQP',[])),
            native_time_limits=sum(r.get('status')==9 for r in result.get('MIQP',[])),
            native_improvements=sum(e.get('native_improved_initial',False) for e in events),
            validation_mse=result['selected'].get('validation_mse')))
