"""Select a feasible historical focus without altering either complete model."""
from copy import deepcopy
import numpy as np
from research_experiments.wzzt_dr_selection.dr_frozen_distance.design import build_design as single_design


def choose_design(pool,history):
    """Largest common inventory1..5, then frozen column contrast, then labels.

    Multiple disputed blocks remain in the model difference. A declared added
    block is only the local resource-search focus. No candidate/weight is
    dropped and no physical or test-truth file is read.
    """
    added=pool['dispute_added_supportlabels']
    attempts=[];best=[]
    for units in range(5,0,-1):
        for block in sorted(added,key=lambda x:(len(x),tuple(x))):
            local=deepcopy(pool)
            local['dispute_added_supportlabels']=[block]
            try:
                design=single_design(local,history,units=units)
                q=design['qualification'];chosen=set(q['selected_labels'])
                score=sum(n['unit_response_column_micro'] for n in q['nodes'] if n['node_label'] in chosen)
                best.append((score,tuple(sorted(block)),design))
                attempts.append(dict(units=units,focus=block,status='qualified',score=score))
            except ValueError as error:
                attempts.append(dict(units=units,focus=block,status='not_qualified',reason=str(error)))
        if best:break
    if not best:raise ValueError('no historically supported local resource pair at inventory1..5: '+str(attempts))
    score,block,design=min(best,key=lambda a:(-a[0],a[1]))
    design['qualification'].update(all_added_disputes=deepcopy(added),
        all_removed_disputes=deepcopy(pool['dispute_removed_supportlabels']),
        inventory_rule='largest common integer1..5 feasible for an inside/outside pair, then sum of full-model difference column norms',
        focus_rule='one added historical support for resource search only; both complete R/X unchanged',
        attempts=attempts,chosen_column_score=score)
    return design
