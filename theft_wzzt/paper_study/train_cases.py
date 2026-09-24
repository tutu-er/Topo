import json
from pathlib import Path
from time import perf_counter
from theft_wzzt.theft.identified_tree import fit_identified_tree, save_identified
OUT=Path(__file__).resolve().parents[1]/'outputs/theft_paper'
for case in ['soumalas11','flynn16']:
    target=OUT/f'identified_{case}.json'
    if target.exists(): continue
    start=perf_counter()
    print('START',case,flush=True)
    try:
        tree=fit_identified_tree(case,time_limit=600.)
        save_identified(tree,target)
        result=dict(case=case,seconds=perf_counter()-start,status='ok',terminals=len(tree.terminals),supports=len(tree.supports),training_replicate=0,validation_replicate=1)
    except Exception as error:
        result=dict(case=case,seconds=perf_counter()-start,status='failed',error=repr(error))
    (OUT/f'train_{case}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result,flush=True)
