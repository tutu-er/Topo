"""Small reproducible runtime audit; writes only into this document's evidence."""
from __future__ import annotations
import json
import platform
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ROOT / 'rnj_wzzt_core'
sys.path.insert(0, str(CORE))
import numpy as np
import pandas as pd
import scipy
import networkx as nx
from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.estimation.laminar_l1_milp import evaluate_l1_matrices
from rnj_wzzt.pipeline import run

facts = {'python': sys.version, 'platform': platform.platform(), 'versions': {m.__name__: m.__version__ for m in (np, pd, scipy, nx)}, 'cases': []}
for name, builder in CASE_BUILDERS.items():
    net = builder()
    report = net.check_terminal_load_only(strict=True)
    facts['cases'].append({'case': name, 'nodes': len(net.buses), 'edges': len(net.branches), 'terminals': report['observed_terminal_count'], 'hidden': report['hidden_internal_count'], 'base_kv': net.base_kv, 'base_mva': net.base_mva, 'legacy_scale': net.metadata['impedance_scale'], 'violations': report['violations']})

# Existing low-level L1 API aligns columns, but does not align rows by index.
p = pd.DataFrame({101: [1., 2., 3.]}, index=['a', 'b', 'c'])
q = p * 0
target = p * 2
common = {'name': 'row_alignment_probe', 'P_terminal': p, 'Q_terminal': q}
aligned = evaluate_l1_matrices([{**common, 'drop_target': target}], np.array([[2.]]), np.array([[0.]]), fixed_intercepts=np.zeros((1, 1)))[0]
permuted = evaluate_l1_matrices([{**common, 'drop_target': target.iloc[::-1]}], np.array([[2.]]), np.array([[0.]]), fixed_intercepts=np.zeros((1, 1)))[0]
facts['l1_row_alignment_probe'] = {'aligned_mae': aligned, 'same_labeled_rows_reversed_mae': permuted, 'interpretation': 'L1 direct callers must synchronize row order; labels alone do not align time rows.'}
facts['smoke_config'] = {'cases': ['soumalas11'], 'scenario_suite': 'reference', 'scenario_count': 1, 'samples_per_scenario': 8, 'bootstrap_replicates': 2, 'time_limit': 5.0, 'selection_only': False, 'run_baseline': False, 'contract_blocks': True, 'candidate_pool_mode': 'rnj'}
payload = run(HERE / 'evidence' / 'smoke', **{**facts['smoke_config'], 'cases': tuple(facts['smoke_config']['cases'])})
facts['smoke_result'] = payload
loaded = []
for name, module in sorted(sys.modules.items()):
    file = getattr(module, '__file__', None)
    if file:
        try:
            relative = Path(file).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            continue
        if name == '__main__':
            continue
        loaded.append({'module': name, 'path': relative})
facts['loaded_repository_modules'] = loaded
facts['noncore_repository_imports'] = [x for x in loaded if not x['path'].startswith('rnj_wzzt_core/')]
(HERE / 'evidence' / 'runtime_audit.json').write_text(json.dumps(facts, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
print(json.dumps({'cases': facts['cases'], 'row_alignment': facts['l1_row_alignment_probe'], 'noncore_imports': facts['noncore_repository_imports'], 'smoke': payload['milp']}, ensure_ascii=False, indent=2))
