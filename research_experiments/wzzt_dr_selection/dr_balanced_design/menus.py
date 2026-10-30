"""Twenty-two predeclared menus using the unchanged stochastic DR contract.

Only six ordered different-pair schedules are added to the frozen sixteen-menu
library. This module never reads loads, voltages, a fitted model, or truth.
"""
from __future__ import annotations

import importlib.util
from hashlib import sha256
from itertools import combinations, permutations
import os
from pathlib import Path
from random import Random

REPO = Path(os.environ.get('WZZT_REPO_ROOT', r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
FROZEN_SOURCE = REPO / 'research_experiments/wzzt_dr_selection/dr_compensation_design/controls.py'
_spec = importlib.util.spec_from_file_location('_wzzt_b_frozen_controls', FROZEN_SOURCE)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
PROBABILITY = _old.PROBABILITY
N = _old.N


def menus():
    """Original 16, then all 3*2 ordered schedules of distinct opposite pairs."""
    result = _old.menus()
    for first, second in permutations(tuple(combinations(range(N), 2)), 2):
        rows = [tuple(int(i in pair) for i in range(N)) for pair in (first, second)]
        name = f'pair_rotation_{first[0]}_{first[1]}__{second[0]}_{second[1]}_p1'
        result.append(_old._definition(name, 'pair_rotation', rows, (first, second)))
    return result


def resolve(menu):
    if isinstance(menu, str):
        found = [m for m in menus() if m['id'] == menu]
        if not found:
            raise ValueError('unknown menu ID')
        return found[0]
    return _old._menu(menu)


def enumerate_states(menu, dependence='local'):
    return _old.enumerate_states(resolve(menu), dependence)


def probability_of_training_state(menu, state, dependence='local'):
    return _old.probability_of_training_state(resolve(menu), state, dependence)


def probability_of_state(menu, train_state, val_state, dependence='local'):
    return _old.probability_of_state(resolve(menu), train_state, val_state, dependence)


def sample_menu(menu, seed, dependence='local', base_kw=200.):
    return _old.sample_menu(resolve(menu), seed, dependence, base_kw)


def states_from_draw(draw):
    return _old.states_from_draw(draw)


def baseline_menu(kind, seed):
    if kind != 'pair_rotation':
        return _old.baseline_menu(kind, seed)
    seed = _old._integer_seed(seed)
    choices = [m for m in menus() if m['kind'] == kind]
    named = int.from_bytes(sha256(f'wzzt-b-baseline|{kind}|{seed}'.encode()).digest(), 'big')
    return choices[Random(named).randrange(len(choices))]


def expected_payment(menu):
    """Four-event expected delivery payment, not a constraint surrogate."""
    m = resolve(menu)
    return float(sum(c * PROBABILITY[c] for row in m['price_rows'] for c in row) * _old.Fraction(2, 5))


def source_provenance():
    return dict(frozen_controls=dict(path=str(FROZEN_SOURCE), sha256=sha256(FROZEN_SOURCE.read_bytes()).hexdigest()),
                frozen_sampler=_old.original_source(),
                extension='six ordered different opposite-pair schedules; no contract change')
