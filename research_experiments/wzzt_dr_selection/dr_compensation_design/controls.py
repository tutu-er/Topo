"""Sixteen predeclared price menus under the existing whole-event contract.

This module enumerates or samples issued controls only. It does not read load,
voltage, topology, future observations, or fitted model parameters. Training and
validation each contain two events; the same menu repeats in validation but
acceptance and orientation are drawn independently for each whole event.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, permutations, product
import os
from pathlib import Path
from random import Random


PROBABILITY = (Fraction(0), Fraction(1, 2), Fraction(3, 4))
N = 3


def _integer_seed(seed):
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError('nonnegative integer seed required')
    return int(seed)


def _definition(name, kind, prices, pairs):
    prices = tuple(tuple(map(int, row)) for row in prices)
    pairs = tuple(tuple(map(int, row)) for row in pairs)
    _validate_rows(prices, pairs)
    worst = Fraction(2, 5) * sum(map(sum, prices))
    return dict(id=name, kind=kind, price_rows=prices, pair_rows=pairs,
                worst_payment=float(worst), terminal_count=N,
                training_events=2, validation_events=2,
                validation_repeats_prices_not_random_outcomes=True)


def _validate_rows(prices, pairs):
    if len(prices) != 2 or any(len(row) != N or any(x not in (0, 1, 2) for x in row) for row in prices):
        raise ValueError('two rows of three prices in {0,1,2} required')
    if len(pairs) != 2 or any(len(row) != 2 or row[0] == row[1] or any(x not in range(N) for x in row) for row in pairs):
        raise ValueError('two rows of distinct declared endpoints required')
    if any(prices[e][i] and i not in pairs[e] for e in range(2) for i in range(N)):
        raise ValueError('all offered nodes must be in the declared pair')
    if Fraction(2, 5) * sum(map(sum, prices)) > Fraction(8, 5):
        raise ValueError('four-event worst payment exceeds 1.6')


def menus():
    """Fresh menu dictionaries: zero, six singles, six rotations, three pairs."""
    result = [_definition('zero', 'zero', ((0, 0, 0),) * 2, ((0, 1),) * 2)]
    for i in range(N):
        for price in (1, 2):
            row = tuple(price if j == i else 0 for j in range(N))
            result.append(_definition(f'single_{i}_p{price}', f'single{price}',
                                      (row, row), ((i, (i + 1) % N),) * 2))
    for i, j in permutations(range(N), 2):
        rows = [tuple(2 if k == active else 0 for k in range(N)) for active in (i, j)]
        pairs = [(active, (active + 1) % N) for active in (i, j)]
        result.append(_definition(f'rotation_{i}_{j}_p2', 'rotation', rows, pairs))
    for i, j in combinations(range(N), 2):
        row = tuple(int(k in (i, j)) for k in range(N))
        result.append(_definition(f'pair_{i}_{j}_p1', 'pair1', (row, row), ((i, j),) * 2))
    return result


def _menu(menu):
    if isinstance(menu, str):
        found = [v for v in menus() if v['id'] == menu]
        if not found:
            raise ValueError('unknown menu ID')
        return found[0]
    if not isinstance(menu, dict):
        raise ValueError('menu dictionary or ID required')
    prices = tuple(tuple(row) for row in menu['price_rows'])
    pairs = tuple(tuple(row) for row in menu['pair_rows'])
    _validate_rows(prices, pairs)
    return dict(menu, price_rows=prices, pair_rows=pairs)


def _acceptance_patterns(prices, dependence):
    """Exact Bernoulli law; common means one common uniform per event."""
    rates = tuple(PROBABILITY[x] for x in prices)
    if dependence == 'local':
        out = []
        for accepted in product((0, 1), repeat=N):
            probability = Fraction(1)
            for z, p in zip(accepted, rates):
                probability *= p if z else 1 - p
            if probability:
                out.append((accepted, probability))
        return out
    if dependence == 'common':
        cuts = sorted({Fraction(0), Fraction(1), *rates})
        out = {}
        for lo, hi in zip(cuts, cuts[1:]):
            if hi == lo:
                continue
            u = (lo + hi) / 2
            accepted = tuple(int(u < p) for p in rates)
            out[accepted] = out.get(accepted, Fraction(0)) + hi - lo
        return list(out.items())
    raise ValueError('dependence must be local or common')


def _event_law(prices, pair, dependence):
    result = {}
    first, second = pair
    for accepted, probability in _acceptance_patterns(prices, dependence):
        for orientation in (-1, 1):
            direction = [0] * N
            direction[first] = orientation * accepted[first]
            direction[second] = -orientation * accepted[second]
            key = tuple(direction)
            result[key] = result.get(key, Fraction(0)) + probability / 2
    return result


@lru_cache(maxsize=64)
def _state_law(prices, pairs, dependence):
    first, second = [_event_law(row, pair, dependence) for row, pair in zip(prices, pairs)]
    result = {}
    for (s0, p0), (s1, p1) in product(first.items(), second.items()):
        state = (s0, s1)
        result[state] = result.get(state, Fraction(0)) + p0 * p1
    assert sum(result.values(), Fraction(0)) == 1
    return tuple(sorted(result.items()))


def enumerate_states(menu, dependence='local'):
    """Return [(two_event_direction_state, probability)], merging zero codes.

    A direction is accepted * orientation * declared endpoint sign, so every
    entry is -1, 0 or 1. The waveform itself remains the original 8-quarter task.
    This law applies separately to training and validation.
    """
    m = _menu(menu)
    return [(s, float(p)) for s, p in _state_law(m['price_rows'], m['pair_rows'], dependence)]


def _state(state):
    try:
        state = tuple(tuple(row) for row in state)
    except TypeError as exc:
        raise ValueError('state must be two rows of three signed integers') from exc
    if len(state) != 2 or any(len(row) != N or any(isinstance(x, bool) or x not in (-1, 0, 1) or int(x) != x for x in row) for row in state):
        raise ValueError('state must be two rows of three signed integers')
    return tuple(tuple(map(int, row)) for row in state)


def probability_of_training_state(menu, train_state, dependence='local'):
    m = _menu(menu)
    law = dict(_state_law(m['price_rows'], m['pair_rows'], dependence))
    return float(law.get(_state(train_state), Fraction(0)))


def probability_of_state(menu, train_state, val_state, dependence='local'):
    """Joint probability: validation is an independent draw of the same rows."""
    m = _menu(menu)
    law = dict(_state_law(m['price_rows'], m['pair_rows'], dependence))
    return float(law.get(_state(train_state), Fraction(0)) * law.get(_state(val_state), Fraction(0)))


def original_source():
    repo = Path(os.environ.get('WZZT_REPO_ROOT', r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
    path = repo / 'research_experiments/wzzt_dr_selection/dr_random_observation_gate/random_code.py'
    return dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest())


@lru_cache(maxsize=1)
def _original_sampler():
    # Execute the unchanged source in an isolated namespace without writing a
    # __pycache__ into the frozen research directory or changing sys.path.
    source = original_source()
    path = Path(source['path'])
    namespace = {'__name__': '_frozen_wzzt_random_control', '__file__': str(path)}
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), namespace)
    return namespace['sample_controls']


def sample_menu(menu, seed, dependence='local', base_kw=200.):
    """Call the original unchanged sampler; returns issued controls, not loads."""
    m = _menu(menu)
    seed = _integer_seed(seed)
    draw = _original_sampler()(m['price_rows'], m['pair_rows'], seed=seed,
                               dependence=dependence, base_kw=base_kw)
    return dict(draw, menu_id=m.get('id', 'custom'), menu_kind=m.get('kind', 'custom'))


def states_from_draw(draw):
    """Extract separate training/validation states, keeping whole events."""
    states = []
    if len(draw['orientation']) != 4:
        raise ValueError('four original whole events required')
    for e in range(4):
        a, b = map(int, draw['pairs'][e])
        row = [0] * N
        row[a] = int(draw['orientation'][e]) * int(draw['accepted'][e][a])
        row[b] = -int(draw['orientation'][e]) * int(draw['accepted'][e][b])
        states.append(tuple(row))
    return tuple(states[:2]), tuple(states[2:])


def baseline_menu(kind, seed):
    """Uniform menu selection before outcomes; supported fair exploration arms."""
    seed = _integer_seed(seed)
    if kind not in ('single2', 'rotation', 'pair1'):
        raise ValueError('baseline kind must be single2, rotation or pair1')
    choices = [m for m in menus() if m['kind'] == kind]
    named_seed = int.from_bytes(sha256(f'wzzt-baseline|{kind}|{seed}'.encode()).digest(), 'big')
    return choices[Random(named_seed).randrange(len(choices))]
