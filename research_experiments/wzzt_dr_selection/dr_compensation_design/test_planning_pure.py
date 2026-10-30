"""Independent pure checks of compensation algebra and the price replay.

No test constructs PoolLP, calls a solver, generates a future load/voltage, or
reads a saved trajectory. Algebra uses explicitly declared synthetic arrays.
"""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import controls
import planning as p
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import select as original_select


LEAVES = [[0], [1], [2]]
CAL = dict(a=1., b=1., median_root_depth=1.)


def point(pair=(0, 1), c=.15):
    return dict(supports=deepcopy(LEAVES)+[[0, 1, 2], list(pair)],
                r=[1., 1., 1., .2, c], x=[0.] * 5,
                objective=0., validation_mae=0.)


class PlanningPureTests(unittest.TestCase):
    def test_synchronous_input_has_exact_weight_compensation(self):
        scalar = np.array([.9, 1., 1.1, .85])
        load = np.repeat(scalar[:, None], 3, axis=1)
        first = point(); second = deepcopy(first)
        delta = .03
        second['r'][0] -= 2*delta
        second['r'][1] -= 2*delta
        second['r'][-1] += delta
        np.testing.assert_allclose(p.predict(load, .35*load, first),
                                   p.predict(load, .35*load, second), atol=1e-14, rtol=0)
        self.assertTrue(all(0 <= w <= 2 for w in second['r']+second['x']))

    def test_node_two_breaks_compensation_node_three_does_not(self):
        load = np.repeat(np.array([.9, 1., 1.1, .85])[:, None], 3, axis=1)
        first = point(); second = deepcopy(first); delta = .03
        second['r'][0] -= 2*delta; second['r'][1] -= 2*delta; second['r'][-1] += delta
        u = np.array([.1, -.04, .03, -.02])
        at_two = load.copy(); at_two[:, 1] += u
        actual = p.predict(at_two, .35*at_two, second)-p.predict(at_two, .35*at_two, first)
        np.testing.assert_allclose(actual, u[:, None]*np.array([delta, -delta, 0.]), atol=1e-14, rtol=0)
        at_three = load.copy(); at_three[:, 2] += u
        np.testing.assert_allclose(p.predict(at_three, .35*at_three, first),
                                   p.predict(at_three, .35*at_three, second), atol=1e-14, rtol=0)

    def test_prediction_jacobian_is_actual_fixed_family_feature(self):
        load = np.array([[.1, .2, .3], [.3, .1, .2], [.2, .3, .1]])
        reactive = .35*load
        base = point(); family = p.family(base)
        ordered = dict(supports=[list(s) for s in family], r=[.2]*len(family), x=[.1]*len(family))
        feature = p.features(load, reactive, family)
        epsilon = 1e-6
        for j in range(len(family)):
            for offset, name in ((0, 'r'), (len(family), 'x')):
                changed = deepcopy(ordered); changed[name][j] += epsilon
                difference = (p.predict(load, reactive, changed)-p.predict(load, reactive, ordered)).ravel()/epsilon
                np.testing.assert_allclose(difference, feature[:, j+offset], atol=1e-9, rtol=0)

    def test_fixed_power_factor_keeps_rx_compensation(self):
        load = np.array([[.1, .2, .3], [.3, .1, .2]])
        first = point(); first['x'][-1] = .1
        second = deepcopy(first); t = .04
        second['r'][-1] += .35*t; second['x'][-1] -= t
        np.testing.assert_allclose(p.predict(load, .35*load, first),
                                   p.predict(load, .35*load, second), atol=1e-14, rtol=0)

    def test_zero_and_weak_target_not_output_contracted(self):
        zero = point(c=0.); weak = point(c=.01)
        self.assertNotIn((0, 1), p.effective(zero))
        self.assertIn((0, 1), p.effective(weak))
        final, raw, index = p.choose([weak], CAL)
        self.assertEqual(index, 0)
        self.assertIn((0, 1), p.effective(raw))
        self.assertNotIn((0, 1), p.effective(final))
        self.assertEqual(p.errors(final, p.effective(weak)), 1)

    def test_state_forecast_matches_original_issued_controls(self):
        mean = np.repeat((.005 + np.arange(96)*1e-6)[:, None], 3, axis=1)
        for menu in controls.menus():
            draw = controls.sample_menu(menu, seed=71193, dependence='local')
            for event_group, state in zip(((0, 1), (2, 3)), controls.states_from_draw(draw)):
                load, reactive = p.input_for_state(mean, .35, state)
                self.assertEqual(load.shape, (192, 3))
                expected = np.tile(mean, (2, 1))
                for local_event, original_event in enumerate(event_group):
                    start = local_event*96+76
                    expected[start:start+8] += draw['delta_p_kw'][original_event]/200.
                np.testing.assert_allclose(load, expected, atol=1e-15, rtol=0)
                np.testing.assert_allclose(reactive, .35*expected, atol=1e-15, rtol=0)

    def test_state_price_only_changes_probability_not_conditional_input(self):
        mean = np.full((96, 3), .005)
        state = ((1, 0, 0), (0, 0, 0))
        one = next(m for m in controls.menus() if m['id'] == 'single_0_p1')
        two = next(m for m in controls.menus() if m['id'] == 'single_0_p2')
        self.assertNotEqual(controls.probability_of_training_state(one, state),
                            controls.probability_of_training_state(two, state))
        first = p.input_for_state(mean, .35, state)
        second = p.input_for_state(mean, .35, state)
        for a, b in zip(first, second): np.testing.assert_array_equal(a, b)

    def test_expected_payment_four_events(self):
        for menu in controls.menus():
            price = np.asarray(menu['price_rows'])
            formula = .4 * np.sum(price * np.array([0., .5, .75])[price])
            law = controls.enumerate_states(menu)
            from_distribution = sum(prob * .4 * np.sum(price * np.abs(state)) for state, prob in law)
            self.assertAlmostEqual(float(formula), float(from_distribution), places=14)
            self.assertLessEqual(float(formula), menu['worst_payment']+1e-14)

    def test_choose_matches_original_three_percent_then_contract(self):
        cases = []
        leaf = dict(supports=deepcopy(LEAVES), r=[1., 1., 1.], x=[0., 0., 0.], validation_mae=1.)
        root = dict(supports=deepcopy(LEAVES)+[[0, 1, 2]], r=[1., 1., 1., .2], x=[0.]*4, validation_mae=.99)
        weak = point(c=.01); weak['validation_mae'] = .98
        cases.append([leaf, root, weak])
        for val in (1.03-2e-12, 1.03, 1.03+2e-12):
            simple = deepcopy(leaf); simple['validation_mae'] = val
            complicated = deepcopy(root); complicated['validation_mae'] = 1.
            cases.append([simple, complicated])
        duplicate = deepcopy(root); duplicate['validation_mae'] = root['validation_mae']
        cases.append([root, duplicate])
        for points in cases:
            original = original_select(deepcopy(points), CAL)
            selected, raw, index = p.choose(deepcopy(points), CAL)
            self.assertEqual(index, original['selection_index'])
            for key in ('supports', 'r', 'x'):
                self.assertEqual(selected[key], original[key])
                self.assertEqual(raw[key], original['selected_uncontracted'][key])

    def test_candidate_crossing_deletion_preserves_legality(self):
        for F in p.base.legal_families(3):
            for C in p.base.internal_universe(3):
                if C in F: continue
                G = p.base.canonical([s for s in F if p.base.compatible(s, C)]+[C])
                self.assertEqual(p.base.validate_supports(G, 3, family=True), G)
                self.assertIn(C, G)
                self.assertTrue(all(p.base.compatible(s, C) for s in G))

    def test_validation_array_not_shadowed_by_loop_probability(self):
        source = ast.parse(Path(p.__file__).read_text(encoding='utf-8'))
        run = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        validation_names = []
        for node in run.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == 'arrays' and call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == 'valid':
                    validation_names.extend(item.id for target in node.targets if isinstance(target, (ast.Tuple, ast.List)) for item in target.elts if isinstance(item, ast.Name))
        loop_targets = {node.id for loop in ast.walk(run) if isinstance(loop, (ast.For, ast.comprehension))
                        for node in ast.walk(loop.target) if isinstance(node, ast.Name)}
        self.assertTrue(validation_names)
        self.assertFalse(set(validation_names) & loop_targets,
                         'validation array overwritten by a loop scalar: '+str(set(validation_names) & loop_targets))

    def test_tie_selection_uses_independent_seed_all_ties(self):
        stage = Path(__file__).resolve().parent
        source = ast.parse((stage/'study.py').read_text(encoding='utf-8'))
        run = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        selection_loop = next(node for node in ast.walk(run) if isinstance(node, ast.For)
                              and isinstance(node.target, ast.Name) and node.target.id == 'flow'
                              and isinstance(node.iter, ast.Subscript) and isinstance(node.iter.value, ast.Name)
                              and node.iter.value.id == 'cfg')
        module = ast.fix_missing_locations(ast.Module(body=[deepcopy(selection_loop)], type_ignores=[]))
        ties = ['single_0_p2', 'single_1_p2', 'single_2_p2']
        byid = {m['id']: m for m in controls.menus()}
        flows = [dict(id=str(i), menu_seed=i, baseline_seed=900+i) for i in range(32)]
        namespace = dict(np=np, cfg=dict(actual_flows=flows), actions=[], byid=byid,
                         ties=ties, baseline_menu=controls.baseline_menu)
        exec(compile(module, '<actual study tie loop>', 'exec'), namespace)
        observed = [row['menus']['guided']['id'] for row in namespace['actions']]
        expected = [ties[int(np.random.default_rng(flow['menu_seed']).integers(len(ties)))] for flow in flows]
        self.assertEqual(observed, expected)
        self.assertEqual(set(observed), set(ties))


if __name__ == '__main__':
    unittest.main(verbosity=2)
