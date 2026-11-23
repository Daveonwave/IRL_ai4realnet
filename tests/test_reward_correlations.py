"""Check the notebook's actual helper functions without loading large archives."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

nb = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/02_reward_correlations.ipynb').read_text())
namespace = dict(np=np, pd=pd, ELEMENT_FEATURES={}, DENSE_REWARDS=['economic', 'distance', 'closeToOverflow'], MIN_STEPS=3)
for index in [3, 5, 11]:
    tree = ast.parse(''.join(nb['cells'][index]['source']))
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    exec(compile(functions, 'notebook_helpers', 'exec'), namespace)


def obs(rho, bus=1):
    return dict(rho=[rho], topo_vect=[bus], line_status=[bus != -1],
                load_p=[10], gen_p=[11], current_step=[0], delta_time=[5])


class CorrelationTests(unittest.TestCase):
    def test_constant_is_undefined(self):
        correlation, n = namespace['safe_correlation']([1, 1, 1], [0, 1, 2])
        self.assertTrue(np.isnan(correlation)); self.assertEqual(n, 3)

    def test_rank_ties_and_missing_pairs(self):
        correlation, n = namespace['safe_correlation']([1, 1, 2, np.nan], [3, 3, 4, 10])
        self.assertAlmostEqual(correlation, 1.); self.assertEqual(n, 3)

    def test_episode_transition_alignment(self):
        e = dict(chronic='x', observations=[obs(.8), obs(.9)], actions=[{}, {}],
                 rewards=[0., 1.], infos=[{'rewards': {'economic': .3}}, {'rewards': {'economic': .4}}])
        rows = list(namespace['episode_rows']('normal', '0', e))
        self.assertEqual(rows[0]['obs__rho_max'], .8)
        self.assertEqual(rows[0]['post__rho_max'], .9)
        self.assertEqual(rows[0]['reward_economic'], .3)
        self.assertNotIn('post__rho_max', rows[1])
        self.assertFalse(rows[1]['terminal_flag_available'])
        self.assertTrue(rows[1]['is_last'])

    def test_terminal_state_not_healthy(self):
        self.assertEqual(namespace['observation_features'](obs(0, -1)), {})

    def test_action_requests_and_noop(self):
        features = namespace['action_features']({'curtail': [-1], 'set_bus': [0]}, obs(.5))
        self.assertEqual(features['intervention'], 0)
        features = namespace['action_features']({'set_bus': [1]}, obs(.5))
        self.assertEqual(features['bus_commands'], 1)
        self.assertEqual(features['bus_changes'], 0)

if __name__ == '__main__':
    unittest.main()
