import tempfile
import unittest
from pathlib import Path
import numpy as np
from dataset_analysis import save_dataset, load_dataset, build_tables, paired_episode_means


def obs(rho, step=0, dead=False):
    return dict(rho=[rho], line_status=[not dead], topo_vect=[-1 if dead else 1],
                load_p=[10], gen_p=[11], current_step=[step], delta_time=[5], max_step=[4])


def episode(legacy=False):
    e = dict(chronic='example', observations=[obs(.8), obs(1.1, 1)],
             actions=[{}, {'set_bus': [2]}], rewards=[0., -1.],
             infos=[{'rewards': {'economic': .5}},
                    {'is_illegal': True, 'rewards': {'economic': .2},
                     'exception': [ValueError('invalid')]}])
    if not legacy:
        e.update(dones=[False, True], final_observation=obs(0., 2, True), env_seed=7)
    return e


class AnalysisTests(unittest.TestCase):
    def test_portable_roundtrip_and_flags(self):
        e = episode()
        e['observations'][0]['rho'] = np.array([.8])
        e['infos'][0]['extra'] = np.float32(np.nan)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'test.jsonl.gz'
            save_dataset({0: e}, path, {'policy': 'test'})
            loaded, meta = load_dataset(path)
            self.assertEqual(meta['policy'], 'test')
            self.assertTrue(np.isnan(loaded['0']['infos'][0]['extra']))
            self.assertEqual(loaded['0']['infos'][1]['exception'][0]['exception_type'], 'ValueError')
            t, s = build_tables({'normal': loaded})
            self.assertAlmostEqual(t.iloc[0].delta_rho, .3)
            self.assertTrue(np.isnan(t.iloc[1].delta_rho))
            self.assertTrue(t.iloc[1].flagged_action)
            self.assertEqual(t.iloc[1].action_type, 'topology')
            self.assertEqual(s.iloc[0].reward_last, -1.)

    def test_legacy_alignment_does_not_cross_episodes(self):
        t, _ = build_tables({'normal': {0: episode(True), 1: episode(True)}})
        self.assertEqual(t.next_state_available.tolist(), [True, False, True, False])
        self.assertTrue(t.loc[[1, 3], 'delta_rho'].isna().all())
        self.assertTrue(t.done.isna().all())

    def test_corrupt_lengths_rejected(self):
        e = episode(); e['rewards'] = [0]
        with self.assertRaises(ValueError):
            build_tables({'normal': {0: e}})

    def test_bootstrap_equal_episode_weight_and_pairing(self):
        import pandas as pd
        e = pd.DataFrame([dict(agent=a, pair_id=p, score=v)
                          for a in ['a', 'b'] for p,v in [('x', 0), ('y', 10)]])
        result = paired_episode_means(e, ['score'])
        self.assertTrue((result['mean'] == 5).all())
        self.assertTrue((result.n_pairs == 2).all())
        with self.assertRaises(ValueError):
            paired_episode_means(pd.concat([e,e.iloc[:1]]), ['score'])

    def test_explicit_curtail_noop(self):
        e = episode(); e['actions'][0] = {'curtail': [-1], 'set_bus': [0]}
        t, _ = build_tables({'normal': {0:e}})
        self.assertEqual(t.iloc[0].action_type, 'no-op')

if __name__ == '__main__':
    unittest.main()
