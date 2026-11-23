"""Rank unitary set_bus actions by accepted use with the repository's GreedyAgent.

Example (reserves 20% of scenarios, without creating Grid2Op split directories):
  python collect_filtered_actions.py --env l2rpn_case14_sandbox 
      --episodes 1000 --top-k 100 250 --output-dir action_spaces/case14_run1

For an existing training environment, pass --env NAME_train --train-fraction 1.
Use --test-env --episodes 1 --max-steps 2 for a small bundled-data smoke run.
"""

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from agent import GreedyAgent


class RecordingGreedyAgent(GreedyAgent):
    """Record the selected search candidate before act() adds reconnection."""

    def act(self, observation, reward=None, done=None):
        self.selected_candidate = None
        return super().act(observation, reward, done)

    def get_best_action_sim(self, obs, actions):
        action = super().get_best_action_sim(obs, actions)
        self.selected_candidate = action
        return action


def action_key(action):
    return json.dumps(action.as_serializable_dict(), sort_keys=True, separators=(",", ":"))


def selection_accepted(info):
    """Acceptance concerns the submitted action, independently of terminal outcome."""
    return not any(info.get(flag, False) for flag in (
        'is_illegal', 'is_ambiguous', 'is_illegal_reco', 'failed_redispatching',
    ))


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', default='l2rpn_case14_sandbox', help='Grid2Op name or local environment path')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for rankings and provenance')
    parser.add_argument('--episodes', type=positive_int, default=1000, help='Maximum distinct training chronics to run')
    parser.add_argument('--top-k', type=positive_int, nargs='+', default=[1000, 2000])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-fraction', type=float, default=0.8, help='Fraction of shuffled chronics eligible for collection; use 1 for a pre-split training environment')
    parser.add_argument('--max-steps', type=positive_int, help='Optional per-episode collection cap; default runs to termination')
    parser.add_argument('--n-workers', type=positive_int, default=1, help='Parallel candidate simulations; >1 requires fork support')
    parser.add_argument('--action-thr', type=float, default=0.95)
    parser.add_argument('--revert-thr', type=float, default=0.85)
    parser.add_argument('--backend', choices=['lightsim', 'pandapower'], default='lightsim')
    parser.add_argument('--with-opponent', action='store_true', help='Keep the environment-configured opponent')
    parser.add_argument('--test-env', action='store_true', help='Use Grid2Op bundled test data, not a production dataset')
    args = parser.parse_args()
    
    if not 0 < args.train_fraction <= 1:
        parser.error('--train-fraction must be in (0, 1]')
    
    if not (0 <= args.revert_thr <= args.action_thr) or not np.isfinite(args.action_thr):
        parser.error('thresholds must be finite and satisfy 0 <= revert-thr <= action-thr')
    
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32 - 1')
    
    if args.n_workers > 1 and 'fork' not in mp.get_all_start_methods():
        parser.error('the current GreedyAgent requires fork for multiple workers; use --n-workers 1')
    
    if args.output_dir.exists():
        parser.error('--output-dir must not already exist (prevents overwriting a previous run)')
    return args


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    args = parse_args()
    
    import grid2op
    from grid2op.Opponent import BaseOpponent
    
    if args.backend == 'lightsim':
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
    else:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
    
    kwargs = {} if args.with_opponent else {'opponent_class': BaseOpponent}
    env = grid2op.make(args.env, backend=backend, test=args.test_env, **kwargs)
    
    try:
        chronics = sorted(str(c) for c in env.chronics_handler.available_chronics())
        if not chronics:
            raise ValueError('Environment has no available chronics')
    
        rng = np.random.default_rng(args.seed)
        shuffled = [chronics[i] for i in rng.permutation(len(chronics))]
        n_train = int(len(shuffled) * args.train_fraction)
    
        if n_train == 0:
            raise ValueError('Training split is empty; increase --train-fraction (use 1 for one chronic)')
    
        chosen = shuffled[:min(args.episodes, n_train)]
        print(f'Generating unitary set_bus actions for {env.name}...', flush=True)
    
        candidates = env.action_space.get_all_unitary_topologies_set(env.action_space)
    
        # Deduplicate and remove no-op; keep generator order for the agent's tie-breaking.
        catalog = {}
        for action in candidates:
            if action.as_serializable_dict():
                catalog.setdefault(action_key(action), action)
    
        if not catalog:
            raise ValueError('Environment has no unitary set_bus candidates')
    
        keys = list(catalog)
        key_to_index = {key: i for i, key in enumerate(keys)}
        payloads = [json.loads(key) for key in keys]
    
        agent = RecordingGreedyAgent(env.action_space, list(catalog.values()),
                                    revert_thr=args.revert_thr, action_thr=args.action_thr,
                                    n_workers=args.n_workers)
        accepted_counts = Counter()
        attempted_counts = Counter()
        terminal_counts = Counter()
    
        manifest = {
            'arguments': {**vars(args), 'output_dir': str(args.output_dir)},
            'grid2op_version': grid2op.__version__,
            'agent_sha256': hashlib.sha256(Path(__file__).with_name('agent.py').read_bytes()).hexdigest(),
            'candidate_count': len(catalog),
            'schema_version': 2,
            'ranking_metric': 'accepted_count',
            'counting_rule': 'Search candidates only; accepted means the submitted action has no illegal, ambiguous, illegal-reconnection, or redispatch-failure flags. Terminal steps are included when accepted; acceptance does not imply a safe or successful outcome. Rejection applies to the submitted combination and does not prove the topology candidate alone is invalid. Combined reconnections are not part of the saved candidate. Reversion and no-op are not counted.',
            'training_chronics': shuffled[:n_train],
            'held_out_chronics': shuffled[n_train:],
            'selected_chronics': chosen,
            'equipment_names': {name: getattr(env, name).tolist() for name in
                                ['name_line', 'name_sub', 'name_gen', 'name_load']},
            'episodes': [], 'complete': False,
        }
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / 'candidate_actions.json', payloads)

        def save():
            ranked = sorted(accepted_counts, key=lambda i: (-accepted_counts[i], i))
            attempted = sorted(attempted_counts, key=lambda i: (-accepted_counts[i], i))
            write_json(args.output_dir / 'action_counts.json',
                       [{'candidate_index': i,
                         'accepted_count': accepted_counts[i],
                         'attempted_count': attempted_counts[i],
                         'rejected_count': attempted_counts[i] - accepted_counts[i],
                         'terminal_count': terminal_counts[i],
                         'action': payloads[i]} for i in attempted])
            for k in set(args.top_k):
                write_json(args.output_dir / f'top_actions_{k}.json', [payloads[i] for i in ranked[:k]])
            manifest['observed_action_count'] = len(attempted)
            manifest['accepted_action_count'] = len(ranked)
            write_json(args.output_dir / 'collection_manifest.json', manifest)

        save()
        print(f'{len(catalog)} candidates; collecting {len(chosen)} chronics; {len(shuffled)-n_train} held out.', flush=True)
        
        for chronic in chosen:
            episode_seed = int(rng.integers(0, 2**31 - 1))
            env.set_id(chronic)
            obs = env.reset(seed=episode_seed)
            agent.reset(obs)
            steps = counted = rejected = 0
            done = False
            
            while not done and (args.max_steps is None or steps < args.max_steps):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                steps += 1
                selected = agent.selected_candidate
                if selected is not None:
                    index = key_to_index.get(action_key(selected))
                    if index is not None:
                        attempted_counts[index] += 1
                        terminal_counts[index] += int(done)
                        if selection_accepted(info):
                            accepted_counts[index] += 1
                            counted += 1
                        else:
                            rejected += 1
            manifest['episodes'].append({'chronic': chronic, 'seed': episode_seed, 'steps': steps,
                                         'done': bool(done), 'counted': counted, 'excluded_selections': rejected})
            save()
            print(f'{len(manifest["episodes"])}/{len(chosen)}: {Path(chronic).name}: '
                  f'{steps} steps, {counted} counted, {rejected} excluded', flush=True)
        
        manifest['complete'] = True
        save()
        print(f'Saved {len(accepted_counts)} accepted actions to {args.output_dir}. '
              'Top-K files may contain fewer than K actions; empty files mean no qualifying selections.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
