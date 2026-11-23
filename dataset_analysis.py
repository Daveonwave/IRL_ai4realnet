"""Portable episode storage and transition summaries for notebook 01.

No simulator is needed to load/analyze archives. Archives retain numeric observations,
requested action components, rewards and info; they are not live Grid2Op objects.
"""
from pathlib import Path
import gzip
import json
import math
import os
import numpy as np
import pandas as pd

ACTION_FIELDS = {
    'set_bus': '_set_topo_vect', 
    'change_bus': '_change_bus_vect',
    'set_line_status': '_set_line_status', 
    'change_line_status': '_switch_line_status',
    'redispatch': '_redispatch', 
    'curtail': '_curtail',
    'storage_power': '_storage_power', 
    'raise_alarm': '_raise_alarm',
}


def portable(value):
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [portable(v) for v in value]
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'__nonfinite__': str(value)}
    if isinstance(value, BaseException):
        return {'exception_type': type(value).__name__, 'message': str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported archive value: {type(value).__name__}')


def _decode(d):
    if set(d) == {'__nonfinite__'}:
        return float(d['__nonfinite__'])
    return d


def observation_dict(obs):
    return obs if isinstance(obs, dict) else obs.to_json()


def action_dict(action):
    if isinstance(action, dict):
        return action
    return {name: getattr(action, attr).copy() for name, attr in ACTION_FIELDS.items()}


def normalize_episode(episode):
    n = len(episode['actions'])
    for key in ['observations', 'rewards', 'infos']:
        if len(episode[key]) != n:
            raise ValueError(f'{key} length does not match {n} transitions')
    for key in ['dones', 'next_observations']:
        if key in episode and len(episode[key]) != n:
            raise ValueError(f'{key} length does not match {n} transitions')
    out = dict(episode)
    out['observations'] = [observation_dict(o) for o in episode['observations']]
    out['actions'] = [action_dict(a) for a in episode['actions']]
    if 'next_observations' in episode:
        out['next_observations'] = [observation_dict(o) if o is not None else None
                                    for o in episode['next_observations']]
    if episode.get('final_observation') is not None:
        out['final_observation'] = observation_dict(episode['final_observation'])
    return out


def save_dataset(dataset, path, metadata=None):
    """Stream one episode per JSON line; replace the archive only after success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with gzip.open(tmp, 'wt', encoding='utf-8') as stream:
            header = {'schema_version': 1, 'metadata': metadata or {}, 'episodes': len(dataset)}
            stream.write(json.dumps(portable(header), allow_nan=False) + '\n')
            for key, episode in dataset.items():
                payload = {'episode_id': str(key), 'episode': normalize_episode(episode)}
                stream.write(json.dumps(portable(payload), allow_nan=False) + '\n')
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_dataset(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream), object_hook=_decode)
        if header.get('schema_version') != 1:
            raise ValueError('Unsupported dataset schema')
        episodes = {}
        for line in stream:
            item = json.loads(line, object_hook=_decode)
            if item['episode_id'] in episodes:
                raise ValueError('Duplicate episode ID')
            episodes[item['episode_id']] = normalize_episode(item['episode'])
    if len(episodes) != header['episodes']:
        raise ValueError('Incomplete archive')
    return episodes, header['metadata']


def array(obs, key):
    return np.asarray(obs.get(key, []), dtype=float).reshape(-1)


def scalar(obs, key, default=np.nan):
    a = array(obs, key)
    return float(a[0]) if a.size else default


def state_features(obs):
    if obs is None:
        return dict(max_rho=np.nan, overloaded=np.nan, disconnected=np.nan,
                    bus2=np.nan, load_mw=np.nan, generation_mw=np.nan, valid=False)
    rho, status, topo = (array(obs, k) for k in ['rho', 'line_status', 'topo_vect'])
    valid = bool(rho.size and np.isfinite(rho).all() and
                 not (status.size and not status.any() and topo.size and (topo == -1).all()))
    return dict(max_rho=float(rho.max()) if valid else np.nan,
                overloaded=int((rho > 1).sum()) if valid else np.nan,
                disconnected=int((status == 0).sum()) if status.size else np.nan,
                bus2=int((topo == 2).sum()) if topo.size else np.nan,
                load_mw=float(array(obs, 'load_p').sum()),
                generation_mw=float(array(obs, 'gen_p').sum()), valid=valid)


def action_features(action, obs):
    """Classify requested commands, not inferred execution or the agent's intent."""
    a = action_dict(action)
    active = []
    count = 0
    for name in ACTION_FIELDS:
        v = np.asarray(a.get(name, []))
        mask = (v != -1) if name == 'curtail' else (v != 0)
        if mask.any():
            active.append(name)
            count += int(mask.sum())
    topo_request = any(k in active for k in ['set_bus', 'change_bus'])
    line_request = any(k in active for k in ['set_line_status', 'change_line_status'])
    if not active:
        kind = 'no-op'
    elif topo_request and not line_request and set(active) <= {'set_bus', 'change_bus'}:
        kind = 'topology'
    elif line_request and not topo_request and set(active) <= {'set_line_status', 'change_line_status'}:
        kind = 'line status'
    else:
        kind = 'combined / other'
    set_bus = np.asarray(a.get('set_bus', []))
    topo = array(obs, 'topo_vect')
    effective_targets = ((set_bus != 0) & (set_bus != topo)) if set_bus.size == topo.size else np.array([], bool)
    return dict(action_type=kind, active_action=bool(active), requested_components=count,
                requested_bus_changes=int(effective_targets.sum()),
                requests_bus1_only=bool(effective_targets.any() and (set_bus[effective_targets] == 1).all()))




def build_tables(datasets, selected_episodes:list[str]=None):
    transitions, episodes = [], []
    for agent, dataset in datasets.items():
        for episode_id, raw in dataset.items():
            if selected_episodes is not None and episode_id not in selected_episodes:
                continue
            e = normalize_episode(raw)
            obs_list, actions = e['observations'], e['actions']
            n = len(actions)
            chronic = str(e.get('chronic', episode_id))
            seed = e.get('env_seed')
            
            # Missing legacy seeds remain explicitly unknown, no claim of identical RNG state.
            pair_id = chronic + (f"|seed={seed}" if seed is not None else "")
            local = []
            for t, (obs, action, reward, info) in enumerate(zip(obs_list, actions, e['rewards'], e['infos'])):
                if 'next_observations' in e:
                    nxt = e['next_observations'][t]
                else:
                    nxt = obs_list[t+1] if t+1 < n else e.get('final_observation')
                done = bool(e['dones'][t]) if 'dones' in e else None
                sf, nf = state_features(obs), state_features(nxt)
                record = dict(agent=agent, episode_id=str(episode_id), chronic=chronic,
                              env_seed=seed, pair_id=pair_id, t=t,
                              step=scalar(obs, 'current_step', t),
                              hours=scalar(obs, 'current_step', t)*scalar(obs, 'delta_time', 5)/60,
                              reward=float(reward), done=done,
                              next_state_available=nxt is not None,
                              next_max_rho=nf['max_rho'], **sf, **action_features(action, obs))
                record['flagged_action'] = any(bool(info.get(k, False)) for k in
                    ['is_illegal', 'is_ambiguous', 'failed_redispatching', 'is_illegal_reco'])
                record['is_illegal'] = bool(info.get('is_illegal', False))
                record['is_ambiguous'] = bool(info.get('is_ambiguous', False))
                # Unknown legacy final state cannot be fabricated from the preceding state.
                record['delta_rho'] = (nf['max_rho'] - sf['max_rho']) if sf['valid'] and nf['valid'] and done is not True else np.nan
                for name, value in info.get('rewards', {}).items():
                    record['reward_' + name] = float(value)
                transitions.append(record)
                local.append(record)
            
            if not local:
                continue
            
            frame = pd.DataFrame(local)
            horizon = scalar(obs_list[0], 'max_step')
            summary = dict(agent=agent, episode_id=str(episode_id), chronic=chronic,
                           env_seed=seed, pair_id=pair_id, steps=n, horizon=horizon,
                           observed_horizon_fraction=n/horizon if horizon > 0 else np.nan,
                           terminal_recorded='dones' in e and bool(e['dones'][-1]),
                           final_state_available=e.get('final_observation') is not None or 'next_observations' in e,
                           mean_max_rho=frame.max_rho.mean(),
                           overload_fraction=frame.loc[frame.valid, 'overloaded'].gt(0).mean(),
                           action_rate=frame.active_action.mean(),
                           flagged_action_rate=frame.flagged_action.mean(),
                           action_count=int(frame.active_action.sum()))
            for col in [c for c in frame if c == 'reward' or c.startswith('reward_')]:
                summary[col+'_mean'] = frame[col].mean()
                summary[col+'_sum'] = frame[col].sum(min_count=1)
                summary[col+'_last'] = frame[col].iloc[-1]
            episodes.append(summary)
    return pd.DataFrame(transitions), pd.DataFrame(episodes)


def paired_episode_means(episodes, columns, bootstrap_samples=2000, seed=42):
    """
    Equal-weight episodes; same bootstrap chronic/seed samples for every agent.
    """
    if episodes.duplicated(['agent', 'pair_id']).any():
        raise ValueError('Repeated chronic/seed pairs: resolve duplicates before paired analysis.')
    agents = list(episodes.agent.unique())
    common = set.intersection(*(set(episodes.loc[episodes.agent == a, 'pair_id']) for a in agents))
    rng = np.random.default_rng(seed)
    rows = []
    for column in columns:
        wide = episodes.pivot(index='pair_id', columns='agent', values=column).reindex(sorted(common)).dropna()
        if wide.empty:
            continue
        indices = rng.integers(0, len(wide), size=(bootstrap_samples, len(wide)))
        for a in agents:
            x = wide[a].to_numpy()
            means = x[indices].mean(axis=1)
            rows.append(dict(metric=column, agent=a, n_pairs=len(x), mean=x.mean(),
                             low=np.quantile(means, .025), high=np.quantile(means, .975)))
    return pd.DataFrame(rows)
