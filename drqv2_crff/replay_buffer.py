# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import os
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    from ml_logger import logger
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, dir):
        from ml_logger import logger
        logger.print(f"making dir: {dir}", color='green')
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, obs, reward, done, action, discount):
        # NOTE: reward = 0, action = [0, ..., 0], discount = 1.0 at the very first transition
        name2val = dict(observation=obs, reward=reward, action=action, discount=discount)
        for name, val in name2val.items():
            # NOTE:
            # NG: reward.shape == (256,)
            # OK: reward.shape == (256, 1)
            if np.isscalar(val):
                dtype = np.float32 if name in ['reward', 'discount'] else self._get_dtype(val)
                val = np.full((1,), val, dtype)
            # assert spec.shape == value.shape and spec.dtype == value.dtype
            self.current_episode[name].append(val)

        from ml_logger import logger
        if done:
            episode = dict()
            for name, val in name2val.items():
                cur_episode = self.current_episode[name]
                dtype = np.float32 if name in ['reward', 'discount'] else self._get_dtype(val)
                episode[name] = np.array(cur_episode, dtype)
            self.current_episode = defaultdict(list)
            self._store_episode(episode)

    def _get_dtype(self, val):
        if isinstance(val, np.ndarray):
            return val.dtype
        elif isinstance(val, float):
            return np.float32
        elif isinstance(val, int):
            return np.int32

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self.dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self.dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        """Load from eps_fn and add its episodes to replay buffer in FIFO fashion."""
        try:
            episode = load_episode(eps_fn)
        except FileNotFoundError:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)  # Remove the oldest file
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len

            # This method returns False only when loading file fails
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            from ml_logger import logger
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader


class Replay:
    _iter = None

    def __init__(self, cache_dir, args):
        self.cache_dir = cache_dir
        print('making storage')
        self.storage = ReplayBufferStorage(cache_dir)
        print('making loader')
        self.loader = make_replay_loader(cache_dir, args.replay_buffer_size,
                                         args.batch_size, args.replay_buffer_num_workers,
                                         save_snapshot=True, nstep=args.nstep, discount=args.discount)

    @property
    def iterator(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        return self._iter
