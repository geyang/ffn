# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from rff_kernels import models

import os
import numpy as np
from os.path import join as pJoin
from pathlib import Path
from warnings import filterwarnings  # noqa

from drqv2_crff.replay_buffer import Replay
from drqv2_crff import utils
from drqv2_crff.config import Args, Agent
from params_proto.neo_proto import PrefixProto

from .env_helpers import get_env
import shutil

class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class View(nn.Module):
    def __init__(self, *dims, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dims = dims

    def forward(self, x):
        if self.batch_first:
            return x.view(-1, *self.dims)
        else:
            return x.view(*self.dims)


class Join(nn.Module):
    def __init__(self, *modules):
        """Join Module

        :param modules: assumes that each module takes only 1 input.
        """
        super().__init__()
        self.modules = modules

    def forward(self, *inputs):
        return torch.cat([net(x) for x, net in zip(inputs, self.modules)], dim=1)


class YComb(nn.Module):
    def __init__(self, left, right, split):
        """Join Module

        :param modules: assumes that each module takes only 1 input.
        """
        super().__init__()
        self.left = left
        self.right = right
        self.split = split

    def forward(self, inputs):
        left_input, right_input = inputs.split(self.split, dim=1)
        return torch.cat([self.left(left_input), self.right(right_input)], dim=1)


class Identity(nn.Module):
    out_features: int

    def __init__(self, in_features: int):
        super().__init__()
        self.out_features = in_features

    def forward(self, obs, **_):
        return obs


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

class Encoder(nn.Module):
    def __init__(self, obs_shape, fourier_features=None, scale=None, rff=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.rff = rff

        if self.rff:
            self.convnet = nn.Sequential(models.CLFF(obs_shape[0], fourier_features, scale=scale),
                                         nn.Conv2d(fourier_features, 32, 3, stride=2),
                                         nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                         nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                         nn.ReLU())
        else:
            self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                        nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 stddev_schedule, stddev_clip,
                 rff=False, conv_fourier_features=None, scale=None,):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape, fourier_features=conv_fourier_features, scale=scale, rff=rff).to(device)

        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    # todo: remove step from this method.
    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            # todo: remove this logic, use random flag instead. This is bad code.
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        from ml_logger import logger
        logger.store_metrics(
            critic_target_q=target_Q.mean().item(),
            critic_q1=Q1.mean().item(),
            critic_q2=Q2.mean().item(),
            critic_loss=critic_loss.item(),
        )

    def update_actor(self, obs, step):

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        from ml_logger import logger
        logger.store_metrics(
            actor_loss=actor_loss.item(),
            actor_logprob=log_prob.mean().item(),
            actor_ent=dist.entropy().sum(dim=-1).mean().item()
        )

    def update(self, replay_iter, step):

        batch = next(replay_iter)
        # obs: [256, 9, 84, 84], action: [256, 1], reward: [256, 1], discount: [256, 1]
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        from ml_logger import logger
        logger.store_metrics(batch_reward=reward.mean().item())

        # update critic
        self.update_critic(obs, action, reward, discount, next_obs, step)
        # update actor
        self.update_actor(obs.detach(), step)

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

class Progress(PrefixProto, cli=False):
    step = 0
    episode = 0
    wall_time = 0
    frame = 0


def save_checkpoint(agent, replay_cache_dir, checkpoint_root):
    from ml_logger import logger

    replay_checkpoint = os.path.join(checkpoint_root, logger.prefix, 'replay.tar')
    logger.save_torch(agent, checkpoint_root, logger.prefix, 'agent.pkl')
    logger.duplicate("metrics.pkl", "metrics_latest.pkl")
    logger.upload_dir(str(replay_cache_dir), replay_checkpoint)
    # Save the progress.pkl last as a fail-safe. To make sure the checkpoints are saving correctly.
    logger.log_params(Progress=vars(Progress), path="progress.pkl", silent=True)
    logger.job_running()


def load_checkpoint(replay_cache_dir, checkpoint_root):
    from ml_logger import logger

    replay_checkpoint = os.path.join(checkpoint_root, logger.prefix, 'replay.tar')

    agent = logger.load_torch(checkpoint_root, logger.prefix, 'agent.pkl', map_location=Args.device)
    logger.duplicate("metrics_latest.pkl", to="metrics.pkl")
    logger.download_dir(replay_checkpoint, str(replay_cache_dir))

    return agent, logger.read_params(path="progress.pkl")


def eval(env, agent, global_step, to_video=None):
    from ml_logger import logger

    step, total_reward = 0, 0
    for episode in range(Args.num_eval_episodes):
        obs = env.reset()
        frames = []
        done = False
        while not done:
            with torch.no_grad(), utils.eval_mode(agent):
                # todo: remove global_step, replace with random-on, passed-in.
                action = agent.act(obs, global_step, eval_mode=True)
            obs, reward, done, info = env.step(action)
            if episode == 0 and to_video:
                # todo: use gym.env.render('rgb_array') instead
                frames.append(env.physics.render(height=256, width=256, camera_id=0))
            total_reward += reward
            step += 1

        if episode == 0 and to_video:
            logger.save_video(frames, to_video)

    logger.log(episode_reward=total_reward / episode, episode_length=step * Args.action_repeat / episode)


def run(train_env, eval_env, agent, replay, progress: Progress, checkpoint_root, time_limit=None):
    from ml_logger import logger

    init_transition = dict(
        obs=None, reward=0.0, done=False, discount=1.0,
        action=np.full(eval_env.action_space.shape, 0.0, dtype=eval_env.action_space.dtype)
    )

    episode_step, episode_reward = 0, 0
    obs = train_env.reset()
    transition = {**init_transition, 'obs': obs}
    replay.storage.add(**transition)
    done = False
    for progress.step in range(progress.step, Args.train_frames // Args.action_repeat + 1):
        progress.wall_time = logger.since('start')
        progress.frame = progress.step * Args.action_repeat

        if done:
            progress.episode += 1

            # log stats
            episode_frame = episode_step * Args.action_repeat
            logger.log(fps=episode_frame / logger.split('episode'),
                       episode_reward=episode_reward,
                       episode_length=episode_frame,
                       buffer_size=len(replay.storage))

            # reset env
            obs = train_env.reset()
            done = False
            transition = {**init_transition, 'obs': obs}
            replay.storage.add(**transition)
            # try to save snapshot
            if time_limit and logger.since('run') > time_limit:
                logger.print(f'local time_limit: {time_limit} (sec) has reached!')
                raise TimeoutError

            episode_step, episode_reward = 0, 0

        # try to evaluate
        if logger.every(Args.eval_every_frames // Args.action_repeat, key="eval"):
            with logger.Prefix(metrics="eval"):
                path = f'videos/{progress.step * Args.action_repeat:09d}_eval.mp4'
                eval(eval_env, agent, progress.step, to_video=path if Args.save_video else None)
                logger.log(**vars(progress))
                logger.flush()

        # sample action
        done_warming_up = progress.step * Args.action_repeat > Args.num_seed_frames

        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, progress.step, eval_mode=False)

        # try to update the agent
        if logger.every(Args.update_freq, key="update") and done_warming_up:
            agent.update(replay.iterator, progress.step)  # checkpoint.step is used for scheduling

        if logger.every(Args.log_freq, key="log", start_on=1) and done_warming_up:
            logger.log_metrics_summary(vars(progress), default_stats='mean')

        if logger.every(Args.checkpoint_freq, key='checkpoint', start_on=1) and done_warming_up:
            with logger.time('checkpoint'):
                save_checkpoint(agent, replay.cache_dir, checkpoint_root=checkpoint_root)

        # take env step
        obs, reward, done, info = train_env.step(action)
        episode_reward += reward

        # TODO: is it ok to always use discount = 1.0 ?
        # we should actually take a look at time_step.discount
        transition = dict(obs=obs, reward=reward, done=done, discount=1.0, action=action)
        replay.storage.add(**transition)
        episode_step += 1

def train(**kwargs):
    from ml_logger import logger, RUN
    # get the directory where this file is located

    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)

    logger.start('run')

    assert logger.prefix, "you will overwrite the experiment root"

    try:  # completion protection
        assert logger.read_params('job.completionTime')
        logger.print(f'job.completionTime is set. This job seems to have been completed already.')
        return
    except KeyError:
        pass

    replay_cache_dir = Path(Args.tmp_dir) / logger.prefix / 'replay'
    shutil.rmtree(replay_cache_dir, ignore_errors=True)
    replay_checkpoint = os.path.join(Args.checkpoint_root, logger.prefix, 'replay.tar')
    snapshot_dir = pJoin(Args.checkpoint_root, logger.prefix)

    if logger.glob('progress.pkl'):
        Args._update(**logger.read_params("Args"))
        Agent._update(**logger.read_params('Agent'))

        agent, progress_cache = load_checkpoint(replay_cache_dir, Args.checkpoint_root)

        Progress._update(progress_cache)
        logger.start('start', 'episode')
        logger.timer_cache['start'] = logger.timer_cache['start'] - Progress.wall_time

    else:
        logger.print("Start training from scratch.")
        # load parameters
        Args._update(kwargs)
        Agent._update(kwargs)
        logger.log_params(Args=vars(Args), Agent=vars(Agent))

        # todo: this needs to be fixed.
        logger.log_text("""
            keys:
            - Args.env_name
            - Args.seed
            charts:
            - yKey: eval/episode_reward
              xKey: eval/frame
            - yKey: episode_reward
              xKey: frame
            - yKeys: ["batch_reward/mean", "critic_loss/mean"]
              xKey: frame
            """, ".charts.yml", overwrite=True, dedent=True)

        logger.start('start', 'episode')

    utils.set_seed_everywhere(Args.seed)
    train_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed)
    eval_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed)

    if 'agent' not in locals():
        agent = DrQV2Agent(obs_shape=train_env.observation_space.shape,
                           action_shape=train_env.action_space.shape,
                           device=Args.device, **vars(Agent))

    replay = Replay(cache_dir=replay_cache_dir, args=Args)

    # Load from local file
    assert logger.prefix, "you will overwrite the experiment root with an empty logger.prefix"
    run(train_env, eval_env, agent, replay, progress=Progress, checkpoint_root=Args.checkpoint_root)