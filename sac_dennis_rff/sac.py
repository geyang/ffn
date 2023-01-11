import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gym
from copy import deepcopy
from tqdm import trange

from . import utils
from .actor import DiagGaussianActor
from .critic import DoubleQCritic
from .replay_buffer import ReplayBuffer
from params_proto import ParamsProto, PrefixProto, Proto
from .config import Args, Actor, Critic, Agent
from rff_kernels import models

class Progress(PrefixProto, cli=False):
    step = 0
    episode = 0

class SACAgent(nn.Module):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic,
                 actor, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = critic

        if self.critic_tau is None:
            self.critic_target = self.critic
        else:
            self.critic_target = deepcopy(critic)
            self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = actor

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done):
        from ml_logger import logger

        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        logger.store_metrics({'train/critic_loss': critic_loss.item()})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log()

    def update_actor_and_alpha(self, obs):
        from ml_logger import logger

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.store_metrics({'train/actor_loss': actor_loss.item()})
        logger.store_metrics({'train/actor_target_entropy': self.target_entropy})
        logger.store_metrics({'train/actor_entropy': -log_prob.mean().item()})

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.store_metrics({'train/alpha_loss': alpha_loss.item()})
            logger.store_metrics({'train/alpha_value': self.alpha.item()})
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step):
        from ml_logger import logger

        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        logger.store_metrics({'train/batch_reward': reward.mean().item()})

        self.update_critic(obs, action, reward, next_obs, not_done_no_max)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            if self.critic_tau is not None:
                utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)

def make_env(env_name, seed, from_pixels=True, dmc=True, image_size=None, frame_stack=None,
             normalize_obs=False, obs_bias=None, obs_scale=None):
    """Helper function to create dm_control environment"""
    from ml_logger import logger

    domain_name, task_name, *_ = env_name.split(":")[-1].split('-')
    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'Quadruped' else 0

    if dmc:
        env = gym.make(env_name,
                       visualize_reward=False,
                       from_pixels=from_pixels,
                       height=image_size,
                       width=image_size,
                       frame_skip=1,
                       camera_id=camera_id)
    else:
        env = gym.make(env_name)

    if normalize_obs:
        logger.print(f'obs bias is {obs_bias}', color="green")
        logger.print(f'obs scale is {obs_scale}', color="green")

    env = utils.NormalizedBoxEnv(env, obs_mean=obs_bias, obs_std=obs_scale)

    if from_pixels and frame_stack:
        env = utils.FrameStack(env, k=frame_stack)

    env.seed(seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def train(**deps):
    from ml_logger import logger, RUN

    RUN._update(deps)
    if RUN.resume and logger.glob("checkpoint.pkl"):
        deps = logger.read_params()
    else:
        RUN.resume = False

    Args._update(deps)
    Actor._update(deps)
    Critic._update(deps)
    Agent._update(deps)

    if RUN.resume:
        logger.print("Loading from checkpoint...", color="yellow")
        logger.duplicate("metrics_latest.pkl", to="metrics.pkl")
        Progress._update(logger.read_params(path="checkpoint.pkl"))
        # note: maybe remove the error later after the run stablizes
        logger.remove("traceback.err")
        if Progress.episode > 0:  # the episode never got completed
            Progress.episode -= 1
    else:
        logger.remove('metrics.pkl', 'checkpoint.pkl', 'metrics_latest.pkl', "traceback.err")
        logger.log_params(RUN=vars(RUN), Args=vars(Args), Actor=vars(Actor), Agent=vars(Agent), Critic=vars(Critic))
        logger.log_text("""
            charts:
            - yKey: train/episode_reward/mean
              xKey: step
            - yKey: eval/episode_reward/mean
              xKey: step
            """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = True
    utils.set_seed_everywhere(Args.seed)

    env = make_env(Args.env_name, seed=Args.seed,
                   from_pixels=Args.from_pixels,
                   dmc=Args.dmc,
                   image_size=Args.image_size,
                   frame_stack=Args.frame_stack,
                   normalize_obs=Args.normalize_obs,
                   obs_bias=Args.obs_bias,
                   obs_scale=Args.obs_scale)

    eval_env = make_env(Args.env_name, seed=Args.seed,
                        from_pixels=Args.from_pixels,
                        dmc=Args.dmc,
                        image_size=Args.image_size,
                        frame_stack=Args.frame_stack,
                        normalize_obs=Args.normalize_obs,
                        obs_bias=Args.obs_bias,
                        obs_scale=Args.obs_scale)

    if RUN.resume:
        agent = logger.load_torch(Args.checkpoint_root, logger.prefix, 'checkpoint/agent.pkl', map_location=Args.device)
        replay_buffer = logger.load_torch(Args.checkpoint_root, logger.prefix, 'checkpoint/replay_buffer.pkl')
    else:
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]

        if Agent.use_rff:
            actor_rff = models.LFF(in_features=obs_shape[0],
                                   out_features=Agent.actor_fourier_features,
                                   scale=Agent.scale)
            critic_1_rff = models.LFF(in_features=obs_shape[0]+action_shape[0],
                                   out_features=Agent.critic_fourier_features,
                                   scale=Agent.scale)
            critic_2_rff = models.LFF(in_features=obs_shape[0]+action_shape[0],
                                      out_features=Agent.critic_fourier_features,
                                      scale=Agent.scale)
        else:
            actor_rff = None
            critic_1_rff = None
            critic_2_rff = None

        actor = DiagGaussianActor(
            obs_dim = obs_shape[0],
            action_dim = action_shape[0],
            hidden_dim = Actor.hidden_features,
            hidden_depth = Actor.hidden_layers,
            log_std_bounds = Actor.log_std_bounds,
            use_rff = Agent.use_rff,
            actor_rff = actor_rff,
        )

        critic = DoubleQCritic(
            obs_dim=obs_shape[0],
            action_dim=action_shape[0],
            hidden_dim=Critic.hidden_features,
            hidden_depth=Critic.hidden_layers,
            use_rff = Agent.use_rff,
            critic_1_rff = critic_1_rff,
            critic_2_rff = critic_2_rff,
        )

        agent = SACAgent(
            obs_dim=obs_shape[0],
            action_dim=action_shape[0],
            action_range=action_range,
            device=Args.device,
            critic=critic,
            actor=actor,
            discount=Agent.discount,
            init_temperature=Agent.init_temperature,
            alpha_lr=Agent.alpha_lr,
            alpha_betas=Agent.alpha_betas,
            actor_lr=Agent.actor_lr,
            actor_betas=Agent.actor_betas,
            actor_update_frequency=Agent.actor_update_frequency,
            critic_lr=Agent.critic_lr,
            critic_betas=Agent.critic_betas,
            critic_tau=Agent.critic_tau,
            critic_target_update_frequency=Agent.critic_target_update_frequency,
            batch_size=Agent.batch_size,
            learnable_temperature=Agent.learnable_temperature,
        )

        agent.to(Args.device)

        replay_buffer = ReplayBuffer(obs_shape, action_shape, capacity=Args.replay_buffer_size, device=Args.device)

    logger.print('now start running', color="green")

    run(env, eval_env, agent, replay_buffer,
        progress=Progress,
        train_steps=Args.train_frames,
        seed_steps=Args.seed_frames, **vars(Args))


def evaluate(env, agent, step, n_episode, save_video=None, compute_rank=False):
    from ml_logger import logger

    average_episode_reward = 0
    pred_q_lst = []
    pred_q1_lst = []
    pred_q2_lst = []
    true_q_lst = []
    frames = []
    if compute_rank:
        feats_1 = []
        feats_2 = []
    for episode in trange(n_episode):
        obs = env.reset()
        done = False
        episode_reward = 0
        averaged_true_q = 0
        averaged_pred_q = 0
        averaged_pred_q1 = 0
        averaged_pred_q2 = 0
        episode_step = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            obs_tensor = torch.as_tensor(obs[None], device=agent.device).float()
            action_tensor = torch.as_tensor(action[None], device=agent.device).float()
            # Calculating sum of predicted Q values along the trajectory
            if compute_rank:
                raise NotImplementedError
            else:
                q1, q2 = agent.critic(obs_tensor, action_tensor)
            averaged_pred_q += torch.min(q1, q2).item()
            averaged_pred_q1 = q1.item()
            averaged_pred_q2 = q2.item()
            obs, reward, done, info = env.step(action)
            if Args.from_pixels:
                img = obs.transpose([1, 2, 0])[:, :, :3]
            else:
                img = env.render("rgb_array", width=Args.image_size, height=Args.image_size)
            if save_video:
                frames.append(img)
            episode_reward += reward
            episode_step += 1
            # Calculating sum of Q values along the trajectory
            averaged_true_q += reward * (1 - (agent.discount ** episode_step)) / (1 - agent.discount)
        average_episode_reward += episode_reward
        # Dividing by episode step to calculate average of Q values along trajectory
        averaged_true_q = averaged_true_q / episode_step
        # Dividing by episode step to calculate average of predicted Q values along trajectory
        averaged_pred_q = averaged_pred_q / episode_step
        averaged_pred_q1 = averaged_pred_q1 / episode_step
        averaged_pred_q2 = averaged_pred_q2 / episode_step

        true_q_lst.append(averaged_true_q)
        pred_q_lst.append(averaged_pred_q)
        pred_q1_lst.append(averaged_pred_q1)
        pred_q2_lst.append(averaged_pred_q2)
    if save_video:
        logger.save_video(frames, save_video)
    average_episode_reward /= n_episode

    logger.store_metrics(metrics={'eval/episode_reward': average_episode_reward,
                                  'eval/avg_pred_q': np.mean(pred_q_lst),
                                  'eval/avg_pred_q1': np.mean(pred_q1_lst),
                                  'eval/avg_pred_q2': np.mean(pred_q2_lst),
                                  'eval/avg_true_q': np.mean(true_q_lst)})

    if compute_rank:
        raise NotImplementedError


def run(env, eval_env, agent, replay_buffer, progress, seed_steps, train_steps,
        eval_frequency, eval_episodes, log_frequency_step, checkpoint_freq, checkpoint_root, save_video,
        save_final_replay_buffer, seed, report_rank, **_):
    from ml_logger import logger

    episode_reward, episode_step, done = 0, 1, True
    logger.start('episode')
    start_step = progress.step
    for progress.step in range(start_step, train_steps + 1):

        # evaluate agent periodically
        if progress.step % eval_frequency == 0:
            evaluate(eval_env, agent, progress.step, n_episode=eval_episodes,
                     save_video=f'videos/{progress.step:07d}.mp4' if save_video else None,
                     compute_rank=report_rank)

        if progress.step % checkpoint_freq == 0:
            logger.job_running()  # mark the job to be running.
            logger.print(f"saving checkpoint: {checkpoint_root}/{logger.prefix}", color="green")
            with logger.time('checkpoint.agent'):
                logger.save_torch(agent, checkpoint_root, logger.prefix, 'checkpoint/agent.pkl')
            with logger.time('checkpoint.buffer'):
                logger.save_torch(replay_buffer, checkpoint_root, logger.prefix, 'checkpoint/replay_buffer.pkl')
            logger.duplicate("metrics.pkl", "metrics_latest.pkl")
            logger.log_params(Progress=vars(progress), path="checkpoint.pkl", silent=True)

        if done:
            logger.store_metrics({'train/episode_reward': episode_reward})
            dt_episode = logger.split('episode')

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            progress.episode += 1

        # sample action for data collection
        if progress.step < seed_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)

        # run training update
        if progress.step >= seed_steps:
            agent.update(replay_buffer, progress.step)

            if progress.step % log_frequency_step == 0:
                logger.log_metrics_summary({"episode": progress.episode,
                                            "step": progress.step,
                                            "frames": progress.step,
                                            "dt_episode": dt_episode}, default_stats='mean')

        next_obs, reward, done, info = env.step(action)

        # allow infinite bootstrap
        done = float(done)
        done_no_max = 0 if episode_step + 1 == env._max_episode_steps else done
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

        obs = next_obs
        episode_step += 1

    if save_final_replay_buffer:
        logger.print("saving replay buffer", color="green")
        logger.save_torch(replay_buffer, checkpoint_root, logger.prefix, 'checkpoint/replay_buffer.pkl')