import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, use_rff=False, critic_1_rff=None, critic_2_rff=None):
        super().__init__()
        self.use_rff = use_rff

        if self.use_rff:
            self.Q1 = utils.mlp(critic_1_rff.out_features, hidden_dim, 1, hidden_depth)
            self.Q2 = utils.mlp(critic_2_rff.out_features, hidden_dim, 1, hidden_depth)
        else:
            self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
            self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

        self.critic_1_rff = critic_1_rff
        self.critic_2_rff = critic_2_rff

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=-1)

        if self.use_rff:
            q1 = self.Q1(self.critic_1_rff(obs_action))
            q2 = self.Q2(self.critic_2_rff(obs_action))
        else:
            q1 = self.Q1(obs_action)
            q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self):
        from ml_logger import logger
        for k, v in self.outputs.items():
            logger.store_metrics({f"train/{k}_mean": v.mean().item()})
            logger.store_metrics({f"train/{k}_min": v.min().item()})
            logger.store_metrics({f"train/{k}_max": v.max().item()})