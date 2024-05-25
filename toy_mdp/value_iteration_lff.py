import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange


def plot_value(states, q_values, losses, *std_mean, fig_prefix, title=None, doc=doc):
    plt.plot(states, q_values[0], label="action 1")
    plt.plot(states, q_values[1], label="action 2")
    if title:
        plt.title(title)
    plt.legend()
    plt.xlabel('State [0, 1)')
    plt.ylabel('Value')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()

    for loss, plot_title in zip([losses, *std_mean], ['loss', 'stddev', 'mean']):
        plt.plot(loss)
        if plot_title == "loss":
            plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
        plt.title(plot_title.capitalize())
        plt.xlabel('Optimization Steps')
        doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}_{plot_title}.png?ts={doc.now("%f")}', dpi=300,
                    zoom=0.3)
        plt.close()


def perform_vi(states, rewards, dyn_mats, gamma=0.9, eps=1e-5):
    # Assume discrete actions and states
    q_values = np.zeros(dyn_mats.shape[:2])

    deltas = []
    while not deltas or deltas[-1] >= eps:
        old = q_values
        q_max = q_values.max(axis=0)
        q_values = rewards + gamma * dyn_mats @ q_max

        deltas.append(np.abs(old - q_values).max())

    return q_values, deltas


def supervised(states, values, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=100):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        nn.Linear(1, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    values = torch.FloatTensor(values)

    losses = []

    for epoch in trange(n_epochs + 1):
        values_bar = Q(states)
        loss = l1(values_bar, values.T)
        losses.append(loss.detach().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()

    q_values = values_bar.T.detach().numpy()
    return q_values, losses


def perform_deep_vi(states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        nn.Linear(1, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )
    Q_target = deepcopy(Q)

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    rewards = torch.FloatTensor(rewards)
    dyn_mats = torch.FloatTensor(dyn_mats)

    losses = []

    for epoch in trange(n_epochs + 1):
        if epoch % 1 == 0:
            Q_target.load_state_dict(Q.state_dict())

        q_max, actions = Q_target(states).max(dim=-1)
        td_target = rewards + gamma * dyn_mats @ q_max
        td_loss = l1(Q(states), td_target.T)
        losses.append(td_loss.detach().numpy())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    return q_values, losses


# class LFF(nn.Module):
#     """
#     get torch.std_mean(self.B)
#     """
#
#     def __init__(self, input_dim, mapping_size, scale=1, grad_multiplier=1):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = mapping_size * 2
#         self.B = nn.Parameter(torch.Tensor(size=(mapping_size, self.input_dim)))
#         nn.init.normal_(self.B, 0, scale)
#         self.B.register_hook(lambda grad: grad_multiplier * grad)
#
#     def forward(self, x):
#         return torch.cat([torch.cos(2 * np.pi * x @ self.B.T),
#                           torch.sin(2 * np.pi * x @ self.B.T)], dim=1)

class LFF(nn.Module):
    """
    get torch.std_mean(self.B)
    """

    def __init__(self, input_dim, mapping_size, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = mapping_size * 2
        self.linear = nn.Linear(input_dim, self.output_dim)
        nn.init.normal_(self.linear.weight, 0, scale / self.input_dim)

    def forward(self, x):
        x = self.linear(2 * np.pi * x)
        return torch.sin(x)


class RFF(LFF):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__(input_dim, mapping_size, scale=scale)
        self.linear.requires_grad = False


def perform_deep_vi_lff(Q, states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, target_freq=1, ):
    Q_target = deepcopy(Q) if target_freq else Q

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    rewards = torch.FloatTensor(rewards)
    dyn_mats = torch.FloatTensor(dyn_mats)

    stats = defaultdict(list)

    for epoch in trange(n_epochs + 1):
        if target_freq and epoch % target_freq == 0:
            Q_target.load_state_dict(Q.state_dict())

        q_max, actions = Q_target(states).max(dim=-1)
        td_target = rewards + gamma * dyn_mats @ q_max.detach()
        td_loss = l1(Q(states), td_target.T)
        stats['losses'].append(td_loss.detach().item())
        with torch.no_grad():
            std, mean = torch.std_mean(Q[0].linear.weight)
            stats['stds'].append(std.item())
            stats['means'].append(mean.item())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    return q_values, *stats.values()


def supervised_lff(states, values, lr=1e-4, n_epochs=400, B_scale=1):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        LFF(1, 50, scale=B_scale),
        nn.Linear(100, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    values = torch.FloatTensor(values)

    stats = defaultdict(list)
    for epoch in trange(n_epochs + 1):
        values_bar = Q(states)
        loss = l1(values_bar, values.T)
        stats['losses'].append(loss.detach().item())

        with torch.no_grad():
            std, mean = torch.std_mean(Q[0].linear.weight)
            stats['stds'].append(std.item())
            stats['means'].append(mean.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

    q_values = values_bar.T.detach().numpy()
    return q_values, *stats.values()


def eval_q_policy(q, num_eval=100):
    """Assumes discrete action such that policy is derived by argmax a Q(s,a)"""
    from rand_mdp import RandMDP
    torch.manual_seed(0)
    env = RandMDP(seed=0, option='fixed')
    returns = []

    for i in range(num_eval):
        done = False
        obs = env.reset()
        total_rew = 0
        while not done:
            obs = torch.FloatTensor(obs).unsqueeze(-1)
            q_max, action = q(obs).max(dim=-1)
            obs, rew, done, _ = env.step(action.item())
            total_rew += rew
        returns.append(total_rew)

    return np.mean(returns)


if __name__ == "__main__":
    doc @ """
    ## Tabular Q-learning (Ground-truth)
    
    Here is the ground truth value function generated via tabular
    value iteration. It shows even for simple dynamics, the value
    function can be exponentially more complex.
    """
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    with doc:
        num_states = 200
        torch.manual_seed(0)
        mdp = RandMDP(seed=0, option='fixed')
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, losses = perform_vi(states, rewards, dyn_mats)

    gt_q_values = q_values  # used later

    plot_value(states, q_values, losses, fig_prefix="value_iteration",
               title="Value Iteration on Toy MDP", doc=doc.table().figure_row())

    doc @ """
    ## DQN w/ LFF
    
    Here we plot the value function learned via deep Q Learning (DQN) using a learned random
    fourier feature network.
    """

    with doc:
        def get_Q_lff(B_scale):
            return nn.Sequential(
                LFF(1, 50, scale=B_scale),
                nn.Linear(100, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 2),
            )

        Q = get_Q_lff(B_scale=10)
        q_values, losses, B_stds, B_means = perform_deep_vi_lff(Q, states, rewards, dyn_mats)
        returns = eval_q_policy(Q)

        doc.print(f"Avg return for DQN+LFF (sigma 5) is {returns}")
    plot_value(states, q_values, losses, B_stds, B_means, fig_prefix="dqn_lff", title="DQN w/ LFF",
               doc=doc.table().figure_row())

    doc.flush()
