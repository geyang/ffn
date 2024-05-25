import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange


class Q_implicit(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=400, ff=None, B_scale=1, latent_scale=1):
        """Assumes discrete actions"""
        super(Q_implicit, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        if ff == 'RFF':
            self.Q = nn.Sequential(
                RFF(state_dim + action_dim, latent_dim//2, scale=B_scale),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 1),
            )
        elif ff == 'LFF':
            self.Q = nn.Sequential(
                LFF(state_dim+action_dim, latent_dim // 2, scale=B_scale),
                LFF(latent_dim, latent_dim // 2, scale=latent_scale),
                LFF(latent_dim, latent_dim // 2, scale=latent_scale),
                LFF(latent_dim, latent_dim // 2, scale=latent_scale),
                nn.Linear(latent_dim, 1),
            )
        else:
            self.Q = nn.Sequential(
                nn.Linear(state_dim + action_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 1),
            )
    def forward(self, states):
        batch_size = states.shape[0]
        value_lst =  []
        for act in range(self.action_dim):
            actions = act*torch.ones((batch_size, self.action_dim))
            values = self.Q(torch.cat([states, actions], dim=1))
            value_lst.append(values)
        return torch.cat(value_lst, dim=1)


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
    Q = Q_implicit(state_dim=1, action_dim=2)

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
    Q = Q_implicit(state_dim=1, action_dim=2)

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
        # nn.init.normal_(self.linear.weight, 0, scale / self.input_dim)
        nn.init.uniform_(self.linear.weight, -scale / self.input_dim, scale / self.input_dim)
        nn.init.uniform_(self.linear.bias, -1, 1)

    def forward(self, x):
        x = self.linear(x)
        return torch.sin(2 * np.pi * x)


class RFF(LFF):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__(input_dim, mapping_size, scale=scale)
        self.linear.requires_grad = False


def supervised_rff(states, values, lr=1e-4, n_epochs=100, B_scale=1):
    # Ge: need to initialize the Q function at zero
    Q = Q_implicit(state_dim=1, action_dim=2, ff='RFF', B_scale=B_scale)

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


def perform_deep_vi_rff(states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, B_scale=1, target_freq=1):
    # Ge: need to initialize the Q function at zero
    Q = Q_implicit(state_dim=1, action_dim=2, ff='RFF', B_scale=B_scale)

    Q_target = deepcopy(Q) if target_freq else Q

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    rewards = torch.FloatTensor(rewards)
    dyn_mats = torch.FloatTensor(dyn_mats)

    losses = []

    for epoch in trange(n_epochs + 1):
        if target_freq and epoch % target_freq == 0:
            Q_target.load_state_dict(Q.state_dict())

        q_max, actions = Q_target(states).max(dim=-1)
        td_target = rewards + gamma * dyn_mats @ q_max.detach()
        td_loss = l1(Q(states), td_target.T)
        losses.append(td_loss.detach().numpy())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    return q_values, losses


def perform_deep_vi_lff_mlp(states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, B_scale=1, target_freq=1,
                            latent_scale=5):
    # Ge: need to initialize the Q function at zero
    Q = Q_implicit(state_dim=1, action_dim=2, ff='LFF', B_scale=B_scale, latent_scale=latent_scale)
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
            std, mean = torch.std_mean(Q.Q[0].linear.weight)
            stats['stds'].append(std.item())
            stats['means'].append(mean.item())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    returns = eval_q_policy(Q)

    print(f"Avg returns is {returns}")

    return q_values, *stats.values()


def supervised_lff_mlp(states, values, lr=1e-4, n_epochs=400, B_scale=1, latent_scale=1, latent_dim=400):
    # Ge: need to initialize the Q function at zero
    Q = Q_implicit(state_dim=1, action_dim=2, latent_dim=latent_dim, ff='LFF', B_scale=B_scale, latent_scale=latent_scale)

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
            std, mean = torch.std_mean(Q.Q[0].linear.weight)
            stats['stds'].append(std.item())
            stats['means'].append(mean.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

    q_values = values_bar.T.detach().numpy()
    return q_values, *stats.values()


if __name__ == "__main__":
    doc @ """
    ## Learned Fourier Features

    We use stacked, four-layer Learned Fourier Networks (LFN) to fit to a complex value function.

    The figure table below shows that with correct scaling, the spectral bias persist across networks
    of different width across 8 octaves of latent dimension.
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

    # doc @ """
    # # Supervised Learning with Learned Random Fourier Features (LFF)
    #
    # The random matrix simply does not update that much!
    # """
    # with doc, doc.table() as table:
    #     for dim, n_epochs in zip([24, 50, 100, 200, 400, 800, 1600], [4000, 2000, 500, 250, 100, 100, 100]):
    #         r = table.figure_row()
    #         all_losses = {}
    #         for scale in [0.1, 1, 10, 20]:
    #             q_values, losses, B_stds, B_means = supervised_lff_mlp(states, gt_q_values, B_scale=8,
    #                                                                    n_epochs=n_epochs, latent_dim=dim,
    #                                                                    latent_scale=scale)
    #             plt.figure('losses')
    #             plt.plot(losses, label=f"scale={scale}")
    #
    #             plt.figure(f'scale {scale}')
    #             plt.plot(states, q_values[0], label="action 1")
    #             plt.plot(states, q_values[1], label="action 2")
    #             r.savefig(os.path.basename(__file__)[:-3] + f"/supervised_lff_mlp_dim-{dim}_sig-{scale}.png",
    #                       title=f"dim={dim} $\sigma={scale}$" if scale == 0.1 else f"$\sigma={scale}$")
    #             plt.close()
    #
    #         plt.figure('losses')
    #         plt.legend(frameon=False)
    #         r.savefig(os.path.basename(__file__)[:-3] + f"/supervised_lff_mlp_loss_dim-{dim}_sig-{scale}.png")
    #         plt.close()

    doc @ """
    ## DQN w/ LFF

    Here we plot the value function learned via deep Q Learning (DQN) using a learned random
    fourier feature network.
    """

    with doc:
        q_values, losses, B_stds, B_means = perform_deep_vi_lff_mlp(states, rewards, dyn_mats, B_scale=8, n_epochs=100)

    plot_value(states, q_values, losses, B_stds, B_means, fig_prefix="dqn_lff_mlp", title="DQN w/ LFF",
               doc=doc.table().figure_row())
    doc.flush()
