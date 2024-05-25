import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange


def plot_value(states, q_values, losses, fig_prefix, title=None, doc=doc):
    plt.plot(states, q_values[0], color="#23aaff", linewidth=4, label="action 1")
    plt.plot(states, q_values[1], color="#444", linewidth=4, label="action 2")
    if title:
        plt.title(title)
    plt.legend()
    plt.xlabel('State [0, 1)')
    plt.ylabel('Value')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()

    plt.plot(losses)
    plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
    plt.title("Loss")
    plt.xlabel('Optimization Steps')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}_loss.png?ts={doc.now("%f")}', dpi=300,
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


def supervised(states, values, dyn_mats, lr=4e-4, gamma=0.9, n_epochs=100):
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


def perform_deep_vi(states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, target_freq=10):
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
        if epoch % target_freq == 0:
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


class RFF(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.output_dim = mapping_size * 2
        self.B = torch.normal(0, scale, size=(mapping_size * 2, self.input_dim))
        self.B.requires_grad = False

    def forward(self, x):
        return torch.cat([torch.cos(2 * np.pi * x @ self.B[:self.mapping_size].T),
                          torch.sin(2 * np.pi * x @ self.B[self.mapping_size:].T)], dim=1)


def supervised_rff(states, values, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=100, B_scale=1):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        RFF(1, 200, scale=B_scale),
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


def perform_deep_vi_rff(states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, B_scale=1, target_freq=10):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        RFF(1, 200, scale=B_scale),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )
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
        # q_max, _ = Q(states)[:, actions]
        td_target = rewards + gamma * dyn_mats @ q_max.detach()
        td_loss = l1(Q(states), td_target.T)
        losses.append(td_loss.detach().numpy())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).T.detach().numpy()
    return q_values, losses


if __name__ == "__main__":
    doc @ """
    ## Learning without a target network
    
    We need to first evaluate the bias using a target network. 
    """
    from off_policy_divergence.rand_mdp import RandMDP
    from matplotlib import pyplot as plt, patches

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
    ## Compare between the three Architectures
    """

    with doc:
        mlp_q_values, losses = perform_deep_vi(states, rewards, dyn_mats, n_epochs=500)
        rff_q_values, losses = perform_deep_vi_rff(
            states, rewards, dyn_mats, n_epochs=500, B_scale=5)
        rff_no_tgt_q_values, losses = perform_deep_vi_rff(
            states, rewards, dyn_mats, n_epochs=500, B_scale=5, target_freq=False)

    with doc:
        plt.figure(figsize=(6.4, 4.8))
        plt.plot(states, gt_q_values[0], color="black", linewidth=1, label="Ground Truth", zorder=5)
        plt.plot(states, rff_no_tgt_q_values[0], color="#23aaff", linewidth=4, label="FFN (No Target)", alpha=0.8)
        plt.plot(states, rff_q_values[0], color="orange", linewidth=3, label="FFN", alpha=0.9)
        plt.plot(states, mlp_q_values[0], color="red", linewidth=3, label="MLP", alpha=0.3)
        plt.title("Neural Fitted Q Iteration")
        plt.xlabel("State [0, 1)")
        plt.ylabel("Value")

        rect = patches.Rectangle([-0.02, 5], 0.205, 2, fill=True, facecolor="white", linewidth=0, zorder=5)
        plt.gca().add_patch(rect)
        plt.legend(loc=(0.025, 0.5), framealpha=0, borderaxespad=-10)
        plt.ylim(3, 7.5)
        plt.xlim(-0.05, 1.05)
        plt.tight_layout()
        doc.savefig(f'{Path(__file__).stem}/q_value_comparison.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.savefig(f'{Path(__file__).stem}/q_value_comparison.pdf', dpi=300)

    doc.flush()
