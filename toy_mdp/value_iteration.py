import os
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange


def plot_value(states, q_values, losses, fig_prefix, title=None, doc=doc):
    plt.plot(states, q_values[0], label="action 1")
    plt.plot(states, q_values[1], label="action 2")
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
        self.output_dim = mapping_size * 2
        self.B = torch.normal(0, scale, size=(mapping_size, self.input_dim))
        self.B.requires_grad = False

    def forward(self, x):
        return torch.cat([torch.cos(2 * np.pi * x @ self.B.T),
                          torch.sin(2 * np.pi * x @ self.B.T)], dim=1)


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
    ## Tabular Q-learning (Ground-truth)
    
    Here is the ground truth value function generated via tabular
    value iteration. It shows even for simple dynamics, the value
    function can be exponentially more complex.
    """
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    with doc:
        num_states = 20
        torch.manual_seed(0)
        mdp = RandMDP(seed=0, option='fixed')
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, losses = perform_vi(states, rewards, dyn_mats)

    # os.makedirs('data', exist_ok=True)
    # np.savetxt("data/q_values.csv", q_values, delimiter=',')

    gt_q_values = q_values  # used later

    plot_value(states, q_values, losses, fig_prefix="value_iteration",
               title="Value Iteration on Toy MDP", doc=doc.table().figure_row())

    doc @ """
    ## DQN w/ Function Approximator
    
    Here we plot the value function learned via deep Q Learning 
    (DQN) using a neural network function approximator.
    """

    with doc:
        q_values, losses = perform_deep_vi(states, rewards, dyn_mats)

    plot_value(states, q_values, losses, fig_prefix="dqn",
               title="DQN on Toy MDP", doc=doc.table().figure_row())

    doc @ """
    ## A Supervised Baseline
    
    **But can the function learn these value functions?** As it turned out, no.
    Even with a supervised learning objective, the learned value function is
    not able to produce a good approximation of the value landscape. Not
    with 20 states, and even less so with 200.
    """
    with doc:
        q_values, losses = supervised(states, gt_q_values, dyn_mats)

    plot_value(states, q_values, losses, fig_prefix="supervised",
               title="Supervised Value Function", doc=doc.table().figure_row())

    doc @ """
    ## Now use RFF (supervised)
    
    The same supervised experiment, instantly improve in fit if we 
    replace the input layer with RFF embedding.
    """
    with doc:
        q_values, losses = supervised_rff(states, gt_q_values, dyn_mats, B_scale=10)

    plot_value(states, q_values, losses, fig_prefix="supervised_over_param",
               title=f"RFF Supervised {10}", doc=doc.table().figure_row())
    doc @ """
    ## DQN with RFF 
    
    We can now apply this to DQN and it works right away!
    """
    with doc:
        q_values, losses = perform_deep_vi_rff(states, rewards, dyn_mats, n_epochs=500, B_scale=10)

    plot_value(states, q_values, losses, fig_prefix="dqn_over_param",
               title=f"DQN w/ RFF {10}", doc=doc.table().figure_row())

    doc @ """
    ## DQN with RFF without Target
    
    Try removing the target network
    """
    with doc:
        q_values, losses = perform_deep_vi_rff(states, rewards, dyn_mats, n_epochs=500, B_scale=10,
                                               target_freq=None)

    plot_value(states, q_values, losses, fig_prefix="dqn_over_param_no_target",
               title=f"DQN w/ RFF {10}", doc=doc.table().figure_row())
    doc.flush()
