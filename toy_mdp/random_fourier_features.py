import numpy as np
import torch.optim
from cmx import doc
from torch import nn
from tqdm import trange


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


def plot_value(states, q_values, losses, fig_prefix, doc=doc):
    plt.plot(states, q_values[0], label="action 1")
    plt.plot(states, q_values[1], label="action 2")
    plt.title("Toy MDP")
    plt.legend()
    plt.xlabel('State [0, 1)')
    plt.ylabel('Value')
    doc.savefig(f'figures/{fig_prefix}.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()

    plt.plot(losses)
    plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
    plt.title("Loss")
    plt.xlabel('Optimization Steps')
    doc.savefig(f'figures/{fig_prefix}_loss.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()


if __name__ == "__main__":
    doc @ """
    ## Tabular Q-learning (Ground-truth)
    
    Here is the ground truth value function generated via tabular
    value iteration. It shows that even for simple dynamics, the
    value function can be exponentially complex due to recursion.
    """
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    with doc:
        num_states = 200
        mdp = RandMDP(seed=0, option='fixed')
        # states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        gt_q_values = np.loadtxt("data/q_values.csv", delimiter=',')
        states = np.loadtxt("data/states.csv", delimiter=',')
        rewards = np.loadtxt('data/rewards.csv', delimiter=',')

    with doc.table().figure_row() as r:
        plt.plot(states, gt_q_values[0], label="action 1")
        plt.plot(states, gt_q_values[1], label="action 2")
        plt.title("Toy MDP")
        plt.legend()
        plt.xlabel('State [0, 1)')
        plt.ylabel('Value')
        r.savefig(f'figures/toy_mdp.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
        plt.close()

    doc @ """
    ## A Supervised Baseline
    
    **Can the function learn these value functions?** As it turned out, no.
    Even with a supervised learning objective, the learned value function is
    not able to produce a good approximation of the value landscape. Not
    with 20 states, and even less so with 200.
    """
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    num_states = 200
    mdp = RandMDP(seed=0, option='fixed')
    states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
    with doc:
        q_values, losses = supervised(states, gt_q_values, dyn_mats, lr=3e-4)
    with doc.table().figure_row() as r:
        plot_value(states, q_values, losses, f"supervised", doc=r)

    doc @ """
    ## Supervised, Random Fourier Features
    
    **Can the function learn these value functions?** As it turned out, no.
    Even with a supervised learning objective, the learned value function is
    not able to produce a good approximation of the value landscape. Not
    with 20 states, and even less so with 200.
    """
    from rand_mdp import RandMDP
    from matplotlib import pyplot as plt

    num_states = 200
    mdp = RandMDP(seed=0, option='fixed')
    states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
    with doc, doc.table() as t:
        for scale in [1, 10, 100]:
            q_values, losses = supervised_rff(states, gt_q_values, dyn_mats, lr=3e-4, B_scale=scale)
            r = t.figure_row()
            plot_value(states, q_values, losses, f"rff_{scale}", doc=r)
