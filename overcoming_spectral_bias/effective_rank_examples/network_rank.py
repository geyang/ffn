import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange


def plot_value(states, target, fit, losses, fig_prefix, title=None, doc=doc):
    plt.plot(states, fit[0], linewidth=2, label="Action I", color='red')
    plt.plot(states, fit[1], linewidth=2, label="Action II", color="#5a5a6e")
    if title:
        plt.title(title)
    # plt.ylim(0, 12)
    # plt.legend(loc=(0.65, 0.7), handlelength=1)
    plt.legend(loc='upper right', handlelength=1.0)
    plt.xlabel('State [0, 1)')
    plt.ylabel('Value')
    doc.savefig(f'{Path(__file__).stem}/{fig_prefix}.png?ts={doc.now("%f")}', zoom=0.3)
    plt.title("")  # remove title in PDF for use in LaTeX
    plt.savefig(f'{Path(__file__).stem}/{fig_prefix}.pdf')
    plt.close()

    plt.plot(losses, color='#23aaff', linewidth=2)
    plt.hlines(0, 0, len(losses), linestyle='--', color='gray', linewidth=1)
    plt.title("Loss")
    plt.xlabel('Optimization Steps')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}_loss.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()


class Buffer:
    def __init__(self, *data, batch_size=None):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data[0])

    def sample(self, batch_size):
        inds = torch.rand(size=(self.__len__(),)).argsort()
        from more_itertools import chunked
        for batch_inds in chunked(inds, n=batch_size):
            yield [torch.Tensor(d[torch.stack(batch_inds)]) for d in self.data]


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


def supervised(f, xs, ys, lr=1e-4, n_epochs=100, batch_size=None):
    # Ge: need to initialize the Q function at zero

    optim = torch.optim.RMSprop(f.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    dataset = Buffer(xs[..., None], ys)

    losses = []
    for epoch in trange(n_epochs + 1, desc=f"batch_size={batch_size or 'all'}"):
        for x_batch, y_batch in dataset.sample(batch_size=batch_size or len(dataset)):
            y_bar = f(x_batch)
            loss = l1(y_bar, y_batch)
            losses.append(loss.detach().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

    with torch.no_grad():
        all_xs = torch.Tensor(xs[..., None])
        return f(all_xs).squeeze().numpy().T, losses


def perform_deep_vi(Q, states, rewards, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, batch_size=None, target_freq=10):
    # Ge: need to initialize the Q function at zero
    Q_target = deepcopy(Q) if target_freq else Q

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    rewards = torch.FloatTensor(rewards)
    dyn_mats = torch.FloatTensor(dyn_mats)

    n = len(states)
    dataset = Buffer(states, np.arange(n))

    losses = []

    for epoch in trange(n_epochs + 1, desc=f"batch_size={batch_size or 'all'}"):
        for state_batch, inds in dataset.sample(batch_size=batch_size or len(dataset)):
            if target_freq and epoch % target_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            inds = inds.numpy().astype(int)
            next_inds = torch.argmax(dyn_mats[:, inds, :], dim=-1)

            q_max, actions = Q_target(states[next_inds.reshape(-1)]).max(dim=-1)
            td_target = rewards[:, inds] + gamma * q_max.reshape(2, -1).detach()
            td_loss = l1(Q(state_batch), td_target.T)
            losses.append(td_loss.detach().numpy())

            optim.zero_grad()
            td_loss.backward()
            optim.step()

    q_values = Q(states).T.detach().numpy()
    return q_values, losses


def kernel_Q(q_values, states):
    pass


def eval_q_policy(q, num_eval=100):
    """Assumes discrete action such that policy is derived by argmax a Q(s,a)"""
    from rand_mdp import ToyMDP
    torch.manual_seed(0)
    env = ToyMDP(seed=0, k=10)
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


class Q_mlp(nn.Sequential):
    def __init__(self, n_layers=4):
        self.n_layers = n_layers
        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(400, 400), nn.ReLU()]
        super().__init__(
            nn.Linear(1, 400), nn.ReLU(),
            *layers,
            nn.Linear(400, 2),
        )

    def feat(self, xs):
        *all_modules, last = list(self)
        for module in all_modules:
            xs = module(xs)
        return xs


class Q_rff(nn.Sequential):
    def __init__(self, B_scale, n_layers=4):
        self.B_scale = B_scale
        self.n_layers = n_layers
        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(400, 400), nn.ReLU()]
        super().__init__(
            RFF(1, 200, scale=B_scale),
            *layers,
            nn.Linear(400, 2),
        )

    def feat(self, xs):
        *all_modules, last = list(self)
        for module in all_modules:
            xs = module(xs)
        return xs


@torch.no_grad()
def effective_rank(f, xs, ):
    feats = f(xs)
    feat_matrix = feats @ feats.T
    sgv = torch.linalg.svdvals(feat_matrix)
    ps = sgv / sgv.sum()
    return torch.exp(- torch.sum(ps * torch.log(ps)))


if __name__ == "__main__":
    doc @ """
    ## Effective Rank
    
    Here is the ground truth value function generated via tabular
    value iteration. 
    """
    from rand_mdp import ToyMDP
    from matplotlib import pyplot as plt

    with doc:
        num_states = 200
        torch.manual_seed(0)
        mdp = ToyMDP(seed=0, k=10)
        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
        q_values, losses = perform_vi(states, rewards, dyn_mats)

    gt_q_values = q_values  # used later

    plot_value(states, gt_q_values, q_values, losses, fig_prefix="value_iteration",
               title="Value Iteration on Toy MDP", doc=doc.table().figure_row())

    doc @ """
    ## A Supervised Baseline
    """
    with doc:
        Q = Q_mlp()
        q_values, losses = supervised(Q, states, gt_q_values.T, n_epochs=100)
        returns = eval_q_policy(Q)

        doc.print(f"Avg return for NN+sup is {returns}")

    plot_value(states, gt_q_values, q_values, losses, fig_prefix="supervised", title="Supervised Value Function",
               doc=doc.table().figure_row())

    rank = effective_rank(Q.feat, torch.Tensor(states)[:, None])
    doc.print("rank", rank)

    doc @ """
    ## DQN w/ Function Approximator
    """
    with doc:
        Q = Q_mlp()
        q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32)
        returns = eval_q_policy(Q)
        doc.print(f"Avg return for DQN is {returns}")

    plot_value(states, gt_q_values, q_values, losses, fig_prefix="dqn",
               title="DQN on Toy MDP", doc=doc.table().figure_row())

    rank = effective_rank(Q.feat, torch.Tensor(states)[:, None])
    doc.print("rank - DQN", rank)

    # doc @ """
    # ## A Supervised Baseline with SGD
    # 
    # SGD does improve the fit.
    # """
    # with doc:
    #     Q = Q_mlp()
    #     q_values, losses = supervised(Q, states, gt_q_values.T, n_epochs=2000, batch_size=32)
    #     returns = eval_q_policy(Q)
    # 
    #     doc.print(f"Avg return for NN+sup is {returns}")
    # 
    # plot_value(states, gt_q_values, q_values, losses, fig_prefix="supervised_sgd",
    #            title="Supervised Value Function (SGD)", doc=doc.table().figure_row())
    # 
    # doc @ """
    # ## A Supervised SGD with Deeper Network (8)
    # 
    # """
    # with doc:
    #     Q = Q_mlp(n_layers=8)
    #     q_values, losses = supervised(Q, states, gt_q_values.T, n_epochs=2000, batch_size=32)
    #     returns = eval_q_policy(Q)
    # 
    #     doc.print(f"Avg return for NN+sup is {returns}")
    # 
    # plot_value(states, gt_q_values, q_values, losses, fig_prefix="supervised_sgd_deep",
    #            title="Supervised Value Function (SGD)", doc=doc.table().figure_row())
    # 
    # doc @ """
    # ## A Supervised SGD with Even Deeper Network (12)
    # 
    # """
    # with doc:
    #     Q = Q_mlp(n_layers=12)
    #     q_values, losses = supervised(Q, states, gt_q_values.T, n_epochs=2000, batch_size=32)
    #     returns = eval_q_policy(Q)
    # 
    #     doc.print(f"Avg return for NN+sup is {returns}")
    # 
    # plot_value(states, gt_q_values, q_values, losses, fig_prefix="supervised_sgd_deeper",
    #            title="Supervised Value Function (SGD)", doc=doc.table().figure_row())
    # 
    # doc @ """
    # ## A Supervised Baseline with SGD but more epochs
    # 
    # SGD does improve the fit.
    # """
    # with doc:
    #     Q = Q_mlp()
    #     q_values, losses = supervised(Q, states, gt_q_values.T, n_epochs=20000, batch_size=32)
    #     returns = eval_q_policy(Q)
    # 
    #     doc.print(f"Avg return for NN+sup is {returns}")
    # 
    # plot_value(states, gt_q_values, q_values, losses, fig_prefix="supervised_sgd_longer",
    #            title="Supervised Value Function (SGD)", doc=doc.table().figure_row())

    doc @ """
    ## Now use RFF (supervised)

    The same supervised experiment, instantly improve in fit if we
    replace the input layer with RFF embedding.
    """
    with doc:

        Q = Q_rff(B_scale=10)
        q_values, losses = supervised(Q, states, gt_q_values.T, batch_size=32)
        returns = eval_q_policy(Q)

        doc.print(f"Avg return for NN+RFF+sup is {returns}")

    rank = effective_rank(Q.feat, torch.Tensor(states)[:, None])
    doc.print("rank - DQN", rank)

    plot_value(states, gt_q_values, q_values, losses, fig_prefix="supervised_rff", title=f"RFF Supervised {10}",
               doc=doc.table().figure_row())

    doc.flush()
    exit()

    doc @ """
    ## DQN with RFF 
    
    We can now apply this to DQN and it works right away! Using scale of 10
    """
    with doc:
        Q = Q_rff(B_scale=10)
        q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32)
        returns = eval_q_policy(Q)

        doc.print(f"Avg return for DQN+RFF is {returns}")

    plot_value(states, gt_q_values, q_values, losses, fig_prefix=f"dqn_rff_{10}",
               title=f"DQN RFF $\sigma={10}$", doc=doc.table().figure_row())

    doc @ """
    ## DQN without the Target Q
    
    Setting the target network to off
    """
    with doc:
        Q = Q_rff(B_scale=10)
        q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32, target_freq=None)
        returns = eval_q_policy(Q)

        doc.print(f"Avg return for DQN+RFF-tgt is {returns}")

    plot_value(states, gt_q_values, q_values, losses, fig_prefix=f"dqn_rff_no_target", title=f"DQN RFF No Target",
               doc=doc.table().figure_row())

    doc @ """
    ## Sweeping Different $\sigma$
    
    We can experiment with different scaling $\sigma$
    """
    for sigma in [1, 3, 5, 10]:
        r = doc.table().figure_row()
        Q = Q_rff(B_scale=sigma)
        q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats)
        returns = eval_q_policy(Q)

        doc @ f"$\sigma={sigma}$"
        doc.print(f"Avg return for DQN+RFF (sigma {sigma}) is {returns}")

        plot_value(states, gt_q_values, q_values, losses, fig_prefix=f"dqn_rff_{sigma}",
                   title=f"DQN RFF $\sigma={sigma}$", doc=r)

    doc.flush()
