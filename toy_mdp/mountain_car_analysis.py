import itertools
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from gym.envs.classic_control import MountainCarEnv
from matplotlib import pyplot as plt
from tqdm import trange

gamma = 0.995
device = 'cpu'
env = MountainCarEnv()
env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


# %%

def plot_q_vals(states, q_vals, color='b'):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.scatter(states[:, 0], states[:, 1], q_vals, color=color)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_xlim(states[:, 0].min(), states[:, 0].max())
    ax.set_ylim(states[:, 1].min(), states[:, 1].max())
    ax.set_zlim(q_vals.min(), q_vals.max())
    plt.show()


# %%

def get_discrete_mdp(num_bins=100):
    env = MountainCarEnv()
    env.reset()

    low = [-1.2, -0.07]
    high = [0.6, 0.07]

    state_ranges = [np.linspace(low[i], high[i], num_bins) for i in range(len(low))]
    states = [np.array(x) for x in itertools.product(*state_ranges)]
    states = np.array(states)

    num_states = states.shape[0]
    num_actions = 3

    rewards = np.zeros((num_actions, num_states))
    dones = np.zeros((num_actions, num_states))

    dyn_mats = np.zeros((num_actions, num_states, num_states))

    for state_id in range(num_states):
        this_state = states[state_id]
        for action in range(num_actions):
            env.state = this_state.tolist()
            new_state, reward, done, _ = env.step(action)
            rewards[action, state_id] = reward
            dones[action, state_id] = float(done)
            dst = np.linalg.norm(states - new_state, axis=1)
            next_state_id = np.argmin(dst)
            dyn_mats[action, state_id, next_state_id] = 1.0

    return states, rewards, dones, dyn_mats


# %%

def perform_vi(states, rewards, dones, dyn_mats, gamma=0.9, eps=1e-5):
    # Assume discrete actions and states
    q_values = np.zeros(dyn_mats.shape[:2])

    deltas = []
    while not deltas or deltas[-1] >= eps:
        old = q_values
        q_max = q_values.max(axis=0)
        q_values = rewards + gamma * (1 - dones) * (dyn_mats @ q_max)

        deltas.append(np.abs(old - q_values).max())

    return q_values, deltas


# %%

def perform_deep_vi(Q, states, rewards, dones, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, target_freq=10):
    # Ge: need to initialize the Q function at zero
    Q.to(device)
    Q_target = deepcopy(Q) if target_freq else Q

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)
    dyn_mats = torch.FloatTensor(dyn_mats).to(device)

    losses = []

    for epoch in trange(n_epochs + 1):
        if target_freq and epoch % target_freq == 0:
            Q_target.load_state_dict(Q.state_dict())

        q_max, actions = Q_target(states).max(dim=-1)
        td_target = rewards + gamma * (1 - dones) * (dyn_mats @ q_max.detach())
        td_loss = l1(Q(states), td_target.T)
        losses.append(td_loss.detach().cpu().item())

        optim.zero_grad()
        td_loss.backward()
        optim.step()

    q_values = Q(states).cpu().T.detach().numpy()
    return q_values, losses


# %%

class Q_table_wrapper:
    def __init__(self, states, q_vals):
        self.states = states
        self.q_vals = q_vals

    def __call__(self, state):
        state = state.cpu().numpy().flatten()
        dst = np.linalg.norm(self.states - state, axis=1)
        idx = np.argmin(dst)
        return torch.FloatTensor(self.q_vals[:, idx])


class RFF(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = mapping_size * 2
        self.B = torch.normal(0, scale, size=(mapping_size, self.input_dim)).to(device)
        self.B.requires_grad = False

    def forward(self, x):
        return torch.cat([torch.cos(2 * np.pi * x @ self.B.T),
                          torch.sin(2 * np.pi * x @ self.B.T)], dim=1)


def get_Q_mlp():
    return nn.Sequential(
        nn.Linear(state_dim, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, action_dim),
    )


def get_Q_rff(B_scale):
    return nn.Sequential(
        RFF(state_dim, 200, scale=B_scale),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, action_dim),
    )


# %%

def eval_q_policy(q, num_eval=100):
    """Assumes discrete action such that policy is derived by argmax a Q(s,a)"""
    env = gym.make('MountainCar-v0')
    returns = []

    for i in range(num_eval):
        done = False
        obs = env.reset()
        total_rew = 0
        while not done:
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
            q_max, action = q(obs).max(dim=-1)
            obs, rew, done, _ = env.step(action.item())
            total_rew += rew
        returns.append(total_rew)

    return np.array(returns)


# %%

states, rewards, dones, dyn_mats = get_discrete_mdp(num_bins=150)
q_values, losses = perform_vi(states, rewards, dones, dyn_mats, gamma=gamma)

# %%

q_fn = Q_table_wrapper(states, q_values)
returns = eval_q_policy(q_fn)
print(returns.mean())
plt.imshow(q_values)
plt.show()
q_values.shape
plot_q_vals(states, q_values[0], color='r')
plot_q_vals(states, q_values[1], color='b')
plot_q_vals(states, q_values[2], color='g')

# %%

q_nn = get_Q_mlp()
q_nn_values, losses_nn = perform_deep_vi(q_nn, states, rewards, dones, dyn_mats, lr=1e-4, gamma=0.995, n_epochs=2000,
                                         target_freq=1)

# %%

returns_nn = eval_q_policy(q_nn)
print(returns_nn.mean())

# %%

plot_q_vals(states, q_nn_values[0], color='r')
plot_q_vals(states, q_nn_values[1], color='b')
plot_q_vals(states, q_nn_values[2], color='g')

# %%

q_rff = get_Q_rff(B_scale=3)
q_rff_values, losses_rff = perform_deep_vi(q_rff, states, rewards, dones, dyn_mats, lr=1e-4, gamma=0.995, n_epochs=2000,
                                           target_freq=1)

# %%

returns_rff = eval_q_policy(q_rff)
print(returns_rff.mean())

# %%

plot_q_vals(states, q_rff_values[0], color='r')
plot_q_vals(states, q_rff_values[1], color='b')
plot_q_vals(states, q_rff_values[2], color='g')

# %%

q_nn_values.shape

# %%

torch.cuda.empty_cache()

# %%
