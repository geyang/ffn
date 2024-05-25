import itertools
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from gym.envs.classic_control import MountainCarEnv
from matplotlib import pyplot as plt
from params_proto.neo_proto import PrefixProto
from tqdm import trange


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


def get_discrete_mdp(num_bins=100, seed=100):
    env = MountainCarEnv()
    env.seed(seed)
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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return states, rewards, dones, dyn_mats, state_dim, action_dim


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


def perform_vi_mlp(Q, states, rewards, dones, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=400, target_freq=10, device='cpu'):
    # Ge: need to initialize the Q function at zero
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
        state = state.detach().cpu().numpy().flatten()
        dst = np.linalg.norm(self.states - state, axis=1)
        idx = np.argmin(dst)
        return torch.FloatTensor(self.q_vals[:, idx])


class FF(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = mapping_size * 2
        self.B = nn.Parameter(torch.normal(0, scale, size=(mapping_size, self.input_dim)))
        self.B.requires_grad = False

    def forward(self, x):
        return torch.cat([torch.cos(2 * np.pi * x @ self.B.T),
                          torch.sin(2 * np.pi * x @ self.B.T)], dim=1)


class LFF(nn.Module):
    """
    get torch.std_mean(self.B)
    """

    def __init__(self, in_features, mapping_size, scale=1.0):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = mapping_size
        self.linear = nn.Linear(in_features, self.output_dim)
        nn.init.uniform_(self.linear.weight, - scale / self.input_dim, scale / self.input_dim)
        nn.init.uniform_(self.linear.bias, -1, 1)

    def forward(self, x):
        x = np.pi * self.linear(x)
        return torch.sin(x)


class RFF(LFF):
    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__(input_dim, mapping_size, scale=scale)
        self.linear.requires_grad = False


def get_Q_mlp(in_feat, out_feat, n_layers=4):
    layers = []
    for _ in range(n_layers - 1):
        layers += [nn.Linear(400, 400), nn.ReLU()]
    return nn.Sequential(
        nn.Linear(in_feat, 400), nn.ReLU(),
        *layers,
        nn.Linear(400, out_feat))


def get_Q_rff(in_feat, out_feat, B_scale):
    return nn.Sequential(
        RFF(in_feat, 400, scale=B_scale),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, out_feat),
    )


def eval_q_policy(q, num_eval=100, seed=100, device='cpu'):
    """Assumes discrete action such that policy is derived by argmax a Q(s,a)"""
    env = gym.make('MountainCar-v0')
    env.seed(seed)
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


def train_vi(num_bins=150, seed=100, gamma=0.995, eps=1e-5, **_):
    from ml_logger import logger
    states, rewards, dones, dyn_mats, state_dim, action_dim = get_discrete_mdp(num_bins=num_bins, seed=seed)

    q_values, losses = perform_vi(states, rewards, dones, dyn_mats, gamma=gamma, eps=eps)
    q_fn = Q_table_wrapper(states, q_values)

    returns = eval_q_policy(q_fn, seed=seed + 42)
    logger.log(returns=returns.mean(), flush=True)
    logger.print('vi', returns.mean())
    logger.save_pkl(q_values, 'q_vi.pkl')

    # q_values.shape = 3, 150, 150
    logger.save_image(q_values.T.reshape([num_bins, num_bins, 3]), key="figures/vi_tabular.png", normalize="individual")
    for i in range(3):
        logger.save_image(q_values[0].reshape([num_bins, num_bins]), key=f"figures/vi_tabular_{i}.png",
                          normalize="individual")


def train_vi_mlp(num_bins=150, seed=100, n_epochs=2000, lr=1e-4, gamma=0.995, target_freq=1, device=None, n_layers=4,
                 **_):
    from ml_logger import logger
    states, rewards, dones, dyn_mats, state_dim, action_dim = get_discrete_mdp(num_bins=num_bins, seed=seed)

    q_nn = get_Q_mlp(state_dim, action_dim, n_layers).to(device)
    q_values, losses_nn = perform_vi_mlp(q_nn, states, rewards, dones, dyn_mats, lr=lr, gamma=gamma,
                                         n_epochs=n_epochs, target_freq=target_freq, device=device)

    returns = eval_q_policy(q_nn, seed=seed + 42, device=device)
    logger.log(returns=returns.mean(), flush=True)
    logger.print('vi + mlp', returns.mean())
    logger.save_pkl(q_values, 'q_mlp.pkl')

    # q_values.shape = 3, 150, 150
    logger.save_image(q_values.T.reshape([num_bins, num_bins, 3]), key="figures/vi_mlp.png",
                      normalize='individual')
    for i in range(3):
        logger.save_image(q_values[i].reshape([num_bins, num_bins]), key=f"figures/vi_mlp_{i}.png",
                          normalize='individual')


def train_vi_rff(num_bins=150, seed=100, n_epochs=2000, lr=1e-4, gamma=0.995, target_freq=1, B_scale=3, device=None,
                 **_):
    from ml_logger import logger
    states, rewards, dones, dyn_mats, state_dim, action_dim = get_discrete_mdp(num_bins=num_bins, seed=seed)

    q_rff = get_Q_rff(state_dim, action_dim, B_scale=B_scale).to(device)
    q_values, losses_rff = perform_vi_mlp(q_rff, states, rewards, dones, dyn_mats, lr=lr, gamma=gamma,
                                          n_epochs=n_epochs, target_freq=target_freq, device=device)

    returns = eval_q_policy(q_rff, seed=seed + 42, device=device)
    logger.log(returns=returns.mean(), flush=True)
    logger.print('vi + rff', returns.mean())
    logger.save_pkl(q_values, 'q_rff.pkl')
    # logger.print('vi + rff q range:', q_values.min(dim=[1, 2]), q_values.max(dim=[1, 2]))

    # q_values.shape = 3, 150, 150
    logger.save_image(q_values.T.reshape([num_bins, num_bins, 3]), key="figures/vi_rff.png",
                      normalize='individual')
    for i in range(3):
        logger.save_image(q_values[i].reshape([num_bins, num_bins]), key=f"figures/vi_rff_{i}.png",
                          normalize='individual')


class MntCar(PrefixProto):
    gamma = 0.995
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 100
    num_bins = 150
    n_epochs = 2000
    n_layers = 4


def launch(fn_name, prefix, **kwargs):
    from overcoming_spectral_bias.mountain_car_analysis.mountain_car_fitted_q_iteration import train_vi, train_vi_mlp, \
        train_vi_rff, MntCar
    from ml_logger import logger
    logger.configure(prefix=prefix)

    logger.print("torch cuda is available:", torch.cuda.is_available())
    MntCar._update(**kwargs)
    logger.log_params(MntCar=vars(MntCar))
    logger.job_started()

    torch.random.manual_seed(MntCar.seed)
    np.random.seed(MntCar.seed)

    try:
        eval(fn_name)(**vars(MntCar))
        logger.job_completed()
    except Exception as e:
        logger.print(e, file="output.error")
        logger.job_errored()
        raise e


if __name__ == '__main__':
    import jaynes
    from ml_logger import logger

    # now = logger.now()
    # prefix = f"conjugate-kernel/conjugate-kernel/mountain_car/mountain_car_analysis/{now:%Y-%m-%d}/{now:%H%M%S.%f}"
    # for seed in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
    #     prefix = f"conjugate-kernel/conjugate-kernel/mountain_car/mountain_car_analysis/tabular/epoch_{MntCar.n_epochs}/s{seed}"
    #     jaynes.config(runner=dict(job_name=prefix[-60:]))
    #     jaynes.run(launch, 'train_vi', prefix=prefix)
    #
    # for seed in [100, 200]:
    #     prefix = f"conjugate-kernel/conjugate-kernel/mountain_car/mountain_car_analysis/mlp-deep/epoch_{MntCar.n_epochs}/s{seed}"
    #     jaynes.config(runner=dict(job_name=prefix[-60:]))
    #     jaynes.run(launch, 'train_vi_mlp', prefix=prefix, seed=seed, n_layes=8)

    for B_scale in [5, 10, 20, 40]:
        for seed in [100, ]:
            prefix = f"conjugate-kernel/conjugate-kernel/mountain_car/mountain_car_analysis/rff_2/epoch_{MntCar.n_epochs}/B_scale/1/B_{B_scale}/s{seed}"
            jaynes.config(runner=dict(job_name=prefix[-60:]))
            jaynes.run(launch, 'train_vi_rff', prefix=prefix, B_scale=B_scale, seed=seed)

    jaynes.listen()
