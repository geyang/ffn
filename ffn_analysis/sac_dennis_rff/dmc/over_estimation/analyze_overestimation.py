from ml_logger import ML_Logger
from sac_rff import Args
import numpy as np
from matplotlib import pyplot as plt
import os

def plot_line(path, color, label, seeds = [100, 200, 300, 400, 500], style='-'):
    epochs = [1500, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000,
              1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000,]
    residuals = []

    for seed in seeds:
        residuals.append([])
        for epoch in epochs:
            this_path = path + f'{seed}/analysis/results/estimate_{epoch:08d}.pkl'
            file = loader.load_pkl(this_path)
            residual = np.abs(np.array(file[0]['value_estimate']) - np.array(file[0]['ep_return'])).mean()
            residuals[-1].append(residual)

    residuals = np.array(residuals)
    mean = residuals.mean(axis=0)
    std = residuals.std(axis=0)
    plt.plot(epochs, mean, color=color, label=label, linestyle=style)
    plt.fill_between(epochs, mean-std, mean+std, alpha=0.1, color=color)
    plt.xlabel('Frames', fontsize=18)
    plt.ylabel('Value estimation error', fontsize=18)


exp_root = "/model-free/model-free/sac_dennis_rff/dmc/over_estimation/value_estimation"
loader = ML_Logger(prefix=exp_root)
colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247']

# env_names = ['Cheetah-run', 'Acrobot-swingup', 'Hopper-hop', 'Quadruped-walk',
#              'Quadruped-run', 'Humanoid-run', 'Finger-turn_hard', 'Walker-run']

env_names = ['Humanoid-run']
scales = [0.001]

# scales = [0.001, 0.003, 0.003, 0.0003,
#           0.0001, 0.001, 0.001, 0.001]

for (env_name, scale) in zip(env_names, scales):
    mlp_path = f'mlp/{env_name}/'
    plot_line(mlp_path, color='black', label='MLP')
    lff_path = f'lff/{env_name}/alpha_tune/scale-{scale}/'
    plot_line(lff_path, color=colors[0], label='LFF')
    plt.title(env_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{os.path.basename(__file__)[:-3]}/{env_name}.pdf', dpi=300, zoom=0.3)
    plt.close()