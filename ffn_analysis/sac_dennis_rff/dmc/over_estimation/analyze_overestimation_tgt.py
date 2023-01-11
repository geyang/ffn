from ml_logger import ML_Logger
from sac_rff import Args
import numpy as np
from matplotlib import pyplot as plt
import os

def plot_line(path, color, label, seeds = [100, 200, 300, 400, 500], style='-'):
    epochs = [1500, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000,]
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

env_names = ['Walker-run', 'Quadruped-walk', 'Quadruped-run', 'Hopper-hop']
scales = [0.001, 0.0003, 0.0001, 0.003]
no_tgt_scales = [0.0003, 0.0001, 0.0003, 0.003]

for (env_name, scale, no_tgt_scale) in zip(env_names, scales, no_tgt_scales):
    mlp_path = f'mlp/{env_name}/'

    if env_name == 'Quadruped-walk':
        plot_line(mlp_path, color='black', label='MLP', seeds=[100, 200, 300, 400])
    else:
        plot_line(mlp_path, color='black', label='MLP', seeds=[100, 200, 300, 400, 500])

    lff_path = f'lff/{env_name}/alpha_tune/scale-{scale}/'
    plot_line(lff_path, color=colors[0], label='FFN')
    no_tgt_lff_path = f'no_target/lff/{env_name}/alpha_tune/scale-{no_tgt_scale}/'
    plot_line(no_tgt_lff_path, color=colors[0], label='FFN (no-tgt)', style='--')
    plt.title(env_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{os.path.basename(__file__)[:-3]}/{env_name}.png', dpi=300, zoom=0.3)
    plt.savefig(f'{os.path.basename(__file__)[:-3]}/{env_name}.pdf', dpi=300, zoom=0.3)
    plt.close()