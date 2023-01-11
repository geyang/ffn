from ml_logger import ML_Logger
from sac_rff import Args
import numpy as np
from matplotlib import pyplot as plt
import os
from cmx import doc

def plot_line(path, color, marker=None, label=None, style='-'):
    y_key = 'weight_diff'
    x_key = 'epoch'

    mean, low, high, step, = loader.read_metrics(f"{y_key}@mean",
                                                 f"{y_key}@16%",
                                                 f"{y_key}@84%",
                                                 x_key=f"{x_key}@min",
                                                 path=path,
                                                 dropna=True)

    if label:
        plt.plot(step.to_list(), mean.to_list(), color=color, label=label, linestyle=style, marker=marker, markersize=3.0, linewidth=1.0)
    else:
        plt.plot(step.to_list(), mean.to_list(), color=color, linestyle=style, marker=marker, markersize=3.0, linewidth=1.0)

    plt.fill_between(step, low, high, alpha=0.1, color=color)

    # if label:
    #     plt.errorbar(step, mean, yerr=std, color=color, label=label, linestyle=style, linewidth=1.0, marker=marker, markersize=2.0)
    # else:
    #     plt.errorbar(step, mean, yerr=std, color=color, linestyle=style, linewidth=1.0, marker=marker, markersize=2.0)

    plt.xlabel('Frames', fontsize=18)
    plt.ylabel('Relative weight change', fontsize=18)


exp_root = "/model-free/model-free/sac_dennis_rff/dmc/over_estimation/value_estimation"
loader = ML_Logger(prefix=exp_root)
colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247']

env_names = ['Cheetah-run', 'Acrobot-swingup', 'Hopper-hop', 'Quadruped-walk',
             'Quadruped-run', 'Humanoid-run', 'Finger-turn_hard', 'Walker-run']

scales = [0.001, 0.003, 0.003, 0.0003,
          0.0001, 0.001, 0.001, 0.001]

with doc:
    for (i, (env_name, scale)) in enumerate(zip(env_names, scales)):

        if i % 4 == 0:
            r = doc.table().figure_row()

        mlp_path = f'mlp/{env_name}/**/weight_diff_first/weight_diff.pkl'
        plot_line(mlp_path, color='cornflowerblue', label='layer 1')
        lff_path = f'lff/{env_name}/alpha_tune/scale-{scale}/**/weight_diff_first/weight_diff.pkl'
        plot_line(lff_path, color='cornflowerblue', marker='o')

        mlp_path = f'mlp/{env_name}/**/weight_diff_after_first/weight_diff.pkl'
        plot_line(mlp_path, color=colors[-1], label='layer 2-n')
        lff_path = f'lff/{env_name}/alpha_tune/scale-{scale}/**/weight_diff_after_first/weight_diff.pkl'
        plot_line(lff_path, color=colors[-1], marker='o')

        mlp_path = f'mlp/{env_name}/**/weight_diff_after_first/weight_diff.pkl'
        plot_line(mlp_path, color=colors[-1], label='MLP')
        lff_path = f'lff/{env_name}/alpha_tune/scale-{scale}/**/weight_diff_after_first/weight_diff.pkl'
        plot_line(lff_path, color=colors[-1], marker='o', label='FFN')

        plt.title(env_name)
        plt.legend()
        plt.tight_layout()
        [line.set_zorder(100) for line in plt.gca().lines]
        [spine.set_zorder(100) for spine in plt.gca().collections]
        r.savefig(f'weight_diff/{env_name}.png', dpi=300, zoom=0.3, title=env_name)
        plt.savefig(f'weight_diff/{env_name}.pdf', dpi=300, zoom=0.3)
        plt.close()