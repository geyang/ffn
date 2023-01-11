import os

import matplotlib.pyplot as plt
from cmx import doc
from ml_logger import ML_Logger, memoize
from tqdm import tqdm

if __name__ == "__main__":
    # envs = ['Cheetah-run', 'Acrobot-swingup', 'Hopper-hop', 'Quadruped-walk',
    #         'Quadruped-run', 'Humanoid-run', 'Finger-turn_hard', 'Walker-run']
    #
    # scales = [0.001, 0.003, 0.003, 0.0003,
    #           0.0001, 0.001, 0.001, 0.001]

    envs = ['Quadruped-run',]
    scales = [0.0001,]

    colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247']

    import os

    with doc @ """# MUJOCO Comparisons""":
        loader = ML_Logger(prefix="/model-free/model-free/sac_dennis_rff/dmc/")

    # loader.glob = memoize(loader.glob)
    # loader.read_metrics = memoize(loader.read_metrics)

    with doc:
        def plot_line(path, color, label, x_key, y_key, linestyle='-'):
            mean, low, high, step, = loader.read_metrics(f"{y_key}@mean",
                                                         f"{y_key}@16%",
                                                         f"{y_key}@84%",
                                                         x_key=f"{x_key}@min", path=path, dropna=True)
            plt.xlabel('Frames', fontsize=18)
            plt.ylabel('Rewards', fontsize=18)

            plt.plot(step.to_list(), mean.to_list(), color=color, label=label, linestyle=linestyle)
            plt.fill_between(step, low, high, alpha=0.1, color=color)

    with doc:
        for (e, (env, scale)) in enumerate(tqdm(zip(envs, scales), desc="env-scales")):

            if e % 4 == 0:
                r = doc.table().figure_row()

            plot_line(path=f"3_layer/mlp/{env}/**/metrics.pkl", color='black', label='MLP', x_key='frames',
                      y_key="eval/episode_reward/mean")

            plot_line(path=f"2_layer/lff/{env}/alpha_tune/scale-{scale}/**/metrics.pkl", color=colors[0], label=f'LFF',
                      x_key='frames', y_key="eval/episode_reward/mean")

            for (j, update_freq) in enumerate([3,4]):
                plot_line(path=f"compute/lff/{env}/alpha_tune/scale-{scale}/update_freq-{update_freq}/**/metrics.pkl", color=colors[j+1],
                          label=f'LFF (update freq {update_freq})',
                          x_key='frames', y_key="eval/episode_reward/mean")
                # plot_line(path=f"compute/mlp/{env}/update_freq-{update_freq}/**/metrics.pkl",
                #           color=colors[j + 1],
                #           label=f'LFF (update freq {update_freq})',
                #           x_key='frames', y_key="eval/episode_reward/mean", linestyle='--')

            plt.title(env)
            plt.legend()
            plt.tight_layout()
            r.savefig(f'{os.path.basename(__file__)[:-3]}/{env}.png', dpi=300, zoom=0.3, title=env)
            plt.savefig(f'{os.path.basename(__file__)[:-3]}/{env}.pdf', dpi=300, zoom=0.3)
            plt.close()

    doc.flush()
