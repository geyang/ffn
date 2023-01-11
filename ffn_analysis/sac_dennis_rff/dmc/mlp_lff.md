# MUJOCO Comparisons
```python
loader = ML_Logger(prefix="/model-free/model-free/sac_dennis_rff/dmc/")
```
```python
def plot_line(path, color, label, x_key, y_key, linestyle='-'):
    mean, low, high, step, = loader.read_metrics(f"{y_key}@mean",
                                                 f"{y_key}@16%",
                                                 f"{y_key}@84%",
                                                 x_key=f"{x_key}@min", path=path, dropna=True)
    plt.xlabel('Frames', fontsize=18)
    plt.ylabel('Rewards', fontsize=18)

    plt.plot(step.to_list(), mean.to_list(), color=color, label=label, linestyle=linestyle)
    plt.fill_between(step, low, high, alpha=0.1, color=color)
```
```python
for (e, (env, scale, nt_scale)) in enumerate(tqdm(zip(envs, scales, no_tgt_scales), desc="env-scales")):

    if e % 4 == 0:
        r = doc.table().figure_row()
        
    plot_line(path=f"3_layer/mlp/{env}/**/metrics.pkl", color='black', label='MLP', x_key='frames', y_key="eval/episode_reward/mean")

    plot_line(path=f"3_layer/mlp_no_tgt/{env}/**/metrics.pkl", color='black', label='MLP (no-tgt)', x_key='frames',
                  y_key="eval/episode_reward/mean", linestyle='--')

    plot_line(path=f"2_layer/lff/{env}/alpha_tune/scale-{scale}/**/metrics.pkl", color=colors[0], label=f'LFF',
              x_key='frames', y_key="eval/episode_reward/mean")

    if env == 'Quadruped-run':
        plot_line(path=f"2_layer/lff_no_tgt/{env}/alpha_tune/scale-{nt_scale}/**/metrics.pkl", color=colors[0], label=f'LFF (no-tgt)',
                x_key='frames', y_key="eval/episode_reward/mean", linestyle='--')
    else:
        plot_line(path=f"2_layer/lff_no_tgt/{env}/scale-{nt_scale}/**/metrics.pkl", color=colors[0],
                  label=f'LFF (no-tgt)', x_key='frames', y_key="eval/episode_reward/mean", linestyle='--')
    plt.title(env)
    plt.legend()
    plt.tight_layout()
    r.savefig(f'{os.path.basename(__file__)[:-3]}/{env}.png', dpi=300, zoom=0.3, title=env)
    plt.savefig(f'{os.path.basename(__file__)[:-3]}/{env}.pdf', dpi=300, zoom=0.3)
    plt.close()
```

| **Cheetah-run** | **Acrobot-swingup** | **Hopper-hop** | **Quadruped-walk** |
|:---------------:|:-------------------:|:--------------:|:------------------:|
| <img style="align-self:center; zoom:0.3;" src="mlp_lff/Cheetah-run.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="mlp_lff/Acrobot-swingup.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="mlp_lff/Hopper-hop.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="mlp_lff/Quadruped-walk.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |

| **Quadruped-run** | **Humanoid-run** | **Finger-turn_hard** | **Walker-run** |
|:-----------------:|:----------------:|:--------------------:|:--------------:|
| <img style="align-self:center; zoom:0.3;" src="mlp_lff/Quadruped-run.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="mlp_lff/Humanoid-run.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="mlp_lff/Finger-turn_hard.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="mlp_lff/Walker-run.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
