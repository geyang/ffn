# MUJOCO Comparisons
```python
loader = ML_Logger(prefix="model-free/model-free")
```
```python
def plot_line(path, color, label, x_key):
    mean, low, high, step, = loader.read_metrics("eval/episode_reward@mean",
                                                 "eval/episode_reward@16%",
                                                 "eval/episode_reward@84%",
                                                 bin_size=2, x_key=f"{x_key}@min", path=path)
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Rewards', fontsize=18)

    mean = mean[:51]
    low = low[:51]
    high = high[:51]
    step = step[:51]

    if color is None:
        plt.plot(step.to_list(), mean.to_list(), label=label)
        plt.fill_between(step, low, high, alpha=0.1)
    else:
        plt.plot(step.to_list(), mean.to_list(), color=color, label=label)
        plt.fill_between(step, low, high, alpha=0.1, color=color)
```
```python
r = doc.table().figure_row()
for (env, b_val) in tqdm(zip(envs, b_vals), desc="env"):
    plot_line(path=f"drqv2_crff_analysis/wavelet/dmc/mlp/mlp/{env}/**/metrics.pkl", color='black',
              label='CNN', x_key='eval/frame')
    plot_line(path=f"drqv2_crff_analysis/rff/dmc/rff/scale-{b_val}/{env}/**/metrics.pkl",
              label='F-CNN', x_key='eval/frame', color=colors[0])

    plt.title(env)
    plt.legend()
    plt.tight_layout()
    [line.set_zorder(100) for line in plt.gca().lines]
    [spine.set_zorder(100) for spine in plt.gca().collections]
    r.savefig(f'{os.path.basename(__file__)[:-3]}/{env}.png', dpi=300, zoom=0.3, title=env)
    plt.savefig(f'{os.path.basename(__file__)[:-3]}/{env}.pdf', dpi=300, zoom=0.3)
    plt.close()
```

| **Cheetah-run** | **Hopper-hop** | **Quadruped-walk** | **Quadruped-run** |
|:---------------:|:--------------:|:------------------:|:-----------------:|
| <img style="align-self:center; zoom:0.3;" src="cnn_lff/Cheetah-run.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="cnn_lff/Hopper-hop.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="cnn_lff/Quadruped-walk.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="cnn_lff/Quadruped-run.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
