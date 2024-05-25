
# Spectral Bias

here is a random superposition of 10 fourier components (random phase).

```python
xs = np.linspace(0, 1, 401)
ys = np.stack([np.sin(np.random.random() + 2 * np.pi * k * xs) for k in range(5, 55, 5)]).sum(axis=0)
```
```python
ys_bar, losses = supervised(xs, ys[..., None], batch_size=32, n_epochs=80000)
```
| <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd/spectral_bias_sgd.png?ts=024667" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd/spectral_bias_sgd_loss.png?ts=494515" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Supervised Learning with MLP and RFF

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
states = np.loadtxt("data/states.csv", delimiter=',')
gt_q_values = np.loadtxt("data/q_values.csv", delimiter=',')
```

## Supervised Learning with SGD

Here we use the a batch size of 32

```python
gt_q_values = np.loadtxt("data/q_values.csv", delimiter=',')
```
| <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd/supervised_sgd.png?ts=586084" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd/supervised_sgd_loss.png?ts=339626" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Supervised Learning with GD

Here we use the entire dataset as a single batch (GD).

```python
title="Supervised with SGD", doc=doc.table().figure_row())
```
| <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd/supervised_gd.png?ts=128570" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd/supervised_gd_loss.png?ts=897842" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
