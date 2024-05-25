
# Spectral Bias

here is a random superposition of 10 fourier components (random phase).

```python
xs = np.linspace(0, 1, 401)
ys = np.stack([np.sin(np.random.random() + 2 * np.pi * k * xs) for k in range(5, 55, 5)]).sum(axis=0)
```
```python
ys_bar, losses = supervised(xs, ys[..., None], batch_size=32, n_epochs=400)
```
| <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd_3/spectral_bias_sgd.png?ts=493605" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd_3/spectral_bias_sgd_loss.png?ts=940125" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


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
q_values, losses = supervised(states, gt_q_values.T, batch_size=32, n_epochs=400)
```
| <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd_3/supervised_sgd.png?ts=591772" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd_3/supervised_sgd_loss.png?ts=010328" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Supervised Learning with GD

Here we use the entire dataset as a single batch (GD).

```python
q_values, losses = supervised(states, gt_q_values.T, n_epochs=400)
```
| <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd_3/supervised_gd.png?ts=023729" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="sgd_vs_gd_3/supervised_gd_loss.png?ts=399112" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
