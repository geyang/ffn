
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
num_states = 200
torch.manual_seed(0)
mdp = ToyMDP(seed=0, k=10)
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/value_iteration.png?ts=861894" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/value_iteration_loss.png?ts=457191" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ Function Approximator

Here we plot the value function learned via deep Q Learning 
(DQN) using a neural network function approximator.

```python
def get_Q_mlp(n_layers=4):
    layers = []
    for _ in range(n_layers - 1):
        layers += [nn.Linear(400, 400), nn.ReLU()]
    return nn.Sequential(
        nn.Linear(1, 400), nn.ReLU(),
        *layers,
        nn.Linear(400, 2),
    )


Q = get_Q_mlp()
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats)
returns = eval_q_policy(Q)
doc.print(f"Avg return for DQN is {returns}")
```

```
Avg return for DQN is 3.2462685004525436
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn.png?ts=740897" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_loss.png?ts=263226" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

**But can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
Q = get_Q_mlp()
q_values, losses = supervised(Q, states, gt_q_values.T)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+sup is {returns}")
```

```
Avg return for NN+sup is 3.2744987001797523
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised.png?ts=701090" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_loss.png?ts=245530" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline with SGD

SGD does improve the fit.

```python
Q = get_Q_mlp()
q_values, losses = supervised(Q, states, gt_q_values.T, batch_size=32)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+sup is {returns}")
```

```
Avg return for NN+sup is 3.2759822502771554
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_sgd.png?ts=038401" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_sgd_loss.png?ts=555211" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


And we can let it run for longer

```python
Q = get_Q_mlp()
q_values, losses = supervised(Q, states, gt_q_values.T, n_epochs=2000, batch_size=32)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+sup is {returns}")
```

```
Avg return for NN+sup is 3.811454439797045
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_sgd_2000.png?ts=131179" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_sgd_2000_loss.png?ts=881185" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised SGD with Deeper Network (8)


```python
Q = get_Q_mlp(n_layers=8)
q_values, losses = supervised(Q, states, gt_q_values.T, batch_size=32)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+sup is {returns}")
```

```
Avg return for NN+sup is 3.2895100397628223
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_sgd_deep.png?ts=905482" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_sgd_deep_loss.png?ts=579273" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised SGD with Even Deeper Network (12)


```python
Q = get_Q_mlp(n_layers=12)
q_values, losses = supervised(Q, states, gt_q_values.T, batch_size=32)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+sup is {returns}")
```

```
Avg return for NN+sup is 3.8278396091901588
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_sgd_deeper.png?ts=633069" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_sgd_deeper_loss.png?ts=239146" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Now use RFF (supervised)

The same supervised experiment, instantly improve in fit if we 
replace the input layer with RFF embedding.

```python
def get_Q_rff(B_scale, n_layers=4):
    layers = []
    for _ in range(n_layers - 1):
        layers += [
            nn.Linear(400, 400),
            nn.ReLU(),
        ]
    return nn.Sequential(
        RFF(1, 200, scale=B_scale),
        *layers,
        nn.Linear(400, 2),
    )


Q = get_Q_rff(B_scale=10)
q_values, losses = supervised(Q, states, gt_q_values.T)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+RFF+sup is {returns}")
```

```
Avg return for NN+RFF+sup is 3.8648945470034595
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_rff.png?ts=660193" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/supervised_rff_loss.png?ts=227766" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with RFF 

We can now apply this to DQN and it works right away! Using scale of 10

```python
Q = get_Q_rff(B_scale=10)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF is {returns}")
```

```
Avg return for DQN+RFF is 3.8648327894362398
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_10.png?ts=185344" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_10_loss.png?ts=125868" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with a shallower RFF network

We can now apply this to DQN and it works right away! Using scale of 10

```python
Q = get_Q_rff(B_scale=10, n_layers=2)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF is {returns}")
```

```
Avg return for DQN+RFF is 3.860389719139694
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_shallow_10.png?ts=845276" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_shallow_10_loss.png?ts=425720" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN without the Target Q

Setting the target network to off

```python
Q = get_Q_rff(B_scale=10)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32, target_freq=None)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF-tgt is {returns}")
```

```
Avg return for DQN+RFF-tgt is 3.9086844258323934
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_no_target_10.png?ts=421851" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_no_target_10_loss.png?ts=172210" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Sweeping Different $\sigma$

We can experiment with different scaling $\sigma$

| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_1.png?ts=135981" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_1_loss.png?ts=877141" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=1$
```
Avg return for DQN+RFF (sigma 1) is 3.866187865830957
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_3.png?ts=186507" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_3_loss.png?ts=758163" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=3$
```
Avg return for DQN+RFF (sigma 3) is 3.938207012635294
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_5.png?ts=522102" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_5_loss.png?ts=118641" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=5$
```
Avg return for DQN+RFF (sigma 5) is 3.9392321782018245
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_10.png?ts=784287" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd/dqn_rff_10_loss.png?ts=480405" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=10$
```
Avg return for DQN+RFF (sigma 10) is 3.874372812112704
```
