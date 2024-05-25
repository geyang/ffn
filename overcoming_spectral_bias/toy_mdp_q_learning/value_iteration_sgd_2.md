
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
num_states = 200
torch.manual_seed(0)
mdp = ToyMDP(seed=0, use_example=True)
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/value_iteration.png?ts=431761" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/value_iteration_loss.png?ts=044642" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


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
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32)
returns = eval_q_policy(Q)
doc.print(f"Avg return for DQN is {returns}")
```

```
Avg return for DQN is 4.020977576327217
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn.png?ts=567316" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_loss.png?ts=110715" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

```python
def get_Q_rff(B_scale):
    return nn.Sequential(
        RFF(1, 200, scale=B_scale),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )
```

## DQN with RFF 

We can now apply this to DQN and it works right away! Using scale of 10

```python
Q = get_Q_rff(B_scale=10)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF is {returns}")
```

```
Avg return for DQN+RFF is 3.7034528501237163
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_10.png?ts=379754" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_10_loss.png?ts=134738" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN without the Target Q

Setting the target network to off

```python
Q = get_Q_rff(B_scale=10)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32, target_freq=None)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF-tgt is {returns}")
```

```
Avg return for DQN+RFF-tgt is 3.7311736995570435
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_no_target.png?ts=645241" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_no_target_loss.png?ts=164405" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Sweeping Different $\sigma$

We can experiment with different scaling $\sigma$

| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_1.png?ts=373304" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_1_loss.png?ts=969294" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=1$
```
Avg return for DQN+RFF (sigma 1) is 3.8745176853810044
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_3.png?ts=080795" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_3_loss.png?ts=635358" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=3$
```
Avg return for DQN+RFF (sigma 3) is 3.7227939530921415
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_5.png?ts=915150" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_5_loss.png?ts=454607" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=5$
```
Avg return for DQN+RFF (sigma 5) is 3.7325171123246053
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_10.png?ts=370164" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_sgd_2/dqn_rff_10_loss.png?ts=931338" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=10$
```
Avg return for DQN+RFF (sigma 10) is 3.68082451539094
```
