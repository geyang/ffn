
## Effective Rank

Here is the ground truth value function generated via tabular
value iteration. 

```python
num_states = 200
torch.manual_seed(0)
mdp = ToyMDP(seed=0, k=10)
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="network_rank/value_iteration.png?ts=971181" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="network_rank/value_iteration_loss.png?ts=597439" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

```python
Q = Q_mlp()
q_values, losses = supervised(Q, states, gt_q_values.T, n_epochs=100)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+sup is {returns}")
```

```
Avg return for NN+sup is 4.279941206988688
```
| <img style="align-self:center; zoom:0.3;" src="network_rank/supervised.png?ts=359609" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="network_rank/supervised_loss.png?ts=901952" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

```
rank tensor(1.9032)
```


## DQN w/ Function Approximator

```python
Q = Q_mlp()
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, batch_size=32)
returns = eval_q_policy(Q)
doc.print(f"Avg return for DQN is {returns}")
```

```
Avg return for DQN is 3.2523405086658994
```
| <img style="align-self:center; zoom:0.3;" src="network_rank/dqn.png?ts=078518" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="network_rank/dqn_loss.png?ts=579195" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

```
rank - DQN tensor(2.4796)
```


## Now use RFF (supervised)

The same supervised experiment, instantly improve in fit if we
replace the input layer with RFF embedding.

```python
Q = Q_rff(B_scale=10)
q_values, losses = supervised(Q, states, gt_q_values.T, batch_size=32)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+RFF+sup is {returns}")
```

```
Avg return for NN+RFF+sup is 3.8866797254344054
```
```
rank - DQN tensor(2.6985)
```

| <img style="align-self:center; zoom:0.3;" src="network_rank/supervised_rff.png?ts=687580" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/> | <img style="align-self:center; zoom:0.3;" src="network_rank/supervised_rff_loss.png?ts=221466" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
