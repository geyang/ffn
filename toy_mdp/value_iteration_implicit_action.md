
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
num_states = 200
torch.manual_seed(0)
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/value_iteration.png?ts=734180" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/value_iteration_loss.png?ts=374142" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ Function Approximator

Here we plot the value function learned via deep Q Learning 
(DQN) using a neural network function approximator.

```python
Q = Q_implicit(state_dim=1, action_dim=2)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats)
returns = eval_q_policy(Q)
doc.print(f"Avg return for DQN is {returns}")
```

```
Avg return for DQN is 4.334120562485238
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn.png?ts=302600" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_loss.png?ts=695518" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

**But can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
Q = Q_implicit(state_dim=1, action_dim=2)
q_values, losses = supervised(Q, states, gt_q_values, n_epochs=2000)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+sup is {returns}")
```

```
Avg return for NN+sup is 4.875278930566616
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/supervised.png?ts=060059" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/supervised_loss.png?ts=682513" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Now use RFF (supervised)

The same supervised experiment, instantly improve in fit if we 
replace the input layer with RFF embedding.

```python
Q = Q_implicit(state_dim=1, action_dim=2, rff=True, B_scale=10)
q_values, losses = supervised(Q, states, gt_q_values)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+RFF+sup is {returns}")
```

```
Avg return for NN+RFF+sup is 4.866300616615212
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/supervised_rff.png?ts=142827" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/supervised_rff_loss.png?ts=866022" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with RFF 

We can now apply this to DQN and it works right away! Using scale of 10

```python
Q = Q_implicit(state_dim=1, action_dim=2, rff=True, B_scale=10)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, )
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF is {returns}")
```

```
Avg return for DQN+RFF is 4.86064766773929
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_10.png?ts=845102" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_10_loss.png?ts=832865" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN without the Target Q

Setting the target network to off

```python
Q = Q_implicit(state_dim=1, action_dim=2, rff=True, B_scale=10)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, target_freq=None)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF-tgt is {returns}")
```

```
Avg return for DQN+RFF-tgt is 4.86064766773929
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_no_target.png?ts=257424" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_no_target_loss.png?ts=949801" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


We can experiment with different scaling $\sigma$

| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_1.png?ts=851598" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_1_loss.png?ts=624691" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=1$
```
Avg return for DQN+RFF (sigma 1) is 4.886965028306951
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_3.png?ts=966012" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_3_loss.png?ts=596483" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=3$
```
Avg return for DQN+RFF (sigma 3) is 4.847344025893762
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_5.png?ts=516936" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_implicit_action/dqn_rff_5_loss.png?ts=110882" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=5$
```
Avg return for DQN+RFF (sigma 5) is 4.906351557874841
```
