
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
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/value_iteration.png?ts=818117" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/value_iteration_loss.png?ts=209116" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ Function Approximator

Here we plot the value function learned via deep Q Learning 
(DQN) using a neural network function approximator.

```python
def get_Q_mlp():
    return nn.Sequential(
        nn.Linear(1, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )


Q = get_Q_mlp()
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats)
returns = eval_q_policy(Q)
doc.print(f"Avg return for DQN is {returns}")
```

```
Avg return for DQN is 4.883710136213835
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn.png?ts=084401" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_loss.png?ts=417289" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

**But can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
Q = get_Q_mlp()
q_values, losses = supervised(Q, states, gt_q_values, n_epochs=2000)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+sup is {returns}")
```

```
Avg return for NN+sup is 4.860335613303359
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised.png?ts=304633" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_loss.png?ts=725419" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Now use RFF (supervised)

The same supervised experiment, instantly improve in fit if we 
replace the input layer with RFF embedding.

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

Q = get_Q_rff(B_scale=10)
q_values, losses = supervised(Q, states, gt_q_values)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+RFF+sup is {returns}")
```

```
Avg return for NN+RFF+sup is 6.240936462960335
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_rff.png?ts=363558" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_rff_loss.png?ts=798291" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with RFF 

We can now apply this to DQN and it works right away! Using scale of 5

```python
Q = get_Q_rff(B_scale=10)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, )
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF is {returns}")
```

```
Avg return for DQN+RFF is 6.247257888800383
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_10.png?ts=797060" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_10_loss.png?ts=757007" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN without the Target Q

Setting the target network to off

```python
Q = get_Q_rff(B_scale=10)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, target_freq=None)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+RFF-tgt is {returns}")
```

```
Avg return for DQN+RFF-tgt is 6.264328494839547
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_no_target.png?ts=786772" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_no_target_loss.png?ts=197453" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


We can experiment with different scaling $\sigma$

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_1.png?ts=930255" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_1_loss.png?ts=333297" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=1$
```
Avg return for DQN+RFF (sigma 1) is 5.986098792751864
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_3.png?ts=763228" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_3_loss.png?ts=364467" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=3$
```
Avg return for DQN+RFF (sigma 3) is 6.2352399541306385
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_5.png?ts=541247" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_5_loss.png?ts=965049" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$\sigma=5$
```
Avg return for DQN+RFF (sigma 5) is 6.263865783084839
```
