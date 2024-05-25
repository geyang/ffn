
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
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/value_iteration.png?ts=766080" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/value_iteration_loss.png?ts=251353" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
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
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn.png?ts=887601" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_loss.png?ts=339767" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
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
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised.png?ts=126276" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_loss.png?ts=658762" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Now use FF (supervised)

The same supervised experiment, instantly improve in fit if we 
replace the input layer with FF embedding.

**Notice** that the FF encoding wraps the [0, 1) around on
both ends, the value is continuous at $0.0$ and $1.0$. This is 
an artifact due to the encoding.

```python
def get_Q_rff(band_limit=24, p: float = 0.0):
    return nn.Sequential(
        FF(band_limit, p=p),
        nn.Linear(band_limit * 2, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )


Q = get_Q_rff(p=0.0)
q_values, losses = supervised(Q, states, gt_q_values)
returns = eval_q_policy(Q)

doc.print(f"Avg return for NN+FF+sup is {returns}")
```

```
Avg return for NN+FF+sup is 6.243202148748859
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_rff.png?ts=397307" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/supervised_rff_loss.png?ts=813482" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with FF 

We can now apply this to DQN and it works right away! Using scale of 5

```python
Q = get_Q_rff(p=0.0)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, )
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+FF is {returns}")
```

```
Avg return for DQN+FF is 6.268218360940236
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_10.png?ts=144161" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_10_loss.png?ts=999489" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN without the Target Q

Setting the target network to off

```python
Q = get_Q_rff(p=0.0)
q_values, losses = perform_deep_vi(Q, states, rewards, dyn_mats, target_freq=None)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+FF-tgt is {returns}")
```

```
Avg return for DQN+FF-tgt is 6.1420902499262455
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_no_target.png?ts=073944" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_rff_no_target_loss.png?ts=632196" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


We can experiment with different scaling $\sigma$

$p=0$
```
Avg return for DQN+RFF (p=0) is 6.1420902499262455
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_0.png?ts=920476" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_0_loss.png?ts=389849" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$p=0.5$
```
Avg return for DQN+RFF (p=0.5) is 6.168606658796324
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_0.5.png?ts=675159" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_0.5_loss.png?ts=127875" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$p=1$
```
Avg return for DQN+RFF (p=1) is 6.236701118765187
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_1.png?ts=738847" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_1_loss.png?ts=136309" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$p=1.5$
```
Avg return for DQN+RFF (p=1.5) is 6.092641373380598
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_1.5.png?ts=689587" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_1.5_loss.png?ts=124623" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$p=2$
```
Avg return for DQN+RFF (p=2) is 5.875328861946326
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_2.png?ts=057515" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_2_loss.png?ts=516102" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

$p=inf$
```
Avg return for DQN+RFF (p=inf) is 5.659290829148359
```

| <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_inf.png?ts=594952" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_fine/dqn_ff_inf_loss.png?ts=169783" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
