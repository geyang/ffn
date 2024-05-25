
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows even for simple dynamics, the value
function can be exponentially more complex.

```python
num_states = 20
torch.manual_seed(0)
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration/value_iteration.png?ts=437203" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration/value_iteration_loss.png?ts=834097" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ Function Approximator

Here we plot the value function learned via deep Q Learning 
(DQN) using a neural network function approximator.

```python
q_values, losses = perform_deep_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration/dqn.png?ts=002675" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration/dqn_loss.png?ts=395483" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

**But can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
q_values, losses = supervised(states, gt_q_values, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration/supervised.png?ts=058338" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration/supervised_loss.png?ts=466929" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Now use RFF (supervised)

The same supervised experiment, instantly improve in fit if we 
replace the input layer with RFF embedding.

```python
q_values, losses = supervised_rff(states, gt_q_values, dyn_mats, B_scale=10)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration/supervised_over_param.png?ts=458496" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration/supervised_over_param_loss.png?ts=896356" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with RFF 

We can now apply this to DQN and it works right away!

```python
q_values, losses = perform_deep_vi_rff(states, rewards, dyn_mats, n_epochs=500, B_scale=10)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration/dqn_over_param.png?ts=603887" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration/dqn_over_param_loss.png?ts=945961" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN with RFF without Target

Try removing the target network

```python
q_values, losses = perform_deep_vi_rff(states, rewards, dyn_mats, n_epochs=500, B_scale=10,
                                       target_freq=None)
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration/dqn_over_param_no_target.png?ts=568546" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration/dqn_over_param_no_target_loss.png?ts=972447" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
