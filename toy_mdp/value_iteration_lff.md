
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
| <img style="align-self:center; zoom:0.3;" src="value_iteration_lff/value_iteration.png?ts=560936" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff/value_iteration_loss.png?ts=963360" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## DQN w/ LFF

Here we plot the value function learned via deep Q Learning (DQN) using a learned random
fourier feature network.

```python
def get_Q_lff(B_scale):
    return nn.Sequential(
        LFF(1, 50, scale=B_scale),
        nn.Linear(100, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )

Q = get_Q_lff(B_scale=10)
q_values, losses, B_stds, B_means = perform_deep_vi_lff(Q, states, rewards, dyn_mats)
returns = eval_q_policy(Q)

doc.print(f"Avg return for DQN+LFF (sigma 5) is {returns}")
```

```
Avg return for DQN+LFF (sigma 5) is 6.271293828408695
```
| <img style="align-self:center; zoom:0.3;" src="value_iteration_lff/dqn_lff.png?ts=647288" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff/dqn_lff_loss.png?ts=999627" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff/dqn_lff_stddev.png?ts=279350" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="value_iteration_lff/dqn_lff_mean.png?ts=599553" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
