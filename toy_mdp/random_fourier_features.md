
## Tabular Q-learning (Ground-truth)

Here is the ground truth value function generated via tabular
value iteration. It shows that even for simple dynamics, the
value function can be exponentially complex due to recursion.

```python
num_states = 20
mdp = RandMDP(seed=0, option='fixed')
# states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
gt_q_values = np.loadtxt("data/q_values.csv", delimiter=',')
states = np.loadtxt("data/states.csv", delimiter=',')
rewards = np.loadtxt('data/rewards.csv', delimiter=',')
```
| <img style="align-self:center; zoom:0.3;" src="figures/toy_mdp.png?ts=347150" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## A Supervised Baseline

**Can the function learn these value functions?** As it turned out, no.
Even with a supervised learning objective, the learned value function is
not able to produce a good approximation of the value landscape. Not
with 20 states, and even less so with 200.

```python
q_values, losses = supervised(states, gt_q_values, dyn_mats, lr=3e-4)
```

