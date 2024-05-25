
## Learning without a target network

We need to first evaluate the bias using a target network. 

```python
num_states = 200
torch.manual_seed(0)
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
gt_q_values, losses = perform_vi(states, rewards, dyn_mats)
```
