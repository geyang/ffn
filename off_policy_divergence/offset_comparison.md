
## Learning without a target network

We need to first evaluate the bias using a target network. 

```python
num_states = 200
torch.manual_seed(0)
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
q_values, losses = perform_vi(states, rewards, dyn_mats)
```
| <img style="align-self:center; zoom:0.3;" src="offset_comparison/value_iteration.png?ts=250758" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="offset_comparison/value_iteration_loss.png?ts=435755" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|


## Compare between the three Architectures

```python
mlp_q_values, losses = perform_deep_vi(states, rewards, dyn_mats, n_epochs=500)
rff_q_values, losses = perform_deep_vi_rff(
    states, rewards, dyn_mats, n_epochs=500, B_scale=5)
rff_no_tgt_q_values, losses = perform_deep_vi_rff(
    states, rewards, dyn_mats, n_epochs=500, B_scale=5, target_freq=False)
```
```python
plt.figure(figsize=(6.4, 4.8))
plt.plot(states, gt_q_values[0], color="black", linewidth=1, label="Ground Truth", zorder=5)
plt.plot(states, rff_no_tgt_q_values[0], color="#23aaff", linewidth=4, label="FFN (No Target)", alpha=0.8)
plt.plot(states, rff_q_values[0], color="orange", linewidth=3, label="FFN", alpha=0.9)
plt.plot(states, mlp_q_values[0], color="red", linewidth=3, label="MLP", alpha=0.3)
plt.title("Neural Fitted Q Iteration")
plt.xlabel("State [0, 1)")
plt.ylabel("Value")

rect = patches.Rectangle([-0.02, 5], 0.205, 2, fill=True, facecolor="white", linewidth=0, zorder=5)
plt.gca().add_patch(rect)
plt.legend(loc=(0.025, 0.5), framealpha=0, borderaxespad=-10)
plt.ylim(3, 7.5)
plt.xlim(-0.05, 1.05)
plt.tight_layout()
doc.savefig(f'{Path(__file__).stem}/q_value_comparison.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
plt.savefig(f'{Path(__file__).stem}/q_value_comparison.pdf', dpi=300)
```

<img style="align-self:center; zoom:0.3;" src="offset_comparison/q_value_comparison.png?ts=266650" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/>
