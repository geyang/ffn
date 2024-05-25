
## Learning without a target network

We need to first evaluate the bias using a target network. 

```python
num_states = 200
torch.manual_seed(0)
mdp = RandMDP(seed=0, option='fixed')
states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=num_states)
gt_q_values, losses = perform_vi(states, rewards, dyn_mats)
```

Now plot the comparison

```python
plt.plot(states, gt_q_values[0], color="black", linewidth=1, label="Ground Truth", zorder=2)
plt.plot(states, rff_no_tgt_q_values[0], color="#23aaff", linewidth=4, label="FFN (No Target)", alpha=0.8)
plt.plot(states, rff_q_values[0], color="orange", linewidth=3, label="FFN", alpha=0.9)
plt.plot(states, mlp_q_values[0], color="red", linewidth=3, label="MLP", alpha=0.3)
plt.title("Neural Fitted Q Iteration")
plt.xlabel("State [0, 1)")
plt.ylabel("Value")
plt.legend(loc="upper left", framealpha=0.8)
plt.ylim(3, 7.5)
plt.tight_layout()
doc.savefig(f'{Path(__file__).stem}/comparison.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
plt.savefig(f'{Path(__file__).stem}/comparison.pdf', dpi=300)
```

<img style="align-self:center; zoom:0.3;" src="train_vi/comparison.png?ts=927330" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/>
