```python
def vi_by_horizon(rewards, dyn_mats, gamma=0.9, H=10):
    q_values = np.zeros(dyn_mats.shape[:2])

    for step in trange(H, desc="value iteration"):
        q_max = q_values.max(axis=0)
        q_values = rewards + gamma * (dyn_mats @ q_max)

    return q_values
```

Collect the spectrum from a collection of MDPs.

```python
for seed in range(100):
    torch.manual_seed(seed)
    mdp = ToyMDP(seed=seed, n=4)

    states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=2000)
    q_1, q_2 = vi_by_horizon(rewards, dyn_mats, gamma=0.9, H=200)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title('Dynamics')
    plt.plot(mdp.kinks, mdp.nodes[0])
    plt.plot(mdp.kinks, mdp.nodes[1])
    plt.ylim(0, 1)

    plt.subplot(122)
    plt.title('Value')
    plt.plot(states, q_1)
    plt.plot(states, q_2)
    plt.tight_layout()
    doc.savefig(f'{Path(__file__).stem}/toy_mdp_{seed}.png')
```

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_0.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_1.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_2.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_3.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_4.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_5.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_6.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_7.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_8.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_9.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_10.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_11.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_12.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_13.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_14.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_15.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_16.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_17.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_18.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_19.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_20.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_21.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_22.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_23.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_24.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_25.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_26.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_27.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_28.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_29.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_30.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_31.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_32.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_33.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_34.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_35.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_36.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_37.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_38.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_39.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_40.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_41.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_42.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_43.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_44.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_45.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_46.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_47.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_48.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_49.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_50.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_51.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_52.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_53.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_54.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_55.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_56.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_57.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_58.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_59.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_60.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_61.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_62.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_63.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_64.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_65.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_66.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_67.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_68.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_69.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_70.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_71.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_72.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_73.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_74.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_75.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_76.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_77.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_78.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_79.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_80.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_81.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_82.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_83.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_84.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_85.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_86.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_87.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_88.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_89.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_90.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_91.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_92.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_93.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_94.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_95.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_96.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_97.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_98.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>

<img style="align-self:center;" src="visualize_mdp_all/toy_mdp_99.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None"/>
