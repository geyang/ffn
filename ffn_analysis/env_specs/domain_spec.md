```python
for env_name in all_envs:
    env = gym.make(f"dmc:{env_name}-v1")
    doc.print(env_name, env.observation_space, env.action_space, sep="|")
```

```
Acrobot-swingup|Box(6,)|Box(1,)
Quadruped-run|Box(78,)|Box(12,)
Quadruped-walk|Box(78,)|Box(12,)
Humanoid-run|Box(67,)|Box(21,)
Finger-turn_hard|Box(12,)|Box(2,)
Walker-run|Box(24,)|Box(6,)
Cheetah-run|Box(17,)|Box(6,)
Hopper-hop|Box(15,)|Box(4,)
```
