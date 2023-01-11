| Name             | Observation   | Action  |
| ---------------- | ------------- | ------- |
| Acrobot-swingup  | Box(6,)       | Box(1,) |
| Finger-turn_hard | Box(12,)      | Box(2,) |
| Hopper-hop       | Box(15,)      | Box(4,) |
| Cheetah-run      | Box(17,)      | Box(6,) |
| Walker-run       | Box(24,)      | Box(6,) |
| Humanoid-run     | Box(67,)      | Box(21,)|
| Quadruped-run    | Box(78,)      | Box(12,)|
| Quadruped-walk   | Box(78,)      | Box(12,)|

| env_name | DDPG + MLP | DDPG + RFF |
| -------- | ---------- | --------- |
| Acrobot-swingup | $174.3 \pm 92.6$ | $172.9 \pm 108.2$ |
| Finger-turn_hard | $816.8 \pm 92.1$ | $864.1 \pm 53.8$ |
| Hopper-hop | $282.1 \pm 189.9$ | $353.1 \pm 106.9$ |
| Cheetah-run | $787.2 \pm 77.9$ | $\boldsymbol{890.5 \pm 18.9}$ |
| Walker-run | $750.9 \pm 17.9$ | $\boldsymbol{772.1 \pm 11.5}$ |
| Quadruped-run | $290.9 \pm 119.3$ | $\boldsymbol{799.4 \pm 75.9}$ |
| Quadruped-walk | $311.1 \pm 143.4$ | $\boldsymbol{939.6 \pm 12.3}$ |
| Humanoid-run | $1.3 \pm 0.1$ | $\boldsymbol{177.9 \pm 14.7}$ |

**Fourier Features on SAC and DeepMind Control Tasks**
| env_name | SAC + MLP | SAC + RFF |
| -------- | --------- | --------- | 
| Acrobot-swingup | $101.7 \pm 87.2$ | $219.8 \pm 61.4$ | 
| Finger-turn_hard | $836.2 \pm 80.8$ | $552.2 \pm 182.0$|
| Hopper-hop | $147.8 \pm 88.9$ | $\boldsymbol{322.8 \pm 27.8}$ |
| Cheetah-run | $829.3 \pm 59.8$ | $\boldsymbol{881.0 pm 12.9}$ |
| Walker-run | $852.4 \pm 7.1$ | $811.7 \pm 22.1$ |
| Quadruped-run | $582.2 \pm 216.9$ | $\boldsymbol{919.4 \pm 20.4}$ | 
| Quadruped-walk | $485.1 \pm 280.5$ | $\boldsymbol{944.6 \pm 14.1}$ | 
| Humanoid-run | $191.5 \pm 40.3$ | $\boldsymbol{261.6 \pm 35.4}$ |

## Combined Table


| Name             | Observation | Action   | DDPG + MLP | DDPG + RFF |
| ---------------- | ----------- | -------- | ---------- | --------- |
| Acrobot-swingup  | Box(6,)     | Box(1,)  | $174.3 \pm 92.6$ | $172.9 \pm 108.2$ |
| Finger-turn_hard | Box(12,)    | Box(2,)  | $816.8 \pm 92.1$ | $864.1 \pm 53.8$ |
| Hopper-hop       | Box(15,)    | Box(4,)  | $282.1 \pm 189.9$ | $353.1 \pm 106.9$ |
| Cheetah-run      | Box(17,)    | Box(6,)  | $787.2 \pm 77.9$ | $\boldsymbol{890.5 \pm 18.9}$ |
| Walker-run       | Box(24,)    | Box(6,)  | $750.9 \pm 17.9$ | $\boldsymbol{772.1 \pm 11.5}$ |
| Humanoid-run     | Box(67,)    | Box(21,) | $290.9 \pm 119.3$ | $\boldsymbol{799.4 \pm 75.9}$ |
| Quadruped-run    | Box(78,)    | Box(12,) | $311.1 \pm 143.4$ | $\boldsymbol{939.6 \pm 12.3}$ |
| Quadruped-walk   | Box(78,)    | Box(12,) | $1.3 \pm 0.1$ | $\boldsymbol{177.9 \pm 14.7}$ |











