
## Spectral Analysis on the Value Function

Here we evaluate the spectral profile of the value function

```python
rewards = np.loadtxt("data/rewards.csv", delimiter=',')
states = np.loadtxt("data/states.csv", delimiter=',')
gt_q_values = np.loadtxt("data/q_values.csv", delimiter=',')
```
| <img style="align-self:center; zoom:0.3;" src="spectral_analysis/reward_power_analysis.png?ts=565557" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="spectral_analysis/reward_power_analysis_Spectrum.png?ts=996943" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

| <img style="align-self:center; zoom:0.3;" src="spectral_analysis/value_power_analysis.png?ts=332022" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center; zoom:0.3;" src="spectral_analysis/value_power_analysis_Spectrum.png?ts=716133" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
