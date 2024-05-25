import gym
import numpy as np

from gym import spaces


class RandMDP(gym.Env):
    def __init__(self, seed=0, option='rand'):
        super(RandMDP, self).__init__()
        assert option in ('rand', 'semi_rand', 'fixed')
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
        self.time = 0
        self.rng = np.random.RandomState(seed)
        self.obs = self.rng.rand()
        if option == 'rand':
            self.kinks = self.rng.rand(2, 2)
            self.kinks.sort(axis=1)
            self.values = self.rng.rand(2, 4)
        elif option == 'semi_rand':
            self.kinks = np.array([[1 / 3, 2 / 3],
                                   [1 / 3, 2 / 3]])
            self.values = np.array([[0.35 * self.rng.rand(), 0.65 + 0.35 * self.rng.rand(), 0.35 * self.rng.rand(),
                                     0.65 + 0.35 * self.rng.rand()],
                                    [0.35 * self.rng.rand(), 0.65 + 0.35 * self.rng.rand(), 0.35 * self.rng.rand(),
                                     0.65 + 0.35 * self.rng.rand()]])
        else:
            self.kinks = np.array([[1 / 3, 2 / 3],
                                   [1 / 3, 2 / 3]])
            self.values = np.array([[0.69, 0.131, 0.907, 0.079],
                                    [0.865, 0.134, 0.75, 0.053]])

    def step(self, action):
        self.time += 1
        kink = self.kinks[action]
        value = self.values[action]
        rew = np.copy(self.obs)

        if self.obs < kink[0]:
            self.obs = value[0] + (value[1] - value[0]) / kink[0] * self.obs
        elif self.obs >= kink[0] and self.obs < kink[1]:
            self.obs = value[1] + (value[2] - value[1]) / (kink[1] - kink[0]) * (self.obs - kink[0])
        else:
            self.obs = value[2] + (value[3] - value[2]) / (1 - kink[1]) * (self.obs - kink[1])
        assert 0 <= self.obs <= 1

        return self.obs, rew, (self.time >= 10), {}

    def reset(self):
        self.time = 0
        self.obs = np.array([self.rng.random()])
        return self.obs

    def get_discrete_mdp(self, num_states=100):
        states = np.linspace(0, 1, num_states)
        rewards = states[None].repeat(2, axis=0)
        dyn_mats = np.zeros((2, num_states, num_states))
        for state_id in range(num_states):
            this_state = states[state_id]
            for action in [0, 1]:
                self.obs = this_state
                self.step(action)
                dst = np.abs(states - self.obs)
                next_state_id = np.argmin(dst)
                dyn_mats[action, state_id, next_state_id] = 1.0

        return states, rewards, dyn_mats
