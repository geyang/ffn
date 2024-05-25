import gym
import numpy as np

from gym import spaces


class ToyMDP(gym.Env):
    example = np.array([[0.69, 0.131, 0.907, 0.079],
                        [0.865, 0.134, 0.75, 0.053]])

    def __init__(self, seed=0, k=4):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(), dtype=np.float32)
        self.time = 0
        self.rng = np.random.RandomState(seed)
        self.obs = self.rng.rand()
        self.n = k
        self.kinks = np.linspace(0, 1, k)
        self.nodes = self.rng.rand(2, k)

    def step(self, action):
        self.time += 1

        if self.obs == 1.0:
            self.obs = self.nodes[action, -1]
        else:
            left = np.floor(self.obs * (self.n - 1)).astype(dtype=int)
            right = left + 1
            w1 = float(right) - self.obs * float(self.n - 1)
            w2 = self.obs * float(self.n - 1) - float(left)

            self.obs = w1 * self.nodes[action, left] + w2 * self.nodes[action, right]

        rew = np.copy(self.obs)
        assert 0 <= self.obs <= 1

        return self.obs, rew, (self.time >= 10), {}

    def reset(self):
        self.time = 0
        self.obs = np.array([self.rng.random()])
        return self.obs

    def get_discrete_mdp(self, num_states=100):
        states = np.linspace(0, 1, num_states)
        rewards = np.sin(2 * np.pi * states)[None].repeat(2, axis=0)
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
