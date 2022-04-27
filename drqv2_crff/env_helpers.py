# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_env import StepType, specs
from gym import ObservationWrapper


class VstackWrapper(ObservationWrapper):
    def __init__(self, env):
        from gym import spaces
        super(VstackWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=(self.observation_space.shape[0] * self.observation_space.shape[1], *self.observation_space.shape[2:])
        )
        # self.observation_space = self.observation_space.shape[0] * self.observation_space.shape[1]

    def observation(self, lazy_frames):
        return np.vstack(lazy_frames)


def get_env(name, frame_stack, action_repeat, seed):
    import os
    import gym

    # Common render settings
    module, env_name = name.split(':', 1)
    camera_id = 2 if env_name.startswith('Quadruped') else 0  # zoom in camera for quadruped
    render_kwargs = dict(height=84, width=84, camera_id=camera_id)

    # Default gdc background path: $HOME/datasets/DAVIS/JPEGImages/480p/
    extra_kwargs = (
        dict(background_data_path=os.environ.get("DC_BG_PATH", None))
        if module == "distracting_control"
        else dict()
    )
    print('making environment:', name)
    # NOTE: name format:
    #   f'distracting_control:{domain_name.capitalize()}-{task_name}-{difficulty}-v1'
    #   f'dmc:{domain_name.capitalize()}-{task_name}-v1'
    env = gym.make(name, from_pixels=True, frame_skip=action_repeat, channels_first=True,
                   **render_kwargs, **extra_kwargs)
    env.seed(seed)

    # Wrappers
    env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    env = gym.wrappers.FrameStack(env, frame_stack)
    env = VstackWrapper(env)

    # HACK: video_recorder requires access to env.physics.render
    env.physics = env.unwrapped.env.physics
    return env
