from gym import Env
import gym.spaces
import numpy as np

class DummyEnv(Env):
    image_shape = (64, 64, 3)
    state_shape = (5,)
    action_shape = (4,)

    observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, image_shape),
            'state': gym.spaces.Box(-1, 1, state_shape)
        })

    action_space = gym.spaces.Box(-1, 1, action_shape)

    def step(self, action):
        return {
            'image': np.zeros(self.image_shape).flatten(),
            'state': np.zeros(self.state_shape)
        }, 0, 0, {}

    def reset(self):
        return {
            'image': np.zeros(self.image_shape).flatten(),
            'state': np.zeros(self.state_shape)
        }

