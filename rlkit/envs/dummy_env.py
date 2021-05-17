import gym
from roboverse.bullet.serializable import Serializable

class DummyEnv(gym.Env, Serializable):
    def __init__(self):
        super()
