from abc import ABC
import numpy as np

from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def act(self, observation):
        raise NotImplementedError


class RandomGamepadAgent(Agent):
    def act(self, observation):
        return np.random.uniform([0, 0, -1], [1, 1, 1], size=(3,))

class RandomArrowsAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        action = self.action_space.sample()
        action[0] = 1
        action[1] = 0
        action[2] = 0

        return action
