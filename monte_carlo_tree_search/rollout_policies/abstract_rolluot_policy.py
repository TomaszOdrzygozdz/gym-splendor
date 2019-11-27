from abc import abstractmethod

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State


class RolloutPolicy:
    def __init__(self, name: str = ''):
        self.name = 'Rollout policy: ' + name

    @abstractmethod
    def choose_action(self, state : State)->Action:
        raise NotImplementedError


