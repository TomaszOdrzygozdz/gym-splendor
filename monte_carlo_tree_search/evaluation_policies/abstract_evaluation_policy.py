from abc import abstractmethod
from typing import Tuple, List

import gym

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State


class EvaluationPolicy:
    def __init__(self, name: str = ''):
        self.name = 'Evaluation policy: ' + name
        self.env = gym.make('splendor-v0')

    @abstractmethod
    def evaluate_state(self, state : State, list_ofactions: List[Action])-> Tuple[float]:
        raise NotImplementedError