from typing import List

import numpy as np

from agents.abstract_agent import Agent
from agents.greedysearch_agent import GreedySearchAgent
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.mechanics.abstract_observation import SplendorObservation
from gym_splendor_code.envs.mechanics.action import Action


class RandomizedAgent(Agent):

    def __init__(self, epsilon):
        super().__init__()
        self.smart_agent = GreedySearchAgent()
        self.epsilon = epsilon
        self.random_agent = RandomAgent(distribution='uniform')

    def choose_action(self, observation : SplendorObservation, previous_actions : List[Action]):
        p = np.random.uniform(0,1)
        if p < self.epsilon:
            return self.random_agent.choose_action(observation, previous_actions)
        else:
            return self.smart_agent.choose_action(observation, previous_actions)



