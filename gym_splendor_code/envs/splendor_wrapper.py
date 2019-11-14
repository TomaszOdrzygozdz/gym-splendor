import time
import numpy as np
from gym_splendor_code.envs.splendor import SplendorEnv
from gym_splendor_code.envs.mechanics.game_settings import *

from typing import List

class SplendorWrapperEnv(SplendorEnv):
    """ Description:
        This environment runs the game Splendor."""

    def __init__(self, thread_str=''):
        super().__init__(thread_str)

    def vectorize_observation_space(self):
        state = self.state_to_dict()
        output = []
        gem_list = []
        for i in state.keys():
            if isinstance(state[i], dict):
                for j in state[i].keys():
                    if "card" in j:
                        output.extend([1 if y in state[i][j] else 0 for y in np.arange(CARDS_IN_DECK)])
                    elif "noble" in j:
                        output.extend([1 if y in state[i][j] else 0 for y in np.arange(NOBLES_IN_DECK)])
                    elif "gems" in j:
                        gem_list.extend(state[i][j])
        output.extend(gem_list)
        self.observation_space_vec = output
        print(self.observation_space_vec)

    def vectorize_action_space(self):
        act = self.action_space_to_dict()
        self.action_space_vec = []

        for i in act:
            if i["action_type"] == "buy":
                action = [1,0,0]
            elif i["action_type"] == "reserve":
                action = [0,1,0]
            else:
                action = [0,0,1]

            action.extend([1 if y in {i["card"]} else 0 for y in np.arange(CARDS_IN_DECK)])
            action.extend(i["gems_flow"])

            self.action_space_vec.append(action)
