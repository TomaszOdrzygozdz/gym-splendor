import time
import numpy as np
from gym_splendor_code.envs.splendor import SplendorEnv
from gym_splendor_code.envs.mechanics.game_settings import *

from typing import List

class SplendorWrapperEnv(SplendorEnv):
    """ Description:
        This environment runs the game Splendor."""

    def __init__(self):
        super().__init__()

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

    def vectorize_action_space(self, action_as_dict):

        if action_as_dict["action_type"] == "buy":
            action = [1,0,0]
        elif action_as_dict["action_type"] == "reserve":
            action = [0,1,0]
        else:
            action = [0,0,1]

        action.extend([1 if y in {action_as_dict["card"]} else 0 for y in np.arange(CARDS_IN_DECK)])
        action.extend(action_as_dict["gems_flow"])
        return action


