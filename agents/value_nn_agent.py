import random

import numpy as np

from agents.abstract_agent import Agent

from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

class ValueNNAgent(Agent):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def choose_act(self, mode, info=False):

        current_state_as_dict = StateAsDict(self.env.current_state_of_the_game)
        list_of_actions = self.env.action_space.list_of_actions
        if list_of_actions:
            best_action = None
            best_action_value = -100
            for action in list_of_actions:
                state_copy = current_state_as_dict.to_state()
                action.execute(state_copy)
                current_value = self.model.get_value(state_copy)
                if current_value > best_action_value:
                    best_action_value = current_value
                    best_action = action

            if not info:
                return best_action
            if info:
                return best_action, best_action_value

        else:
            if not info:
                return None
            if info:
                return None, -1
