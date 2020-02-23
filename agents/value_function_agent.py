import numpy as np

from agents.abstract_agent import Agent
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from nn_models.value_function_heura.value_function import ValueFunction


class ValueFunctionAgent(Agent):

    def __init__(self):
        super().__init__()
        self.name = 'Value function Agent'
        self.evaluator = ValueFunction()

    def set_weights(self, weights):
        self.evaluator.set_weights(weights)

    def show_weights(self):
        return self.evaluator.weights


    def choose_act(self, mode, info=False):
        current_state_as_dict = StateAsDict(self.env.current_state_of_the_game)
        list_of_actions = self.env.action_space.list_of_actions
        if list_of_actions:
            best_action = None
            best_action_value = -float('inf')
            for action in list_of_actions:
                state_copy = current_state_as_dict.to_state()
                action.execute(state_copy)
                state_copy.change_active_player()
                # print('*******************')
                current_value = self.evaluator.evaluate(state_copy)
                # print(f'State_copy = {StateAsDict(state_copy)}')
                # print(f'Action = {action} val = {current_value}')
                # print('------------------------------------')
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