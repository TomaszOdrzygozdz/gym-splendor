import numpy as np

from archive.states_list import state_2, state_4, state_3, state_3_5, state_1_2, state_1
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.state import State

VALUE_IF_NO_LEGAL_ACTIONS = -1

class StateEvaluatorHeuristic:
    def __init__(self,
                 weight: list = [100,2,2,1,0.1]):
        self.weight = weight
        self.normalize_weight()

    def normalize_weight(self):
        if np.linalg.norm(self.weight) > 0:
            self.weight = self.weight/np.linalg.norm(self.weight)

    def evaluate(self, state_to_eval):
        current_points = state_to_eval.active_players_hand().number_of_my_points()
        legal_actions = generate_all_legal_actions(state_to_eval)
        if len(legal_actions):
            points = []
            for action in legal_actions:
                ae = action.evaluate(state_to_eval)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                    self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))

                points.append(potential_reward)
            return max(points)

        else:
            return VALUE_IF_NO_LEGAL_ACTIONS