from typing import List

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from nn_models.dense_model import DenseModel


class QValueEvaluation(EvaluationPolicy):

    def __init__(self):

        self.model = DenseModel()
        self.model.load_weights('E:\ML_research\gym_splendor\\nn_models\weights\minmax_480_games.h5')

    # def evaluate_state(self, state, action : Action):
    #     #active player evaluation
    #     return self.model.get_q_value(StateAsDict(state), action)

    def evaluate_all_action(self, state: State, list_of_actions : List[Action]):
        return self.model.get_q_value_of_list(StateAsDict(state), list_of_actions)