from typing import List, Tuple

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from nn_models.dense_q_model_v0 import DenseModel


class QValueEvaluator(EvaluationPolicy):

    def __init__(self):

        self.model = DenseModel()
        self.model.create_network()
        self.model.load_weights('E:\ML_research\gym_splendor\\nn_models\weights\minmax_480_games.h5')

    # def evaluate_state(self, state, action : Action):
    #     #active player evaluation
    #     return self.model.get_q_value(StateAsDict(state), action)

    def evaluate_state(self, state : State) -> Tuple[float]:
        #generate all legal actions in the given state
        list_of_actions = generate_all_legal_actions(state)
        return self.model.get_max_q_value(StateAsDict(state), list_of_actions)

