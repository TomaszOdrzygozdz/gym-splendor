from typing import List, Tuple

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from nn_models.value_dense_model import ValueDenseModel


class ValueEvaluator(EvaluationPolicy):

    def __init__(self):

        self.model = ValueDenseModel()
        self.model.create_network()
        self.model.load_weights('E:\ML_research\gym_splendor\\nn_models\weights\\value_random_rollout_960.h5')

    # def evaluate_state(self, state, action : Action):
    #     #active player evaluation
    #     return self.model.get_q_value(StateAsDict(state), action)

    def evaluate_state(self, state : State) -> float:
        #generate all legal actions in the given state
        return self.model.get_value(StateAsDict(state))

