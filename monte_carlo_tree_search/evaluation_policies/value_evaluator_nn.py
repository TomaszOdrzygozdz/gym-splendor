from typing import List
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from nn_models.architectures.average_pool_v0 import ValueRegressor, IdentityTransformer, StateEncoder


class ValueEvaluator(EvaluationPolicy):

    def __init__(self, model = None, weights_file = None):
        super().__init__(name='Value average pool evaluator')
        if model is None:
            final_layer = ValueRegressor()
            data_transformer = IdentityTransformer()
            self.model = StateEncoder(final_layer=final_layer, data_transformer=data_transformer)
            if weights_file is not None:
                self.model.load_weights(weights_file)
            if weights_file is not None:
                self.model.load_weights(file_name=weights_file)
        if model is not None:
            self.model = model

    def load_weights(self, weights_file):
        self.model.load_weights(file_name=weights_file)

    def dump_weights(self, weights_file):
        self.model.dump_weights(file_name=weights_file)

    def evaluate_state(self, state : State, list_of_actions: List[Action] = None) -> float:
        #check if the state is terminal
        if state.active_players_hand().number_of_my_points() >= POINTS_TO_WIN:
            return  -1
        elif state.other_players_hand().number_of_my_points() >= POINTS_TO_WIN:
            return  1

        else:
            return self.model.get_value(state)
