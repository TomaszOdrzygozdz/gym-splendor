from typing import List
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from nn_models.architectures.average_pool_v0 import ValueRegressor, IdentityTransformer, StateEncoder
from nn_models.value_function_heura.value_function import ValueFunction


class HeuraEvaluator(EvaluationPolicy):

    def __init__(self):

        super().__init__(name='Heura Value evaluator')
        self.evaluator = ValueFunction()

    def evaluate_state(self, state : State, list_of_actions: List[Action] = None) -> float:
        #check if the state is terminal
        if state.active_players_hand().number_of_my_points() >= POINTS_TO_WIN:
            return  -1
        elif state.other_players_hand().number_of_my_points() >= POINTS_TO_WIN:
            return  1
        else:
            return self.evaluator.evaluate(state)
