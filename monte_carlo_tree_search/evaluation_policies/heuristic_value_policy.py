
from typing import List

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from nn_models.value_function_heura.value_function import ValueFunction


class HeuristicValuePolicy(EvaluationPolicy):

    def __init__(self):
        super().__init__(name = 'HeuristicValuePolicy')
        self.value_function  = ValueFunction()

    def evaluate_state(self, state : State, list_of_actions: List[Action] = None) -> float:
        inversed_state = StateAsDict(state).to_state()
        inversed_state.change_active_player()
        return self.value_function.evaluate(state) - self.value_function.evaluate(inversed_state)

