from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy


class DummyEval(EvaluationPolicy):

    def evaluate_all_actions(self, state : State, list_of_actions):
        return [7 for x in list_of_actions]

    def evaluate_state(self, state : State):
        return 8