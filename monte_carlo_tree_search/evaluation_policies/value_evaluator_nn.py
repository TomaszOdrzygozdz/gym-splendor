from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from archive.dense_models.value_dense_model_v0 import ValueDenseModel


class ValueEvaluator(EvaluationPolicy):

    def __init__(self, weights_file : str):

        self.model = ValueDenseModel()
        self.model.create_network()
        if weights_file is not None:
            self.model.load_weights(weights_file=weights_file)

    def evaluate_state(self, state : State) -> float:
        #generate all legal actions in the given state
        return self.model.get_value(StateAsDict(state))

    def evaluate_vector(self, vector_of_state):
        return self.model.get_value_of_vector(vector_of_state)

