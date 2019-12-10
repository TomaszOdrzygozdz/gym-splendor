from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.rollout_policies.abstract_rolluot_policy import RolloutPolicy
from nn_models.dense_model import DenseModel


class DenseNNRollout(RolloutPolicy):

    def __init__(self):
        self.model = DenseModel()
        self.model.create_network()
        self.model.load_weights('E:\ML_research\gym_splendor\\nn_models\weights\minmax_480_games.h5')

    def choose_action(self, state : State) ->Action:
        list_of_actions = generate_all_legal_actions(state)
        return self.model.choose_best_action(StateAsDict(state), list_of_actions)

