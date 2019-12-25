from agents.abstract_agent import Agent
from keras import Model

from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from nn_models.dense_model import DenseModel


class QValueAgent(Agent):

    def __init__(self, weights_file=None):
        super().__init__()
        self.name = 'Dense NN Agent'
        self.model = DenseModel()
        self.model.create_network()
        if weights_file is not None:
            self.model.load_weights(weights_file)

    def choose_act(self, mode):
        current_state_as_dict = StateAsDict(self.env.current_state_of_the_game)
        list_of_actions = self.env.action_space.list_of_actions
        return self.model.choose_best_action(current_state_as_dict, list_of_actions)

