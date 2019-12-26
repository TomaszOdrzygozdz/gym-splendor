import random

from agents.abstract_agent import Agent
import numpy as np

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
        self.epsilon = 0.5
        self.explore = False
        self.info = False

    def train_mode(self):
        self.explore = True
        self.info = True

    def test_mode(self):
        self.explore = False
        self.info = False

    def choose_act(self, mode):

        self.epsilon = self.epsilon*0.999
        if not self.explore:
            current_state_as_dict = StateAsDict(self.env.current_state_of_the_game)
            list_of_actions = self.env.action_space.list_of_actions
            if not self.info:
                return self.model.choose_best_action(current_state_as_dict, list_of_actions)
            if self.info:
                return self.model.choose_best_action_with_q_value(current_state_as_dict, list_of_actions)

        if self.explore:
            current_state_as_dict = StateAsDict(self.env.current_state_of_the_game)
            list_of_actions = self.env.action_space.list_of_actions
            p = np.random.uniform(0,1)
            best_action, best_eval = self.model.choose_best_action_with_q_value(current_state_as_dict, list_of_actions)
            if p >= self.epsilon:
                actual_action = best_action
                actual_eval = best_eval
            if p < self.epsilon:
                if list_of_actions:
                    actual_action = random.choice(list_of_actions)
                    actual_eval = self.model.get_q_value(current_state_as_dict, actual_action)[0]
                else:
                    actual_action = None
                    actual_eval = -1
                    best_eval = -1

            return actual_action, actual_eval, best_eval

