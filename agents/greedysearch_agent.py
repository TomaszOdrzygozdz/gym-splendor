import random
import numpy as np
from copy import deepcopy

from agent import Agent
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.data.data_loader import load_all_cards, load_all_nobles

class GreedySearchAgent(Agent):

    def __init__(self,
                name: str = "MinMax",
                weight: list = [100,2,2,1,0.1],
                decay: float = 0.9,
                depth: int = 3):
        """:param:
        distribution: argument determining how action is chosen at random. Possible options are:
        uniform - this draws from random distribution on all legal action
        uniform_on_types - first we draw a type of action at random (with uniform distribution on existing types) and
        later choose at random an action of this type from uniform distribution along actions of this type
        first_buy - if it is possible to buy a card we choose buying action at ranodm with uniform distribution, if not
        we choose action at random."""

        super().__init__()

        #we create own gym-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.name = name + ' ' + str(weight)
        self.weight = weight
        self.decay = decay
        self.depth = depth
        self.action_to_avoid = -100

        self.normalize_weight()
        self.env_dict = {lvl : None for lvl in range(1, self.depth)}


    def choose_action(self, observation) -> Action:

        #first we load observation to the private environment
        self.env.load_observation(observation)
        self.env.update_actions()
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()

        if len(self.env.action_space.list_of_actions)>0:
            actions = []
            potential_reward_max = self.action_to_avoid
            numerator = self.depth - 1
            potential_reward_max = self.action_to_avoid

            self.env_dict[numerator] = self.env.vectorize_state(return_var = True)
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                 self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                 self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))

                if potential_reward > potential_reward_max:
                    potential_reward_max = potential_reward
                    actions = []
                    actions.append(action)

                elif potential_reward == potential_reward_max:
                    actions.append(action)
            if len(actions) > 1:
                potential_reward_max = self.action_to_avoid
                for action in actions:
                    ae = action.evaluate(self.env.current_state_of_the_game)
                    potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                     self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                    potential_reward -= self.decay * self.deep_evaluation(action, numerator - 1)

                    if potential_reward > potential_reward_max:
                        potential_reward_max = potential_reward
                        actions = []
                        actions.append(action)

                    elif potential_reward == potential_reward_max:
                        actions.append(action)

                    self.restore_env(numerator)

                self.env.reset()
                self.env.setup_state(self.env_dict[numerator])

            return random.choice(actions)

        else:
            return None


    def deep_evaluation(self, action, numerator):

        self.get_temp_env(action, numerator)

        if numerator > 1:
            current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()
            self.env_dict[numerator] = self.env.vectorize_state(return_var = True)
            if len(self.env.action_space.list_of_actions) > 0:
                potential_reward_list = []
                for action in self.env.action_space.list_of_actions:
                    ae = action.evaluate(self.env.current_state_of_the_game)
                    potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                     self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))

                    if potential_reward > self.action_to_avoid:
                        potential_reward_max = potential_reward
                        actions = []
                        actions.append(action)

                    elif potential_reward == potential_reward_max:
                        actions.append(action)

                for action in actions:
                    ae = action.evaluate(self.env.current_state_of_the_game)
                    potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                     self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                    potential_reward -= self.decay * self.deep_evaluation(action, numerator - 1)
                    potential_reward_list.append(potential_reward)
                    self.restore_env(numerator)

                self.restore_env(numerator + 1)
                reward = max(potential_reward_list)
            else:
                reward = self.action_to_avoid
        else:
            reward = self.evaluate_actions()

        return reward

    def evaluate_actions(self):
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()
        if len(self.env.action_space.list_of_actions) > 0:
            potential_reward_list = []
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                 self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                 self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                potential_reward_list.append(potential_reward)
            reward = max(potential_reward_list)
        else:
            reward = self.action_to_avoid
        return reward

    def get_temp_env(self, action, numerator):
        observation, reward, is_done, info = self.env.step(action, ensure_correctness = False)
        self.env.load_observation(observation)
        self.env.update_actions()

    def restore_env(self, numerator):
        self.env.is_done = False
        self.env.current_state_of_the_game = State(all_cards=self.env.all_cards, all_nobles=self.env.all_nobles)
        self.env.setup_state(self.env_dict[numerator])
        self.env.update_actions()

    def normalize_weight(self):
        if np.linalg.norm(self.weight) > 0:
            self.weight = self.weight/np.linalg.norm(self.weight)
