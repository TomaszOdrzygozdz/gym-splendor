import random
import numpy as np
import pandas as pd

from agents.abstract_agent import Agent
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from nn_models.vectorization import vectorize_state, vectorize_action


class MinMaxAgent(Agent):

    def __init__(self,
                name: str = "MinMax",
                weight: list = [100,2,2,1,0.1],
                decay: float = 0.9,
                depth: int = 3,
                collect_stats: bool=False):

        super().__init__()

        #we create own gym-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.name = name + ' ' + str(weight)
        self.weight = weight
        self.normalize_weight()
        self.decay = decay
        self.depth = depth
        self.action_to_avoid = -100

        self.collect_stats = collect_stats
        if self.collect_stats:
            self.stats_dataframe = pd.DataFrame(columns=('state', 'action', 'evaluation'))
            self.stats_dataframe_vectorized = pd.DataFrame(columns=('state_vector', 'action_vector', 'evaluation'))

        self.env_dict = {lvl : None for lvl in range(1, self.depth)}

    def choose_act(self, mode) -> Action:

        #first we load observation to the private environment
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()

        if len(self.env.action_space.list_of_actions)>0:
            actions = []
            potential_reward_max = self.action_to_avoid
            numerator = self.depth - 1
            primary_state = StateAsDict(self.env.current_state_of_the_game)
            self.env_dict[numerator] = StateAsDict(self.env.current_state_of_the_game)
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                 self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                 self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))

                potential_reward -= self.decay * self.deep_evaluation(action, numerator - 1, mode)
                self.restore_env(numerator)

                if self.collect_stats:
                    self.stats_dataframe = self.stats_dataframe.append({'state' : primary_state,
                                                    'action' : action.to_dict(),
                                                    'evaluation' : potential_reward}, ignore_index=True)
                    self.stats_dataframe_vectorized = self.stats_dataframe_vectorized.append({'state_vector': vectorize_state(primary_state),
                                                                        'action_vector': vectorize_action(action),
                                                                        'evaluation': potential_reward},
                                                                       ignore_index=True)

                if potential_reward > potential_reward_max:
                    potential_reward_max = potential_reward
                    actions = []
                    actions.append(action)
                elif potential_reward == potential_reward_max:
                    actions.append(action)

            self.env.reset()
            self.env.load_state_from_dict(self.env_dict[numerator])

            return random.choice(actions)

        else:
            return None


    def deep_evaluation(self, action, numerator, mode):

        self.get_temp_env(action, numerator, mode)

        if numerator > 1:
            current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()
            self.env_dict[numerator] = StateAsDict(self.env.current_state_of_the_game)
            if len(self.env.action_space.list_of_actions) > 0:
                potential_reward_list = []
                for action in self.env.action_space.list_of_actions:
                    ae = action.evaluate(self.env.current_state_of_the_game)
                    potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                     self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                    potential_reward -= self.decay * self.deep_evaluation(action, numerator - 1, mode) * pow(-1, self.depth - numerator + 1)
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

    def get_temp_env(self, action, numerator, mode):
        observation, reward, is_done, info = self.env.step(mode, action, ensure_correctness = False)
        self.env.load_observation(observation)
        self.env.update_actions_light()

    def restore_env(self, numerator):
        self.env.is_done = False
        self.env.current_state_of_the_game = State(all_cards=self.env.all_cards, all_nobles=self.env.all_nobles)
        self.env.load_state_from_dict(self.env_dict[numerator])
        self.env.update_actions_light()

    def normalize_weight(self):
        if np.linalg.norm(self.weight) > 0:
            self.weight = self.weight/np.linalg.norm(self.weight)

    def dump_action_scores(self, file_name, clean_dict = True):
        if self.collect_stats:
            self.stats_dataframe.to_csv(file_name + '_raw.csv', header=True)
            self.stats_dataframe_vectorized.to_csv(file_name + '_vectorized.csv', header=True)
            if clean_dict:
                self.stats_dataframe = pd.DataFrame(columns=('state', 'action', 'evaluation'))
                self.stats_dataframe_vectorized = pd.DataFrame(columns=('state_vector', 'action_vector', 'evaluation'))


