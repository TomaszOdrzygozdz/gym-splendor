import random
from functools import reduce

import gym
import pandas as pd
import numpy as np

from agents.abstract_agent import Agent
from agents.greedy_agent_boost import GreedyAgentBoost
from agents.value_nn_agent import ValueNNAgent
from arena.multi_arena import MultiArena
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from nn_models.vectorization import vectorize_state

from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
main_process = my_rank == 0

class QLearningTrainer:

    def __init__(self):
        self.agent = ValueNNAgent()
        self.env = gym.make('splendor-v0')
        self.weights_token = 'weights_' + str(random.randint(0,1000000)) + '.h5'

    def _set_token(self, token):
        self.weights_token = token

    def _get_token(self):
        return self.weights_token

    def _save_weights(self):
        self.agent.model.save_weights(self.weights_token)

    def _load_weights(self):
        self.agent.model.load_weights(self.weights_token)

    def new_value_formula(self, old_value, best_value, winner_id, reward, alpha=0.1):
        if winner_id is not None:
            return reward
        if winner_id is None:
            if old_value and best_value is not None:
                return (1-alpha)*old_value + alpha*best_value
            else:
                return None


    def run_one_game_and_collect_data(self, debug_info=True):

        opponent = GreedyAgentBoost()
        last_value = None
        self.env.reset()
        observation = self.env.show_observation('deterministic')
        is_done = False
        number_of_moves = 0

        debug_collected_data = pd.DataFrame(columns=('active_player_id', 'winner_id', 'reward', 'best_value'))
        collected_data = pd.DataFrame(columns=('state_as_vector', 'value'))

        while not is_done and number_of_moves < MAX_NUMBER_OF_MOVES:

            action, best_value = self.agent.choose_action(observation, [None], info=True)
            #print('Action = {}'.format(action))
            #print('best value = {}'.format(best_value))
            observation, reward, is_done, info = self.env.step('deterministic', action)
            winner_id = info['winner_id']

            current_state_as_dict = StateAsDict(self.env.current_state_of_the_game)

            old_value = last_value

            if debug_info:
                debug_collected_data = debug_collected_data.append({
                                                        'new_value': self.new_value_formula(old_value, best_value,
                                                                                            winner_id, reward, alpha=0.1),
                                                        'active_player_id' : self.env.previous_player_id(),
                                                        'winner_id' : winner_id,
                                                        'reward' : reward,
                                                        'best_value' : best_value,
                                                        'old_value': old_value,
                                                        'pa_points' : self.env.previous_players_hand().number_of_my_points()},
                                                        ignore_index=True)


            collected_data = collected_data.append({'state_as_vector' : vectorize_state(current_state_as_dict),
                                                        'value': self.new_value_formula(old_value, best_value,
                                                                                            winner_id, reward, alpha=0.1)},
                                                   ignore_index=True)

            #let the opponent move:
            action = opponent.choose_action(observation, [None], info=False)
            print('Greedy action = {}'.format(action))
            print('Is done = {}'.format(is_done))
            observation, reward, is_done, info = self.env.step('deterministic', action)
            winner_id = info['winner_id']

            number_of_moves += 1

            #update old value:
            if self.env.active_player_id() == 0:
                last_value_for_player_0 = best_value
            if self.env.active_player_id() == 1:
                last_value_for_player_1 = best_value

        if debug_info:
            debug_collected_data.to_csv('debug_info.csv')
        collected_data = collected_data.iloc[2:]
        return collected_data

    def train_network(self, collected_data, epochs):
        #prepare X and Y for training:
        self.agent.model.train_model(data_frame=collected_data, epochs=epochs)


class MultiQLearningTrainer:

    def __init__(self):
        self.multi_arena = MultiArena()
        self.local_trainer = QLearningTrainer()
        token = None
        if main_process:
            token = self.local_trainer._get_token()
            print('Current token = {}'.format(token))
            self.local_trainer._save_weights()

        token = comm.bcast(token, root=0)
        self.local_trainer._set_token(token)



    def combine_dataframes(self, list_of_dataframes):
        combined_data_frame = pd.DataFrame(columns=('state_as_vector', 'value'))
        for data_frame in list_of_dataframes:
            combined_data_frame = combined_data_frame.append(data_frame)
        return combined_data_frame


    def run_one_episode(self, epochs):

        if not main_process:
            self.local_trainer._load_weights()

        collected_data = self.local_trainer.run_one_game_and_collect_data()
        #gather data:
        gathered_data = comm.gather(collected_data, root=0)

        network_updated = False

        if main_process:
            combined_data = self.combine_dataframes(gathered_data)
            self.local_trainer.train_network(combined_data, epochs=epochs)
            self.local_trainer._save_weights()
            network_updated = True

        network_updated = comm.bcast(network_updated, root=0)
        assert network_updated

        #broadcats information about saving weights:

    def run_test(self, opponent: Agent):
        results = self.multi_arena.run_many_duels('deterministic', [self.local_trainer.agent, opponent], n_games=comm.Get_size(),
                                        n_proc_per_agent=1, shuffle=True)
        if main_process:
            print(results)

    def run_full_training(self, n_terations, opponent):
        for i in range(n_terations):
            if main_process:
                print('Game number = {}'.format(i))
            self.run_one_episode(epochs=2)
            self.run_test(opponent=opponent)




