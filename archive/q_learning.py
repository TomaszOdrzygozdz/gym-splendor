
import random

from neptune_settings import USE_NEPTUNE

if USE_NEPTUNE:
    from neptune_settings import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME
    import  neptune
    neptune.init(project_qualified_name=NEPTUNE_PROJECT_NAME, api_token=NEPTUNE_API_TOKEN)

import gym
import pandas as pd

from agents.abstract_agent import Agent
from agents.q_value_agent import QValueAgent
from arena.arena import Arena
from arena.multi_arena import MultiArena
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from archive.vectorization import vectorize_state, vectorize_action

from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
main_process = my_rank == 0

class QLearningTrainer:

    def __init__(self, alpha):
        self.agent = QValueAgent()
        self.env = gym.make('splendor-v0')
        self.weights_token = 'weights_' + str(random.randint(0,1000000)) + '.h5'
        self.arena = Arena()
        self.alpha = alpha


    def _set_token(self, token):
        self.weights_token = token

    def _get_token(self):
        return self.weights_token

    def _save_weights(self):
        self.agent.model.save_weights(self.weights_token)

    def _load_weights(self):
        self.agent.model.load_weights(self.weights_token)

    def new_value_formula(self, old_value, best_value, winner_id, reward, alpha):
        if winner_id is not None:
            return reward
        if winner_id is None:
            if old_value and best_value is not None:
                return (1-alpha)*old_value + alpha*best_value
            else:
                return None


    def run_one_game_and_collect_data(self, debug_info=True):

        there_was_no_action = False
        self.agent.train_mode()
        last_actual_player_0 = None
        last_actual_player_1 = None
        last_state_player_0 = None
        last_state_player_1 = None
        last_action_vec_player_0 = None
        last_action_vec_player_1 = None
        old_value = None
        old_state = None
        old_action_vec = None
        self.env.reset()
        observation = self.env.show_observation('deterministic')
        is_done = False
        number_of_moves = 0

        debug_collected_data = pd.DataFrame(columns=('active_player_id', 'winner_id', 'reward', 'best_value'))
        collected_data = pd.DataFrame(columns=('state_as_vector', 'value'))
        extra_move_done = False

        while not (is_done and extra_move_done) and number_of_moves < MAX_NUMBER_OF_MOVES:

            if is_done:
                extra_move_done = True

            current_state_as_dict = StateAsDict(self.env.current_state_of_the_game)

            actual_action, actual_eval, best_eval = self.agent.choose_action(observation, [None])
            if actual_action is None:
                there_was_no_action = True
                break
            #print('best value = {}'.format(best_value))
            observation, reward, is_done, info = self.env.step('deterministic', actual_action)
            previous_player_id = self.env.previous_player_id()
            winner_id = info['winner_id']



            if previous_player_id == 0:
                old_value = last_actual_player_0
                old_state = last_state_player_0
                old_action_vec = last_action_vec_player_0

            if previous_player_id == 1:
                old_value = last_actual_player_1
                old_state = last_state_player_1
                old_action_vec = last_action_vec_player_1

            if debug_info:
                state_status = old_state.__repr__() if old_state is not None else 'NONE'
                state_vector = vectorize_state(old_state) if old_state is not None else 'NONE'
                debug_collected_data = debug_collected_data.append({
                                                        'state_ex' : state_status,
                                                        'state_vec' : state_vector,
                                                        'new_value': self.new_value_formula(old_value, best_eval,
                                                                                            winner_id, reward, self.alpha),
                                                        'active_player_id' : self.env.previous_player_id(),
                                                        'winner_id' : winner_id,
                                                        'reward' : reward,
                                                        'best_eval' : best_eval,
                                                        'actual_eval' : actual_eval,
                                                        'old_value': old_value,
                                                        'pa_points' : self.env.previous_players_hand().number_of_my_points()},
                                                        ignore_index=True)


            if old_state is not None:
                collected_data = collected_data.append({'state_as_vector' : vectorize_state(old_state),
                                                        'action_vector' : old_action_vec,
                                                            'value': self.new_value_formula(old_value, best_eval,
                                                                                                winner_id, reward, self.alpha)},
                                                       ignore_index=True)



            if previous_player_id == 0:
                last_actual_player_0 = actual_eval
                last_state_player_0 = current_state_as_dict
                last_action_vec_player_0 = vectorize_action(actual_action)
            if previous_player_id == 1:
                last_actual_player_1 = actual_eval
                last_state_player_1 = current_state_as_dict
                last_action_vec_player_1 = vectorize_action(actual_action)

            #let the opponent move:
            number_of_moves += 1

        if debug_info:
            debug_collected_data.to_csv('debug_info.csv')
        collected_data = collected_data.iloc[0:]
        self.agent.test_mode()
        return collected_data

    def train_network(self, collected_data, epochs):
        #prepare X and Y for training:
        self.agent.model.train_model(data_frame=collected_data, epochs=epochs)

    def run_test(self, opponent: Agent):
        results = self.arena.run_many_duels('deterministic', [self.local_trainer.agent, opponent], n_games=comm.Get_size(),
                                        n_proc_per_agent=1, shuffle=True)
        if main_process:
            print(results)

    def run_training(self, n_iterations, opponent):

        if USE_NEPTUNE:
            neptune.create_experiment('Q learning alpha = '.format(self.alpha))
        experience_replay_buffer = None
        for i in range(n_iterations):
            collected_data, there_was_no_action = self.run_one_game_and_collect_data(debug_info=True)
            if not there_was_no_action:
                self.agent.model.train_model(data_frame=collected_data, epochs=1)
                if experience_replay_buffer is None:
                    experience_replay_buffer = collected_data
                else:
                    experience_replay_buffer = experience_replay_buffer.append(collected_data)
            #Run test
            print('Game number = {}'.format(i))
            if i%20 == 0 and i > 0:
                self.agent.model.train_model(data_frame=experience_replay_buffer, epochs=2)

            if i%100 == 0 and i > 0:
                experience_replay_buffer = None
                print('Clearing buffer')

            if i%10 == 0:
                if USE_NEPTUNE:
                    neptune.send_metric('epsilon', x=self.agent.epsilon)
                results = self.arena.run_many_duels('deterministic', [self.agent, opponent], number_of_games=50)
                print(results)
                if USE_NEPTUNE:
                    for pair in results.data.keys():
                        neptune.send_metric(pair[0] + '_wins', x=i, y=results.data[pair].wins)
                        neptune.send_metric(pair[0] + '_reward', x=i, y=results.data[pair].reward)
                        neptune.send_metric(pair[0] + '_victory_points', x=i, y=results.data[pair].victory_points)



        if USE_NEPTUNE:
            neptune.stop()



class MultiQLearningTrainer:

    def __init__(self, alpha):
        if USE_NEPTUNE and main_process:
            neptune.create_experiment('Q learning M alpha = {}'.format(alpha))

        self.multi_arena = MultiArena()
        self.local_trainer = QLearningTrainer(alpha=alpha)
        token = None
        if main_process:
            token = self.local_trainer._get_token()
            print('Current token = {}'.format(token))
            if USE_NEPTUNE:
                neptune.send_text('weights token', x=token)
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

    def run_test(self, opponent: Agent, x_coord):
        results = self.multi_arena.run_many_duels('deterministic', [self.local_trainer.agent, opponent], n_games=2*comm.Get_size(),
                                        n_proc_per_agent=1, shuffle=True)


        if main_process:
            print(results)
            if USE_NEPTUNE and main_process:
                for pair in results.data.keys():
                    neptune.send_metric(pair[0] + '_wins', x=x_coord, y=results.data[pair].wins)
                    neptune.send_metric(pair[0] + '_reward', x=x_coord,  y=results.data[pair].reward)
                    neptune.send_metric(pair[0] + '_victory_points', x=x_coord, y=results.data[pair].victory_points)


    def run_full_training(self, n_iterations, opponent):


        for i in range(n_iterations):
            if main_process:
                print('Game number = {}'.format(i))
            self.run_one_episode(epochs=2)
            if i %2 == 0:
                self.run_test(opponent=opponent, x_coord=i)

        if USE_NEPTUNE and main_process:
            neptune.stop()




# opponent_x = RandomAgent(distribution='first_buy')
