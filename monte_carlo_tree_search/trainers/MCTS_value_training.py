import neptune
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from PIL import Image

from agents.random_agent import RandomAgent
from agents.single_mcts_agent import SingleMCTSAgent
from agents.value_nn_agent import ValueNNAgent
from archive.neptune_settings import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME_NN_TRAINING
from arena.arena import Arena
from arena.arena_multi_thread import ArenaMultiThread
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator

from nn_models.architectures.average_pool_v0 import StateEncoder, IdentityTransformer, ValueRegressor, \
    DataTransformerExp
from nn_models.tree_data_collector import TreeDataCollector

from mpi4py import MPI
mpi_communicator = MPI.COMM_WORLD
comm_size = mpi_communicator.size
main_process = mpi_communicator.Get_rank() == 0

class ReplayBuffer:
    def __init__(self):
        self.games_data = []

    def add_game(self, data):
        self.games_data.append(data)

    def flatten_data(self, list_of_games):
        merged = {'state' : [], 'mcts_value' : []}
        for game in list_of_games:
            merged['state'] += game['state']
            merged['mcts_value'] += game['mcts_value']
        return merged

    def data_from_last_games(self, k):
        if k > len(self.games_data):
            return self.flatten_data(self.games_data)
        else:
            return self.flatten_data(self.games_data[-k:])



class MCTS_value_trainer:
    def __init__(self, weights = None):
        self.data_transformer = IdentityTransformer()
        self.model = StateEncoder(final_layer=ValueRegressor(), data_transformer=self.data_transformer)
        if weights is not None:
            self.model.load_weights(weights)
        self.value_policy = ValueEvaluator(model = self.model, weights_file=None)
        self.opponent_value_policy = ValueEvaluator(model=self.model, weights_file=None)
        self.data_collector = TreeDataCollector()
        self.params = {}
        self.arena = MultiArena()
        self.params_files = []
        self.replay_buffer = {'state' : [], 'mcts_value' : []}
        if main_process:
            self.model.dump_weights('initial_weights.h5')
            self.initial_weights_file = 'initial_weights.h5'
        self.replay_buffer = ReplayBuffer()

    def reset_weights(self):
        self.model.load_weights(self.initial_weights_file)

    def create_neptune_experiment(self, experiment_name, source_files):
        if main_process:
            neptune.init(project_qualified_name=NEPTUNE_PROJECT_NAME_NN_TRAINING, api_token=NEPTUNE_API_TOKEN)
            neptune.create_experiment(name=experiment_name, description='training MCTS value', params=self.params,
                                      upload_source_files=source_files)
        else:
            pass

    def flatten_data(self, gathered_data):
        comm_states = {'state': [], 'mcts_value' : []}
        for local_data in gathered_data:
            comm_states['state'] += local_data['state']
            comm_states['mcts_value'] += local_data['mcts_value']
        return comm_states

    def include_params_file(self, file):
        self.params_files.append(file)

    def parse_params_files(self):
        for file in self.params_files:
            with open(file) as inputfile:
                for line in inputfile:
                    to_log = line.split('=')
                    if len(to_log) == 2:
                        self.params[to_log[0]] = to_log[1]

    def run_training_games_multi_process(self,
                                         opponent_to_train,
                                         baselines,
                                         epochs,
                                         n_test_games,
                                         mcts_passes,
                                         exploration_ceofficient,
                                         experiment_name: str = 'MCTS value training',
                                         weights_path = None,
                                         confidence_threshold: float = 0.1,
                                         confidence_limit:int = 2,
                                         count_ratio: float = 6,
                                         replay_buffer_n_games:int = 10,
                                         neural_network_train_epochs:int  = 2,
                                         reset_network: bool = True,
                                         create_visualizer:bool=True,
                                         use_neptune: bool = True,
                                         tags = ['experiment'],
                                         source_files = None):

        count_threshold = int(count_ratio * mcts_passes)
        if main_process:
            self.params['mcts passes'] = mcts_passes
            self.params['exploration coefficient'] = exploration_ceofficient
            self.params['n test games'] = n_test_games
            self.params['n proc'] = comm_size
            self.params['replay buffer games'] = replay_buffer_n_games
            self.params['opponent name'] = opponent_to_train.name if opponent_to_train != 'self' else 'self-play'
            self.params['train_epochs'] = neural_network_train_epochs
            self.params['count threshold'] = count_threshold
            self.parse_params_files()

        self.mcts_agent = SingleMCTSAgent(mcts_passes, self.value_policy, exploration_ceofficient,
                                          create_visualizer=create_visualizer, show_unvisited_nodes=False,
                                          log_to_neptune=(main_process and use_neptune))

        if opponent_to_train == 'self':
            self.opponent = SingleMCTSAgent(mcts_passes, self.opponent_value_policy, exploration_ceofficient,
                                          create_visualizer=False, show_unvisited_nodes=False,
                                          log_to_neptune=False)
            self.opponent.name = 'MCTS - opponent'
        else:
            self.opponent = opponent_to_train

        if main_process and use_neptune:
            self.create_neptune_experiment(experiment_name=experiment_name, source_files=source_files)
            if opponent_to_train == 'self':
                tags.append('self-play')
            neptune.append_tag(tags)

        for epoch_idx in range(epochs):

            if n_test_games > 0:

                for baseline in baselines:
                    results_with_baseline = self.arena.run_many_duels('deterministic', [self.mcts_agent, baseline], n_games=n_test_games,
                                                     n_proc_per_agent=1, shuffle=False)

                    if main_process:
                        print(results_with_baseline)
                        _, _, baseline_win_rate, baseline_victory_points = results_with_baseline.return_stats()
                        neptune.send_metric(f'Win rate vs {baseline.name}', x=epoch_idx+1, y=baseline_win_rate/n_test_games)
                        neptune.send_metric(f'Win points vs {baseline.name}', x=epoch_idx + 1,
                                            y=baseline_victory_points / n_test_games)

            if main_process:
                print('============ \n Running MCTS games \n============')
            results = self.arena.run_many_duels('deterministic', [self.mcts_agent, self.opponent], n_games=comm_size,
                                                n_proc_per_agent=1, shuffle=False)
            if main_process:
                print(results)
            self.data_collector.setup_root(self.mcts_agent.mcts_algorithm.original_root)
            local_data_for_training = self.data_collector.generate_all_tree_data_as_list(confidence_threshold,
                                                                                         count_threshold, confidence_limit)
            combined_data = mpi_communicator.gather(local_data_for_training, root=0)
            if main_process:
                data_from_this_epoch = self.flatten_data(combined_data)
                self.replay_buffer.add_game(data_from_this_epoch)
                data_for_training = self.replay_buffer.data_from_last_games(replay_buffer_n_games)
                _, _, mcts_win_rate, mcts_victory_points = results.return_stats()
                if use_neptune:
                    neptune.log_metric('MCTS train win rate', x=epoch_idx, y=mcts_win_rate/comm_size)
                    neptune.log_metric('MCTS train victory points', x=epoch_idx, y=mcts_victory_points/comm_size)
                plt.hist(data_for_training['mcts_value'], bins=100)
                plt.savefig('epoch_histogram.png')
                plt.clf()
                img_histogram = Image.open('epoch_histogram.png')
                if use_neptune:
                    neptune.send_image(f'Train set histogram epoch = {epoch_idx}', img_histogram)
                self.data_collector.clean_memory()
                if reset_network:
                    self.reset_weights()
                fit_history = self.model.train_on_mcts_data(data_for_training, train_epochs=neural_network_train_epochs)
                if use_neptune:
                    neptune.send_metric('training set size', x=epoch_idx, y=len(data_for_training['mcts_value']))
                    neptune.send_metric('loss', x=epoch_idx, y=fit_history.history['loss'][0])
                self.mcts_agent.dump_weights(weights_file=weights_path + f'epoch_{epoch_idx}.h5')

                saved = main_process
                weights_saved = mpi_communicator.bcast(saved, root=0)

            if not main_process:
                self.mcts_agent.load_weights(weights_file=weights_path + f'epoch_{epoch_idx}.h5')
                self.opponent.load_weights(weights_file=weights_path + f'epoch_{epoch_idx}.h5')

        if main_process and use_neptune:
            neptune.stop()