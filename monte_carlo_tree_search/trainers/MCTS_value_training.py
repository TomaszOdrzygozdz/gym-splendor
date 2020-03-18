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

class MCTS_value_trainer:
    def __init__(self, weights = None):
        self.data_transformer = IdentityTransformer()
        self.model = StateEncoder(final_layer=ValueRegressor(), data_transformer=self.data_transformer)
        if weights is not None:
            self.model.load_weights(weights)
        self.value_policy = ValueEvaluator(model = self.model, weights_file=None)
        self.data_collector = TreeDataCollector()
        self.params = {}
        self.arena = MultiArena()
        self.replay_buffer = {'state' : [], 'mcts_value' : []}


    def create_neptune_experiment(self, experiment_name):
        if main_process:
            neptune.init(project_qualified_name=NEPTUNE_PROJECT_NAME_NN_TRAINING, api_token=NEPTUNE_API_TOKEN)
            neptune.create_experiment(name=experiment_name, description='training MCTS value', params=self.params)
        else:
            pass

    def add_to_replay_buffer(self, new_data, value_threshold):
            for idx in range(len(new_data['mcts_value'])):
                if abs(new_data['mcts_value'][idx]) > value_threshold:
                    #print('Added')
                    self.replay_buffer['state'] += new_data['state']
                    self.replay_buffer['mcts_value'] += new_data['mcts_value']

    def combine_with_buffer(self, new_data):
        combined_data = {'state' : self.replay_buffer['state'] + new_data['state'],
                         'mcts_value' : self.replay_buffer['mcts_value'] + new_data['mcts_value']}
        return combined_data

    def crop_replay_buffer(self, top_r):
        top_idx = int((1-top_r)*len(self.replay_buffer['state']))
        self.replay_buffer['state'] = self.replay_buffer['state'][top_idx:]
        self.replay_buffer['mcts_value'] = self.replay_buffer['mcts_value'][top_idx:]

    def flatten_data(self, gathered_data):
        comm_states = {'state': [], 'mcts_value' : []}
        comm_mcts_values = []
        for local_data in gathered_data:
            comm_states['state'] += local_data['state']
            comm_states['mcts_value'] += local_data['mcts_value']
        return comm_states
    # def run_training_games(self, epochs, n_test_games, mcts_passes, exploration_ceofficient, experiment_name: str = 'MCTS value training',
    #                        opponents : List[str] = [], value_threshold = 0.8):
    #
    #     self.params['mcts passes'] = mcts_passes
    #     self.params['exploration coefficient'] = exploration_ceofficient
    #     self.params['n test games'] = n_test_games
    #     #self.params['replay buffer crop ceofficient'] = replay_buffer_crop_ceofficient
    #     self.params['value threshold'] = value_threshold
    #
    #     self.create_neptune_experiment(experiment_name)
    #
    #     self.mcts_agent = SingleMCTSAgent(mcts_passes, self.value_policy, exploration_ceofficient,
    #                                       create_visualizer=True, show_unvisited_nodes=False)
    #
    #
    #     for epoch_idx in range(epochs):
    #         results = self.arena.run_many_duels('deterministic', [self.mcts_agent, self.opponent], 1,
    #                                             comm_size)
    #         if main_process:
    #             _, _, mcts_win_rate = results.return_stats()
    #             neptune.log_metric('mcts_win_rate', x=epoch_idx, y=mcts_win_rate)
    #             self.data_collector.setup_root(self.mcts_agent.mcts_algorithm.original_root)
    #             data_for_training = self.data_collector.generate_all_tree_data_as_list()
    #             plt.hist(data_for_training['mcts_value'], bins=100)
    #             plt.savefig('epoch_histogram.png')
    #             plt.clf()
    #             img_histogram = Image.open('epoch_histogram.png')
    #             neptune.send_image(f'train set histogram epoch = {epoch_idx}', img_histogram)
    #             print('max = {}, min = {}  mean = {}'.format(max(data_for_training['mcts_value']),
    #                                                          min(data_for_training['mcts_value']),
    #                                                          np.mean(data_for_training['mcts_value'])))
    #             self.data_collector.clean_memory()
    #             self.add_to_replay_buffer(data_for_training, value_threshold)
    #             neptune.log_metric('replay buffer size', x=epoch_idx, y=len(self.replay_buffer['state']))
    #             fit_history = self.model.train_on_mcts_data(self.replay_buffer)
    #             neptune.send_metric('training set size', x=epoch_idx, y=len(data_for_training['mcts_value']))
    #             neptune.send_metric('loss', x=epoch_idx, y=fit_history.history['loss'][0])
    #             win_rates = self.model.check_performance(n_test_games, opponents)
    #             for opponent_name in opponents:
    #                 neptune.send_metric(f'greedy win rate vs {opponent_name}', x=epoch_idx, y=win_rates[opponent_name])
    #     neptune.stop()

    def run_training_games_multi_process(self, opponent, epochs, n_test_games, mcts_passes, exploration_ceofficient, experiment_name: str = 'MCTS value training',
                            value_threshold = 0.8, weights_path = None, confidence_threshold: float = 0.1, count_threshold: int = 6):


        if main_process:
            self.params['mcts passes'] = mcts_passes
            self.params['exploration coefficient'] = exploration_ceofficient
            self.params['n test games'] = n_test_games
            self.params['n proc'] = comm_size
            #self.params['replay buffer crop ceofficient'] = replay_buffer_crop_ceofficient
            self.params['value threshold'] = value_threshold
            #self.create_neptune_experiment(experiment_name)

        self.mcts_agent = SingleMCTSAgent(mcts_passes, self.value_policy, exploration_ceofficient,
                                          create_visualizer=True, show_unvisited_nodes=False)
        self.create_neptune_experiment('Multi Process training')

        for epoch_idx in range(epochs):
            results = self.arena.run_many_duels('deterministic', [self.mcts_agent, opponent], n_games=comm_size,
                                                n_proc_per_agent=1, shuffle=False)
            self.data_collector.setup_root(self.mcts_agent.mcts_algorithm.original_root)
            local_data_for_training = self.data_collector.generate_all_tree_data_as_list(confidence_threshold, count_threshold)
            combined_data = mpi_communicator.gather(local_data_for_training, root=0)

            if main_process:
                print(results)
                data_for_training = self.flatten_data(combined_data)
                _, _, mcts_win_rate = results.return_stats()

                neptune.log_metric('mcts_win_rate', x=epoch_idx, y=mcts_win_rate/comm_size)
                plt.hist(data_for_training['mcts_value'], bins=100)
                plt.savefig('epoch_histogram.png')
                plt.clf()
                img_histogram = Image.open('epoch_histogram.png')
                neptune.send_image(f'train set histogram epoch = {epoch_idx}', img_histogram)
                print('max = {}, min = {}  mean = {}'.format(max(data_for_training['mcts_value']),
                                                             min(data_for_training['mcts_value']),
                                                             np.mean(data_for_training['mcts_value'])))
                self.data_collector.clean_memory()
                #self.add_to_replay_buffer(data_for_training, value_threshold)
                #neptune.log_metric('replay buffer size', x=epoch_idx, y=len(self.replay_buffer['state']))
                #self.add_to_replay_buffer(data_for_training, value_threshold)
                fit_history = self.model.train_on_mcts_data(data_for_training)
                neptune.send_metric('training set size', x=epoch_idx, y=len(data_for_training['mcts_value']))
                neptune.send_metric('loss', x=epoch_idx, y=fit_history.history['loss'][0])

                self.mcts_agent.dump_weights(weights_file=weights_path + f'epoch_{epoch_idx}.h5')

            saved = main_process
            weights_saved = mpi_communicator.bcast(saved, root=0)

            if not main_process:
                self.mcts_agent.load_weights(weights_file=weights_path + f'epoch_{epoch_idx}.h5')

            greedy_agent = ValueNNAgent(model = self.mcts_agent.evaluation_policy.model)
            results_with_greedy = self.arena.run_many_duels('deterministic', [greedy_agent, opponent], n_games=n_test_games,
                                                n_proc_per_agent=1, shuffle=False)

            if main_process:
                _, _, greedy_win_rate = results_with_greedy.return_stats()
                neptune.send_metric(f'greedy win rate vs random', x=epoch_idx, y=greedy_win_rate/n_test_games)

        if main_process:
            neptune.stop()
