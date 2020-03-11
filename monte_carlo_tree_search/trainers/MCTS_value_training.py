import neptune
from typing import List
import matplotlib.pyplot as plt
from PIL import Image

from agents.random_agent import RandomAgent
from agents.single_mcts_agent import SingleMCTSAgent
from archive.neptune_settings import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME_NN_TRAINING
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
    def __init__(self):
        self.data_transformer = IdentityTransformer()
        self.model = StateEncoder(final_layer=ValueRegressor(), data_transformer=self.data_transformer)
        self.value_policy = ValueEvaluator(model = self.model, weights_file=None)
        self.data_collector = TreeDataCollector()
        self.params = {}
        self.arena = MultiArena()
        self.opponent = RandomAgent(distribution='first_buy')
        self.replay_buffer = None

    def create_neptune_experiment(self, experiment_name):
        if main_process:
            neptune.init(project_qualified_name=NEPTUNE_PROJECT_NAME_NN_TRAINING, api_token=NEPTUNE_API_TOKEN)
            neptune.create_experiment(name=experiment_name, description='training MCTS value', params=self.params)
        else:
            pass

    def add_to_replay_buffer(self, new_data):
        if self.replay_buffer is None:
            self.replay_buffer = new_data
        else:
            self.replay_buffer['state'] += new_data['state']
            self.replay_buffer['mcts_value'] += new_data['mcts_value']

    def crop_replay_buffer(self, top_r):
        top_idx = int((1-top_r)*len(self.replay_buffer['state']))
        self.replay_buffer['state'] = self.replay_buffer['state'][top_idx:]
        self.replay_buffer['mcts_value'] = self.replay_buffer['mcts_value'][top_idx:]

    def run_training_games(self, epochs, n_test_games, mcts_passes, exploration_ceofficient, experiment_name: str = 'MCTS value training',
                           opponents : List[str] = [], replay_buffer_crop_ceofficient: float = 0.8):
        self.params['mcts passes'] = mcts_passes
        self.params['exploration coefficient'] = exploration_ceofficient
        self.params['n test games'] = n_test_games
        self.params['replay buffer crop ceofficient'] = replay_buffer_crop_ceofficient

        self.create_neptune_experiment(experiment_name)

        self.mcts_agent = SingleMCTSAgent(mcts_passes, self.value_policy, exploration_ceofficient,
                                          create_visualizer=True, show_unvisited_nodes=True)


        for epoch_idx in range(epochs):
            results = self.arena.run_many_duels('deterministic', [self.mcts_agent, self.opponent], 1,
                                                comm_size, shuffle=True)
            if main_process:
                _, _, mcts_win_rate = results.return_stats()
                neptune.log_metric('mcts_win_rate', x=epoch_idx, y=mcts_win_rate)
                self.data_collector.setup_root(self.mcts_agent.mcts_algorithm.original_root)
                data_for_training = self.data_collector.generate_all_tree_data_as_list()
                plt.hist(data_for_training['mcts_value'])
                plt.savefig('epoch_histogram.png')
                plt.clf()
                img_histogram = Image.open('epoch_histogram.png')
                neptune.send_image(f'train set histogram epoch = {epoch_idx}', img_histogram)
                print('max = {}, min = {}'.format(max(data_for_training['mcts_value']), min(data_for_training['mcts_value'])))
                self.data_collector.clean_memory()
                self.add_to_replay_buffer(data_for_training)
                if epoch_idx > 1:
                    self.crop_replay_buffer(replay_buffer_crop_ceofficient)
                neptune.log_metric('replay buffer size', x=epoch_idx, y=len(self.replay_buffer['state']))
                fit_history = self.model.train_on_mcts_data(self.replay_buffer)
                neptune.send_metric('training set size', x=epoch_idx, y=len(data_for_training['mcts_value']))
                neptune.send_metric('loss', x=epoch_idx, y=fit_history.history['loss'][0])
                win_rates = self.model.check_performance(n_test_games, opponents)
                for opponent_name in opponents:
                    neptune.send_metric(f'greedy win rate vs {opponent_name}', x=epoch_idx, y=win_rates[opponent_name])



        neptune.stop()

