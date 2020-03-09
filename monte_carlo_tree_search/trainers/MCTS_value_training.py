import neptune

from agents.multi_process_mcts_agent import MultiMCTSAgent
from agents.random_agent import RandomAgent
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
        self.model = StateEncoder(final_layer=ValueRegressor(), data_transformer=DataTransformerExp(0.2))
        self.value_policy = ValueEvaluator(model = self.model, weights_file=None)
        self.data_collector = TreeDataCollector()
        self.params = {}
        self.arena = MultiArena()
        self.opponent = RandomAgent(distribution='first_buy')


    def create_neptune_experiment(self, experiment_name: str = 'MCTS value training'):
        if main_process:
            neptune.init(project_qualified_name=NEPTUNE_PROJECT_NAME_NN_TRAINING, api_token=NEPTUNE_API_TOKEN)
            neptune.create_experiment(name=experiment_name, description='training MCTS value', params=self.params)
        else:
            pass

    def run_training_games(self, epochs, n_test_games, mcts_passes, exploration_ceofficient):
        self.params['mcts passes'] = mcts_passes
        self.params['exploration coefficient'] = exploration_ceofficient
        self.params['n test games'] = n_test_games

        self.mcts_agent = MultiMCTSAgent(mcts_passes, 1, rollout_policy=None, evaluation_policy=self.value_policy,
                                         exploration_coefficient=exploration_ceofficient, rollout_repetition=0,
                                         create_visualizer=True, show_unvisited_nodes=False)
        for epoch_idx in range(epochs):
            results = self.arena.run_many_duels('deterministic', [self.mcts_agent, self.opponent], 1,
                                                comm_size, shuffle=True)
            if main_process:
                _, _, mcts_win_rate = results.return_stats()
                neptune.log_metric('mcts_win_rate', x=epoch_idx, y=mcts_win_rate)
                self.data_collector.setup_root(self.mcts_agent.mcts_algorithm.original_root())
                data_for_training = self.data_collector.generate_all_tree_data()
                self.data_collector.clean_memory()
                fit_history = self.model.train_on_mcts_data(data_for_training)
                neptune.send_metric('training set size', x=epoch_idx, y=len(data_for_training['mcts_value']))
                neptune.send_metric('loss', x=epoch_idx, y=fit_history.history['loss'][0])
                greedy_win_rate = self.model.check_performance(n_test_games)
                neptune.send_metric('greedy win rate', x=epoch_idx, y=greedy_win_rate)

        neptune.stop()

