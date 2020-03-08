import neptune

from agents.multi_process_mcts_agent import MultiMCTSAgent
from agents.random_agent import RandomAgent
from archive.neptune_settings import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME_NN_TRAINING
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator

from nn_models.architectures.average_pool_v0 import StateEncoder, IdentityTransformer, ValueRegressor
from nn_models.tree_data_collector import TreeDataCollector

from mpi4py import MPI
mpi_communicator = MPI.COMM_WORLD
main_process = mpi_communicator.Get_rank() == 0

class MCTS_value_trainer:
    def __init__(self):
        self.model = StateEncoder(final_layer=ValueRegressor(), data_transformer=IdentityTransformer())
        self.value_policy = ValueEvaluator(model = self.model, weights_file=None)
        self.data_collector = TreeDataCollector()
        self.params = {}
        self.arena = MultiArena()
        self.opponent = RandomAgent(distribution='first_buy')


    def create_neptune_experiment(self, experiment_name: str = 'MCTS value training'):
        neptune.init(project_qualified_name=NEPTUNE_PROJECT_NAME_NN_TRAINING, api_token=NEPTUNE_API_TOKEN)
        neptune.create_experiment(name=experiment_name, description='training MCTS value', params=self.params)

    def run_training_games(self, epochs, n_games_per_epoch, n_test_games, mcts_passes, exploration_ceofficient):
        self.mcts_agent = MultiMCTSAgent(mcts_passes, 1, rollout_policy=None, evaluation_policy=)
        for epoch_idx in range(epochs):
            results = self.arena.run_many_duels('deterministic', [])

