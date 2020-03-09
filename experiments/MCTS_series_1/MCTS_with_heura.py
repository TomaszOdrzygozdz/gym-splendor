import gin
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, IdentityTransformer

gin.parse_config_file('experiments/MCTS_series_1/params.gin')
from monte_carlo_tree_search.trainers.MCTS_value_training import MCTS_value_trainer

def run_experiment():
    trainer = MCTS_value_trainer()
    trainer.create_neptune_experiment()
    trainer.run_training_games(epochs=100, n_test_games=50, mcts_passes=250, exploration_ceofficient=0.61)