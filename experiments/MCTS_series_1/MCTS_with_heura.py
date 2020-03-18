import gin
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, IdentityTransformer

gin.parse_config_file('experiments/MCTS_series_1/params.gin')
from monte_carlo_tree_search.trainers.MCTS_value_training import MCTS_value_trainer

CLUSTER = True

def run_experiment():
    #trainer = MCTS_value_trainer(weights='/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/weights/w1.h5')
    trainer = MCTS_value_trainer()

    if not CLUSTER:
        trainer.run_training_games_multi_process(epochs=1, n_test_games=120, mcts_passes=120, exploration_ceofficient=0.61,
                                   experiment_name='MCTS with NN', value_threshold=0.8,
                                                 weights_path='/home/tomasz/ML_Research/splendor/gym-splendor/archive/weights_tt1/',
                                                 count_threshold=6)

    if CLUSTER:
        trainer.run_training_games_multi_process(epochs=200, n_test_games=120, mcts_passes=120, exploration_ceofficient=0.61,
                                   experiment_name='MCTS with NN', value_threshold=0.8,
                                                 weights_path='/net/archive/groups/plggluna/plgtodrzygozdz/weights_temp/',
                                                 count_threshold=6)
