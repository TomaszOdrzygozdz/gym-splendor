import gin

from agents.greedy_agent_boost import GreedyAgentBoost
from gym_splendor_code.envs.utils.cluster_detection import CLUSTER
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, IdentityTransformer

gin.parse_config_file('experiments/MCTS_series_1/params.gin')
from monte_carlo_tree_search.trainers.MCTS_value_training import MCTS_value_trainer

def run_experiment():
    #trainer = MCTS_value_trainer(weights='/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/weights/w1.h5')
    trainer = MCTS_value_trainer()

    if not CLUSTER:
        trainer.run_training_games_multi_process(GreedyAgentBoost(), epochs=1, mcts_passes=50, n_test_games=10, exploration_ceofficient=0.61,
                                   experiment_name='MCTS with NN', value_threshold=0.8,
                                                 weights_path='/home/tomasz/ML_Research/splendor/gym-splendor/archive/weights_tt1/',
                                                 confidence_threshold=0.1, count_threshold=50)

    if CLUSTER:
        trainer.run_training_games_multi_process(GreedyAgentBoost(), epochs=200, mcts_passes=150, n_test_games=96,  exploration_ceofficient=0.61,
                                   experiment_name='MCTS with NN', value_threshold=0.8,
                                                 weights_path='/net/archive/groups/plggluna/plgtodrzygozdz/weights_temp/',
                                                 confidence_threshold=0.1, count_threshold=50)