import gin

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.utils.cluster_detection import CLUSTER
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, IdentityTransformer

gin.parse_config_file('experiments/MCTS_series_1/params.gin')
from monte_carlo_tree_search.trainers.MCTS_value_training import MCTS_value_trainer

def run_experiment():
    #trainer = MCTS_value_trainer(weights='/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/weights/w1.h5')
    trainer = MCTS_value_trainer()

    if not CLUSTER:
        trainer.run_training_games_multi_process(RandomAgent(distribution='first_buy'), epochs=1, mcts_passes=50, n_test_games=4, exploration_ceofficient=0.41,
                                   experiment_name='MCTS with NN', value_threshold=0.8,
                                                 weights_path='/home/tomasz/ML_Research/splendor/gym-splendor/archive/weights_tt1/',
                                                 confidence_threshold=1,  confidence_limit=2, count_threshold=15,
                                                 replay_buffer_n_games=50)

    if CLUSTER:
        trainer.run_training_games_multi_process(GreedyAgentBoost(), epochs=250, mcts_passes=50, n_test_games=48,  exploration_ceofficient=0.41,
                                   experiment_name='MCTS with NN', value_threshold=0.8,
                                                 weights_path='/net/archive/groups/plggluna/plgtodrzygozdz/weights_temp/',
                                                 confidence_threshold=1, confidence_limit=2, count_threshold=90,
                                                 replay_buffer_n_games=50)